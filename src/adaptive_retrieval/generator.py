"""Generator interface and backends.

Two real backends:
- ``HFGenerator``: HuggingFace transformers. Works on Mac (MPS/CPU) and Linux/CUDA.
  Slow on CPU but lets the rest of the pipeline run end-to-end locally.
- ``VLLMGenerator``: vLLM batched inference. Linux + CUDA only. Used on the GPU node
  for the ~90k-generation main experiments and the ~250k-pair Estimator training run.

Plus ``MockGenerator`` for unit tests and ``CachedGenerator`` (a transparent
disk-backed cache wrapper) for the experiment runner.
"""
from __future__ import annotations

import hashlib
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Generation:
    prediction: str
    token_ids: list[int]
    token_probs: np.ndarray
    token_entropies: np.ndarray
    latency_ms: float


class Generator:
    """Abstract base. Subclasses implement generate()/generate_batch()."""

    def generate(self, prompt: str) -> Generation:
        raise NotImplementedError

    def generate_batch(self, prompts: list[str]) -> list[Generation]:
        return [self.generate(p) for p in prompts]


# ---------- HuggingFace backend ----------

class HFGenerator(Generator):
    """Transformers `model.generate()` with `output_scores=True`.

    Greedy decoding (``temperature=0``); top-K logprobs computed from scores
    via ``log_softmax``. Entropy is over the top-K, not the full vocabulary
    (per IMPLEMENTATION_GUIDE §C.2 — same approximation as the vLLM path).
    """

    def __init__(
        self,
        model_name: str,
        max_tokens: int = 50,
        top_k_for_entropy: int = 50,
        device: str | None = None,
        dtype: str | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        if dtype is None:
            torch_dtype = torch.float16 if device != "cpu" else torch.float32
        else:
            torch_dtype = getattr(torch, dtype)

        self.max_tokens = max_tokens
        self.top_k = top_k_for_entropy
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype
        ).to(device)
        self.model.eval()
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id or self.eos_token_id

    def generate(self, prompt: str) -> Generation:
        import torch

        t0 = time.time()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.pad_token_id,
            )

        gen_ids = out.sequences[0, prompt_len:].tolist()
        # Strip trailing EOS if present, and align scores 1:1 with kept tokens.
        if self.eos_token_id is not None and self.eos_token_id in gen_ids:
            cut = gen_ids.index(self.eos_token_id)
            gen_ids = gen_ids[:cut]
        scores = list(out.scores)[: len(gen_ids)]

        token_probs: list[float] = []
        token_entropies: list[float] = []
        for step_scores, chosen in zip(scores, gen_ids):
            logits = step_scores[0].float()
            logprobs = torch.log_softmax(logits, dim=-1)
            chosen_lp = float(logprobs[chosen].item())
            token_probs.append(float(np.exp(chosen_lp)))
            topk = torch.topk(logprobs, min(self.top_k, logprobs.shape[-1]))
            topk_lp = topk.values.cpu().numpy().astype(np.float64)
            probs = np.exp(topk_lp)
            probs = probs / probs.sum()
            entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, None))))
            token_entropies.append(entropy)

        prediction = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        t1 = time.time()
        return Generation(
            prediction=prediction,
            token_ids=gen_ids,
            token_probs=np.array(token_probs, dtype=np.float32),
            token_entropies=np.array(token_entropies, dtype=np.float32),
            latency_ms=(t1 - t0) * 1000.0,
        )


# ---------- vLLM backend ----------

class VLLMGenerator(Generator):
    """vLLM continuous-batched inference. Use on the GPU node."""

    def __init__(
        self,
        model_name: str,
        max_tokens: int = 50,
        top_k_for_entropy: int = 50,
        dtype: str = "bfloat16",
    ):
        from vllm import LLM, SamplingParams

        # max_logprobs caps how many per-token logprobs the engine will
        # return. vLLM >=0.10 defaults to 20; we ask for top_k_for_entropy
        # (=50, matching CARD §3.4 Table 1) and the engine refuses unless
        # we raise the ceiling explicitly.
        self.llm = LLM(
            model=model_name,
            dtype=dtype,
            max_logprobs=top_k_for_entropy,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            logprobs=top_k_for_entropy,
        )
        # Cached for the over-length guard (_fit_prompt). vLLM aborts the entire
        # EngineCore if a single prompt exceeds the context window, so we
        # defensively truncate before every generate call.
        self._max_tokens = max_tokens
        self._tokenizer = self.llm.get_tokenizer()
        self._max_model_len = self._detect_max_model_len()

    def _detect_max_model_len(self, default: int = 16384) -> int:
        """Best-effort read of the engine's context window, safe-defaulted.

        The attribute path has moved across vLLM versions, so probe a couple of
        known locations and fall back to the value the engine logs for
        CodeLlama (16384) if neither resolves.
        """
        for getter in (
            lambda: self.llm.llm_engine.model_config.max_model_len,
            lambda: self.llm.llm_engine.vllm_config.model_config.max_model_len,
        ):
            try:
                v = int(getter())
                if v > 0:
                    return v
            except Exception:
                pass
        return default

    def _fit_prompt(self, prompt: str) -> str:
        """Guarantee ``prompt`` fits the context window so an over-long input
        can never crash the engine (which kills the whole run).

        Keeps the TAIL of the prompt: for a FIM prompt that preserves the
        ``<SUF>``/``<MID>`` markers and the context nearest the hole, dropping
        only the far-left prefix head. Over-long prompts are rare and
        pathological (minified/data files in calibration; very large retrieved
        contexts at inference), so truncating beats aborting the job. The
        128-token margin absorbs BOS / round-trip tokenization differences.
        Normal-length prompts are returned unchanged.
        """
        budget = self._max_model_len - self._max_tokens - 128
        if budget <= 0:
            return prompt
        try:
            ids = self._tokenizer.encode(prompt)
            if len(ids) <= budget:
                return prompt
            return self._tokenizer.decode(ids[-budget:])
        except Exception:
            # Last-resort char cap (encode should never fail) — still can't
            # blow the window.
            cap = budget * 2
            return prompt if len(prompt) <= cap else prompt[-cap:]

    @staticmethod
    def _parse(output) -> Generation:
        token_probs: list[float] = []
        token_entropies: list[float] = []
        chosen_ids = list(output.outputs[0].token_ids)
        for step_logprobs, chosen_tok in zip(output.outputs[0].logprobs, chosen_ids):
            chosen_lp = step_logprobs[chosen_tok].logprob
            token_probs.append(float(np.exp(chosen_lp)))
            lp_arr = np.array(
                [lp.logprob for lp in step_logprobs.values()], dtype=np.float64
            )
            probs = np.exp(lp_arr)
            probs = probs / probs.sum()
            entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, None))))
            token_entropies.append(entropy)
        latency_ms = 0.0
        m = getattr(output, "metrics", None)
        if m is not None:
            try:
                latency_ms = (m.last_token_time - m.first_scheduled_time) * 1000.0
            except (TypeError, AttributeError):
                latency_ms = 0.0
        return Generation(
            prediction=output.outputs[0].text,
            token_ids=chosen_ids,
            token_probs=np.array(token_probs, dtype=np.float32),
            token_entropies=np.array(token_entropies, dtype=np.float32),
            latency_ms=latency_ms,
        )

    def generate(self, prompt: str) -> Generation:
        outputs = self.llm.generate([self._fit_prompt(prompt)], self.sampling_params)
        return self._parse(outputs[0])

    def generate_batch(self, prompts: list[str]) -> list[Generation]:
        prompts = [self._fit_prompt(p) for p in prompts]
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [self._parse(o) for o in outputs]


# ---------- Mock for tests ----------

class MockGenerator(Generator):
    """Deterministic, no model. Lets us unit-test pipelines without GPU."""

    def __init__(
        self,
        prompt_to_prediction: dict[str, str] | None = None,
        default_prediction: str = "MOCK",
        token_prob: float = 0.7,
        token_entropy: float = 0.5,
        latency_ms: float = 1.0,
    ):
        self.prompt_to_prediction = prompt_to_prediction or {}
        self.default = default_prediction
        self.token_prob = token_prob
        self.token_entropy = token_entropy
        self.latency_ms = latency_ms
        self.call_log: list[str] = []

    def generate(self, prompt: str) -> Generation:
        self.call_log.append(prompt)
        pred = self.prompt_to_prediction.get(prompt, self.default)
        # Tokenize naively: one "token" per whitespace-separated word, minimum 1.
        n = max(1, len(pred.split()))
        return Generation(
            prediction=pred,
            token_ids=list(range(n)),
            token_probs=np.full(n, self.token_prob, dtype=np.float32),
            token_entropies=np.full(n, self.token_entropy, dtype=np.float32),
            latency_ms=self.latency_ms,
        )


# ---------- mlx-lm backend (Apple Silicon) ----------

class MLXGenerator(Generator):
    """mlx-lm backend optimised for Apple Silicon.

    Roughly 2–3× faster than ``HFGenerator`` on M-series chips. Uses
    ``mlx_lm.generate.generate_step`` which yields ``(token_id, log_probs)``
    per step — exactly the per-token signal CARD's features need.

    Note: installing mlx-lm pulls in transformers >= 5.0, which requires
    torch >= 2.4. If you have the project's pinned torch 2.3.0, the
    HFGenerator backend won't work in the same env — pick one.
    """

    def __init__(
        self,
        model_name: str,
        max_tokens: int = 50,
        top_k_for_entropy: int = 50,
    ):
        from mlx_lm import load

        self.max_tokens = max_tokens
        self.top_k = top_k_for_entropy
        self.model, self.tokenizer = load(model_name)
        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

    def generate(self, prompt: str) -> Generation:
        import mlx.core as mx
        from mlx_lm.generate import generate_step

        t0 = time.time()
        prompt_tokens = self.tokenizer.encode(prompt)
        prompt_arr = mx.array(prompt_tokens)

        def argmax_sampler(logprobs):
            return mx.argmax(logprobs, axis=-1)

        token_ids: list[int] = []
        token_probs: list[float] = []
        token_entropies: list[float] = []

        for token, log_probs in generate_step(
            prompt_arr,
            self.model,
            max_tokens=self.max_tokens,
            sampler=argmax_sampler,
        ):
            # mlx-lm 0.31.x yields the token as a Python int, not mx.array
            # (despite the docstring). int(...) handles both: Python int
            # is idempotent, mx.array 0-D supports __int__.
            tid = int(token)
            if self.eos_token_id is not None and tid == self.eos_token_id:
                break
            token_ids.append(tid)

            # log_probs is a bfloat16 mx.array on Apple Silicon. numpy can't
            # read the bf16 buffer (PEP 3118 mismatch), so cast to float32
            # in mlx-land first, then move to numpy.
            if hasattr(log_probs, "astype"):
                lp = np.asarray(log_probs.astype(mx.float32)).astype(np.float64)
            else:
                lp = np.asarray(log_probs, dtype=np.float64)
            # lp is shape (vocab_size,)
            chosen_lp = float(lp[tid])
            token_probs.append(float(np.exp(chosen_lp)))

            # Top-K entropy
            k = min(self.top_k, lp.shape[-1])
            topk_lp = np.sort(lp)[-k:]
            probs = np.exp(topk_lp)
            probs = probs / probs.sum()
            entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, None))))
            token_entropies.append(entropy)

        prediction = self.tokenizer.decode(token_ids) if token_ids else ""
        t1 = time.time()
        return Generation(
            prediction=prediction,
            token_ids=token_ids,
            token_probs=np.array(token_probs, dtype=np.float32),
            token_entropies=np.array(token_entropies, dtype=np.float32),
            latency_ms=(t1 - t0) * 1000.0,
        )


# ---------- Disk cache wrapper ----------

class CachedGenerator(Generator):
    """Disk-backed cache around any Generator.

    Cache key: ``sha256(model_name :: prompt :: max_tokens)`` per
    IMPLEMENTATION_GUIDE §15.3. Stored as pickle files under
    ``cache_dir/<first2chars>/<full_hash>.pkl``.

    This is the mechanism that makes re-running metrics or sweeping
    thresholds cheap — the 90k-generation main experiment matrix is
    dominated by the Generator, so cache hits across configs (e.g. the
    zero-shot generation is shared by C1, C3, and C4) save substantial
    GPU time.
    """

    def __init__(
        self,
        inner: Generator,
        cache_dir: str | Path,
        model_name: str,
        max_tokens: int = 50,
    ):
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.hits = 0
        self.misses = 0

    def _key(self, prompt: str) -> str:
        h = hashlib.sha256(
            f"{self.model_name}::{prompt}::{self.max_tokens}".encode("utf-8")
        ).hexdigest()
        return h

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.pkl"

    def generate(self, prompt: str) -> Generation:
        key = self._key(prompt)
        path = self._path(key)
        if path.exists():
            try:
                with open(path, "rb") as f:
                    gen = pickle.load(f)
                if isinstance(gen, Generation):
                    self.hits += 1
                    return gen
            except (pickle.UnpicklingError, EOFError, AttributeError):
                # Corrupt cache entry — fall through and regenerate.
                pass
        self.misses += 1
        out = self.inner.generate(prompt)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(out, f)
        return out

    def generate_batch(self, prompts: list[str]) -> list[Generation]:
        # Cache hits per-prompt; misses delegated as a batch to the inner
        # generator for batching efficiency.
        results: list[Generation | None] = [None] * len(prompts)
        miss_prompts: list[str] = []
        miss_idx: list[int] = []
        for i, p in enumerate(prompts):
            key = self._key(p)
            path = self._path(key)
            if path.exists():
                try:
                    with open(path, "rb") as f:
                        gen = pickle.load(f)
                    if isinstance(gen, Generation):
                        self.hits += 1
                        results[i] = gen
                        continue
                except (pickle.UnpicklingError, EOFError, AttributeError):
                    pass
            miss_prompts.append(p)
            miss_idx.append(i)

        if miss_prompts:
            generated = self.inner.generate_batch(miss_prompts)
            for idx, p, gen in zip(miss_idx, miss_prompts, generated):
                results[idx] = gen
                self.misses += 1
                path = self._path(self._key(p))
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "wb") as f:
                    pickle.dump(gen, f)

        # All entries should now be filled.
        assert all(r is not None for r in results)
        return [r for r in results if r is not None]


# ---------- Factory ----------

def make_generator(
    model_name: str,
    backend: str = "auto",
    **kwargs,
) -> Generator:
    """Pick a backend. ``auto`` prefers vLLM if installed, else HF."""
    if backend == "auto":
        try:
            import vllm  # noqa: F401

            backend = "vllm"
        except ImportError:
            try:
                import mlx_lm  # noqa: F401

                backend = "mlx"
            except ImportError:
                backend = "hf"
    if backend == "vllm":
        return VLLMGenerator(model_name, **kwargs)
    if backend == "mlx":
        return MLXGenerator(model_name, **kwargs)
    if backend == "hf":
        return HFGenerator(model_name, **kwargs)
    if backend == "mock":
        return MockGenerator(**kwargs)
    raise ValueError(f"Unknown backend: {backend!r}")
