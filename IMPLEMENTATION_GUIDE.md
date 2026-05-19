# Adaptive Retrieval for Repository-Level Code Completion

**Project type.** Empirical software-engineering research, 4 weeks, 5 contributors. Deliverable is an 8–10-page two-column paper plus code release.

**This file is the single source of truth for the implementation.** It is written for a Claude Code session (or equivalent autonomous engineering agent) that is going to build the system end-to-end. Every component, file, dependency, command, dataset, metric, and validation check is specified explicitly. Where ambiguity is unavoidable, it is flagged with `**DECISION REQUIRED**`.

---

## Table of contents

1. [Project goals](#1-project-goals)
2. [Research questions and hypotheses](#2-research-questions-and-hypotheses)
3. [System overview](#3-system-overview)
4. [Datasets](#4-datasets)
5. [Environment setup](#5-environment-setup)
6. [File structure](#6-file-structure)
7. [Module: Generator](#7-module-generator)
8. [Module: Retriever (BM25)](#8-module-retriever-bm25)
9. [Module: CARD reimplementation](#9-module-card-reimplementation)
10. [Module: Static analysis](#10-module-static-analysis)
11. [Module: The cascade (our contribution)](#11-module-the-cascade-our-contribution)
12. [Module: Baselines](#12-module-baselines)
13. [Metrics](#13-metrics)
14. [Experiment design](#14-experiment-design)
15. [Logging and reproducibility](#15-logging-and-reproducibility)
16. [Implementation order and validation checkpoints](#16-implementation-order-and-validation-checkpoints)
17. [Known risks and mitigations](#17-known-risks-and-mitigations)
18. [Appendix A: CARD paper reference numbers](#appendix-a-card-paper-reference-numbers)
19. [Appendix B: Code skeletons](#appendix-b-code-skeletons)
20. [Appendix C: Common pitfalls](#appendix-c-common-pitfalls)
21. [Appendix D: Dataset schemas](#appendix-d-dataset-schemas)
22. [Appendix E: Extended static analysis tests](#appendix-e-extended-static-analysis-tests)
23. [Appendix F: End-to-end smoke test](#appendix-f-end-to-end-smoke-test)

---

## 1. Project goals

We are studying whether **adaptive retrieval** — selectively triggering retrieval-augmented generation only on instances where it would help — outperforms always-retrieve and never-retrieve baselines for repository-level code completion. The novel contribution is **adding a static-analysis "second-chance" gate to the CARD framework**: CARD (Zhang et al. 2024, arXiv:2406.10263) uses logit-based uncertainty signals to decide whether to retrieve; we add a parallel signal from static analysis of the model's prediction that fires retrieval on instances where CARD would otherwise abstain.

The hypothesis is that CARD's uncertainty signal misses a specific class of failures: confidently-generated identifiers that don't actually exist in the repository (hallucinations). Static analysis can detect these directly by checking whether predicted identifiers resolve in the repo's symbol table. Adding this signal should reduce the hallucination rate without significantly increasing retrieval cost.

**Concrete deliverables:**
- A working reimplementation of CARD (since no public code release exists).
- The cascade architecture: CARD + static-analysis fallback.
- No-retrieve and always-retrieve baselines.
- A measurement pipeline for hallucination rate, plus all standard accuracy metrics.
- A full ablation matrix on CrossCodeEval-Python and RepoEval.
- A research paper writing up the findings.

---

## 2. Research questions and hypotheses

### RQ1 — Cascade vs. always/never baselines

> How does the proposed CARD + static-analysis cascade perform against always-retrieve and never-retrieve baselines on CrossCodeEval-Python and RepoEval (line + API subsets), measured by accuracy (EM, ES, Identifier-F1) and efficiency (% retrieval, end-to-end latency)?

**Hypothesis H1.** The cascade matches or exceeds always-retrieve on accuracy while performing fewer retrievals (latency win), and strictly outperforms never-retrieve on accuracy.

### RQ2 — Hallucination reduction from static analysis

> Does adding the static-analysis cascade to CARD reduce the rate of identifier hallucinations in generated code, compared to vanilla CARD?

**Hypothesis H2.** The cascade reduces the per-instance hallucination rate by at least 20% relative to vanilla CARD, with statistical significance under McNemar's test (p < 0.05).

### RQ3 — CARD vs. baselines

> How does vanilla CARD compare against always-retrieve and never-retrieve on the same datasets?

**Hypothesis H3.** CARD achieves accuracy comparable to always-retrieve (within ±1 ES) while performing 20–46% fewer retrievals, replicating the CARD paper's published results on RepoEval and generalising to CrossCodeEval-Python.

RQ3 is necessary because the CARD paper compares against iterative-RAG systems (RepoCoder), not directly against the two simple baselines our project description requires.

---

## 3. System overview

### High-level architecture

```
                    ┌─────────────────────────────┐
                    │   (X_left, X_right), repo R │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │   Generator G (no retrieval)│
                    │   ŷ₀, logits₀ ← G(X_l, X_r) │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │  CARD.isRetrieve(ŷ₀, logits₀)│
                    └──────┬───────────────┬──────┘
                       YES │               │ NO
                           ▼               ▼
              ┌──────────────────┐  ┌────────────────────────────┐
              │   Retrieve CC    │  │  StaticAnalysis(ŷ₀, R)     │  ◄── NEW
              │   ŷ ← G(X, CC)   │  │                            │
              └────────┬─────────┘  └─────┬─────────────────┬────┘
                       │              YES │             NO  │
                       │                  ▼                 ▼
                       │      ┌──────────────────┐  ┌──────────────┐
                       │      │   Retrieve CC    │  │   return ŷ₀  │
                       │      │   ŷ ← G(X, CC)   │  │              │
                       │      └────────┬─────────┘  └──────┬───────┘
                       │               │                   │
                       └───────────────┴───────────────────┘
                                       │
                                       ▼
                              ┌──────────────┐
                              │  return ŷ    │
                              └──────────────┘
```

### Key design choices

- **Asymmetric cascade.** The static-analysis stage can only *add* retrievals to CARD's decisions, never remove them. This bounds the worst-case retrieval count (≤ always-retrieve) and frames the central question as "does the extra retrieval budget reduce hallucinations?" rather than a confounded accuracy-vs-cost trade-off.

- **Static analysis on the prediction, not just the prompt.** The novel signal is "does ŷ₀ reference identifiers that don't resolve in the repository?" — this is directly diagnostic of hallucination, not a proxy.

- **CARD as a faithful reimplementation.** We follow the paper's spec (§§2–3 of Zhang et al. 2024) exactly. The cascade adds Stage 3 without modifying CARD's internals.

- **One-shot, not iterative.** We use CARD's single-RAG variant (§2.4 of the CARD paper, "Single RAG"). The iterative variant adds complexity that isn't needed for our research question. If time permits in Week 4, we can extend to iterative.

---

## 4. Datasets

### 4.1 Primary evaluation: CrossCodeEval (Python subset)

- **Source.** Ding et al. 2023, arXiv:2310.11248. HuggingFace: `crosscodeeval` (or `microsoft/CrossCodeEval`).
- **Size.** ~2,460 Python instances.
- **Why.** Has identifier-F1 metric; cross-file dependency is explicitly annotated; well-known.
- **How to load.**
  ```python
  from datasets import load_dataset
  ds = load_dataset("microsoft/CrossCodeEval", "python", split="test")
  ```
- **Fields used.** `prompt` (in-file left context), `right_context`, `groundtruth`, `crossfile_context`, `repository`, `path`.

### 4.2 Secondary evaluation: RepoEval

- **Source.** Zhang et al. 2023 (RepoCoder paper), available at `https://github.com/microsoft/CodeT/tree/main/RepoCoder`.
- **Size.** 1,600 line completion + 1,600 API completion + 373 function completion (Python only).
- **Why.** CARD reports published numbers on RepoEval → direct sanity check for our reimplementation.
- **How to load.** Clone the RepoCoder repo; the data files are JSONL. Each line has `prompt`, `metadata.task_id`, `metadata.fpath_tuple`, `metadata.ground_truth`.
  ```bash
  git clone https://github.com/microsoft/CodeT
  # Files at CodeT/RepoCoder/datasets/
  ```

### 4.3 Training-data source: The Stack (subset)

- **Source.** `bigcode/the-stack-smol` on HuggingFace (1% sample, 3.1GB compressed).
- **Why.** Need ~11k Python repos with 50–100 files each to construct CARD's training set per §3.4 of the CARD paper.
- **How to load.**
  ```python
  ds = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train")
  ```
- **Note.** The full `bigcode/the-stack` requires accepting a data agreement. The Smol subset is sufficient because we only need 11k repos and the Smol subset contains far more.

### 4.4 Skipped datasets

- **RepoBench.** Skipped because its cross_file/in_file labels make the always/never baselines trivial — there's no real adaptive question to answer there.
- **CrossCodeLongEval.** Optional extension if time permits in Week 4. It's the natural-distribution counterpart to CrossCodeEval and would strengthen the RQ1 results.

---

## 5. Environment setup

### 5.1 Hardware

- **Training Estimator.** CPU only, ~5 minutes.
- **Generating training data for Estimator.** 1× NVIDIA A100 40GB or 80GB, ~24 hours.
- **Main experiments.** 1× A100 80GB recommended. The full experiment matrix (6 configs × 3 datasets × ~5000 instances) is ~90k generations, runnable in ~48 GPU-hours with vLLM batching.

### 5.2 Python

- **Version.** Python 3.11 (3.10 also works; avoid 3.12 — tree-sitter bindings have intermittent issues).

### 5.3 Dependencies

Create `requirements.txt` with pinned versions:

```text
# Core
numpy==1.26.4
scipy==1.13.0
scikit-learn==1.5.0

# LLM serving
vllm==0.5.4
torch==2.3.0
transformers==4.43.0
accelerate==0.32.0

# Data
datasets==2.20.0
huggingface-hub==0.24.0

# Static analysis
tree-sitter==0.22.3
tree-sitter-python==0.21.0

# Retrieval
rank-bm25==0.2.2

# Metrics
python-Levenshtein==0.25.1

# ML for CARD's Estimator
lightgbm==4.5.0
xgboost==2.1.0      # for ablation only

# Utilities
tqdm==4.66.4
pydantic==2.8.0
jsonlines==4.0.0
click==8.1.7

# Dev
pytest==8.3.2
ruff==0.5.5
```

Install:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 5.4 Models

**DECISION REQUIRED.** Choose generator(s):
- **Option A (recommended for paper):** `Qwen/Qwen2.5-Coder-7B` (base, fill-in-the-middle capable, strongest available 7B code model in 2026). Use this for headline results.
- **Option B (for CARD validation):** `codellama/CodeLlama-7b-hf` (base, matches CARD paper exactly). Use this to validate the CARD reimplementation against the paper's reported numbers.

Recommended approach: train two Estimators, one per generator. Run RepoEval with both for the CARD-validation table; run CrossCodeEval-Python with Qwen2.5-Coder-7B for headline numbers.

### 5.5 Cluster/compute

DelftBlue allocation (30 hours mentioned in initial notes) is **tight**. Plan:
- Use A100 80GB for vLLM batching efficiency.
- Run experiments in Week 3 only — do all infrastructure validation on smaller subsets in Weeks 1–2.
- Cache aggressively: generations should be cached to disk so a re-run of metrics doesn't re-generate.

---

## 6. File structure

Create the project structure exactly as below:

```
adaptive-retrieval/
├── README.md
├── IMPLEMENTATION_GUIDE.md          # This file
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── data/
│   ├── crosscodeeval/
│   ├── repoeval/
│   ├── stack_subset/                # Cached Stack repos for training data
│   └── training_data/               # (features, scores) pairs for Estimator
│       ├── codellama_7b.parquet
│       └── qwen25_coder_7b.parquet
│
├── models/
│   ├── estimator_codellama_7b.lgb   # Trained LightGBM
│   └── estimator_qwen25_coder_7b.lgb
│
├── src/adaptive_retrieval/
│   ├── __init__.py
│   ├── generator.py                 # vLLM wrapper, returns logits
│   ├── retriever.py                 # BM25 retrieval
│   ├── prompt.py                    # Prompt assembly
│   ├── cascade.py                   # Our main contribution
│   ├── baselines.py                 # No-retrieve, always-retrieve
│   ├── metrics.py                   # All metrics
│   │
│   ├── card/
│   │   ├── __init__.py
│   │   ├── features.py              # Table 1 of CARD paper
│   │   ├── estimator.py             # LightGBM train/inference
│   │   ├── pipeline.py              # Algorithm 1
│   │   └── train_data.py            # Construct training dataset
│   │
│   ├── static_analysis/
│   │   ├── __init__.py
│   │   ├── parser.py                # tree-sitter setup
│   │   ├── symbol_table.py          # Repository symbol table
│   │   ├── scope.py                 # In-file scope analyzer
│   │   └── analyzer.py              # PredictionAnalyzer
│   │
│   └── eval/
│       ├── __init__.py
│       ├── datasets.py              # Loaders for CrossCodeEval, RepoEval
│       └── runner.py                # Experiment runner
│
├── scripts/
│   ├── 01_construct_training_data.py
│   ├── 02_train_estimator.py
│   ├── 03_validate_card.py
│   ├── 04_run_experiment.py
│   ├── 05_compute_metrics.py
│   └── 06_analysis.py
│
├── results/
│   └── (JSON logs per experiment)
│
└── tests/
    ├── test_features.py
    ├── test_estimator.py
    ├── test_static_analysis.py
    ├── test_metrics.py
    └── test_cascade.py
```

---

## 7. Module: Generator

**File:** `src/adaptive_retrieval/generator.py`

### Interface

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class Generation:
    prediction: str                  # decoded text
    token_ids: list[int]             # generated token IDs
    token_probs: np.ndarray          # shape (N,), p_t(ŷ_t) of chosen tokens
    token_entropies: np.ndarray      # shape (N,), H_t over top-k vocab
    latency_ms: float

class Generator:
    def __init__(self, model_name: str, max_tokens: int = 50, top_k_for_entropy: int = 50):
        ...

    def generate(self, prompt: str) -> Generation:
        ...

    def generate_batch(self, prompts: list[str]) -> list[Generation]:
        ...
```

### Implementation notes

Use **vLLM** for batched inference with logprob extraction:

```python
from vllm import LLM, SamplingParams

class Generator:
    def __init__(self, model_name, max_tokens=50, top_k_for_entropy=50):
        self.llm = LLM(model=model_name, dtype="bfloat16")
        self.sampling_params = SamplingParams(
            temperature=0.0,                  # greedy
            max_tokens=max_tokens,
            logprobs=top_k_for_entropy,       # top-K logprobs per token
        )

    def generate_batch(self, prompts):
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [self._parse(o) for o in outputs]

    def _parse(self, output):
        # output.outputs[0].logprobs is a list[dict[int, Logprob]] per token
        # For each step t:
        #   - chosen_token = output.outputs[0].token_ids[t]
        #   - p_t = exp(logprobs[t][chosen_token].logprob)
        #   - H_t = entropy over the top-K logprobs at step t
        token_probs = []
        token_entropies = []
        for step_logprobs, chosen_tok in zip(output.outputs[0].logprobs,
                                              output.outputs[0].token_ids):
            chosen_logprob = step_logprobs[chosen_tok].logprob
            token_probs.append(np.exp(chosen_logprob))
            # Entropy over top-K only (approximation; sufficient for K=50)
            logprobs_array = np.array([lp.logprob for lp in step_logprobs.values()])
            probs_array = np.exp(logprobs_array)
            probs_array = probs_array / probs_array.sum()  # renormalise
            token_entropies.append(-np.sum(probs_array * np.log(probs_array + 1e-12)))
        return Generation(
            prediction=output.outputs[0].text,
            token_ids=list(output.outputs[0].token_ids),
            token_probs=np.array(token_probs),
            token_entropies=np.array(token_entropies),
            latency_ms=(output.metrics.last_token_time - output.metrics.first_scheduled_time) * 1000,
        )
```

### Sanity checks

```python
def test_generator_basic():
    gen = Generator("Qwen/Qwen2.5-Coder-7B")
    out = gen.generate("def hello():\n    return ")
    assert len(out.prediction) > 0
    assert out.token_probs.shape == out.token_entropies.shape
    assert all(0 <= p <= 1 for p in out.token_probs)
    assert all(h >= 0 for h in out.token_entropies)
```

---

## 8. Module: Retriever (BM25)

**File:** `src/adaptive_retrieval/retriever.py`

### Interface

```python
@dataclass
class RetrievedChunk:
    text: str
    file_path: str
    start_line: int
    end_line: int
    score: float

class BM25Retriever:
    def __init__(self, repo_files: dict[str, str], chunk_size: int = 20, stride: int = 10):
        """repo_files: {file_path: file_contents}"""
        ...

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        ...
```

### Implementation notes

- **Chunking.** For each file, slide a window of `chunk_size=20` lines with `stride=10`. Matches CARD/RepoCoder exactly.
- **Query construction.** Use the last 20 non-blank lines of `X_left` as the query. (RepoCoder convention.)
- **Tokenisation.** Whitespace + punctuation split (`re.findall(r'\w+', text)`), lowercased.
- **Library.** Use `rank_bm25.BM25Okapi`.

```python
import re
from rank_bm25 import BM25Okapi

def _tokenize(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())

class BM25Retriever:
    def __init__(self, repo_files, chunk_size=20, stride=10):
        self.chunks = []
        for path, content in repo_files.items():
            lines = content.splitlines()
            for i in range(0, max(1, len(lines) - chunk_size + 1), stride):
                chunk_text = "\n".join(lines[i:i + chunk_size])
                self.chunks.append({
                    "text": chunk_text, "file_path": path,
                    "start_line": i, "end_line": min(i + chunk_size, len(lines))
                })
        tokenized = [_tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query, top_k=10):
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_idx = scores.argsort()[::-1][:top_k]
        return [RetrievedChunk(**self.chunks[i], score=scores[i]) for i in top_idx]
```

### Prompt assembly with retrieved chunks

**File:** `src/adaptive_retrieval/prompt.py`

For fill-in-the-middle generators (Qwen2.5-Coder, CodeLlama):

```python
def build_fim_prompt(x_left: str, x_right: str,
                     retrieved: list[RetrievedChunk] | None = None,
                     model_family: str = "qwen") -> str:
    if model_family == "qwen":
        fim_prefix, fim_suffix, fim_middle = "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"
    elif model_family == "codellama":
        fim_prefix, fim_suffix, fim_middle = "▁<PRE>", "▁<SUF>", "▁<MID>"
    elif model_family == "starcoder":
        fim_prefix, fim_suffix, fim_middle = "<fim_prefix>", "<fim_suffix>", "<fim_middle>"
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    if retrieved:
        cc_text = "# Here are some relevant code fragments from other files of the repo:\n"
        for chunk in retrieved:
            cc_text += f"# the below code fragment can be found in: {chunk.file_path}\n"
            cc_text += "\n".join("# " + line for line in chunk.text.splitlines())
            cc_text += "\n\n"
        x_left = cc_text + x_left

    # Token budget: truncate left to 1024 tokens worth, right to 512
    # (Use tokenizer to enforce, but for now keep characters proportional)
    return f"{fim_prefix}{x_left}{fim_suffix}{x_right}{fim_middle}"
```

---

## 9. Module: CARD reimplementation

**Reference.** Zhang et al. 2024, arXiv:2406.10263 (PDF in repo at `papers/2406.10263v1.pdf`).

### 9.1 Feature extraction

**File:** `src/adaptive_retrieval/card/features.py`

Implements Table 1 of the CARD paper. The output is a 13-D vector.

```python
import numpy as np

FEATURE_NAMES = [
    "prob_max", "prob_min", "prob_avg", "prob_std", "prob_prod", "prob_geomavg",
    "ent_max", "ent_min", "ent_avg", "ent_std", "ent_prod", "ent_geomavg",
    "length"
]
assert len(FEATURE_NAMES) == 13

def extract_features(token_probs: np.ndarray, token_entropies: np.ndarray) -> np.ndarray:
    """
    Extract 13-D feature vector per Table 1 of CARD paper.

    Args:
        token_probs: shape (N,), p_t(ŷ_t)
        token_entropies: shape (N,), H_t

    Returns:
        np.ndarray of shape (13,) following FEATURE_NAMES order.
    """
    assert token_probs.shape == token_entropies.shape
    assert token_probs.ndim == 1
    N = len(token_probs)
    if N == 0:
        # Edge case: empty generation. Return zeros and length 0.
        return np.zeros(13, dtype=np.float32)

    feats = []
    for x in (token_probs, token_entropies):
        feats.append(float(np.max(x)))
        feats.append(float(np.min(x)))
        feats.append(float(np.mean(x)))
        feats.append(float(np.std(x)))
        # Product and geometric average computed in log-space to avoid underflow.
        # CRITICAL: probabilities multiply down to ~0 over 50 tokens otherwise.
        log_x = np.log(np.clip(x, 1e-12, None))
        feats.append(float(np.exp(np.sum(log_x))))
        feats.append(float(np.exp(np.mean(log_x))))
    feats.append(float(N))
    return np.array(feats, dtype=np.float32)
```

### 9.2 Estimator (LightGBM)

**File:** `src/adaptive_retrieval/card/estimator.py`

```python
import lightgbm as lgb
import numpy as np

class Estimator:
    def __init__(self, model: lgb.Booster | None = None):
        self.model = model

    @classmethod
    def train(cls, features: np.ndarray, scores: np.ndarray,
              val_fraction: float = 0.05, random_state: int = 42):
        """
        Train LightGBM regressor predicting ES from features.

        Args:
            features: shape (N, 13)
            scores: shape (N,) in [0, 1] (edit similarity)
        """
        assert features.shape[1] == 13
        assert features.shape[0] == scores.shape[0]

        rng = np.random.default_rng(random_state)
        n = len(features)
        idx = rng.permutation(n)
        n_val = int(n * val_fraction)
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        train_data = lgb.Dataset(features[train_idx], scores[train_idx])
        val_data = lgb.Dataset(features[val_idx], scores[val_idx], reference=train_data)
        model = lgb.train(
            params={"objective": "regression", "metric": "mse", "verbose": -1},
            train_set=train_data,
            valid_sets=[val_data],
            num_boost_round=500,
            callbacks=[lgb.early_stopping(20)],
        )
        return cls(model=model)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Returns ŝ for one or more feature vectors."""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return self.model.predict(features)

    def save(self, path: str):
        self.model.save_model(path)

    @classmethod
    def load(cls, path: str):
        return cls(model=lgb.Booster(model_file=path))
```

### 9.3 isRetrieve and Select

**File:** `src/adaptive_retrieval/card/pipeline.py`

Implements §§2.2–2.3 of the CARD paper.

```python
import numpy as np
from .features import extract_features

EPSILON = 1e-6

def is_retrieve(estimator, token_probs, token_entropies, T_RAG: float) -> bool:
    """Returns True iff retrieval is needed (i.e., ŝ < T_RAG)."""
    feats = extract_features(token_probs, token_entropies)
    s_hat = float(estimator.predict(feats)[0])
    return s_hat < T_RAG

def select(s_hat_i: float, s_hat_j: float, T_ACC: float) -> bool:
    """Returns True iff we should KEEP the older generation ŷ^i (reject ŷ^j)."""
    return (s_hat_j / (s_hat_i + EPSILON)) < T_ACC
```

### 9.4 Full pipeline (Algorithm 1)

```python
@dataclass
class CARDOutput:
    prediction: str
    n_iterations: int             # 0 = zero-shot kept; 1 = RG1 was triggered, etc.
    s_hats: list[float]           # ŝ at each iteration
    retrieved_at_iter: list[int]  # iterations where retrieval was performed
    latency_ms: float

def card_pipeline(
    generator,
    retriever,
    estimator,
    x_left: str,
    x_right: str,
    T_RAG_schedule: list[float],   # e.g., [0.9, 0.8, 0.7, 0.6]
    T_ACC_schedule: list[float],   # e.g., [0.8, 0.9, 0.95, 0.99]
    max_iter: int = 1,             # 1 for single-RAG (recommended for our project)
    model_family: str = "qwen",
) -> CARDOutput:
    """
    Algorithm 1 of the CARD paper, single-RAG variant.
    For our project, max_iter=1 (zero-shot generation + at most one RAG iteration).
    """
    s_hats = []
    predictions = []
    retrieved_at = []

    # Iteration 0: zero-shot
    prompt = build_fim_prompt(x_left, x_right, retrieved=None, model_family=model_family)
    g0 = generator.generate(prompt)
    feats0 = extract_features(g0.token_probs, g0.token_entropies)
    s_hat_0 = float(estimator.predict(feats0)[0])
    s_hats.append(s_hat_0)
    predictions.append(g0.prediction)

    # CARD's isRetrieve decision
    if max_iter >= 1 and s_hat_0 < T_RAG_schedule[0]:
        # Retrieve and regenerate
        query = "\n".join(x_left.splitlines()[-20:])
        retrieved = retriever.retrieve(query, top_k=10)
        prompt = build_fim_prompt(x_left, x_right, retrieved=retrieved, model_family=model_family)
        g1 = generator.generate(prompt)
        feats1 = extract_features(g1.token_probs, g1.token_entropies)
        s_hat_1 = float(estimator.predict(feats1)[0])
        s_hats.append(s_hat_1)
        predictions.append(g1.prediction)
        retrieved_at.append(1)

        # Select: do we keep ŷ^0 or accept ŷ^1?
        if select(s_hat_0, s_hat_1, T_ACC_schedule[0]):
            final_prediction = predictions[0]   # keep older
            n_iterations = 0
        else:
            final_prediction = predictions[1]   # accept newer
            n_iterations = 1
    else:
        final_prediction = predictions[0]
        n_iterations = 0

    return CARDOutput(
        prediction=final_prediction,
        n_iterations=n_iterations,
        s_hats=s_hats,
        retrieved_at_iter=retrieved_at,
        latency_ms=0,  # accumulate from generations
    )
```

### 9.5 Constructing the training dataset for the Estimator

**File:** `src/adaptive_retrieval/card/train_data.py`

Following §3.4 of the CARD paper exactly.

```python
"""
Constructs (X, y) pairs for training the Estimator.

Spec (CARD paper §3.4):
- 11k Python repos from The Stack with 50-100 files each.
- Files filtered to: ≥3 local imports, >20 non-empty lines.
- (X, y) sampling: y = block of code, |y| in lines ~ Poisson(λ=2); X = 50 lines before y.
- K-Means deduplication with cluster_ratio=0.2 (following Wu et al. 2024 / Repoformer).
- Final dataset: 250k pairs.
- Then: for each pair, run generator on X to produce (ŷ, logits) and compute ES(y, ŷ).
"""

import re
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
import numpy as np

POISSON_LAMBDA = 2
LINES_X = 50
MIN_LOCAL_IMPORTS = 3
MIN_NONEMPTY_LINES = 20
TARGET_PAIRS = 250_000
CLUSTER_RATIO = 0.2  # from Repoformer Appendix D


def count_local_imports(content: str) -> int:
    """Count `from .module import X` and `from .X import Y` style imports."""
    return len(re.findall(r"^from\s+\.\S*\s+import", content, re.MULTILINE))


def is_valid_file(content: str) -> bool:
    nonempty = sum(1 for line in content.splitlines() if line.strip())
    return count_local_imports(content) >= MIN_LOCAL_IMPORTS and nonempty > MIN_NONEMPTY_LINES


def sample_pair(file_content: str, rng) -> tuple[str, str] | None:
    """Sample (X, y) from a file. Returns None if file too short."""
    lines = file_content.splitlines()
    k = max(1, rng.poisson(POISSON_LAMBDA))
    if len(lines) < LINES_X + k + 5:
        return None
    # Random position for y, with at least LINES_X above
    y_start = rng.integers(LINES_X, len(lines) - k)
    y = "\n".join(lines[y_start:y_start + k])
    x = "\n".join(lines[max(0, y_start - LINES_X):y_start])
    return x, y


def kmeans_deduplicate(pairs: list[tuple[str, str]], cluster_ratio=0.2, random_state=42):
    """Keep one representative per cluster."""
    texts = [x + "\n" + y for x, y in pairs]
    vec = TfidfVectorizer(max_features=2000)
    X = vec.fit_transform(texts)
    n_clusters = max(1, int(len(pairs) * cluster_ratio))
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=4096)
    labels = km.fit_predict(X)
    # Keep first sample of each cluster
    seen = set()
    dedup = []
    for pair, label in zip(pairs, labels):
        if label not in seen:
            seen.add(label)
            dedup.append(pair)
    return dedup


def construct_training_data(
    generator,
    n_target_pairs: int = TARGET_PAIRS,
    n_repos: int = 11_000,
    random_state: int = 42,
):
    """Main entry. Returns (features_array, scores_array)."""
    rng = np.random.default_rng(random_state)

    # 1. Load Stack subset, filter
    ds = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train")

    # Group files by repository
    repo_files = {}
    for ex in ds:
        repo = ex["repository_name"]
        repo_files.setdefault(repo, []).append(ex["content"])

    # Filter repos: 50-100 files; filter files: imports/length
    eligible_repos = []
    for repo, files in repo_files.items():
        if not (50 <= len(files) <= 100):
            continue
        valid = [f for f in files if is_valid_file(f)]
        if len(valid) >= 5:
            eligible_repos.append(valid)

    print(f"Found {len(eligible_repos)} eligible repos")

    # 2. Sample (X, y) pairs
    raw_pairs = []
    while len(raw_pairs) < n_target_pairs * 2 and eligible_repos:  # oversample to allow dedup
        repo = eligible_repos[rng.integers(0, len(eligible_repos))]
        file_content = repo[rng.integers(0, len(repo))]
        pair = sample_pair(file_content, rng)
        if pair:
            raw_pairs.append(pair)

    # 3. K-Means dedup
    pairs = kmeans_deduplicate(raw_pairs, cluster_ratio=CLUSTER_RATIO,
                                random_state=random_state)[:n_target_pairs]
    print(f"After dedup: {len(pairs)} pairs")

    # 4. Generate ŷ for each X, compute features and ES
    from ..metrics import edit_similarity
    from .features import extract_features

    features_list = []
    scores_list = []

    batch_size = 32
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        prompts = [x for x, _ in batch]      # Or wrap in FIM
        generations = generator.generate_batch(prompts)
        for (x, y), gen in zip(batch, generations):
            feats = extract_features(gen.token_probs, gen.token_entropies)
            es = edit_similarity(y, gen.prediction)
            features_list.append(feats)
            scores_list.append(es)

    return np.stack(features_list), np.array(scores_list)
```

### 9.6 Thresholds (from CARD paper §3.4)

For **single-RAG** (max_iter=1), only the first threshold in each schedule is used.

| Task | T_RAG (iter 1) | T_ACC (iter 1) |
|------|----------------|----------------|
| Line completion | 0.9 | 0.8 |
| API completion  | 0.9 | 0.8 |
| Function completion | 0.65 | 0.9 |

For the full 4-iteration schedule (if extending later):

| | T_RAG | T_ACC |
|---|---|---|
| Line/API | [0.9, 0.8, 0.7, 0.6] | [0.8, 0.9, 0.95, 0.99] |
| Function | [0.65, 0.45, 0.3, 0.25] | [0.9, 0.9, 0.95, 0.99] |

---

## 10. Module: Static analysis

This is the **novel contribution** and gets the most careful attention.

### 10.1 Tree-sitter setup

**File:** `src/adaptive_retrieval/static_analysis/parser.py`

```python
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

PY_LANGUAGE = Language(tspython.language())

def get_parser():
    parser = Parser(PY_LANGUAGE)
    return parser

def parse(source: str | bytes):
    if isinstance(source, str):
        source = source.encode("utf-8")
    return get_parser().parse(source)
```

### 10.2 Repository symbol table

**File:** `src/adaptive_retrieval/static_analysis/symbol_table.py`

```python
from collections import defaultdict
from pathlib import Path
import builtins as _builtins

PYTHON_BUILTINS = set(dir(_builtins)) | {"self", "cls"}
COMMON_LIBS = {
    "numpy", "np", "pandas", "pd", "torch", "tf", "tensorflow",
    "sklearn", "matplotlib", "plt", "os", "sys", "re", "json",
    "math", "random", "time", "datetime", "collections", "typing",
    # Add more as needed
}

class RepositorySymbolTable:
    """All names defined anywhere in the repository's .py files."""

    def __init__(self, repo_root: str | Path):
        self.symbols: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
        # name -> [(file, kind, line), ...]
        self._build(Path(repo_root))

    def _build(self, root: Path):
        from .parser import parse, PY_LANGUAGE
        for py_file in root.rglob("*.py"):
            try:
                src = py_file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            tree = parse(src)
            self._walk_defs(tree.root_node, src.encode("utf-8"), str(py_file))

    def _walk_defs(self, node, src_bytes, file_path):
        """Collect function defs, class defs, and assignments."""
        if node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = src_bytes[name_node.start_byte:name_node.end_byte].decode()
                self.symbols[name].append((file_path, "function", name_node.start_point[0]))
        elif node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = src_bytes[name_node.start_byte:name_node.end_byte].decode()
                self.symbols[name].append((file_path, "class", name_node.start_point[0]))
        elif node.type == "assignment":
            # Top-level assignments at module level
            left = node.child_by_field_name("left")
            if left and left.type == "identifier":
                name = src_bytes[left.start_byte:left.end_byte].decode()
                self.symbols[name].append((file_path, "variable", left.start_point[0]))
        for child in node.children:
            self._walk_defs(child, src_bytes, file_path)

    def contains(self, name: str) -> bool:
        return name in self.symbols or name in PYTHON_BUILTINS

    def __len__(self):
        return len(self.symbols)
```

### 10.3 In-file scope analyzer

**File:** `src/adaptive_retrieval/static_analysis/scope.py`

Goal: given the file's contents and the hole position (byte offset), return the set of identifiers visible at that point.

```python
import builtins as _builtins
from .parser import parse

PYTHON_BUILTINS = set(dir(_builtins)) | {"self", "cls", "True", "False", "None"}

class InFileScopeAnalyzer:
    def visible_at(self, source: str, hole_byte: int) -> set[str]:
        """Return identifiers visible at the hole position."""
        tree = parse(source)
        src_bytes = source.encode("utf-8")

        visible = set(PYTHON_BUILTINS)

        # 1. Imports anywhere in the file (Python imports are module-level)
        visible |= self._collect_imports(tree.root_node, src_bytes)

        # 2. Top-level function/class names (Python allows forward references for these
        #    if they're only invoked at runtime, but we keep it simple: all top-level names)
        visible |= self._collect_module_level_names(tree.root_node, src_bytes)

        # 3. Local names from the enclosing function/class up to the hole
        visible |= self._collect_local_names_up_to(tree.root_node, src_bytes, hole_byte)

        return visible

    def _collect_imports(self, node, src):
        """Extract names introduced by `import X`, `from X import Y, Z`."""
        names = set()
        def walk(n):
            if n.type == "import_statement":
                # e.g., `import os`, `import numpy as np`
                for child in n.children:
                    if child.type == "dotted_name":
                        names.add(src[child.start_byte:child.end_byte].decode().split(".")[0])
                    elif child.type == "aliased_import":
                        alias = child.child_by_field_name("alias")
                        if alias:
                            names.add(src[alias.start_byte:alias.end_byte].decode())
            elif n.type == "import_from_statement":
                # e.g., `from X import Y` or `from .x import Y`
                for child in n.children:
                    if child.type == "dotted_name" and child.prev_sibling and \
                            src[child.prev_sibling.start_byte:child.prev_sibling.end_byte] == b"import":
                        names.add(src[child.start_byte:child.end_byte].decode().split(".")[0])
                    elif child.type == "aliased_import":
                        alias = child.child_by_field_name("alias")
                        if alias:
                            names.add(src[alias.start_byte:alias.end_byte].decode())
                # Also: names directly listed after `import` keyword
                # tree-sitter-python represents this as identifier children
            for c in n.children:
                walk(c)
        walk(node)
        return names

    def _collect_module_level_names(self, root, src):
        names = set()
        for child in root.children:
            if child.type == "function_definition":
                nn = child.child_by_field_name("name")
                if nn:
                    names.add(src[nn.start_byte:nn.end_byte].decode())
            elif child.type == "class_definition":
                nn = child.child_by_field_name("name")
                if nn:
                    names.add(src[nn.start_byte:nn.end_byte].decode())
            elif child.type == "assignment":
                left = child.child_by_field_name("left")
                if left and left.type == "identifier":
                    names.add(src[left.start_byte:left.end_byte].decode())
        return names

    def _collect_local_names_up_to(self, root, src, hole_byte):
        """Find names bound in scope at hole_byte (assignments, params, for-loop vars, etc.)."""
        names = set()

        def find_enclosing_func(node):
            best = None
            def walk(n):
                nonlocal best
                if n.type in ("function_definition", "lambda"):
                    if n.start_byte <= hole_byte <= n.end_byte:
                        best = n
                for c in n.children:
                    walk(c)
            walk(node)
            return best

        enclosing = find_enclosing_func(root)
        if enclosing is None:
            return names

        # Parameters of the enclosing function
        params = enclosing.child_by_field_name("parameters")
        if params:
            for param in params.children:
                if param.type == "identifier":
                    names.add(src[param.start_byte:param.end_byte].decode())
                elif param.type in ("default_parameter", "typed_parameter",
                                     "typed_default_parameter"):
                    pname = param.child_by_field_name("name")
                    if pname:
                        names.add(src[pname.start_byte:pname.end_byte].decode())

        # Assignments, for-vars, with-vars, except-vars between function start and hole
        def walk_body(n):
            if n.start_byte > hole_byte:
                return
            if n.type == "assignment":
                left = n.child_by_field_name("left")
                if left and left.type == "identifier":
                    names.add(src[left.start_byte:left.end_byte].decode())
                elif left and left.type == "pattern_list":
                    for sub in left.children:
                        if sub.type == "identifier":
                            names.add(src[sub.start_byte:sub.end_byte].decode())
            elif n.type == "for_statement":
                left = n.child_by_field_name("left")
                if left and left.type == "identifier":
                    names.add(src[left.start_byte:left.end_byte].decode())
            elif n.type == "as_pattern":
                alias = n.child_by_field_name("alias")
                if alias and alias.type == "identifier":
                    names.add(src[alias.start_byte:alias.end_byte].decode())
            for c in n.children:
                walk_body(c)

        walk_body(enclosing)
        return names
```

### 10.4 Prediction analyzer (the main novel logic)

**File:** `src/adaptive_retrieval/static_analysis/analyzer.py`

```python
from dataclasses import dataclass
from .parser import parse
from .scope import InFileScopeAnalyzer
from .symbol_table import RepositorySymbolTable

@dataclass
class StaticAnalysisResult:
    fires: bool                       # True = retrieval should be triggered
    unresolved_identifiers: list[str] # names used in prediction but unresolved
    cross_file_identifiers: list[str] # names resolved only in repo (not in-file)
    n_used_identifiers: int           # total non-builtin identifier uses

class PredictionAnalyzer:
    """
    Decides whether retrieval should be triggered by inspecting the prediction.

    A name used in the prediction is classified as:
      - BUILTIN:    Python built-in -> not interesting
      - IN_FILE:    visible in current scope -> not interesting
      - CROSS_FILE: in repo symbol table -> FIRE (retrieval would bring it into context)
      - UNRESOLVED: nowhere -> FIRE (likely hallucination)
    """

    def __init__(self,
                 scope_analyzer: InFileScopeAnalyzer,
                 repo_symbols: RepositorySymbolTable,
                 fire_on_crossfile: bool = True,
                 fire_on_unresolved: bool = True):
        self.scope = scope_analyzer
        self.repo = repo_symbols
        self.fire_on_crossfile = fire_on_crossfile
        self.fire_on_unresolved = fire_on_unresolved

    def analyze(self, prediction: str, x_left: str, x_right: str) -> StaticAnalysisResult:
        """
        Args:
            prediction: the model's output ŷ₀
            x_left, x_right: in-file context surrounding the hole

        Returns: StaticAnalysisResult
        """
        # Build the combined file with the prediction inserted at the hole
        full_source = x_left + prediction + x_right
        hole_byte = len(x_left.encode("utf-8"))

        # Visible names at the hole
        visible = self.scope.visible_at(full_source, hole_byte)

        # Extract identifier uses from the prediction only
        used_names = self._extract_used_identifiers(prediction)

        cross_file = []
        unresolved = []
        for name in used_names:
            if name in visible:
                continue
            if self.repo.contains(name):
                cross_file.append(name)
            else:
                unresolved.append(name)

        fires = (self.fire_on_crossfile and len(cross_file) > 0) or \
                (self.fire_on_unresolved and len(unresolved) > 0)

        return StaticAnalysisResult(
            fires=fires,
            unresolved_identifiers=unresolved,
            cross_file_identifiers=cross_file,
            n_used_identifiers=len(used_names),
        )

    def _extract_used_identifiers(self, code: str) -> set[str]:
        """Extract names that are USED (not defined) in the code snippet."""
        # We parse just the snippet (may have parse errors; tree-sitter is fault-tolerant)
        tree = parse(code)
        src_bytes = code.encode("utf-8")
        used = set()
        defined_locally = set()

        def walk(node):
            # First collect locally defined names so we don't flag them as unresolved.
            if node.type == "assignment":
                left = node.child_by_field_name("left")
                if left and left.type == "identifier":
                    defined_locally.add(src_bytes[left.start_byte:left.end_byte].decode())
            elif node.type == "function_definition":
                nn = node.child_by_field_name("name")
                if nn:
                    defined_locally.add(src_bytes[nn.start_byte:nn.end_byte].decode())
            elif node.type == "for_statement":
                left = node.child_by_field_name("left")
                if left and left.type == "identifier":
                    defined_locally.add(src_bytes[left.start_byte:left.end_byte].decode())

            # Now collect identifier uses
            if node.type == "identifier":
                name = src_bytes[node.start_byte:node.end_byte].decode()
                # Skip if this identifier is the LHS of an assignment
                parent = node.parent
                is_def = False
                if parent:
                    if parent.type == "assignment" and \
                            parent.child_by_field_name("left") == node:
                        is_def = True
                    elif parent.type in ("function_definition", "class_definition") and \
                            parent.child_by_field_name("name") == node:
                        is_def = True
                if not is_def:
                    used.add(name)

            # Attribute access: foo.bar.baz - treat 'foo' as a use, ignore attributes
            # tree-sitter handles this: 'foo' is an identifier child of attribute node

            for c in node.children:
                walk(c)

        walk(tree.root_node)
        return used - defined_locally
```

### 10.5 Static analysis: testing

**File:** `tests/test_static_analysis.py`

```python
from src.adaptive_retrieval.static_analysis.scope import InFileScopeAnalyzer
from src.adaptive_retrieval.static_analysis.symbol_table import RepositorySymbolTable
from src.adaptive_retrieval.static_analysis.analyzer import PredictionAnalyzer
from pathlib import Path

def test_unresolved_identifier_fires(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def known_func():\n    pass\n")

    syms = RepositorySymbolTable(repo)
    scope = InFileScopeAnalyzer()
    analyzer = PredictionAnalyzer(scope, syms)

    x_left = "def use_it():\n    result = "
    x_right = "\n    return result\n"
    prediction = "totally_made_up_function()"

    r = analyzer.analyze(prediction, x_left, x_right)
    assert r.fires
    assert "totally_made_up_function" in r.unresolved_identifiers


def test_in_file_name_does_not_fire(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    syms = RepositorySymbolTable(repo)
    scope = InFileScopeAnalyzer()
    analyzer = PredictionAnalyzer(scope, syms)

    x_left = "def helper():\n    return 1\n\ndef caller():\n    x = "
    x_right = "\n    return x\n"
    prediction = "helper()"   # 'helper' is defined in the file

    r = analyzer.analyze(prediction, x_left, x_right)
    assert not r.fires


def test_cross_file_resolved_fires(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "lib.py").write_text("def cross_file_func():\n    pass\n")
    (repo / "main.py").write_text("from lib import cross_file_func\n")

    syms = RepositorySymbolTable(repo)
    scope = InFileScopeAnalyzer()
    analyzer = PredictionAnalyzer(scope, syms)

    # X_left does NOT import cross_file_func; prediction uses it
    x_left = "def f():\n    return "
    x_right = "\n"
    prediction = "cross_file_func()"

    r = analyzer.analyze(prediction, x_left, x_right)
    assert r.fires
    assert "cross_file_func" in r.cross_file_identifiers
```

---

## 11. Module: The cascade (our contribution)

**File:** `src/adaptive_retrieval/cascade.py`

```python
from dataclasses import dataclass
from .generator import Generator
from .retriever import BM25Retriever
from .card.estimator import Estimator
from .card.features import extract_features
from .static_analysis.analyzer import PredictionAnalyzer
from .prompt import build_fim_prompt

@dataclass
class CascadeOutput:
    prediction: str
    retrieved: bool
    trigger_reason: str          # "none", "card", "static_unresolved", "static_crossfile"
    s_hat_0: float               # CARD's estimated ES for ŷ₀
    static_unresolved: list[str]
    static_crossfile: list[str]
    latency_ms: float

def cascade_pipeline(
    generator: Generator,
    retriever: BM25Retriever,
    estimator: Estimator,
    analyzer: PredictionAnalyzer,
    x_left: str,
    x_right: str,
    T_RAG: float = 0.9,
    model_family: str = "qwen",
) -> CascadeOutput:
    """
    CARD + static-analysis cascade.

    Stage 1: generate ŷ₀ without retrieval.
    Stage 2: CARD's isRetrieve. If yes -> retrieve.
    Stage 3: Static analysis on ŷ₀. If fires -> retrieve.
    """
    # Stage 1: no-retrieval generation
    prompt = build_fim_prompt(x_left, x_right, retrieved=None, model_family=model_family)
    g0 = generator.generate(prompt)
    feats0 = extract_features(g0.token_probs, g0.token_entropies)
    s_hat_0 = float(estimator.predict(feats0)[0])

    # Stage 2: CARD's isRetrieve
    if s_hat_0 < T_RAG:
        return _retrieve_and_regenerate(
            generator, retriever, x_left, x_right, model_family,
            trigger_reason="card", s_hat_0=s_hat_0,
            static_unresolved=[], static_crossfile=[], g0=g0,
        )

    # Stage 3: Static analysis on ŷ₀
    sa_result = analyzer.analyze(g0.prediction, x_left, x_right)
    if sa_result.fires:
        reason = "static_unresolved" if sa_result.unresolved_identifiers else "static_crossfile"
        return _retrieve_and_regenerate(
            generator, retriever, x_left, x_right, model_family,
            trigger_reason=reason, s_hat_0=s_hat_0,
            static_unresolved=sa_result.unresolved_identifiers,
            static_crossfile=sa_result.cross_file_identifiers,
            g0=g0,
        )

    # No retrieval
    return CascadeOutput(
        prediction=g0.prediction,
        retrieved=False, trigger_reason="none",
        s_hat_0=s_hat_0,
        static_unresolved=[], static_crossfile=[],
        latency_ms=g0.latency_ms,
    )


def _retrieve_and_regenerate(generator, retriever, x_left, x_right, model_family,
                              trigger_reason, s_hat_0, static_unresolved, static_crossfile, g0):
    query = "\n".join(x_left.splitlines()[-20:])
    retrieved = retriever.retrieve(query, top_k=10)
    prompt = build_fim_prompt(x_left, x_right, retrieved=retrieved, model_family=model_family)
    g_rag = generator.generate(prompt)
    return CascadeOutput(
        prediction=g_rag.prediction,
        retrieved=True, trigger_reason=trigger_reason,
        s_hat_0=s_hat_0,
        static_unresolved=static_unresolved,
        static_crossfile=static_crossfile,
        latency_ms=g0.latency_ms + g_rag.latency_ms,
    )
```

---

## 12. Module: Baselines

**File:** `src/adaptive_retrieval/baselines.py`

```python
from .generator import Generator
from .retriever import BM25Retriever
from .prompt import build_fim_prompt

def no_retrieve_baseline(generator, x_left, x_right, model_family="qwen"):
    prompt = build_fim_prompt(x_left, x_right, retrieved=None, model_family=model_family)
    return generator.generate(prompt)

def always_retrieve_baseline(generator, retriever, x_left, x_right, model_family="qwen"):
    query = "\n".join(x_left.splitlines()[-20:])
    retrieved = retriever.retrieve(query, top_k=10)
    prompt = build_fim_prompt(x_left, x_right, retrieved=retrieved, model_family=model_family)
    return generator.generate(prompt)
```

---

## 13. Metrics

**File:** `src/adaptive_retrieval/metrics.py`

### 13.1 Standard accuracy metrics

```python
import re
from Levenshtein import distance as lev_distance

def exact_match(reference: str, prediction: str) -> bool:
    """Strict string equality after stripping trailing whitespace."""
    return reference.rstrip() == prediction.rstrip()

def edit_similarity(reference: str, prediction: str) -> float:
    """ES per CARD §2.1. Returns value in [0, 1], higher = better."""
    if not reference and not prediction:
        return 1.0
    return 1.0 - lev_distance(reference, prediction) / max(len(reference), len(prediction))

def _identifiers(text: str) -> list[str]:
    """Extract identifier-like tokens via regex (CrossCodeEval's approach)."""
    return re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text)

def identifier_f1(reference: str, prediction: str) -> float:
    """CrossCodeEval-style Identifier-F1."""
    ref_ids = set(_identifiers(reference))
    pred_ids = set(_identifiers(prediction))
    if not ref_ids and not pred_ids:
        return 1.0
    if not ref_ids or not pred_ids:
        return 0.0
    tp = len(ref_ids & pred_ids)
    precision = tp / len(pred_ids) if pred_ids else 0.0
    recall = tp / len(ref_ids) if ref_ids else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
```

### 13.2 Hallucination metrics (novel for this project)

```python
from .static_analysis.analyzer import PredictionAnalyzer

def repository_symbol_precision(prediction: str, x_left: str, x_right: str,
                                  analyzer: PredictionAnalyzer) -> float:
    """
    Of all non-trivial identifiers in the prediction, what fraction resolve
    (either in-file or anywhere in the repo)?

    Higher = fewer hallucinations.
    """
    result = analyzer.analyze(prediction, x_left, x_right)
    n_total = result.n_used_identifiers
    if n_total == 0:
        return 1.0  # No identifiers used; vacuously precise
    n_resolved = n_total - len(result.unresolved_identifiers)
    # NOTE: cross_file identifiers DO resolve (in the repo), so they count as resolved.
    return n_resolved / n_total


def hallucination_flag(prediction: str, x_left: str, x_right: str,
                       analyzer: PredictionAnalyzer) -> bool:
    """
    True iff prediction contains AT LEAST ONE unresolved identifier.
    The primary per-instance hallucination indicator.
    """
    result = analyzer.analyze(prediction, x_left, x_right)
    return len(result.unresolved_identifiers) > 0
```

### 13.3 Efficiency metrics

These are computed at aggregate level over an experiment run, not per-instance:

```python
def percent_retrieval(records: list[dict]) -> float:
    """% of instances on which retrieval was performed."""
    return 100.0 * sum(r["retrieved"] for r in records) / len(records)

def mean_latency_ms(records: list[dict]) -> float:
    return sum(r["latency_ms"] for r in records) / len(records)
```

### 13.4 Statistical tests

```python
from scipy.stats import binomtest

def mcnemar_test(records_a: list[dict], records_b: list[dict],
                  key: str = "hallucinated") -> dict:
    """
    Paired McNemar test for binary outcomes.
    Used for: hallucination_flag comparison between CARD and Cascade.
    """
    assert len(records_a) == len(records_b)
    # b = cases where A=0, B=1 (B is worse)
    # c = cases where A=1, B=0 (B is better)
    b = sum(1 for a, b in zip(records_a, records_b) if not a[key] and b[key])
    c = sum(1 for a, b in zip(records_a, records_b) if a[key] and not b[key])
    if b + c == 0:
        return {"p_value": 1.0, "b": 0, "c": 0}
    # Exact binomial test
    test_result = binomtest(min(b, c), b + c, p=0.5)
    return {"p_value": test_result.pvalue, "b": b, "c": c}


def paired_bootstrap(records_a, records_b, key: str, n_resamples: int = 10_000):
    """Paired bootstrap for continuous metric differences (e.g. ES)."""
    import numpy as np
    a_vals = np.array([r[key] for r in records_a])
    b_vals = np.array([r[key] for r in records_b])
    diffs = b_vals - a_vals
    rng = np.random.default_rng(42)
    boot_means = []
    n = len(diffs)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot_means.append(diffs[idx].mean())
    boot_means = np.array(boot_means)
    return {
        "mean_diff": diffs.mean(),
        "ci_lower": np.percentile(boot_means, 2.5),
        "ci_upper": np.percentile(boot_means, 97.5),
    }
```

---

## 14. Experiment design

### 14.1 Configurations to run

Each configuration produces a list of records (one per instance) with all metrics.

| Config ID | Name | Description |
|-----------|------|-------------|
| `C1_no_retrieve` | No retrieval | Generator on in-file prompt only |
| `C2_always_retrieve` | Always retrieve | Generator with BM25 top-10 chunks |
| `C3_card` | CARD only | Single-RAG CARD pipeline |
| `C4_cascade` | **CARD + Static (ours)** | The cascade pipeline |
| `C5_static_only` | Static only (ablation) | Trigger retrieval only on static-analysis fire |
| `C6_oracle` | Oracle upper bound | Run both no-retrieve and always-retrieve, pick the one with higher ES against ground truth |

### 14.2 Datasets to run on

- **CrossCodeEval-Python (~2,460 instances)** — primary, all 6 configs.
- **RepoEval-line (1,600 instances)** — secondary, all 6 configs. Used for CARD validation.
- **RepoEval-API (1,600 instances)** — secondary, all 6 configs.
- **RepoEval-function (373 instances)** — optional/Week 4, all 6 configs. Note: ES isn't ideal for function completion; use UT (unit test pass rate) if implementable, else just ES.

### 14.3 Ablations (CrossCodeEval-Python only)

| Ablation | Description |
|----------|-------------|
| A1: Static strictness | Cascade with `fire_on_crossfile=True, fire_on_unresolved=True` vs `True/False` vs `False/True` |
| A2: T_RAG sweep | Cascade with T_RAG ∈ {0.7, 0.8, 0.9} |
| A3: Top-k for retrieval | k ∈ {5, 10, 20} |

### 14.4 Run order

1. **Week 2:** Validate CARD reimplementation on RepoEval-line by reproducing CARD paper's Table 3 within ±1% ES.
2. **Week 3 day 1–2:** Main experiments — C1 through C6 on CrossCodeEval-Python (the largest run).
3. **Week 3 day 3:** C1 through C6 on RepoEval-line and RepoEval-API.
4. **Week 3 day 4:** Ablations A1–A3 on CrossCodeEval-Python.
5. **Week 3 day 5:** Analysis runs (per-trigger-reason breakdown, disagreement analysis).

---

## 15. Logging and reproducibility

### 15.1 Per-instance log format

Every experiment run produces a JSONL file in `results/`. Each line is one instance:

```json
{
  "instance_id": "crosscodeeval_python_0042",
  "config": "C4_cascade",
  "dataset": "crosscodeeval_python",
  "repository": "owner/repo",
  "ground_truth": "self.client.send(message)",
  "prediction": "self.client.send(msg)",
  "retrieved": true,
  "trigger_reason": "static_unresolved",
  "s_hat_0": 0.72,
  "static_unresolved": ["msg"],
  "static_crossfile": [],
  "metrics": {
    "exact_match": false,
    "edit_similarity": 0.92,
    "identifier_f1": 0.857,
    "repo_symbol_precision": 1.0,
    "hallucinated": false
  },
  "latency_ms": 1340.5
}
```

### 15.2 Aggregate results format

After all configs complete, generate `results/aggregate.json`:

```json
{
  "C4_cascade": {
    "dataset": "crosscodeeval_python",
    "n_instances": 2460,
    "n_retrieved": 1480,
    "percent_retrieval": 60.16,
    "metrics": {
      "exact_match": 0.421,
      "edit_similarity": 0.764,
      "identifier_f1": 0.692,
      "repo_symbol_precision": 0.943,
      "hallucination_rate": 0.124
    },
    "latency_ms_mean": 845.3
  }
}
```

### 15.3 Reproducibility

- **Random seeds.** Pin numpy and torch seeds to `42` everywhere a seed is consumed.
- **Generator decoding.** Use greedy decoding (`temperature=0.0`).
- **Cache generations.** Every generator call should hash its prompt and cache the result. A second run with the same prompt returns the cached generation. This is critical: regenerating ~90k completions costs real GPU hours.

```python
import hashlib, json
from pathlib import Path

CACHE_DIR = Path("data/generation_cache")
CACHE_DIR.mkdir(exist_ok=True, parents=True)

def cache_key(model_name: str, prompt: str, max_tokens: int) -> str:
    return hashlib.sha256(f"{model_name}::{prompt}::{max_tokens}".encode()).hexdigest()
```

---

## 16. Implementation order and validation checkpoints

This section is the most important part for a Claude Code agent. Follow this order; don't skip.

### Phase 0 — Setup (day 1)

```bash
mkdir adaptive-retrieval && cd adaptive-retrieval
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir -p src/adaptive_retrieval/{card,static_analysis,eval} scripts results data tests
touch src/adaptive_retrieval/__init__.py
```

**Validation:** `python -c "import vllm; import lightgbm; import tree_sitter_python; print('ok')"` succeeds.

### Phase 1 — Static analysis (days 2–4)

Build `parser.py`, `symbol_table.py`, `scope.py`, `analyzer.py` in that order.

**Validation:** All tests in `tests/test_static_analysis.py` pass:
```bash
pytest tests/test_static_analysis.py -v
```

Manually inspect 10 hand-crafted examples covering:
1. Hallucinated function name → fires
2. Locally defined function → doesn't fire
3. Imported library function → doesn't fire
4. Cross-file repo function → fires
5. Attribute access on imported object → doesn't fire
6. Attribute access with hallucinated attribute name → fires
7. Self-reference in method → doesn't fire
8. List comprehension with locally-bound variable → doesn't fire
9. Decorator usage with imported decorator → doesn't fire
10. Lambda with locally-defined name → doesn't fire

### Phase 2 — Generator + retriever (days 3–4)

Build `generator.py` and `retriever.py`. Get the no-retrieve and always-retrieve baselines running end-to-end on 10 instances of CrossCodeEval-Python.

**Validation:**
- `Generator.generate("def hello():\n    return ")` returns reasonable output with non-empty `token_probs`.
- `BM25Retriever` over a 10-file toy repo returns sensible top-3 chunks for a known query.
- Both baselines complete 10 CrossCodeEval instances; metrics computed without errors.

### Phase 3 — CARD reimplementation (days 5–9)

Build CARD modules in this order:
1. `features.py` (~30 lines, easiest)
2. `pipeline.py` (Algorithm 1 wrapper)
3. `train_data.py` (the time sink — 24-hour data generation)
4. `estimator.py` (~50 lines)

**Validation:**
- `test_features.py`: feature vector has shape (13,), values in expected ranges, log-space computation doesn't overflow/underflow.
- After training Estimator: MSE on validation set < 0.10 (per CARD paper Table 8, their MSE is ~0.07).
- Running CARD single-RAG on **RepoEval-line with CodeLlama-7B** reproduces CARD paper's published `CARD-RG1` numbers within ±1 absolute % on ES.

**Critical sanity check.** From CARD paper Table 3:
- CodeLlama-7B + RepoEval-line: zero-shot ES = 59.42%, RG1 = 71.83%, CARD-RG1 = 72.26%.
- If our reimplementation gives CARD-RG1 ES on RepoEval-line within [71%, 73%], we accept it. Outside that range, debug before proceeding.

### Phase 4 — Cascade integration (days 10–11)

Build `cascade.py`. Wire together generator + retriever + estimator + analyzer.

**Validation:** Run cascade on a curated 20-instance subset of CrossCodeEval-Python. Manually inspect the `trigger_reason` field: it should be a mix of `"card"`, `"static_unresolved"`, `"static_crossfile"`, and `"none"`. If 100% are one reason, something's wrong.

### Phase 5 — Evaluation infrastructure (days 11–12)

Build:
- `eval/datasets.py` — loaders for CrossCodeEval and RepoEval.
- `eval/runner.py` — runs a config over a dataset, writes per-instance JSONL.
- `scripts/04_run_experiment.py` — CLI wrapper.

**Validation:** Run all 6 configs on 50 instances of CrossCodeEval-Python end-to-end. All JSONL files have the correct shape; aggregate metrics computable.

### Phase 6 — Main experiments (week 3)

Run the full experiment matrix per §14. Cache generations aggressively.

### Phase 7 — Analysis (week 3, end)

Build `scripts/06_analysis.py`:
- Per-trigger-reason breakdown of cascade.
- Disagreement analysis: instances where CARD says no but static fires.
- McNemar test for hallucination rate: CARD vs. Cascade.
- Threshold sweep plot for T_RAG.

### Phase 8 — Writing (week 4)

Paper sections:
1. Introduction (1 page).
2. Background and related work (1 page).
3. Method — describe CARD briefly, then the cascade and the static-analysis signal (2 pages).
4. Experiments — config table, headline results, ablations (3 pages).
5. Analysis — disagreement analysis, qualitative examples (2 pages).
6. Discussion and limitations (0.5 page).
7. Conclusion (0.5 page).

---

## 17. Known risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| CARD reimplementation doesn't reproduce paper numbers | Medium | High | Sanity check in Phase 3. If off by >2% on RepoEval-line, debug feature extraction first (most likely culprit), then Estimator hyperparameters, then prompt template. |
| Static analysis has false positives (fires on legitimate predictions) | High | Medium | Build a confusion matrix of static-analysis fire vs. ground-truth correctness on 100 hand-labeled instances. Tune `fire_on_crossfile` and `fire_on_unresolved` flags. |
| Tree-sitter scope analysis misses edge cases (decorators, type hints, etc.) | High | Low | Accept some imprecision; document failures; report headline numbers with the simpler signal (`fire_on_unresolved` only) if needed. |
| Training-data construction for Estimator takes too long | Medium | Medium | Subsample: use 50k pairs instead of 250k. CARD paper Table 8 shows the Estimator's MSE is similar across reasonable dataset sizes. |
| Compute budget runs out during Week 3 experiments | Medium | High | Use vLLM with continuous batching. Cache generations. Run smallest configs first. Drop RepoEval-function if needed. |
| Identifier-F1 doesn't move much between configs | Low | Medium | If true, fall back on the custom hallucination rate metric, which is more sensitive. |
| Repository symbol table is too noisy (matches anything) | Medium | High | Filter out: common variable names (`x`, `i`, `j`, `result`, `data`, etc.) from the symbol table — these are too common to be diagnostic. |

---

## Appendix A: CARD paper reference numbers

For Phase 3 validation. From Table 3 of CARD (Zhang et al. 2024).

### CodeLlama-7B on RepoEval

| Task | Metric | Zero-shot | RG1 | **CARD-RG1** |
|------|--------|-----------|-----|--------------|
| Line | EM | 33.94% | 52.31% | **52.56%** |
| Line | ES | 59.42% | 71.83% | **72.26%** |
| Line | aART | 0 | 1.0 | **0.79 (-21%)** |
| API | EM | 25.31% | 40.50% | **40.38%** |
| API | ES | 54.82% | 66.90% | **67.01%** |
| API | aART | 0 | 1.0 | **0.82 (-19%)** |
| Function | UT | 28.42% | 34.32% | **35.12%** |
| Function | ES | 38.62% | 48.79% | **48.82%** |
| Function | aART | 0 | 1.0 | **0.94 (-6%)** |

### DeepSeek-Coder-7B on RepoEval

| Task | Metric | Zero-shot | RG1 | **CARD-RG1** |
|------|--------|-----------|-----|--------------|
| Line | EM | 36.25% | 54.56% | **54.87%** |
| Line | ES | 60.98% | 73.23% | **73.59%** |
| Line | aART | 0 | 1.0 | **0.77 (-23%)** |

**aART = accumulative Average Retrieval Times.** For single-RAG, aART < 1.0 means CARD skipped retrieval on some instances.

Our reimplementation target: reproduce these to within ±1 percentage point on ES.

---

## Appendix B: Code skeletons

### B.1 `scripts/01_construct_training_data.py`

```python
import click
from pathlib import Path
from adaptive_retrieval.generator import Generator
from adaptive_retrieval.card.train_data import construct_training_data
import numpy as np

@click.command()
@click.option("--model", default="Qwen/Qwen2.5-Coder-7B")
@click.option("--output", default="data/training_data/qwen25_coder_7b.npz")
@click.option("--n-pairs", default=250_000)
def main(model, output, n_pairs):
    gen = Generator(model)
    features, scores = construct_training_data(gen, n_target_pairs=n_pairs)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, features=features, scores=scores)
    print(f"Saved {len(features)} pairs to {output}")

if __name__ == "__main__":
    main()
```

### B.2 `scripts/02_train_estimator.py`

```python
import click, numpy as np
from adaptive_retrieval.card.estimator import Estimator

@click.command()
@click.option("--data", required=True)
@click.option("--output", required=True)
def main(data, output):
    d = np.load(data)
    est = Estimator.train(d["features"], d["scores"])
    est.save(output)
    print(f"Saved estimator to {output}")

if __name__ == "__main__":
    main()
```

### B.3 `scripts/04_run_experiment.py`

```python
import click, json, jsonlines
from pathlib import Path
from tqdm import tqdm
from adaptive_retrieval.generator import Generator
from adaptive_retrieval.retriever import BM25Retriever
from adaptive_retrieval.card.estimator import Estimator
from adaptive_retrieval.static_analysis.symbol_table import RepositorySymbolTable
from adaptive_retrieval.static_analysis.scope import InFileScopeAnalyzer
from adaptive_retrieval.static_analysis.analyzer import PredictionAnalyzer
from adaptive_retrieval.cascade import cascade_pipeline
from adaptive_retrieval.baselines import no_retrieve_baseline, always_retrieve_baseline
from adaptive_retrieval.card.pipeline import card_pipeline
from adaptive_retrieval.eval.datasets import load_crosscodeeval, load_repoeval
from adaptive_retrieval.metrics import (
    exact_match, edit_similarity, identifier_f1,
    repository_symbol_precision, hallucination_flag,
)

@click.command()
@click.option("--config", type=click.Choice(["C1", "C2", "C3", "C4", "C5", "C6"]))
@click.option("--dataset", type=click.Choice(["crosscodeeval_py", "repoeval_line",
                                                "repoeval_api", "repoeval_function"]))
@click.option("--model", default="Qwen/Qwen2.5-Coder-7B")
@click.option("--estimator-path", required=False)
@click.option("--output", required=True)
@click.option("--limit", default=None, type=int)
def main(config, dataset, model, estimator_path, output, limit):
    # Load dataset
    if dataset == "crosscodeeval_py":
        instances = load_crosscodeeval("python")
    elif dataset.startswith("repoeval"):
        task = dataset.split("_")[1]
        instances = load_repoeval(task)
    if limit:
        instances = instances[:limit]

    # Initialize components
    gen = Generator(model)
    estimator = Estimator.load(estimator_path) if estimator_path else None
    scope_an = InFileScopeAnalyzer()

    # Run
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(output, "w") as writer:
        for inst in tqdm(instances):
            # Build per-instance retriever and analyzer
            retriever = BM25Retriever(inst.repo_files)
            repo_syms = RepositorySymbolTable(inst.repo_root) if inst.repo_root else None
            analyzer = PredictionAnalyzer(scope_an, repo_syms) if repo_syms else None

            # Run config
            if config == "C1":
                out = no_retrieve_baseline(gen, inst.x_left, inst.x_right)
                record = _record_baseline(inst, out, retrieved=False)
            elif config == "C2":
                out = always_retrieve_baseline(gen, retriever, inst.x_left, inst.x_right)
                record = _record_baseline(inst, out, retrieved=True)
            elif config == "C3":
                out = card_pipeline(gen, retriever, estimator,
                                      inst.x_left, inst.x_right, T_RAG_schedule=[0.9], T_ACC_schedule=[0.8])
                record = _record_card(inst, out)
            elif config == "C4":
                out = cascade_pipeline(gen, retriever, estimator, analyzer,
                                        inst.x_left, inst.x_right, T_RAG=0.9)
                record = _record_cascade(inst, out, analyzer)
            # ... etc

            writer.write(record)

if __name__ == "__main__":
    main()
```

---

## Appendix C: Common pitfalls

### C.1 Probability underflow in feature extraction

Computing `np.prod(token_probs)` for 50 tokens with probabilities like 0.3 gives ~`8.5e-27` — still representable in float32, but for longer sequences it underflows to 0. **Always use log-space for the Product and Geometric Average operators** (already in the code above).

### C.2 Tokenization mismatch in entropy computation

vLLM's `logprobs=K` returns top-K logprobs per generated token. Computing entropy over only K probabilities **underestimates** the true entropy (which sums over the full vocab). For K=50 the underestimate is ~5% in practice. We accept this as it's consistent across all configs.

### C.3 CrossCodeEval prompt format

The `prompt` field of CrossCodeEval has trailing newlines and doesn't include the right context. Use the `prompt` field as `x_left` and `right_context` field as `x_right`. **Do not include the `crossfile_context` field in the in-file prompt**; this field is the gold cross-file context and would leak.

### C.4 Per-instance repo loading — DECISION

CrossCodeEval and RepoEval provide cross-file context but not always the full repository. For BM25 retrieval and the repository symbol table to work, we need a corpus of "the rest of the repo".

**Decision: use Option A throughout the project.**

- **CrossCodeEval.** Each instance ships with a `crossfile_context` field containing pre-extracted chunks from elsewhere in the repository (with file paths and content). Treat this list of chunks as the per-instance "repository". BM25 indexes these chunks; the repository symbol table is built by parsing the same chunks as if they were source files.
- **RepoEval.** Each instance's `metadata.fpath_tuple` identifies the target file inside the dataset's `repositories/` directory, which is shipped with the RepoCoder GitHub repo (clone it once, the `repositories/` folder is the corpus). Per instance, the "repository" is the directory tree rooted at the repo containing `fpath_tuple[0]`.

Rationale: Option A (use shipped contexts) is reproducible, version-stable, and matches what CARD and Repoformer did. Option B (clone repos from GitHub at original commits) introduces a moving target — repos get deleted, force-pushed, or renamed. The realism loss from Option A is acceptable for our research questions because both the cascade and CARD see the same retrieval corpus.

**Caveat for the static-analysis module.** The repository symbol table built from `crossfile_context` will be incomplete (it only knows about chunks that happened to be selected as cross-file context for some instance). This is fine for our purposes because:
1. The `fire_on_unresolved` signal is unaffected — if a name doesn't appear in any chunk and isn't in-file, it's still classified as unresolved.
2. The `fire_on_crossfile` signal will under-fire on names that exist in the repo but aren't in any cross-file context. Document this limitation in the paper; it makes our hallucination-reduction result a lower bound, which is the safer direction.

### C.5 Static analysis on syntactically broken predictions

Sometimes the model outputs syntactically broken code (e.g., missing closing parenthesis). `tree-sitter` is fault-tolerant and will still parse, but with `ERROR` nodes. The analyzer should:
- Not crash on parse errors.
- Still extract identifiers it can find (best-effort).
- Optionally, treat predictions with too many parse errors as definitely-hallucinated (since they wouldn't compile).

### C.6 Identifier overcounting in attribute access

Tree-sitter parses `foo.bar.baz` as an attribute node containing identifier `foo` and field names `bar`, `baz`. We only count `foo` as a "use" (the receiver). Attribute names (`bar`, `baz`) are not used as standalone identifiers in our analysis. This is consistent with how Python resolves them at runtime.

### C.7 Symbol table noise

A repository's symbol table can contain thousands of short common names (`x`, `i`, `data`, `result`, `tmp`, ...). These match too liberally and produce false "cross-file resolved" classifications. **Filter the symbol table to exclude single-letter names and a curated stoplist of ~30 common short names.**

### C.8 The Stack download size

Even `the-stack-smol` is ~3GB. Stream it rather than downloading fully:
```python
ds = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train",
                  streaming=True)
```

### C.9 Greedy decoding vs. sampling

Throughout the paper and our implementation, **always use greedy decoding** (`temperature=0.0`). The CARD paper does. Sampling would introduce variance that confounds the comparisons.

### C.10 ES vs. ES×100

The CARD paper sometimes reports ES as a percentage (e.g., `72.26%`) and sometimes as a fraction (e.g., `0.7226`). Our code returns the fraction; multiply by 100 for paper-style display. Don't get these confused when validating against published numbers.

---

## Appendix D: Dataset schemas

This appendix documents the exact JSONL field names and types for each dataset, with a per-dataset adapter to the canonical `(x_left, x_right, ground_truth, repo_files)` tuple our code uses internally. **Read this before writing `src/adaptive_retrieval/eval/datasets.py`.**

### D.1 CrossCodeEval

**Source.** GitHub `amazon-science/cceval`. Data ships as `data/crosscodeeval_data.tar.xz` inside the repo. Decompress with:
```bash
git clone https://github.com/amazon-science/cceval
cd cceval && tar -xvJf data/crosscodeeval_data.tar.xz -C data/
```

**File layout after decompression.** `data/crosscodeeval_data/<language>/<task>.jsonl`. Languages: `python`, `java`, `csharp`, `typescript`. Task files (Python example):
- `line_completion.jsonl` — the canonical "no retrieval" version, 2,665 Python instances.
- `line_completion_rg1_bm25.jsonl` — pre-retrieved with BM25.
- `line_completion_rg1_unixcoder_cosine_sim.jsonl` — pre-retrieved with UniXcoder.
- `line_completion_oracle_bm25.jsonl` — oracle retrieval (includes ground truth in query).

**Use `line_completion.jsonl` as the primary file.** It contains the raw instance with cross-file context but no pre-retrieved chunks. Other variants pre-bake retrieval into the prompt, which conflicts with our adaptive policy.

**Per-line JSONL schema.** Each line is one instance:

```json
{
  "task_id": "project_cc_python/42",
  "prompt": "import asyncio\nimport json\n\nfrom server import Server\n\nasync def main():\n    s = Server()\n    ",
  "groundtruth": "await s.start()",
  "right_context": "\n    await asyncio.sleep(1)\n    await s.stop()\n",
  "crossfile_context": [
    {
      "filename": "server.py",
      "retrieved_chunk": "class Server:\n    def __init__(self):\n        self.running = False\n\n    async def start(self):\n        self.running = True\n",
      "score": 0.84
    },
    {
      "filename": "client.py",
      "retrieved_chunk": "...",
      "score": 0.71
    }
  ],
  "repository": "owner/repo-name",
  "metadata": {"file": "main.py", "...": "..."}
}
```

**Field-to-canonical mapping.**

| CrossCodeEval field | Our canonical field | Notes |
|---------------------|---------------------|-------|
| `prompt` | `x_left` | The in-file context up to the hole. |
| `right_context` | `x_right` | The in-file context after the hole. |
| `groundtruth` | `ground_truth` | The string to match against. |
| `crossfile_context` | `repo_files` | List of `{filename, retrieved_chunk}`. **Treat each chunk's `retrieved_chunk` as a synthetic file with name `filename`.** |
| `task_id` | `instance_id` | For logging/cross-referencing. |
| `metadata.file` | `target_file_path` | Path of the current file in the original repo. |

**Adapter signature:**

```python
def load_crosscodeeval_python(path: str = "data/crosscodeeval_data/python/line_completion.jsonl"):
    """Yields Instance(x_left, x_right, ground_truth, repo_files, instance_id, target_file)."""
    import jsonlines
    with jsonlines.open(path) as reader:
        for record in reader:
            repo_files = {
                chunk["filename"]: chunk["retrieved_chunk"]
                for chunk in record.get("crossfile_context", [])
            }
            # Also include the target file itself (reconstructed from x_left + ground_truth + x_right)
            target = record["metadata"].get("file", "current_file.py")
            repo_files[target] = record["prompt"] + record["groundtruth"] + record.get("right_context", "")
            yield Instance(
                x_left=record["prompt"],
                x_right=record.get("right_context", ""),
                ground_truth=record["groundtruth"],
                repo_files=repo_files,
                instance_id=record["task_id"],
                target_file=target,
            )
```

**Sanity check.** After loading, the first record should have:
- `x_left` ending in code, not a newline alone.
- `ground_truth` being a single line or short snippet (~10–30 chars typically).
- `repo_files` containing 1–10 entries (the cross-file chunks plus the synthesised current file).

### D.2 RepoEval

**Source.** GitHub `microsoft/CodeT/RepoCoder`. Data files at `RepoCoder/datasets/datasets.zip`, which contains:
- `function_level_completion_4k_context_codex.test.jsonl` (373 instances)
- `function_level_completion_2k_context_codex.test.jsonl`
- `line_level_completion_4k_context_codex.test.jsonl` (1,600 instances)
- `line_level_completion_2k_context_codex.test.jsonl`
- `line_level_completion_2k_context_codegen.test.jsonl`
- `line_level_completion_1k_context_codegen.test.jsonl`
- `api_level_completion_4k_context_codex.test.jsonl` (1,600 instances)
- `api_level_completion_2k_context_codex.test.jsonl`

**Choice of context size.** The `_4k_context_` files contain longer left-context (more room for retrieved snippets); `_2k_` and `_1k_` are smaller-context variants. **Use `line_level_completion_2k_context_codex.test.jsonl`, `api_level_completion_2k_context_codex.test.jsonl`, and `function_level_completion_2k_context_codex.test.jsonl` for our experiments** — the 2k size matches what fits with our 7B-parameter generators without retrieval, and CARD paper Table 3 uses this size.

**Per-line JSONL schema.** Each line:

```json
{
  "prompt": "...the unfinished code, with retrieved snippets prepended IF rg1...",
  "metadata": {
    "task_id": "huggingface_diffusers/0",
    "ground_truth": "def __init__(self):",
    "fpath_tuple": ["huggingface_diffusers", "src", "diffusers", "models", "unet.py"],
    "context_start_lineno": 0,
    "line_no": 42
  }
}
```

**Important caveats.**

- The `prompt` field is **left-context only**. There is no separate `right_context` field. To recover `x_right`, locate the file at `metadata.fpath_tuple` in the shipped `repositories/` directory and slice from `line_no + len(ground_truth.splitlines())` onward.
- For the no-retrieve baseline, slice the file's contents directly to construct `x_left` and `x_right`; don't use the `prompt` field as-is because some of the JSONL files (the `rg1` variants) have retrieved chunks prepended.
- The `repositories/` directory must be downloaded separately. Follow the RepoCoder README; it's a sibling zip file (~200 MB).

**Adapter signature:**

```python
def load_repoeval(task: str = "line", repo_root: str = "RepoCoder/repositories"):
    """task in {'line', 'api', 'function'}."""
    import jsonlines
    from pathlib import Path
    filename = f"{task}_level_completion_2k_context_codex.test.jsonl"
    path = f"RepoCoder/datasets/{filename}"
    with jsonlines.open(path) as reader:
        for record in reader:
            meta = record["metadata"]
            fpath = Path(repo_root) / Path(*meta["fpath_tuple"])
            full_content = fpath.read_text(encoding="utf-8")
            lines = full_content.splitlines(keepends=True)
            line_no = meta["line_no"]
            x_left = "".join(lines[:line_no])
            gt_line_count = meta["ground_truth"].count("\n") + 1
            x_right = "".join(lines[line_no + gt_line_count:])

            # Build repo_files from all files in the same top-level repo
            repo_name = meta["fpath_tuple"][0]
            repo_dir = Path(repo_root) / repo_name
            repo_files = {}
            for f in repo_dir.rglob("*.py"):
                try:
                    repo_files[str(f.relative_to(repo_dir))] = f.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue

            yield Instance(
                x_left=x_left,
                x_right=x_right,
                ground_truth=meta["ground_truth"],
                repo_files=repo_files,
                instance_id=meta["task_id"],
                target_file=str(Path(*meta["fpath_tuple"])),
            )
```

### D.3 the-stack-smol (for CARD Estimator training data)

**Source.** HuggingFace `bigcode/the-stack-smol`. Requires accepting the data agreement (one-click after HF login).

**Schema** (from HuggingFace dataset card):
```python
Dataset({
    features: ['content', 'avg_line_length', 'max_line_length',
               'alphanum_fraction', 'licenses', 'repository_name', 'path',
               'size', 'lang'],
    num_rows: 300_000  # across all languages
})
```

Loading Python only:
```python
from datasets import load_dataset
ds = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train")
# ~10,000 Python files
```

**IMPORTANT DEVIATION from CARD paper §3.4.** The CARD paper specifies "11k Python repos with 50–100 files each". `the-stack-smol` only contains ~10k random Python *files* (not repos), drawn from many distinct repositories. The "50–100 files per repo" filter therefore cannot be applied at training-data-construction time.

**Adapted recipe (replace §9.5):**

1. Load all ~10k Python files from `the-stack-smol`.
2. Filter files individually: ≥3 local imports, >20 non-empty lines. (Skip the repo-level "50–100 files" filter — we don't have repo structure here.)
3. From each remaining file, sample 25 (X, y) pairs (vs. the implicit ~22 per repo in the CARD recipe, which yields ~250k pairs total).
4. K-Means deduplicate, target final size ~250k.
5. Run generator and collect features/scores.

**Field-to-use mapping.**

| Stack-Smol field | Used for |
|------------------|----------|
| `content` | The Python source for sampling (X, y) pairs |
| `repository_name` | For logging (optional) |
| `lang` | Should always be `"Python"` after filtering |
| Others | Ignored |

**Fallback if Stack-Smol is insufficient.** If 10k Python files don't produce 250k diverse pairs after deduplication, the alternative is `codeparrot/github-code` (HuggingFace, no agreement needed, ~115GB). Use the same recipe. Document this fallback choice in the paper.

### D.4 Canonical `Instance` dataclass

Use this single dataclass for all three datasets:

```python
# src/adaptive_retrieval/eval/datasets.py
from dataclasses import dataclass

@dataclass
class Instance:
    x_left: str
    x_right: str
    ground_truth: str
    repo_files: dict[str, str]   # path -> content
    instance_id: str
    target_file: str
```

---

## Appendix E: Extended static analysis tests

**File:** `tests/test_static_analysis_extended.py`

These 20 tests cover edge cases that the basic tests in §10.5 don't catch. Run them after every change to the static-analysis module.

```python
"""
Extended static-analysis tests covering Python scoping edge cases.

Pattern: each test constructs a minimal "in-file context" and "prediction",
builds a tiny repo symbol table, and asserts the analyzer's decision.
"""
import pytest
from pathlib import Path
from src.adaptive_retrieval.static_analysis.scope import InFileScopeAnalyzer
from src.adaptive_retrieval.static_analysis.symbol_table import RepositorySymbolTable
from src.adaptive_retrieval.static_analysis.analyzer import PredictionAnalyzer


@pytest.fixture
def empty_analyzer(tmp_path):
    """Analyzer with no cross-file symbols (only in-file and builtins resolvable)."""
    repo = tmp_path / "repo"; repo.mkdir()
    syms = RepositorySymbolTable(repo)
    return PredictionAnalyzer(InFileScopeAnalyzer(), syms)


@pytest.fixture
def repo_with(tmp_path):
    """Factory: create a tiny repo with the given filename->content map."""
    def _make(files: dict[str, str]):
        repo = tmp_path / "repo"; repo.mkdir(exist_ok=True)
        for name, content in files.items():
            (repo / name).write_text(content)
        return PredictionAnalyzer(InFileScopeAnalyzer(),
                                   RepositorySymbolTable(repo))
    return _make


# ---------- BASIC RESOLUTION ----------

def test_01_hallucinated_function(empty_analyzer):
    """Name not defined anywhere -> fires."""
    r = empty_analyzer.analyze(
        prediction="totally_fake()",
        x_left="def main():\n    return ",
        x_right="\n",
    )
    assert r.fires
    assert "totally_fake" in r.unresolved_identifiers


def test_02_locally_defined_function(empty_analyzer):
    """Function defined in same file -> doesn't fire."""
    r = empty_analyzer.analyze(
        prediction="helper()",
        x_left="def helper():\n    return 1\n\ndef caller():\n    return ",
        x_right="\n",
    )
    assert not r.fires


def test_03_builtin_function(empty_analyzer):
    """print, len, range, etc -> don't fire."""
    r = empty_analyzer.analyze(
        prediction="print(len(x))",
        x_left="def f(x):\n    ",
        x_right="\n",
    )
    assert not r.fires, f"Unexpected fire: {r.unresolved_identifiers}"


def test_04_cross_file_resolved(repo_with):
    """Name defined in another repo file -> fires as cross_file."""
    analyzer = repo_with({
        "lib.py": "def cross_func():\n    return 42\n",
    })
    r = analyzer.analyze(
        prediction="cross_func()",
        x_left="def use_it():\n    return ",
        x_right="\n",
    )
    assert r.fires
    assert "cross_func" in r.cross_file_identifiers


# ---------- IMPORTS ----------

def test_05_imported_library_function(empty_analyzer):
    """Imported name -> doesn't fire even if not in repo."""
    r = empty_analyzer.analyze(
        prediction="np.array([1, 2])",
        x_left="import numpy as np\n\ndef f():\n    return ",
        x_right="\n",
    )
    assert not r.fires


def test_06_from_import(empty_analyzer):
    """`from X import Y` should bring Y into scope."""
    r = empty_analyzer.analyze(
        prediction="join('a', 'b')",
        x_left="from os.path import join\n\ndef f():\n    return ",
        x_right="\n",
    )
    assert not r.fires


def test_07_aliased_import(empty_analyzer):
    """`import X as Y` should bring Y (not X) into scope."""
    r = empty_analyzer.analyze(
        prediction="pd.DataFrame()",
        x_left="import pandas as pd\n\ndef f():\n    return ",
        x_right="\n",
    )
    assert not r.fires


# ---------- ATTRIBUTE ACCESS ----------

def test_08_attribute_on_imported(empty_analyzer):
    """`os.path.join` -> 'os' is the use; attributes are not flagged."""
    r = empty_analyzer.analyze(
        prediction="os.path.join('a', 'b')",
        x_left="import os\n\ndef f():\n    return ",
        x_right="\n",
    )
    assert not r.fires


def test_09_attribute_on_self(empty_analyzer):
    """`self.x` is fine; 'self' is treated as a builtin-like name."""
    r = empty_analyzer.analyze(
        prediction="self.value + 1",
        x_left="class C:\n    def m(self):\n        return ",
        x_right="\n",
    )
    assert not r.fires


# ---------- SCOPING ----------

def test_10_function_parameter(empty_analyzer):
    """Function parameter used in body -> doesn't fire."""
    r = empty_analyzer.analyze(
        prediction="x + 1",
        x_left="def f(x):\n    return ",
        x_right="\n",
    )
    assert not r.fires


def test_11_for_loop_variable(empty_analyzer):
    """Variable bound by `for` loop -> usable inside loop."""
    r = empty_analyzer.analyze(
        prediction="item * 2",
        x_left="def f(items):\n    for item in items:\n        return ",
        x_right="\n",
    )
    assert not r.fires


def test_12_list_comprehension_variable(empty_analyzer):
    """Comprehension scope: 'i' should be visible within the comprehension itself."""
    r = empty_analyzer.analyze(
        prediction="[i * 2 for i in range(10)]",
        x_left="def f():\n    return ",
        x_right="\n",
    )
    assert not r.fires


def test_13_with_statement_binding(empty_analyzer):
    """`with X as Y:` binds Y."""
    r = empty_analyzer.analyze(
        prediction="f.read()",
        x_left="def reader(path):\n    with open(path) as f:\n        return ",
        x_right="\n",
    )
    assert not r.fires


def test_14_multiple_assignment(empty_analyzer):
    """`a, b = c, d` defines a and b."""
    r = empty_analyzer.analyze(
        prediction="a + b",
        x_left="def f():\n    a, b = 1, 2\n    return ",
        x_right="\n",
    )
    assert not r.fires


def test_15_walrus_operator(empty_analyzer):
    """`(x := 5)` should bind x. Acceptable if test fails;
    walrus is a known edge case we may not support."""
    r = empty_analyzer.analyze(
        prediction="x + 1",
        x_left="def f():\n    if (x := compute()) > 0:\n        return ",
        x_right="\n",
    )
    # Soft assertion: walrus is hard; document if we can't handle it.
    # If this fails, set fire_on_unresolved=False to avoid false positives in real data.
    if r.fires:
        pytest.xfail("Walrus operator scoping not supported; documented limitation.")
    assert not r.fires


# ---------- DECORATORS AND TYPE HINTS ----------

def test_16_decorator_imported(empty_analyzer):
    """Imported decorator -> doesn't fire."""
    r = empty_analyzer.analyze(
        prediction="cached(f)",
        x_left="from functools import cache as cached\n\ndef use(f):\n    return ",
        x_right="\n",
    )
    assert not r.fires


def test_17_type_hint_imported(empty_analyzer):
    """Type hint with imported type -> doesn't fire."""
    # The prediction itself uses Optional; if our extractor walks type annotations as uses, this matters.
    r = empty_analyzer.analyze(
        prediction="Optional[int]",
        x_left="from typing import Optional\n\nx: ",
        x_right="\n",
    )
    assert not r.fires


# ---------- LAMBDA AND NESTED FUNCTIONS ----------

def test_18_lambda_parameter(empty_analyzer):
    """Lambda parameter -> usable in lambda body."""
    r = empty_analyzer.analyze(
        prediction="lambda x: x * 2",
        x_left="def f():\n    return ",
        x_right="\n",
    )
    assert not r.fires


def test_19_nested_function_closure(empty_analyzer):
    """Inner function uses outer's local variable -> doesn't fire."""
    r = empty_analyzer.analyze(
        prediction="inner()",
        x_left="def outer():\n    y = 5\n    def inner():\n        return y\n    return ",
        x_right="\n",
    )
    assert not r.fires


# ---------- INHERITANCE ----------

def test_20_super_call(empty_analyzer):
    """super().__init__() -> 'super' is builtin, doesn't fire."""
    r = empty_analyzer.analyze(
        prediction="super().__init__()",
        x_left="class C(Base):\n    def __init__(self):\n        ",
        x_right="\n",
    )
    # 'Base' might fire as unresolved; that's fine for our purposes.
    # But 'super' must not.
    assert "super" not in r.unresolved_identifiers


# ---------- ROBUSTNESS ----------

def test_21_syntactically_broken_prediction(empty_analyzer):
    """Analyzer must not crash on broken Python."""
    r = empty_analyzer.analyze(
        prediction="foo(",   # missing closing paren
        x_left="def f():\n    return ",
        x_right="\n",
    )
    # tree-sitter is fault-tolerant; should still extract 'foo' as a used name.
    # Whether it fires depends on resolution.
    assert "foo" in r.unresolved_identifiers


def test_22_empty_prediction(empty_analyzer):
    """Empty prediction must not crash."""
    r = empty_analyzer.analyze(
        prediction="",
        x_left="def f():\n    return ",
        x_right="\n",
    )
    assert not r.fires
    assert r.n_used_identifiers == 0
```

**Acceptance criteria.** All 22 tests pass except possibly `test_15_walrus_operator` (acceptable to xfail with documented limitation) and `test_17_type_hint_imported` (depends on whether our extractor visits annotation nodes — also acceptable to xfail if documented).

---

## Appendix F: End-to-end smoke test

**File:** `scripts/00_smoke_test.py`

This script runs **one synthetic instance** through all six experimental configurations to catch integration bugs before burning GPU time on the full experiments. Run it after Phase 5 of the implementation order (§16) and before Phase 6.

```python
"""
End-to-end smoke test for the entire experimental pipeline.

Constructs a single synthetic instance, runs all six configs on it,
asserts each returns a valid record, and prints a summary.

Run with:
    python scripts/00_smoke_test.py --model Qwen/Qwen2.5-Coder-7B \
        --estimator models/estimator_qwen25_coder_7b.lgb
"""
import click
import json
import sys
from src.adaptive_retrieval.generator import Generator
from src.adaptive_retrieval.retriever import BM25Retriever
from src.adaptive_retrieval.card.estimator import Estimator
from src.adaptive_retrieval.card.pipeline import card_pipeline
from src.adaptive_retrieval.static_analysis.symbol_table import RepositorySymbolTable
from src.adaptive_retrieval.static_analysis.scope import InFileScopeAnalyzer
from src.adaptive_retrieval.static_analysis.analyzer import PredictionAnalyzer
from src.adaptive_retrieval.cascade import cascade_pipeline
from src.adaptive_retrieval.baselines import no_retrieve_baseline, always_retrieve_baseline
from src.adaptive_retrieval.metrics import (
    exact_match, edit_similarity, identifier_f1,
    repository_symbol_precision, hallucination_flag,
)


# ---------- Synthetic test instance ----------

SYNTHETIC_REPO = {
    "server.py": (
        "class Server:\n"
        "    def __init__(self, port=8080):\n"
        "        self.port = port\n"
        "        self.running = False\n"
        "\n"
        "    async def start(self):\n"
        "        self.running = True\n"
        "        print(f'Server started on port {self.port}')\n"
        "\n"
        "    async def stop(self):\n"
        "        self.running = False\n"
        "        print('Server stopped')\n"
    ),
    "client.py": (
        "import asyncio\n"
        "\n"
        "class Client:\n"
        "    def __init__(self, host='localhost', port=8080):\n"
        "        self.host = host\n"
        "        self.port = port\n"
        "\n"
        "    async def connect(self):\n"
        "        print(f'Connecting to {self.host}:{self.port}')\n"
    ),
}

INSTANCE = {
    "x_left": (
        "import asyncio\n"
        "import json\n"
        "\n"
        "from server import Server\n"
        "\n"
        "async def main():\n"
        "    s = Server()\n"
        "    "
    ),
    "x_right": (
        "\n"
        "    await asyncio.sleep(1)\n"
        "    await s.stop()\n"
    ),
    "ground_truth": "await s.start()",
}


# ---------- Validators ----------

def assert_record_shape(record: dict, config_name: str):
    """Every record must have the fields below, regardless of config."""
    required = {"prediction", "retrieved", "metrics", "latency_ms"}
    missing = required - set(record.keys())
    assert not missing, f"[{config_name}] missing fields: {missing}"
    assert isinstance(record["prediction"], str), f"[{config_name}] prediction not str"
    assert isinstance(record["retrieved"], bool), f"[{config_name}] retrieved not bool"
    assert isinstance(record["latency_ms"], (int, float)), f"[{config_name}] latency not numeric"

    metrics = record["metrics"]
    for m in ["exact_match", "edit_similarity", "identifier_f1",
              "repo_symbol_precision", "hallucinated"]:
        assert m in metrics, f"[{config_name}] metric '{m}' missing"

    print(f"  ✓ [{config_name}] record shape OK")


# ---------- Per-config runners ----------

def compute_metrics(prediction: str, analyzer, x_left, x_right, ground_truth):
    return {
        "exact_match": exact_match(ground_truth, prediction),
        "edit_similarity": edit_similarity(ground_truth, prediction),
        "identifier_f1": identifier_f1(ground_truth, prediction),
        "repo_symbol_precision": repository_symbol_precision(
            prediction, x_left, x_right, analyzer),
        "hallucinated": hallucination_flag(prediction, x_left, x_right, analyzer),
    }


def run_c1_no_retrieve(generator, analyzer, instance):
    out = no_retrieve_baseline(generator, instance["x_left"], instance["x_right"])
    return {
        "prediction": out.prediction,
        "retrieved": False,
        "trigger_reason": "none",
        "latency_ms": out.latency_ms,
        "metrics": compute_metrics(out.prediction, analyzer, instance["x_left"],
                                     instance["x_right"], instance["ground_truth"]),
    }


def run_c2_always_retrieve(generator, retriever, analyzer, instance):
    out = always_retrieve_baseline(generator, retriever, instance["x_left"], instance["x_right"])
    return {
        "prediction": out.prediction,
        "retrieved": True,
        "trigger_reason": "always",
        "latency_ms": out.latency_ms,
        "metrics": compute_metrics(out.prediction, analyzer, instance["x_left"],
                                     instance["x_right"], instance["ground_truth"]),
    }


def run_c3_card(generator, retriever, estimator, analyzer, instance):
    out = card_pipeline(generator, retriever, estimator,
                         instance["x_left"], instance["x_right"],
                         T_RAG_schedule=[0.9], T_ACC_schedule=[0.8], max_iter=1)
    return {
        "prediction": out.prediction,
        "retrieved": len(out.retrieved_at_iter) > 0,
        "trigger_reason": "card" if out.retrieved_at_iter else "none",
        "s_hat_0": out.s_hats[0],
        "latency_ms": out.latency_ms,
        "metrics": compute_metrics(out.prediction, analyzer, instance["x_left"],
                                     instance["x_right"], instance["ground_truth"]),
    }


def run_c4_cascade(generator, retriever, estimator, analyzer, instance):
    out = cascade_pipeline(generator, retriever, estimator, analyzer,
                            instance["x_left"], instance["x_right"], T_RAG=0.9)
    return {
        "prediction": out.prediction,
        "retrieved": out.retrieved,
        "trigger_reason": out.trigger_reason,
        "s_hat_0": out.s_hat_0,
        "static_unresolved": out.static_unresolved,
        "static_crossfile": out.static_crossfile,
        "latency_ms": out.latency_ms,
        "metrics": compute_metrics(out.prediction, analyzer, instance["x_left"],
                                     instance["x_right"], instance["ground_truth"]),
    }


def run_c5_static_only(generator, retriever, analyzer, instance):
    # Generate ŷ_0, then check static. If fires, retrieve. Else return ŷ_0.
    out_g0 = no_retrieve_baseline(generator, instance["x_left"], instance["x_right"])
    sa = analyzer.analyze(out_g0.prediction, instance["x_left"], instance["x_right"])
    if sa.fires:
        out_rag = always_retrieve_baseline(generator, retriever,
                                             instance["x_left"], instance["x_right"])
        return {
            "prediction": out_rag.prediction,
            "retrieved": True,
            "trigger_reason": "static",
            "latency_ms": out_g0.latency_ms + out_rag.latency_ms,
            "metrics": compute_metrics(out_rag.prediction, analyzer, instance["x_left"],
                                         instance["x_right"], instance["ground_truth"]),
        }
    return {
        "prediction": out_g0.prediction,
        "retrieved": False,
        "trigger_reason": "none",
        "latency_ms": out_g0.latency_ms,
        "metrics": compute_metrics(out_g0.prediction, analyzer, instance["x_left"],
                                     instance["x_right"], instance["ground_truth"]),
    }


def run_c6_oracle(generator, retriever, analyzer, instance):
    # Run both, pick higher ES against ground truth.
    out_no = no_retrieve_baseline(generator, instance["x_left"], instance["x_right"])
    out_yes = always_retrieve_baseline(generator, retriever,
                                         instance["x_left"], instance["x_right"])
    es_no = edit_similarity(instance["ground_truth"], out_no.prediction)
    es_yes = edit_similarity(instance["ground_truth"], out_yes.prediction)
    chosen = out_yes if es_yes > es_no else out_no
    return {
        "prediction": chosen.prediction,
        "retrieved": (chosen is out_yes),
        "trigger_reason": "oracle",
        "latency_ms": out_no.latency_ms + out_yes.latency_ms,
        "metrics": compute_metrics(chosen.prediction, analyzer, instance["x_left"],
                                     instance["x_right"], instance["ground_truth"]),
    }


# ---------- Main ----------

@click.command()
@click.option("--model", default="Qwen/Qwen2.5-Coder-7B")
@click.option("--estimator", default=None, help="Path to trained LightGBM file")
@click.option("--skip-card", is_flag=True, help="Skip CARD/Cascade if no estimator")
def main(model, estimator, skip_card):
    print("=" * 60)
    print("SMOKE TEST: running synthetic instance through all 6 configs")
    print("=" * 60)

    # Build components from the synthetic instance
    print("\n[1/3] Building components...")
    generator = Generator(model)
    retriever = BM25Retriever(SYNTHETIC_REPO)

    # Write the synthetic repo to a temporary directory for the symbol table
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        for name, content in SYNTHETIC_REPO.items():
            with open(os.path.join(tmp, name), "w") as f:
                f.write(content)
        repo_syms = RepositorySymbolTable(tmp)
    scope = InFileScopeAnalyzer()
    analyzer = PredictionAnalyzer(scope, repo_syms)

    if estimator and not skip_card:
        est = Estimator.load(estimator)
    else:
        est = None
        if not skip_card:
            print("WARN: No estimator provided. Skipping CARD and Cascade.")
            skip_card = True

    print("\n[2/3] Running 6 configs...")
    results = {}
    results["C1"] = run_c1_no_retrieve(generator, analyzer, INSTANCE)
    assert_record_shape(results["C1"], "C1")

    results["C2"] = run_c2_always_retrieve(generator, retriever, analyzer, INSTANCE)
    assert_record_shape(results["C2"], "C2")

    if not skip_card:
        results["C3"] = run_c3_card(generator, retriever, est, analyzer, INSTANCE)
        assert_record_shape(results["C3"], "C3")

        results["C4"] = run_c4_cascade(generator, retriever, est, analyzer, INSTANCE)
        assert_record_shape(results["C4"], "C4")

    results["C5"] = run_c5_static_only(generator, retriever, analyzer, INSTANCE)
    assert_record_shape(results["C5"], "C5")

    results["C6"] = run_c6_oracle(generator, retriever, analyzer, INSTANCE)
    assert_record_shape(results["C6"], "C6")

    # ---------- Sanity checks ----------
    print("\n[3/3] Cross-config sanity checks...")

    # All non-skipped configs should produce a non-empty prediction
    for cfg, rec in results.items():
        assert rec["prediction"], f"[{cfg}] empty prediction"

    # C1 should never be flagged as retrieved
    assert results["C1"]["retrieved"] is False

    # C2 must be retrieved
    assert results["C2"]["retrieved"] is True

    # C5: trigger_reason in {"static", "none"}
    assert results["C5"]["trigger_reason"] in {"static", "none"}

    # If CARD ran: C4's trigger_reason must be one of the four valid strings
    if "C4" in results:
        assert results["C4"]["trigger_reason"] in {
            "card", "static_unresolved", "static_crossfile", "none"
        }

    # ---------- Summary ----------
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'config':<6} {'retrieved':<10} {'trigger':<20} {'ES':<6} {'hall':<5} prediction")
    print("-" * 80)
    for cfg, rec in results.items():
        ret = "yes" if rec["retrieved"] else "no"
        trig = rec.get("trigger_reason", "-")
        es = f"{rec['metrics']['edit_similarity']:.2f}"
        hall = "yes" if rec["metrics"]["hallucinated"] else "no"
        pred_preview = rec["prediction"][:30].replace("\n", " ")
        print(f"{cfg:<6} {ret:<10} {trig:<20} {es:<6} {hall:<5} {pred_preview!r}")

    print("\nGround truth:", repr(INSTANCE["ground_truth"]))
    print("\n✓ Smoke test passed — pipeline is integration-correct.")
    print("Now safe to run scripts/04_run_experiment.py on the real datasets.")


if __name__ == "__main__":
    main()
```

**Expected output (rough — depends on the model):**

```
============================================================
SMOKE TEST: running synthetic instance through all 6 configs
============================================================
[1/3] Building components...
[2/3] Running 6 configs...
  ✓ [C1] record shape OK
  ✓ [C2] record shape OK
  ✓ [C3] record shape OK
  ✓ [C4] record shape OK
  ✓ [C5] record shape OK
  ✓ [C6] record shape OK
[3/3] Cross-config sanity checks...

============================================================
RESULTS SUMMARY
============================================================
config retrieved  trigger              ES     hall  prediction
--------------------------------------------------------------------------------
C1     no         none                 0.87   no    'await s.start()'
C2     yes        always               1.00   no    'await s.start()'
C3     yes        card                 1.00   no    'await s.start()'
C4     no         none                 0.87   no    'await s.start()'
C5     no         none                 0.87   no    'await s.start()'
C6     yes        oracle               1.00   no    'await s.start()'

Ground truth: 'await s.start()'

✓ Smoke test passed — pipeline is integration-correct.
```

The exact numbers will vary; what matters is:
1. Every config produces a record without errors.
2. The trigger reasons are sensible.
3. No assertion fails.

If the smoke test passes, the pipeline is end-to-end functional. If a config returns nonsensical output (e.g., empty string, garbage), check the prompt formatting for the model family in `prompt.py`.

---

## Final note

This document is intended to be the complete specification. If the implementing agent finds a section that's ambiguous or incomplete, the most likely fix is to:
1. Check the CARD paper PDF (in `papers/2406.10263v1.pdf`) for §§2–3 of the paper.
2. Check the Repoformer paper (Wu et al. 2024, arXiv:2403.10059) for shared infrastructure conventions (K-Means dedup, RepoEval format).
3. Read Appendix D for dataset schemas, Appendix E for static-analysis edge cases, and Appendix F for the end-to-end smoke test that must pass before running the main experiments.
4. Choose the conservative implementation and document the choice.

**Last update:** Project kickoff, Week 1 day 1, with extensions for dataset schemas, extended static-analysis tests, and end-to-end smoke test.
