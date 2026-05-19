"""Cascade pipeline smoke test on real CrossCodeEval-Python instances.

Phase 4 §16 validation gate:
    "Run cascade on a curated 20-instance subset of CrossCodeEval-Python.
    Manually inspect the trigger_reason field: it should be a mix of
    'card', 'static_unresolved', 'static_crossfile', and 'none'. If 100%
    are one reason, something's wrong."

This script uses MockGenerator + a synthetic Estimator, so the trigger
distribution depends on the mock's confidence patterns and the predictions
we choose. We assert the cascade exercises at least 2 distinct trigger
reasons; with a real model on the GPU node, all 4 should appear naturally.

Run:
    python scripts/05_smoke_cascade.py --n 20
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from itertools import islice
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import click  # noqa: E402
import numpy as np  # noqa: E402

from adaptive_retrieval.card.estimator import Estimator  # noqa: E402
from adaptive_retrieval.card.features import extract_features  # noqa: E402
from adaptive_retrieval.cascade import cascade_pipeline  # noqa: E402
from adaptive_retrieval.eval.datasets import load_crosscodeeval_python  # noqa: E402
from adaptive_retrieval.generator import Generation, MockGenerator  # noqa: E402
from adaptive_retrieval.retriever import BM25Retriever  # noqa: E402
from adaptive_retrieval.static_analysis.analyzer import PredictionAnalyzer  # noqa: E402
from adaptive_retrieval.static_analysis.scope import InFileScopeAnalyzer  # noqa: E402
from adaptive_retrieval.static_analysis.symbol_table import RepositorySymbolTable  # noqa: E402


# --------- synthetic Estimator helpers (lifted from scripts/04_smoke_card.py) ---------

def _synthetic_training_data(n: int = 1500, seed: int = 0):
    rng = np.random.default_rng(seed)
    features = np.zeros((n, 13), dtype=np.float32)
    scores = np.zeros(n, dtype=np.float32)
    for i in range(n):
        seq_len = int(rng.integers(5, 50))
        if rng.random() < 0.6:
            base_prob = float(rng.uniform(0.65, 0.95))
            base_ent = float(rng.uniform(0.1, 0.6))
            score = base_prob + float(rng.normal(0, 0.05))
        else:
            base_prob = float(rng.uniform(0.05, 0.45))
            base_ent = float(rng.uniform(0.8, 2.5))
            score = base_prob * 0.7 + float(rng.normal(0, 0.05))
        probs = np.clip(rng.normal(base_prob, 0.05, seq_len), 1e-3, 1.0)
        ents = np.clip(rng.normal(base_ent, 0.1, seq_len), 0.0, 5.0)
        features[i] = extract_features(probs, ents)
        scores[i] = float(np.clip(score, 0.0, 1.0))
    return features, scores


# --------- mock that varies prediction TEXT and confidence by call index ---------

class _DistributionMock(MockGenerator):
    """Cycles through prediction patterns so the cascade can exercise each
    trigger branch. Pattern rotation per zero-shot call:
        0: "pass"                      (no identifiers, clean) → "none" if conf high
        1: low confidence, any pred    → "card" (CARD fires first)
        2: "totally_made_up_name()"    → "static_unresolved" if conf high
        3: high confidence + cross-file (uses a name from repo)
    """

    def __init__(self):
        super().__init__()
        self._zs_call_idx = 0

    def generate(self, prompt: str) -> Generation:
        self.call_log.append(prompt)
        is_rag = "# Here are some relevant code fragments" in prompt
        if is_rag:
            # RAG path: emit a reasonable-looking RAG response, with arbitrary stats.
            pred = "real_rag_response()"
            probs = np.full(6, 0.85, dtype=np.float32)
            ents = np.full(6, 0.2, dtype=np.float32)
        else:
            pattern = self._zs_call_idx % 4
            self._zs_call_idx += 1
            if pattern == 0:
                pred = "pass"
                probs = np.full(2, 0.90, dtype=np.float32)
                ents = np.full(2, 0.15, dtype=np.float32)
            elif pattern == 1:
                pred = "uncertain_output()"
                probs = np.full(6, 0.35, dtype=np.float32)
                ents = np.full(6, 1.20, dtype=np.float32)
            elif pattern == 2:
                pred = "totally_made_up_name()"
                probs = np.full(8, 0.88, dtype=np.float32)
                ents = np.full(8, 0.18, dtype=np.float32)
            else:
                pred = "extract_features([1,2])"  # likely matches a name in some repo_files
                probs = np.full(6, 0.87, dtype=np.float32)
                ents = np.full(6, 0.20, dtype=np.float32)
        n = max(1, len(probs))
        return Generation(
            prediction=pred,
            token_ids=list(range(n)),
            token_probs=probs,
            token_entropies=ents,
            latency_ms=1.0,
        )


@click.command()
@click.option("--n", default=20, type=int, help="CCE instances to run")
@click.option(
    "--t-rag",
    default=0.7,
    type=float,
    help="T_RAG threshold for CARD's is_retrieve. Lower than the paper's 0.9 "
    "because our synthetic Estimator caps below 0.9.",
)
def main(n: int, t_rag: float) -> None:
    print("=" * 60)
    print("Cascade smoke test (no GPU)")
    print("=" * 60)

    print("\n[1/3] Train synthetic Estimator")
    feats, scores = _synthetic_training_data(n=1500)
    est = Estimator.train(feats, scores, val_fraction=0.1)
    in_sample_mse = float(np.mean((est.predict(feats) - scores) ** 2))
    print(f"  in-sample MSE: {in_sample_mse:.4f}")

    print(f"\n[2/3] Load {n} CCE-Python instances and run cascade")
    instances = list(islice(load_crosscodeeval_python(), n))
    print(f"  loaded {len(instances)} instances; t_rag={t_rag}")

    gen = _DistributionMock()
    t0 = time.time()
    outputs = []
    for inst in instances:
        retriever = BM25Retriever(inst.repo_files)
        analyzer = PredictionAnalyzer(
            InFileScopeAnalyzer(),
            RepositorySymbolTable.from_files(inst.repo_files),
        )
        out = cascade_pipeline(
            gen, retriever, est, analyzer,
            x_left=inst.x_left, x_right=inst.x_right,
            t_rag=t_rag,
        )
        outputs.append((inst.instance_id, out))
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.2f}s")

    print("\n[3/3] Trigger-reason distribution")
    counts = Counter(o.trigger_reason for _, o in outputs)
    for reason in ("none", "card", "static_unresolved", "static_crossfile"):
        n_count = counts.get(reason, 0)
        bar = "#" * n_count
        print(f"  {reason:<22} {n_count:>3}/{len(outputs):<3} {bar}")
    distinct = len([k for k, v in counts.items() if v > 0])
    print(f"\n  distinct trigger reasons seen: {distinct}/4")

    # Examples per reason
    print("\n  one example per reason:")
    seen = set()
    for inst_id, out in outputs:
        if out.trigger_reason in seen:
            continue
        seen.add(out.trigger_reason)
        pred_preview = out.prediction[:40].replace("\n", " ")
        extras = ""
        if out.static_unresolved:
            extras += f" unresolved={out.static_unresolved[:3]}"
        if out.static_crossfile:
            extras += f" crossfile={out.static_crossfile[:3]}"
        print(
            f"    [{out.trigger_reason:<20}] id={inst_id[-20:]:<20} "
            f"ŝ₀={out.s_hat_0:.3f} pred={pred_preview!r}{extras}"
        )

    # Gate: at least 2 distinct trigger reasons fired.
    assert distinct >= 2, (
        f"Cascade only exercised {distinct} trigger reason(s); something is wrong "
        f"(see §16 Phase 4 gate)"
    )

    # Asymmetric-cascade invariant: retrieval count must be >= CARD's would-be count.
    retrieved = sum(1 for _, o in outputs if o.retrieved)
    card_only = sum(1 for _, o in outputs if o.s_hat_0 < t_rag)
    assert retrieved >= card_only, (
        f"Cascade retrieved {retrieved} but CARD alone would have done {card_only} — "
        "static stage should only ADD retrievals, never remove them"
    )
    print(f"\n  asymmetric-cascade invariant: cascade_retrieved={retrieved} "
          f">= card_alone={card_only} ✓")

    print("\nOK — cascade exercises multiple branches and preserves the "
          "asymmetric-cascade invariant.")


if __name__ == "__main__":
    main()
