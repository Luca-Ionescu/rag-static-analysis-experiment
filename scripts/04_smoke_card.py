"""End-to-end CARD smoke test, no GPU required.

1. Generate synthetic (features, ES-score) pairs with a learnable signal.
2. Train an Estimator on them; assert MSE < 0.10 (Phase 3 paper-target gate).
3. Run the full CARD single-RAG pipeline on N real CrossCodeEval-Python
   instances using MockGenerator, and verify the trigger distribution.

This catches integration bugs in the CARD pipeline without burning GPU time.
The actual CARD reproduction (±1% ES vs CARD paper on RepoEval-line) still
needs a real model on the GPU node.
"""
from __future__ import annotations

import statistics
import sys
import time
from itertools import islice
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import click  # noqa: E402
import numpy as np  # noqa: E402

from adaptive_retrieval.card.estimator import Estimator  # noqa: E402
from adaptive_retrieval.card.features import extract_features  # noqa: E402
from adaptive_retrieval.card.pipeline import card_pipeline  # noqa: E402
from adaptive_retrieval.eval.datasets import load_crosscodeeval_python  # noqa: E402
from adaptive_retrieval.generator import Generation, MockGenerator  # noqa: E402
from adaptive_retrieval.retriever import BM25Retriever  # noqa: E402


def _synthetic_training_data(n: int = 1500, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic (features, scores) where prob_avg dominates ES, plus mild noise."""
    rng = np.random.default_rng(seed)
    features = np.zeros((n, 13), dtype=np.float32)
    scores = np.zeros(n, dtype=np.float32)
    for i in range(n):
        seq_len = int(rng.integers(5, 50))
        # Vary confidence: a fraction of "confident" generations, rest noisy.
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


class ConfidenceVaryingMock(MockGenerator):
    """MockGenerator that returns different token_probs by call index, simulating
    instances of varying CARD confidence so the pipeline branches both ways.
    """

    def __init__(self):
        super().__init__(default_prediction="MOCK_OUT")
        self._n_calls = 0

    def generate(self, prompt: str) -> Generation:
        self.call_log.append(prompt)
        is_rag_call = "# Here are some relevant code fragments" in prompt
        if self._n_calls % 4 == 0:
            probs = np.full(8, 0.90, dtype=np.float32)
            ents = np.full(8, 0.15, dtype=np.float32)
        elif self._n_calls % 4 == 1:
            probs = np.full(8, 0.40, dtype=np.float32)
            ents = np.full(8, 1.20, dtype=np.float32)
        elif self._n_calls % 4 == 2:
            probs = np.full(8, 0.70, dtype=np.float32) if is_rag_call else np.full(8, 0.35, dtype=np.float32)
            ents = np.full(8, 0.40, dtype=np.float32) if is_rag_call else np.full(8, 1.30, dtype=np.float32)
        else:
            probs = np.full(8, 0.85, dtype=np.float32)
            ents = np.full(8, 0.20, dtype=np.float32)
        self._n_calls += 1
        pred = "rag_pred()" if is_rag_call else "zs_pred()"
        return Generation(
            prediction=pred,
            token_ids=list(range(len(probs))),
            token_probs=probs,
            token_entropies=ents,
            latency_ms=1.0,
        )


@click.command()
@click.option("--n", default=20, type=int, help="CrossCodeEval instances to run")
@click.option("--mse-gate", default=0.10, type=float, help="Estimator MSE upper bound")
@click.option(
    "--t-rag",
    default=0.7,
    type=float,
    help="T_RAG threshold. Smoke default is below the paper's 0.9 because the "
    "synthetic Estimator caps below 0.9; both branches need to fire here.",
)
def main(n: int, mse_gate: float, t_rag: float) -> None:
    print("=" * 60)
    print("CARD smoke test (no GPU)")
    print("=" * 60)

    # ---------- [1/3] train Estimator on synthetic data ----------
    print("\n[1/3] Train Estimator on synthetic data ...")
    t0 = time.time()
    feats, scores = _synthetic_training_data(n=1500)
    est = Estimator.train(feats, scores, val_fraction=0.1)
    train_time = time.time() - t0

    pred = est.predict(feats)
    mse = float(np.mean((pred - scores) ** 2))
    pearson = float(np.corrcoef(pred, scores)[0, 1])
    print(f"  trained in {train_time:.2f}s")
    print(f"  in-sample MSE: {mse:.4f}  (gate: <{mse_gate})")
    print(f"  in-sample Pearson: {pearson:.3f}")
    assert mse < mse_gate, f"Estimator MSE {mse:.4f} above gate {mse_gate}"

    # ---------- [2/3] run CARD pipeline on real CCE instances ----------
    print(f"\n[2/3] Run CARD single-RAG on {n} CrossCodeEval-Python instances ...")
    instances = list(islice(load_crosscodeeval_python(), n))
    print(f"  loaded {len(instances)} instances")

    gen = ConfidenceVaryingMock()
    t0 = time.time()
    outputs = []
    for inst in instances:
        retriever = BM25Retriever(inst.repo_files)
        out = card_pipeline(
            gen, retriever, est,
            x_left=inst.x_left, x_right=inst.x_right,
            t_rag_schedule=[t_rag],
        )
        outputs.append((inst.instance_id, out))
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.2f}s")

    # ---------- [3/3] trigger-distribution checks ----------
    print("\n[3/3] Trigger distribution")
    retrieved_count = sum(1 for _, o in outputs if o.retrieved_at_iter)
    kept_zs_after_retrieve = sum(
        1 for _, o in outputs if o.retrieved_at_iter and o.n_iterations == 0
    )
    accepted_rag = sum(
        1 for _, o in outputs if o.retrieved_at_iter and o.n_iterations == 1
    )
    no_retrieve = sum(1 for _, o in outputs if not o.retrieved_at_iter)
    s_hats_zero = [o.s_hats[0] for _, o in outputs]
    print(f"  retrieved (CARD fired): {retrieved_count}/{len(outputs)}")
    print(f"    accepted RAG ŷ¹:      {accepted_rag}")
    print(f"    kept ZS ŷ⁰:           {kept_zs_after_retrieve}")
    print(f"  no retrieval:           {no_retrieve}")
    print(f"  ŝ₀ stats: min={min(s_hats_zero):.3f}  "
          f"mean={statistics.mean(s_hats_zero):.3f}  max={max(s_hats_zero):.3f}")

    # Sanity: pipeline should exercise both branches at least once.
    assert retrieved_count > 0, "Pipeline never triggered retrieval — check ŝ₀ values"
    assert no_retrieve > 0, "Pipeline always retrieved — check ŝ₀ values"

    print("\nOK — CARD pipeline wires together and behaves as expected.")
    print("    (Real ES-vs-paper reproduction still needs the GPU node.)")


if __name__ == "__main__":
    main()
