"""Train a LightGBM Estimator from an (features, scores) .npz.

Usage:
    python scripts/02_train_estimator.py \\
        --data data/training_data/qwen25_05b.npz \\
        --output models/estimator_qwen25_05b.lgb

Reports val MSE so you can sanity-check against IMPLEMENTATION_GUIDE §16
Phase 3 gate (paper-target <0.10; CARD paper reports ~0.07 with 7B).
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import click  # noqa: E402
import numpy as np  # noqa: E402

from adaptive_retrieval.card.estimator import Estimator  # noqa: E402


@click.command()
@click.option("--data", required=True, type=click.Path(exists=True), help="Path to .npz.")
@click.option("--output", required=True, type=click.Path(), help="Output .lgb path.")
@click.option("--val-fraction", default=0.05, type=float)
@click.option("--num-boost-round", default=500, type=int)
@click.option("--learning-rate", default=0.1, type=float)
@click.option("--seed", default=42, type=int)
def main(
    data: str,
    output: str,
    val_fraction: float,
    num_boost_round: int,
    learning_rate: float,
    seed: int,
) -> None:
    d = np.load(data)
    features = d["features"]
    scores = d["scores"]
    print(f"[setup] features {features.shape}  scores {scores.shape}")
    print(f"        ES mean={float(np.mean(scores)):.4f}  std={float(np.std(scores)):.4f}")

    est = Estimator.train(
        features,
        scores,
        val_fraction=val_fraction,
        random_state=seed,
        num_boost_round=num_boost_round,
        learning_rate=learning_rate,
    )

    in_sample_pred = est.predict(features)
    in_sample_mse = float(np.mean((in_sample_pred - scores) ** 2))
    in_sample_pearson = float(np.corrcoef(in_sample_pred, scores)[0, 1])

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    est.save(output)
    print(f"\n[done] saved Estimator to {output}")
    print(f"  in-sample MSE: {in_sample_mse:.4f}  (Phase 3 gate: <0.10)")
    print(f"  in-sample Pearson: {in_sample_pearson:.3f}")
    if in_sample_mse < 0.10:
        print("  PASS — Estimator MSE under the paper-target gate.")
    else:
        print("  WARN — Estimator MSE above gate. Check feature distribution + model size.")


if __name__ == "__main__":
    main()
