"""LightGBM Estimator predicting Edit Similarity from CARD features.

CARD paper §3.4: regress ES against the 13-D feature vector. Per Table 8 of
the paper their MSE on the validation split is ~0.07; we target <0.10.
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np


class Estimator:
    """Thin wrapper around a single LightGBM regressor."""

    def __init__(self, model: lgb.Booster | None = None):
        self.model = model

    @classmethod
    def train(
        cls,
        features: np.ndarray,
        scores: np.ndarray,
        val_fraction: float = 0.05,
        random_state: int = 42,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 20,
        learning_rate: float = 0.1,
    ) -> "Estimator":
        """Train an Estimator on (features, scores) pairs.

        Args:
            features: shape (N, 13)
            scores: shape (N,), edit similarities in [0, 1]
        """
        features = np.asarray(features, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != 13:
            raise ValueError(f"features must be (N, 13), got {features.shape}")
        if features.shape[0] != scores.shape[0]:
            raise ValueError(
                f"features and scores N mismatch: {features.shape[0]} vs {scores.shape[0]}"
            )

        rng = np.random.default_rng(random_state)
        n = features.shape[0]
        idx = rng.permutation(n)
        n_val = max(1, int(n * val_fraction))
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        train_data = lgb.Dataset(features[train_idx], scores[train_idx])
        val_data = lgb.Dataset(
            features[val_idx], scores[val_idx], reference=train_data
        )
        params = {
            "objective": "regression",
            "metric": "mse",
            "learning_rate": learning_rate,
            "verbose": -1,
            "seed": random_state,
        }
        model = lgb.train(
            params=params,
            train_set=train_data,
            valid_sets=[val_data],
            num_boost_round=num_boost_round,
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        return cls(model=model)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Returns shape (N,). Accepts (13,) or (N, 13)."""
        if self.model is None:
            raise RuntimeError("Estimator is empty; train or load first")
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        preds = self.model.predict(features)
        return np.asarray(preds, dtype=np.float32)

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("nothing to save")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "Estimator":
        return cls(model=lgb.Booster(model_file=str(path)))


class MockEstimator:
    """Deterministic Estimator for unit tests. Returns scripted ŝ values."""

    def __init__(self, s_hat_values: list[float] | float):
        if isinstance(s_hat_values, (int, float)):
            s_hat_values = [float(s_hat_values)]
        self.s_hat_values = list(s_hat_values)
        self.call_count = 0

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.call_count >= len(self.s_hat_values):
            val = self.s_hat_values[-1]  # repeat last
        else:
            val = self.s_hat_values[self.call_count]
        self.call_count += 1
        return np.array([val], dtype=np.float32)
