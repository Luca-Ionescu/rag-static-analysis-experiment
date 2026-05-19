"""Estimator (LightGBM) tests."""
from __future__ import annotations

import numpy as np
import pytest

from adaptive_retrieval.card.estimator import Estimator, MockEstimator


def _make_synthetic_data(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    features = rng.uniform(0, 1, size=(n, 13)).astype(np.float32)
    true_w = rng.normal(0, 1, size=13)
    raw = features @ true_w
    # Map to [0, 1] via sigmoid + small noise
    scores = (1.0 / (1.0 + np.exp(-raw)) + rng.normal(0, 0.02, n)).clip(0, 1)
    return features, scores.astype(np.float32)


def test_train_returns_estimator_with_booster():
    features, scores = _make_synthetic_data(n=300)
    est = Estimator.train(features, scores, val_fraction=0.2)
    assert est.model is not None


def test_estimator_learns_linear_pattern():
    """Synthetic check matching the CARD Phase 3 MSE gate: <0.10."""
    features, scores = _make_synthetic_data(n=1500)
    est = Estimator.train(features, scores, val_fraction=0.1)
    pred = est.predict(features)
    mse = float(np.mean((pred - scores) ** 2))
    assert mse < 0.05, f"MSE too high: {mse}"


def test_predict_handles_1d_input():
    features, scores = _make_synthetic_data(n=200)
    est = Estimator.train(features, scores, val_fraction=0.1)
    single = features[0]  # shape (13,)
    out = est.predict(single)
    assert out.shape == (1,)


def test_predict_handles_2d_batch():
    features, scores = _make_synthetic_data(n=200)
    est = Estimator.train(features, scores, val_fraction=0.1)
    batch = features[:5]  # (5, 13)
    out = est.predict(batch)
    assert out.shape == (5,)


def test_invalid_feature_shape_rejected():
    rng = np.random.default_rng(0)
    bad = rng.uniform(0, 1, size=(10, 7))  # only 7 features
    scores = rng.uniform(0, 1, size=10)
    with pytest.raises(ValueError):
        Estimator.train(bad, scores)


def test_features_scores_length_mismatch_rejected():
    rng = np.random.default_rng(0)
    features = rng.uniform(0, 1, size=(10, 13))
    scores = rng.uniform(0, 1, size=9)
    with pytest.raises(ValueError):
        Estimator.train(features, scores)


def test_save_and_load_roundtrip(tmp_path):
    features, scores = _make_synthetic_data(n=200)
    est = Estimator.train(features, scores, val_fraction=0.1)
    pred_before = est.predict(features[:3])

    path = tmp_path / "model.lgb"
    est.save(path)
    assert path.exists()

    est2 = Estimator.load(path)
    pred_after = est2.predict(features[:3])
    assert np.allclose(pred_before, pred_after)


def test_predict_on_empty_estimator_raises():
    est = Estimator()
    with pytest.raises(RuntimeError):
        est.predict(np.zeros((1, 13), dtype=np.float32))


def test_save_on_empty_estimator_raises(tmp_path):
    est = Estimator()
    with pytest.raises(RuntimeError):
        est.save(tmp_path / "model.lgb")


def test_mock_estimator_returns_scripted_values():
    m = MockEstimator([0.9, 0.5, 0.3])
    feats = np.zeros((13,), dtype=np.float32)
    assert m.predict(feats)[0] == pytest.approx(0.9)
    assert m.predict(feats)[0] == pytest.approx(0.5)
    assert m.predict(feats)[0] == pytest.approx(0.3)


def test_mock_estimator_repeats_last_value():
    m = MockEstimator([0.8])
    feats = np.zeros((13,), dtype=np.float32)
    assert m.predict(feats)[0] == pytest.approx(0.8)
    # Subsequent calls reuse the final scripted value
    assert m.predict(feats)[0] == pytest.approx(0.8)
    assert m.predict(feats)[0] == pytest.approx(0.8)
