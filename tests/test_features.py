"""CARD feature-extraction tests (IMPLEMENTATION_GUIDE §16 Phase 3 gate)."""
from __future__ import annotations

import numpy as np
import pytest

from adaptive_retrieval.card.features import FEATURE_NAMES, extract_features


def test_feature_names_count_is_13():
    assert len(FEATURE_NAMES) == 13


def test_shape_is_13():
    probs = np.array([0.5, 0.7, 0.3])
    ents = np.array([1.0, 2.0, 0.5])
    feats = extract_features(probs, ents)
    assert feats.shape == (13,)
    assert feats.dtype == np.float32


def test_empty_input_yields_zeros():
    feats = extract_features(np.array([]), np.array([]))
    assert feats.shape == (13,)
    assert (feats == 0).all()


def test_known_values_for_probs():
    probs = np.array([0.5, 0.7, 0.3], dtype=np.float64)
    ents = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    feats = extract_features(probs, ents)
    # Indices follow FEATURE_NAMES order: prob_{max,min,avg,std,prod,geomavg}
    assert feats[0] == pytest.approx(0.7, abs=1e-5)
    assert feats[1] == pytest.approx(0.3, abs=1e-5)
    assert feats[2] == pytest.approx(0.5, abs=1e-5)
    assert feats[3] == pytest.approx(np.std(probs), abs=1e-5)
    assert feats[4] == pytest.approx(0.5 * 0.7 * 0.3, abs=1e-5)  # prod
    assert feats[5] == pytest.approx((0.5 * 0.7 * 0.3) ** (1 / 3), abs=1e-5)  # geomavg


def test_known_values_for_entropies():
    probs = np.array([0.5, 0.5, 0.5])
    ents = np.array([0.1, 0.4, 0.9], dtype=np.float64)
    feats = extract_features(probs, ents)
    assert feats[6] == pytest.approx(0.9, abs=1e-5)
    assert feats[7] == pytest.approx(0.1, abs=1e-5)
    assert feats[8] == pytest.approx(np.mean(ents), abs=1e-5)
    assert feats[9] == pytest.approx(np.std(ents), abs=1e-5)
    assert feats[10] == pytest.approx(np.prod(ents), abs=1e-5)
    assert feats[11] == pytest.approx(np.prod(ents) ** (1 / 3), abs=1e-5)


def test_length_feature():
    probs = np.full(7, 0.5)
    ents = np.full(7, 0.3)
    feats = extract_features(probs, ents)
    assert feats[12] == 7.0


def test_log_space_does_not_overflow_or_nan_for_long_sequence():
    # 500 tokens at p=0.1: naive prod is 1e-500 (underflows even float64).
    # Log-space sum is ~-1151, and exp(-1151) underflows to 0.0 cleanly.
    # The key is: result is finite (no NaN, no inf), and prod/geomavg
    # are 0.0 / a small positive number respectively.
    probs = np.full(500, 0.1, dtype=np.float64)
    ents = np.full(500, 1.0, dtype=np.float64)
    feats = extract_features(probs, ents)
    assert np.isfinite(feats).all(), "Features contain NaN or inf"
    assert feats[4] >= 0.0  # prob_prod
    assert feats[5] > 0.0   # prob_geomavg should still be ~0.1


def test_log_space_geomavg_correct_for_long_sequence():
    """Geometric average is robust to N: geomavg([0.5]*1000) == 0.5."""
    probs = np.full(1000, 0.5, dtype=np.float64)
    ents = np.full(1000, 1.0, dtype=np.float64)
    feats = extract_features(probs, ents)
    assert feats[5] == pytest.approx(0.5, abs=1e-5)   # prob_geomavg
    assert feats[11] == pytest.approx(1.0, abs=1e-5)  # ent_geomavg


def test_zero_probability_does_not_explode():
    """A token with probability 0 (extreme low confidence) should not produce NaN."""
    probs = np.array([0.5, 0.0, 0.5], dtype=np.float64)
    ents = np.array([1.0, 1.0, 1.0])
    feats = extract_features(probs, ents)
    assert np.isfinite(feats).all()
    # log-space: log(clip(0, 1e-12, None)) = log(1e-12) ≈ -27.6
    # geomavg = exp(mean) ≈ exp((log(0.5) + -27.6 + log(0.5))/3) ≈ a small positive value
    assert feats[5] > 0


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        extract_features(np.array([0.5]), np.array([0.5, 0.5]))


def test_non_1d_input_raises():
    with pytest.raises(ValueError):
        extract_features(np.array([[0.5, 0.5]]), np.array([[1.0, 1.0]]))


def test_features_finite_for_arbitrary_random_input():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = int(rng.integers(1, 200))
        probs = rng.uniform(0.0, 1.0, size=n)
        ents = rng.uniform(0.0, 5.0, size=n)
        feats = extract_features(probs, ents)
        assert np.isfinite(feats).all()
