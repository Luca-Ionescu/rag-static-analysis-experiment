"""CARD's 13-D feature vector (Table 1 of Zhang et al. 2024).

Six statistics over per-token chosen-token probabilities (max, min, avg, std,
prod, geomavg), the same six over per-token entropies, plus generation
length. ``prod`` and ``geomavg`` are computed in log-space to avoid float
underflow on long sequences — for 50 tokens with avg prob 0.1 a naive
``np.prod`` would already approach the float32 floor.
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES: tuple[str, ...] = (
    "prob_max", "prob_min", "prob_avg", "prob_std", "prob_prod", "prob_geomavg",
    "ent_max", "ent_min", "ent_avg", "ent_std", "ent_prod", "ent_geomavg",
    "length",
)
assert len(FEATURE_NAMES) == 13


def extract_features(
    token_probs: np.ndarray,
    token_entropies: np.ndarray,
) -> np.ndarray:
    """Return a 13-D float32 feature vector following ``FEATURE_NAMES`` order.

    Args:
        token_probs: shape (N,), per-token chosen-token probabilities in [0, 1].
        token_entropies: shape (N,), per-token entropies in [0, +inf).

    Returns:
        np.ndarray of shape (13,).
    """
    token_probs = np.asarray(token_probs)
    token_entropies = np.asarray(token_entropies)
    if token_probs.shape != token_entropies.shape:
        raise ValueError(
            f"token_probs/token_entropies shape mismatch: "
            f"{token_probs.shape} vs {token_entropies.shape}"
        )
    if token_probs.ndim != 1:
        raise ValueError(f"Expected 1-D arrays, got {token_probs.ndim}-D")

    n = len(token_probs)
    if n == 0:
        return np.zeros(13, dtype=np.float32)

    feats: list[float] = []
    for x in (token_probs, token_entropies):
        feats.append(float(np.max(x)))
        feats.append(float(np.min(x)))
        feats.append(float(np.mean(x)))
        feats.append(float(np.std(x)))
        # Log-space prod and geomavg. exp(sum(log)) underflows to 0 for huge N,
        # which is the correct float representation — not NaN/inf.
        log_x = np.log(np.clip(x, 1e-12, None))
        feats.append(float(np.exp(np.sum(log_x))))
        feats.append(float(np.exp(np.mean(log_x))))
    feats.append(float(n))

    # Clip to float32 range before downcast. ``ent_prod`` can overflow
    # float32 when entropies are large and N is moderate (e.g. 5^170);
    # the float64 computation is fine but the cast would produce inf.
    f32_max = float(np.finfo(np.float32).max)
    arr = np.clip(np.asarray(feats, dtype=np.float64), -f32_max, f32_max)
    return arr.astype(np.float32)
