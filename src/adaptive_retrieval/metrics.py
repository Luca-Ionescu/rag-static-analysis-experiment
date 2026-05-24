"""Accuracy, hallucination, efficiency, and statistical-test metrics.

Per IMPLEMENTATION_GUIDE §13. ``repository_symbol_precision`` and
``hallucination_flag`` depend on the static-analysis ``PredictionAnalyzer``;
all other metrics are self-contained.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
from Levenshtein import distance as lev_distance
from scipy.stats import binomtest

from .static_analysis.analyzer import PredictionAnalyzer


# ---------- accuracy ----------

def exact_match(reference: str, prediction: str) -> bool:
    return reference.rstrip() == prediction.rstrip()


def edit_similarity(reference: str, prediction: str) -> float:
    """ES per CARD §2.1, value in [0, 1] (higher = better)."""
    if not reference and not prediction:
        return 1.0
    denom = max(len(reference), len(prediction))
    if denom == 0:
        return 1.0
    return 1.0 - lev_distance(reference, prediction) / denom


_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def _identifiers(text: str) -> list[str]:
    return _IDENT_RE.findall(text)


def identifier_f1(reference: str, prediction: str) -> float:
    """CrossCodeEval-style Identifier-F1. Set-based P/R over identifier tokens."""
    ref_ids = set(_identifiers(reference))
    pred_ids = set(_identifiers(prediction))
    if not ref_ids and not pred_ids:
        return 1.0
    if not ref_ids or not pred_ids:
        return 0.0
    tp = len(ref_ids & pred_ids)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_ids)
    recall = tp / len(ref_ids)
    return 2 * precision * recall / (precision + recall)


# ---------- hallucination ----------

def repository_symbol_precision(
    prediction: str,
    x_left: str,
    x_right: str,
    analyzer: PredictionAnalyzer,
) -> float:
    """Fraction of identifiers in the prediction that are visible at the hole.

    Under the unified Tier 1 design this is purely a scope check — it does
    not consult the repository symbol table. The metric is a descriptive
    continuous companion to the binary ``hallucination_flag``.
    """
    result = analyzer.analyze(prediction, x_left, x_right)
    n_total = result.n_used_identifiers
    if n_total == 0:
        return 1.0
    n_out_of_scope = len(result.out_of_scope_identifiers)
    return (n_total - n_out_of_scope) / n_total


def hallucination_flag(
    prediction: str,
    x_left: str,
    x_right: str,
    analyzer: PredictionAnalyzer,
) -> bool:
    """True iff any static-analysis tier found something: an out-of-scope
    identifier in a significant position (Tier 1), a signature mismatch
    (Tier 2), or a wrong-origin import (Tier 3). Independent of which tiers
    are enabled at trigger time — reflects what the analyzer actually found,
    not whether the cascade would have acted on it.
    """
    result = analyzer.analyze(prediction, x_left, x_right)
    return (
        bool(result.significant_out_of_scope)
        or bool(result.signature_issues)
        or bool(result.import_issues)
    )


# ---------- efficiency (aggregate) ----------

def percent_retrieval(records: Iterable[dict]) -> float:
    records = list(records)
    if not records:
        return 0.0
    return 100.0 * sum(1 for r in records if r.get("retrieved")) / len(records)


def mean_latency_ms(records: Iterable[dict]) -> float:
    records = list(records)
    if not records:
        return 0.0
    return sum(r.get("latency_ms", 0.0) for r in records) / len(records)


# ---------- statistical tests ----------

def mcnemar_test(
    records_a: list[dict],
    records_b: list[dict],
    key: str = "hallucinated",
) -> dict:
    """Paired McNemar test for a binary outcome.

    Returns ``{"p_value", "b", "c"}`` where:
    - b: count of (A=0, B=1) — instances where B is worse
    - c: count of (A=1, B=0) — instances where B is better

    Uses the exact binomial test (no continuity correction) which is
    appropriate for the small b+c counts we expect.
    """
    if len(records_a) != len(records_b):
        raise ValueError(
            f"Paired test needs equal lengths: {len(records_a)} vs {len(records_b)}"
        )
    b = sum(1 for ra, rb in zip(records_a, records_b) if not ra[key] and rb[key])
    c = sum(1 for ra, rb in zip(records_a, records_b) if ra[key] and not rb[key])
    if b + c == 0:
        return {"p_value": 1.0, "b": 0, "c": 0}
    result = binomtest(min(b, c), b + c, p=0.5)
    return {"p_value": float(result.pvalue), "b": b, "c": c}


def paired_bootstrap(
    records_a: list[dict],
    records_b: list[dict],
    key: str,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap 95% CI for ``mean(B[key] - A[key])``."""
    if len(records_a) != len(records_b):
        raise ValueError(
            f"Paired bootstrap needs equal lengths: {len(records_a)} vs {len(records_b)}"
        )
    a_vals = np.asarray([r[key] for r in records_a], dtype=np.float64)
    b_vals = np.asarray([r[key] for r in records_b], dtype=np.float64)
    diffs = b_vals - a_vals
    n = len(diffs)
    if n == 0:
        return {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = diffs[idx].mean(axis=1)
    return {
        "mean_diff": float(diffs.mean()),
        "ci_lower": float(np.percentile(boot_means, 2.5)),
        "ci_upper": float(np.percentile(boot_means, 97.5)),
    }
