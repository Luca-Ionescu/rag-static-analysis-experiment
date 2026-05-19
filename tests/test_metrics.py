"""Tests for metrics: accuracy, hallucination, efficiency, statistical tests."""
from __future__ import annotations

from adaptive_retrieval.metrics import (
    edit_similarity,
    exact_match,
    hallucination_flag,
    identifier_f1,
    mcnemar_test,
    mean_latency_ms,
    paired_bootstrap,
    percent_retrieval,
    repository_symbol_precision,
)
from adaptive_retrieval.static_analysis.analyzer import PredictionAnalyzer
from adaptive_retrieval.static_analysis.scope import InFileScopeAnalyzer
from adaptive_retrieval.static_analysis.symbol_table import RepositorySymbolTable


# ---------- accuracy ----------

def test_exact_match_strips_trailing_whitespace():
    assert exact_match("foo", "foo")
    assert exact_match("foo\n", "foo")
    assert exact_match("foo  ", "foo\n")
    assert not exact_match("foo", "bar")
    assert not exact_match("foo", " foo")  # leading whitespace differs


def test_edit_similarity_endpoints():
    assert edit_similarity("abc", "abc") == 1.0
    assert edit_similarity("", "") == 1.0
    assert edit_similarity("abc", "") == 0.0
    assert edit_similarity("", "abc") == 0.0


def test_edit_similarity_partial_match():
    es = edit_similarity("self.client.send(message)", "self.client.send(msg)")
    # Differ in only a few chars; ES should be high.
    assert 0.7 < es < 1.0


def test_identifier_f1_perfect_match():
    assert identifier_f1("foo bar baz", "foo bar baz") == 1.0


def test_identifier_f1_disjoint_sets():
    assert identifier_f1("foo bar", "baz qux") == 0.0


def test_identifier_f1_partial_overlap():
    # ref={a,b,c}, pred={a,b,d}: tp=2, p=2/3, r=2/3, f1=2/3
    val = identifier_f1("a b c", "a b d")
    assert abs(val - 2 / 3) < 1e-9


def test_identifier_f1_both_empty_strings():
    assert identifier_f1("", "") == 1.0
    assert identifier_f1("x", "") == 0.0
    assert identifier_f1("", "x") == 0.0


# ---------- hallucination ----------

def test_repo_symbol_precision_and_flag(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    syms = RepositorySymbolTable(repo)
    analyzer = PredictionAnalyzer(InFileScopeAnalyzer(), syms)

    # Hallucinated call: "totally_fake" not visible anywhere.
    rsp = repository_symbol_precision(
        prediction="totally_fake()",
        x_left="def main():\n    return ",
        x_right="\n",
        analyzer=analyzer,
    )
    assert 0.0 <= rsp < 1.0
    assert hallucination_flag(
        prediction="totally_fake()",
        x_left="def main():\n    return ",
        x_right="\n",
        analyzer=analyzer,
    )


def test_repo_symbol_precision_clean_prediction(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    syms = RepositorySymbolTable(repo)
    analyzer = PredictionAnalyzer(InFileScopeAnalyzer(), syms)

    # All names resolved (locally defined + builtin).
    rsp = repository_symbol_precision(
        prediction="helper(x)",
        x_left="def helper(x):\n    return x\n\ndef main(x):\n    return ",
        x_right="\n",
        analyzer=analyzer,
    )
    assert rsp == 1.0
    assert not hallucination_flag(
        prediction="helper(x)",
        x_left="def helper(x):\n    return x\n\ndef main(x):\n    return ",
        x_right="\n",
        analyzer=analyzer,
    )


# ---------- efficiency ----------

def test_percent_retrieval():
    records = [{"retrieved": True}, {"retrieved": False}, {"retrieved": True}]
    assert abs(percent_retrieval(records) - 66.666) < 0.01
    assert percent_retrieval([]) == 0.0


def test_mean_latency_ms():
    records = [{"latency_ms": 100.0}, {"latency_ms": 200.0}]
    assert mean_latency_ms(records) == 150.0
    assert mean_latency_ms([]) == 0.0


# ---------- statistical tests ----------

def test_mcnemar_no_disagreement():
    a = [{"hallucinated": True}, {"hallucinated": False}]
    b = [{"hallucinated": True}, {"hallucinated": False}]
    r = mcnemar_test(a, b)
    assert r == {"p_value": 1.0, "b": 0, "c": 0}


def test_mcnemar_b_and_c_counts():
    # 3 cases where A=0 and B=1 (B worse) — b=3
    # 0 cases where A=1 and B=0 — c=0
    a = [{"h": False}] * 3 + [{"h": True}] * 2
    b = [{"h": True}] * 3 + [{"h": True}] * 2
    r = mcnemar_test(a, b, key="h")
    assert r["b"] == 3 and r["c"] == 0
    assert 0.0 <= r["p_value"] <= 1.0


def test_mcnemar_significant():
    # Strong imbalance: b=10, c=0 → very low p
    a = [{"h": False}] * 10
    b = [{"h": True}] * 10
    r = mcnemar_test(a, b, key="h")
    assert r["b"] == 10 and r["c"] == 0
    assert r["p_value"] < 0.01


def test_mcnemar_unequal_lengths_raises():
    import pytest
    with pytest.raises(ValueError):
        mcnemar_test([{"h": True}], [{"h": True}, {"h": False}])


def test_paired_bootstrap_known_diff():
    a = [{"es": 0.5} for _ in range(100)]
    b = [{"es": 0.7} for _ in range(100)]
    r = paired_bootstrap(a, b, key="es")
    assert abs(r["mean_diff"] - 0.2) < 1e-9
    # CI must contain the true diff (zero variance, so CI tightly around 0.2).
    # Allow tiny float64 rounding slack at the boundaries.
    assert r["ci_lower"] <= 0.2 + 1e-9
    assert r["ci_upper"] >= 0.2 - 1e-9


def test_paired_bootstrap_with_variance():
    import numpy as np
    rng = np.random.default_rng(0)
    a = [{"es": float(rng.uniform(0.3, 0.5))} for _ in range(200)]
    b = [{"es": a[i]["es"] + 0.1 + float(rng.normal(0, 0.05))} for i in range(200)]
    r = paired_bootstrap(a, b, key="es")
    # mean diff should be near 0.1; CI should contain 0.1
    assert 0.05 < r["mean_diff"] < 0.15
    assert r["ci_lower"] < r["mean_diff"] < r["ci_upper"]


def test_paired_bootstrap_empty_inputs():
    r = paired_bootstrap([], [], key="es")
    assert r == {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
