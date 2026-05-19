"""Tests for Phase 7 analysis: trigger breakdown, disagreement, statistical tests, threshold sweep."""
from __future__ import annotations

import jsonlines
import pytest

from adaptive_retrieval.eval.analysis import (
    disagreement_analysis,
    es_paired_bootstrap,
    hallucination_mcnemar,
    load_records,
    threshold_sweep_from_card,
    threshold_sweep_paired,
    trigger_reason_breakdown,
)


def _record(
    iid: str,
    *,
    config: str = "C4_cascade",
    retrieved: bool = False,
    trigger_reason: str = "none",
    s_hat_0: float | None = None,
    es: float = 0.5,
    hall: bool = False,
    latency_ms: float = 100.0,
):
    return {
        "instance_id": iid,
        "config": config,
        "retrieved": retrieved,
        "trigger_reason": trigger_reason,
        "s_hat_0": s_hat_0,
        "latency_ms": latency_ms,
        "metrics": {
            "exact_match": False,
            "edit_similarity": es,
            "identifier_f1": 0.0,
            "repo_symbol_precision": 1.0,
            "hallucinated": hall,
        },
    }


# ---------- load_records ----------

def test_load_records_reads_jsonl(tmp_path):
    p = tmp_path / "x.jsonl"
    with jsonlines.open(p, "w") as w:
        w.write(_record("a"))
        w.write(_record("b"))
    recs = load_records(p)
    assert len(recs) == 2
    assert recs[0]["instance_id"] == "a"
    assert recs[1]["instance_id"] == "b"


# ---------- trigger_reason_breakdown ----------

def test_trigger_breakdown_orders_known_reasons():
    records = [
        _record("a", trigger_reason="card", es=0.8),
        _record("b", trigger_reason="card", es=0.6),
        _record("c", trigger_reason="none", es=0.4),
        _record("d", trigger_reason="static_unresolved", es=0.7, hall=True),
        _record("e", trigger_reason="static_crossfile", es=0.9),
    ]
    rows = trigger_reason_breakdown(records)
    order = [r["trigger_reason"] for r in rows]
    assert order == ["none", "card", "static_unresolved", "static_crossfile"]


def test_trigger_breakdown_counts_and_means():
    records = [
        _record("a", trigger_reason="card", es=0.8, hall=False),
        _record("b", trigger_reason="card", es=0.4, hall=True),
        _record("c", trigger_reason="none", es=1.0),
    ]
    rows = trigger_reason_breakdown(records)
    card_row = next(r for r in rows if r["trigger_reason"] == "card")
    assert card_row["n"] == 2
    assert card_row["fraction"] == pytest.approx(2 / 3)
    assert card_row["edit_similarity"] == pytest.approx(0.6)
    assert card_row["hallucination_rate"] == pytest.approx(0.5)


def test_trigger_breakdown_handles_empty_records():
    assert trigger_reason_breakdown([]) == []


def test_trigger_breakdown_unknown_reason_appears_after_known():
    records = [
        _record("a", trigger_reason="always"),  # C2 records use this
        _record("b", trigger_reason="card"),
    ]
    rows = trigger_reason_breakdown(records)
    # Known reasons come first (card), then unknown (always).
    assert [r["trigger_reason"] for r in rows] == ["card", "always"]


# ---------- disagreement_analysis ----------

def test_disagreement_four_way_split():
    card = [
        _record("i1", retrieved=False, es=0.5),  # card_no
        _record("i2", retrieved=False, es=0.3),  # card_no
        _record("i3", retrieved=True, es=0.9, trigger_reason="card"),
    ]
    cascade = [
        _record("i1", retrieved=False, es=0.5),  # both skipped
        _record("i2", retrieved=True, es=0.7),   # cascade added retrieval
        _record("i3", retrieved=True, es=0.9, trigger_reason="card"),
    ]
    d = disagreement_analysis(card, cascade)
    assert d["n_shared"] == 3
    assert d["card_no_cascade_no"]["n"] == 1
    assert d["card_no_cascade_yes"]["n"] == 1   # the static-analysis save
    assert d["card_yes_cascade_yes"]["n"] == 1
    # card_no_cascade_yes should show ES improvement
    save = d["card_no_cascade_yes"]
    assert save["card_mean_es"] == pytest.approx(0.3)
    assert save["cascade_mean_es"] == pytest.approx(0.7)


def test_disagreement_raises_on_no_overlap():
    a = [_record("x")]
    b = [_record("y")]
    with pytest.raises(ValueError):
        disagreement_analysis(a, b)


def test_disagreement_card_yes_cascade_no_is_impossible_under_invariant():
    # The asymmetric cascade should never produce this combination, but the
    # analysis function still tolerates it gracefully (n=0).
    card = [_record("i1", retrieved=True, es=0.9)]
    cascade = [_record("i1", retrieved=False, es=0.9)]
    d = disagreement_analysis(card, cascade)
    # We DO get the combination here because the test data violates the
    # invariant — the analysis function reports it honestly.
    assert d["card_yes_cascade_no"]["n"] == 1


# ---------- hallucination_mcnemar ----------

def test_mcnemar_no_disagreement_returns_p1():
    card = [_record("a", hall=True), _record("b", hall=False)]
    cas = [_record("a", hall=True), _record("b", hall=False)]
    out = hallucination_mcnemar(card, cas)
    assert out == {"p_value": 1.0, "b": 0, "c": 0, "n": 2}


def test_mcnemar_cascade_better_lower_p():
    card = [_record(f"i{k}", hall=True) for k in range(10)]
    cas = [_record(f"i{k}", hall=False) for k in range(10)]
    out = hallucination_mcnemar(card, cas)
    # b=0 (cascade ok in all 10), c=10 (CARD hallucinated 10 cases, cascade fixed)
    assert out["b"] == 0 and out["c"] == 10
    assert out["p_value"] < 0.01


def test_mcnemar_uses_intersection_of_ids():
    card = [_record("a", hall=True), _record("b", hall=True)]
    cas = [_record("a", hall=False), _record("c", hall=False)]
    out = hallucination_mcnemar(card, cas)
    # Only 'a' is shared. b=0, c=1.
    assert out["n"] == 1


# ---------- es_paired_bootstrap ----------

def test_paired_bootstrap_recovers_mean_diff():
    card = [_record(f"i{k}", es=0.5) for k in range(50)]
    cas = [_record(f"i{k}", es=0.7) for k in range(50)]
    out = es_paired_bootstrap(card, cas)
    assert out["mean_diff"] == pytest.approx(0.2, abs=1e-9)
    assert out["ci_lower"] <= 0.2 + 1e-9
    assert out["ci_upper"] >= 0.2 - 1e-9


def test_paired_bootstrap_only_counts_shared_ids():
    card = [_record("a", es=0.5)]
    cas = [_record("a", es=0.7), _record("b", es=0.9)]
    out = es_paired_bootstrap(card, cas)
    assert out["n"] == 1


# ---------- threshold sweeps ----------

def test_threshold_sweep_from_card_counts_retrievals_at_each_t():
    records = [
        _record("a", s_hat_0=0.4),
        _record("b", s_hat_0=0.8),
        _record("c", s_hat_0=0.6),
    ]
    rows = threshold_sweep_from_card(records, [0.5, 0.7, 0.9])
    # t=0.5: only s_hat=0.4 would retrieve → 1
    # t=0.7: s_hat=0.4 and 0.6 retrieve → 2
    # t=0.9: all three retrieve → 3
    assert rows[0]["would_retrieve"] == 1
    assert rows[1]["would_retrieve"] == 2
    assert rows[2]["would_retrieve"] == 3


def test_threshold_sweep_paired_picks_correct_branch_per_t():
    # ŷ⁰ has ES=0.5 everywhere; ŷ_rag has ES=0.9 everywhere.
    no = [_record("a", es=0.5), _record("b", es=0.5)]
    yes = [_record("a", es=0.9), _record("b", es=0.9)]
    s_hats = {"a": 0.6, "b": 0.8}

    # At t=0.5: no instance below threshold → all keep ŷ⁰ → mean ES 0.5
    # At t=0.7: a (0.6) below threshold → swap to RAG; b stays → mean ES (0.9+0.5)/2 = 0.7
    # At t=0.9: both below → both swap → mean ES 0.9
    rows = threshold_sweep_paired(no, yes, s_hats, [0.5, 0.7, 0.9])
    assert rows[0]["percent_retrieval"] == 0.0
    assert rows[0]["mean_edit_similarity"] == pytest.approx(0.5)
    assert rows[1]["percent_retrieval"] == 50.0
    assert rows[1]["mean_edit_similarity"] == pytest.approx(0.7)
    assert rows[2]["percent_retrieval"] == 100.0
    assert rows[2]["mean_edit_similarity"] == pytest.approx(0.9)


def test_threshold_sweep_paired_handles_missing_s_hats():
    no = [_record("a", es=0.5)]
    yes = [_record("a", es=0.9), _record("b", es=0.9)]
    s_hats = {"a": 0.5}  # 'b' has no s_hat
    rows = threshold_sweep_paired(no, yes, s_hats, [0.7])
    # Only 'a' is shared across all three inputs.
    assert rows[0]["n"] == 1
