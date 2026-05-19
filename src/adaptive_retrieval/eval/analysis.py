"""Phase 7 analyses: trigger breakdown, disagreement, McNemar, bootstrap, threshold sweep.

Operates on the JSONL records written by ``runner.run_experiment``. Each
function returns a Python dict / list-of-dicts that's both directly
interpretable and easy to dump to JSON for the paper.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import jsonlines

from ..metrics import mcnemar_test, paired_bootstrap


# ---------- loading ----------

def load_records(path: str | Path) -> list[dict]:
    with jsonlines.open(path) as r:
        return list(r)


# ---------- per-trigger-reason breakdown (cascade only) ----------

def trigger_reason_breakdown(records: list[dict]) -> list[dict]:
    """For each trigger_reason in the cascade output, report count and mean
    accuracy / hallucination / latency.

    Returns a list ordered: none, card, static_unresolved, static_crossfile,
    plus any other reasons that appear (e.g. ``always`` for C2 records,
    ``oracle`` for C6).
    """
    by_reason: dict[str, list[dict]] = {}
    for r in records:
        reason = r.get("trigger_reason", "none")
        by_reason.setdefault(reason, []).append(r)

    ordered_keys = []
    for k in ("none", "card", "static_unresolved", "static_crossfile"):
        if k in by_reason:
            ordered_keys.append(k)
    for k in by_reason:
        if k not in ordered_keys:
            ordered_keys.append(k)

    out = []
    for k in ordered_keys:
        sub = by_reason[k]
        n = len(sub)
        out.append(
            {
                "trigger_reason": k,
                "n": n,
                "fraction": n / len(records) if records else 0.0,
                "edit_similarity": _mean(sub, "edit_similarity"),
                "identifier_f1": _mean(sub, "identifier_f1"),
                "hallucination_rate": _mean_bool(sub, "hallucinated"),
                "mean_latency_ms": (
                    sum(r.get("latency_ms", 0.0) for r in sub) / n if n else 0.0
                ),
            }
        )
    return out


def _mean(records: list[dict], key: str) -> float:
    if not records:
        return 0.0
    return sum(float(r["metrics"][key]) for r in records) / len(records)


def _mean_bool(records: list[dict], key: str) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if r["metrics"][key]) / len(records)


# ---------- disagreement analysis ----------

def disagreement_analysis(
    card_records: list[dict],
    cascade_records: list[dict],
) -> dict:
    """Among instances where CARD said "no retrieve", how many did the cascade
    retrieve (because static analysis fired)? What's the accuracy difference?

    Returns a dict with the four-way breakdown:
        card_no_cascade_no    (both skipped)
        card_no_cascade_yes   (the static-analysis save)
        card_yes_cascade_no   (impossible by the asymmetric cascade)
        card_yes_cascade_yes  (both retrieved)
    plus per-bucket mean ES.
    """
    by_id_card = {r["instance_id"]: r for r in card_records}
    by_id_cascade = {r["instance_id"]: r for r in cascade_records}
    shared = sorted(set(by_id_card) & set(by_id_cascade))
    if not shared:
        raise ValueError("No overlapping instance_ids between CARD and cascade records")

    buckets: dict[str, list[tuple[dict, dict]]] = {
        "card_no_cascade_no": [],
        "card_no_cascade_yes": [],
        "card_yes_cascade_no": [],
        "card_yes_cascade_yes": [],
    }
    for iid in shared:
        c = by_id_card[iid]
        x = by_id_cascade[iid]
        k = ("card_yes" if c["retrieved"] else "card_no") + (
            "_cascade_yes" if x["retrieved"] else "_cascade_no"
        )
        buckets[k].append((c, x))

    summary = {"n_shared": len(shared)}
    for k, pairs in buckets.items():
        n = len(pairs)
        if n == 0:
            summary[k] = {"n": 0, "fraction": 0.0}
            continue
        card_es = sum(float(c["metrics"]["edit_similarity"]) for c, _ in pairs) / n
        cas_es = sum(float(x["metrics"]["edit_similarity"]) for _, x in pairs) / n
        card_hall = sum(1 for c, _ in pairs if c["metrics"]["hallucinated"]) / n
        cas_hall = sum(1 for _, x in pairs if x["metrics"]["hallucinated"]) / n
        summary[k] = {
            "n": n,
            "fraction": n / len(shared),
            "card_mean_es": card_es,
            "cascade_mean_es": cas_es,
            "card_hallucination_rate": card_hall,
            "cascade_hallucination_rate": cas_hall,
        }
    return summary


# ---------- statistical tests ----------

def hallucination_mcnemar(
    card_records: list[dict],
    cascade_records: list[dict],
) -> dict:
    """Paired McNemar's exact test on hallucinated flag, aligned by instance_id.

    Returns ``{p_value, b, c, n}`` where:
      b = #(CARD ok, cascade hallucinated)  — cascade is worse
      c = #(CARD hallucinated, cascade ok)  — cascade is better
    """
    card_by_id = {r["instance_id"]: r for r in card_records}
    cas_by_id = {r["instance_id"]: r for r in cascade_records}
    shared = sorted(set(card_by_id) & set(cas_by_id))
    a = [{"h": bool(card_by_id[i]["metrics"]["hallucinated"])} for i in shared]
    b = [{"h": bool(cas_by_id[i]["metrics"]["hallucinated"])} for i in shared]
    out = mcnemar_test(a, b, key="h")
    out["n"] = len(shared)
    return out


def es_paired_bootstrap(
    a_records: list[dict],
    b_records: list[dict],
    n_resamples: int = 10_000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap 95% CI for mean ES(b) - mean ES(a), aligned by instance_id."""
    a_by_id = {r["instance_id"]: r for r in a_records}
    b_by_id = {r["instance_id"]: r for r in b_records}
    shared = sorted(set(a_by_id) & set(b_by_id))
    a_aligned = [{"es": float(a_by_id[i]["metrics"]["edit_similarity"])} for i in shared]
    b_aligned = [{"es": float(b_by_id[i]["metrics"]["edit_similarity"])} for i in shared]
    out = paired_bootstrap(a_aligned, b_aligned, key="es", n_resamples=n_resamples, seed=seed)
    out["n"] = len(shared)
    return out


# ---------- T_RAG threshold sweep ----------

def threshold_sweep_from_card(
    card_records_with_s_hat: list[dict],
    thresholds: Iterable[float],
) -> list[dict]:
    """Recover what CARD's accuracy / retrieval-rate would have been at
    different T_RAG thresholds, *given the existing JSONL records*.

    Requires the JSONL to include ``s_hat_0`` per record. For each candidate
    threshold:
      - if s_hat_0 < threshold: the prediction is the RAG one (if available)
      - else: the prediction is the zero-shot one (the recorded one when
        retrieved=False, OR derived from a companion no-retrieve run).

    Limitation: when ``retrieved=False`` in the JSONL, we only have ŷ⁰
    (which is also the final prediction). When ``retrieved=True``, we only
    have ŷ_rag. We can therefore only sweep thresholds in a single direction
    from the original threshold — pair with a C1 no-retrieve run and a C2
    always-retrieve run on the same instances to do a full sweep.

    For exact sweeps, use ``threshold_sweep_paired`` below.
    """
    rows = []
    for t in thresholds:
        n = 0
        n_retrieved = 0
        for r in card_records_with_s_hat:
            s = r.get("s_hat_0")
            if s is None:
                continue
            would_retrieve = s < t
            n += 1
            if would_retrieve:
                n_retrieved += 1
        rows.append(
            {
                "t_rag": float(t),
                "n": n,
                "would_retrieve": n_retrieved,
                "percent_retrieval": 100.0 * n_retrieved / n if n else 0.0,
            }
        )
    return rows


def threshold_sweep_paired(
    no_retrieve_records: list[dict],
    always_retrieve_records: list[dict],
    s_hats_by_id: dict[str, float],
    thresholds: Iterable[float],
) -> list[dict]:
    """Paired T_RAG sweep — picks ŷ⁰ vs ŷ_rag per instance based on whether
    ``s_hat < t``, then reports mean ES, mean hallucination, retrieval %.

    This is the version you'd plot for the paper.
    """
    no_by_id = {r["instance_id"]: r for r in no_retrieve_records}
    yes_by_id = {r["instance_id"]: r for r in always_retrieve_records}
    shared = sorted(set(no_by_id) & set(yes_by_id) & set(s_hats_by_id))

    rows = []
    for t in thresholds:
        es_sum = 0.0
        hall_sum = 0
        n_retr = 0
        for iid in shared:
            s = s_hats_by_id[iid]
            chosen = yes_by_id[iid] if s < t else no_by_id[iid]
            es_sum += float(chosen["metrics"]["edit_similarity"])
            hall_sum += int(bool(chosen["metrics"]["hallucinated"]))
            if s < t:
                n_retr += 1
        n = len(shared)
        rows.append(
            {
                "t_rag": float(t),
                "n": n,
                "percent_retrieval": 100.0 * n_retr / n if n else 0.0,
                "mean_edit_similarity": es_sum / n if n else 0.0,
                "hallucination_rate": hall_sum / n if n else 0.0,
            }
        )
    return rows
