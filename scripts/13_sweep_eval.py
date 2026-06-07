"""Post-hoc t_rag sweep + the five metrics, dataset-aware, from cached C1/C2/C3.

Given the per-config JSONL a generation run produced (C1 no-retrieve, C2
always-retrieve, C3 card — C3 only needed for its stored ŝ₀), this replays:
  * CARD     : retrieve iff ŝ₀ < t
  * cascade  : retrieve iff ŝ₀ < t  OR  pyflakes fires on the zero-shot output
for every t in the grid (default 0.05…0.95 step 0.05) — no GPU, no regeneration
(ŝ₀ and the generations are frozen). It also reports the two t-independent
baselines C1 and C2.

Five metrics per row: exact_match, edit_similarity, identifier_f1, hallucination
(A4∧B2, with A4-only alongside), and latency (synthesised from the cached
per-instance generation latencies: zero-shot + retrieved-if-retrieved).

Dataset-aware scoring (via datasets.MULTILINE_DATASETS):
  * crosscodeeval_py   -> one scoring mode: "line"  (first line)
  * repoeval_function  -> TWO modes: "body" (truncate_to_function_body) and
                          "full" (raw, FIM-stripped) — both saved & evaluated.

Output: a long-format CSV (one row per dataset×mode×config×t) + a printed
summary. Circularity note: the cascade trigger and B2 are both pyflakes, so the
cascade's hallucination drop is partly structural; A4-only is the independent
column.

    python scripts/13_sweep_eval.py --results-dir results/qwen25_1.5b_crosscodeeval_py \\
        --dataset crosscodeeval_py --out-csv results/qwen25_1.5b_crosscodeeval_py/sweep.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import click  # noqa: E402
import numpy as np  # noqa: E402

from adaptive_retrieval.eval.datasets import (  # noqa: E402
    DATASET_LOADERS,
    LINE_COUNT_DATASETS,
    MULTILINE_DATASETS,
)
from adaptive_retrieval.metrics import (  # noqa: E402
    edit_similarity,
    exact_match,
    hallucinated_identifier_flag,
    identifier_f1,
    invented_identifier_flag,
    truncate_to_function_body,
    truncate_to_line_count,
)
from adaptive_retrieval.static_analysis.pyflakes_checker import PyflakesChecker  # noqa: E402

_FIM = ("<|", "▁<", "<fim", "<PRE>", "<SUF>", "<MID>", "<EOT>", "</s>", "<｜",
        "<repo_name>", "<file_sep>", "<|endoftext|>")


def _strip(text: str) -> str:
    for m in _FIM:
        i = text.find(m)
        if i != -1:
            text = text[:i]
    return text


def _clean(raw: str, gold: str, mode: str) -> str:
    s = _strip(raw)
    if mode == "line":
        return s.split("\n", 1)[0]
    if mode == "lines":
        # Fixed-size block: keep the gold's non-empty line count (Repoformer
        # chunk metric). Distinct from "line" (single-line CCE) — chunk golds
        # span 1-6 lines, so first-line-only would wreck multi-line blocks.
        return truncate_to_line_count(gold, s)
    if mode == "body":
        return truncate_to_function_body(gold, s)
    return s  # "full"


@click.command()
@click.option("--results-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--dataset", required=True, type=click.Choice(list(DATASET_LOADERS)))
@click.option("--out-csv", required=True, type=click.Path())
@click.option("--t-grid", default=",".join(f"{t:.2f}" for t in np.arange(0.05, 0.96, 0.05)))
def main(results_dir: str, dataset: str, out_csv: str, t_grid: str) -> None:
    D = Path(results_dir)
    load = lambda n: {r["instance_id"]: r for r in (
        json.loads(l) for l in open(D / n, encoding="utf-8") if l.strip())}
    c1, c2, c3 = load("C1_no_retrieve.jsonl"), load("C2_always_retrieve.jsonl"), load("C3_card.jsonl")
    insts = {i.instance_id: i for i in DATASET_LOADERS[dataset]()}
    ids = [i for i in sorted(set(c1) & set(c2) & set(c3) & set(insts))
           if c3[i].get("s_hat_0") is not None]
    n = len(ids)
    if dataset in MULTILINE_DATASETS:
        modes = ["body", "full"]          # function: dedent body + raw
    elif dataset in LINE_COUNT_DATASETS:
        modes = ["lines", "full"]         # chunk: gold-line-count + raw
    else:
        modes = ["line"]                  # single-line (CCE)
    thresholds = [round(float(x), 4) for x in t_grid.split(",")]
    pf = PyflakesChecker()
    LAT = lambda r: float(r.get("latency_ms") or 0.0)
    print(f"[setup] dataset={dataset}  n={n}  modes={modes}  thresholds={len(thresholds)}")

    shat = {i: float(c3[i]["s_hat_0"]) for i in ids}
    l1 = {i: LAT(c1[i]) for i in ids}
    l2 = {i: LAT(c2[i]) for i in ids}

    # Per mode: cleaned predictions + per-instance metrics + the cascade trigger.
    es1, es2, em1, em2, f11, f12 = ({m: {} for m in modes} for _ in range(6))
    hAB1, hAB2, hA1, hA2, trig = ({m: {} for m in modes} for _ in range(5))
    for i in ids:
        inst = insts[i]
        gold = inst.ground_truth
        for m in modes:
            p1, p2 = _clean(c1[i]["prediction"], gold, m), _clean(c2[i]["prediction"], gold, m)
            es1[m][i], es2[m][i] = edit_similarity(gold, p1), edit_similarity(gold, p2)
            em1[m][i], em2[m][i] = exact_match(gold, p1), exact_match(gold, p2)
            f11[m][i], f12[m][i] = identifier_f1(gold, p1), identifier_f1(gold, p2)
            u1 = set(pf.analyze(p1, inst.x_left, inst.x_right).significant_out_of_scope)
            u2 = set(pf.analyze(p2, inst.x_left, inst.x_right).significant_out_of_scope)
            hAB1[m][i] = 1 if invented_identifier_flag(gold, p1, u1) else 0
            hAB2[m][i] = 1 if invented_identifier_flag(gold, p2, u2) else 0
            hA1[m][i] = 1 if hallucinated_identifier_flag(gold, p1) else 0
            hA2[m][i] = 1 if hallucinated_identifier_flag(gold, p2) else 0
            trig[m][i] = bool(u1)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    def agg(mode, pick, two_pass):  # pick(i)->True uses C2 (retrieved) else C1
        g = lambda d1, d2: np.mean([d2[mode][i] if pick(i) else d1[mode][i] for i in ids])
        if two_pass:   # CARD/cascade: always a zero-shot probe + a conditional retrieved gen
            lat = np.mean([l1[i] + (l2[i] if pick(i) else 0.0) for i in ids])
        else:          # baselines: exactly one generation
            lat = np.mean([l2[i] if pick(i) else l1[i] for i in ids])
        retr = np.mean([1 if pick(i) else 0 for i in ids])
        return dict(retrieval_pct=100 * retr, exact_match=g(em1, em2), edit_similarity=g(es1, es2),
                    identifier_f1=g(f11, f12), hall_A4B2=g(hAB1, hAB2), hall_A4=g(hA1, hA2),
                    latency_ms=lat)

    for mode in modes:
        rows.append(dict(dataset=dataset, scoring=mode, config="C1_no_retrieve", t_rag="",
                         **agg(mode, lambda i: False, two_pass=False)))
        rows.append(dict(dataset=dataset, scoring=mode, config="C2_always_retrieve", t_rag="",
                         **agg(mode, lambda i: True, two_pass=False)))
        for t in thresholds:
            rows.append(dict(dataset=dataset, scoring=mode, config="C3_card", t_rag=t,
                             **agg(mode, lambda i, t=t: shat[i] < t, two_pass=True)))
            rows.append(dict(dataset=dataset, scoring=mode, config="C4_cascade", t_rag=t,
                             **agg(mode, lambda i, t=t: shat[i] < t or trig[mode][i], two_pass=True)))

    fields = ["dataset", "scoring", "config", "t_rag", "retrieval_pct", "exact_match",
              "edit_similarity", "identifier_f1", "hall_A4B2", "hall_A4", "latency_ms"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"[done] wrote {len(rows)} rows -> {out_path}")

    # Printed summary: baselines + cascade-vs-CARD at a few thresholds.
    for mode in modes:
        print(f"\n=== {dataset} [{mode}] ===")
        base = {r["config"]: r for r in rows if r["scoring"] == mode and r["config"].startswith(("C1", "C2"))}
        for c in ("C1_no_retrieve", "C2_always_retrieve"):
            b = base[c]
            print(f"  {c:<20} EM={b['exact_match']:.3f} ES={b['edit_similarity']:.3f} "
                  f"hallA4B2={b['hall_A4B2']:.4f} lat={b['latency_ms']:.0f}ms")
        print(f"  {'t':>5} | {'CARD retr':>9} {'ES':>6} {'hallA4B2':>9} | {'CASC retr':>9} {'ES':>6} {'hallA4B2':>9} {'Δhall':>8}")
        for t in thresholds:
            cd = next(r for r in rows if r["scoring"] == mode and r["config"] == "C3_card" and r["t_rag"] == t)
            cx = next(r for r in rows if r["scoring"] == mode and r["config"] == "C4_cascade" and r["t_rag"] == t)
            print(f"  {t:>5.2f} | {cd['retrieval_pct']:>8.1f}% {cd['edit_similarity']:>6.3f} {cd['hall_A4B2']:>9.4f} | "
                  f"{cx['retrieval_pct']:>8.1f}% {cx['edit_similarity']:>6.3f} {cx['hall_A4B2']:>9.4f} {cx['hall_A4B2']-cd['hall_A4B2']:>+8.4f}")


if __name__ == "__main__":
    main()
