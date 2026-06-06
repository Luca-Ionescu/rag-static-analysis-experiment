"""T_RAG sweep with the A4∧B2 (pyflakes) hallucination metric, dataset-aware.

Replays, from cached C1/C2/C3:
  * CARD     : retrieve iff ŝ₀ < t
  * cascade  : retrieve iff ŝ₀ < t  OR  pyflakes fires on the zero-shot line
choosing C1 (no-retrieve) or C2 (always-retrieve) per instance, and scoring
A4∧B2 on the chosen output. Thresholds default to the ŝ₀ deciles, so CARD spans
~0–100% retrieval instead of collapsing to full-retrieve at T_RAG=0.9.

Circularity note: trigger and the metric's B2 part are both pyflakes, so the
cascade's drop is partly structural ("of the lines pyflakes flags on zero-shot,
does retrieval fix them?"). A4 (gold) only removes flags that matched the gold.

    python scripts/12_cascade_sweep_hall.py --results-dir results/cce_py_1.5b --dataset crosscodeeval_py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import click  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

from adaptive_retrieval.eval.datasets import DATASET_LOADERS, MULTILINE_DATASETS  # noqa: E402
from adaptive_retrieval.metrics import (  # noqa: E402
    edit_similarity,
    invented_identifier_flag,
    truncate_to_function_body,
)
from adaptive_retrieval.static_analysis.pyflakes_checker import PyflakesChecker  # noqa: E402

_FIM = ("<|", "▁<", "<fim", "<PRE>", "<SUF>", "<MID>", "<EOT>", "</s>", "<｜",
        "<repo_name>", "<file_sep>")


def _clean(raw: str, ref: str, multiline: bool) -> str:
    for m in _FIM:
        i = raw.find(m)
        if i != -1:
            raw = raw[:i]
    return truncate_to_function_body(ref, raw) if multiline else raw.split("\n", 1)[0]


@click.command()
@click.option("--results-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--dataset", required=True, type=click.Choice(list(DATASET_LOADERS)))
@click.option("--t-grid", default="", help="Comma-separated; default = ŝ₀ deciles.")
def main(results_dir: str, dataset: str, t_grid: str) -> None:
    D = Path(results_dir)
    load = lambda n: {r["instance_id"]: r for r in (
        json.loads(l) for l in open(D / n, encoding="utf-8") if l.strip())}
    c1, c2, c3 = load("C1_no_retrieve.jsonl"), load("C2_always_retrieve.jsonl"), load("C3_card.jsonl")
    multiline = dataset in MULTILINE_DATASETS
    insts = {i.instance_id: i for i in DATASET_LOADERS[dataset]()}
    ids = [i for i in sorted(set(c1) & set(c2) & set(c3) & set(insts))
           if c3[i].get("s_hat_0") is not None]
    n = len(ids)
    pf = PyflakesChecker()
    print(f"[setup] dataset={dataset} multiline={multiline}  n={n}")

    shat, es1, es2, h1, h2, trig = {}, {}, {}, {}, {}, {}
    for i in ids:
        inst = insts[i]
        gold = inst.ground_truth
        p1 = _clean(c1[i]["prediction"], gold, multiline)
        p2 = _clean(c2[i]["prediction"], gold, multiline)
        u1 = set(pf.analyze(p1, inst.x_left, inst.x_right).significant_out_of_scope)
        u2 = set(pf.analyze(p2, inst.x_left, inst.x_right).significant_out_of_scope)
        shat[i] = float(c3[i]["s_hat_0"])
        es1[i], es2[i] = edit_similarity(gold, p1), edit_similarity(gold, p2)
        h1[i] = 1 if invented_identifier_flag(gold, p1, u1) else 0
        h2[i] = 1 if invented_identifier_flag(gold, p2, u2) else 0
        trig[i] = bool(u1)  # pyflakes fires on zero-shot

    s = np.array([shat[i] for i in ids])
    print(f"        ŝ₀: min={s.min():.3f} median={np.median(s):.3f} max={s.max():.3f}")
    print(f"        pyflakes trigger fires on zero-shot: {np.mean([trig[i] for i in ids]):.1%}")
    print(f"        A4∧B2 baselines: C1(no-retrieve)={np.mean([h1[i] for i in ids]):.4f}  "
          f"C2(always)={np.mean([h2[i] for i in ids]):.4f}")

    if t_grid.strip():
        ts = [float(x) for x in t_grid.split(",")]
    else:
        ts = [float(np.quantile(s, q)) for q in np.arange(0.1, 1.0, 0.1)]

    print(f"\n{'t':>6} | {'CARD retr':>9} {'ES':>6} {'A4∧B2':>7} | "
          f"{'CASC retr':>9} {'ES':>6} {'A4∧B2':>7} {'Δhall':>8} {'McNemar p':>10}")
    for t in ts:
        cret = [shat[i] < t for i in ids]
        xret = [shat[i] < t or trig[i] for i in ids]
        ch = [h2[i] if r else h1[i] for i, r in zip(ids, cret)]
        xh = [h2[i] if r else h1[i] for i, r in zip(ids, xret)]
        ces = np.mean([es2[i] if r else es1[i] for i, r in zip(ids, cret)])
        xes = np.mean([es2[i] if r else es1[i] for i, r in zip(ids, xret)])
        b = sum(1 for a, x in zip(ch, xh) if a == 0 and x == 1)
        cc = sum(1 for a, x in zip(ch, xh) if a == 1 and x == 0)
        p = binomtest(min(b, cc), b + cc, 0.5).pvalue if (b + cc) else 1.0
        print(f"{t:>6.3f} | {np.mean(cret):>8.1%} {ces:>6.3f} {np.mean(ch):>7.4f} | "
              f"{np.mean(xret):>8.1%} {xes:>6.3f} {np.mean(xh):>7.4f} "
              f"{np.mean(xh)-np.mean(ch):>+8.4f} {p:>10.2g}")


if __name__ == "__main__":
    main()
