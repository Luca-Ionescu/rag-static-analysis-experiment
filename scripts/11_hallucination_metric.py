"""Dataset-aware hallucination metric: A4 ∧ B2 (invented-identifier rate).

  A4 (reference): an emitted identifier absent from the gold completion.
  B2 (static, pyflakes): an emitted identifier pyflakes flags as 'undefined
      name' (F821) in the reconstructed file (x_left + prediction + x_right).
  A4 ∧ B2: an identifier that is BOTH — the model invented a name that is wrong
      AND resolves nowhere.

The metric is computed on the FINAL prediction of each config (after any
retrieval), so it measures hallucinations that survived to evaluation: if the
static gate triggered retrieval on the zero-shot output, a genuine
undefined-name should be fixed in the retrieved output and no longer flagged.

Dataset-aware (via datasets.MULTILINE_DATASETS):
  * line-completion (crosscodeeval_py) -> truncate prediction to the first line;
  * multi-line bodies (repoeval_function) -> keep the body
    (metrics.truncate_to_function_body).
B2 uses pyflakes, NOT the project's static analyzer, so the metric is
independent of the cascade's own resolver — except that if pyflakes is also the
retrieval trigger, only A4 is fully trigger-independent (see project note §1.2).

    python scripts/11_hallucination_metric.py \\
        --results-dir results/cce_py_1.5b --dataset crosscodeeval_py
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
    exact_match,
    hallucinated_identifier_flag,
    identifier_f1,
    invented_identifier_flag,
    truncate_to_function_body,
)
from adaptive_retrieval.static_analysis.pyflakes_checker import PyflakesChecker  # noqa: E402

# Sentinel/FIM markers to cut off before scoring (old runs lacked a stop string).
_FIM_MARKERS = (
    "<|", "▁<", "<fim", "<PRE>", "<SUF>", "<MID>", "<EOT>", "</s>", "<｜",
    "<repo_name>", "<file_sep>", "<|endoftext|>",
)
_CONFIGS = ("C1_no_retrieve", "C2_always_retrieve", "C3_card", "C4_cascade")


def _strip_sentinels(text: str) -> str:
    for m in _FIM_MARKERS:
        i = text.find(m)
        if i != -1:
            text = text[:i]
    return text


def _clean(raw: str, reference: str, multiline: bool) -> str:
    """Dataset-aware: strip FIM tail, then truncate to the unit being scored."""
    pred = _strip_sentinels(raw)
    if multiline:
        return truncate_to_function_body(reference, pred)
    return pred.split("\n", 1)[0]


@click.command()
@click.option("--results-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--dataset", required=True, type=click.Choice(list(DATASET_LOADERS)))
@click.option("--configs", default=",".join(_CONFIGS))
def main(results_dir: str, dataset: str, configs: str) -> None:
    D = Path(results_dir)
    multiline = dataset in MULTILINE_DATASETS
    print(f"[setup] dataset={dataset}  multiline={multiline}  results-dir={D}")
    insts = {i.instance_id: i for i in DATASET_LOADERS[dataset]()}
    print(f"        loaded {len(insts)} instances")
    pf = PyflakesChecker()

    inv_flags: dict[str, dict[str, int]] = {}   # config -> {iid: A4∧B2 flag}
    print(f"\n{'config':<20}{'n':>6}{'EM':>7}{'ES':>7}{'idF1':>7}{'A4':>7}{'B2':>7}{'A4∧B2':>8}")
    for cfg in configs.split(","):
        path = D / f"{cfg}.jsonl"
        if not path.exists():
            print(f"{cfg:<20}   (file not found — skipped)")
            continue
        recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        n = 0
        em = es = f1 = 0.0
        a4 = b2 = inv = 0
        per: dict[str, int] = {}
        for r in recs:
            inst = insts.get(r["instance_id"])
            if inst is None:
                continue
            n += 1
            gold = inst.ground_truth
            pred = _clean(r["prediction"], gold, multiline)
            em += exact_match(gold, pred)
            es += edit_similarity(gold, pred)
            f1 += identifier_f1(gold, pred)
            undefined = set(pf.analyze(pred, inst.x_left, inst.x_right).significant_out_of_scope)
            a4 += hallucinated_identifier_flag(gold, pred)
            b2 += 1 if undefined else 0
            flag = 1 if invented_identifier_flag(gold, pred, undefined) else 0
            inv += flag
            per[r["instance_id"]] = flag
        inv_flags[cfg] = per
        if n:
            print(f"{cfg:<20}{n:>6}{em/n:>7.3f}{es/n:>7.3f}{f1/n:>7.3f}"
                  f"{a4/n:>7.3f}{b2/n:>7.3f}{inv/n:>8.4f}")
    print(f"\n[note] A4∧B2 is the headline hallucination rate. "
          f"pyflakes parse-failures (B2 → 0 there): {pf.parse_failures}")

    if "C3_card" in inv_flags and "C4_cascade" in inv_flags:
        ids = sorted(set(inv_flags["C3_card"]) & set(inv_flags["C4_cascade"]))
        a = [inv_flags["C3_card"][i] for i in ids]
        b = [inv_flags["C4_cascade"][i] for i in ids]
        bb = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
        cc = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
        p = binomtest(min(bb, cc), bb + cc, 0.5).pvalue if (bb + cc) else 1.0
        print(f"\n=== RQ2 (A4∧B2): CARD (C3) vs cascade (C4) on {len(ids)} shared ===")
        print(f"  hall  CARD={np.mean(a):.4f}  cascade={np.mean(b):.4f}  "
              f"Δ={np.mean(b)-np.mean(a):+.4f}  McNemar p={p:.2g}  (cascade better={cc}, worse={bb})")


if __name__ == "__main__":
    main()
