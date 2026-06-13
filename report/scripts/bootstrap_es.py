#!/usr/bin/env python
"""Paired bootstrap 95% CIs on edit-similarity (ES) differences, computed from
the SAVED per-instance predictions only -- no model re-run, no GPU.

  RQ1: dES(C2 - C1) per (model x dataset)  -- is the retrieval gain significant?
  RQ4: dES(C4 - C3) at T_RAG=0.05          -- do the cascade's extra retrievals
                                              hurt accuracy (non-inferiority)?

ES is recomputed from predictions with the SAME truncation 13_sweep_eval uses
(_clean), so these numbers match the reported tables exactly. Run from repo root:
    PYTHONPATH=src python report/scripts/bootstrap_es.py
"""
import sys, json, csv
import numpy as np
from importlib import import_module
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
_clean = import_module("13_sweep_eval")._clean
from adaptive_retrieval.metrics import edit_similarity
from adaptive_retrieval.eval.datasets import DATASET_LOADERS
from adaptive_retrieval.static_analysis.pyflakes_checker import PyflakesChecker
from scipy.stats import wilcoxon

# FIM-strip the raw generation exactly as 13_sweep_eval / mcnemar_check do, so the
# static trigger is computed on the same text the sweep uses.
_FIM = ("<|", "▁<", "<fim", "<PRE>", "<SUF>", "<MID>", "<EOT>", "</s>",
        "<｜", "<repo_name>", "<file_sep>", "<|endoftext|>")
def _strip(t):
    for m in _FIM:
        i = t.find(m)
        if i != -1:
            t = t[:i]
    return t

RES = "data/_resweep"
MODELS = [("qwen25_0.5b", "0.5B"), ("qwen25_1.5b", "1.5B"), ("codellama_7b", "7B")]
DS = [("crosscodeeval_py", "CCE-line", "line"),
      ("repoeval_function", "RepoEval-fn", "body"),
      ("crosscodelongeval_function", "CCLE-fn", "body"),
      ("crosscodelongeval_chunk", "CCLE-chunk", "lines")]
T, B = 0.05, 10000
rng = np.random.default_rng(42)

def load(tag, ds, cfg):
    return {json.loads(l)["instance_id"]: json.loads(l)
            for l in open(f"{RES}/{tag}_{ds}/{cfg}.jsonl")}

def es(rec, mode):
    g = rec["ground_truth"]
    return edit_similarity(g, _clean(rec["prediction"], g, mode))

def boot_ci(deltas):
    d = np.asarray(deltas, float)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    means = d[idx].mean(axis=1)
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)

rows = []
print("=== RQ1: dES(C2-C1), 95%% bootstrap CI over %d resamples (CI>0 => retrieval helps) ===" % B)
for tag, ml in MODELS:
    for ds, dl, mode in DS:
        c1, c2 = load(tag, ds, "C1_no_retrieve"), load(tag, ds, "C2_always_retrieve")
        ids = sorted(set(c1) & set(c2))
        e1 = [es(c1[i], mode) for i in ids]; e2 = [es(c2[i], mode) for i in ids]
        d = [b - a for a, b in zip(e1, e2)]
        m, lo, hi = boot_ci(d)
        w = wilcoxon(e2, e1, zero_method="zsplit").pvalue
        sig = "YES" if lo > 0 else ("NEG" if hi < 0 else "ns")
        print(f"  {ml:4s} {dl:11s} n={len(ids):5d}  dES={m:+.3f}  CI[{lo:+.3f},{hi:+.3f}]  wilcoxon p={w:.1e}  {sig}")
        rows.append((dl, ml, "C2-C1", len(ids), round(m, 4), round(lo, 4), round(hi, 4), w))

print("\n=== RQ4: dES(C4-C3) at T=0.05, 95%% CI (CI excludes negatives => extra retrievals do not hurt ES) ===")
pf = PyflakesChecker()
for ds, dl, mode in DS:
    try:
        insts = {i.instance_id: i for i in DATASET_LOADERS[ds]()}  # x_left/x_right shared across models
    except FileNotFoundError:
        print(f"  -- {dl}: raw benchmark context not on disk; skipping (deterministic asymmetry still holds, Sec. RQ4)")
        continue
    for tag, ml in MODELS:
        c1, c2, c3 = load(tag, ds, "C1_no_retrieve"), load(tag, ds, "C2_always_retrieve"), load(tag, ds, "C3_card")
        ids = [i for i in sorted(set(c1) & set(c2) & set(c3) & set(insts)) if c3[i].get("s_hat_0") is not None]
        d, n_extra = [], 0
        for i in ids:
            inst = insts[i]
            shat = float(c3[i]["s_hat_0"])
            p1 = _strip(c1[i]["prediction"])
            trig = bool(pf.analyze(p1, inst.x_left, inst.x_right).significant_out_of_scope)
            e1, e2 = es(c1[i], mode), es(c2[i], mode)
            c3v = e2 if shat < T else e1
            c4v = e2 if (shat < T or trig) else e1
            if c4v != c3v:
                n_extra += 1
            d.append(c4v - c3v)
        m, lo, hi = boot_ci(d)
        print(f"  {ml:4s} {dl:11s} n={len(ids):5d} extra-retr={n_extra:4d}  dES={m:+.4f}  CI[{lo:+.4f},{hi:+.4f}]")
        rows.append((dl, ml, "C4-C3@.05", len(ids), round(m, 5), round(lo, 5), round(hi, 5), ""))

with open("/tmp/report_work/bootstrap_es_results.csv", "w", newline="") as f:
    wtr = csv.writer(f); wtr.writerow(["dataset", "model", "comparison", "n", "mean_dES", "ci_lo", "ci_hi", "wilcoxon_p"])
    wtr.writerows(rows)
print("\nsaved /tmp/report_work/bootstrap_es_results.csv")
