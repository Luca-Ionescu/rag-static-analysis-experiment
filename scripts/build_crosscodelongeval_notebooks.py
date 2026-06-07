"""Emit three CrossCodeLongEval Colab notebooks (one per generator).

Sibling of ``build_experiment_notebooks.py`` — it reuses that module's shared
cell templates (GitHub REST helpers, repo pull, install, calibration, sweep,
summary, …) and only overrides three things:

  * the dataset list  -> the two new CrossCodeLongEval tasks, **function first,
    then chunk** (function is the multi-line body we care about most);
  * the data-provisioning cell -> downloads the two CrossCodeLongEval tarballs
    from the Repoformer repo (sparse checkout) and extracts them where the
    loader expects (``data/crosscodelongeval/cceval_{function,chunk}_eval_data/``);
  * per-dataset ``MAX_TOKENS`` -> function 400, chunk 80 (p99 gold length).

SMOKE is left ON (SMOKE_LIMIT=3) so each notebook does a tiny end-to-end dry run
first; flip ``SMOKE = False`` in the Config cell for the full 5000+5000 sweep.

    python scripts/build_crosscodelongeval_notebooks.py
"""
from __future__ import annotations

import json

import build_experiment_notebooks as base  # noqa: E402  (shared cell templates)
from build_experiment_notebooks import (  # noqa: E402
    CALIBRATE_CELL,
    CLEAR_CACHE,
    GENERATE,
    GENERATORS,
    GH_BRANCH,
    GH_HELPERS,
    HF_LOGIN,
    INSTALL,
    OUT_DIR,
    PULL,
    SUMMARY,
    SWEEP,
    VERIFY,
    code,
    md,
)

# Config cell: function FIRST, then chunk. Only DATASETS + MAX_TOKENS differ from
# the base CONFIG; everything else (git, sweep grid, configs) is identical.
CONFIG_CCLE = """# ---- experiment knobs ----
SMOKE = True            # True: tiny end-to-end check. Flip OFF for the full run.
SMOKE_LIMIT = 3

MODEL = '__MODEL__'
MODEL_FAMILY = '__FAMILY__'
ESTIMATOR = '__ESTIMATOR__'
CALIBRATE = __CALIBRATE__          # 0.5B: calibrate a fresh estimator on the-stack-dedup
RESULTS_TAG = '__TAG__'

# Function first (multi-line body, the interesting case), then chunk (short block).
DATASETS = ['crosscodelongeval_function', 'crosscodelongeval_chunk']
# Per-dataset generation budget = p99 of gold length in tokens, via the project's
# validated char->token ratio (RepoEval-function: 280 tok == 968-char p99):
# function p99 1378 chars -> ~399 -> 400; chunk p99 227 chars -> ~66 -> 80.
# Over-budget is harmless (stop tokens + body truncation), so we round up.
MAX_TOKENS = {'crosscodelongeval_function': 400, 'crosscodelongeval_chunk': 80}
GEN_T_RAG = 0.5         # only used so C3 stores s_hat_0; the real sweep is post-hoc
TOP_K = 10
T_GRID = ','.join(f'{t:.2f}' for t in [0.05*i for i in range(1, 20)])  # 0.05..0.95
GEN_CONFIGS = ['C1_no_retrieve', 'C2_always_retrieve', 'C3_card']      # C4 + sweep are post-hoc

# ---- git (REST API; no clone/push) ----
REPO = 'Luca-Ionescu/rag-static-analysis-experiment'
SRC_REF = 'main'
GH_RESULTS_BRANCH = 'colab-results'
WORK_DIR = '/content/rag-static-analysis-experiment'
print('SMOKE' if SMOKE else 'FULL', '| model', MODEL, '| calibrate', CALIBRATE)
"""

# CrossCodeLongEval ships as two tarballs committed in the Repoformer repo. We
# sparse-checkout just the crosscodelongeval/ folder and extract both tarballs
# into data/crosscodelongeval/ — the layout load_crosscodelongeval() expects.
PROVISION_CCLE = """import os, subprocess, tarfile, glob
os.chdir(WORK_DIR)
os.makedirs('data/crosscodelongeval', exist_ok=True)
if not os.path.exists('/content/Repoformer'):
    subprocess.run(['git','clone','--depth','1','--filter=blob:none','--sparse',
                    'https://github.com/amazon-science/Repoformer.git','/content/Repoformer'], check=True)
    subprocess.run(['git','-C','/content/Repoformer','sparse-checkout','set','crosscodelongeval'], check=True)
RF = '/content/Repoformer/crosscodelongeval'
for tb in ('cceval_function_eval_data.tar.gz', 'cceval_chunk_eval_data.tar.gz'):
    with tarfile.open(f'{RF}/{tb}') as t: t.extractall('data/crosscodelongeval')
fn = glob.glob('data/crosscodelongeval/cceval_function_eval_data/*sparse_rg1*.jsonl')
ch = glob.glob('data/crosscodelongeval/cceval_chunk_eval_data/*sparse_rg1*.jsonl')
print('CCLE function:', fn, '->', sum(1 for _ in open(fn[0])) if fn else 0, 'instances')
print('CCLE chunk   :', ch, '->', sum(1 for _ in open(ch[0])) if ch else 0, 'instances')
"""


def build_ccle(gen):
    cells = [
        md(f"# Cascade experiment · {gen['model']} · C1–C4 × CrossCodeLongEval "
           "{function, chunk}\n\n"
           "Function (multi-line body, scored **body** and **full**) runs first, then "
           "chunk (short block, scored as-is). Full `t_rag` sweep (0.05–0.95), five metrics.\n\n"
           f"GPU: {gen['gpu_note']}\n\n"
           "Setup: `Runtime → GPU`; Colab Secret `LUCA_GITHUB_PAT` (Contents: read/write)"
           + ("; `HF_TOKEN` (the-stack-dedup license accepted)." if gen['calibrate'] else ".")
           + "\n\n**SMOKE is ON** (3 instances/config) for a fast dry run — flip "
           "`SMOKE = False` in the Config cell for the full run."),
        md("## 1. Config"),
        code(CONFIG_CCLE.replace("__MODEL__", gen["model"]).replace("__FAMILY__", gen["family"])
             .replace("__ESTIMATOR__", gen["estimator"]).replace("__CALIBRATE__", str(gen["calibrate"]))
             .replace("__TAG__", gen["tag"])),
        md("## 2. GPU sanity"),
        code("import subprocess\ntry:\n    print(subprocess.check_output(['nvidia-smi'], text=True))\nexcept Exception as e:\n    print('No GPU — Runtime -> Change runtime type -> GPU.', e)"),
        md("## 3. GitHub token + REST helpers"),
        code(GH_HELPERS),
        md("## 4. Pull repo (tarball REST API)"),
        code(PULL),
        md("## 5. Install dependencies (several minutes)"),
        code(INSTALL),
        md("## 6. HF token (for gated the-stack-dedup; 0.5B calibration)"),
        code(HF_LOGIN),
        md("## 7. Provision CrossCodeLongEval data (Repoformer tarballs)"),
        code(PROVISION_CCLE),
        md("## 8. Results branch + uploader (REST)"),
        code(GH_BRANCH),
    ]
    if gen["calibrate"]:
        cells += [md("## 8b. Calibrate the 0.5B estimator\n"
                     "FULL: the-stack-dedup (gated → needs `HF_TOKEN`), ~30 min. "
                     "SMOKE: tiny public `the-stack-smol` sample, relaxed guards, a few min "
                     "(rough estimator → isolated `_smoke` paths, not pushed). The estimator is "
                     "per-**model**, not per-dataset, so it is identical to the one used for the "
                     "CCE/RepoEval runs."),
                  code(CALIBRATE_CELL)]
    cells += [
        md("## 9. Verify dataset loaders"),
        code(VERIFY),
        md("## 10. Clear the generation cache (stop-strings changed → avoid stale cache)"),
        code(CLEAR_CACHE),
        md("## 11. Generate C1/C2/C3 per dataset (function first, push each)\n"
           "C4 + the full t_rag sweep are produced post-hoc in step 12 — C3 here is only "
           "for its stored ŝ₀ (its zero-shot prompt is a cache hit on C1)."),
        code(GENERATE),
        md("## 12. Sweep t_rag (0.05–0.95) + five metrics → sweep.csv\n"
           "`13_sweep_eval.py` replays CARD (C3) and cascade (C4) at every t_rag, computes "
           "EM/ES/idF1/latency and hallucination (A4∧B2 + A4-only). Function is scored **body** "
           "and **full**; chunk is scored as-is (no body truncation)."),
        code(SWEEP),
        md("## 13. Summary"),
        code(SUMMARY),
    ]
    return {
        "cells": cells,
        "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


def main():
    for gen in GENERATORS:
        nb = build_ccle(gen)
        path = OUT_DIR / f"crosscodelongeval_{gen['tag']}.ipynb"
        path.write_text(json.dumps(nb, indent=1))
        print("wrote", path, f"({len(nb['cells'])} cells)")


if __name__ == "__main__":
    # base import kept explicit so static checkers don't flag the module as unused
    assert base.OUT_DIR == OUT_DIR
    main()
