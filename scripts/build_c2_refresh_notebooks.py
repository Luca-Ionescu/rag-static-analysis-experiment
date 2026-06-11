"""Emit three C2-refresh Colab notebooks (one per generator).

Purpose: regenerate ONLY C2_always_retrieve for all four datasets after the
retrieval-corpus fix (BM25Retriever now excludes the completion's own file, so
the ground truth can no longer leak into the retrieved prompt). C1/C3 outputs
and the CARD estimators are unaffected (zero-shot side) and are reused; the
t_rag sweep is recomputed locally from the refreshed C2 + existing C1/C3.

    python scripts/build_c2_refresh_notebooks.py
"""
from __future__ import annotations

import json

from build_experiment_notebooks import (  # noqa: E402  (shared cell templates)
    GENERATORS,
    GH_BRANCH,
    GH_HELPERS,
    HF_LOGIN,
    INSTALL,
    OUT_DIR,
    PROVISION,
    PULL,
    code,
    md,
)
from build_crosscodelongeval_notebooks import PROVISION_CCLE  # noqa: E402

CONFIG_C2 = """# ---- experiment knobs (C2 refresh: leak-free retrieval corpus) ----
SMOKE = False
SMOKE_LIMIT = 32

MODEL = '__MODEL__'
MODEL_FAMILY = '__FAMILY__'
RESULTS_TAG = '__TAG__'

DATASETS = ['crosscodeeval_py', 'repoeval_function',
            'crosscodelongeval_function', 'crosscodelongeval_chunk']
MAX_TOKENS = {'crosscodeeval_py': 50, 'repoeval_function': 280,
              'crosscodelongeval_function': 400, 'crosscodelongeval_chunk': 80}
TOP_K = 10
BATCH_SIZE = 256
GEN_CONFIGS = ['C2_always_retrieve']    # ONLY C2: zero-shot side (C1/C3) is unchanged

# ---- git (REST API; no clone/push) ----
REPO = 'Luca-Ionescu/rag-static-analysis-experiment'
SRC_REF = 'main'
GH_RESULTS_BRANCH = 'colab-results'
WORK_DIR = '/content/rag-static-analysis-experiment'
print('C2-REFRESH | model', MODEL, '| datasets', len(DATASETS))
"""

GENERATE_C2 = """import os, subprocess, sys
os.chdir(WORK_DIR)
# Sanity: the leak fix must be present in the pulled source.
src = open('src/adaptive_retrieval/retriever.py').read()
assert 'exclude_file' in src, 'retriever fix missing on SRC_REF!'
src2 = open('src/adaptive_retrieval/eval/runner.py').read()
assert 'exclude_file=inst.target_file' in src2, 'runner fix missing on SRC_REF!'
print('leak-fix present in source: OK')

def run_config(cfg, ds, subdir):
    out = f'{subdir}/{cfg}.jsonl'
    cmd = [sys.executable, 'scripts/04_run_experiment.py',
           '--config', cfg, '--dataset', ds, '--backend', 'vllm',
           '--model', MODEL, '--model-family', MODEL_FAMILY,
           '--max-tokens', str(MAX_TOKENS[ds]),
           '--top-k', str(TOP_K),
           '--batch-size', str(BATCH_SIZE),
           '--output', out, '--cache-dir', 'data/generation_cache']
    if SMOKE:
        cmd += ['--limit', str(SMOKE_LIMIT)]
    print('>>>', ' '.join(cmd)); subprocess.run(cmd, check=True)
    gh_upload(out, subdir)

for ds in DATASETS:
    subdir = f'results/{RESULTS_TAG}_{ds}'
    os.makedirs(subdir, exist_ok=True)
    for cfg in GEN_CONFIGS:
        run_config(cfg, ds, subdir)
print('C2 refresh done (all datasets pushed)')
"""


def build_c2(gen):
    cells = [
        md(f"# C2 refresh (leak-free retrieval) · {gen['model']}\n\n"
           "Regenerates **only C2 always-retrieve** for all four datasets after the\n"
           "retrieval-corpus fix (the completion's own file is now excluded from the\n"
           "BM25 corpus, so the gold can no longer appear in the retrieved context).\n\n"
           f"GPU: {gen['gpu_note']}\n\n"
           "Setup: `Runtime → A100 high-RAM`; Colab Secret `LUCA_GITHUB_PAT`."),
        md("## 1. Config"),
        code(CONFIG_C2.replace("__MODEL__", gen["model"]).replace("__FAMILY__", gen["family"])
             .replace("__TAG__", gen["tag"])),
        md("## 2. GPU sanity"),
        code("import subprocess\ntry:\n    print(subprocess.check_output(['nvidia-smi'], text=True))\nexcept Exception as e:\n    print('No GPU — Runtime -> Change runtime type -> GPU.', e)"),
        md("## 3. GitHub token + REST helpers"),
        code(GH_HELPERS),
        md("## 4. Pull repo (tarball REST API)"),
        code(PULL.replace("if not CALIBRATE:\n    assert os.path.exists(ESTIMATOR), f'missing {ESTIMATOR} on {SRC_REF}'\n    print('estimator present:', ESTIMATOR)\n", "")),
        md("## 5. Install dependencies"),
        code(INSTALL),
        md("## 6. HF token (public models; secret optional)"),
        code(HF_LOGIN),
        md("## 7. Provision CCE + RepoEval"),
        code(PROVISION),
        md("## 7b. Provision CrossCodeLongEval"),
        code(PROVISION_CCLE),
        md("## 8. Results branch + uploader (REST)"),
        code(GH_BRANCH),
        md("## 9. Generate C2 per dataset (push each)"),
        code(GENERATE_C2),
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
        nb = build_c2(gen)
        path = OUT_DIR / f"c2refresh_{gen['tag']}.ipynb"
        path.write_text(json.dumps(nb, indent=1))
        print("wrote", path, f"({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
