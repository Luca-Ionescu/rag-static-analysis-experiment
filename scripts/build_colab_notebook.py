"""Generate the Colab notebook for the RepoEval-function Qwen-1.5B run.

Produces ``notebooks/repoeval_function_qwen15.ipynb``. Run:

    python scripts/build_colab_notebook.py

The notebook itself: clones the repo, installs deps, provisions
RepoEval-function (microsoft/CodeT), runs C1-C4 with Qwen2.5-Coder-1.5B via
vLLM using the committed Qwen estimator, then commits+pushes results to a
branch. A SMOKE flag (LIMIT=3) gates a quick end-to-end check before the
full 455-instance run.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_HTTPS = "github.com/Luca-Ionescu/rag-static-analysis-experiment.git"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


CELLS: list[dict] = []

CELLS.append(md(
    "# RepoEval-function · Qwen2.5-Coder-1.5B · C1–C4\n"
    "\n"
    "Runs the adaptive-retrieval cascade on **RepoEval function-body completion** "
    "with **Qwen2.5-Coder-1.5B** (vLLM), using the committed Qwen estimator. "
    "Generation is **FIM** (unchanged pipeline). Results are committed and pushed "
    "back to a results branch.\n"
    "\n"
    "**Runtime:** GPU (T4/L4/A100). Set `Runtime → Change runtime type → GPU` first.\n"
    "\n"
    "**Flow:** clone → install → provision RepoEval → run C1–C4 → push results.\n"
    "\n"
    "Flip `SMOKE` off to go from a 3-instance check to the full 455."
))

# --- Config cell ---
CELLS.append(md("## 1. Configuration — edit these"))
CELLS.append(code(
    "# ---- experiment knobs ----\n"
    "SMOKE = True          # True: 3 instances (quick check). False: all 455.\n"
    "SMOKE_LIMIT = 3\n"
    "\n"
    "MODEL = 'Qwen/Qwen2.5-Coder-1.5B'\n"
    "MODEL_FAMILY = 'qwen'\n"
    "DATASET = 'repoeval_function'\n"
    "MAX_TOKENS = 512       # function bodies are multi-line; 512 avoids truncation\n"
    "T_RAG = 0.9\n"
    "TOP_K = 10\n"
    "ESTIMATOR = 'models/estimator_qwen25_1.5b.lgb'   # committed Qwen estimator\n"
    "CONFIGS = ['C1_no_retrieve', 'C2_always_retrieve', 'C3_card', 'C4_cascade']\n"
    "\n"
    "# ---- git push (results -> branch) ----\n"
    "GIT_BRANCH = 'results/repoeval-qwen15'   # branch the results get pushed to\n"
    "GIT_USER_NAME = 'colab-runner'\n"
    "GIT_USER_EMAIL = 'colab@example.com'\n"
    "# Paste a GitHub Personal Access Token (repo scope) when prompted below.\n"
    "# It is read via getpass so it is not stored in the notebook.\n"
    "BRANCH_FROM = 'main'\n"
    "\n"
    "REPO_HTTPS = '" + REPO_HTTPS + "'\n"
    "WORK_DIR = '/content/rag-static-analysis-experiment'\n"
    "RESULTS_SUBDIR = f'results/qwen25_1.5b_{DATASET}'\n"
    "print('SMOKE' if SMOKE else 'FULL', '| model', MODEL, '| dataset', DATASET,\n"
    "      '| max_tokens', MAX_TOKENS)"
))

# --- GPU check ---
CELLS.append(md("## 2. GPU sanity check"))
CELLS.append(code(
    "import subprocess\n"
    "try:\n"
    "    print(subprocess.check_output(['nvidia-smi'], text=True))\n"
    "except Exception as e:\n"
    "    print('No GPU visible — set Runtime -> Change runtime type -> GPU.', e)"
))

# --- Token + clone ---
CELLS.append(md("## 3. Token & clone\nPaste a GitHub PAT (repo scope). Used only to clone+push over HTTPS."))
CELLS.append(code(
    "import os, getpass, subprocess\n"
    "GH_TOKEN = getpass.getpass('GitHub PAT (repo scope): ').strip()\n"
    "auth_url = f'https://{GH_TOKEN}@{REPO_HTTPS}'\n"
    "if os.path.isdir(WORK_DIR):\n"
    "    subprocess.run(['rm', '-rf', WORK_DIR], check=True)\n"
    "subprocess.run(['git', 'clone', auth_url, WORK_DIR], check=True)\n"
    "os.chdir(WORK_DIR)\n"
    "subprocess.run(['git', 'checkout', BRANCH_FROM], check=True)\n"
    "subprocess.run(['git', 'pull', '--ff-only'], check=True)\n"
    "print('cloned at', os.getcwd())\n"
    "print(subprocess.check_output(['git', 'log', '-1', '--oneline'], text=True))"
))

# --- Install deps ---
CELLS.append(md("## 4. Install dependencies\nvLLM + the project requirements. Takes several minutes."))
CELLS.append(code(
    "import subprocess, sys\n"
    "# vLLM (brings a compatible torch) + the static/retrieval/metrics stack.\n"
    "pkgs = [\n"
    "    'vllm==0.10.2',\n"
    "    'transformers>=4.55.2,<5.0',\n"
    "    'tree-sitter==0.23.2', 'tree-sitter-python==0.23.6',\n"
    "    'rank-bm25', 'python-Levenshtein', 'lightgbm',\n"
    "    'jsonlines', 'click', 'pyflakes', 'tqdm', 'scipy', 'scikit-learn',\n"
    "]\n"
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *pkgs], check=True)\n"
    "print('deps installed')"
))

# --- Provision RepoEval ---
CELLS.append(md(
    "## 5. Provision RepoEval-function\n"
    "Clones microsoft/CodeT (sparse) and unzips the function-level task JSONLs + "
    "repositories into `data/repoeval/` where `load_repoeval` expects them."
))
CELLS.append(code(
    "import os, subprocess, zipfile, glob\n"
    "os.chdir(WORK_DIR)\n"
    "os.makedirs('data/repoeval/datasets', exist_ok=True)\n"
    "os.makedirs('data/repoeval/repositories', exist_ok=True)\n"
    "\n"
    "if not os.path.exists('/content/CodeT'):\n"
    "    subprocess.run(['git','clone','--depth','1','--filter=blob:none','--sparse',\n"
    "                    'https://github.com/microsoft/CodeT.git','/content/CodeT'], check=True)\n"
    "    subprocess.run(['git','-C','/content/CodeT','sparse-checkout','set','RepoCoder'], check=True)\n"
    "\n"
    "RC = '/content/CodeT/RepoCoder'\n"
    "with zipfile.ZipFile(f'{RC}/datasets/datasets.zip') as z:\n"
    "    z.extractall('data/repoeval/datasets')\n"
    "with zipfile.ZipFile(f'{RC}/repositories/function_level.zip') as z:\n"
    "    z.extractall('data/repoeval/repositories')\n"
    "\n"
    "ds = glob.glob('data/repoeval/datasets/function_level_completion_2k*.jsonl')\n"
    "print('function-level task file:', ds)\n"
    "print('repos provisioned:', len(os.listdir('data/repoeval/repositories')))"
))

# --- Verify loader ---
CELLS.append(md("## 6. Verify the loader is lossless\nConfirms the RepoEval fix: `x_left + ground_truth + x_right == file` for every instance."))
CELLS.append(code(
    "import sys, os\n"
    "os.chdir(WORK_DIR)\n"
    "sys.path.insert(0, 'src')\n"
    "from pathlib import Path\n"
    "from adaptive_retrieval.eval.datasets import load_repoeval\n"
    "n=bad=xr=0\n"
    "for inst in load_repoeval(task='function'):\n"
    "    n += 1\n"
    "    full = (Path('data/repoeval/repositories')/Path(*inst.target_file.split('/'))).read_text(errors='replace')\n"
    "    if inst.x_left + inst.ground_truth + inst.x_right != full: bad += 1\n"
    "    if inst.x_right.strip(): xr += 1\n"
    "print(f'instances={n}  reconstruction_failures={bad}  with_right_context={xr}')\n"
    "assert bad == 0, 'loader not lossless!'\n"
    "print('OK — loader lossless')"
))

# --- Run configs ---
CELLS.append(md(
    "## 7. Run C1–C4\n"
    "Each config writes a per-instance JSONL via the existing "
    "`scripts/04_run_experiment.py`. C3/C4 use the Qwen estimator. "
    "A shared generation cache makes the zero-shot pass reusable across configs."
))
CELLS.append(code(
    "import os, subprocess, sys\n"
    "os.chdir(WORK_DIR)\n"
    "os.makedirs(RESULTS_SUBDIR, exist_ok=True)\n"
    "os.makedirs('data/generation_cache', exist_ok=True)\n"
    "\n"
    "def run_config(cfg):\n"
    "    out = f'{RESULTS_SUBDIR}/{cfg}.jsonl'\n"
    "    cmd = [sys.executable, 'scripts/04_run_experiment.py',\n"
    "           '--config', cfg, '--dataset', DATASET,\n"
    "           '--backend', 'vllm', '--model', MODEL,\n"
    "           '--model-family', MODEL_FAMILY,\n"
    "           '--max-tokens', str(MAX_TOKENS),\n"
    "           '--t-rag', str(T_RAG), '--top-k', str(TOP_K),\n"
    "           '--output', out, '--cache-dir', 'data/generation_cache']\n"
    "    if cfg in ('C3_card', 'C4_cascade'):\n"
    "        cmd += ['--estimator-path', ESTIMATOR]\n"
    "    if SMOKE:\n"
    "        cmd += ['--limit', str(SMOKE_LIMIT)]\n"
    "    print('>>>', ' '.join(cmd))\n"
    "    subprocess.run(cmd, check=True)\n"
    "\n"
    "for cfg in CONFIGS:\n"
    "    run_config(cfg)\n"
    "print('all configs done')"
))

# --- Summarize ---
CELLS.append(md("## 8. Quick summary\nAggregates each config's JSONL (EM/ES/IdF1/hallucination/retrieval%)."))
CELLS.append(code(
    "import os, json, sys\n"
    "os.chdir(WORK_DIR)\n"
    "sys.path.insert(0, 'src')\n"
    "from adaptive_retrieval.eval.runner import aggregate_from_jsonl\n"
    "rows = []\n"
    "for cfg in CONFIGS:\n"
    "    path = f'{RESULTS_SUBDIR}/{cfg}.jsonl'\n"
    "    if not os.path.exists(path):\n"
    "        continue\n"
    "    s = aggregate_from_jsonl(path)\n"
    "    rows.append((cfg, s.n_instances, round(s.percent_retrieval,1), s.metrics))\n"
    "summary = {\n"
    "    'model': MODEL, 'dataset': DATASET, 'smoke': SMOKE,\n"
    "    'max_tokens': MAX_TOKENS, 't_rag': T_RAG,\n"
    "    'configs': {r[0]: {'n': r[1], 'retrieval_pct': r[2], **r[3]} for r in rows},\n"
    "}\n"
    "with open(f'{RESULTS_SUBDIR}/summary.json', 'w') as f:\n"
    "    json.dump(summary, f, indent=2)\n"
    "print(json.dumps(summary, indent=2))"
))

# --- Push results ---
CELLS.append(md(
    "## 9. Commit & push results\n"
    "`results/` is gitignored, so we **force-add** the run's JSONLs + summary and "
    "push to the results branch. The SMOKE run is tagged in the commit message."
))
CELLS.append(code(
    "import os, subprocess\n"
    "os.chdir(WORK_DIR)\n"
    "subprocess.run(['git','config','user.name', GIT_USER_NAME], check=True)\n"
    "subprocess.run(['git','config','user.email', GIT_USER_EMAIL], check=True)\n"
    "# fresh results branch off main\n"
    "subprocess.run(['git','checkout','-B', GIT_BRANCH], check=True)\n"
    "# force-add: results/ is gitignored by design\n"
    "subprocess.run(['git','add','-f', RESULTS_SUBDIR], check=True)\n"
    "tag = 'SMOKE' if SMOKE else 'FULL'\n"
    "msg = f'RepoEval-function Qwen-1.5B results ({tag}, max_tokens={MAX_TOKENS})'\n"
    "rc = subprocess.run(['git','commit','-m', msg])\n"
    "if rc.returncode == 0:\n"
    "    subprocess.run(['git','push','-u','origin', GIT_BRANCH, '--force'], check=True)\n"
    "    print('pushed to', GIT_BRANCH)\n"
    "else:\n"
    "    print('nothing to commit (no new results?)')"
))

NB = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "notebooks" / "repoeval_function_qwen15.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(NB, indent=1))
    print(f"wrote {out}")
