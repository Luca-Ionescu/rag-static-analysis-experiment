"""Generate the Colab notebook for the RepoEval-function Qwen-1.5B run.

Produces ``notebooks/repoeval_function_qwen15.ipynb``. Run:

    python scripts/build_colab_notebook.py

Git auth follows the proven pattern from the MasterThesis lightccn notebooks:
the GitHub REST API (Bearer PAT) is used for BOTH pulling the repo (tarball
endpoint) and pushing results (Contents API, per-file) — no ``git clone`` /
``git push`` (which fail in Colab with fine-grained tokens). The PAT is read
from a Colab Secret named ``LUCA_GITHUB_PAT`` (fallback: env var, then getpass).
Results are uploaded incrementally as each config finishes, so a Colab
disconnect never loses completed work.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = "Luca-Ionescu/rag-static-analysis-experiment"


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
    "Adaptive-retrieval cascade on **RepoEval function-body completion** with "
    "**Qwen2.5-Coder-1.5B** (vLLM), using the committed Qwen estimator. "
    "Generation is **FIM** (pipeline unchanged).\n"
    "\n"
    "**Git via REST API (no clone/push):** the repo is pulled with the GitHub "
    "tarball endpoint and results are pushed with the Contents API, **per config "
    "as it finishes** (crash-safe). PAT comes from a Colab Secret `LUCA_GITHUB_PAT`.\n"
    "\n"
    "**Setup:** `Runtime → Change runtime type → GPU`, then add a Colab Secret "
    "named `LUCA_GITHUB_PAT` (🔑 panel, left sidebar) with a token that has "
    "**Contents: Read and write** on this repo.\n"
    "\n"
    "Flip `SMOKE` off to go from a 3-instance check to the full 455."
))

# --- Config ---
CELLS.append(md("## 1. Configuration"))
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
    "# ---- git (REST API) ----\n"
    "REPO = '" + REPO + "'\n"
    "SRC_REF = 'main'                       # ref to pull the code from\n"
    "GH_RESULTS_BRANCH = 'colab-results'    # branch results are pushed to (auto-created)\n"
    "WORK_DIR = '/content/rag-static-analysis-experiment'\n"
    "RESULTS_SUBDIR = f'results/qwen25_1.5b_{DATASET}'\n"
    "GH_RESULTS_PREFIX = RESULTS_SUBDIR     # path inside the results branch\n"
    "print('SMOKE' if SMOKE else 'FULL', '| model', MODEL, '| dataset', DATASET,\n"
    "      '| max_tokens', MAX_TOKENS)"
))

# --- GPU ---
CELLS.append(md("## 2. GPU sanity check"))
CELLS.append(code(
    "import subprocess\n"
    "try:\n"
    "    print(subprocess.check_output(['nvidia-smi'], text=True))\n"
    "except Exception as e:\n"
    "    print('No GPU visible — set Runtime -> Change runtime type -> GPU.', e)"
))

# --- PAT + GitHub REST helpers ---
CELLS.append(md(
    "## 3. GitHub token + REST helpers\n"
    "Reads `LUCA_GITHUB_PAT` from Colab Secrets (🔑). No `git clone`/`push` — all "
    "GitHub I/O goes through the REST API, which works with fine-grained tokens."
))
CELLS.append(code(
    "import os, sys, base64, json as _gjson, urllib.request, urllib.error\n"
    "\n"
    "def _get_pat():\n"
    "    try:\n"
    "        from google.colab import userdata\n"
    "        pat = userdata.get('LUCA_GITHUB_PAT')\n"
    "        if pat: return pat\n"
    "    except Exception as e:\n"
    "        print('  secret read:', e)\n"
    "    pat = os.environ.get('LUCA_GITHUB_PAT')\n"
    "    if pat: return pat\n"
    "    import getpass\n"
    "    return getpass.getpass('LUCA_GITHUB_PAT (Contents: read/write): ').strip()\n"
    "\n"
    "_GH_PAT = _get_pat()\n"
    "assert _GH_PAT, 'no PAT provided'\n"
    "\n"
    "def _gh_req(method, url, data=None, raw=False):\n"
    "    req = urllib.request.Request(url, method=method)\n"
    "    req.add_header('Authorization', f'Bearer {_GH_PAT}')\n"
    "    req.add_header('Accept', 'application/vnd.github.v3.raw' if raw else 'application/vnd.github+json')\n"
    "    req.add_header('User-Agent', 'colab-runner')\n"
    "    req.add_header('X-GitHub-Api-Version', '2022-11-28')\n"
    "    body = None\n"
    "    if data is not None:\n"
    "        body = _gjson.dumps(data).encode()\n"
    "        req.add_header('Content-Type', 'application/json')\n"
    "    with urllib.request.urlopen(req, body, timeout=180) as resp:\n"
    "        out = resp.read()\n"
    "        return out if raw else _gjson.loads(out.decode())\n"
    "\n"
    "# auth smoke test\n"
    "_me = _gh_req('GET', 'https://api.github.com/user')\n"
    "print('authenticated as:', _me.get('login'))"
))

# --- Pull repo via tarball API ---
CELLS.append(md(
    "## 4. Pull the repo (tarball API — no git clone)\n"
    "Downloads the repo at `SRC_REF` as a tarball through the REST API and "
    "extracts it to `WORK_DIR`. This is the auth path that works with "
    "fine-grained PATs where `git clone` returns exit 128."
))
CELLS.append(code(
    "import os, io, tarfile, shutil, urllib.request\n"
    "if os.path.isdir(WORK_DIR):\n"
    "    shutil.rmtree(WORK_DIR)\n"
    "os.makedirs(WORK_DIR, exist_ok=True)\n"
    "\n"
    "url = f'https://api.github.com/repos/{REPO}/tarball/{SRC_REF}'\n"
    "blob = _gh_req('GET', url, raw=True)\n"
    "with tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz') as tar:\n"
    "    members = tar.getmembers()\n"
    "    root = members[0].name.split('/')[0]   # GitHub wraps in <owner>-<repo>-<sha>/\n"
    "    tar.extractall('/content/_repo_extract')\n"
    "for name in os.listdir(f'/content/_repo_extract/{root}'):\n"
    "    shutil.move(f'/content/_repo_extract/{root}/{name}', f'{WORK_DIR}/{name}')\n"
    "shutil.rmtree('/content/_repo_extract')\n"
    "os.chdir(WORK_DIR)\n"
    "print('repo at', os.getcwd())\n"
    "assert os.path.exists(ESTIMATOR), f'missing {ESTIMATOR} — is it committed on {SRC_REF}?'\n"
    "print('estimator present:', ESTIMATOR)"
))

# --- Install deps ---
CELLS.append(md("## 5. Install dependencies\nvLLM + the project stack. Several minutes."))
CELLS.append(code(
    "import subprocess, sys\n"
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
    "## 6. Provision RepoEval-function\n"
    "Clones microsoft/CodeT (sparse, public — plain git is fine here) and unzips "
    "the function-level task JSONLs + repositories into `data/repoeval/`."
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
    "print('function task file:', glob.glob('data/repoeval/datasets/function_level_completion_2k*.jsonl'))\n"
    "print('repos provisioned:', len(os.listdir('data/repoeval/repositories')))"
))

# --- Verify loader ---
CELLS.append(md("## 7. Verify the loader is lossless"))
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

# --- Results-branch setup (REST) ---
CELLS.append(md(
    "## 8. Prepare the results branch (REST)\n"
    "Auto-creates `colab-results` off the default branch if missing, and defines "
    "`gh_upload(path)` to PUT a file into it (incremental, per file)."
))
CELLS.append(code(
    "def _gh_ensure_branch():\n"
    "    refbase = f'https://api.github.com/repos/{REPO}/git/refs/heads/'\n"
    "    try:\n"
    "        _gh_req('GET', refbase + GH_RESULTS_BRANCH); return\n"
    "    except urllib.error.HTTPError as e:\n"
    "        if e.code != 404: raise\n"
    "    info = _gh_req('GET', f'https://api.github.com/repos/{REPO}')\n"
    "    head = _gh_req('GET', refbase + info['default_branch'])\n"
    "    _gh_req('POST', f'https://api.github.com/repos/{REPO}/git/refs',\n"
    "            {'ref': f'refs/heads/{GH_RESULTS_BRANCH}', 'sha': head['object']['sha']})\n"
    "    print('created results branch:', GH_RESULTS_BRANCH)\n"
    "\n"
    "def _gh_get_sha(dest):\n"
    "    url = f'https://api.github.com/repos/{REPO}/contents/{dest}?ref={GH_RESULTS_BRANCH}'\n"
    "    try:\n"
    "        return _gh_req('GET', url).get('sha')\n"
    "    except Exception:\n"
    "        return None\n"
    "\n"
    "def gh_upload(path):\n"
    "    from pathlib import Path as _P\n"
    "    p = _P(path)\n"
    "    if not p.exists():\n"
    "        print('  skip (missing):', path); return\n"
    "    dest = f'{GH_RESULTS_PREFIX}/{p.name}'\n"
    "    payload = {\n"
    "        'message': f'colab results: {p.name} ({\"SMOKE\" if SMOKE else \"FULL\"})',\n"
    "        'content': base64.b64encode(p.read_bytes()).decode(),\n"
    "        'branch': GH_RESULTS_BRANCH,\n"
    "    }\n"
    "    sha = _gh_get_sha(dest)\n"
    "    if sha: payload['sha'] = sha\n"
    "    _gh_req('PUT', f'https://api.github.com/repos/{REPO}/contents/{dest}', payload)\n"
    "    print('  pushed:', dest)\n"
    "\n"
    "_gh_ensure_branch()\n"
    "print('results branch ready:', GH_RESULTS_BRANCH)"
))

# --- Run + push per config ---
CELLS.append(md(
    "## 9. Run C1–C4 (push each as it finishes)\n"
    "Each config writes its JSONL via `04_run_experiment.py`, then is uploaded "
    "immediately — so a disconnect never loses a completed config."
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
    "    gh_upload(out)        # push this config's results immediately\n"
    "\n"
    "for cfg in CONFIGS:\n"
    "    run_config(cfg)\n"
    "print('all configs done + pushed')"
))

# --- Summary + push ---
CELLS.append(md("## 10. Summary (computed + pushed)"))
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
    "spath = f'{RESULTS_SUBDIR}/summary.json'\n"
    "with open(spath, 'w') as f:\n"
    "    json.dump(summary, f, indent=2)\n"
    "gh_upload(spath)\n"
    "print(json.dumps(summary, indent=2))\n"
    "print('\\nResults: https://github.com/' + REPO + '/tree/' + GH_RESULTS_BRANCH + '/' + GH_RESULTS_PREFIX)"
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
