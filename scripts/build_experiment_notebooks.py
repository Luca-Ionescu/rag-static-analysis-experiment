"""Emit the three experiment Colab notebooks (one per generator) from a shared
cell template + per-generator config. Modeled on notebooks/repoeval_function_qwen15.ipynb.

    python scripts/build_experiment_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "notebooks"

# Per-generator config (only this differs between the three notebooks).
GENERATORS = [
    dict(tag="codellama_7b", model="codellama/CodeLlama-7b-hf", family="codellama",
         estimator="models/estimator_codellama_7b.lgb", calibrate=False,
         gpu_note="Needs A100 or L4 (Pro+). T4 will OOM on bf16 7B."),
    dict(tag="qwen25_1.5b", model="Qwen/Qwen2.5-Coder-1.5B", family="qwen",
         estimator="models/estimator_qwen25_1.5b.lgb", calibrate=False,
         gpu_note="Fits T4/L4."),
    dict(tag="qwen25_0.5b", model="Qwen/Qwen2.5-Coder-0.5B", family="qwen",
         estimator="models/estimator_qwen25_0.5b.lgb", calibrate=True,
         gpu_note="Fits T4/L4. Calibrates its estimator first (the-stack-dedup, gated)."),
]


def md(src):  # markdown cell
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):  # code cell
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


CONFIG = """# ---- experiment knobs ----
SMOKE = True            # True: tiny check. Flip OFF for the full run.
SMOKE_LIMIT = 3

MODEL = '__MODEL__'
MODEL_FAMILY = '__FAMILY__'
ESTIMATOR = '__ESTIMATOR__'
CALIBRATE = __CALIBRATE__          # 0.5B: calibrate a fresh estimator on the-stack-dedup
RESULTS_TAG = '__TAG__'

# CrossCodeLongEval lives in its own notebooks (build_crosscodelongeval_notebooks.py),
# which carry the extra tarball-provisioning cell; keep this builder on CCE+RepoEval.
DATASETS = ['crosscodeeval_py', 'repoeval_function']
MAX_TOKENS = {'crosscodeeval_py': 50, 'repoeval_function': 280}
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

GH_HELPERS = """import os, sys, base64, json as _gjson, urllib.request, urllib.error

def _get_pat():
    try:
        from google.colab import userdata
        pat = userdata.get('LUCA_GITHUB_PAT')
        if pat: return pat
    except Exception as e:
        print('  secret read:', e)
    pat = os.environ.get('LUCA_GITHUB_PAT')
    if pat: return pat
    import getpass
    return getpass.getpass('LUCA_GITHUB_PAT (Contents: read/write): ').strip()

_GH_PAT = _get_pat(); assert _GH_PAT, 'no PAT provided'

def _gh_req(method, url, data=None, raw=False):
    req = urllib.request.Request(url, method=method)
    req.add_header('Authorization', f'Bearer {_GH_PAT}')
    req.add_header('Accept', 'application/vnd.github.v3.raw' if raw else 'application/vnd.github+json')
    req.add_header('User-Agent', 'colab-runner')
    req.add_header('X-GitHub-Api-Version', '2022-11-28')
    body = None
    if data is not None:
        body = _gjson.dumps(data).encode(); req.add_header('Content-Type', 'application/json')
    with urllib.request.urlopen(req, body, timeout=180) as resp:
        out = resp.read()
        return out if raw else _gjson.loads(out.decode())

print('authenticated as:', _gh_req('GET', 'https://api.github.com/user').get('login'))
"""

PULL = """import os, io, tarfile, shutil
if os.path.isdir(WORK_DIR): shutil.rmtree(WORK_DIR)
os.makedirs(WORK_DIR, exist_ok=True)
blob = _gh_req('GET', f'https://api.github.com/repos/{REPO}/tarball/{SRC_REF}', raw=True)
with tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz') as tar:
    root = tar.getmembers()[0].name.split('/')[0]
    tar.extractall('/content/_repo_extract')
for name in os.listdir(f'/content/_repo_extract/{root}'):
    shutil.move(f'/content/_repo_extract/{root}/{name}', f'{WORK_DIR}/{name}')
shutil.rmtree('/content/_repo_extract'); os.chdir(WORK_DIR)
print('repo at', os.getcwd())
if not CALIBRATE:
    assert os.path.exists(ESTIMATOR), f'missing {ESTIMATOR} on {SRC_REF}'
    print('estimator present:', ESTIMATOR)
"""

INSTALL = """import subprocess, sys
pkgs = ['vllm==0.10.2', 'transformers>=4.55.2,<5.0',
        'tree-sitter==0.23.2', 'tree-sitter-python==0.23.6',
        'rank-bm25', 'python-Levenshtein', 'lightgbm', 'jsonlines', 'click',
        'pyflakes', 'tqdm', 'scipy', 'scikit-learn', 'datasets', 'huggingface-hub']
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *pkgs], check=True)
print('deps installed')
"""

HF_LOGIN = """# HF token: required for the-stack-dedup (gated) used by 0.5B calibration; the
# code models themselves are public. Reads Colab secret HF_TOKEN if present.
import os
try:
    from google.colab import userdata
    tok = userdata.get('HF_TOKEN')
    if tok: os.environ['HF_TOKEN'] = tok
except Exception:
    pass
print('HF_TOKEN set:', bool(os.environ.get('HF_TOKEN')))
"""

PROVISION = """import os, subprocess, zipfile, glob
os.chdir(WORK_DIR)
# CrossCodeEval (committed gzip asset)
os.makedirs('data/crosscodeeval/crosscodeeval_data/python', exist_ok=True)
cce = 'data/crosscodeeval/crosscodeeval_data/python/line_completion_rg1_bm25.jsonl'
if not os.path.exists(cce):
    subprocess.run(f'gunzip -c scripts/runpod/assets/cce_python_rg1_bm25.jsonl.gz > {cce}', shell=True, check=True)
print('CCE instances:', sum(1 for _ in open(cce)))
# RepoEval-function (microsoft/CodeT RepoCoder)
os.makedirs('data/repoeval/datasets', exist_ok=True)
os.makedirs('data/repoeval/repositories', exist_ok=True)
if not os.path.exists('/content/CodeT'):
    subprocess.run(['git','clone','--depth','1','--filter=blob:none','--sparse',
                    'https://github.com/microsoft/CodeT.git','/content/CodeT'], check=True)
    subprocess.run(['git','-C','/content/CodeT','sparse-checkout','set','RepoCoder'], check=True)
RC = '/content/CodeT/RepoCoder'
with zipfile.ZipFile(f'{RC}/datasets/datasets.zip') as z: z.extractall('data/repoeval/datasets')
with zipfile.ZipFile(f'{RC}/repositories/function_level.zip') as z: z.extractall('data/repoeval/repositories')
print('RepoEval function task:', glob.glob('data/repoeval/datasets/function_level_completion_2k*.jsonl'))
"""

CALIBRATE_CELL = """# ---- 0.5B ONLY: calibrate a fresh estimator, reuse the one .lgb for both datasets ----
# FULL : the-stack-dedup (gated -> needs HF_TOKEN), real guards, ~30 min.
# SMOKE: tiny public the-stack-smol sample, relaxed guards, isolated *_smoke paths
#        (a few min, rough estimator -- wiring check only; not pushed).
import os, subprocess, sys
os.chdir(WORK_DIR)
if SMOKE:
    ESTIMATOR = 'models/estimator_qwen25_0.5b_smoke.lgb'   # don't clobber a real estimator
    npz = 'data/training_data/qwen25_0.5b_smoke.npz'
    build = ['--source','the-stack-smol','--file-limit','300','--n-pairs','3000',
             '--per-file','25','--batch-size','256','--min-files','40','--min-pairs','400']
    train = ['--num-boost-round','60','--min-skill','-1.0']
else:
    npz = 'data/training_data/qwen25_0.5b.npz'
    build = ['--source','the-stack-dedup','--file-limit','15000','--n-pairs','250000',
             '--per-file','25','--batch-size','256','--min-files','8000','--min-pairs','20000']
    train = ['--num-boost-round','100']
if not os.path.exists(ESTIMATOR):
    if not os.path.exists(npz):
        subprocess.run([sys.executable, 'scripts/01_construct_training_data.py',
            '--backend','vllm','--model', MODEL, '--model-family', MODEL_FAMILY,
            '--max-tokens','50','--output', npz, *build], check=True)
    subprocess.run([sys.executable, 'scripts/02_train_estimator.py',
        '--data', npz, '--output', ESTIMATOR, *train], check=True)
    if not SMOKE:
        gh_upload(ESTIMATOR, 'models'); gh_upload(npz, 'data/training_data')
print('estimator ready:', ESTIMATOR, os.path.exists(ESTIMATOR), '|', 'SMOKE' if SMOKE else 'FULL')
"""

VERIFY = """import sys, os
os.chdir(WORK_DIR); sys.path.insert(0, 'src')
from adaptive_retrieval.eval.datasets import DATASET_LOADERS
for ds in DATASETS:
    n = sum(1 for _ in DATASET_LOADERS[ds]())
    print(f'{ds}: {n} instances loadable')
"""

GH_BRANCH = """def _gh_ensure_branch():
    refbase = f'https://api.github.com/repos/{REPO}/git/refs/heads/'
    try:
        _gh_req('GET', refbase + GH_RESULTS_BRANCH); return
    except urllib.error.HTTPError as e:
        if e.code != 404: raise
    info = _gh_req('GET', f'https://api.github.com/repos/{REPO}')
    head = _gh_req('GET', refbase + info['default_branch'])
    _gh_req('POST', f'https://api.github.com/repos/{REPO}/git/refs',
            {'ref': f'refs/heads/{GH_RESULTS_BRANCH}', 'sha': head['object']['sha']})

def _gh_get_sha(dest):
    try:
        return _gh_req('GET', f'https://api.github.com/repos/{REPO}/contents/{dest}?ref={GH_RESULTS_BRANCH}').get('sha')
    except Exception:
        return None

def gh_upload(path, prefix):
    from pathlib import Path as _P
    p = _P(path)
    if not p.exists():
        print('  skip (missing):', path); return
    dest = f'{prefix}/{p.name}'
    payload = {'message': f'colab: {dest}', 'content': base64.b64encode(p.read_bytes()).decode(),
               'branch': GH_RESULTS_BRANCH}
    sha = _gh_get_sha(dest)
    if sha: payload['sha'] = sha
    _gh_req('PUT', f'https://api.github.com/repos/{REPO}/contents/{dest}', payload)
    print('  pushed:', dest)

_gh_ensure_branch(); print('results branch ready:', GH_RESULTS_BRANCH)
"""

CLEAR_CACHE = """import shutil, os
os.chdir(WORK_DIR); cache = 'data/generation_cache'
if os.path.isdir(cache):
    shutil.rmtree(cache); print('cleared generation cache')
os.makedirs(cache, exist_ok=True)
"""

GENERATE = """import os, subprocess, sys
os.chdir(WORK_DIR)

def run_config(cfg, ds, subdir):
    out = f'{subdir}/{cfg}.jsonl'
    cmd = [sys.executable, 'scripts/04_run_experiment.py',
           '--config', cfg, '--dataset', ds, '--backend', 'vllm',
           '--model', MODEL, '--model-family', MODEL_FAMILY,
           '--max-tokens', str(MAX_TOKENS[ds]),
           '--t-rag', str(GEN_T_RAG), '--top-k', str(TOP_K),
           '--output', out, '--cache-dir', 'data/generation_cache']
    if cfg == 'C3_card':
        cmd += ['--estimator-path', ESTIMATOR]
    if SMOKE:
        cmd += ['--limit', str(SMOKE_LIMIT)]
    print('>>>', ' '.join(cmd)); subprocess.run(cmd, check=True)
    gh_upload(out, subdir)

for ds in DATASETS:
    subdir = f'results/{RESULTS_TAG}_{ds}'
    os.makedirs(subdir, exist_ok=True)
    for cfg in GEN_CONFIGS:
        run_config(cfg, ds, subdir)
print('generation done (C1/C2/C3 per dataset)')
"""

SWEEP = """import os, subprocess, sys
os.chdir(WORK_DIR)
for ds in DATASETS:
    subdir = f'results/{RESULTS_TAG}_{ds}'
    out_csv = f'{subdir}/sweep.csv'
    cmd = [sys.executable, 'scripts/13_sweep_eval.py',
           '--results-dir', subdir, '--dataset', ds, '--out-csv', out_csv, '--t-grid', T_GRID]
    print('>>>', ' '.join(cmd)); subprocess.run(cmd, check=True)
    gh_upload(out_csv, subdir)
print('sweep + metrics done; CSVs pushed')
"""

SUMMARY = """import os, csv
os.chdir(WORK_DIR)
for ds in DATASETS:
    path = f'results/{RESULTS_TAG}_{ds}/sweep.csv'
    if not os.path.exists(path): continue
    print(f'\\n===== {ds} =====')
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        if r['config'] in ('C1_no_retrieve','C2_always_retrieve'):
            print(f"  [{r['scoring']}] {r['config']:<20} EM={r['exact_match']} ES={r['edit_similarity']} "
                  f"hallA4B2={r['hall_A4B2']} lat={r['latency_ms']}ms")
print('\\nResults: https://github.com/' + REPO + '/tree/' + GH_RESULTS_BRANCH)
"""


def build(gen):
    cells = [
        md(f"# Cascade experiment · {gen['model']} · C1–C4 × {{CCE, RepoEval-function}}\n\n"
           f"Full `t_rag` sweep (0.05–0.95), five metrics, RepoEval scored body **and** full. "
           f"GPU: {gen['gpu_note']}\n\n"
           "Setup: `Runtime → GPU`; Colab Secret `LUCA_GITHUB_PAT` (Contents: read/write)"
           + ("; `HF_TOKEN` (the-stack-dedup license accepted)." if gen['calibrate'] else ".")),
        md("## 1. Config"),
        code(CONFIG.replace("__MODEL__", gen["model"]).replace("__FAMILY__", gen["family"])
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
        md("## 7. Provision data (CrossCodeEval asset + RepoEval/RepoCoder)"),
        code(PROVISION),
        md("## 8. Results branch + uploader (REST)"),
        code(GH_BRANCH),
    ]
    if gen["calibrate"]:
        cells += [md("## 8b. Calibrate the 0.5B estimator\n"
                     "FULL: the-stack-dedup (gated → needs `HF_TOKEN`), ~30 min. "
                     "SMOKE: tiny public `the-stack-smol` sample, relaxed guards, a few min "
                     "(rough estimator → isolated `_smoke` paths, not pushed)."),
                  code(CALIBRATE_CELL)]
    cells += [
        md("## 9. Verify dataset loaders"),
        code(VERIFY),
        md("## 10. Clear the generation cache (stop-strings changed → avoid stale cache)"),
        code(CLEAR_CACHE),
        md("## 11. Generate C1/C2/C3 per dataset (push each)\n"
           "C4 + the full t_rag sweep are produced post-hoc in step 12 — C3 here is only "
           "for its stored ŝ₀ (its zero-shot prompt is a cache hit on C1)."),
        code(GENERATE),
        md("## 12. Sweep t_rag (0.05–0.95) + five metrics → sweep.csv\n"
           "`13_sweep_eval.py` replays CARD (C3) and cascade (C4) at every t_rag, computes "
           "EM/ES/idF1/latency and hallucination (A4∧B2 + A4-only), and scores RepoEval **body** and **full**."),
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
        nb = build(gen)
        path = OUT_DIR / f"experiment_{gen['tag']}.ipynb"
        path.write_text(json.dumps(nb, indent=1))
        print("wrote", path, f"({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
