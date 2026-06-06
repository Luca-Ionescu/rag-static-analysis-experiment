#!/usr/bin/env bash
# Fire-and-forget CARD-identical calibration + C1-C4 benchmark on a RunPod A100.
#
# Designed to be run once on a fresh RunPod A100 pod, then walked away from.
# All artifacts (.lgb estimator, .npz training pairs, JSONL records, logs) are
# uploaded to a private HuggingFace Hub dataset repo at the end. The pod can
# optionally self-terminate once the upload succeeds.
#
# Required env vars (set in the RunPod pod template, or before launching the
# script):
#   HF_TOKEN              HuggingFace access token (read + write). The account
#                         behind it MUST have accepted the gated dataset license
#                         at https://huggingface.co/datasets/bigcode/the-stack-dedup
#                         (one-time click) — calibration streams that corpus.
#   HF_DATASET_REPO       Private dataset repo, e.g. "luca-ionescu/card-7b-results".
#                         Created if missing.
#
# PREREQUISITE: accept the the-stack-dedup license (link above) with the same HF
# account as HF_TOKEN before launching, or Phase 3 aborts at the --min-files
# guardrail (by design — better a fast abort than a strawman Estimator).
#
# Optional env vars (defaults are CARD-paper-identical):
#   MODEL                 default: codellama/CodeLlama-7b-hf
#   MODEL_FAMILY          default: codellama (FIM token set)
#   MAX_TOKENS            default: 50
#   NPAIRS                default: 250000 (upper cap on pairs after dedup)
#   BATCH_SIZE            default: 256 (vLLM chunk size)
#   FILE_LIMIT            default: 15000 (target valid files streamed from the Stack)
#   MIN_FILES             default: 8000 (abort before GPU if fewer valid files stream out)
#   MIN_PAIRS             default: 20000 (abort rather than write a tiny calibration set)
#   DATASET               default: crosscodeeval_py
#   TRAG                  default: 0.9 (CARD-RG1 threshold)
#   WORK_DIR              default: /workspace/project-group-17
#   REPO_URL              default: this project's GitHub URL
#   AUTO_TERMINATE        default: false. If "true", the pod self-terminates
#                         after a successful upload. Requires RUNPOD_API_KEY.
#   RUNPOD_API_KEY        only required if AUTO_TERMINATE=true
#   RUNPOD_POD_ID         set automatically by RunPod inside the pod
#
# Idempotency: each phase checks whether its output already exists and skips if
# so. Safe to re-run after a crash (e.g. pod reboot).
#
# Usage:
#   tmux new -s card                  # detachable session, survives SSH drop
#   export HF_TOKEN=hf_xxx
#   export HF_DATASET_REPO=username/card-7b-results
#   bash scripts/runpod/run_card_pipeline.sh
#   # Ctrl+B then D to detach. SSH back and `tmux attach -t card` to resume.
#
set -euo pipefail

# ---------- config ----------
HF_DATASET_REPO="${HF_DATASET_REPO:?Set HF_DATASET_REPO (e.g. username/card-7b-results)}"
MODEL="${MODEL:-codellama/CodeLlama-7b-hf}"
MODEL_FAMILY="${MODEL_FAMILY:-codellama}"
MAX_TOKENS="${MAX_TOKENS:-50}"
NPAIRS="${NPAIRS:-250000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
FILE_LIMIT="${FILE_LIMIT:-15000}"
MIN_FILES="${MIN_FILES:-8000}"
MIN_PAIRS="${MIN_PAIRS:-20000}"
DATASET="${DATASET:-crosscodeeval_py}"
TRAG="${TRAG:-0.9}"
WORK_DIR="${WORK_DIR:-/workspace/project-group-17}"
REPO_URL="${REPO_URL:-https://github.com/Luca-Ionescu/rag-static-analysis-experiment.git}"
AUTO_TERMINATE="${AUTO_TERMINATE:-false}"

# ---------- env validation ----------
: "${HF_TOKEN:?Set HF_TOKEN env var (pod template or shell)}"
if [[ "$AUTO_TERMINATE" == "true" ]]; then
    : "${RUNPOD_API_KEY:?Set RUNPOD_API_KEY when AUTO_TERMINATE=true}"
    : "${RUNPOD_POD_ID:?RUNPOD_POD_ID should be set by RunPod automatically}"
fi
export HF_TOKEN

# Force the HuggingFace cache onto the container disk (~70 GB free).
# RunPod's pytorch image pre-sets HF_HOME=/workspace/.cache/huggingface,
# but /workspace is a network mount with a per-pod 20 GB quota that
# CodeLlama-7B (13 GB raw + xet scratch) blows through. Override
# unconditionally — using `=` not `:=` so we replace any inherited value.
export HF_HOME=/root/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"
echo "[setup] HF_HOME=$HF_HOME"

# Logging timestamp (the LOG_DIR is created AFTER the clone — creating it
# beforehand would materialize WORK_DIR as a side effect and break the
# subsequent git clone).
LOG_TS=$(date +%Y%m%d_%H%M%S)

phase() {
    echo
    echo "============================================================"
    echo "[$(date +%H:%M:%S)] PHASE: $*"
    echo "============================================================"
}

# ---------- Phase 1: setup ----------
phase "1/7 Setup"

# Pick the Python interpreter that pip targets (the runpod/pytorch image
# ships multiple Python versions; pip and the image's preinstalled
# packages live under the highest one, often python3.13). All subsequent
# `python` / `pip` invocations use this binary to avoid the install-here-
# import-there split.
if command -v python3.13 >/dev/null 2>&1; then
    PY=python3.13
elif command -v python3.12 >/dev/null 2>&1; then
    PY=python3.12
elif command -v python3.11 >/dev/null 2>&1; then
    PY=python3.11
else
    PY=python3
fi
echo "[setup] using $PY ($($PY --version 2>&1))"
PIP="$PY -m pip"

if [[ -d "$WORK_DIR" && ! -d "$WORK_DIR/.git" ]]; then
    echo "[setup] $WORK_DIR exists but is not a git checkout; clearing it"
    rm -rf "$WORK_DIR"
fi
if [[ ! -d "$WORK_DIR/.git" ]]; then
    git clone "$REPO_URL" "$WORK_DIR"
fi
cd "$WORK_DIR"
git fetch origin
git checkout main
git pull --rebase --autostash || true

# Now that WORK_DIR is a valid clone, set up the structured log file.
LOG_DIR="${WORK_DIR}/logs/runpod_${LOG_TS}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/master.log") 2>&1
echo "[setup] log file: $LOG_DIR/master.log"

# Install the runpod-specific requirements (matches the
# runpod/pytorch:0.7.0-cu1281-torch271 image). The main requirements.txt
# is for local Mac/mlx development and pins an older vLLM/torch combo.
if [[ -f requirements-runpod.txt ]]; then
    # Show pip progress live (no -q) so users can see what's happening
    # during the 5-15 minute vllm + torch install.
    $PIP install -r requirements-runpod.txt --progress-bar on
else
    echo "[error] requirements-runpod.txt not found. Are you on main?" >&2
    exit 1
fi

mkdir -p data/training_data models "results/codellama_7b" data/generation_cache

# The runpod/pytorch image exports HF_HUB_ENABLE_HF_TRANSFER=1 to use the fast
# Rust downloader, but the `hf_transfer` package isn't always present (and the
# HF client then hard-fails instead of falling back). requirements-runpod.txt
# installs it; if it's still not importable, drop back to the standard
# downloader so the model pull can't crash the run.
if ! $PY -c "import hf_transfer" >/dev/null 2>&1; then
    export HF_HUB_ENABLE_HF_TRANSFER=0
    echo "[setup] hf_transfer unavailable -> HF_HUB_ENABLE_HF_TRANSFER=0 (standard download)"
else
    echo "[setup] hf_transfer present -> fast downloads enabled"
fi

# ---------- Phase 2: model download ----------
phase "2/7 Model download (CodeLlama-7B)"
$PY - <<PY
import os
from huggingface_hub import snapshot_download
snapshot_download("$MODEL", token=os.environ["HF_TOKEN"])
print("Model cached.")
PY

# ---------- Phase 3: calibration training data ----------
TRAIN_NPZ="data/training_data/codellama_7b.npz"
if [[ -f "$TRAIN_NPZ" ]]; then
    echo "[skip] $TRAIN_NPZ already exists"
else
    phase "3/7 Calibration training data (~2-3h)"
    $PY scripts/01_construct_training_data.py \
        --source the-stack-dedup \
        --file-limit "$FILE_LIMIT" \
        --backend vllm \
        --model "$MODEL" \
        --model-family "$MODEL_FAMILY" \
        --max-tokens "$MAX_TOKENS" \
        --n-pairs "$NPAIRS" \
        --per-file 25 \
        --batch-size "$BATCH_SIZE" \
        --min-files "$MIN_FILES" \
        --min-pairs "$MIN_PAIRS" \
        --output "$TRAIN_NPZ" \
        2>&1 | tee "$LOG_DIR/calibration.log"
fi

# ---------- Phase 4: train Estimator ----------
ESTIMATOR="models/estimator_codellama_7b.lgb"
if [[ -f "$ESTIMATOR" ]]; then
    echo "[skip] $ESTIMATOR already exists"
else
    phase "4/7 Train LightGBM Estimator"
    $PY scripts/02_train_estimator.py \
        --data "$TRAIN_NPZ" \
        --output "$ESTIMATOR" \
        --num-boost-round 100 \
        2>&1 | tee "$LOG_DIR/train_estimator.log"
fi

# ---------- Phase 5: prepare benchmark data ----------
# CrossCodeEval ships as a committed gzipped asset (decompressed below).
# RepoEval is too large to commit, so it must be provisioned once under
# data/repoeval/ on the persistent volume (see load_repoeval).
phase "5/7 Prepare benchmark data ($DATASET)"
if [[ "$DATASET" == crosscodeeval_py ]]; then
    CCE_JSONL="data/crosscodeeval/crosscodeeval_data/python/line_completion_rg1_bm25.jsonl"
    CCE_ARCHIVE="scripts/runpod/assets/cce_python_rg1_bm25.jsonl.gz"
    EXPECTED_CCE_LINES=2665
    if [[ -s "$CCE_JSONL" ]]; then
        echo "[skip] $CCE_JSONL already present ($(wc -l < "$CCE_JSONL") lines)"
    else
        if [[ ! -f "$CCE_ARCHIVE" ]]; then
            echo "[error] benchmark archive missing: $CCE_ARCHIVE" >&2
            echo "        It is committed to the repo — are you on an up-to-date main?" >&2
            exit 1
        fi
        mkdir -p "$(dirname "$CCE_JSONL")"
        gunzip -c "$CCE_ARCHIVE" > "$CCE_JSONL"
        lines=$(wc -l < "$CCE_JSONL")
        echo "[prep] decompressed $CCE_ARCHIVE -> $CCE_JSONL ($lines lines)"
        if [[ "$lines" -ne "$EXPECTED_CCE_LINES" ]]; then
            echo "[error] expected $EXPECTED_CCE_LINES instances, got $lines — archive looks corrupt." >&2
            rm -f "$CCE_JSONL"
            exit 1
        fi
    fi
elif [[ "$DATASET" == repoeval_* ]]; then
    # RepoEval data lives on the persistent volume (too big to commit). Provision once, e.g.:
    #   git clone --depth 1 https://github.com/microsoft/CodeT /tmp/CodeT
    #   mkdir -p data/repoeval && cp -r /tmp/CodeT/RepoCoder/datasets data/repoeval/datasets
    #   # then extract the repositories archive into data/repoeval/repositories/
    if [[ ! -d data/repoeval/datasets || ! -d data/repoeval/repositories ]]; then
        echo "[error] RepoEval data not found (data/repoeval/{datasets,repositories} missing)." >&2
        echo "        Provision the RepoCoder datasets + repositories under data/repoeval/ then re-run:" >&2
        echo "        https://github.com/microsoft/CodeT/tree/main/RepoCoder" >&2
        exit 1
    fi
    echo "[prep] RepoEval data present ($(ls data/repoeval/datasets/*.jsonl 2>/dev/null | wc -l | tr -d ' ') dataset files)"
    if [[ "$DATASET" == repoeval_function && "$MAX_TOKENS" -lt 128 ]]; then
        echo "[warn] repoeval_function with MAX_TOKENS=$MAX_TOKENS — function bodies need more; set MAX_TOKENS=256+." >&2
    fi
else
    echo "[error] unknown DATASET=$DATASET (expected crosscodeeval_py or repoeval_*)" >&2
    exit 1
fi

# ---------- Phase 6: run C1-C4 ----------
phase "6/7 Benchmark C1-C4 on $DATASET (~1.5h)"
# Keep CrossCodeEval at the existing path (back-compat); separate other datasets
# so results never collide when switching benchmarks.
if [[ "$DATASET" == crosscodeeval_py ]]; then
    RESULTS_DIR="results/codellama_7b"
else
    RESULTS_DIR="results/codellama_7b_${DATASET}"
fi

run_config() {
    local cfg=$1
    shift
    local out="$RESULTS_DIR/${cfg}.jsonl"
    if [[ -f "$out" ]]; then
        echo "[skip] $out already exists"
        return 0
    fi
    $PY scripts/04_run_experiment.py \
        --config "$cfg" \
        --dataset "$DATASET" \
        --backend vllm \
        --model "$MODEL" \
        --model-family "$MODEL_FAMILY" \
        --max-tokens "$MAX_TOKENS" \
        --t-rag "$TRAG" \
        --output "$out" \
        --cache-dir data/generation_cache \
        "$@" 2>&1 | tee "$LOG_DIR/${cfg}.log"
}

run_config C1_no_retrieve
run_config C2_always_retrieve
run_config C3_card --estimator-path "$ESTIMATOR"
run_config C4_cascade --estimator-path "$ESTIMATOR"

# ---------- Phase 7: upload artifacts to HF Hub ----------
phase "7/7 Upload artifacts to HuggingFace Hub"

# Bundle the generation cache (pickled Generations keyed by SHA-256(prompt)) so
# every post-hoc T_RAG / t_acc sweep and cascade-logic change can be replayed on
# CPU with `04_run_experiment.py --backend mock --cache-dir <unpacked>` instead
# of re-running this GPU pipeline. The cache holds the zero-shot AND retrieved
# generation for every benchmarked instance (~5k small pkls, a few MB gzipped).
CACHE_TARBALL="$RESULTS_DIR/generation_cache.tar.gz"
if [[ -d data/generation_cache ]]; then
    n_pkl=$(find data/generation_cache -name '*.pkl' | wc -l | tr -d ' ')
    echo "[cache] bundling $n_pkl cached generations -> $CACHE_TARBALL"
    tar -czf "$CACHE_TARBALL" -C data generation_cache
    echo "[cache] $(du -h "$CACHE_TARBALL" | cut -f1) tarball ready"
else
    echo "[warn] data/generation_cache missing; post-hoc CPU replay will be impossible" >&2
fi

$PY - <<PY
import os, pathlib, glob, sys
from huggingface_hub import HfApi, create_repo

token = os.environ["HF_TOKEN"]
repo_id = "$HF_DATASET_REPO"
api = HfApi(token=token)

create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True, token=token)

paths = [
    "$ESTIMATOR",
    "$TRAIN_NPZ",
    "$CACHE_TARBALL",
] + glob.glob("$RESULTS_DIR/*.jsonl") + glob.glob("$LOG_DIR/*.log")

uploaded = 0
for path in paths:
    p = pathlib.Path(path)
    if not p.exists():
        print(f"[warn] missing: {path}", file=sys.stderr)
        continue
    api.upload_file(
        path_or_fileobj=str(p),
        path_in_repo=str(p),
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"[upload] {path}")
    uploaded += 1
print(f"Uploaded {uploaded} files to https://huggingface.co/datasets/{repo_id}")
PY

# ---------- Optional: self-terminate the pod ----------
if [[ "$AUTO_TERMINATE" == "true" ]]; then
    phase "Self-terminate (AUTO_TERMINATE=true)"
    # Give the upload a moment to settle before the pod dies.
    sleep 10
    curl -sS -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }\"}" \
        || echo "[warn] self-terminate API call failed; terminate manually from the web UI."
fi

echo
echo "============================================================"
echo "[$(date +%H:%M:%S)] PIPELINE COMPLETE"
echo "============================================================"
echo
echo "Artifacts: https://huggingface.co/datasets/$HF_DATASET_REPO"
echo "Logs:      $LOG_DIR"
