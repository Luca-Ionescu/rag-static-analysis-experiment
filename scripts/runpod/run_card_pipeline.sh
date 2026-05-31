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
#   HF_TOKEN              HuggingFace access token (read + write).
#   HF_DATASET_REPO       Private dataset repo, e.g. "luca-ionescu/card-7b-results".
#                         Created if missing.
#
# Optional env vars (defaults are CARD-paper-identical):
#   MODEL                 default: codellama/CodeLlama-7b-hf
#   MODEL_FAMILY          default: codellama (FIM token set)
#   MAX_TOKENS            default: 50
#   NPAIRS                default: 250000 (target after dedup)
#   BATCH_SIZE            default: 256 (vLLM chunk size)
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
phase "1/6 Setup"
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
    pip install -r requirements-runpod.txt --progress-bar on
else
    echo "[error] requirements-runpod.txt not found. Are you on main?" >&2
    exit 1
fi

mkdir -p data/training_data models "results/codellama_7b" data/generation_cache

# ---------- Phase 2: model download ----------
phase "2/6 Model download (CodeLlama-7B)"
python - <<PY
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
    phase "3/6 Calibration training data (~2-3h)"
    python scripts/01_construct_training_data.py \
        --source the-stack-smol \
        --backend vllm \
        --model "$MODEL" \
        --model-family "$MODEL_FAMILY" \
        --max-tokens "$MAX_TOKENS" \
        --n-pairs "$NPAIRS" \
        --per-file 25 \
        --batch-size "$BATCH_SIZE" \
        --output "$TRAIN_NPZ" \
        2>&1 | tee "$LOG_DIR/calibration.log"
fi

# ---------- Phase 4: train Estimator ----------
ESTIMATOR="models/estimator_codellama_7b.lgb"
if [[ -f "$ESTIMATOR" ]]; then
    echo "[skip] $ESTIMATOR already exists"
else
    phase "4/6 Train LightGBM Estimator"
    python scripts/02_train_estimator.py \
        --data "$TRAIN_NPZ" \
        --output "$ESTIMATOR" \
        --num-boost-round 100 \
        2>&1 | tee "$LOG_DIR/train_estimator.log"
fi

# ---------- Phase 5: run C1-C4 on CCE-Python ----------
phase "5/6 Benchmark C1-C4 on CCE-Python (~1.5h)"
RESULTS_DIR="results/codellama_7b"

run_config() {
    local cfg=$1
    shift
    local out="$RESULTS_DIR/${cfg}.jsonl"
    if [[ -f "$out" ]]; then
        echo "[skip] $out already exists"
        return 0
    fi
    python scripts/04_run_experiment.py \
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

# ---------- Phase 6: upload artifacts to HF Hub ----------
phase "6/6 Upload artifacts to HuggingFace Hub"
python - <<PY
import os, pathlib, glob, sys
from huggingface_hub import HfApi, create_repo

token = os.environ["HF_TOKEN"]
repo_id = "$HF_DATASET_REPO"
api = HfApi(token=token)

create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True, token=token)

paths = [
    "$ESTIMATOR",
    "$TRAIN_NPZ",
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
