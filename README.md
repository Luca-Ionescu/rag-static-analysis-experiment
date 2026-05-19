# Adaptive Retrieval for Repository-Level Code Completion

A CARD-based adaptive-retrieval framework with a static-analysis cascade
stage that catches hallucinated identifiers CARD's uncertainty signal misses.

See [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) for the full spec
and [project_report.pdf](project_report.pdf) for the current status report.

## Setup

```bash
# Mac (Apple Silicon)
conda create -n adaptive-retrieval python=3.11
conda activate adaptive-retrieval
pip install -r requirements-dev.txt

# Linux + CUDA (GPU node)
pip install -r requirements.txt
```

## Run a smoke test (no GPU)

```bash
# 1. Train a synthetic Estimator
python -c "
import sys, numpy as np
sys.path.insert(0, 'src')
from adaptive_retrieval.card.estimator import Estimator
from adaptive_retrieval.card.features import extract_features
rng = np.random.default_rng(0)
feats = np.array([extract_features(
    np.clip(rng.normal(0.7, 0.2, 30), 0.01, 1),
    np.clip(rng.normal(0.5, 0.3, 30), 0, 5)) for _ in range(500)])
scores = feats[:, 2].astype(np.float32)  # ES ~ prob_avg
Estimator.train(feats, scores).save('models/estimator_synthetic.lgb')
print('saved')
"

# 2. End-to-end smoke through all 6 configs
python scripts/00_smoke_test.py --estimator models/estimator_synthetic.lgb
```

## Run the full matrix (Phase 6)

```bash
# Local: Apple Silicon + mlx-lm
python scripts/01_construct_training_data.py \
    --backend mlx --model Qwen/Qwen2.5-Coder-0.5B \
    --n-pairs 50000 --output data/training_data/qwen25_05b.npz
python scripts/02_train_estimator.py \
    --data data/training_data/qwen25_05b.npz \
    --output models/estimator_qwen25_05b.lgb
python scripts/run_full_matrix.py \
    --backend mlx --model Qwen/Qwen2.5-Coder-0.5B \
    --estimator-path models/estimator_qwen25_05b.lgb \
    --datasets crosscodeeval_py
```

## Analysis (Phase 7)

```bash
python scripts/06_analysis.py \
    --c1 results/crosscodeeval_py/C1_no_retrieve.jsonl \
    --c2 results/crosscodeeval_py/C2_always_retrieve.jsonl \
    --c3 results/crosscodeeval_py/C3_card.jsonl \
    --c4 results/crosscodeeval_py/C4_cascade.jsonl \
    --output-dir analysis/cce
```

## Tests

```bash
pytest tests/             # 167+ tests
ruff check src/ tests/ scripts/
```

## Layout

```
src/adaptive_retrieval/      modules: generator, retriever, prompt, baselines,
                             cascade, metrics, card/, static_analysis/, eval/
scripts/                     CLIs: smoke, training-data, estimator-train,
                             run_experiment, run_full_matrix, analysis,
                             compute_metrics, build_project_report
tests/                       per-module test suites
data/                        crosscodeeval/ (added), repoeval/ (download
                             separately), stack_subset/, training_data/
models/                      saved Estimator .lgb files
results/                     per-config per-dataset JSONL
analysis/                    Phase 7 summaries and plots
paper/                       LaTeX skeleton ready for results
```

## Project status

See [project_report.pdf](project_report.pdf). Quick summary:

- **Phases 0–5 implemented**, including the novel static-analysis cascade.
- **167+ tests passing**, lint clean.
- **All implementation gates met locally**; the CARD-paper reproduction
  gate needs CodeLlama-7B which requires a GPU node.
- **Phase 6/7 infrastructure ready**: scripts to run the full matrix and
  produce the paper's headline numbers from the resulting JSONL.
- **Phase 8 paper skeleton** ready at `paper/main.tex` with TODO
  placeholders for results.

## References

Papers in `papers/`:
- 2303.12570 — RepoCoder (Zhang et al. 2023): introduces RepoEval.
- 2305.06983 — FLARE (Jiang et al. 2023): active retrieval.
- 2310.11248 — CrossCodeEval (Ding et al. 2023): primary benchmark.
- 2403.10059 — Repoformer (Wu et al. 2024): K-means dedup recipe.
- 2406.10263 — CARD (Zhang et al. 2024): the framework we extend.
