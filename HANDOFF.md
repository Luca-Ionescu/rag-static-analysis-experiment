# Handoff: Adaptive Retrieval for Repository-Level Code Completion

A self-contained orientation for the next agent picking up this project.
Read this first; then [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) is
the long-form spec; then [README.md](README.md) is the user-facing recipe.

---

## 1. What this project is

A 4-week empirical research project. We reimplement CARD (Zhang et al. 2024,
[papers/2406.10263v1.pdf](papers/2406.10263v1.pdf)) — an adaptive
retrieval-augmented code-completion framework — and add a **static-analysis
cascade stage** as the novel contribution.

**Core hypothesis**: CARD's logit-based uncertainty signal misses a specific
failure mode — confidently-generated identifiers that don't actually exist
in the repository. Static analysis of the model's prediction can catch
these and trigger retrieval that CARD would otherwise skip.

**The cascade is asymmetric**: static analysis can only *add* retrievals to
CARD's decisions, never remove them. This bounds the worst-case retrieval
count to "always-retrieve" and frames the research question cleanly as
"does the extra retrieval budget reduce hallucinations?"

Three research questions: RQ1 cascade vs no/always-retrieve baselines, RQ2
hallucination reduction over vanilla CARD (McNemar's test), RQ3 CARD vs
baselines (paper reproduction). Full spec in
[IMPLEMENTATION_GUIDE.md §2](IMPLEMENTATION_GUIDE.md).

---

## 2. State of the world

**Phases 0–5 done. Phases 6–8 scaffolded but not run.** 167 tests passing,
ruff clean.

Status snapshot table:

| Phase | Status | What it produced |
|---|---|---|
| 0 setup | done | conda env `adaptive-retrieval` (Py 3.11.15), dir scaffolding |
| 1 static analysis | done | `parser`, `symbol_table`, `scope`, `analyzer` modules + 25 tests |
| 2 generator/retriever/baselines/metrics | done | 4 generator backends, BM25, FIM prompts, C1/C2, full metric suite + 41 tests |
| 3 CARD reimplementation | done (modulo GPU validation) | features, Estimator, pipeline, train_data + 47 tests |
| 4 cascade integration | done | `cascade.py` + 10 tests covering all 4 trigger reasons |
| 5 evaluation infrastructure | done | runner, generation cache, CLI, all 6 configs run on 50 CCE instances |
| 6 main experiments | **scaffolded only** | orchestrator script ready; no real-model runs yet |
| 7 analysis | scaffolded + tested | analysis module + 16 tests; CLI awaits Phase 6 JSONL |
| 8 paper | skeleton only | `paper/main.tex` with `\todo{}` placeholders for results |

**Validation gates** (IMPLEMENTATION_GUIDE §16):

| Gate | Status |
|---|---|
| 0: deps importable (relaxed, vLLM deferred) | PASS |
| 1: 22 Appendix-E static-analysis tests | PASS |
| 2: 10 CCE instances × baselines, metrics no errors | PASS |
| 3a: feature-vector shape and stability | PASS |
| 3b: Estimator MSE on synthetic data < 0.10 | PASS (0.0014) |
| 3c: CARD vs paper Table 3 within ±1% ES | **deferred — needs GPU + CodeLlama-7B** |
| 4: cascade exercises 2+ trigger reasons on real data | PASS (3/4 with mock) |
| 5: 50 instances × 6 configs end-to-end | PASS |

---

## 3. Environment

Conda env `adaptive-retrieval` (Python 3.11.15) installed at
`/Users/lucaionescu/miniconda3/envs/adaptive-retrieval`. Activate with:

```bash
conda activate adaptive-retrieval
# Or invoke directly:
/Users/lucaionescu/miniconda3/envs/adaptive-retrieval/bin/python ...
/Users/lucaionescu/miniconda3/envs/adaptive-retrieval/bin/pytest ...
```

Two requirements files exist:
- [requirements.txt](requirements.txt) — full GPU stack (vLLM, CUDA torch). Pinned.
- [requirements-dev.txt](requirements-dev.txt) — Mac-installable subset.
  Excludes vLLM; **includes mlx-lm and matplotlib**.

**The platform is macOS (M4 Pro, Apple Silicon, 24 GB RAM)**. Local runs use
mlx-lm. GPU runs (DelftBlue or similar) would use vLLM.

---

## 4. Code map

```
src/adaptive_retrieval/
  generator.py        # Generator interface; HF/vLLM/MLX/Mock/Cached backends
  retriever.py        # BM25 (20-line/stride-10 chunks); make_query()
  prompt.py           # FIM templates for qwen/codellama/starcoder
  baselines.py        # C1 no-retrieve, C2 always-retrieve
  cascade.py          # C4: CARD + static-analysis cascade (the contribution)
  metrics.py          # EM, ES, IdF1, hallucination_flag, McNemar, bootstrap
  card/
    features.py       # 13-D Table 1 vector, log-space prod/geomavg
    estimator.py      # LightGBM wrapper + MockEstimator
    pipeline.py       # CARD's Algorithm 1, single-RAG variant
    train_data.py     # Stack sampling + K-means dedup + generation
  static_analysis/
    parser.py         # tree-sitter setup
    symbol_table.py   # Repo-wide name table; in-memory or filesystem
    scope.py          # InFileScopeAnalyzer.visible_at(source, hole_byte)
    analyzer.py       # PredictionAnalyzer — the novel signal
  eval/
    datasets.py       # Instance dataclass, load_crosscodeeval_python, load_repoeval
    runner.py         # run_experiment() for all 6 configs + aggregates
    analysis.py       # Phase 7: trigger breakdown, disagreement, sweeps
scripts/
  00_smoke_test.py            # Appendix F: 1 synthetic instance × 6 configs
  01_construct_training_data.py  # CARD Estimator training-data CLI (GPU job)
  02_smoke_generator.py       # Manual Generator HF backend check
  02_train_estimator.py       # Train Estimator from (features, scores) npz
  03_smoke_pipeline.py        # 10 CCE instances × baselines
  04_run_experiment.py        # CLI for any (config, dataset) combination
  04_smoke_card.py            # CARD pipeline smoke with synthetic Estimator
  05_compute_metrics.py       # Recompute aggregates from JSONL
  05_smoke_cascade.py         # Cascade smoke with trigger-distribution report
  06_analysis.py              # Phase 7 full analysis (McNemar, bootstrap, sweep, plot)
  build_project_report.py     # Generates project_report.pdf
  run_full_matrix.py          # Phase 6 orchestrator
paper/
  main.tex                    # 2-column skeleton with \todo{} placeholders
  refs.bib                    # CARD, RepoCoder, CrossCodeEval, Repoformer, FLARE, Stack
  README.md                   # How to fill in the TODOs from analysis output
tests/                        # 14 test files, 167 tests, all passing
data/
  crosscodeeval/              # downloaded (~42 MB .tar.xz + extracted)
  repoeval/                   # NOT downloaded yet
  stack_subset/               # not populated; loaded via HF on demand
  training_data/              # destination for 01_construct_training_data.py npz
models/                       # destination for trained Estimators
results/                      # destination for per-config JSONL
analysis/                     # destination for Phase 7 summary.json + plots
```

Notable scripts run in this layout: `scripts/00_smoke_test.py` runs all six
configs against one synthetic instance — run this first to confirm wiring.

---

## 5. Non-obvious facts and gotchas

**Things that aren't visible from reading the code**:

### 5.1 Deviations from IMPLEMENTATION_GUIDE.md

Each one is documented inline in the relevant module.

- **tree-sitter version**: guide pins 0.22.3 + tree-sitter-python 0.21.0;
  reality is 0.23.2 + 0.23.6 (no Py3.11 wheels for the older pins). Same
  API. See [requirements.txt:17](requirements.txt:17).
- **vLLM is not installed locally** (Linux/CUDA only). Local dev uses
  MockGenerator, HFGenerator, or MLXGenerator. The dev variant of
  requirements ships mlx-lm 0.31.3 which **upgrades transformers to 5.8.1**
  and breaks `HFGenerator` (transformers 5.x requires torch ≥ 2.4; we have
  2.3.0). Use mlx-lm locally, vLLM on GPU node. Don't try to use HFGenerator
  in the local env unless you downgrade transformers.
- **CrossCodeEval schema**: the raw `line_completion.jsonl` has
  `crossfile_context: None`. The chunks live in the `_rg1_*` variants.
  Loader defaults to `line_completion_rg1_bm25.jsonl`. The `prompt`,
  `groundtruth`, `right_context` are byte-identical across variants — we're
  only borrowing the chunk list, not the retrieval-baked prompt. See
  [src/adaptive_retrieval/eval/datasets.py:5-13](src/adaptive_retrieval/eval/datasets.py).
- **The Stack subset**: guide describes 11k repos × 50–100 files; reality
  is ~10k random Python files. Recipe adapted: sample 25 (X, y) pairs per
  file. See [src/adaptive_retrieval/card/train_data.py](src/adaptive_retrieval/card/train_data.py) module docstring.

### 5.2 Tree-sitter 0.23.x identity quirk

`child_by_field_name` returns a **fresh Python wrapper on each call**, so
the `is` identity comparison fails (returns False for the same logical
node). Always compare nodes via `node.id ==`, not `is` or `==`. Without
this, every attribute name (`np.array`'s `array`, `self.x`'s `x`,
`f.read()`'s `read`) would be falsely flagged as unresolved. Already fixed,
but if you add new tree-sitter walking code, remember this. See
[src/adaptive_retrieval/static_analysis/analyzer.py:151](src/adaptive_retrieval/static_analysis/analyzer.py:151).

### 5.3 float32 overflow in CARD features

The entropy product feature `ent_prod` can overflow float32 when entropies
are large and N is moderate (e.g. 5^170 → inf). Computation is done in
float64 then clipped to ±finfo(float32).max before downcast. Don't change
this without re-running the random-input finite-output test. See
[src/adaptive_retrieval/card/features.py:55-60](src/adaptive_retrieval/card/features.py:55).

### 5.4 The Estimator is generator-specific

CARD's 13-D feature vector is built from the generator's logits. The
mapping from features to ES is generator-specific. **Train the Estimator
with the same generator you use at inference.** Don't mix a 0.5B-trained
Estimator with a 7B inference run.

### 5.5 Generation cache is shared across configs

[CachedGenerator](src/adaptive_retrieval/generator.py) keys on `sha256(model :: prompt :: max_tokens)`. C1's
zero-shot prompts are identical to C3/C4's zero-shot prompts → 100% cache
hit. After C1+C2 warm the cache (~5,330 generations for CCE-Python), every
subsequent config completes for free. This is the §15.3 efficiency win.
**Don't disable the cache unless you're debugging the generator** — the
full matrix without it is multi-hour even with mlx-lm.

### 5.6 Static analysis trigger-reason priority

In the cascade, when both `unresolved` AND `crossfile` identifiers fire on
the same prediction, the trigger reason is `static_unresolved` (the
stronger hallucination signal). Both fields are populated in the
CascadeOutput regardless of which is the trigger reason — useful for
diagnostics.

### 5.7 Model-size discussion summary

If the agent is asked about running with a smaller model, the relevant
tradeoffs are: (a) Estimator MSE rises (0.07 paper → 0.10–0.18 expected at
0.5B), (b) baseline landscape shifts (always-retrieve becomes stronger
relative to no-retrieve), (c) hallucinations are more abundant but
qualitatively different (more obvious tokens). The cascade-vs-baselines
*direction* should hold; the magnitudes won't transfer to 7B. CARD paper
reproduction specifically requires CodeLlama-7B.

---

## 6. How to pick up the work

### 6.1 Quick orientation (< 5 minutes)

```bash
conda activate adaptive-retrieval
pytest tests/                            # should show 167 passed
ruff check src/ tests/ scripts/          # should be clean
python scripts/00_smoke_test.py \
    --estimator models/estimator_synthetic.lgb
# Expect: all 6 configs run, ES≈1.0 (MockGenerator is lucky on this synthetic)
```

If any of those fail, something broke since 2026-05-19 and the rest of
this document might be stale. Inspect first.

A trained synthetic Estimator already exists at
`models/estimator_synthetic.lgb` from the Phase 5 smoke. Real-model
training is Phase 6.

### 6.2 Reading order for cold start

1. [README.md](README.md) — user-facing recipe (3 min)
2. [project_report.pdf](project_report.pdf) — full project context (10 min)
3. [src/adaptive_retrieval/cascade.py](src/adaptive_retrieval/cascade.py) — the contribution in 70 lines
4. [src/adaptive_retrieval/static_analysis/analyzer.py](src/adaptive_retrieval/static_analysis/analyzer.py) — the novel signal
5. [tests/test_cascade.py](tests/test_cascade.py) — every trigger reason exercised

Skip until needed: the IMPLEMENTATION_GUIDE itself (~3000 lines), papers/.

---

## 7. What still needs doing

In rough priority order.

### 7.1 Phase 6 — actual experiment runs

**This is the biggest single piece of remaining work.** The scaffolding is
ready; running it takes hours.

**Local recipe (Apple Silicon, mlx-lm)**:
```bash
# 1. Estimator training data (~5–10 h on M4 Pro for 50k pairs)
python scripts/01_construct_training_data.py \
    --backend mlx --model Qwen/Qwen2.5-Coder-0.5B \
    --n-pairs 50000 \
    --output data/training_data/qwen25_05b.npz

# 2. Train Estimator (~1 min)
python scripts/02_train_estimator.py \
    --data data/training_data/qwen25_05b.npz \
    --output models/estimator_qwen25_05b.lgb

# 3. Smoke before committing to the full run
python scripts/00_smoke_test.py \
    --backend mlx --model Qwen/Qwen2.5-Coder-0.5B \
    --estimator models/estimator_qwen25_05b.lgb

# 4. Full matrix on CCE-Python (3–6 h with cache)
python scripts/run_full_matrix.py \
    --backend mlx --model Qwen/Qwen2.5-Coder-0.5B \
    --estimator-path models/estimator_qwen25_05b.lgb \
    --datasets crosscodeeval_py
```

**GPU recipe (DelftBlue or similar Linux+CUDA)**: same scripts, swap
`--backend mlx` for `--backend vllm` and adjust the model. The full
requirements.txt with vLLM is the right env.

**Pilot first**: always pass `--limit 50` (or `100`) to validate the matrix
on a subset before running the full 2,665-instance pass. The full matrix
without a limit on CCE-Python alone is ~16,000 generations (6 configs ×
2,665, with cache reuse cutting it dramatically).

### 7.2 RepoEval data download

The loader [load_repoeval](src/adaptive_retrieval/eval/datasets.py) is
written but the data isn't downloaded. Needed for: secondary evaluation
tables and the CARD reproduction gate.

```bash
# Clone the RepoCoder repo to get datasets.zip + repositories archive
git clone https://github.com/microsoft/CodeT
cd CodeT/RepoCoder
# Follow the README to extract datasets.zip and the repositories archive
# Move/symlink to data/repoeval/{datasets,repositories}
```

When data is in place, `load_repoeval(task='line')` will work, and
`scripts/run_full_matrix.py --datasets repoeval_line` is unblocked.

### 7.3 Phase 7 — analysis numbers

Once Phase 6 has produced JSONL files at
`results/<dataset>/<config>.jsonl`:

```bash
python scripts/06_analysis.py \
    --c1 results/crosscodeeval_py/C1_no_retrieve.jsonl \
    --c2 results/crosscodeeval_py/C2_always_retrieve.jsonl \
    --c3 results/crosscodeeval_py/C3_card.jsonl \
    --c4 results/crosscodeeval_py/C4_cascade.jsonl \
    --output-dir analysis/cce
```

Produces `analysis/cce/summary.json` (every paper number) and
`analysis/cce/t_rag_sweep.png` (the threshold-sweep figure).

The aggregate logic is in
[src/adaptive_retrieval/eval/analysis.py](src/adaptive_retrieval/eval/analysis.py) — extend it
for any additional analyses the paper needs. Tests in [tests/test_analysis.py](tests/test_analysis.py).

### 7.4 Phase 8 — paper

[paper/main.tex](paper/main.tex) has section structure but `\todo{}`
placeholders. Each TODO maps to a specific field in
`analysis/<dataset>/summary.json`:

- `\todo{XX%}` for retrieval reduction → `aggregate[].percent_retrieval`
- `\todo{XX%}` for hallucination reduction → `mcnemar.c - mcnemar.b` / `mcnemar.n`
- McNemar `p < ` → `mcnemar.p_value`
- Headline tables → `aggregate[]`
- T_RAG sweep figure → `paper/figures/t_rag_sweep.pdf` (convert from PNG)

See [paper/README.md](paper/README.md) for the substitution recipe.

### 7.5 CARD paper reproduction (gate 3c)

**Requires CodeLlama-7B specifically + GPU**. The script exists
implicitly: run `01_construct_training_data.py` and the full matrix with
`--model codellama/CodeLlama-7b-hf` on RepoEval-line, then compare:

```
RepoEval-line CodeLlama-7B (CARD paper Table 3):
  Zero-shot: EM=33.94%  ES=59.42%
  RG1:       EM=52.31%  ES=71.83%
  CARD-RG1:  EM=52.56%  ES=72.26%

Our targets (within ±1% ES):
  CARD-RG1 ES ∈ [71.26%, 73.26%]
```

If outside that range: debug feature extraction first (most likely),
then Estimator hyperparameters, then prompt template.

### 7.6 Optional extensions (Week-4 nice-to-haves)

- Ablations A1/A2/A3 (static strictness, T_RAG sweep, top-k sweep). The
  current `run_full_matrix.py` doesn't loop over T_RAG or top-k — add a
  thin wrapper or run multiple matrix calls.
- CrossCodeLongEval (the natural-distribution counterpart to
  CrossCodeEval). Optional per the guide.
- RepoEval-function with UT (unit-test pass rate) metric instead of ES.

---

## 8. Conventions used in this codebase

**Style**

- Line length 100 (per [pyproject.toml](pyproject.toml)).
- Type hints, dataclasses where they help.
- Docstrings explain *why*, not *what*. Avoid restating obvious code.
- No emojis in code, comments, or output.
- Prefer imperative module docstrings over noun-phrase ones.

**Tests**

- One test file per module (`test_<module>.py`).
- Each test name describes the expected behaviour: `test_X_does_Y_when_Z`.
- Mocks live next to or inside the test files; `MockGenerator` and
  `MockEstimator` are project-wide and live in their respective modules.
- Use synthetic data with known structure rather than mocking statistical
  functions. The Estimator MSE test trains on real LightGBM with synthetic
  features that have a clear signal.
- Avoid changing tests to make them pass. Change the code instead, unless
  the test is testing a bug or an over-strict assertion (those tests
  should be relaxed with a comment explaining why).

**Commands**

- Run tests with the full path or after activation:
  `/Users/lucaionescu/miniconda3/envs/adaptive-retrieval/bin/pytest tests/`.
- Run ruff before committing: `ruff check src/ tests/ scripts/`.
- Don't auto-fix lint without inspecting — sometimes the warning points at
  a real bug. (`--fix` is safe for the f-string-without-placeholder family,
  which is what we used during this session.)
- Generate the project report after major changes:
  `python scripts/build_project_report.py`.

**What not to do**

- Don't add backwards-compat shims for code that hasn't been deployed.
- Don't write comments that describe what the code does — only why.
- Don't run destructive git operations without explicit user confirmation.
- Don't try to run vLLM locally (no macOS wheels).
- Don't run HFGenerator + mlx-lm in the same env (transformers conflict).

---

## 9. Active discussions / decisions pending user input

If the user picks up the project, these are the open questions to flag:

1. **Model choice for Phase 6**: Qwen2.5-Coder-0.5B (fast, ~5–10h locally),
   1.5B (better quality, ~8–20h), or 7B (paper-grade, needs GPU)?
   Discussed in [project_report.pdf](project_report.pdf) §5.3.

2. **CARD reproduction gate**: skip and rely on internal validation, or
   carve out GPU time for CodeLlama-7B on RepoEval-line? The reviewer-
   facing risk is non-trivial (see project_report §5.3).

3. **Ablation coverage**: minimum is RQ1 + RQ2 results on CCE-Python; the
   paper benefits from RepoEval-line/-api numbers and the A1/A2/A3
   ablations on CCE. With limited compute, prioritise.

4. **§16 Phase 1 manual-check #6 inconsistency**: the guide says
   "Attribute access with hallucinated attribute name → fires" but §C.6
   says we don't flag attribute names (only receivers). We followed §C.6.
   Worth surfacing if a reviewer raises it.

---

## 10. Final test runs as of this handoff

```
$ pytest tests/
167 passed, 13 warnings in 5.00s

$ ruff check src/ tests/ scripts/
All checks passed!

$ python scripts/00_smoke_test.py --estimator models/estimator_synthetic.lgb --t-rag 0.7
... 6 configs, all ES=1.0 on synthetic instance, no errors

$ python scripts/06_analysis.py --c1 results/phase5_smoke/C1_*.jsonl ... (see §7.3)
... wrote analysis/phase5_smoke/summary.json + t_rag_sweep.png
```

These should all still pass when you take over. If they don't, that's the
first thing to investigate.

---

## 11. Files to NOT touch without thinking

- `IMPLEMENTATION_GUIDE.md` — the project spec. Documented deviations live
  in the relevant modules, not in this file.
- `papers/` — read-only references (PDFs).
- `data/crosscodeeval/` — the .tar.xz and extracted dataset. Re-downloading
  costs 42 MB and a few minutes; not destructive but a waste.
- `models/estimator_synthetic.lgb` — used by Phase 5 smoke. Not real.
  Real Phase 6 produces `estimator_qwen25_05b.lgb` etc.

Everything in `results/phase5_smoke/` is synthetic data from the Phase 5
validation run. Safe to delete and regenerate.
