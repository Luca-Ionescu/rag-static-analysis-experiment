# Implementation Guide — three Colab notebooks for the cascade experiment

Audience: a capable coding agent (Claude 4.8 Max) building/running the notebooks. Read this top to bottom; it encodes decisions **and** the landmines already hit, so they aren't repeated.

## 0. What you're building

Three Colab notebooks, **one per generator**, each running the adaptive-retrieval cascade C1–C4 on **two datasets** with a **full `t_rag` sweep** and the **five metrics** (EM, ES, retrieval %, latency, hallucination). Model the notebooks on `notebooks/repoeval_function_qwen15.ipynb` (same REST-API git I/O, same cell flow) — you are *extending* it, not reinventing it.

| Generator | model id | `model_family` | estimator |
|---|---|---|---|
| CodeLlama-7B | `codellama/CodeLlama-7b-hf` | `codellama` | reuse `models/estimator_codellama_7b.lgb` |
| Qwen2.5-Coder-1.5B | `Qwen/Qwen2.5-Coder-1.5B` | `qwen` | reuse `models/estimator_qwen25_1.5b.lgb` |
| Qwen2.5-Coder-0.5B | `Qwen/Qwen2.5-Coder-0.5B` | `qwen` | **calibrate fresh** → `models/estimator_qwen25_0.5b.lgb` |

Datasets: `crosscodeeval_py` and `repoeval_function`. Configs: C1 no-retrieve, C2 always-retrieve, C3 CARD, C4 cascade.

## 1. Finalized decisions (do not re-litigate)

- **0.5B = Qwen2.5-Coder-0.5B** (FIM-capable code model; the plain general model would break FIM infilling).
- **`max_tokens`: CCE = 50, RepoEval-function = 280.**
- **RepoEval-function is scored BOTH ways:** truncated to the function body *and* non-truncated (raw). Save and report both. (CCE is always first-line.)
- **Fresh regeneration** of everything with the current code (stop-strings, full-pipeline latency, pyflakes trigger, A4∧B2). The old cached 7B/1.5B results are stale — ignore them.
- **Hallucination metric = A4∧B2 (pyflakes)**; also report **A4-only** for transparency. Accept the trigger=metric circularity (no Pyright).
- **`t_rag` sweep = 0.05 → 0.95 step 0.05**, report every value.
- **Static checker = pyflakes** (no AST/Tiers/symbol-table).
- **GPU**: 7B needs A100/L4 (Pro+); 1.5B/0.5B fit T4.

## 2. Pipeline shape — what is GPU vs CPU (this is the key efficiency idea)

The `t_rag` sweep is **free**: ŝ₀ and the generations are frozen, so CARD/cascade at any threshold are pure replays. So:

- **GPU, once per (generator, dataset):** generate **C1, C2, C3** with `04_run_experiment.py`. C1 = zero-shot, C2 = always-retrieve, **C3 is run only to store ŝ₀** (its zero-shot prompt is a cache hit on C1, so it's cheap). You do **not** run C4 on GPU.
- **CPU, post-hoc:** `scripts/13_sweep_eval.py` replays **C3 (CARD) and C4 (cascade)** across the whole `t_rag` grid, computes all five metrics, and (for RepoEval) scores both truncation modes. This is the engine; it is already written and tested.

So per notebook the GPU work is 3 generators-agnostic passes × 2 datasets = generate C1/C2/C3 twice; the 0.5B notebook additionally calibrates.

## 3. Notebook cell flow (extends the example)

1. **Config** — `MODEL`, `MODEL_FAMILY`, `ESTIMATOR`, `DATASETS=['crosscodeeval_py','repoeval_function']`, `MAX_TOKENS={'crosscodeeval_py':50,'repoeval_function':280}`, `T_GRID="0.05,…,0.95"`, `TOP_K=10`, `SMOKE` flag, git settings (REST API, `LUCA_GITHUB_PAT`, `colab-results` branch). One generation `T_RAG` for the C3 ŝ₀ pass (value irrelevant — use 0.5).
2. **GPU sanity** (`nvidia-smi`).
3. **GitHub REST helpers** — copy verbatim from the example (PAT from Colab secret; `_gh_req`, `gh_upload`, branch creation). No `git clone`/`push`.
4. **Pull repo** via tarball REST API (verbatim from example).
5. **Install deps** — example list **+ `pyflakes`** (already in the example's list; confirm). Add `datasets` for the 0.5B calibration notebook.
6. **Provision data:**
   - **CCE**: `gunzip scripts/runpod/assets/cce_python_rg1_bm25.jsonl.gz → data/crosscodeeval/crosscodeeval_data/python/line_completion_rg1_bm25.jsonl` (assert 2665 lines).
   - **RepoEval**: clone `microsoft/CodeT` sparse, unzip `RepoCoder/datasets/datasets.zip → data/repoeval/datasets` and `RepoCoder/repositories/function_level.zip → data/repoeval/repositories` (verbatim from example cell 12).
7. **[0.5B notebook only] Calibrate estimator** (see §5).
8. **Verify loaders lossless** — example cell 14 for RepoEval; add a quick CCE load check.
9. **Results branch + `gh_upload`** (verbatim).
10. **Clear generation cache** (verbatim — critical: the cache key omits sampling params, so a stale cache would serve pre-stop-string generations).
11. **Generate C1/C2/C3 per dataset** — loop over `DATASETS`; for each, run `04_run_experiment.py` for `C1_no_retrieve`, `C2_always_retrieve`, `C3_card` with that dataset's `MAX_TOKENS`, `--backend vllm`, the estimator for C3, `--cache-dir data/generation_cache`, a **per-(generator,dataset) results subdir**. Push each JSONL as it finishes. (Skip C4 on GPU.)
12. **Sweep + metrics per dataset** — run `scripts/13_sweep_eval.py --results-dir <subdir> --dataset <ds> --out-csv <subdir>/sweep.csv`. It emits the long-format CSV (all t, both RepoEval modes). Push the CSV.
13. **Summary** — print/save the sweep CSVs; link the `colab-results` branch.

## 4. The scripts the notebooks call (contracts)

**Generation —** `scripts/04_run_experiment.py` (one call per config×dataset):
```
--config {C1_no_retrieve|C2_always_retrieve|C3_card} --dataset {crosscodeeval_py|repoeval_function}
--backend vllm --model <MODEL> --model-family <FAMILY> --max-tokens <50|280>
--t-rag 0.5 --top-k 10 --estimator-path <ESTIMATOR>   # estimator only for C3
--output <subdir>/<config>.jsonl --cache-dir data/generation_cache  [--limit N for SMOKE]
```

**Sweep + metrics —** `scripts/13_sweep_eval.py` (one call per dataset):
```
--results-dir <subdir> --dataset <ds> --out-csv <subdir>/sweep.csv
# default --t-grid is 0.05..0.95 step 0.05
```
Output CSV columns: `dataset, scoring, config, t_rag, retrieval_pct, exact_match, edit_similarity, identifier_f1, hall_A4B2, hall_A4, latency_ms`.
- `scoring` ∈ {`line`} for CCE, {`body`,`full`} for RepoEval-function (the two modes).
- `config` ∈ {C1, C2 (t_rag blank), C3_card, C4_cascade (one row per t)}.
- `latency_ms` is synthesized: baselines = one generation; CARD/cascade = zero-shot probe + conditional retrieved gen (from the cached per-instance generation latencies).

## 5. 0.5B estimator calibration (extra cells in the 0.5B notebook only)

Calibrate on the-stack-dedup, **reuse the one `.lgb` for both datasets** (as done for 7B/1.5B):
```
python scripts/01_construct_training_data.py --source the-stack-dedup --file-limit 15000 \
  --backend vllm --model Qwen/Qwen2.5-Coder-0.5B --model-family qwen --max-tokens 50 \
  --n-pairs 250000 --per-file 25 --batch-size 256 --min-files 8000 --min-pairs 20000 \
  --output data/training_data/qwen25_0.5b.npz
python scripts/02_train_estimator.py --data data/training_data/qwen25_0.5b.npz \
  --output models/estimator_qwen25_0.5b.lgb --num-boost-round 100
```
Gotchas: the-stack-dedup is **gated** (accept the license on the PAT/HF account); calibration uses the **dedup-speed fix** already in `train_data.py` (`n_init=1`, `max_clusters=30_000`) so it finishes in ~30 min not hours; the `--min-pairs`/held-out-MSE (`--min-skill`) guardrails must pass or it aborts by design. Push the `.lgb` to the results branch.

**SMOKE-aware calibration:** the calibration cell branches on `SMOKE`. `SMOKE=True` uses the tiny **public** `the-stack-smol` sample with relaxed guards (`--min-files 40 --min-pairs 400 --min-skill -1.0`, ~few min, no `HF_TOKEN` needed) and writes to **isolated `_smoke` paths** (`estimator_qwen25_0.5b_smoke.lgb`, reassigning the `ESTIMATOR` global so downstream C3 uses it) — a rough estimator for wiring only, **not pushed**. `SMOKE=False` does the real gated calibration to the canonical paths and pushes. So a smoke pass never clobbers or gets reused by a later full run.

## 6. The careful bits / gotchas (why each matters)

- **Per-dataset `max_tokens`** — 50 truncates CCE to its single line; 280 covers ~p99 of RepoEval-function bodies. Wrong values cut bodies or waste compute.
- **Stop-strings** are in `generator.py` (`stop=_FIM_STOP_STRINGS`, Qwen+CodeLlama). They keep generations clean; the metric scripts also strip FIM tails as a backstop. **Always clear the cache** before a run or you serve pre-stop-string garbage.
- **Dataset-aware scoring** is keyed by `datasets.MULTILINE_DATASETS = {"repoeval_function"}`: CCE → first line, RepoEval → `truncate_to_function_body` (mode `body`) and raw (mode `full`). `13_sweep_eval.py` handles both automatically.
- **ŝ₀ scale is estimator-specific** (7B ≈ [0.05,0.47], 1.5B ≈ [0.29,0.50], 0.5B unknown). That's *why* you sweep the whole 0.05–0.95 grid rather than fix one threshold — some thresholds will be degenerate (0% or 100% retrieval) for a given generator; the non-degenerate band (read `retrieval_pct`) is where the CARD-vs-cascade contrast lives. The cascade's retrieval floor is the pyflakes trigger rate (higher for smaller models).
- **Hallucination = A4∧B2** (pyflakes "undefined name" ∩ "absent from gold") — `metrics.invented_identifier_flag`. Report `hall_A4` (gold-only, fully independent) beside it. **Circularity caveat:** trigger and B2 are both pyflakes, so the cascade's drop is partly structural — state this; `hall_A4` is the independent number.
- **Latency = full pipeline** (`card_pipeline`/`cascade_pipeline`/baselines now time gen + estimator + gate + retrieval, cache-robust via `Generation.latency_ms`). In the sweep it's synthesized from C1/C2 gen latencies; for an exact figure at one operating point, run the runner's C3/C4 at that `t_rag` (small CPU overhead it adds).
- **Expect near-zero hallucination on the 7B/1.5B** (~0.6–1.1% A4∧B2). The 0.5B should hallucinate more — that's the point of including it.
- **GPU sizing**: 7B notebook must use A100/L4; T4 will OOM on bf16 7B + KV cache.

## 7. Outputs (per notebook)

For each `(generator, dataset)`, pushed to the `colab-results` branch under `results/<gen>_<dataset>/`:
- `C1_no_retrieve.jsonl`, `C2_always_retrieve.jsonl`, `C3_card.jsonl` (generations + ŝ₀)
- `sweep.csv` (the headline artifact: every metric × every `t_rag` × scoring mode)
- 0.5B notebook also pushes `models/estimator_qwen25_0.5b.lgb` and the `.npz`.

## 8. Caveats to carry into the write-up

1. Hallucination is rare under a correct metric on capable models → RQ2 contrast is small; the 0.5B is where it should show.
2. Trigger=metric (both pyflakes) makes the cascade's hallucination reduction partly structural; `hall_A4` is the independent check.
3. Estimator ŝ₀ is compressed (calibrated on un-truncated ES) → operating points read off `retrieval_pct`, not the raw threshold.
4. Latency in the sweep is generation-synthesized (excludes ~ms of estimator/gate/BM25 CPU); a fresh runner run captures it exactly.
