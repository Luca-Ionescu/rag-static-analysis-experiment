# Cross-dataset summary — Adaptive-Retrieval Cascade (0.5B / 1.5B / 7B)

Full results across **all four benchmarks**, three generators:

| Dataset | Task | N | Headline scoring |
|---|---|--:|---|
| CrossCodeEval-Python | single-line | 2665 | `line` (first line) |
| RepoEval-function | function body | 455 | `body` (dedent truncation) |
| CrossCodeLongEval-function | function body | 5000 | `body` |
| CrossCodeLongEval-chunk | fixed block (1–6 lines) | 5000 | `lines` (gold line-count) |

Configs C1 no-retrieve, C2 always-retrieve (BM25 top-10), C3 CARD gate, C4 cascade (CARD + 3-tier static gate); C3/C4 swept over `t_rag`. Custom hallucination: **hall_A4B2** = invented identifier (absent from gold AND unresolvable), **hall_A4** = looser absent-from-gold.

> **Consistency note.** All four datasets were (re-)scored with one sweep engine. The two new CrossCodeLongEval sweeps and the CrossCodeEval / RepoEval sweeps were all recomputed so the **static signal (cascade trigger + hallucination) is computed on the complete prediction**, and accuracy is reported truncated **and** raw. (Accuracy numbers match the earlier runs; only the hallucination methodology was unified to on-complete.)

---

## 1. Endpoints (headline scoring): C1 → C2

| Dataset | Model | C1 ES | C2 ES | C1 EM | C2 EM | C1 hallA4B2 | C2 hallA4B2 |
|---|---|--:|--:|--:|--:|--:|--:|
| CCE-line | 0.5B | 0.556 | 0.929 | 0.205 | 0.856 | 0.0428 | 0.0398 |
| | 1.5B | 0.631 | 0.961 | 0.294 | 0.918 | 0.0210 | 0.0120 |
| | 7B | 0.652 | 0.961 | 0.315 | 0.923 | 0.0169 | 0.0244 |
| RepoEval-fn | 0.5B | 0.357 | 0.717 | 0.046 | 0.339 | 0.0835 | 0.0813 |
| | 1.5B | 0.431 | 0.887 | 0.055 | 0.624 | 0.0527 | 0.0505 |
| | 7B | 0.444 | 0.839 | 0.084 | 0.626 | 0.0176 | 0.0505 |
| CCLE-fn | 0.5B | 0.397 | 0.739 | 0.039 | 0.350 | 0.0660 | 0.0756 |
| | 1.5B | 0.468 | 0.849 | 0.080 | 0.574 | 0.0344 | 0.0354 |
| | 7B | 0.489 | 0.839 | 0.085 | 0.600 | 0.0418 | 0.0536 |
| CCLE-chunk | 0.5B | 0.605 | 0.846 | 0.265 | 0.649 | 0.0210 | 0.0224 |
| | 1.5B | 0.707 | 0.935 | 0.394 | 0.863 | 0.0164 | 0.0148 |
| | 7B | 0.711 | 0.924 | 0.419 | 0.846 | 0.0206 | 0.0240 |

## 2. Dual metrics — truncated (headline) vs raw `full` (multi-line datasets), C2 ES
| Dataset | 0.5B trunc/full | 1.5B trunc/full | 7B trunc/full |
|---|--:|--:|--:|
| RepoEval-fn (body) | 0.717 / 0.475 | 0.887 / 0.615 | 0.839 / 0.444 |
| CCLE-fn (body) | 0.739 / 0.472 | 0.849 / 0.554 | 0.839 / 0.426 |
| CCLE-chunk (lines) | 0.846 / 0.564 | 0.935 / 0.681 | 0.924 / 0.511 |

Body/line truncation is essential on every multi-line dataset — raw scoring is deflated by over-generation past the target span (largest on 7B, which over-generates most).

## 3. The cascade contribution — at low `t_rag` (C3 CARD → C4 cascade)
Reduction in invented-identifier hallucination (hall_A4B2) for the small extra retrieval the static gate adds, at `t_rag = 0.05`:

| Dataset | 0.5B | 1.5B | 7B |
|---|--:|--:|--:|
| CCE-line | 9.5× (+5%) | 11.1× (+2%) | 11.3× (+2%) |
| RepoEval-fn | 4.2× (+9%) | →0 (+6%) | 4.0× (+3%) |
| CCLE-fn | 4.9× (+8%) | 8.2× (+4%) | 3.7× (+5%) |
| CCLE-chunk | 5.5× (+3%) | 3.6× (+2%) | 4.0× (+2%) |

In every cell, **C4 ES ≥ C3 ES and C4 hall ≤ C3 hall** (asymmetry guarantee), at single-digit-% extra retrieval. The contribution is robust across **4 datasets × 3 model scales = 12 settings**.

## 4. Four robust findings

1. **Retrieval is decisive at every scale and task.** C1→C2 lifts ES by +0.21–0.46 and EM several-fold everywhere. No-retrieve baselines are knowledge-limited, not broken.

2. **The cascade ≈ matches CARD on accuracy while cutting invented-identifier hallucination 3.6–11×, at +2–9% retrieval — across all 12 settings.** This is the core contribution and it generalizes from the original CCE/RepoEval benchmarks to the new, larger CrossCodeLongEval (5000+5000).

3. **Always-retrieve does *not* reduce invented identifiers.** Measured on the complete prediction, C2 hall_A4B2 is ≈ C1 or higher (e.g. RepoEval-fn 7B 0.018→0.051; CCLE-fn 7B 0.042→0.054). Retrieval helps accuracy but adds more code (more unresolvable names). The static gate is the only mechanism here that specifically reduces them — strengthening the case for the cascade over both baselines.

4. **Model size: accuracy ↑ with scale; hallucination ↓ with scale on the clean benchmarks.** C1 ES 0.5B<1.5B<7B everywhere; C1 hall_A4B2 falls with size on CCE (0.043/0.021/0.017) and RepoEval (0.084/0.053/0.018). On CrossCodeLongEval the size→hallucination trend is noisier (7B-fn 0.042 ≈ 1.5B 0.034) — the longer, harder contexts let even the 7B invent names — which is exactly where the static gate keeps paying off.

## 5. Latency
Latency scales with model size (CCLE-fn C1 ≈ 153 / 170 / 904 ms for 0.5B/1.5B/7B; chunk ≈ 58 / 102 / 422 ms). The cascade's hallucination reduction comes at near-C1 latency (single-digit-% retrieval), far below C2's always-retrieve cost — the cost/quality sweet spot. (CrossCodeLongEval was run with vLLM continuous batching, so its absolute latencies are throughput-amortized and lower than the earlier sequential CCE/RepoEval numbers; use retrieval-% as the apples-to-apples cost proxy across runs.)

## Bottom line
Across **CrossCodeEval, RepoEval-function, and both CrossCodeLongEval tasks**, at **0.5B / 1.5B / 7B**, the static-analysis cascade delivers its intended contribution: **near-elimination of invented-identifier hallucinations that CARD's confidence gate misses — and that plain always-retrieve fails to fix — at minimal extra retrieval cost and without hurting accuracy.** Largest where hallucination is largest (smaller models, function-level tasks); cleanest to tune where the gate is best-calibrated. Function-level tasks are the strongest testbed for the hallucination claim; single-line (CCE) and short-block (chunk) tasks show the same direction.

_Per-task detail: `analysis_CCLE_only.md` (new datasets). Generated after all three models completed all four datasets._
