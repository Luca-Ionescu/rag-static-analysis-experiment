# CrossCodeLongEval — Adaptive-Retrieval Cascade (0.5B / 1.5B / 7B)

New benchmark added this run: **CrossCodeLongEval** (Repoformer, ICML 2024), Python, **5000 instances each** for two tasks:
- **function** — multi-line function-body completion (the hallucination-relevant case);
- **chunk** — fixed-size block completion (1–6 gold lines).

Generators: Qwen2.5-Coder-0.5B / 1.5B, CodeLlama-7B. Configs: **C1** no-retrieve, **C2** always-retrieve (BM25 top-10), **C3** CARD uncertainty gate, **C4** cascade (CARD + 3-tier static gate); C3/C4 swept over `t_rag` (0.05–0.95). All runs use vLLM continuous batching on A100-80GB.

## Scoring (dual metrics, per your spec)
Every accuracy metric is reported **two ways**, for **both** datasets:
- **function** — `body` (prediction truncated to the generated function body at the dedent boundary; the headline) **and** `full` (raw, FIM-stripped).
- **chunk** — `lines` (prediction truncated to the gold's non-empty line count, matching Repoformer's chunk metric; the headline) **and** `full` (raw).

The **static-analysis signal** (cascade trigger + hallucination metric) is computed **once on the complete, untruncated prediction** — never on the truncated text. Rationale: truncating an over-generated span to an oracle length can delete definitions the rest of the file uses (manufacturing/masking undefined-name flags) and the gold length isn't available at inference. So accuracy is scored on the target span while the model's *actual* output is what's judged for hallucination.

Hallucination: **hall_A4B2** = invented identifier (absent from gold **and** unresolvable by pyflakes) — the strict "made-up name" signal; **hall_A4** = looser absent-from-gold rate.

---

## 1. Endpoints — retrieval helps at every scale (headline scoring)

### function (`body`) — ES / EM / hall_A4B2
| Model | C1 ES | C2 ES | C1 EM | C2 EM | C1 hallA4B2 | C2 hallA4B2 |
|---|--:|--:|--:|--:|--:|--:|
| 0.5B | 0.397 | 0.739 | 0.039 | 0.350 | 0.0660 | 0.0756 |
| 1.5B | 0.468 | 0.849 | 0.080 | 0.574 | 0.0344 | 0.0354 |
| 7B   | 0.489 | 0.839 | 0.085 | 0.600 | 0.0418 | 0.0536 |

### chunk (`lines`) — ES / EM / hall_A4B2
| Model | C1 ES | C2 ES | C1 EM | C2 EM | C1 hallA4B2 | C2 hallA4B2 |
|---|--:|--:|--:|--:|--:|--:|
| 0.5B | 0.605 | 0.846 | 0.265 | 0.649 | 0.0210 | 0.0224 |
| 1.5B | 0.707 | 0.935 | 0.394 | 0.863 | 0.0164 | 0.0148 |
| 7B   | 0.711 | 0.924 | 0.419 | 0.846 | 0.0206 | 0.0240 |

Retrieval (C1→C2) lifts function ES by **+0.34–0.38** and chunk ES by **+0.21–0.24**; EM jumps several-fold. The no-retrieve baselines are "correctly low" (knowledge-limited), not broken — verified by inspecting predictions (coherent code, wrong because the cross-file context is missing).

## 2. Truncated vs raw (dual metrics) — truncation is essential
C2 ES, headline (`body`/`lines`) **vs** `full` (raw):

| Model | function body | function full | chunk lines | chunk full |
|---|--:|--:|--:|--:|
| 0.5B | 0.739 | 0.472 | 0.846 | 0.564 |
| 1.5B | 0.849 | 0.554 | 0.935 | 0.681 |
| 7B   | 0.839 | 0.426 | 0.924 | 0.511 |

The raw view is heavily deflated by over-generation past the target span (e.g. 7B function C2 **0.839 body → 0.426 full**). This is why the headline uses span-matched truncation (function body / chunk gold-line-count); `full` is retained for transparency.

## 3. The cascade contribution — replicates cleanly on CrossCodeLongEval
At low `t_rag` CARD (C3) is conservative and retrieves ~nothing, so its zero-shot output carries the model's invented identifiers. The **static gate (C4) adds a few % of retrieval and drives hall_A4B2 down ~4–8×, while ES slightly *increases*** (`t_rag = 0.05`):

### function
| Model | C3 retr → C4 retr | C3 hallA4B2 → C4 | reduction | ES (C4 vs C3) |
|---|--:|--:|--:|--:|
| 0.5B | 0% → 8%  | 0.0660 → 0.0134 | **4.9×** | 0.424 vs 0.397 |
| 1.5B | 0% → 4%  | 0.0344 → 0.0042 | **8.2×** | 0.489 vs 0.468 |
| 7B   | 0% → 5%  | 0.0418 → 0.0114 | **3.7×** | 0.504 vs 0.489 |

### chunk
| Model | C3 retr → C4 retr | C3 hallA4B2 → C4 | reduction | ES (C4 vs C3) |
|---|--:|--:|--:|--:|
| 0.5B | 0% → 3% | 0.0210 → 0.0038 | **5.5×** | 0.613 vs 0.605 |
| 1.5B | 0% → 2% | 0.0164 → 0.0046 | **3.6×** | 0.713 vs 0.707 |
| 7B   | 0% → 2% | 0.0206 → 0.0052 | **4.0×** | 0.718 vs 0.711 |

The **asymmetry guarantee holds at every `t_rag`** (C4 retrieval ≥ C3, C4 ES ≥ C3, C4 hall ≤ C3) for both datasets, all three models.

## 4. A telling negative result — always-retrieve does *not* fix invented identifiers
On the complete prediction, **C2 (always-retrieve) hallucination is ≈ C1 or slightly higher** (function: 0.5B 0.066→0.076, 7B 0.042→0.054). Retrieval improves *accuracy* but lets the model write more/longer code, introducing more unresolvable names. **Only the static gate specifically targets and reduces invented identifiers** — which is precisely the cascade's contribution: it removes a failure mode that neither raw retrieval nor the confidence gate addresses.

## 5. Latency (per-instance, batched inference)
| Model | function C1 / C2 | chunk C1 / C2 |
|---|--:|--:|
| 0.5B | 153 / 186 ms | 58 / 96 ms |
| 1.5B | 170 / 262 ms | 102 / 174 ms |
| 7B   | 904 / 1594 ms | 422 / 725 ms |

Latency scales with model size; C2 > C1 (retrieval adds prefill). The cascade buys its hallucination reduction at **single-digit-% extra retrieval** → near-C1 latency, far below C2's always-retrieve cost. (These are batched-throughput latencies, not directly comparable to the earlier sequential per-instance numbers.)

## Bottom line (CrossCodeLongEval)
On both new tasks and all three scales, the static-analysis cascade **near-eliminates invented-identifier hallucinations that the CARD confidence gate misses (3.6–8.2×), at +2–8% retrieval, without hurting accuracy** — and does so where plain always-retrieve actually leaves hallucination flat-to-higher. The effect is largest where hallucination is largest (smaller models / function task). Function is the stronger testbed for the hallucination story (clean semantic boundary); chunk shows the same direction at smaller magnitude.

_Generated after all three CrossCodeLongEval runs completed both datasets (0.5B/1.5B/7B, A100-80GB, batched)._
