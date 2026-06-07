# Analysis — Qwen2.5-Coder-1.5B (RepoEval-function + CrossCodeEval-Python)

Generator: **Qwen/Qwen2.5-Coder-1.5B** (vLLM, FIM, greedy, max_tokens=280, stop-tokens on).
Configs: **C1** no-retrieve, **C2** always-retrieve (BM25 top-10), **C3** CARD gate, **C4** cascade (CARD + 3-tier static gate). C3/C4 swept over T_RAG ∈ [0.05, 0.95].
Metrics: **EM** exact match, **ES** edit similarity, **IdF1** identifier-F1, **hall_A4** = emitted identifier absent from gold (reference-based, upper bound), **hall_A4B2** = invented identifier (absent from gold AND resolves nowhere — the strict "made-up name" signal), **retrieval%**, **latency** (ms/instance).
Run completed 2026-06-07 ~02:55 CEST. Datasets: CrossCodeEval-Python (line, 2665) and RepoEval-function (455; scored two ways — **body** = dedent-truncated, the headline; **full** = untruncated).

---

## 1. Headline (endpoints: C1 vs C2)

### CrossCodeEval-Python (line completion)
| Config | Retr% | EM | ES | IdF1 | hall_A4B2 | hall_A4 | Latency |
|---|--:|--:|--:|--:|--:|--:|--:|
| C1 no-retrieve     | 0.0   | 0.294 | 0.631 | 0.603 | 0.0113 | 0.582 | 200 |
| C2 always-retrieve | 100.0 | **0.918** | **0.961** | **0.958** | **0.0011** | **0.061** | 234 |

### RepoEval-function (body-scored — headline)
| Config | Retr% | EM | ES | IdF1 | hall_A4B2 | hall_A4 | Latency |
|---|--:|--:|--:|--:|--:|--:|--:|
| C1 no-retrieve     | 0.0   | 0.055 | 0.431 | 0.585 | 0.0703 | 0.730 | 1106 |
| C2 always-retrieve | 100.0 | **0.624** | **0.888** | **0.933** | **0.0176** | **0.156** | 1399 |

**Retrieval is decisive on both.** On CCE, ES 0.63→0.96 and EM 0.29→0.92. On RepoEval-function (body), ES 0.43→0.89 and EM 0.055→0.624 — an 11× EM jump. This confirms the C1 baseline is "correctly low" (knowledge-limited), not broken: when the cross-file context is supplied, the small model completes well.

---

## 2. The cascade vs CARD (C4 vs C3) — does the static gate help?

The interesting region is **low T_RAG**, where CARD retrieves little/nothing and the static gate can add value. At high T_RAG both gates retrieve everything and C3≡C4.

### RepoEval-function (body), selected T_RAG
| T_RAG | C3 retr% | C3 ES | C3 hallA4B2 | C4 retr% | C4 ES | C4 hallA4B2 |
|--:|--:|--:|--:|--:|--:|--:|
| 0.05–0.25 | 0.0 | 0.431 | 0.0703 | **7.9** | **0.466** | **0.0044** |
| 0.35 | 12.7 | 0.502 | 0.0637 | **20.0** | **0.533** | **0.0044** |
| 0.40 | 57.1 | 0.739 | 0.0352 | 59.8 | 0.749 | **0.0154** |
| 0.45 | 88.4 | 0.846 | 0.0198 | 89.0 | 0.847 | 0.0176 |
| ≥0.50 | 100 | 0.888 | 0.0176 | 100 | 0.888 | 0.0176 |

### CrossCodeEval (line), selected T_RAG
| T_RAG | C3 retr% | C3 ES | C3 hallA4B2 | C4 retr% | C4 ES | C4 hallA4B2 |
|--:|--:|--:|--:|--:|--:|--:|
| 0.05–0.25 | 0.0 | 0.631 | 0.0113 | **1.2** | **0.638** | **0.0004** |
| 0.30 | 7.4 | 0.669 | 0.0090 | 8.3 | 0.675 | **0.0004** |
| 0.35 | 33.1 | 0.792 | 0.0068 | 33.8 | 0.796 | **0.0004** |
| 0.40 | 66.0 | 0.905 | 0.0023 | 66.3 | 0.906 | **0.0004** |

**Findings.**
1. **The static gate adds retrievals exactly where CARD is silent.** At low T_RAG (CARD retr%=0), C4 fires on **7.9%** of RepoEval-function instances and **1.2%** of CCE — the instances whose zero-shot output contains an out-of-scope/invalid identifier.
2. **Those extra retrievals are high-value for hallucination.** The strict invented-identifier rate (hall_A4B2) drops sharply: RepoEval-function **0.070→0.004** (16×↓), CCE **0.011→0.0004** (28×↓) at low T_RAG. This is the core RQ2 result: the cascade removes nearly all "made-up name" hallucinations CARD's confidence gate misses.
3. **Accuracy also nudges up** for a tiny retrieval budget: RepoEval-function ES +0.035 (0.431→0.466) and EM +0.035 (0.055→0.090) at ~8% retrieval; CCE ES +0.007 at ~1% retrieval.
4. **Asymmetry holds:** C4 retr% ≥ C3 retr% everywhere, and C4 ES ≥ C3 ES — the static stage only ever adds retrievals, never hurts.

---

## 3. Hallucination metric behavior

- **hall_A4 (reference-based, loose)** tracks accuracy inversely: C1 RepoEval-function 0.730 → C2 0.156. It's an upper bound (counts any non-gold identifier, incl. valid alternatives).
- **hall_A4B2 (invented, strict)** is the meaningful signal — it's already low at C1 (0.070 body / 0.011 CCE) because most non-gold identifiers *do* resolve; the cascade drives it toward **~0** at low retrieval cost.
- The cascade's effect is **much larger on hall_A4B2 than on ES/EM** — i.e. its primary benefit is removing genuine hallucinations, not raw accuracy. This is the intended contribution.

---

## 4. Efficiency (latency & retrieval cost)

| | C1 | C2 | C4 @ low T_RAG |
|---|--:|--:|--:|
| RepoEval-function latency (ms) | 1106 | 1399 | ~1250 (7.9% retr) |
| CCE latency (ms) | 200 | 234 | ~204 (1.2% retr) |

The cascade buys most of the hallucination reduction for a **single-digit % retrieval budget** — near-C1 latency, far below C2's always-retrieve cost. This is the cost/quality sweet spot the framework targets.

---

## 5. body vs full scoring (RepoEval-function)

The dedent **body** truncation matters: C2 ES is **0.888 (body)** vs **0.615 (full)** — the untruncated scoring is dragged down by over-generation past the function body. Headline numbers use **body**; full is reported for completeness. (C1 body 0.431 vs full 0.374.)

---

## 6. Takeaways
- **1.5B + retrieval works**: ES ~0.89 (RepoEval-fn) / ~0.96 (CCE) with always-retrieve.
- **Cascade (C4) ≈ matches CARD on accuracy while nearly eliminating invented-identifier hallucinations** at low retrieval cost — the central hypothesis, supported on both datasets.
- The static gate's value concentrates at **low T_RAG** (where CARD is conservative); at high T_RAG the two converge, as designed.
- Note: at 1.5B, CARD's gate is coarse (retr% jumps 0→100 across a narrow T_RAG band 0.3–0.5) — the 7B run should give a smoother operating curve.

_Generated by the Colab monitor after the 1.5B run completed both datasets._
