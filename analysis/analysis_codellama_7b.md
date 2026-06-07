# Analysis — CodeLlama-7B (RepoEval-function + CrossCodeEval-Python)

Generator: **codellama/CodeLlama-7b-hf** (vLLM, FIM, greedy, max_tokens=280, stop-tokens on).
Configs: **C1** no-retrieve, **C2** always-retrieve (BM25 top-10), **C3** CARD gate, **C4** cascade (CARD + 3-tier static gate). C3/C4 swept over T_RAG ∈ [0.05, 0.95].
Metrics: **EM**, **ES**, **IdF1**, **hall_A4** (emitted id absent from gold — loose upper bound), **hall_A4B2** (invented id: absent from gold AND resolves nowhere — strict "made-up name"), **retrieval%**, **latency** (ms/instance).
Completed 2026-06-07 ~04:19 CEST. Datasets: CrossCodeEval-Python (line, 2665) + RepoEval-function (455; **body**=dedent-truncated headline, **full**=untruncated).

---

## 1. Headline (C1 vs C2)

### CrossCodeEval-Python (line)
| Config | Retr% | EM | ES | IdF1 | hall_A4B2 | hall_A4 | Latency |
|---|--:|--:|--:|--:|--:|--:|--:|
| C1 no-retrieve     | 0   | 0.315 | 0.652 | 0.626 | 0.0060 | 0.541 | 502 |
| C2 always-retrieve | 100 | **0.923** | **0.961** | **0.958** | **0.0019** | **0.055** | 844 |

### RepoEval-function (body — headline)
| Config | Retr% | EM | ES | IdF1 | hall_A4B2 | hall_A4 | Latency |
|---|--:|--:|--:|--:|--:|--:|--:|
| C1 no-retrieve     | 0   | 0.084 | 0.444 | 0.559 | 0.0242 | 0.706 | 3607 |
| C2 always-retrieve | 100 | **0.626** | **0.839** | **0.865** | 0.0154 | **0.226** | 4916 |

Retrieval is again decisive: CCE ES 0.65→0.96, EM 0.32→0.92; RepoEval-fn ES 0.44→0.84, EM 0.084→0.626 (7.5× EM).

---

## 2. Cascade (C4) vs CARD (C3) — the static gate

### CrossCodeEval (line), selected T_RAG
| T_RAG | C3 retr% | C3 ES | C3 hallA4B2 | C4 retr% | C4 ES | C4 hallA4B2 |
|--:|--:|--:|--:|--:|--:|--:|
| 0.05 | 0.0 | 0.652 | 0.0060 | **0.7** | 0.656 | **0.0004** |
| 0.10 | 2.1 | 0.662 | 0.0060 | 2.8 | 0.666 | **0.0004** |
| 0.15 | 7.4 | 0.688 | 0.0060 | 8.0 | 0.692 | **0.0004** |
| 0.20 | 17.7 | 0.733 | 0.0049 | 18.3 | 0.736 | **0.0004** |
| 0.25 | 30.0 | 0.790 | 0.0041 | 30.4 | 0.792 | 0.0015 |
| 0.30 | 59.6 | 0.901 | 0.0030 | 59.8 | 0.901 | 0.0019 |

### RepoEval-function (body), selected T_RAG
| T_RAG | C3 retr% | C3 ES | C3 hallA4B2 | C4 retr% | C4 ES | C4 hallA4B2 |
|--:|--:|--:|--:|--:|--:|--:|
| 0.05 | 0.0 | 0.444 | 0.0242 | **3.7** | 0.461 | **0.0088** |
| 0.10 | 2.0 | 0.461 | 0.0242 | 5.7 | 0.478 | **0.0088** |
| 0.20 | 3.1 | 0.470 | 0.0264 | 6.8 | 0.487 | 0.0110 |
| 0.25 | 5.5 | 0.485 | 0.0264 | 9.0 | 0.501 | 0.0110 |
| 0.30 | 53.8 | 0.683 | 0.0198 | 55.8 | 0.690 | 0.0132 |
| 0.35 | 76.9 | 0.793 | 0.0154 | 77.8 | 0.797 | 0.0132 |

**Findings.**
1. **Static gate adds retrievals where CARD is silent/conservative** and consistently **lifts ES and cuts invented-identifier hallucination**. At T_RAG=0.05: CCE hall_A4B2 0.0060→0.0004 (15×↓) for +0.7% retrieval; RepoEval-fn 0.0242→0.0088 (2.7×↓) for +3.7% retrieval.
2. **Effect persists across the low/mid range** (the whole 0.05–0.30 band), not just one point — C4 ES ≥ C3 ES and C4 hall_A4B2 ≤ C3 hall_A4B2 at every T_RAG.
3. **Asymmetry preserved:** C4 retr% ≥ C3 retr%, accuracy never drops.

---

## 3. 7B vs 1.5B — CARD's gate is much smoother at 7B

The key model-size effect: at **1.5B**, CARD's retrieval rate jumped **0→100% across a narrow T_RAG band (0.3–0.5)** — a coarse, near-binary gate. At **7B**, CARD's logits are sharper/better-calibrated, so retr% rises **gradually** (CCE: 0→2→7→18→30→60→81→91→100 over T_RAG 0.05→0.45). This means:
- 7B gives a **usable cost/quality operating curve** — you can pick e.g. T_RAG=0.25 (CCE) for ES 0.79 at 30% retrieval, impossible to target cleanly at 1.5B.
- The cascade's marginal value is **clearer and spread across the curve** at 7B.

| | 1.5B | 7B |
|---|--:|--:|
| C1 ES (CCE) | 0.631 | **0.652** |
| C1 ES (RepoEval-fn body) | 0.431 | **0.444** |
| C2 ES (CCE) | 0.961 | 0.961 |
| C2 ES (RepoEval-fn body) | **0.888** | 0.839 |
| C1 latency (RepoEval-fn) | 1106 ms | 3607 ms |

Interesting: **C2 RepoEval-function ES is slightly LOWER at 7B (0.839) than 1.5B (0.888)**. Likely the larger model's retrieved-context completions diverge stylistically from gold more often on this small 455-set; not a degeneracy (full-scoring and CCE both behave normally). Worth noting, not alarming.

---

## 4. Hallucination metric

- **hall_A4B2 (strict, invented)** is the headline hallucination signal. The cascade drives it toward ~0 at low T_RAG on both datasets — its core purpose.
- **hall_A4 (loose)** tracks accuracy inversely (C1 RepoEval-fn 0.706 → C2 0.226). On RepoEval-function **full** scoring, hall_A4 stays high even for C2 (0.859) — an artifact of untruncated over-generation introducing many non-gold identifiers; the **body** scoring (0.226) is the correct view.

---

## 5. Efficiency
- **7B is ~3× slower than 1.5B per instance** (RepoEval-fn C1: 3607 vs 1106 ms; CCE C1: 502 vs 200 ms) — expected.
- Cascade keeps most hallucination reduction at **single-digit % retrieval** near C1 latency (e.g. RepoEval-fn T_RAG=0.05: 3.7% retrieval, hall_A4B2 2.7×↓).

---

## 6. body vs full (RepoEval-function)
C2 ES **0.839 (body)** vs **0.444 (full)** — over-generation past the body massively deflates full scoring. Headline uses body; full reported for completeness.

---

## 7. Takeaways
- **Retrieval decisive** on both datasets at 7B (ES → 0.96 CCE / 0.84 RepoEval-fn).
- **Cascade ≈ matches/edges CARD on accuracy while cutting invented-identifier hallucinations** across the whole low/mid T_RAG range — RQ2 supported, and more cleanly than at 1.5B thanks to CARD's smoother gate.
- **Model-size story:** 7B's better-calibrated confidence gives a graded retrieval curve (vs 1.5B's near-binary), making adaptive retrieval genuinely tunable — the strongest argument for the 7B setting.

_Generated by the Colab monitor after the CodeLlama-7B run completed both datasets._
