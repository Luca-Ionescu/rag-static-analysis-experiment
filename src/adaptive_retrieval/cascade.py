"""The cascade pipeline: CARD's gate + a static-analysis second-chance gate.

IMPLEMENTATION_GUIDE §11. The cascade is asymmetric: static analysis can only
*add* retrievals to CARD's decisions, never remove them. This bounds the
worst-case retrieval count to "always-retrieve" and frames the research
question as "does the extra retrieval budget reduce hallucinations?" rather
than a confounded accuracy-vs-cost tradeoff.

Stage 1: ŷ₀ ← zero-shot Generator(x_left, x_right)
Stage 2: if CARD's is_retrieve(ŷ₀, …) → retrieve and return ŷ_rag
Stage 3: else if PredictionAnalyzer(ŷ₀, …).fires → retrieve and return ŷ_rag
Otherwise: return ŷ₀.

Stage 3 fires on a single signal: any used identifier in a structurally
significant position that's not visible at the hole. The cross-file vs
unresolved distinction the analyzer used to make is now diagnostic only —
both lead to the same retrieve-and-regenerate action.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .card.estimator import Estimator
from .card.features import extract_features
from .generator import Generator
from .prompt import build_fim_prompt
from .retriever import BM25Retriever, make_query
from .static_analysis.analyzer import PredictionAnalyzer


@dataclass
class CascadeOutput:
    prediction: str
    retrieved: bool
    trigger_reason: str            # one of "none", "card", "static"
    s_hat_0: float                 # CARD's estimated ES for ŷ₀
    static_out_of_scope: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


def cascade_pipeline(
    generator: Generator,
    retriever: BM25Retriever,
    estimator: Estimator,
    analyzer: PredictionAnalyzer,
    x_left: str,
    x_right: str,
    t_rag: float = 0.9,
    model_family: str = "qwen",
    top_k: int = 10,
) -> CascadeOutput:
    """Run CARD + static-analysis cascade. Single-RAG (no Select stage).

    Args:
        t_rag: CARD's retrieval threshold. ŝ₀ < t_rag → retrieve.
        top_k: BM25 top-k chunks when retrieval fires.
    """
    # Stage 1: zero-shot generation
    prompt_zs = build_fim_prompt(x_left, x_right, retrieved=None, model_family=model_family)
    g0 = generator.generate(prompt_zs)
    feats0 = extract_features(g0.token_probs, g0.token_entropies)
    s_hat_0 = float(estimator.predict(feats0)[0])

    # Stage 2: CARD's is_retrieve gate
    if s_hat_0 < t_rag:
        g_rag = _retrieve_and_regenerate(
            generator, retriever, x_left, x_right, model_family, top_k
        )
        return CascadeOutput(
            prediction=g_rag.prediction,
            retrieved=True,
            trigger_reason="card",
            s_hat_0=s_hat_0,
            latency_ms=g0.latency_ms + g_rag.latency_ms,
        )

    # Stage 3: static-analysis gate on ŷ₀
    sa = analyzer.analyze(g0.prediction, x_left, x_right)
    if sa.fires:
        g_rag = _retrieve_and_regenerate(
            generator, retriever, x_left, x_right, model_family, top_k
        )
        return CascadeOutput(
            prediction=g_rag.prediction,
            retrieved=True,
            trigger_reason="static",
            s_hat_0=s_hat_0,
            static_out_of_scope=list(sa.significant_out_of_scope),
            latency_ms=g0.latency_ms + g_rag.latency_ms,
        )

    # No retrieval — return zero-shot
    return CascadeOutput(
        prediction=g0.prediction,
        retrieved=False,
        trigger_reason="none",
        s_hat_0=s_hat_0,
        latency_ms=g0.latency_ms,
    )


def _retrieve_and_regenerate(
    generator: Generator,
    retriever: BM25Retriever,
    x_left: str,
    x_right: str,
    model_family: str,
    top_k: int,
):
    query = make_query(x_left)
    chunks = retriever.retrieve(query, top_k=top_k)
    prompt_rag = build_fim_prompt(
        x_left, x_right, retrieved=chunks, model_family=model_family
    )
    return generator.generate(prompt_rag)
