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

Trigger-reason priority within Stage 3 (strongest hallucination signal
first):

    static_unresolved > static_signature > static_import > static_crossfile

Rationale:
* ``static_unresolved`` — name resolves nowhere. Near-certain hallucination.
* ``static_signature`` — name exists but is called with wrong arity or
  unknown keyword names against a unique repo signature.
* ``static_import`` — ``from X import Y`` where ``Y`` actually lives in
  ``Z != X`` and isn't re-exported by ``X``.
* ``static_crossfile`` — name exists in the repo but not in-file scope.
  Off by default; retained for the A1 ablation.

By default ``PredictionAnalyzer`` fires on unresolved, signature, and
import. A cross-file identifier alone is evidence the model already
recovered the right name from parametric memory or in-file hints, so
spending retrieval budget there is hard to justify.
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
    trigger_reason: str            # see module docstring for the full set
    s_hat_0: float                 # CARD's estimated ES for ŷ₀
    static_unresolved: list[str] = field(default_factory=list)
    static_crossfile: list[str] = field(default_factory=list)
    # Tier 2 / Tier 3 diagnostic payloads — populated even when those
    # signals didn't trigger so Phase-7 disagreement analysis can mine them.
    signature_issues: list = field(default_factory=list)
    import_issues: list = field(default_factory=list)
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
        # Precedence: unresolved > signature > import > crossfile.
        if sa.unresolved_identifiers:
            reason = "static_unresolved"
        elif sa.signature_issues:
            reason = "static_signature"
        elif sa.import_issues:
            reason = "static_import"
        else:
            reason = "static_crossfile"
        g_rag = _retrieve_and_regenerate(
            generator, retriever, x_left, x_right, model_family, top_k
        )
        return CascadeOutput(
            prediction=g_rag.prediction,
            retrieved=True,
            trigger_reason=reason,
            s_hat_0=s_hat_0,
            static_unresolved=list(sa.unresolved_identifiers),
            static_crossfile=list(sa.cross_file_identifiers),
            signature_issues=list(sa.signature_issues),
            import_issues=list(sa.import_issues),
            latency_ms=g0.latency_ms + g_rag.latency_ms,
        )

    # No retrieval — return zero-shot. Carry diagnostics forward so Phase 7
    # disagreement analysis can mine cases where static analysis ran but
    # didn't fire (e.g. cross-file detected with the ablation default off).
    return CascadeOutput(
        prediction=g0.prediction,
        retrieved=False,
        trigger_reason="none",
        s_hat_0=s_hat_0,
        static_unresolved=list(sa.unresolved_identifiers),
        static_crossfile=list(sa.cross_file_identifiers),
        signature_issues=list(sa.signature_issues),
        import_issues=list(sa.import_issues),
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
