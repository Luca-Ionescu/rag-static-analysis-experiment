"""The cascade pipeline: CARD's gate + a three-tier static-analysis gate.

IMPLEMENTATION_GUIDE §11. The cascade is asymmetric: static analysis can only
*add* retrievals to CARD's decisions, never remove them. This bounds the
worst-case retrieval count to "always-retrieve" and frames the research
question as "does the extra retrieval budget reduce hallucinations?" rather
than a confounded accuracy-vs-cost tradeoff.

Stage 1: ŷ₀ ← zero-shot Generator(x_left, x_right)
Stage 2: if CARD's is_retrieve(ŷ₀, …) → retrieve and return ŷ_rag
Stage 3: else if PredictionAnalyzer(ŷ₀, …).fires → retrieve and return ŷ_rag
Otherwise: return ŷ₀.

Stage 3 runs three independent static-analysis checks (see
:mod:`static_analysis.analyzer`):

  Tier 1 (out-of-scope)    — single signal, no symbol-table dependency
  Tier 2 (signature)       — call shape against repo signatures
  Tier 3 (import)          — ``from X import Y`` against repo module layout

Any tier firing triggers retrieval. The recorded trigger_reason follows a
precedence (strongest-signal-first) for paper diagnostics:

    static_out_of_scope > static_signature > static_import

Each tier can be toggled independently on ``PredictionAnalyzer`` so the
A1 ablation matrix is measurable from a single set of cached generations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass

from .card.estimator import Estimator
from .card.features import extract_features
from .generator import Generator
from .prompt import build_fim_prompt
from .retriever import BM25Retriever, make_query
from .static_analysis.analyzer import PredictionAnalyzer


def _serialise(items) -> list[dict]:
    """Convert a list of dataclasses (CallIssue / ImportIssue) to dicts."""
    out: list[dict] = []
    for it in items or ():
        if is_dataclass(it):
            out.append(asdict(it))
        elif isinstance(it, dict):
            out.append(it)
    return out


@dataclass
class CascadeOutput:
    prediction: str
    retrieved: bool
    trigger_reason: str            # "none", "card", or "static_{out_of_scope|signature|import}"
    s_hat_0: float                 # CARD's estimated ES for ŷ₀
    static_out_of_scope: list[str] = field(default_factory=list)
    signature_issues: list[dict] = field(default_factory=list)
    import_issues: list[dict] = field(default_factory=list)
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
    importing_file: str = "",
) -> CascadeOutput:
    """Run CARD + three-tier static-analysis cascade. Single-RAG (no Select).

    Args:
        t_rag: CARD's retrieval threshold. ŝ₀ < t_rag → retrieve.
        top_k: BM25 top-k chunks when retrieval fires.
        importing_file: path of the file the prediction is being inserted
            into, used by Tier 3 to resolve relative imports. Pass ``""``
            if not meaningful.
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

    # Stage 3: three-tier static analysis on ŷ₀
    sa = analyzer.analyze(g0.prediction, x_left, x_right, importing_file=importing_file)
    if sa.fires:
        # Precedence for attribution: out_of_scope > signature > import.
        # Out-of-scope is the strongest hallucination signal; signature and
        # import are more structural ("wrong shape" / "wrong source"). Each
        # branch checks the corresponding flag so disabled tiers don't claim
        # attribution.
        if analyzer.fire_on_out_of_scope and sa.significant_out_of_scope:
            reason = "static_out_of_scope"
        elif analyzer.fire_on_signature and sa.signature_issues:
            reason = "static_signature"
        elif analyzer.fire_on_import and sa.import_issues:
            reason = "static_import"
        else:
            # Defensive — analyzer.fires was True so at least one branch
            # should match. Fall back to a generic tag.
            reason = "static"
        g_rag = _retrieve_and_regenerate(
            generator, retriever, x_left, x_right, model_family, top_k
        )
        return CascadeOutput(
            prediction=g_rag.prediction,
            retrieved=True,
            trigger_reason=reason,
            s_hat_0=s_hat_0,
            static_out_of_scope=list(sa.significant_out_of_scope),
            signature_issues=_serialise(sa.signature_issues),
            import_issues=_serialise(sa.import_issues),
            latency_ms=g0.latency_ms + g_rag.latency_ms,
        )

    # No retrieval — return zero-shot. Carry diagnostics forward even when
    # nothing fired so Phase 7 disagreement analysis can mine them.
    return CascadeOutput(
        prediction=g0.prediction,
        retrieved=False,
        trigger_reason="none",
        s_hat_0=s_hat_0,
        static_out_of_scope=list(sa.significant_out_of_scope),
        signature_issues=_serialise(sa.signature_issues),
        import_issues=_serialise(sa.import_issues),
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
