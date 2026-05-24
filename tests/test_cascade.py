"""Cascade pipeline tests covering all four trigger reasons.

Strategy: ``MockGenerator`` returns prompt-keyed predictions so we can
distinguish zero-shot vs. RAG output, and ``MockEstimator`` returns scripted
ŝ values. We control the prediction text to land on each branch.
"""
from __future__ import annotations

import numpy as np
import pytest

from adaptive_retrieval.card.estimator import MockEstimator
from adaptive_retrieval.cascade import CascadeOutput, cascade_pipeline
from adaptive_retrieval.generator import Generation, MockGenerator
from adaptive_retrieval.retriever import BM25Retriever
from adaptive_retrieval.static_analysis.analyzer import PredictionAnalyzer
from adaptive_retrieval.static_analysis.scope import InFileScopeAnalyzer
from adaptive_retrieval.static_analysis.symbol_table import RepositorySymbolTable


REPO = {"lib.py": "def cross_func():\n    return 42\n"}


class _BranchedMock(MockGenerator):
    """Mock that returns ``rag_pred`` if the prompt contains the retrieval
    header, otherwise ``zs_pred``. Lets us identify which generation path
    produced the final cascade output.
    """

    def __init__(self, zs_pred: str, rag_pred: str, latency_ms: float = 1.0):
        super().__init__(default_prediction=zs_pred, latency_ms=latency_ms)
        self.zs_pred = zs_pred
        self.rag_pred = rag_pred

    def generate(self, prompt: str) -> Generation:
        self.call_log.append(prompt)
        is_rag = "# Here are some relevant code fragments" in prompt
        pred = self.rag_pred if is_rag else self.zs_pred
        n = max(1, len(pred.split()))
        return Generation(
            prediction=pred,
            token_ids=list(range(n)),
            token_probs=np.full(n, 0.7, dtype=np.float32),
            token_entropies=np.full(n, 0.5, dtype=np.float32),
            latency_ms=self.latency_ms,
        )


def _make_analyzer(repo_files=None):
    syms = (
        RepositorySymbolTable.from_files(repo_files)
        if repo_files
        else RepositorySymbolTable.from_files({})
    )
    return PredictionAnalyzer(InFileScopeAnalyzer(), syms)


# ---------- branch coverage ----------

def test_card_below_threshold_triggers_retrieval():
    # ŝ₀=0.5 < t_rag=0.9 → CARD fires. Static analysis is NOT consulted.
    gen = _BranchedMock(zs_pred="hallucinated_func()", rag_pred="real_func()")
    est = MockEstimator([0.5])
    retriever = BM25Retriever(REPO)
    analyzer = _make_analyzer(REPO)

    out = cascade_pipeline(
        gen, retriever, est, analyzer,
        x_left="def f():\n    return ", x_right="\n",
    )
    assert isinstance(out, CascadeOutput)
    assert out.retrieved
    assert out.trigger_reason == "card"
    assert out.s_hat_0 == pytest.approx(0.5)
    assert out.prediction == "real_func()"
    # CARD path doesn't populate the static field.
    assert out.static_out_of_scope == []
    assert len(gen.call_log) == 2  # zero-shot + RAG


def test_high_confidence_clean_prediction_no_retrieval():
    # ŝ₀=0.95 above threshold; prediction "pass" has no identifiers to fire on.
    gen = _BranchedMock(zs_pred="pass", rag_pred="should_not_be_used()")
    est = MockEstimator([0.95])
    retriever = BM25Retriever(REPO)
    analyzer = _make_analyzer(REPO)

    out = cascade_pipeline(
        gen, retriever, est, analyzer,
        x_left="def f():\n    ", x_right="\n",
    )
    assert not out.retrieved
    assert out.trigger_reason == "none"
    assert out.prediction == "pass"
    assert out.static_out_of_scope == []
    # Only the zero-shot generator call happened.
    assert len(gen.call_log) == 1


def test_high_confidence_unresolved_identifier_triggers_static():
    # ŝ₀ high; prediction uses a name that's not in-file, not in repo, not builtin.
    gen = _BranchedMock(
        zs_pred="totally_made_up_name()",
        rag_pred="real_func()",
    )
    est = MockEstimator([0.95])
    retriever = BM25Retriever(REPO)
    analyzer = _make_analyzer(REPO)  # repo has only "cross_func"

    out = cascade_pipeline(
        gen, retriever, est, analyzer,
        x_left="def f():\n    return ", x_right="\n",
    )
    assert out.retrieved
    assert out.trigger_reason == "static_out_of_scope"
    assert "totally_made_up_name" in out.static_out_of_scope
    # Final prediction is the RAG one.
    assert out.prediction == "real_func()"
    assert len(gen.call_log) == 2


def test_high_confidence_crossfile_identifier_triggers_static():
    # Prediction uses a name that resolves in the repo (cross_func in lib.py)
    # but not in the in-file context. Under the unified trigger this still
    # fires with trigger_reason="static" — we no longer distinguish.
    gen = _BranchedMock(
        zs_pred="cross_func()",
        rag_pred="local_helper_after_retrieve()",
    )
    est = MockEstimator([0.95])
    retriever = BM25Retriever(REPO)
    analyzer = _make_analyzer(REPO)

    out = cascade_pipeline(
        gen, retriever, est, analyzer,
        x_left="def f():\n    return ", x_right="\n",
    )
    assert out.retrieved
    assert out.trigger_reason == "static_out_of_scope"
    assert "cross_func" in out.static_out_of_scope


def test_mixed_unresolved_and_crossfile_both_in_out_of_scope_list():
    # Prediction has BOTH a cross-file name AND an unresolved name. Both
    # appear in the unified out-of-scope list.
    gen = _BranchedMock(
        zs_pred="totally_fake() + cross_func()",
        rag_pred="real_func()",
    )
    est = MockEstimator([0.95])
    retriever = BM25Retriever(REPO)
    analyzer = _make_analyzer(REPO)

    out = cascade_pipeline(
        gen, retriever, est, analyzer,
        x_left="def f():\n    return ", x_right="\n",
    )
    assert out.retrieved
    assert out.trigger_reason == "static_out_of_scope"
    assert "totally_fake" in out.static_out_of_scope
    assert "cross_func" in out.static_out_of_scope


# ---------- semantic checks ----------

def test_card_path_does_not_consult_static_analyzer():
    """If CARD fires first, ŷ₀ never gets static-analyzed. Use a custom
    analyzer that would crash to prove it wasn't called.
    """
    class _ExplodingAnalyzer:
        def analyze(self, *args, **kwargs):  # noqa: D401
            raise AssertionError("Static analyzer should NOT be called when CARD fires")

    gen = _BranchedMock(zs_pred="anything", rag_pred="rag_out")
    est = MockEstimator([0.5])  # below default t_rag=0.9
    retriever = BM25Retriever(REPO)

    out = cascade_pipeline(
        gen, retriever, est, _ExplodingAnalyzer(),  # type: ignore[arg-type]
        x_left="def f():\n    return ", x_right="\n",
    )
    assert out.trigger_reason == "card"


def test_rag_prompt_contains_retrieved_chunks():
    gen = _BranchedMock(zs_pred="zs_out", rag_pred="rag_out")
    est = MockEstimator([0.5])
    retriever = BM25Retriever(REPO)
    analyzer = _make_analyzer(REPO)

    cascade_pipeline(
        gen, retriever, est, analyzer,
        x_left="def f():\n    return cross_func", x_right="\n",
    )
    # First call = zero-shot (no retrieval header), second = RAG (with header).
    assert "# Here are some relevant code fragments" not in gen.call_log[0]
    assert "# Here are some relevant code fragments" in gen.call_log[1]
    assert "# lib.py" in gen.call_log[1]


def test_latency_accumulates_when_retrieval_fires():
    gen = _BranchedMock(zs_pred="zs", rag_pred="rag", latency_ms=37.0)
    est = MockEstimator([0.5])
    retriever = BM25Retriever(REPO)
    analyzer = _make_analyzer(REPO)
    out = cascade_pipeline(
        gen, retriever, est, analyzer,
        x_left="L", x_right="R",
    )
    assert out.latency_ms == pytest.approx(74.0)  # 2 generations × 37ms


def test_latency_is_single_gen_when_no_retrieval():
    gen = _BranchedMock(zs_pred="pass", rag_pred="never_called", latency_ms=22.0)
    est = MockEstimator([0.95])
    retriever = BM25Retriever(REPO)
    analyzer = _make_analyzer(REPO)
    out = cascade_pipeline(
        gen, retriever, est, analyzer,
        x_left="L", x_right="R",
    )
    assert out.latency_ms == pytest.approx(22.0)


def test_custom_t_rag_threshold():
    # ŝ₀=0.85 — would fire at default 0.9, not at 0.7.
    gen = _BranchedMock(zs_pred="pass", rag_pred="rag_out")
    est = MockEstimator([0.85])
    retriever = BM25Retriever(REPO)
    analyzer = _make_analyzer(REPO)

    out_low = cascade_pipeline(
        gen, retriever, est, analyzer,
        x_left="L", x_right="R", t_rag=0.7,
    )
    assert out_low.trigger_reason == "none"

    # Reset mocks for second call
    gen2 = _BranchedMock(zs_pred="pass", rag_pred="rag_out")
    est2 = MockEstimator([0.85])
    out_high = cascade_pipeline(
        gen2, retriever, est2, analyzer,
        x_left="L", x_right="R", t_rag=0.9,
    )
    assert out_high.trigger_reason == "card"
