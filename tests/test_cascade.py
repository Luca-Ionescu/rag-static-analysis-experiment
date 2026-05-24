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
    # CARD path doesn't populate static fields.
    assert out.static_unresolved == []
    assert out.static_crossfile == []
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
    assert out.static_unresolved == []
    assert out.static_crossfile == []
    # Only the zero-shot generator call happened.
    assert len(gen.call_log) == 1


def test_high_confidence_unresolved_identifier_triggers_static_unresolved():
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
    assert out.trigger_reason == "static_unresolved"
    assert "totally_made_up_name" in out.static_unresolved
    # Final prediction is the RAG one.
    assert out.prediction == "real_func()"
    assert len(gen.call_log) == 2


def test_high_confidence_crossfile_identifier_does_not_fire_by_default():
    # Prediction uses a name that resolves in the repo (cross_func in lib.py)
    # but not in the in-file context. With the default fire_on_crossfile=False,
    # the cascade should NOT retrieve — the model already recovered the right
    # cross-file name from parametric memory.
    gen = _BranchedMock(
        zs_pred="cross_func()",
        rag_pred="local_helper_after_retrieve()",
    )
    est = MockEstimator([0.95])
    retriever = BM25Retriever(REPO)
    analyzer = _make_analyzer(REPO)  # defaults: fire_on_crossfile=False

    out = cascade_pipeline(
        gen, retriever, est, analyzer,
        x_left="def f():\n    return ", x_right="\n",
    )
    assert not out.retrieved
    assert out.trigger_reason == "none"
    assert out.prediction == "cross_func()"
    # Only the zero-shot generation happened.
    assert len(gen.call_log) == 1


def test_high_confidence_crossfile_identifier_fires_when_flag_enabled():
    # The A1 ablation path: explicit fire_on_crossfile=True restores the
    # cross-file trigger so the 2x2 (crossfile, unresolved) matrix is still
    # measurable.
    gen = _BranchedMock(
        zs_pred="cross_func()",
        rag_pred="local_helper_after_retrieve()",
    )
    est = MockEstimator([0.95])
    retriever = BM25Retriever(REPO)
    analyzer = PredictionAnalyzer(
        InFileScopeAnalyzer(),
        RepositorySymbolTable.from_files(REPO),
        fire_on_crossfile=True,
    )

    out = cascade_pipeline(
        gen, retriever, est, analyzer,
        x_left="def f():\n    return ", x_right="\n",
    )
    assert out.retrieved
    assert out.trigger_reason == "static_crossfile"
    assert "cross_func" in out.static_crossfile
    # Unresolved should be empty for this case (no hallucinated names).
    assert out.static_unresolved == []


def test_unresolved_takes_precedence_over_crossfile():
    # Prediction has BOTH a cross-file name AND an unresolved name.
    # By design, `static_unresolved` wins (stronger hallucination signal).
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
    assert out.trigger_reason == "static_unresolved"
    assert "totally_fake" in out.static_unresolved
    assert "cross_func" in out.static_crossfile  # still populated for diagnostics


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


# ---------- new Tier 2 / Tier 3 trigger reasons ----------

def test_signature_mismatch_triggers_static_signature():
    """CARD says skip; prediction calls a known repo function with the
    wrong arity. The static-signature gate should fire."""
    repo_files = {
        "pkg/__init__.py": "",
        "pkg/core.py": "def foo(x):\n    return x\n",
    }
    gen = _BranchedMock(
        zs_pred="foo(1, 2, 3)",  # foo only takes one positional arg
        rag_pred="foo(1)",
    )
    est = MockEstimator([0.95])  # CARD says skip
    retriever = BM25Retriever(repo_files)
    analyzer = _make_analyzer(repo_files)

    out = cascade_pipeline(
        gen, retriever, est, analyzer,
        x_left="from pkg.core import foo\n\ndef caller():\n    return ",
        x_right="\n",
    )
    assert out.retrieved
    assert out.trigger_reason == "static_signature"
    assert out.signature_issues
    assert out.signature_issues[0].kind == "wrong_arity"
    assert out.prediction == "foo(1)"


def test_wrong_import_triggers_static_import():
    """CARD says skip; prediction imports a known repo symbol from the
    wrong module. The static-import gate should fire."""
    repo_files = {
        "pkg/__init__.py": "",
        "pkg/core.py": "def Foo():\n    return 1\n",
        "pkg/other.py": "",
    }
    gen = _BranchedMock(
        zs_pred="from pkg.other import Foo",
        rag_pred="from pkg.core import Foo",
    )
    est = MockEstimator([0.95])
    retriever = BM25Retriever(repo_files)
    analyzer = _make_analyzer(repo_files)

    out = cascade_pipeline(
        gen, retriever, est, analyzer,
        x_left="",
        x_right="\n",
    )
    assert out.retrieved
    assert out.trigger_reason == "static_import"
    assert out.import_issues
    assert out.import_issues[0].kind == "wrong_origin"


def test_unresolved_wins_over_signature_and_import():
    """Precedence check: unresolved > signature > import > crossfile."""
    repo_files = {
        "pkg/__init__.py": "",
        "pkg/core.py": "def foo(x):\n    return x\n",
    }
    # Prediction triggers BOTH a hallucinated identifier AND wrong arity.
    gen = _BranchedMock(
        zs_pred="totally_fake() + foo(1, 2, 3)",
        rag_pred="real()",
    )
    est = MockEstimator([0.95])
    retriever = BM25Retriever(repo_files)
    analyzer = _make_analyzer(repo_files)

    out = cascade_pipeline(
        gen, retriever, est, analyzer,
        x_left="from pkg.core import foo\n\ndef caller():\n    return ",
        x_right="\n",
    )
    assert out.retrieved
    assert out.trigger_reason == "static_unresolved"
    assert "totally_fake" in out.static_unresolved
    # Diagnostics for the lower-priority signal are still surfaced.
    assert out.signature_issues

