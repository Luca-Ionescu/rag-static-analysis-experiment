"""CARD Algorithm 1 (single-RAG) pipeline tests using MockGenerator + MockEstimator."""
from __future__ import annotations

import pytest

from adaptive_retrieval.card.estimator import MockEstimator
from adaptive_retrieval.card.pipeline import (
    EPSILON,
    CARDOutput,
    card_pipeline,
    is_retrieve,
    select,
)
from adaptive_retrieval.generator import MockGenerator
from adaptive_retrieval.retriever import BM25Retriever


REPO = {"lib.py": "def special_helper(x):\n    return x + 100\n"}


# ---------- helper unit tests ----------

def test_is_retrieve_below_threshold_triggers():
    import numpy as np

    est = MockEstimator([0.5])
    assert is_retrieve(est, np.full(3, 0.7), np.full(3, 0.5), t_rag=0.9)


def test_is_retrieve_at_or_above_threshold_skips():
    import numpy as np

    est = MockEstimator([0.95])
    assert not is_retrieve(est, np.full(3, 0.7), np.full(3, 0.5), t_rag=0.9)


def test_select_keeps_older_when_newer_is_relatively_worse():
    # ratio = 0.4 / 0.8 = 0.5 < t_acc=0.8 → keep older
    assert select(s_hat_i=0.8, s_hat_j=0.4, t_acc=0.8)


def test_select_accepts_newer_when_better():
    # ratio = 0.95 / 0.5 = 1.9 >= t_acc=0.8 → accept newer
    assert not select(s_hat_i=0.5, s_hat_j=0.95, t_acc=0.8)


def test_select_handles_zero_denominator_via_epsilon():
    # i is 0; ratio = j / EPSILON which is huge → accept newer
    assert not select(s_hat_i=0.0, s_hat_j=0.3, t_acc=0.8)
    # but very small j keeps older
    assert select(s_hat_i=0.0, s_hat_j=EPSILON / 10, t_acc=0.8)


# ---------- card_pipeline tests ----------

def test_high_confidence_skips_retrieval():
    gen = MockGenerator(default_prediction="ZS_OUT")
    est = MockEstimator([0.95])  # ŝ₀=0.95 > T_RAG=0.9
    retriever = BM25Retriever(REPO)

    out = card_pipeline(
        gen, retriever, est,
        x_left="def f():\n    return ", x_right="\n",
    )
    assert isinstance(out, CARDOutput)
    assert out.prediction == "ZS_OUT"
    assert out.retrieved_at_iter == []
    assert out.n_iterations == 0
    assert out.s_hats == pytest.approx([0.95])
    # Exactly one generator call (zero-shot only).
    assert len(gen.call_log) == 1


def test_low_confidence_triggers_retrieval_and_accepts_newer():
    # ŝ₀=0.5 < 0.9 → retrieve; ŝ₁=0.85; ratio 0.85/0.5=1.7 >= 0.8 → accept newer
    gen = MockGenerator(
        prompt_to_prediction={},
        default_prediction="GEN_OUT",
    )
    est = MockEstimator([0.5, 0.85])
    retriever = BM25Retriever(REPO)

    out = card_pipeline(
        gen, retriever, est,
        x_left="def f():\n    return special_helper", x_right="\n",
        t_rag_schedule=[0.9],
        t_acc_schedule=[0.8],
    )
    assert out.retrieved_at_iter == [1]
    assert out.n_iterations == 1
    assert out.s_hats == pytest.approx([0.5, 0.85])
    # Two generator calls — zero-shot + RAG.
    assert len(gen.call_log) == 2
    # RAG prompt should include retrieved chunks.
    assert "# Here are some relevant code fragments" in gen.call_log[1]
    assert "# Here are some relevant code fragments" not in gen.call_log[0]


def test_low_confidence_triggers_but_keeps_older_when_newer_is_worse():
    # ŝ₀=0.5; ŝ₁=0.1; ratio=0.1/0.5=0.2 < 0.8 → keep older (ŷ₀)
    zs_out = "ZS_OUT"
    rag_out = "RAG_OUT"

    # MockGenerator does exact-prompt lookup, but here we want behaviour keyed
    # on "is this a RAG prompt or not". Emulate with a subclass that returns
    # rag_out iff the prompt contains the retrieval header.
    class TwoCallMock(MockGenerator):
        def __init__(self):
            super().__init__(default_prediction=zs_out)
            self._n = 0

        def generate(self, prompt):
            self._n += 1
            self.call_log.append(prompt)
            pred = rag_out if "# Here are some relevant" in prompt else zs_out
            import numpy as np
            return type(super().generate(prompt))(
                prediction=pred,
                token_ids=[0],
                token_probs=np.array([0.5], dtype=np.float32),
                token_entropies=np.array([0.5], dtype=np.float32),
                latency_ms=1.0,
            )

    gen2 = TwoCallMock()
    est = MockEstimator([0.5, 0.1])
    retriever = BM25Retriever(REPO)
    out = card_pipeline(
        gen2, retriever, est,
        x_left="def f():\n    return ", x_right="\n",
    )
    assert out.retrieved_at_iter == [1]  # retrieval ran
    assert out.n_iterations == 0          # but Select kept ŷ₀
    assert out.prediction == zs_out
    assert out.s_hats == pytest.approx([0.5, 0.1])


def test_max_iter_zero_disables_retrieval():
    gen = MockGenerator(default_prediction="ZS_OUT")
    est = MockEstimator([0.1])  # very low confidence; would normally retrieve
    retriever = BM25Retriever(REPO)
    out = card_pipeline(
        gen, retriever, est,
        x_left="def f():\n    return ", x_right="\n",
        max_iter=0,
    )
    assert out.n_iterations == 0
    assert out.retrieved_at_iter == []
    assert len(gen.call_log) == 1


def test_latency_accumulates():
    gen = MockGenerator(default_prediction="OUT", latency_ms=42.0)
    est = MockEstimator([0.5, 0.5])
    retriever = BM25Retriever(REPO)
    out = card_pipeline(
        gen, retriever, est,
        x_left="def f():\n    return ", x_right="\n",
    )
    # Two generations × 42ms each = 84ms
    assert out.latency_ms == pytest.approx(84.0)


def test_default_schedule_thresholds_used():
    """Confirm the defaults (0.9 / 0.8) are applied when schedules are None."""
    gen = MockGenerator(default_prediction="OUT")
    est = MockEstimator([0.89])  # just below 0.9
    retriever = BM25Retriever(REPO)
    out = card_pipeline(gen, retriever, est, x_left="L", x_right="R")
    # ŝ₀=0.89 < default T_RAG=0.9 → retrieval triggered
    assert out.retrieved_at_iter == [1]
