"""Baselines test with MockGenerator — verifies prompt assembly and retrieval
get plumbed through correctly without needing a real model.
"""
from __future__ import annotations

from adaptive_retrieval.baselines import always_retrieve_baseline, no_retrieve_baseline
from adaptive_retrieval.generator import MockGenerator
from adaptive_retrieval.prompt import FIM_TOKENS
from adaptive_retrieval.retriever import BM25Retriever


REPO = {
    "lib.py": (
        "def special_helper(x):\n"
        "    return x + 100\n"
        "\n"
        "def another_helper(y):\n"
        "    return y * 2\n"
    ),
}


def test_no_retrieve_passes_in_file_only_prompt():
    gen = MockGenerator(default_prediction="OUT")
    out = no_retrieve_baseline(gen, x_left="def main():\n    return ", x_right="\n")
    assert out.prediction == "OUT"
    assert len(gen.call_log) == 1
    prompt = gen.call_log[0]
    pre, suf, mid = FIM_TOKENS["qwen"]
    assert prompt.startswith(pre) and prompt.endswith(mid)
    assert "# Here are some relevant code fragments" not in prompt


def test_always_retrieve_includes_retrieved_chunks_in_prompt():
    gen = MockGenerator(default_prediction="OUT")
    retriever = BM25Retriever(REPO, chunk_size=5, stride=3)
    out = always_retrieve_baseline(
        gen,
        retriever,
        x_left="def main():\n    return special_helper",
        x_right="(5)\n",
        top_k=3,
    )
    assert out.prediction == "OUT"
    prompt = gen.call_log[0]
    assert "# Here are some relevant code fragments" in prompt
    assert "# lib.py" in prompt
    assert "special_helper" in prompt


def test_always_retrieve_handles_empty_repo():
    gen = MockGenerator(default_prediction="OUT")
    retriever = BM25Retriever({}, chunk_size=5, stride=3)
    out = always_retrieve_baseline(
        gen, retriever, x_left="def main():\n    return ", x_right="\n"
    )
    # Should still generate (just without retrieved chunks)
    assert out.prediction == "OUT"
    prompt = gen.call_log[0]
    assert "# Here are some relevant code fragments" not in prompt


def test_model_family_switches_fim_tokens():
    gen = MockGenerator(default_prediction="OUT")
    no_retrieve_baseline(
        gen, x_left="L", x_right="R", model_family="codellama"
    )
    pre, suf, mid = FIM_TOKENS["codellama"]
    assert gen.call_log[0] == f"{pre}L{suf}R{mid}"
