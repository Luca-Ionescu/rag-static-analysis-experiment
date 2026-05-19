"""FIM prompt-assembly tests."""
from __future__ import annotations

import pytest

from adaptive_retrieval.prompt import FIM_TOKENS, build_fim_prompt
from adaptive_retrieval.retriever import RetrievedChunk


def _chunk(path="lib.py", text="def foo():\n    return 1") -> RetrievedChunk:
    return RetrievedChunk(
        text=text, file_path=path, start_line=0, end_line=text.count("\n") + 1, score=1.0
    )


def test_qwen_tokens_appear():
    p = build_fim_prompt("LEFT", "RIGHT", retrieved=None, model_family="qwen")
    pre, suf, mid = FIM_TOKENS["qwen"]
    assert p == f"{pre}LEFT{suf}RIGHT{mid}"


def test_codellama_tokens_appear():
    p = build_fim_prompt("LEFT", "RIGHT", retrieved=None, model_family="codellama")
    pre, suf, mid = FIM_TOKENS["codellama"]
    assert p == f"{pre}LEFT{suf}RIGHT{mid}"


def test_starcoder_tokens_appear():
    p = build_fim_prompt("LEFT", "RIGHT", retrieved=None, model_family="starcoder")
    pre, suf, mid = FIM_TOKENS["starcoder"]
    assert p == f"{pre}LEFT{suf}RIGHT{mid}"


def test_unknown_family_raises():
    with pytest.raises(ValueError):
        build_fim_prompt("L", "R", retrieved=None, model_family="codet5")


def test_no_retrieval_does_not_prepend_anything():
    p = build_fim_prompt("LEFT", "RIGHT", retrieved=None, model_family="qwen")
    assert "# Here are some relevant code fragments" not in p


def test_retrieval_prepends_commented_chunks():
    p = build_fim_prompt(
        "LEFT", "RIGHT", retrieved=[_chunk("lib.py", "x = 1\ny = 2")], model_family="qwen"
    )
    pre, suf, mid = FIM_TOKENS["qwen"]
    # The retrieval header sits between prefix and the original x_left.
    assert p.startswith(pre)
    assert p.endswith(f"{suf}RIGHT{mid}")
    assert "# Here are some relevant code fragments" in p
    assert "# the below code fragment can be found in:" in p
    assert "# lib.py" in p
    assert "# x = 1" in p
    assert "# y = 2" in p
    # Original x_left preserved
    assert "LEFT" in p


def test_retrieval_preserves_order_and_paths():
    chunks = [
        _chunk("a.py", "def a():\n    pass"),
        _chunk("b.py", "def b():\n    pass"),
        _chunk("c.py", "def c():\n    pass"),
    ]
    p = build_fim_prompt("L", "R", retrieved=chunks, model_family="qwen")
    # All three paths appear in order
    ia = p.find("# a.py")
    ib = p.find("# b.py")
    ic = p.find("# c.py")
    assert -1 < ia < ib < ic


def test_empty_retrieval_list_treated_as_no_retrieval():
    p = build_fim_prompt("LEFT", "RIGHT", retrieved=[], model_family="qwen")
    assert "# Here are some relevant code fragments" not in p
