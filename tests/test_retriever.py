"""BM25 retriever tests on a toy multi-file repo."""
from __future__ import annotations

import pytest

from adaptive_retrieval.retriever import BM25Retriever, make_query


TOY_REPO = {
    "math_utils.py": (
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "def multiply(x, y):\n"
        "    return x * y\n"
        "\n"
        "def subtract(p, q):\n"
        "    return p - q\n"
    ),
    "string_utils.py": (
        "def reverse_string(s):\n"
        "    return s[::-1]\n"
        "\n"
        "def to_upper_case(s):\n"
        "    return s.upper()\n"
        "\n"
        "def split_string(s, sep):\n"
        "    return s.split(sep)\n"
    ),
    "data_loader.py": (
        "import json\n"
        "import csv\n"
        "\n"
        "def load_json_file(path):\n"
        "    with open(path) as f:\n"
        "        return json.load(f)\n"
        "\n"
        "def load_csv_file(path):\n"
        "    with open(path) as f:\n"
        "        return list(csv.reader(f))\n"
    ),
}


def test_chunks_built_for_every_file():
    r = BM25Retriever(TOY_REPO, chunk_size=5, stride=3)
    paths = {c["file_path"] for c in r.chunks}
    assert paths == set(TOY_REPO.keys())
    assert len(r) >= len(TOY_REPO)


def test_chunk_text_within_file_bounds():
    r = BM25Retriever(TOY_REPO, chunk_size=5, stride=3)
    for c in r.chunks:
        file_lines = TOY_REPO[c["file_path"]].splitlines()
        assert c["start_line"] >= 0
        assert c["end_line"] <= len(file_lines)
        assert c["end_line"] - c["start_line"] <= 5


def test_retrieve_top_chunk_matches_query_topic():
    r = BM25Retriever(TOY_REPO, chunk_size=5, stride=3)
    results = r.retrieve("reverse string upper case split", top_k=3)
    assert len(results) > 0
    # Top hit should be from string_utils.py — that file owns all those tokens.
    assert results[0].file_path == "string_utils.py"


def test_retrieve_json_query_hits_data_loader():
    r = BM25Retriever(TOY_REPO, chunk_size=5, stride=3)
    results = r.retrieve("load json file open", top_k=3)
    assert results[0].file_path == "data_loader.py"


def test_retrieve_scores_descending():
    r = BM25Retriever(TOY_REPO, chunk_size=5, stride=3)
    results = r.retrieve("add multiply subtract", top_k=5)
    assert all(
        results[i].score >= results[i + 1].score for i in range(len(results) - 1)
    )


def test_empty_corpus_returns_no_results():
    r = BM25Retriever({}, chunk_size=5, stride=3)
    assert r.retrieve("anything") == []


def test_empty_query_returns_no_results():
    r = BM25Retriever(TOY_REPO, chunk_size=5, stride=3)
    assert r.retrieve("") == []


def test_short_file_emits_one_chunk():
    r = BM25Retriever({"tiny.py": "x = 1\n"}, chunk_size=20, stride=10)
    assert len(r.chunks) == 1
    assert r.chunks[0]["file_path"] == "tiny.py"


def test_invalid_chunk_params_rejected():
    with pytest.raises(ValueError):
        BM25Retriever({"a.py": "x = 1\n"}, chunk_size=0, stride=1)
    with pytest.raises(ValueError):
        BM25Retriever({"a.py": "x = 1\n"}, chunk_size=5, stride=0)


def test_make_query_keeps_last_n_lines():
    text = "\n".join(f"line{i}" for i in range(50))
    q = make_query(text, n_lines=5)
    assert q == "line45\nline46\nline47\nline48\nline49"


def test_make_query_handles_fewer_lines_than_n():
    q = make_query("only one line", n_lines=20)
    assert q == "only one line"
