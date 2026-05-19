"""No-retrieve and always-retrieve baselines (configs C1 and C2)."""
from __future__ import annotations

from .generator import Generation, Generator
from .prompt import build_fim_prompt
from .retriever import BM25Retriever, make_query


def no_retrieve_baseline(
    generator: Generator,
    x_left: str,
    x_right: str,
    model_family: str = "qwen",
) -> Generation:
    """C1: generate from the in-file FIM prompt only."""
    prompt = build_fim_prompt(x_left, x_right, retrieved=None, model_family=model_family)
    return generator.generate(prompt)


def always_retrieve_baseline(
    generator: Generator,
    retriever: BM25Retriever,
    x_left: str,
    x_right: str,
    model_family: str = "qwen",
    top_k: int = 10,
) -> Generation:
    """C2: always run BM25 retrieval and prepend the top-k chunks."""
    query = make_query(x_left)
    retrieved = retriever.retrieve(query, top_k=top_k)
    prompt = build_fim_prompt(
        x_left, x_right, retrieved=retrieved, model_family=model_family
    )
    return generator.generate(prompt)
