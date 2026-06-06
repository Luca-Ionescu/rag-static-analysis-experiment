"""No-retrieve and always-retrieve baselines (configs C1 and C2)."""
from __future__ import annotations

import time

from .generator import Generation, Generator, LatencyProxy
from .prompt import build_fim_prompt
from .retriever import BM25Retriever, make_query


def no_retrieve_baseline(
    generator: Generator,
    x_left: str,
    x_right: str,
    model_family: str = "qwen",
) -> Generation:
    """C1: generate from the in-file FIM prompt only.

    ``latency_ms`` is the full per-instance pipeline time (here ≈ generation),
    measured cache-robustly so it stays comparable to CARD/cascade.
    """
    t0 = time.perf_counter()
    gen = LatencyProxy(generator)
    prompt = build_fim_prompt(x_left, x_right, retrieved=None, model_family=model_family)
    g = gen.generate(prompt)
    g.latency_ms = gen.reported_ms + (time.perf_counter() - t0 - gen.gen_wall_s) * 1000.0
    return g


def always_retrieve_baseline(
    generator: Generator,
    retriever: BM25Retriever,
    x_left: str,
    x_right: str,
    model_family: str = "qwen",
    top_k: int = 10,
) -> Generation:
    """C2: always run BM25 retrieval and prepend the top-k chunks.

    ``latency_ms`` covers the BM25 retrieval **and** generation, so it is
    comparable to CARD/cascade (which also pay for retrieval).
    """
    t0 = time.perf_counter()
    gen = LatencyProxy(generator)
    query = make_query(x_left)
    retrieved = retriever.retrieve(query, top_k=top_k)
    prompt = build_fim_prompt(
        x_left, x_right, retrieved=retrieved, model_family=model_family
    )
    g = gen.generate(prompt)
    g.latency_ms = gen.reported_ms + (time.perf_counter() - t0 - gen.gen_wall_s) * 1000.0
    return g
