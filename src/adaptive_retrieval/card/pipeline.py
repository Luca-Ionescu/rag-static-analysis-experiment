"""CARD's Algorithm 1 (single-RAG variant).

Single-RAG (``max_iter=1``) is what our project uses for the main experiments.
The iterative variant of CARD adds complexity not needed for our research
question. Per Zhang et al. 2024 §§2.2–2.4 and Table 3 of the paper.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..generator import Generator, LatencyProxy
from ..prompt import build_fim_prompt
from ..retriever import BM25Retriever, make_query
from .estimator import Estimator
from .features import extract_features

EPSILON = 1e-6


def is_retrieve(
    estimator: Estimator,
    token_probs,
    token_entropies,
    t_rag: float,
) -> bool:
    """True iff retrieval is needed (i.e., ŝ < T_RAG)."""
    feats = extract_features(token_probs, token_entropies)
    s_hat = float(estimator.predict(feats)[0])
    return s_hat < t_rag


def select(s_hat_i: float, s_hat_j: float, t_acc: float) -> bool:
    """True iff we should KEEP the older generation ŷ^i (reject ŷ^j)."""
    return (s_hat_j / (s_hat_i + EPSILON)) < t_acc


@dataclass
class CARDOutput:
    prediction: str
    n_iterations: int                       # 0 = zero-shot kept; 1 = RG1 accepted
    s_hats: list[float] = field(default_factory=list)
    retrieved_at_iter: list[int] = field(default_factory=list)
    latency_ms: float = 0.0


def card_pipeline(
    generator: Generator,
    retriever: BM25Retriever,
    estimator: Estimator,
    x_left: str,
    x_right: str,
    t_rag_schedule: list[float] | None = None,
    t_acc_schedule: list[float] | None = None,
    max_iter: int = 1,
    model_family: str = "qwen",
    top_k: int = 10,
) -> CARDOutput:
    """Run CARD's Algorithm 1 (single-RAG by default).

    Args:
        t_rag_schedule: only ``t_rag_schedule[0]`` is consulted for max_iter=1.
        t_acc_schedule: same.
        max_iter: 1 for single-RAG (project default).
    """
    if t_rag_schedule is None:
        t_rag_schedule = [0.9]
    if t_acc_schedule is None:
        t_acc_schedule = [0.8]

    s_hats: list[float] = []
    predictions: list[str] = []
    retrieved_at: list[int] = []
    t0 = time.perf_counter()
    gen = LatencyProxy(generator)

    # Iteration 0: zero-shot
    prompt = build_fim_prompt(x_left, x_right, retrieved=None, model_family=model_family)
    g0 = gen.generate(prompt)
    feats0 = extract_features(g0.token_probs, g0.token_entropies)
    s_hat_0 = float(estimator.predict(feats0)[0])
    s_hats.append(s_hat_0)
    predictions.append(g0.prediction)

    if max_iter >= 1 and s_hat_0 < t_rag_schedule[0]:
        query = make_query(x_left)
        retrieved = retriever.retrieve(query, top_k=top_k)
        prompt_rag = build_fim_prompt(
            x_left, x_right, retrieved=retrieved, model_family=model_family
        )
        g1 = gen.generate(prompt_rag)
        feats1 = extract_features(g1.token_probs, g1.token_entropies)
        s_hat_1 = float(estimator.predict(feats1)[0])
        s_hats.append(s_hat_1)
        predictions.append(g1.prediction)
        retrieved_at.append(1)

        if select(s_hat_0, s_hat_1, t_acc_schedule[0]):
            final_prediction = predictions[0]
            n_iterations = 0
        else:
            final_prediction = predictions[1]
            n_iterations = 1
    else:
        final_prediction = predictions[0]
        n_iterations = 0

    return CARDOutput(
        prediction=final_prediction,
        n_iterations=n_iterations,
        s_hats=s_hats,
        retrieved_at_iter=retrieved_at,
        # Full-pipeline latency: generation (cache-robust) + estimator gate,
        # select, and retrieval as live wall-clock.
        latency_ms=gen.reported_ms + (time.perf_counter() - t0 - gen.gen_wall_s) * 1000.0,
    )
