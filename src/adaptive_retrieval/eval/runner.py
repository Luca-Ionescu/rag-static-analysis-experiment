"""Experiment runner: runs a single config over a dataset, writes JSONL.

Configs (IMPLEMENTATION_GUIDE §14.1):
    C1_no_retrieve     — generator on in-file prompt only
    C2_always_retrieve — generator with BM25 top-k chunks
    C3_card            — CARD single-RAG
    C4_cascade         — CARD + static analysis (our contribution)
    C5_static_only     — retrieve only when static analysis fires
    C6_oracle          — run both no- and always-retrieve; keep higher-ES one

Per-instance JSONL schema follows §15.1. Aggregates follow §15.2.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import jsonlines

from ..baselines import always_retrieve_baseline, no_retrieve_baseline
from ..card.estimator import Estimator
from ..card.pipeline import card_pipeline
from ..cascade import cascade_pipeline
from ..generator import Generator
from ..metrics import (
    edit_similarity,
    exact_match,
    hallucination_flag,
    identifier_f1,
    repository_symbol_precision,
)
from ..retriever import BM25Retriever
from ..static_analysis.analyzer import PredictionAnalyzer
from ..static_analysis.scope import InFileScopeAnalyzer
from ..static_analysis.symbol_table import RepositorySymbolTable
from .datasets import Instance

VALID_CONFIGS: tuple[str, ...] = (
    "C1_no_retrieve",
    "C2_always_retrieve",
    "C3_card",
    "C4_cascade",
    "C5_static_only",
    "C6_oracle",
)


@dataclass
class RunSummary:
    config: str
    dataset: str
    n_instances: int
    n_retrieved: int
    percent_retrieval: float
    mean_latency_ms: float
    metrics: dict[str, float]


# ---------- per-instance dispatch ----------

def _build_record(
    inst: Instance,
    prediction: str,
    retrieved: bool,
    trigger_reason: str,
    latency_ms: float,
    analyzer: PredictionAnalyzer,
    s_hat_0: float | None = None,
    static_unresolved: list[str] | None = None,
    static_crossfile: list[str] | None = None,
) -> dict:
    return {
        "instance_id": inst.instance_id,
        "repository": inst.repository,
        "target_file": inst.target_file,
        "ground_truth": inst.ground_truth,
        "prediction": prediction,
        "retrieved": retrieved,
        "trigger_reason": trigger_reason,
        "s_hat_0": s_hat_0,
        "static_unresolved": static_unresolved or [],
        "static_crossfile": static_crossfile or [],
        "metrics": {
            "exact_match": exact_match(inst.ground_truth, prediction),
            "edit_similarity": edit_similarity(inst.ground_truth, prediction),
            "identifier_f1": identifier_f1(inst.ground_truth, prediction),
            "repo_symbol_precision": repository_symbol_precision(
                prediction, inst.x_left, inst.x_right, analyzer
            ),
            "hallucinated": hallucination_flag(
                prediction, inst.x_left, inst.x_right, analyzer
            ),
        },
        "latency_ms": latency_ms,
    }


def _run_single_instance(
    config: str,
    inst: Instance,
    generator: Generator,
    retriever: BM25Retriever,
    analyzer: PredictionAnalyzer,
    estimator: Estimator | None,
    model_family: str,
    t_rag: float,
    top_k: int,
) -> dict:
    if config == "C1_no_retrieve":
        out = no_retrieve_baseline(generator, inst.x_left, inst.x_right, model_family)
        return _build_record(
            inst, out.prediction, retrieved=False, trigger_reason="none",
            latency_ms=out.latency_ms, analyzer=analyzer,
        )

    if config == "C2_always_retrieve":
        out = always_retrieve_baseline(
            generator, retriever, inst.x_left, inst.x_right, model_family, top_k=top_k
        )
        return _build_record(
            inst, out.prediction, retrieved=True, trigger_reason="always",
            latency_ms=out.latency_ms, analyzer=analyzer,
        )

    if config == "C3_card":
        if estimator is None:
            raise ValueError("C3_card requires an estimator")
        card_out = card_pipeline(
            generator, retriever, estimator,
            x_left=inst.x_left, x_right=inst.x_right,
            t_rag_schedule=[t_rag], t_acc_schedule=[0.8],
            model_family=model_family, top_k=top_k,
        )
        retrieved = bool(card_out.retrieved_at_iter)
        return _build_record(
            inst, card_out.prediction, retrieved=retrieved,
            trigger_reason="card" if retrieved else "none",
            latency_ms=card_out.latency_ms, analyzer=analyzer,
            s_hat_0=card_out.s_hats[0] if card_out.s_hats else None,
        )

    if config == "C4_cascade":
        if estimator is None:
            raise ValueError("C4_cascade requires an estimator")
        casc = cascade_pipeline(
            generator, retriever, estimator, analyzer,
            x_left=inst.x_left, x_right=inst.x_right,
            t_rag=t_rag, model_family=model_family, top_k=top_k,
        )
        return _build_record(
            inst, casc.prediction, retrieved=casc.retrieved,
            trigger_reason=casc.trigger_reason, latency_ms=casc.latency_ms,
            analyzer=analyzer, s_hat_0=casc.s_hat_0,
            static_unresolved=casc.static_unresolved,
            static_crossfile=casc.static_crossfile,
        )

    if config == "C5_static_only":
        # Generate zero-shot, then static-analyze. If static fires, retrieve.
        out_zs = no_retrieve_baseline(generator, inst.x_left, inst.x_right, model_family)
        sa = analyzer.analyze(out_zs.prediction, inst.x_left, inst.x_right)
        if sa.fires:
            out_rag = always_retrieve_baseline(
                generator, retriever, inst.x_left, inst.x_right, model_family, top_k=top_k
            )
            return _build_record(
                inst, out_rag.prediction, retrieved=True, trigger_reason="static",
                latency_ms=out_zs.latency_ms + out_rag.latency_ms,
                analyzer=analyzer,
                static_unresolved=list(sa.unresolved_identifiers),
                static_crossfile=list(sa.cross_file_identifiers),
            )
        return _build_record(
            inst, out_zs.prediction, retrieved=False, trigger_reason="none",
            latency_ms=out_zs.latency_ms, analyzer=analyzer,
        )

    if config == "C6_oracle":
        out_no = no_retrieve_baseline(generator, inst.x_left, inst.x_right, model_family)
        out_yes = always_retrieve_baseline(
            generator, retriever, inst.x_left, inst.x_right, model_family, top_k=top_k
        )
        es_no = edit_similarity(inst.ground_truth, out_no.prediction)
        es_yes = edit_similarity(inst.ground_truth, out_yes.prediction)
        chose_rag = es_yes > es_no
        chosen = out_yes if chose_rag else out_no
        return _build_record(
            inst, chosen.prediction, retrieved=chose_rag, trigger_reason="oracle",
            latency_ms=out_no.latency_ms + out_yes.latency_ms,
            analyzer=analyzer,
        )

    raise ValueError(f"Unknown config: {config!r}. Valid: {VALID_CONFIGS}")


# ---------- top-level runner ----------

def run_experiment(
    config: str,
    dataset_name: str,
    instances: Iterable[Instance],
    generator: Generator,
    estimator: Estimator | None,
    output_path: str | Path,
    model_family: str = "qwen",
    t_rag: float = 0.9,
    top_k: int = 10,
    progress: bool = True,
) -> RunSummary:
    """Run one config over a dataset. Writes per-instance records to JSONL.

    Returns aggregate ``RunSummary``. Side effect: ``output_path`` is written.
    """
    if config not in VALID_CONFIGS:
        raise ValueError(f"Unknown config: {config!r}. Valid: {VALID_CONFIGS}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    iterator = instances
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(instances, desc=f"{config}/{dataset_name}")
        except ImportError:
            pass

    n = 0
    n_retrieved = 0
    sum_em = sum_es = sum_f1 = sum_rsp = sum_hall = 0.0
    sum_latency = 0.0

    with jsonlines.open(output_path, "w") as writer:
        for inst in iterator:
            retriever = BM25Retriever(inst.repo_files)
            analyzer = PredictionAnalyzer(
                InFileScopeAnalyzer(),
                RepositorySymbolTable.from_files(inst.repo_files),
            )
            record = _run_single_instance(
                config, inst, generator, retriever, analyzer, estimator,
                model_family, t_rag, top_k,
            )
            record["dataset"] = dataset_name
            record["config"] = config
            writer.write(record)

            n += 1
            if record["retrieved"]:
                n_retrieved += 1
            m = record["metrics"]
            sum_em += float(m["exact_match"])
            sum_es += float(m["edit_similarity"])
            sum_f1 += float(m["identifier_f1"])
            sum_rsp += float(m["repo_symbol_precision"])
            sum_hall += float(m["hallucinated"])
            sum_latency += float(record["latency_ms"])

    return RunSummary(
        config=config,
        dataset=dataset_name,
        n_instances=n,
        n_retrieved=n_retrieved,
        percent_retrieval=100.0 * n_retrieved / n if n else 0.0,
        mean_latency_ms=sum_latency / n if n else 0.0,
        metrics={
            "exact_match": sum_em / n if n else 0.0,
            "edit_similarity": sum_es / n if n else 0.0,
            "identifier_f1": sum_f1 / n if n else 0.0,
            "repo_symbol_precision": sum_rsp / n if n else 0.0,
            "hallucination_rate": sum_hall / n if n else 0.0,
        },
    )


def aggregate_from_jsonl(path: str | Path) -> RunSummary:
    """Recompute aggregates from a JSONL written by ``run_experiment``.

    Useful for re-running metrics without re-running the generator.
    """
    p = Path(path)
    records: list[dict] = []
    with jsonlines.open(p) as r:
        for rec in r:
            records.append(rec)

    n = len(records)
    if n == 0:
        raise ValueError(f"No records in {p}")
    config = records[0].get("config", "")
    dataset = records[0].get("dataset", "")
    n_retrieved = sum(1 for r in records if r["retrieved"])
    return RunSummary(
        config=config,
        dataset=dataset,
        n_instances=n,
        n_retrieved=n_retrieved,
        percent_retrieval=100.0 * n_retrieved / n,
        mean_latency_ms=sum(r.get("latency_ms", 0.0) for r in records) / n,
        metrics={
            "exact_match": sum(float(r["metrics"]["exact_match"]) for r in records) / n,
            "edit_similarity": sum(float(r["metrics"]["edit_similarity"]) for r in records) / n,
            "identifier_f1": sum(float(r["metrics"]["identifier_f1"]) for r in records) / n,
            "repo_symbol_precision": sum(
                float(r["metrics"]["repo_symbol_precision"]) for r in records
            ) / n,
            "hallucination_rate": sum(
                float(r["metrics"]["hallucinated"]) for r in records
            ) / n,
        },
    )
