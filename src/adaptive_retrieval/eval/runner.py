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

from dataclasses import asdict, dataclass, is_dataclass
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
    truncate_to_function_body,
)
from ..retriever import BM25Retriever
from ..static_analysis.analyzer import PredictionAnalyzer
from ..static_analysis.scope import InFileScopeAnalyzer
from ..static_analysis.symbol_table import RepositorySymbolTable
from .datasets import MULTILINE_DATASETS, Instance, build_repo_chunks_index


def _serialise(items) -> list[dict]:
    """Convert a list of CallIssue / ImportIssue dataclasses to dicts."""
    out: list[dict] = []
    for it in items or ():
        if is_dataclass(it):
            out.append(asdict(it))
        elif isinstance(it, dict):
            out.append(it)
    return out


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
    # Aggregates over the raw (untruncated) per-instance metrics. Only populated
    # for multi-line datasets where ``metrics`` is the truncated (main) view;
    # None for single-line datasets where there is only one metric set.
    metrics_raw: dict[str, float] | None = None


def _mean_metrics(records: list[dict], key: str) -> dict[str, float]:
    """Average a per-instance metric block (``metrics`` or ``metrics_raw``)."""
    n = len(records)
    present = [r for r in records if key in r]
    if not present:
        return {}
    m = len(present)
    return {
        "exact_match": sum(float(r[key]["exact_match"]) for r in present) / m,
        "edit_similarity": sum(float(r[key]["edit_similarity"]) for r in present) / m,
        "identifier_f1": sum(float(r[key]["identifier_f1"]) for r in present) / m,
        "repo_symbol_precision": sum(float(r[key]["repo_symbol_precision"]) for r in present) / m,
        "hallucination_rate": sum(float(r[key]["hallucinated"]) for r in present) / m,
    }


# ---------- per-instance dispatch ----------

def _score(inst: Instance, prediction: str, analyzer: PredictionAnalyzer) -> dict:
    """Compute the metric dict for one prediction string."""
    return {
        "exact_match": exact_match(inst.ground_truth, prediction),
        "edit_similarity": edit_similarity(inst.ground_truth, prediction),
        "identifier_f1": identifier_f1(inst.ground_truth, prediction),
        "repo_symbol_precision": repository_symbol_precision(
            prediction, inst.x_left, inst.x_right, analyzer
        ),
        "hallucinated": hallucination_flag(
            prediction, inst.x_left, inst.x_right, analyzer
        ),
    }


def _build_record(
    inst: Instance,
    prediction: str,
    retrieved: bool,
    trigger_reason: str,
    latency_ms: float,
    analyzer: PredictionAnalyzer,
    s_hat_0: float | None = None,
    s_hat_1: float | None = None,
    static_out_of_scope: list[str] | None = None,
    signature_issues: list | None = None,
    import_issues: list | None = None,
    multiline: bool = False,
) -> dict:
    record = {
        "instance_id": inst.instance_id,
        "repository": inst.repository,
        "target_file": inst.target_file,
        "ground_truth": inst.ground_truth,
        "prediction": prediction,
        "retrieved": retrieved,
        "trigger_reason": trigger_reason,
        "s_hat_0": s_hat_0,
        # ŝ₁ = predicted ES of the retrieved generation (None when no retrieval
        # happened). Emitting it makes the CARD (C3) T_RAG/t_acc sweep fully
        # replayable from JSONL alone — see scripts/08_selectivity_check.py and
        # the post-hoc re-thresholding path. It does NOT change CARD's logic;
        # the value is already computed by card_pipeline.
        "s_hat_1": s_hat_1,
        "static_out_of_scope": static_out_of_scope or [],
        "signature_issues": signature_issues or [],
        "import_issues": import_issues or [],
        "latency_ms": latency_ms,
    }
    if multiline:
        # Multi-line task (RepoEval-function): the model over-generates past the
        # function body, so the headline metrics are scored on the prediction
        # truncated to the end of the generated body (dedent boundary). The raw,
        # untruncated metrics are kept alongside for comparison.
        pred_trunc = truncate_to_function_body(inst.ground_truth, prediction)
        record["prediction_truncated"] = pred_trunc
        record["metrics"] = _score(inst, pred_trunc, analyzer)        # main
        record["metrics_raw"] = _score(inst, prediction, analyzer)    # secondary
    else:
        record["metrics"] = _score(inst, prediction, analyzer)
    return record


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
    multiline: bool = False,
) -> dict:
    # Bind the multi-line flag so every _rec(...) below emits both the
    # truncated (main) and raw metric sets for function-level datasets.
    from functools import partial
    _rec = partial(_build_record, multiline=multiline)
    if config == "C1_no_retrieve":
        out = no_retrieve_baseline(generator, inst.x_left, inst.x_right, model_family)
        return _rec(
            inst, out.prediction, retrieved=False, trigger_reason="none",
            latency_ms=out.latency_ms, analyzer=analyzer,
        )

    if config == "C2_always_retrieve":
        out = always_retrieve_baseline(
            generator, retriever, inst.x_left, inst.x_right, model_family, top_k=top_k
        )
        return _rec(
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
        return _rec(
            inst, card_out.prediction, retrieved=retrieved,
            trigger_reason="card" if retrieved else "none",
            latency_ms=card_out.latency_ms, analyzer=analyzer,
            s_hat_0=card_out.s_hats[0] if card_out.s_hats else None,
            s_hat_1=card_out.s_hats[1] if len(card_out.s_hats) > 1 else None,
        )

    if config == "C4_cascade":
        if estimator is None:
            raise ValueError("C4_cascade requires an estimator")
        casc = cascade_pipeline(
            generator, retriever, estimator, analyzer,
            x_left=inst.x_left, x_right=inst.x_right,
            t_rag=t_rag, model_family=model_family, top_k=top_k,
        )
        return _rec(
            inst, casc.prediction, retrieved=casc.retrieved,
            trigger_reason=casc.trigger_reason, latency_ms=casc.latency_ms,
            analyzer=analyzer, s_hat_0=casc.s_hat_0,
            static_out_of_scope=casc.static_out_of_scope,
            signature_issues=casc.signature_issues,
            import_issues=casc.import_issues,
        )

    if config == "C5_static_only":
        # Generate zero-shot, then static-analyze. If static fires, retrieve.
        out_zs = no_retrieve_baseline(generator, inst.x_left, inst.x_right, model_family)
        sa = analyzer.analyze(out_zs.prediction, inst.x_left, inst.x_right)
        if sa.fires:
            out_rag = always_retrieve_baseline(
                generator, retriever, inst.x_left, inst.x_right, model_family, top_k=top_k
            )
            return _rec(
                inst, out_rag.prediction, retrieved=True, trigger_reason="static",
                latency_ms=out_zs.latency_ms + out_rag.latency_ms,
                analyzer=analyzer,
                static_out_of_scope=list(sa.significant_out_of_scope),
                signature_issues=_serialise(sa.signature_issues),
                import_issues=_serialise(sa.import_issues),
            )
        return _rec(
            inst, out_zs.prediction, retrieved=False, trigger_reason="none",
            latency_ms=out_zs.latency_ms, analyzer=analyzer,
            static_out_of_scope=list(sa.significant_out_of_scope),
            signature_issues=_serialise(sa.signature_issues),
            import_issues=_serialise(sa.import_issues),
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
        return _rec(
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
    use_repo_union: bool = True,
) -> RunSummary:
    """Run one config over a dataset. Writes per-instance records to JSONL.

    Args:
        use_repo_union: when True, the per-instance analyzer's symbol table
            sees every cross-file chunk shipped by *any* instance of the same
            repository (not just the current instance's 5 chunks). This is
            the default since it materially reduces false-positive
            hallucination flags without changing what gets retrieved at
            inference. Disable for the static-strictness ablation (A1).
    """
    if config not in VALID_CONFIGS:
        raise ValueError(f"Unknown config: {config!r}. Valid: {VALID_CONFIGS}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    instances_list = list(instances)
    repo_index = build_repo_chunks_index(instances_list) if use_repo_union else {}

    iterator = instances_list
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(instances_list, desc=f"{config}/{dataset_name}")
        except ImportError:
            pass

    # Multi-line datasets (function-body completion) get truncated (main) +
    # raw metric sets per record; single-line datasets get one set.
    multiline = dataset_name in MULTILINE_DATASETS

    records: list[dict] = []
    n_retrieved = 0
    sum_latency = 0.0

    with jsonlines.open(output_path, "w") as writer:
        for inst in iterator:
            retriever = BM25Retriever(inst.repo_files)
            # Symbol table: per-repo union (if enabled) merged with this
            # instance's target file. BM25 retrieval is unchanged — that uses
            # only the instance's own 5 chunks.
            sym_files: dict[str, str] = {}
            if use_repo_union and inst.repository:
                sym_files.update(repo_index.get(inst.repository, {}))
            sym_files.update(inst.repo_files)
            analyzer = PredictionAnalyzer(
                InFileScopeAnalyzer(),
                RepositorySymbolTable.from_files(sym_files),
            )
            record = _run_single_instance(
                config, inst, generator, retriever, analyzer, estimator,
                model_family, t_rag, top_k, multiline=multiline,
            )
            record["dataset"] = dataset_name
            record["config"] = config
            writer.write(record)
            records.append(record)
            if record["retrieved"]:
                n_retrieved += 1
            sum_latency += float(record["latency_ms"])

    n = len(records)
    return RunSummary(
        config=config,
        dataset=dataset_name,
        n_instances=n,
        n_retrieved=n_retrieved,
        percent_retrieval=100.0 * n_retrieved / n if n else 0.0,
        mean_latency_ms=sum_latency / n if n else 0.0,
        metrics=_mean_metrics(records, "metrics"),
        metrics_raw=_mean_metrics(records, "metrics_raw") or None,
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
        metrics=_mean_metrics(records, "metrics"),
        metrics_raw=_mean_metrics(records, "metrics_raw") or None,
    )
