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

import time
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
    truncate_to_line_count,
)
from ..prompt import build_fim_prompt
from ..retriever import BM25Retriever, make_query
from ..static_analysis.analyzer import PredictionAnalyzer
from ..static_analysis.scope import InFileScopeAnalyzer
from ..static_analysis.symbol_table import RepositorySymbolTable
from .datasets import (
    LINE_COUNT_DATASETS,
    MULTILINE_DATASETS,
    Instance,
    build_repo_chunks_index,
)


def _truncation_for(dataset_name: str):
    """Pick the scoring-time truncation function for a dataset (None = score the
    raw prediction; single metric set)."""
    if dataset_name in MULTILINE_DATASETS:
        return truncate_to_function_body          # function body (dedent boundary)
    if dataset_name in LINE_COUNT_DATASETS:
        return truncate_to_line_count             # fixed-size block (gold line count)
    return None

# Configs whose generation is a single pass per instance with no adaptive,
# logit-dependent control flow — so all their prompts can be built up front and
# sent to the generator as one batch (vLLM continuous batching). The adaptive
# configs (C3/C4/C5/C6) stay on the per-instance path, but ride the shared
# generation cache: their zero-shot call hits C1 and their retrieved call hits
# C2, so batching C1+C2 first makes them cheap too.
_BATCHABLE_CONFIGS = ("C1_no_retrieve", "C2_always_retrieve")

# Configs whose per-instance logic genuinely needs the in-house static analyzer
# (the cascade's static gate / the static-only trigger). For every other config
# the analyzer is only used for the descriptive scope-precision + in-house
# hallucination flag, which are redundant with the sweep's pyflakes pass — so we
# skip the costly per-instance symbol-table build and stay generation-bound.
_STATIC_CONFIGS = ("C4_cascade", "C5_static_only")


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

def _score(inst: Instance, prediction: str, analyzer: PredictionAnalyzer | None) -> dict:
    """Compute the metric dict for one prediction string.

    When ``analyzer is None`` the static-analysis metrics (scope precision,
    hallucination flag) are skipped and given schema-preserving placeholders.
    Those depend on a per-instance repository symbol table (tree-sitter parse of
    every cross-file fragment), which dominates runtime at ~2.5s/instance and
    swamps the batched generator. The authoritative hallucination signal is
    recomputed once in the post-hoc sweep (pyflakes, on the complete
    prediction), so the baselines (C1/C2/C3) skip it here for throughput.
    """
    m = {
        "exact_match": exact_match(inst.ground_truth, prediction),
        "edit_similarity": edit_similarity(inst.ground_truth, prediction),
        "identifier_f1": identifier_f1(inst.ground_truth, prediction),
    }
    if analyzer is not None:
        m["repo_symbol_precision"] = repository_symbol_precision(
            prediction, inst.x_left, inst.x_right, analyzer
        )
        m["hallucinated"] = hallucination_flag(
            prediction, inst.x_left, inst.x_right, analyzer
        )
    else:
        m["repo_symbol_precision"] = 1.0   # placeholder; sweep is authoritative
        m["hallucinated"] = False          # placeholder; sweep is authoritative
    return m


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
    truncate_fn=None,
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
    if truncate_fn is not None:
        # Truncated task: the model over-generates past the target span (function
        # body for *_function, the fixed-size block for *_chunk), so the headline
        # metrics are scored on the truncated prediction and the raw, untruncated
        # metrics are kept alongside for comparison.
        pred_trunc = truncate_fn(inst.ground_truth, prediction)
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
    truncate_fn=None,
) -> dict:
    # Bind the truncation fn so every _rec(...) below emits both the truncated
    # (main) and raw metric sets for truncated datasets (function / chunk).
    from functools import partial
    _rec = partial(_build_record, truncate_fn=truncate_fn)
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


# ---------- shared helpers ----------

def _make_analyzer_factory(instances_list, use_repo_union):
    """Return ``analyzer(inst) -> PredictionAnalyzer`` with the per-repo symbol
    union (if enabled) pre-indexed once."""
    repo_index = build_repo_chunks_index(instances_list) if use_repo_union else {}

    def make(inst: Instance) -> PredictionAnalyzer:
        sym_files: dict[str, str] = {}
        if use_repo_union and inst.repository:
            sym_files.update(repo_index.get(inst.repository, {}))
        sym_files.update(inst.repo_files)
        return PredictionAnalyzer(
            InFileScopeAnalyzer(),
            RepositorySymbolTable.from_files(sym_files),
        )

    return make


def _batched_prompt(config: str, inst: Instance, model_family: str, top_k: int) -> str:
    """Build the FIM prompt for a batchable config exactly as its baseline does
    (so cache keys match the per-instance path and C3/C4 still hit the cache)."""
    if config == "C1_no_retrieve":
        return build_fim_prompt(
            inst.x_left, inst.x_right, retrieved=None, model_family=model_family
        )
    # C2_always_retrieve. exclude_file keeps the corpus cross-file only — the
    # loaders put the (gold-containing) current file into repo_files for the
    # symbol-table consumers, and it must never be retrievable.
    retriever = BM25Retriever(inst.repo_files, exclude_file=inst.target_file)
    chunks = retriever.retrieve(make_query(inst.x_left), top_k=top_k)
    return build_fim_prompt(
        inst.x_left, inst.x_right, retrieved=chunks, model_family=model_family
    )


def _run_experiment_batched(
    config, dataset_name, instances_list, generator, output_path,
    model_family, top_k, truncate_fn, make_analyzer, batch_size, progress,
) -> "RunSummary":
    """Batched fast-path for C1/C2: build all prompts, generate in chunks via
    ``generator.generate_batch`` (vLLM continuous batching), then score.

    Identical records to the per-instance path (same prompts -> same predictions
    and cache keys); only far faster. Writes JSONL per chunk so a disconnect
    mid-run keeps everything generated so far. Per-instance ``latency_ms`` is the
    generator's per-sequence latency when vLLM reports it, else the chunk
    wall-clock amortised over the chunk (the honest throughput-based per-item
    cost under batching); BM25 retrieval for C2 happens in the build loop and is
    not folded into per-item latency (it is ~ms vs seconds of generation).
    """
    retrieved_flag = config == "C2_always_retrieve"
    trigger = "always" if retrieved_flag else "none"

    starts = range(0, len(instances_list), batch_size)
    if progress:
        try:
            from tqdm import tqdm
            starts = tqdm(list(starts), desc=f"{config}/{dataset_name} [batch {batch_size}]")
        except ImportError:
            pass

    records: list[dict] = []
    n_retrieved = 0
    sum_latency = 0.0
    with jsonlines.open(output_path, "w") as writer:
        for start in starts:
            batch = instances_list[start:start + batch_size]
            prompts = [_batched_prompt(config, i, model_family, top_k) for i in batch]
            t0 = time.perf_counter()
            gens = generator.generate_batch(prompts)
            amort_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(batch))
            for inst, gen in zip(batch, gens):
                latency = gen.latency_ms if gen.latency_ms else amort_ms
                record = _build_record(
                    inst, gen.prediction, retrieved=retrieved_flag,
                    trigger_reason=trigger, latency_ms=latency,
                    analyzer=make_analyzer(inst) if make_analyzer else None,
                    truncate_fn=truncate_fn,
                )
                record["dataset"] = dataset_name
                record["config"] = config
                writer.write(record)
                records.append(record)
                if retrieved_flag:
                    n_retrieved += 1
                sum_latency += latency

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
    batch_size: int = 1,
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

    # Truncated datasets (function-body or fixed-size block) get truncated (main)
    # + raw metric sets per record; single-line datasets get one set.
    truncate_fn = _truncation_for(dataset_name)

    # Only build the (expensive) per-instance analyzer factory for configs that
    # actually consult the static analyzer; otherwise skip it entirely so the
    # symbol table is never built (the sweep computes hallucination post-hoc).
    needs_static = config in _STATIC_CONFIGS
    make_analyzer = (
        _make_analyzer_factory(instances_list, use_repo_union) if needs_static else None
    )

    # Fast path: the two single-pass baselines (C1/C2) are batched through the
    # generator (vLLM continuous batching) — ~order-of-magnitude faster than
    # one generate() call per instance, and produces identical records.
    if batch_size and batch_size > 1 and config in _BATCHABLE_CONFIGS:
        return _run_experiment_batched(
            config, dataset_name, instances_list, generator, output_path,
            model_family, top_k, truncate_fn, make_analyzer, batch_size, progress,
        )

    iterator = instances_list
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(instances_list, desc=f"{config}/{dataset_name}")
        except ImportError:
            pass

    records: list[dict] = []
    n_retrieved = 0
    sum_latency = 0.0

    with jsonlines.open(output_path, "w") as writer:
        for inst in iterator:
            retriever = BM25Retriever(inst.repo_files, exclude_file=inst.target_file)
            # Symbol table only built for configs that consult the static gate;
            # None otherwise (the sweep computes hallucination post-hoc).
            analyzer = make_analyzer(inst) if make_analyzer else None
            record = _run_single_instance(
                config, inst, generator, retriever, analyzer, estimator,
                model_family, t_rag, top_k, truncate_fn=truncate_fn,
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
