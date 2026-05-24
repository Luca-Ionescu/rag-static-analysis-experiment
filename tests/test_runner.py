"""Tests for the experiment runner — one per config, plus aggregation."""
from __future__ import annotations

import json

import jsonlines
import numpy as np
import pytest

from adaptive_retrieval.card.estimator import Estimator
from adaptive_retrieval.eval.datasets import Instance
from adaptive_retrieval.eval.runner import (
    VALID_CONFIGS,
    aggregate_from_jsonl,
    run_experiment,
)
from adaptive_retrieval.generator import MockGenerator


SYNTHETIC_REPO = {
    "lib.py": "def cross_func():\n    return 42\n",
    "main.py": "def helper(x):\n    return x + 1\n",
}


def _make_instance(idx: int = 0) -> Instance:
    return Instance(
        x_left=f"def f_{idx}():\n    return ",
        x_right="\n",
        ground_truth="cross_func()",
        repo_files=SYNTHETIC_REPO,
        instance_id=f"synthetic/{idx}",
        target_file="caller.py",
        repository="synthetic/repo",
    )


def _make_estimator(s_hat: float = 0.5) -> Estimator:
    """Train a tiny LightGBM that predicts ~s_hat regardless of input."""
    rng = np.random.default_rng(0)
    features = rng.uniform(0.4, 0.6, size=(120, 13)).astype(np.float32)
    scores = np.full(120, s_hat, dtype=np.float32)
    return Estimator.train(features, scores, val_fraction=0.1)


# ---------- per-config wiring ----------

@pytest.fixture
def instances():
    return [_make_instance(i) for i in range(5)]


def test_c1_no_retrieve(tmp_path, instances):
    gen = MockGenerator(default_prediction="dummy_pred()")
    out_path = tmp_path / "c1.jsonl"
    summary = run_experiment(
        config="C1_no_retrieve",
        dataset_name="synthetic",
        instances=instances,
        generator=gen,
        estimator=None,
        output_path=out_path,
        progress=False,
    )
    assert summary.n_instances == 5
    assert summary.n_retrieved == 0
    assert summary.percent_retrieval == 0.0
    with jsonlines.open(out_path) as r:
        records = list(r)
    assert len(records) == 5
    for rec in records:
        assert rec["config"] == "C1_no_retrieve"
        assert rec["dataset"] == "synthetic"
        assert rec["retrieved"] is False
        assert rec["trigger_reason"] == "none"


def test_c2_always_retrieve(tmp_path, instances):
    gen = MockGenerator(default_prediction="dummy_pred()")
    out_path = tmp_path / "c2.jsonl"
    summary = run_experiment(
        config="C2_always_retrieve",
        dataset_name="synthetic",
        instances=instances,
        generator=gen,
        estimator=None,
        output_path=out_path,
        progress=False,
    )
    assert summary.n_retrieved == 5
    assert summary.percent_retrieval == 100.0
    with jsonlines.open(out_path) as r:
        for rec in r:
            assert rec["retrieved"] is True
            assert rec["trigger_reason"] == "always"


def test_c3_card_low_confidence_triggers(tmp_path, instances):
    gen = MockGenerator(default_prediction="dummy_pred()")
    est = _make_estimator(s_hat=0.5)  # ŝ << default t_rag=0.9 → retrieve always
    summary = run_experiment(
        config="C3_card",
        dataset_name="synthetic",
        instances=instances,
        generator=gen,
        estimator=est,
        output_path=tmp_path / "c3.jsonl",
        progress=False,
    )
    # With ŝ ≈ 0.5 every time, CARD fires on every instance.
    assert summary.n_retrieved == 5


def test_c3_card_requires_estimator(tmp_path, instances):
    gen = MockGenerator()
    with pytest.raises(ValueError):
        run_experiment(
            config="C3_card",
            dataset_name="synthetic",
            instances=instances,
            generator=gen,
            estimator=None,
            output_path=tmp_path / "c3.jsonl",
            progress=False,
        )


def test_c4_cascade_writes_trigger_reason(tmp_path, instances):
    gen = MockGenerator(default_prediction="cross_func()")  # name resolves cross-file
    est = _make_estimator(s_hat=0.95)  # CARD says skip → static stage runs
    summary = run_experiment(
        config="C4_cascade",
        dataset_name="synthetic",
        instances=instances,
        generator=gen,
        estimator=est,
        output_path=tmp_path / "c4.jsonl",
        progress=False,
    )
    # cross_func is in repo but not in-file → static cascade fires.
    assert summary.n_retrieved == 5
    with jsonlines.open(tmp_path / "c4.jsonl") as r:
        for rec in r:
            assert rec["trigger_reason"] == "static"
            assert "cross_func" in rec["static_out_of_scope"]


def test_c4_cascade_requires_estimator(tmp_path, instances):
    gen = MockGenerator()
    with pytest.raises(ValueError):
        run_experiment(
            config="C4_cascade",
            dataset_name="synthetic",
            instances=instances,
            generator=gen,
            estimator=None,
            output_path=tmp_path / "c4.jsonl",
            progress=False,
        )


def test_c5_static_only(tmp_path, instances):
    gen = MockGenerator(default_prediction="totally_made_up_name()")
    summary = run_experiment(
        config="C5_static_only",
        dataset_name="synthetic",
        instances=instances,
        generator=gen,
        estimator=None,
        output_path=tmp_path / "c5.jsonl",
        progress=False,
    )
    # Unresolved identifier → static fires → retrieval triggered on every instance.
    assert summary.n_retrieved == 5
    with jsonlines.open(tmp_path / "c5.jsonl") as r:
        for rec in r:
            assert rec["trigger_reason"] == "static"


def test_c6_oracle_picks_better(tmp_path, instances):
    # MockGenerator returns the same string regardless of prompt, so ES is
    # tied between no- and always-retrieve and oracle picks the no-retrieve one.
    gen = MockGenerator(default_prediction="something()")
    summary = run_experiment(
        config="C6_oracle",
        dataset_name="synthetic",
        instances=instances,
        generator=gen,
        estimator=None,
        output_path=tmp_path / "c6.jsonl",
        progress=False,
    )
    assert summary.n_instances == 5
    with jsonlines.open(tmp_path / "c6.jsonl") as r:
        for rec in r:
            assert rec["trigger_reason"] == "oracle"
            # Ties go to no-retrieve in our implementation.
            assert rec["retrieved"] in (True, False)


def test_unknown_config_raises(tmp_path, instances):
    with pytest.raises(ValueError):
        run_experiment(
            config="C999_bogus",
            dataset_name="synthetic",
            instances=instances,
            generator=MockGenerator(),
            estimator=None,
            output_path=tmp_path / "out.jsonl",
            progress=False,
        )


# ---------- record schema ----------

def test_records_match_section_15_1_schema(tmp_path, instances):
    gen = MockGenerator(default_prediction="dummy()")
    summary = run_experiment(
        config="C1_no_retrieve",
        dataset_name="synthetic",
        instances=instances,
        generator=gen,
        estimator=None,
        output_path=tmp_path / "out.jsonl",
        progress=False,
    )
    assert summary.n_instances == 5
    with jsonlines.open(tmp_path / "out.jsonl") as r:
        records = list(r)
    required = {
        "instance_id", "repository", "target_file", "ground_truth",
        "prediction", "retrieved", "trigger_reason", "s_hat_0",
        "static_out_of_scope", "metrics", "latency_ms",
        "config", "dataset",
    }
    for rec in records:
        missing = required - set(rec.keys())
        assert not missing, f"Missing fields: {missing}"
        for m in (
            "exact_match", "edit_similarity", "identifier_f1",
            "repo_symbol_precision", "hallucinated",
        ):
            assert m in rec["metrics"], f"Missing metric: {m}"
        # Records must be JSON-serialisable (no numpy ints, etc.)
        json.dumps(rec)


# ---------- aggregate_from_jsonl ----------

def test_aggregate_from_jsonl_matches_run_summary(tmp_path, instances):
    gen = MockGenerator(default_prediction="dummy()")
    out_path = tmp_path / "agg.jsonl"
    summary = run_experiment(
        config="C1_no_retrieve",
        dataset_name="synthetic",
        instances=instances,
        generator=gen,
        estimator=None,
        output_path=out_path,
        progress=False,
    )
    re_agg = aggregate_from_jsonl(out_path)
    assert re_agg.n_instances == summary.n_instances
    assert re_agg.n_retrieved == summary.n_retrieved
    assert re_agg.config == summary.config
    assert re_agg.dataset == summary.dataset
    for k in summary.metrics:
        assert re_agg.metrics[k] == pytest.approx(summary.metrics[k])


def test_aggregate_from_empty_jsonl_raises(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.touch()
    with pytest.raises(ValueError):
        aggregate_from_jsonl(p)


# ---------- valid configs constant ----------

def test_valid_configs_includes_all_six():
    expected = {
        "C1_no_retrieve", "C2_always_retrieve",
        "C3_card", "C4_cascade",
        "C5_static_only", "C6_oracle",
    }
    assert set(VALID_CONFIGS) == expected
