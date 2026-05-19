"""Tests for eval/datasets.py — runs against the real shipped JSONL."""
from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_retrieval.eval.datasets import (
    DEFAULT_CCE_PYTHON_PATH,
    Instance,
    load_crosscodeeval_python,
)

_REAL_DATA = Path(DEFAULT_CCE_PYTHON_PATH)

requires_data = pytest.mark.skipif(
    not _REAL_DATA.exists(),
    reason=f"CrossCodeEval data missing at {_REAL_DATA}",
)


@requires_data
def test_loader_yields_instances():
    it = load_crosscodeeval_python()
    first = next(it)
    assert isinstance(first, Instance)


@requires_data
def test_first_instance_shape():
    inst = next(load_crosscodeeval_python())
    assert inst.x_left, "x_left should be non-empty"
    assert inst.x_right is not None, "x_right should exist (may be empty)"
    assert inst.ground_truth, "ground_truth should be non-empty"
    assert inst.instance_id, "instance_id should be set"
    assert inst.target_file, "target_file should be set"
    # rg1_bm25 variant ships chunks; expect at least one.
    assert len(inst.repo_files) >= 1


@requires_data
def test_repo_files_includes_target_file():
    inst = next(load_crosscodeeval_python())
    assert inst.target_file in inst.repo_files
    # The synthesised target should contain the ground truth somewhere.
    assert inst.ground_truth in inst.repo_files[inst.target_file]


@requires_data
def test_repo_files_chunks_are_strings():
    inst = next(load_crosscodeeval_python())
    for path, content in inst.repo_files.items():
        assert isinstance(path, str) and path
        assert isinstance(content, str)


@requires_data
def test_loader_streams_many_instances():
    """Smoke: pull the first 20 to check streaming works without crashes."""
    it = load_crosscodeeval_python()
    instances = [next(it) for _ in range(20)]
    assert len(instances) == 20
    # task_ids should be unique across the batch
    ids = [i.instance_id for i in instances]
    assert len(set(ids)) == 20


@requires_data
def test_can_exclude_target_file():
    inst = next(load_crosscodeeval_python(include_target_file=False))
    assert inst.target_file not in inst.repo_files


def test_loader_raises_on_missing_path(tmp_path):
    bogus = tmp_path / "no_such_file.jsonl"
    with pytest.raises(FileNotFoundError):
        next(load_crosscodeeval_python(bogus))
