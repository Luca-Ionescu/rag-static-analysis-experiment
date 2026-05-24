"""Tests for the symbol-table-coverage improvements:

A1 — cross-instance per-repo chunk union (``build_repo_chunks_index``).
A3 — implicit top-level package whitelist (``IMPLICIT_TOPLEVEL_PACKAGES``).
"""
from __future__ import annotations

from adaptive_retrieval.eval.datasets import Instance, build_repo_chunks_index
from adaptive_retrieval.static_analysis.analyzer import PredictionAnalyzer
from adaptive_retrieval.static_analysis.scope import (
    IMPLICIT_TOPLEVEL_PACKAGES,
    InFileScopeAnalyzer,
)
from adaptive_retrieval.static_analysis.symbol_table import RepositorySymbolTable


# ---------- A1: per-repo chunk union ----------

def _make_instance(
    instance_id: str,
    repository: str,
    chunk_files: dict[str, str],
    target_file: str = "main.py",
    x_left: str = "def f():\n    return ",
    x_right: str = "\n",
    ground_truth: str = "1",
) -> Instance:
    repo_files = {**chunk_files, target_file: x_left + ground_truth + x_right}
    return Instance(
        x_left=x_left,
        x_right=x_right,
        ground_truth=ground_truth,
        repo_files=repo_files,
        instance_id=instance_id,
        target_file=target_file,
        repository=repository,
    )


def test_repo_index_unions_chunks_across_instances():
    """Two instances of the same repo with different chunk sets should
    produce a union containing both."""
    inst_a = _make_instance(
        "owner/repo/1",
        "owner/repo",
        {"utils.py": "def helper_a():\n    pass\n"},
    )
    inst_b = _make_instance(
        "owner/repo/2",
        "owner/repo",
        {"models.py": "def helper_b():\n    pass\n"},
    )
    index = build_repo_chunks_index([inst_a, inst_b])
    assert "owner/repo" in index
    repo = index["owner/repo"]
    assert "utils.py" in repo
    assert "models.py" in repo
    assert "helper_a" in repo["utils.py"]
    assert "helper_b" in repo["models.py"]


def test_repo_index_excludes_synthesised_target_files():
    """The synthesised current-file content is instance-specific and must
    not leak into the cross-instance union."""
    inst_a = _make_instance(
        "owner/repo/1",
        "owner/repo",
        {"utils.py": "def helper():\n    pass\n"},
        target_file="main_a.py",
    )
    inst_b = _make_instance(
        "owner/repo/2",
        "owner/repo",
        {"models.py": "def model():\n    pass\n"},
        target_file="main_b.py",
    )
    index = build_repo_chunks_index([inst_a, inst_b])
    repo = index["owner/repo"]
    # Target files are excluded.
    assert "main_a.py" not in repo
    assert "main_b.py" not in repo


def test_repo_index_concatenates_same_filename_across_instances():
    """If two instances both ship a chunk named utils.py with different
    content, both contents survive in the union."""
    inst_a = _make_instance(
        "owner/repo/1",
        "owner/repo",
        {"utils.py": "def helper_one():\n    pass\n"},
    )
    inst_b = _make_instance(
        "owner/repo/2",
        "owner/repo",
        {"utils.py": "def helper_two():\n    pass\n"},
    )
    index = build_repo_chunks_index([inst_a, inst_b])
    repo = index["owner/repo"]
    assert "helper_one" in repo["utils.py"]
    assert "helper_two" in repo["utils.py"]


def test_repo_index_segregates_by_repository():
    """Two repositories' chunks must not mix."""
    inst_a = _make_instance(
        "a/1", "owner/repo-a", {"a.py": "def only_in_a():\n    pass\n"}
    )
    inst_b = _make_instance(
        "b/1", "owner/repo-b", {"b.py": "def only_in_b():\n    pass\n"}
    )
    index = build_repo_chunks_index([inst_a, inst_b])
    assert "only_in_a" in index["owner/repo-a"]["a.py"]
    assert "a.py" not in index.get("owner/repo-b", {})


def test_repo_index_skips_instances_without_repository():
    """Instances missing the repository field shouldn't crash or contaminate."""
    inst_a = _make_instance("a/1", "owner/repo", {"a.py": "def x(): pass\n"})
    inst_no_repo = Instance(
        x_left="", x_right="", ground_truth="",
        repo_files={"orphan.py": "def y(): pass\n"},
        instance_id="orphan",
        target_file="orphan.py",
        repository=None,
    )
    index = build_repo_chunks_index([inst_a, inst_no_repo])
    assert "owner/repo" in index
    assert None not in index
    assert "" not in index


def test_repo_index_handles_empty_input():
    assert build_repo_chunks_index([]) == {}


# ---------- A1: integration with the analyzer ----------

def test_analyzer_with_repo_union_resolves_more_names():
    """A name defined in another instance's chunk should resolve under the
    union but be unresolved if we use only the current instance's chunks."""
    # Instance 1's chunks DON'T contain validators.py
    inst1 = _make_instance(
        "owner/repo/1",
        "owner/repo",
        {"helpers.py": "def help_one():\n    pass\n"},
    )
    # Instance 2's chunks DO contain validators.py
    inst2 = _make_instance(
        "owner/repo/2",
        "owner/repo",
        {"validators.py": "def validate_input():\n    pass\n"},
    )

    # Build the union — both chunks are in scope for either instance.
    index = build_repo_chunks_index([inst1, inst2])
    sym_files = {**index["owner/repo"], **inst1.repo_files}
    analyzer_union = PredictionAnalyzer(
        InFileScopeAnalyzer(),
        RepositorySymbolTable.from_files(sym_files),
    )

    # Without the union — only inst1's chunks
    analyzer_alone = PredictionAnalyzer(
        InFileScopeAnalyzer(),
        RepositorySymbolTable.from_files(inst1.repo_files),
    )

    prediction = "validate_input()"
    x_left = "def f():\n    return "
    x_right = "\n"

    r_union = analyzer_union.analyze(prediction, x_left, x_right)
    r_alone = analyzer_alone.analyze(prediction, x_left, x_right)

    # Tier 1 no longer depends on the symbol table — both analyzers should
    # report the same out-of-scope behaviour. validate_input is not visible
    # in the in-file scope, so it's out of scope in both cases.
    assert "validate_input" in r_union.out_of_scope_identifiers
    assert "validate_input" in r_alone.out_of_scope_identifiers
    assert r_union.significant_out_of_scope == r_alone.significant_out_of_scope
    assert r_union.fires == r_alone.fires


# ---------- A3: implicit top-level package whitelist ----------

def test_implicit_packages_includes_common_stdlib():
    """Sanity: a few canonical names should be present."""
    for name in ("os", "sys", "re", "json", "pathlib", "collections", "typing"):
        assert name in IMPLICIT_TOPLEVEL_PACKAGES


def test_implicit_packages_includes_common_third_party():
    """Sanity: popular libraries should be present."""
    for name in ("numpy", "pandas", "torch", "requests", "pytest"):
        assert name in IMPLICIT_TOPLEVEL_PACKAGES


def test_unimported_stdlib_receiver_does_not_fire(tmp_path):
    """`os.path.join(...)` with no `import os` in scope used to fire on `os`.
    Now `os` is in the implicit whitelist, so it resolves.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    analyzer = PredictionAnalyzer(
        InFileScopeAnalyzer(),
        RepositorySymbolTable.from_files({}),
    )
    r = analyzer.analyze(
        prediction="os.path.join('a', 'b')",
        x_left="def f():\n    return ",
        x_right="\n",
    )
    assert not r.fires
    assert "os" not in r.out_of_scope_identifiers


def test_unimported_numpy_receiver_does_not_fire(tmp_path):
    """`numpy.array([1,2])` without `import numpy` resolves via the whitelist."""
    repo = tmp_path / "repo"
    repo.mkdir()
    analyzer = PredictionAnalyzer(
        InFileScopeAnalyzer(),
        RepositorySymbolTable.from_files({}),
    )
    r = analyzer.analyze(
        prediction="numpy.array([1, 2])",
        x_left="def f():\n    return ",
        x_right="\n",
    )
    assert not r.fires
    assert "numpy" not in r.out_of_scope_identifiers


def test_random_unrecognised_package_still_fires(tmp_path):
    """A name that is NOT in the implicit whitelist and not imported still
    fires when used as a call target / receiver. Sanity for the whitelist
    not being a catch-all."""
    repo = tmp_path / "repo"
    repo.mkdir()
    analyzer = PredictionAnalyzer(
        InFileScopeAnalyzer(),
        RepositorySymbolTable.from_files({}),
    )
    r = analyzer.analyze(
        prediction="some_obscure_package.do_thing()",
        x_left="def f():\n    return ",
        x_right="\n",
    )
    # `some_obscure_package` is attribute receiver, not in whitelist → unresolved + significant → fires.
    assert r.fires
    assert "some_obscure_package" in r.significant_out_of_scope
