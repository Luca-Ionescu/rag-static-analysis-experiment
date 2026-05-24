"""Basic static-analysis tests (IMPLEMENTATION_GUIDE.md §10.5).

Under the unified Tier 1 design (no symbol-table dependency at the trigger
level), both ``unresolved_identifiers`` and ``cross_file_identifiers`` are
folded into a single ``out_of_scope_identifiers`` list and ``fires`` is
based on its significant subset. The cross-file vs unresolved distinction
no longer affects the trigger; both warrant retrieval.
"""
from __future__ import annotations

from adaptive_retrieval.static_analysis.analyzer import PredictionAnalyzer
from adaptive_retrieval.static_analysis.scope import InFileScopeAnalyzer
from adaptive_retrieval.static_analysis.symbol_table import RepositorySymbolTable


def test_unresolved_identifier_fires(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def known_func():\n    pass\n")

    syms = RepositorySymbolTable(repo)
    scope = InFileScopeAnalyzer()
    analyzer = PredictionAnalyzer(scope, syms)

    x_left = "def use_it():\n    result = "
    x_right = "\n    return result\n"
    prediction = "totally_made_up_function()"

    r = analyzer.analyze(prediction, x_left, x_right)
    assert r.fires
    assert "totally_made_up_function" in r.out_of_scope_identifiers
    assert "totally_made_up_function" in r.significant_out_of_scope


def test_in_file_name_does_not_fire(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    syms = RepositorySymbolTable(repo)
    scope = InFileScopeAnalyzer()
    analyzer = PredictionAnalyzer(scope, syms)

    x_left = "def helper():\n    return 1\n\ndef caller():\n    x = "
    x_right = "\n    return x\n"
    prediction = "helper()"  # 'helper' is defined in the file

    r = analyzer.analyze(prediction, x_left, x_right)
    assert not r.fires


def test_cross_file_name_used_in_significant_position_fires(tmp_path):
    """A name defined elsewhere in the repo but not in the in-file scope
    is out of scope at the hole. Tier 1 fires regardless of whether the
    name happens to be in the symbol table."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "lib.py").write_text("def cross_file_func():\n    pass\n")
    (repo / "main.py").write_text("from lib import cross_file_func\n")

    syms = RepositorySymbolTable(repo)
    analyzer = PredictionAnalyzer(InFileScopeAnalyzer(), syms)

    # X_left does NOT import cross_file_func; prediction calls it.
    r = analyzer.analyze(
        prediction="cross_file_func()",
        x_left="def f():\n    return ",
        x_right="\n",
    )
    assert r.fires
    assert "cross_file_func" in r.out_of_scope_identifiers
    assert "cross_file_func" in r.significant_out_of_scope


def test_cross_file_bare_use_does_not_fire(tmp_path):
    """A cross-file name used only as a bare value (not in a significant
    position) is in ``out_of_scope_identifiers`` but not in the significant
    subset, so the cascade does not fire."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "lib.py").write_text("REPO_CONST = 42\n")

    syms = RepositorySymbolTable(repo)
    analyzer = PredictionAnalyzer(InFileScopeAnalyzer(), syms)
    r = analyzer.analyze(
        prediction="x + REPO_CONST",
        x_left="def f(x):\n    return ",
        x_right="\n",
    )
    assert not r.fires
    # Still in the loose list for diagnostics / RSP.
    assert "REPO_CONST" in r.out_of_scope_identifiers
    assert "REPO_CONST" not in r.significant_out_of_scope
