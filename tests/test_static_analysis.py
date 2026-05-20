"""Basic static-analysis tests (IMPLEMENTATION_GUIDE.md §10.5)."""
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
    assert "totally_made_up_function" in r.unresolved_identifiers


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


def test_cross_file_resolved_fires(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "lib.py").write_text("def cross_file_func():\n    pass\n")
    (repo / "main.py").write_text("from lib import cross_file_func\n")

    syms = RepositorySymbolTable(repo)
    scope = InFileScopeAnalyzer()
    # Default is fire_on_crossfile=False (the cross-file path is now an
    # ablation rather than a default signal). Opt in explicitly to exercise
    # the detection logic.
    analyzer = PredictionAnalyzer(scope, syms, fire_on_crossfile=True)

    # X_left does NOT import cross_file_func; prediction uses it
    x_left = "def f():\n    return "
    x_right = "\n"
    prediction = "cross_file_func()"

    r = analyzer.analyze(prediction, x_left, x_right)
    assert r.fires
    assert "cross_file_func" in r.cross_file_identifiers


def test_cross_file_resolved_does_not_fire_by_default(tmp_path):
    """With default fire_on_crossfile=False, a cross-file name alone is a
    diagnostic (still recorded in cross_file_identifiers) but does not fire."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "lib.py").write_text("def cross_file_func():\n    pass\n")

    analyzer = PredictionAnalyzer(InFileScopeAnalyzer(), RepositorySymbolTable(repo))
    r = analyzer.analyze(
        prediction="cross_file_func()",
        x_left="def f():\n    return ",
        x_right="\n",
    )
    assert not r.fires
    assert "cross_file_func" in r.cross_file_identifiers
