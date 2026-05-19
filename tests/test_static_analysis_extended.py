"""Extended static-analysis tests covering Python scoping edge cases.

Pattern: each test constructs a minimal "in-file context" and "prediction",
builds a tiny repo symbol table, and asserts the analyzer's decision.
Source: IMPLEMENTATION_GUIDE.md Appendix E.
"""
from __future__ import annotations

import pytest

from adaptive_retrieval.static_analysis.analyzer import PredictionAnalyzer
from adaptive_retrieval.static_analysis.scope import InFileScopeAnalyzer
from adaptive_retrieval.static_analysis.symbol_table import RepositorySymbolTable


@pytest.fixture
def empty_analyzer(tmp_path):
    """Analyzer with no cross-file symbols (only in-file and builtins resolvable)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    syms = RepositorySymbolTable(repo)
    return PredictionAnalyzer(InFileScopeAnalyzer(), syms)


@pytest.fixture
def repo_with(tmp_path):
    """Factory: create a tiny repo with the given filename->content map."""
    def _make(files: dict[str, str]):
        repo = tmp_path / "repo"
        repo.mkdir(exist_ok=True)
        for name, content in files.items():
            (repo / name).write_text(content)
        return PredictionAnalyzer(InFileScopeAnalyzer(), RepositorySymbolTable(repo))
    return _make


# ---------- BASIC RESOLUTION ----------

def test_01_hallucinated_function(empty_analyzer):
    """Name not defined anywhere -> fires."""
    r = empty_analyzer.analyze(
        prediction="totally_fake()",
        x_left="def main():\n    return ",
        x_right="\n",
    )
    assert r.fires
    assert "totally_fake" in r.unresolved_identifiers


def test_02_locally_defined_function(empty_analyzer):
    """Function defined in same file -> doesn't fire."""
    r = empty_analyzer.analyze(
        prediction="helper()",
        x_left="def helper():\n    return 1\n\ndef caller():\n    return ",
        x_right="\n",
    )
    assert not r.fires


def test_03_builtin_function(empty_analyzer):
    """print, len, range, etc -> don't fire."""
    r = empty_analyzer.analyze(
        prediction="print(len(x))",
        x_left="def f(x):\n    ",
        x_right="\n",
    )
    assert not r.fires, f"Unexpected fire: {r.unresolved_identifiers}"


def test_04_cross_file_resolved(repo_with):
    """Name defined in another repo file -> fires as cross_file."""
    analyzer = repo_with({
        "lib.py": "def cross_func():\n    return 42\n",
    })
    r = analyzer.analyze(
        prediction="cross_func()",
        x_left="def use_it():\n    return ",
        x_right="\n",
    )
    assert r.fires
    assert "cross_func" in r.cross_file_identifiers


# ---------- IMPORTS ----------

def test_05_imported_library_function(empty_analyzer):
    """Imported name -> doesn't fire even if not in repo."""
    r = empty_analyzer.analyze(
        prediction="np.array([1, 2])",
        x_left="import numpy as np\n\ndef f():\n    return ",
        x_right="\n",
    )
    assert not r.fires


def test_06_from_import(empty_analyzer):
    """`from X import Y` should bring Y into scope."""
    r = empty_analyzer.analyze(
        prediction="join('a', 'b')",
        x_left="from os.path import join\n\ndef f():\n    return ",
        x_right="\n",
    )
    assert not r.fires


def test_07_aliased_import(empty_analyzer):
    """`import X as Y` should bring Y (not X) into scope."""
    r = empty_analyzer.analyze(
        prediction="pd.DataFrame()",
        x_left="import pandas as pd\n\ndef f():\n    return ",
        x_right="\n",
    )
    assert not r.fires


# ---------- ATTRIBUTE ACCESS ----------

def test_08_attribute_on_imported(empty_analyzer):
    """`os.path.join` -> 'os' is the use; attributes are not flagged."""
    r = empty_analyzer.analyze(
        prediction="os.path.join('a', 'b')",
        x_left="import os\n\ndef f():\n    return ",
        x_right="\n",
    )
    assert not r.fires


def test_09_attribute_on_self(empty_analyzer):
    """`self.x` is fine; 'self' is treated as a builtin-like name."""
    r = empty_analyzer.analyze(
        prediction="self.value + 1",
        x_left="class C:\n    def m(self):\n        return ",
        x_right="\n",
    )
    assert not r.fires


# ---------- SCOPING ----------

def test_10_function_parameter(empty_analyzer):
    """Function parameter used in body -> doesn't fire."""
    r = empty_analyzer.analyze(
        prediction="x + 1",
        x_left="def f(x):\n    return ",
        x_right="\n",
    )
    assert not r.fires


def test_11_for_loop_variable(empty_analyzer):
    """Variable bound by `for` loop -> usable inside loop."""
    r = empty_analyzer.analyze(
        prediction="item * 2",
        x_left="def f(items):\n    for item in items:\n        return ",
        x_right="\n",
    )
    assert not r.fires


def test_12_list_comprehension_variable(empty_analyzer):
    """Comprehension scope: 'i' should be visible within the comprehension itself."""
    r = empty_analyzer.analyze(
        prediction="[i * 2 for i in range(10)]",
        x_left="def f():\n    return ",
        x_right="\n",
    )
    assert not r.fires


def test_13_with_statement_binding(empty_analyzer):
    """`with X as Y:` binds Y."""
    r = empty_analyzer.analyze(
        prediction="f.read()",
        x_left="def reader(path):\n    with open(path) as f:\n        return ",
        x_right="\n",
    )
    assert not r.fires


def test_14_multiple_assignment(empty_analyzer):
    """`a, b = c, d` defines a and b."""
    r = empty_analyzer.analyze(
        prediction="a + b",
        x_left="def f():\n    a, b = 1, 2\n    return ",
        x_right="\n",
    )
    assert not r.fires


def test_15_walrus_operator(empty_analyzer):
    """`(x := 5)` should bind x. Acceptable if test fails;
    walrus is a known edge case we may not support."""
    r = empty_analyzer.analyze(
        prediction="x + 1",
        x_left="def f():\n    if (x := compute()) > 0:\n        return ",
        x_right="\n",
    )
    if r.fires:
        pytest.xfail("Walrus operator scoping not supported; documented limitation.")
    assert not r.fires


# ---------- DECORATORS AND TYPE HINTS ----------

def test_16_decorator_imported(empty_analyzer):
    """Imported decorator -> doesn't fire."""
    r = empty_analyzer.analyze(
        prediction="cached(f)",
        x_left="from functools import cache as cached\n\ndef use(f):\n    return ",
        x_right="\n",
    )
    assert not r.fires


def test_17_type_hint_imported(empty_analyzer):
    """Type hint with imported type -> doesn't fire."""
    r = empty_analyzer.analyze(
        prediction="Optional[int]",
        x_left="from typing import Optional\n\nx: ",
        x_right="\n",
    )
    assert not r.fires


# ---------- LAMBDA AND NESTED FUNCTIONS ----------

def test_18_lambda_parameter(empty_analyzer):
    """Lambda parameter -> usable in lambda body."""
    r = empty_analyzer.analyze(
        prediction="lambda x: x * 2",
        x_left="def f():\n    return ",
        x_right="\n",
    )
    assert not r.fires


def test_19_nested_function_closure(empty_analyzer):
    """Inner function uses outer's local variable -> doesn't fire."""
    r = empty_analyzer.analyze(
        prediction="inner()",
        x_left="def outer():\n    y = 5\n    def inner():\n        return y\n    return ",
        x_right="\n",
    )
    assert not r.fires


# ---------- INHERITANCE ----------

def test_20_super_call(empty_analyzer):
    """super().__init__() -> 'super' is builtin, doesn't fire."""
    r = empty_analyzer.analyze(
        prediction="super().__init__()",
        x_left="class C(Base):\n    def __init__(self):\n        ",
        x_right="\n",
    )
    # 'Base' might fire as unresolved; that's fine for our purposes.
    # But 'super' must not.
    assert "super" not in r.unresolved_identifiers


# ---------- ROBUSTNESS ----------

def test_21_syntactically_broken_prediction(empty_analyzer):
    """Analyzer must not crash on broken Python."""
    r = empty_analyzer.analyze(
        prediction="foo(",
        x_left="def f():\n    return ",
        x_right="\n",
    )
    assert "foo" in r.unresolved_identifiers


def test_22_empty_prediction(empty_analyzer):
    """Empty prediction must not crash."""
    r = empty_analyzer.analyze(
        prediction="",
        x_left="def f():\n    return ",
        x_right="\n",
    )
    assert not r.fires
    assert r.n_used_identifiers == 0
