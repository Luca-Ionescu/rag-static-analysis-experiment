"""Tests for the structural-significance filter on the static-analysis trigger.

The analyzer's ``fires`` signal now ignores unresolved bare identifiers that
appear only in expression positions (binary-op operands, call arguments,
return values). It only fires on identifiers in structurally significant
positions: call target, attribute receiver, subscript value.

These tests pin down the new behaviour. Existing tests in
``test_static_analysis*.py`` continue to pass because they use function-call
predictions for the "should fire" cases.
"""
from __future__ import annotations

import pytest

from adaptive_retrieval.static_analysis.analyzer import PredictionAnalyzer
from adaptive_retrieval.static_analysis.scope import InFileScopeAnalyzer
from adaptive_retrieval.static_analysis.symbol_table import RepositorySymbolTable


@pytest.fixture
def empty_analyzer(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    return PredictionAnalyzer(InFileScopeAnalyzer(), RepositorySymbolTable(repo))


@pytest.fixture
def repo_with(tmp_path):
    def _make(files):
        repo = tmp_path / "repo"
        repo.mkdir(exist_ok=True)
        for name, content in files.items():
            (repo / name).write_text(content)
        return PredictionAnalyzer(InFileScopeAnalyzer(), RepositorySymbolTable(repo))
    return _make


# ---------- bare identifiers DO NOT fire any longer ----------

def test_bare_unresolved_in_binop_does_not_fire(empty_analyzer):
    """`x + made_up` — both are binary-op operands. Even if made_up is unresolved,
    no significant-position fire.
    """
    r = empty_analyzer.analyze(
        prediction="x + made_up_var",
        x_left="def f(x):\n    return ",
        x_right="\n",
    )
    assert not r.fires
    # But the loose list should still record it for diagnostics.
    assert "made_up_var" in r.unresolved_identifiers


def test_bare_unresolved_arg_to_known_call_does_not_fire(empty_analyzer):
    """`print(unresolved_var)` — print is the call target (builtin, resolves);
    unresolved_var is a bare arg, not significant.
    """
    r = empty_analyzer.analyze(
        prediction="print(unresolved_var)",
        x_left="def f():\n    ",
        x_right="\n",
    )
    assert not r.fires
    assert "unresolved_var" in r.unresolved_identifiers


def test_bare_unresolved_return_value_does_not_fire(empty_analyzer):
    """`return some_local_thing` — bare identifier in return position."""
    r = empty_analyzer.analyze(
        prediction="some_local_thing",
        x_left="def f():\n    return ",
        x_right="\n",
    )
    assert not r.fires
    assert "some_local_thing" in r.unresolved_identifiers


def test_bare_unresolved_assignment_rhs_does_not_fire(empty_analyzer):
    """`result = some_unresolved` — bare identifier on assignment RHS."""
    r = empty_analyzer.analyze(
        prediction="result = some_unresolved",
        x_left="def f():\n    ",
        x_right="\n",
    )
    assert not r.fires


def test_bare_unresolved_in_list_does_not_fire(empty_analyzer):
    """`[a, b, c]` — all bare list elements."""
    r = empty_analyzer.analyze(
        prediction="[some_a, some_b, some_c]",
        x_left="def f():\n    return ",
        x_right="\n",
    )
    assert not r.fires


# ---------- significant positions still DO fire ----------

def test_unresolved_call_target_fires(empty_analyzer):
    """Call target — the canonical hallucination signal."""
    r = empty_analyzer.analyze(
        prediction="fake_function_call()",
        x_left="def f():\n    return ",
        x_right="\n",
    )
    assert r.fires
    assert "fake_function_call" in r.significant_out_of_scope
    assert "fake_function_call" in r.unresolved_identifiers


def test_unresolved_attribute_receiver_fires(empty_analyzer):
    """Attribute receiver — accessing a member on a non-existent object."""
    r = empty_analyzer.analyze(
        prediction="fake_obj.some_attr",
        x_left="def f():\n    return ",
        x_right="\n",
    )
    assert r.fires
    assert "fake_obj" in r.significant_out_of_scope


def test_unresolved_subscript_value_fires(empty_analyzer):
    """Subscript value — indexing into a non-existent object."""
    r = empty_analyzer.analyze(
        prediction="fake_array[0]",
        x_left="def f():\n    return ",
        x_right="\n",
    )
    assert r.fires
    assert "fake_array" in r.significant_out_of_scope


def test_unresolved_call_target_with_bare_args_fires_on_call_only(empty_analyzer):
    """`fake_func(unresolved_x)` — both unresolved, but only fake_func is
    significant. The fire decision uses the significant subset only.
    """
    r = empty_analyzer.analyze(
        prediction="fake_func(unresolved_x)",
        x_left="def f():\n    return ",
        x_right="\n",
    )
    assert r.fires
    assert "fake_func" in r.significant_out_of_scope
    assert "unresolved_x" in r.unresolved_identifiers
    assert "unresolved_x" not in r.significant_out_of_scope


# ---------- cross-file path ----------

def test_crossfile_call_target_fires(repo_with):
    """Cross-file resolved as a call target — fires under the unified signal.

    The cascade no longer distinguishes cross-file vs unresolved at the
    trigger level; both are out-of-scope at a significant position and
    both fire. The loose ``cross_file_identifiers`` list remains for RSP.
    """
    analyzer = repo_with({"lib.py": "def repo_func():\n    return 1\n"})
    r = analyzer.analyze(
        prediction="repo_func()",
        x_left="def f():\n    return ",
        x_right="\n",
    )
    assert r.fires
    assert "repo_func" in r.significant_out_of_scope
    # Still classified in the loose list as cross-file for diagnostics.
    assert "repo_func" in r.cross_file_identifiers


def test_crossfile_bare_does_not_fire(repo_with):
    """A cross-file name used only as a bare value (not called or accessed)
    is not a trigger. Bare uses are too common to be reliable signals.
    """
    analyzer = repo_with({"lib.py": "REPO_CONST = 42\n"})
    r = analyzer.analyze(
        prediction="x + REPO_CONST",
        x_left="def f(x):\n    return ",
        x_right="\n",
    )
    assert not r.fires
    # Still appears in the loose list for diagnostics.
    assert "REPO_CONST" in r.cross_file_identifiers


# ---------- significant subset contract ----------

def test_significant_out_of_scope_is_subset_of_used(empty_analyzer):
    """Invariant: every significant_out_of_scope element is in one of the
    loose lists too (either unresolved or cross_file)."""
    r = empty_analyzer.analyze(
        prediction="fake_func(other_unknown) + bare_unknown.attr",
        x_left="def f():\n    return ",
        x_right="\n",
    )
    for n in r.significant_out_of_scope:
        assert n in r.unresolved_identifiers or n in r.cross_file_identifiers


def test_resolved_names_never_appear_in_either_list(empty_analyzer):
    """`helper(x)` where both are in scope — no entries in either bucket."""
    r = empty_analyzer.analyze(
        prediction="helper(x)",
        x_left="def helper(x):\n    return x\n\ndef caller(x):\n    return ",
        x_right="\n",
    )
    assert r.unresolved_identifiers == []
    assert r.cross_file_identifiers == []
    assert r.significant_out_of_scope == []
    assert not r.fires


# ---------- subtle: chained calls / nested attributes ----------

def test_nested_attribute_resolved_receiver_only_does_not_fire(empty_analyzer):
    """`np.array.shape` — np is imported (in-file). The chain resolves through
    the receiver. Even if mid-chain types aren't in our symbol table, only
    the outermost receiver matters."""
    r = empty_analyzer.analyze(
        prediction="np.array.shape",
        x_left="import numpy as np\n\ndef f():\n    return ",
        x_right="\n",
    )
    assert not r.fires


def test_nested_attribute_unresolved_receiver_fires(empty_analyzer):
    """`fake_thing.something.method()` — fake_thing is the deepest receiver,
    out of scope, so it fires."""
    r = empty_analyzer.analyze(
        prediction="fake_thing.something.method()",
        x_left="def f():\n    return ",
        x_right="\n",
    )
    assert r.fires
    assert "fake_thing" in r.significant_out_of_scope
