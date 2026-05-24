"""Tests for the structurally-significant positions beyond the original three
(call target, attribute receiver, subscript value).

Added in the signal-collapse refactor: class bases, bare decorators, exception
types (including tuple and aliased forms), and raise targets. Each is a
position where a hallucinated identifier reference would constitute a real
semantic error, but tree-sitter classifies them outside the original three.
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


# ---------- class bases ----------

def test_unresolved_class_base_fires(empty_analyzer):
    """`class Foo(NonExistentBase):` — NonExistentBase as a class base
    is now a significant position. Fires."""
    r = empty_analyzer.analyze(
        prediction="class Foo(NonExistentBase):\n    pass",
        x_left="",
        x_right="\n",
    )
    assert r.fires
    assert "NonExistentBase" in r.significant_out_of_scope


def test_resolved_class_base_does_not_fire(empty_analyzer):
    """`class Foo(BaseClass):` where BaseClass is imported. No fire."""
    r = empty_analyzer.analyze(
        prediction="class Foo(BaseClass):\n    pass",
        x_left="from base import BaseClass\n",
        x_right="\n",
    )
    assert not r.fires


def test_multiple_class_bases_all_checked(empty_analyzer):
    """`class Foo(Real, Fake):` — Real resolves via import, Fake doesn't."""
    r = empty_analyzer.analyze(
        prediction="class Foo(Real, Fake):\n    pass",
        x_left="from base import Real\n",
        x_right="\n",
    )
    assert r.fires
    assert "Fake" in r.significant_out_of_scope
    assert "Real" not in r.significant_out_of_scope


# ---------- decorators ----------

def test_unresolved_bare_decorator_fires(empty_analyzer):
    """`@nonexistent_decorator\ndef f():` — bare decorator name not in scope."""
    r = empty_analyzer.analyze(
        prediction="@nonexistent_decorator\ndef new_helper():\n    pass",
        x_left="",
        x_right="\n",
    )
    assert r.fires
    assert "nonexistent_decorator" in r.significant_out_of_scope


def test_resolved_bare_decorator_does_not_fire(empty_analyzer):
    """`@cached\ndef f():` — cached imported from functools."""
    r = empty_analyzer.analyze(
        prediction="@cached\ndef new_helper():\n    pass",
        x_left="from functools import cache as cached\n",
        x_right="\n",
    )
    assert not r.fires


def test_unresolved_decorator_with_call_still_fires(empty_analyzer):
    """`@fake_dec()\ndef f():` — already covered by call-target rule."""
    r = empty_analyzer.analyze(
        prediction="@fake_dec()\ndef new_helper():\n    pass",
        x_left="",
        x_right="\n",
    )
    assert r.fires
    assert "fake_dec" in r.significant_out_of_scope


# ---------- exception types ----------

def test_unresolved_except_type_fires(empty_analyzer):
    """`except CustomError:` where CustomError isn't defined or imported."""
    r = empty_analyzer.analyze(
        prediction="try:\n    pass\nexcept CustomError:\n    pass",
        x_left="",
        x_right="\n",
    )
    assert r.fires
    assert "CustomError" in r.significant_out_of_scope


def test_resolved_except_type_does_not_fire(empty_analyzer):
    """`except ValueError:` — ValueError is a builtin."""
    r = empty_analyzer.analyze(
        prediction="try:\n    pass\nexcept ValueError:\n    pass",
        x_left="",
        x_right="\n",
    )
    assert not r.fires


def test_unresolved_except_tuple_fires(empty_analyzer):
    """`except (E1, E2):` — tuple form. E1 resolves; E2 doesn't."""
    r = empty_analyzer.analyze(
        prediction="try:\n    pass\nexcept (ValueError, CustomError):\n    pass",
        x_left="",
        x_right="\n",
    )
    assert r.fires
    assert "CustomError" in r.significant_out_of_scope
    assert "ValueError" not in r.significant_out_of_scope


def test_unresolved_except_with_alias_fires(empty_analyzer):
    """`except CustomError as e:` — alias `e` is bound; CustomError is the type."""
    r = empty_analyzer.analyze(
        prediction="try:\n    pass\nexcept CustomError as e:\n    print(e)",
        x_left="",
        x_right="\n",
    )
    assert r.fires
    assert "CustomError" in r.significant_out_of_scope


def test_resolved_except_with_alias_does_not_fire(empty_analyzer):
    """`except ValueError as e:` — type is builtin; alias e bound locally."""
    r = empty_analyzer.analyze(
        prediction="try:\n    pass\nexcept ValueError as e:\n    print(e)",
        x_left="",
        x_right="\n",
    )
    assert not r.fires


# ---------- raise targets ----------

def test_unresolved_raise_target_fires(empty_analyzer):
    """`raise CustomError` (no call) — bare raise of unresolved name."""
    r = empty_analyzer.analyze(
        prediction="raise CustomError",
        x_left="def f():\n    ",
        x_right="\n",
    )
    assert r.fires
    assert "CustomError" in r.significant_out_of_scope


def test_resolved_raise_target_does_not_fire(empty_analyzer):
    """`raise ValueError` — builtin exception."""
    r = empty_analyzer.analyze(
        prediction="raise ValueError",
        x_left="def f():\n    ",
        x_right="\n",
    )
    assert not r.fires


def test_unresolved_raise_with_call_already_caught(empty_analyzer):
    """`raise CustomError("msg")` — caught via call-target rule (not new)."""
    r = empty_analyzer.analyze(
        prediction='raise CustomError("msg")',
        x_left="def f():\n    ",
        x_right="\n",
    )
    assert r.fires
    assert "CustomError" in r.significant_out_of_scope


def test_unresolved_raise_from_fires(empty_analyzer):
    """`raise X from Y` — both X and Y are children of raise_statement."""
    r = empty_analyzer.analyze(
        prediction="raise CustomError from OriginalError",
        x_left="def f():\n    ",
        x_right="\n",
    )
    assert r.fires
    # At least one — could be both.
    assert (
        "CustomError" in r.significant_out_of_scope
        or "OriginalError" in r.significant_out_of_scope
    )
