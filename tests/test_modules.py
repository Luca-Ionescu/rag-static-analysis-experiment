"""Tests for static_analysis.modules — path/module conversion and reexport parsing."""
from __future__ import annotations

from adaptive_retrieval.static_analysis.modules import (
    find_repo_root,
    parse_all_export,
    parse_reexports,
    path_to_module,
)


# ---------- path_to_module ----------

def test_path_to_module_simple():
    assert path_to_module("foo.py") == "foo"


def test_path_to_module_nested():
    assert path_to_module("pkg/sub/foo.py") == "pkg.sub.foo"


def test_path_to_module_init():
    assert path_to_module("pkg/__init__.py") == "pkg"


def test_path_to_module_top_level_init():
    assert path_to_module("__init__.py") == ""


def test_path_to_module_strips_repo_root():
    assert path_to_module("src/pkg/foo.py", repo_root="src") == "pkg.foo"


def test_path_to_module_with_trailing_slash_root():
    assert path_to_module("src/pkg/foo.py", repo_root="src/") == "pkg.foo"


def test_path_to_module_root_not_prefix_is_ignored():
    # If the supplied root isn't actually a prefix, we keep the full path.
    assert path_to_module("pkg/foo.py", repo_root="src") == "pkg.foo"


# ---------- find_repo_root ----------

def test_find_repo_root_common_prefix():
    assert find_repo_root(["src/a.py", "src/sub/b.py"]) == "src"


def test_find_repo_root_no_common_prefix():
    assert find_repo_root(["a.py", "b.py"]) == ""


def test_find_repo_root_mixed_top_and_sub():
    assert find_repo_root(["src/a.py", "top.py"]) == ""


def test_find_repo_root_empty():
    assert find_repo_root([]) == ""


# ---------- parse_reexports ----------

def test_parse_reexports_basic():
    src = "from .core import Foo, Bar\n"
    assert parse_reexports(src) == {"Foo": ".core", "Bar": ".core"}


def test_parse_reexports_alias():
    src = "from .util import baz as zap\n"
    assert parse_reexports(src) == {"zap": ".util"}


def test_parse_reexports_absolute_module():
    src = "from pkg.core import Thing\n"
    assert parse_reexports(src) == {"Thing": "pkg.core"}


def test_parse_reexports_empty():
    assert parse_reexports("") == {}


def test_parse_reexports_ignores_star_imports():
    # Star imports can't be statically resolved -> intentionally skipped.
    src = "from .core import *\n"
    assert parse_reexports(src) == {}


# ---------- parse_all_export ----------

def test_parse_all_export_simple():
    assert parse_all_export('__all__ = ["Foo", "Bar"]') == {"Foo", "Bar"}


def test_parse_all_export_tuple():
    assert parse_all_export('__all__ = ("Foo",)') == {"Foo"}


def test_parse_all_export_missing():
    assert parse_all_export("x = 1\n") is None


def test_parse_all_export_dynamic_returns_none():
    # Anything other than a plain list/tuple literal -> abstain (None).
    assert parse_all_export('__all__ = list(NAMES)') is None
