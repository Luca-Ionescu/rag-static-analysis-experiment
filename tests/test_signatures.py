"""Tests for static_analysis.signatures — Signature extraction from tree-sitter."""
from __future__ import annotations

from adaptive_retrieval.static_analysis.parser import parse
from adaptive_retrieval.static_analysis.signatures import (
    Signature,
    extract_signature,
)


def _first_func(src: str):
    tree = parse(src)

    def walk(n):
        if n.type == "function_definition":
            return n
        for c in n.children:
            r = walk(c)
            if r is not None:
                return r
        return None

    return walk(tree.root_node)


def _sig(src: str, is_method: bool = False) -> Signature:
    return extract_signature(_first_func(src), src.encode(), is_method=is_method)


# ---------- positional ----------

def test_simple_positional():
    sig = _sig("def f(a, b, c): pass")
    assert sig.positional == ("a", "b", "c")
    assert sig.min_positional == 3
    assert sig.max_positional == 3


def test_defaults_widen_range():
    sig = _sig("def f(a, b=1, c=2): pass")
    assert sig.n_defaults == 2
    assert sig.min_positional == 1
    assert sig.max_positional == 3


def test_no_args():
    sig = _sig("def f(): pass")
    assert sig.positional == ()
    assert sig.min_positional == 0
    assert sig.max_positional == 0


# ---------- splats ----------

def test_star_args_unbounds_max():
    sig = _sig("def f(a, *args): pass")
    assert sig.has_star_args
    assert sig.max_positional is None


def test_star_kwargs():
    sig = _sig("def f(**kw): pass")
    assert sig.has_star_kwargs
    assert sig.all_keyword_names == frozenset()


# ---------- keyword-only ----------

def test_keyword_only_after_star():
    sig = _sig("def f(a, *, b, c=1): pass")
    assert sig.keyword_only == ("b", "c")
    assert sig.all_keyword_names == frozenset({"a", "b", "c"})


def test_keyword_only_after_star_args():
    sig = _sig("def f(a, *args, b): pass")
    assert sig.has_star_args
    assert sig.keyword_only == ("b",)


# ---------- methods ----------

def test_method_has_self():
    src = "class C:\n    def m(self, x): pass\n"
    sig = _sig(src, is_method=True)
    assert sig.has_self
    assert sig.positional == ("self", "x")
    assert sig.min_positional == 1  # excludes self


def test_method_with_defaults():
    src = "class C:\n    def m(self, x, y=1): pass\n"
    sig = _sig(src, is_method=True)
    assert sig.min_positional == 1
    assert sig.max_positional == 2


def test_classmethod_keeps_cls_as_self():
    src = "class C:\n    @classmethod\n    def m(cls, x): pass\n"
    sig = _sig(src, is_method=True)
    assert sig.has_self  # cls counts as the implicit receiver


def test_staticmethod_strips_self_semantics():
    src = "class C:\n    @staticmethod\n    def m(x, y): pass\n"
    sig = _sig(src, is_method=True)
    assert not sig.has_self
    assert sig.min_positional == 2


# ---------- decorators ----------

def test_unsafe_decorator_flagged():
    src = "@click.command()\ndef f(a, b): pass\n"
    sig = _sig(src)
    assert sig.decorated_unsafe


def test_safe_decorator_not_flagged():
    src = "@functools.lru_cache\ndef f(a): pass\n"
    sig = _sig(src)
    assert not sig.decorated_unsafe


def test_overload_decorator_is_safe():
    src = "@overload\ndef f(a): pass\n"
    sig = _sig(src)
    assert not sig.decorated_unsafe
