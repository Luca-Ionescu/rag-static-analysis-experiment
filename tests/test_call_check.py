"""Tests for static_analysis.call_check — Tier 2 signature mismatch detection."""
from __future__ import annotations

import pytest

from adaptive_retrieval.static_analysis.call_check import check_calls
from adaptive_retrieval.static_analysis.symbol_table import RepositorySymbolTable


@pytest.fixture
def repo():
    """Repo with:
        pkg/core.py:
          def foo(x, y=1): ...
          def bar(*args, **kw): ...
          class Repo:
            def find(self, path): ...
            @staticmethod
            def helper(a, b): ...
        pkg/util.py: def util(a, b, c)
    """
    files = {
        "pkg/__init__.py": "",
        "pkg/core.py": (
            "def foo(x, y=1):\n    return x + y\n\n"
            "def bar(*args, **kwargs):\n    pass\n\n"
            "class Repo:\n"
            "    def find(self, path):\n        return path\n"
            "    @staticmethod\n"
            "    def helper(a, b):\n        return a + b\n"
        ),
        "pkg/util.py": "def util(a, b, c):\n    pass\n",
    }
    return RepositorySymbolTable.from_files(files)


# ---------- positive cases (no issue) ----------

def test_correct_arity(repo):
    assert check_calls("foo(1)", "from pkg.core import foo\n", repo) == []


def test_default_only_argument(repo):
    # foo(x, y=1) with one positional is valid.
    assert check_calls("foo(1)", "from pkg.core import foo\n", repo) == []


def test_known_keyword_argument(repo):
    assert check_calls("foo(1, y=2)", "from pkg.core import foo\n", repo) == []


def test_callee_with_splats_accepts_anything(repo):
    assert (
        check_calls("bar(1, 2, 3, whatever=4)", "from pkg.core import bar\n", repo)
        == []
    )


def test_class_qualified_method_correct(repo):
    assert (
        check_calls("Repo.find(r, 'p')", "from pkg.core import Repo\n", repo) == []
    )


# ---------- wrong arity ----------

def test_too_many_positional(repo):
    issues = check_calls("foo(1, 2, 3)", "from pkg.core import foo\n", repo)
    assert len(issues) == 1
    assert issues[0].kind == "wrong_arity"
    assert issues[0].callee == "foo"


def test_too_few_positional(repo):
    issues = check_calls("foo()", "from pkg.core import foo\n", repo)
    assert len(issues) == 1
    assert issues[0].kind == "wrong_arity"


def test_module_qualified_wrong_arity(repo):
    issues = check_calls(
        "core.foo(1, 2, 3)", "import pkg.core as core\n", repo,
    )
    assert len(issues) == 1
    assert issues[0].kind == "wrong_arity"


def test_class_qualified_method_too_few(repo):
    issues = check_calls("Repo.find()", "from pkg.core import Repo\n", repo)
    assert len(issues) == 1
    assert issues[0].kind == "wrong_arity"


def test_staticmethod_wrong_arity(repo):
    # staticmethod has no implicit self — must supply both a and b.
    issues = check_calls("Repo.helper(1)", "from pkg.core import Repo\n", repo)
    assert len(issues) == 1
    assert issues[0].kind == "wrong_arity"


# ---------- unknown kwarg ----------

def test_unknown_kwarg_flagged(repo):
    issues = check_calls("foo(1, typo=2)", "from pkg.core import foo\n", repo)
    assert len(issues) == 1
    assert issues[0].kind == "unknown_kwarg"
    assert "typo" in issues[0].actual


def test_splat_skips_arity_but_keeps_kwarg(repo):
    issues = check_calls("foo(*xs, typo=2)", "from pkg.core import foo\n", repo)
    # Arity check skipped due to *xs; kwarg check still finds the typo.
    assert len(issues) == 1
    assert issues[0].kind == "unknown_kwarg"


# ---------- abstention cases (no issue) ----------

def test_self_method_call_abstains(repo):
    # Receiver type unknown -> abstain. self.find(...) would need Tier 1.
    assert check_calls("self.find('p', 'q')\n", "", repo) == []


def test_instance_method_call_abstains(repo):
    # `obj.find('p', 'q', 'r')` — obj's type isn't tracked here. Abstain.
    assert check_calls("obj.find('p', 'q', 'r')\n", "", repo) == []


def test_unknown_callee_abstains(repo):
    # `nope` isn't in the repo. The bare-name unresolved check handles it.
    assert check_calls("nope()", "", repo) == []


def test_external_module_call_abstains(repo):
    # numpy isn't part of the repo -> no signatures available.
    assert (
        check_calls("np.array([1, 2, 3])", "import numpy as np\n", repo) == []
    )
