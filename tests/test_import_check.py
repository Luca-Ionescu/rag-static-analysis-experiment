"""Tests for static_analysis.import_check — Tier 3 wrong-source detection."""
from __future__ import annotations

import pytest

from adaptive_retrieval.static_analysis.import_check import (
    check_attribute_usage,
    check_imports,
)
from adaptive_retrieval.static_analysis.symbol_table import RepositorySymbolTable


@pytest.fixture
def repo():
    """Tiny repo with:
        pkg/core.py:  def Foo()
        pkg/util.py:  class Bar
        pkg/other.py: empty
        pkg/__init__.py: re-exports Bar (not Foo); declares __all__ = ["Bar"]
    """
    files = {
        "pkg/__init__.py": 'from .util import Bar\n__all__ = ["Bar"]\n',
        "pkg/core.py": "def Foo():\n    return 1\n",
        "pkg/util.py": "class Bar:\n    pass\n",
        "pkg/other.py": "",
    }
    return RepositorySymbolTable.from_files(files)


# ---------- wrong-origin from-imports ----------

def test_correct_import_no_issue(repo):
    assert check_imports("from pkg.core import Foo\n", "", repo) == []


def test_wrong_origin_flagged(repo):
    issues = check_imports("from pkg.other import Foo\n", "", repo)
    assert len(issues) == 1
    issue = issues[0]
    assert issue.kind == "wrong_origin"
    assert issue.name == "Foo"
    assert issue.expected_modules == ("pkg.core",)


def test_reexport_via_init_is_legitimate(repo):
    # pkg/__init__.py has `from .util import Bar` and `__all__ = ["Bar"]`.
    assert check_imports("from pkg import Bar\n", "", repo) == []


def test_not_reexported_via_init_flagged(repo):
    # Foo is NOT in pkg/__init__.py's reexports or __all__.
    issues = check_imports("from pkg import Foo\n", "", repo)
    assert len(issues) == 1
    assert issues[0].kind == "wrong_origin"


def test_external_module_abstains(repo):
    # numpy is not in the repo -> we can't validate -> abstain.
    assert check_imports("from numpy import array\n", "", repo) == []


def test_unknown_name_not_double_flagged(repo):
    # `bogus` doesn't exist anywhere in the repo. The bare-name unresolved
    # check handles it; import_check abstains.
    assert check_imports("from pkg.core import bogus\n", "", repo) == []


def test_aliased_import_validated(repo):
    issues = check_imports("from pkg.other import Foo as MyFoo\n", "", repo)
    assert len(issues) == 1
    assert issues[0].kind == "wrong_origin"


# ---------- relative imports ----------

def test_relative_import_resolved(repo):
    # `from .other import Foo` inside `pkg/main.py` -> resolves to pkg.other.
    issues = check_imports("from .other import Foo\n", "pkg/main.py", repo)
    assert len(issues) == 1
    assert issues[0].kind == "wrong_origin"
    assert issues[0].expected_modules == ("pkg.core",)


def test_relative_import_without_file_abstains(repo):
    # Can't resolve relative imports without knowing the importing file.
    assert check_imports("from .other import Foo\n", "", repo) == []


def test_wildcard_import_abstains(repo):
    # ``from X import *`` is intentionally not validated.
    assert check_imports("from pkg.other import *\n", "", repo) == []


# ---------- attribute usage ----------

def test_correct_attribute_usage(repo):
    # `import pkg.core as M; M.Foo()` is fine.
    assert (
        check_attribute_usage(
            "M.Foo()\n", repo, in_file_imports=[("M", "pkg.core")],
        )
        == []
    )


def test_missing_attribute_flagged(repo):
    # Bar lives in pkg.util, not pkg.core.
    issues = check_attribute_usage(
        "M.Bar()\n", repo, in_file_imports=[("M", "pkg.core")],
    )
    assert len(issues) == 1
    assert issues[0].kind == "missing_in_module"
    assert issues[0].expected_modules == ("pkg.util",)


def test_attribute_on_external_module_abstains(repo):
    assert (
        check_attribute_usage(
            "np.array([])\n", repo, in_file_imports=[("np", "numpy")],
        )
        == []
    )


def test_attribute_unknown_name_abstains(repo):
    # `Bogus` isn't anywhere in the repo, so unresolved check handles it.
    assert (
        check_attribute_usage(
            "M.Bogus()\n", repo, in_file_imports=[("M", "pkg.core")],
        )
        == []
    )
