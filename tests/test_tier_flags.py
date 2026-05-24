"""Tests for the three-tier flag-toggle behaviour on ``PredictionAnalyzer``.

Each of the three static-analysis tiers can be individually enabled or
disabled at construction time. The trigger fires iff *any* enabled tier's
output is non-empty. Verifying:

  * each tier in isolation fires on its targeted failure mode,
  * disabling a tier suppresses firing on its outputs,
  * the three-tier OR composes correctly,
  * all-tiers-off makes the analyzer permanently silent.
"""
from __future__ import annotations

from adaptive_retrieval.static_analysis.analyzer import PredictionAnalyzer
from adaptive_retrieval.static_analysis.scope import InFileScopeAnalyzer
from adaptive_retrieval.static_analysis.symbol_table import RepositorySymbolTable


def _analyzer(repo_files: dict[str, str] | None = None, **flags) -> PredictionAnalyzer:
    return PredictionAnalyzer(
        InFileScopeAnalyzer(),
        RepositorySymbolTable.from_files(repo_files or {}),
        **flags,
    )


# ---------- Tier 1 alone ----------

def test_tier1_alone_fires_on_unresolved_call_target():
    a = _analyzer(
        fire_on_out_of_scope=True,
        fire_on_signature=False,
        fire_on_import=False,
    )
    r = a.analyze("fake_func()", x_left="def f():\n    return ", x_right="\n")
    assert r.fires
    assert "fake_func" in r.significant_out_of_scope


def test_tier1_off_suppresses_out_of_scope_fire():
    a = _analyzer(
        fire_on_out_of_scope=False,
        fire_on_signature=False,
        fire_on_import=False,
    )
    r = a.analyze("fake_func()", x_left="def f():\n    return ", x_right="\n")
    # The signal is still computed (significant_out_of_scope populated) but
    # the trigger doesn't fire.
    assert not r.fires
    assert "fake_func" in r.significant_out_of_scope


# ---------- Tier 2 alone ----------

def test_tier2_alone_fires_on_signature_mismatch():
    # repo_func takes 1 positional arg; we pass 3.
    a = _analyzer(
        {"lib.py": "def repo_func(x):\n    return x\n"},
        fire_on_out_of_scope=False,
        fire_on_signature=True,
        fire_on_import=False,
    )
    r = a.analyze(
        "repo_func(1, 2, 3)",
        x_left="from lib import repo_func\n\ndef f():\n    return ",
        x_right="\n",
    )
    assert r.fires
    assert r.signature_issues  # at least one


def test_tier2_off_suppresses_signature_fire():
    a = _analyzer(
        {"lib.py": "def repo_func(x):\n    return x\n"},
        fire_on_out_of_scope=False,
        fire_on_signature=False,
        fire_on_import=False,
    )
    r = a.analyze(
        "repo_func(1, 2, 3)",
        x_left="from lib import repo_func\n\ndef f():\n    return ",
        x_right="\n",
    )
    # Tier 2 still detected the issue (lists are populated) but the trigger
    # doesn't fire because the flag is off.
    assert not r.fires
    assert r.signature_issues


# ---------- Tier 3 alone ----------

def test_tier3_alone_fires_on_wrong_origin_import():
    # save_record lives in db.persistence; the prediction imports it from db.client.
    a = _analyzer(
        {
            "db/__init__.py": "",
            "db/client.py": "def connect():\n    pass\n",
            "db/persistence.py": "def save_record():\n    pass\n",
        },
        fire_on_out_of_scope=False,
        fire_on_signature=False,
        fire_on_import=True,
    )
    r = a.analyze(
        "from db.client import save_record",
        x_left="",
        x_right="\n",
    )
    assert r.fires
    assert r.import_issues
    issue = r.import_issues[0]
    assert issue.name == "save_record"
    assert "db.persistence" in issue.expected_modules


def test_tier3_off_suppresses_import_fire():
    a = _analyzer(
        {
            "db/__init__.py": "",
            "db/client.py": "def connect():\n    pass\n",
            "db/persistence.py": "def save_record():\n    pass\n",
        },
        fire_on_out_of_scope=False,
        fire_on_signature=False,
        fire_on_import=False,
    )
    r = a.analyze("from db.client import save_record", x_left="", x_right="\n")
    assert not r.fires
    assert r.import_issues  # signal still computed


# ---------- Composition: OR across enabled tiers ----------

def test_any_enabled_tier_can_fire_independently():
    # Prediction has both an out-of-scope call AND a wrong-origin import.
    a = _analyzer(
        {
            "db/__init__.py": "",
            "db/client.py": "def connect():\n    pass\n",
            "db/persistence.py": "def save_record():\n    pass\n",
        },
        fire_on_out_of_scope=True,
        fire_on_signature=False,
        fire_on_import=True,
    )
    r = a.analyze(
        "from db.client import save_record\nfake_other()",
        x_left="",
        x_right="\n",
    )
    assert r.fires
    assert r.significant_out_of_scope    # Tier 1 found fake_other
    assert r.import_issues               # Tier 3 found wrong-origin import


def test_only_enabled_tier_drives_fire_when_others_silent():
    # Only Tier 3 fires (no out-of-scope identifiers, no signature issues).
    a = _analyzer(
        {
            "db/__init__.py": "",
            "db/client.py": "def connect():\n    pass\n",
            "db/persistence.py": "def save_record():\n    pass\n",
        },
        fire_on_out_of_scope=True,
        fire_on_signature=True,
        fire_on_import=True,
    )
    r = a.analyze("from db.client import save_record", x_left="", x_right="\n")
    assert r.fires
    assert not r.significant_out_of_scope
    assert not r.signature_issues
    assert r.import_issues


def test_all_tiers_off_never_fires():
    """With every flag off, the analyzer is permanently silent on any input."""
    a = _analyzer(
        {"lib.py": "def repo_func(x):\n    return x\n"},
        fire_on_out_of_scope=False,
        fire_on_signature=False,
        fire_on_import=False,
    )
    # All three failure modes present in one prediction.
    r = a.analyze(
        "fake_call()\nrepo_func(1, 2, 3)",
        x_left="from lib import repo_func\n",
        x_right="\n",
    )
    assert not r.fires
    # Signals are still computed for diagnostics.
    assert r.significant_out_of_scope or r.signature_issues


def test_defaults_have_all_tiers_enabled():
    """Default constructor enables all three tiers."""
    a = _analyzer()
    assert a.fire_on_out_of_scope is True
    assert a.fire_on_signature is True
    assert a.fire_on_import is True


# ---------- Tier 1 does not consult the symbol table ----------

def test_tier1_decision_independent_of_symbol_table():
    """Tier 1's significant_out_of_scope is the same whether the symbol
    table contains the name or not. (Tier 2/3 use the symbol table; Tier 1
    deliberately does not.)"""
    pred = "fake_helper()"
    x_left = "def f():\n    return "
    x_right = "\n"

    a_empty = _analyzer(
        fire_on_out_of_scope=True, fire_on_signature=False, fire_on_import=False,
    )
    a_with = _analyzer(
        {"lib.py": "def fake_helper():\n    pass\n"},
        fire_on_out_of_scope=True, fire_on_signature=False, fire_on_import=False,
    )

    r_empty = a_empty.analyze(pred, x_left, x_right)
    r_with = a_with.analyze(pred, x_left, x_right)

    # Symbol-table membership is irrelevant — Tier 1 fires the same way.
    assert r_empty.fires and r_with.fires
    assert (
        r_empty.significant_out_of_scope == r_with.significant_out_of_scope
        == ["fake_helper"]
    )
