"""Accuracy, hallucination, efficiency, and statistical-test metrics.

Per IMPLEMENTATION_GUIDE §13. ``repository_symbol_precision`` and
``hallucination_flag`` depend on the static-analysis ``PredictionAnalyzer``;
all other metrics are self-contained.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
from Levenshtein import distance as lev_distance
from scipy.stats import binomtest

from .static_analysis.analyzer import PredictionAnalyzer


# ---------- prediction post-processing ----------

def _indentation(line: str) -> int:
    """Leading-whitespace width of ``line`` (tabs expanded to 8)."""
    expanded = line.expandtabs(8)
    return len(expanded) - len(expanded.lstrip())


def truncate_to_function_body(reference: str, prediction: str) -> str:
    """Truncate a multi-line prediction to the end of the generated function body.

    A code LM completing a function body keeps generating past the body — into
    the next top-level ``def``/``class``, or into FIM padding. We keep the
    prediction up to (but not including) the first line that *dedents out of the
    body*: any non-blank line whose indentation is shallower than the body's
    own indentation marks the end of the function and is dropped along with
    everything after it.

    The body indentation is taken from the first non-blank line of the
    ground-truth completion (the holes are function bodies, so the gold's first
    line sits at the body's indent level). This is the Python-faithful analogue
    of CARD §3.3's "extract the function body by matching brackets" — Python
    delimits blocks by indentation, not braces, so we use the dedent boundary.

    Unlike a line-count budget, this scores the model's *actual complete
    completion* (the whole body it wrote, ``return`` included) rather than an
    arbitrary first-N-lines slice, so a guard clause or extra statement never
    pushes the real body content out of the scored window.
    """
    if not prediction.strip():
        return ""
    ref_nb = [ln for ln in reference.splitlines() if ln.strip()]
    if not ref_nb:
        return prediction
    # Body indent = the shallowest INDENTED gold line. We can't just take the
    # first line's indent: some benchmarks (Repoformer CrossCodeLongEval) move
    # the gold's first-line leading whitespace into the prompt, so the gold's
    # first line sits at column 0. A function body never legitimately contains a
    # column-0 line, so the shallowest line with indent > 0 is the true body
    # indent; falling back to the first line only when nothing is indented (a
    # degenerate single-token body). This keeps RepoEval unchanged (its gold
    # first line already carries the base indent == min positive indent).
    indents = [_indentation(ln) for ln in ref_nb]
    positive = [i for i in indents if i > 0]
    body_indent = min(positive) if positive else indents[0]

    out: list[str] = []
    seen_body = False
    for line in prediction.splitlines():
        if line.strip():
            # A non-blank line shallower than the body indent ends the function.
            if seen_body and _indentation(line) < body_indent:
                break
            seen_body = True
            out.append(line)
        else:
            # Keep blank lines only while inside the body (they may be trailing
            # padding once the body ends, but we strip trailing blanks below).
            out.append(line)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


# ---------- accuracy ----------

def exact_match(reference: str, prediction: str) -> bool:
    return reference.rstrip() == prediction.rstrip()


def edit_similarity(reference: str, prediction: str) -> float:
    """ES per CARD §2.1, value in [0, 1] (higher = better)."""
    if not reference and not prediction:
        return 1.0
    denom = max(len(reference), len(prediction))
    if denom == 0:
        return 1.0
    return 1.0 - lev_distance(reference, prediction) / denom


_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def _identifiers(text: str) -> list[str]:
    return _IDENT_RE.findall(text)


def identifier_f1(reference: str, prediction: str) -> float:
    """CrossCodeEval-style Identifier-F1. Set-based P/R over identifier tokens."""
    ref_ids = set(_identifiers(reference))
    pred_ids = set(_identifiers(prediction))
    if not ref_ids and not pred_ids:
        return 1.0
    if not ref_ids or not pred_ids:
        return 0.0
    tp = len(ref_ids & pred_ids)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_ids)
    recall = tp / len(ref_ids)
    return 2 * precision * recall / (precision + recall)


# ---------- reference-based hallucination (A4 / A4∧B2) ----------

def hallucinated_identifier_flag(reference: str, prediction: str) -> bool:
    """A4: True iff the prediction emits ≥1 identifier absent from the reference.

    Reference-based and resolver-free (independent of any static checker). It's
    an upper bound on hallucination — a valid alternative identifier also counts.
    Repurposes CrossCodeEval's Identifier-Match precision component.
    """
    return bool(set(_identifiers(prediction)) - set(_identifiers(reference)))


def invented_identifier_flag(reference: str, prediction: str, undefined_names) -> bool:
    """A4 ∧ B2: True iff the prediction emits an identifier that is BOTH absent
    from the reference (A4) AND flagged ``undefined`` by a static checker
    (B2 — e.g. pyflakes F821). The cleanest "the model invented a name" signal:
    the reference never uses it *and* it resolves nowhere.

    ``undefined_names`` is the checker's undefined-name set restricted to the
    prediction span — passed in so this module stays free of any resolver
    dependency (feed pyflakes / Pyright output here).
    """
    return bool(set(undefined_names) - set(_identifiers(reference)))


# ---------- hallucination ----------

def repository_symbol_precision(
    prediction: str,
    x_left: str,
    x_right: str,
    analyzer: PredictionAnalyzer,
) -> float:
    """Fraction of identifiers in the prediction that are visible at the hole.

    Under the unified Tier 1 design this is purely a scope check — it does
    not consult the repository symbol table. The metric is a descriptive
    continuous companion to the binary ``hallucination_flag``.
    """
    result = analyzer.analyze(prediction, x_left, x_right)
    n_total = result.n_used_identifiers
    if n_total == 0:
        return 1.0
    n_out_of_scope = len(result.out_of_scope_identifiers)
    return (n_total - n_out_of_scope) / n_total


def hallucination_flag(
    prediction: str,
    x_left: str,
    x_right: str,
    analyzer: PredictionAnalyzer,
) -> bool:
    """True iff any static-analysis tier found something: an out-of-scope
    identifier in a significant position (Tier 1), a signature mismatch
    (Tier 2), or a wrong-origin import (Tier 3). Independent of which tiers
    are enabled at trigger time — reflects what the analyzer actually found,
    not whether the cascade would have acted on it.
    """
    result = analyzer.analyze(prediction, x_left, x_right)
    return (
        bool(result.significant_out_of_scope)
        or bool(result.signature_issues)
        or bool(result.import_issues)
    )


# ---------- efficiency (aggregate) ----------

def percent_retrieval(records: Iterable[dict]) -> float:
    records = list(records)
    if not records:
        return 0.0
    return 100.0 * sum(1 for r in records if r.get("retrieved")) / len(records)


def mean_latency_ms(records: Iterable[dict]) -> float:
    records = list(records)
    if not records:
        return 0.0
    return sum(r.get("latency_ms", 0.0) for r in records) / len(records)


# ---------- statistical tests ----------

def mcnemar_test(
    records_a: list[dict],
    records_b: list[dict],
    key: str = "hallucinated",
) -> dict:
    """Paired McNemar test for a binary outcome.

    Returns ``{"p_value", "b", "c"}`` where:
    - b: count of (A=0, B=1) — instances where B is worse
    - c: count of (A=1, B=0) — instances where B is better

    Uses the exact binomial test (no continuity correction) which is
    appropriate for the small b+c counts we expect.
    """
    if len(records_a) != len(records_b):
        raise ValueError(
            f"Paired test needs equal lengths: {len(records_a)} vs {len(records_b)}"
        )
    b = sum(1 for ra, rb in zip(records_a, records_b) if not ra[key] and rb[key])
    c = sum(1 for ra, rb in zip(records_a, records_b) if ra[key] and not rb[key])
    if b + c == 0:
        return {"p_value": 1.0, "b": 0, "c": 0}
    result = binomtest(min(b, c), b + c, p=0.5)
    return {"p_value": float(result.pvalue), "b": b, "c": c}


def paired_bootstrap(
    records_a: list[dict],
    records_b: list[dict],
    key: str,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap 95% CI for ``mean(B[key] - A[key])``."""
    if len(records_a) != len(records_b):
        raise ValueError(
            f"Paired bootstrap needs equal lengths: {len(records_a)} vs {len(records_b)}"
        )
    a_vals = np.asarray([r[key] for r in records_a], dtype=np.float64)
    b_vals = np.asarray([r[key] for r in records_b], dtype=np.float64)
    diffs = b_vals - a_vals
    n = len(diffs)
    if n == 0:
        return {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = diffs[idx].mean(axis=1)
    return {
        "mean_diff": float(diffs.mean()),
        "ci_lower": float(np.percentile(boot_means, 2.5)),
        "ci_upper": float(np.percentile(boot_means, 97.5)),
    }
