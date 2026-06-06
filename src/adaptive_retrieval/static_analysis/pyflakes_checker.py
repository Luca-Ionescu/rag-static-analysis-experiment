"""A pyflakes-backed static checker — a drop-in alternative to the in-house
AST :class:`PredictionAnalyzer` for the cascade's Stage-3 gate.

It runs pyflakes on the *reconstructed file* ``x_left + prediction + x_right``
and keeps only the ``undefined name`` (F821) reports that fall on the
prediction's own line(s). Differences from the in-house analyzer:

* **Resolvability, not "significant position".** It flags ANY undefined name,
  so it catches bare-value hallucinations (``total = a + tax_amount``) that the
  significance heuristic misses, and — via pyflakes' complete binding model —
  never false-flags bindings (``with f() as x``, match captures, walrus, ...).
* **Single-file.** It resolves names against the in-file imports/defs only, so
  the reconstructed file must contain them. CCE's ``x_left`` starts at the file
  top (imports included), so it does.
* **Needs a parse.** Unparseable reconstructions yield no signal; an optional
  ``fallback`` analyzer (e.g. the tree-sitter one) is used on parse failure.

Returns a :class:`StaticAnalysisResult` so it slots into ``cascade_pipeline``,
``hallucination_flag`` and the rescore unchanged: undefined names are exposed
through ``significant_out_of_scope`` (and ``out_of_scope_identifiers``); the
signature/import tiers are always empty (pyflakes does not do them).
"""
from __future__ import annotations

import ast

from .analyzer import StaticAnalysisResult

try:
    from pyflakes import api as _pf_api
    from pyflakes import messages as _pf_msg

    _HAVE_PYFLAKES = True
except ImportError:  # pragma: no cover
    _HAVE_PYFLAKES = False


class _Collector:
    """pyflakes reporter that captures UndefinedName warnings + parse status."""

    def __init__(self) -> None:
        self.undefined: list[tuple[int, str]] = []  # (lineno, name)
        self.syntax_error = False

    def unexpectedError(self, filename, msg) -> None:  # noqa: N802 (pyflakes API)
        self.syntax_error = True

    def syntaxError(self, filename, msg, lineno, offset, text) -> None:  # noqa: N802
        self.syntax_error = True

    def flake(self, message) -> None:
        if isinstance(message, _pf_msg.UndefinedName):
            self.undefined.append((message.lineno, message.message_args[0]))


class PyflakesChecker:
    """Drop-in alternative to ``PredictionAnalyzer`` using pyflakes F821.

    Args:
        fallback: optional analyzer with a compatible ``.analyze()`` used when
            the reconstructed file does not parse. ``None`` => return no-fire.
    """

    # Mirror PredictionAnalyzer's tier flags so cascade_pipeline can read them.
    fire_on_out_of_scope = True
    fire_on_signature = False
    fire_on_import = False

    def __init__(self, fallback=None) -> None:
        if not _HAVE_PYFLAKES:
            raise ImportError("pyflakes is not installed (pip install pyflakes)")
        self.fallback = fallback
        self.parse_failures = 0  # counter for diagnostics

    def analyze(
        self,
        prediction: str,
        x_left: str,
        x_right: str = "",
        importing_file: str = "",
    ) -> StaticAnalysisResult:
        if not prediction.strip():
            return StaticAnalysisResult(fires=False)

        full = x_left + prediction + x_right
        start = x_left.count("\n") + 1            # 1-indexed: prediction's first line
        end = start + prediction.count("\n")

        # Guard the parse ourselves (host-Python ast) for a clean parse signal.
        try:
            ast.parse(full)
        except Exception:
            self.parse_failures += 1
            if self.fallback is not None:
                return self.fallback.analyze(
                    prediction, x_left, x_right, importing_file=importing_file
                )
            return StaticAnalysisResult(fires=False)

        col = _Collector()
        try:
            _pf_api.check(full, "<pred>", col)
        except Exception:
            self.parse_failures += 1
            if self.fallback is not None:
                return self.fallback.analyze(
                    prediction, x_left, x_right, importing_file=importing_file
                )
            return StaticAnalysisResult(fires=False)

        names = sorted({n for (ln, n) in col.undefined if start <= ln <= end})
        return StaticAnalysisResult(
            fires=bool(names),
            out_of_scope_identifiers=names,
            significant_out_of_scope=names,
            n_used_identifiers=len(names),
        )
