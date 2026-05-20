"""Tier 3: detect wrong-source imports in the prediction.

Catches the failure where a confidently-generated identifier exists in the
repository but the prediction attributes it to the wrong module. Typical
hallucinations look like::

    from db.client import save_record   # save_record actually lives in
                                          # db.persistence.save_record

The check is deliberately conservative: we abstain whenever the imported
module is external (not part of the repo) or when ``__init__.py``
re-exports could legitimately explain the binding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .modules import path_to_module
from .parser import parse
from .symbol_table import RepositorySymbolTable


@dataclass(frozen=True)
class ImportIssue:
    """A single suspicious import binding."""

    name: str                       # the imported symbol's local name
    imported_from: str              # textual module in the ``from X import Y`` clause
    expected_modules: tuple[str, ...]  # repo modules where ``name`` actually lives
    kind: str                       # "wrong_origin" | "missing_in_module"


def _decode(src: bytes, node) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _module_text(module_node, src: bytes) -> str:
    """Strip leading-dot relativity markers from a module reference.

    ``from .core import X``    -> ``".core"``
    ``from ..pkg.core import X`` -> ``"..pkg.core"``
    We retain the dots so callers can detect relative vs absolute, then
    normalise via :func:`_resolve_module` against the importing file.
    """
    return _decode(src, module_node).strip()


def _resolve_module(
    raw_module: str,
    importing_module: str,
) -> tuple[str, int]:
    """Resolve a (possibly relative) module reference to an absolute path.

    Returns ``(absolute_module, dot_level)`` where ``dot_level`` is the
    number of leading dots in the source (0 for absolute, 1 for ``from .``,
    etc.). The absolute module is the best-effort dotted name relative to
    the repo root.
    """
    dot_level = 0
    for ch in raw_module:
        if ch == ".":
            dot_level += 1
        else:
            break
    tail = raw_module[dot_level:]
    if dot_level == 0:
        return tail, 0
    # Strip ``dot_level - 1`` trailing components from importing_module
    # (the first dot means "the current package").
    parts = importing_module.split(".") if importing_module else []
    # ``importing_module`` for ``pkg/sub/foo.py`` is ``"pkg.sub.foo"``; the
    # current package is ``pkg.sub``. So ``from . import x`` resolves to
    # ``pkg.sub.x`` -> we drop the final segment (the file) once, then
    # ``dot_level - 1`` more for each additional dot.
    drop = dot_level
    if drop > len(parts):
        # Relative import escapes the repo — abstain by returning empty.
        return tail, dot_level
    base = parts[: len(parts) - drop]
    if tail:
        base.append(tail)
    return ".".join(base), dot_level


def _is_external(module: str, repo: RepositorySymbolTable) -> bool:
    """A module is external iff no file in the repo maps to it (or any of
    its parent packages).
    """
    if not module:
        return True
    if repo.file_for_module(module) is not None:
        return False
    # If a child module exists (e.g. ``pkg.sub`` doesn't map but ``pkg.sub.foo``
    # does), the parent is still part of the repo.
    prefix = module + "."
    for info in repo.modules():
        if info.module == module or info.module.startswith(prefix):
            return False
    return True


def _name_is_reexported_from(
    name: str,
    module: str,
    repo: RepositorySymbolTable,
) -> bool:
    """Is ``name`` legitimately accessible via ``from module import name``?

    True if the ``__init__.py`` of ``module`` either:
      * declares ``name`` in ``__all__``, or
      * re-imports ``name`` from a submodule (parsed as a reexport binding).
    """
    info_path = repo.file_for_module(module)
    if info_path is None:
        return False
    info = repo.module_info(info_path)
    if info is None:
        return False
    if info.all_export is not None and name in info.all_export:
        return True
    if name in info.reexports:
        return True
    return False


def _name_is_in_module(
    name: str,
    module: str,
    repo: RepositorySymbolTable,
) -> bool:
    sites = repo.sites(name)
    return any(s.module == module for s in sites)


def check_imports(
    prediction_source: str,
    importing_file: str,
    repo: RepositorySymbolTable,
) -> list[ImportIssue]:
    """Validate ``from X import Y`` style statements in ``prediction_source``.

    ``importing_file`` is the path of the file the prediction is being
    inserted into (used to resolve relative imports). Pass ``""`` if the
    file path isn't meaningful — relative imports will then abstain.
    """
    if not prediction_source.strip():
        return []
    try:
        tree = parse(prediction_source)
    except Exception:
        return []
    src = prediction_source.encode("utf-8")
    importing_module = path_to_module(importing_file) if importing_file else ""
    issues: list[ImportIssue] = []

    def handle_from_import(node) -> None:
        module_node = node.child_by_field_name("module_name")
        if module_node is None:
            return
        raw_module = _module_text(module_node, src)
        resolved, dot_level = _resolve_module(raw_module, importing_module)
        if dot_level > 0 and not importing_module:
            # Relative import but we don't know where we are -- abstain.
            return
        if _is_external(resolved, repo):
            # External package (numpy, requests, stdlib). Can't validate.
            return
        module_id = module_node.id
        for c in node.children:
            if c.id == module_id:
                # Skip the module_name node itself. tree-sitter 0.23.x
                # returns fresh wrappers from child_by_field_name, so ``is``
                # is unreliable — compare ids instead.
                continue
            if c.type == "dotted_name":
                # ``from X import Y`` -- Y is the first identifier.
                first = next(
                    (sub for sub in c.children if sub.type == "identifier"),
                    None,
                )
                if first is None:
                    continue
                name = _decode(src, first)
                _check_one(name, raw_module, resolved)
            elif c.type == "aliased_import":
                name_field = c.child_by_field_name("name")
                if name_field is None:
                    continue
                if name_field.type == "dotted_name":
                    first = next(
                        (sub for sub in name_field.children if sub.type == "identifier"),
                        None,
                    )
                    if first is None:
                        continue
                    name = _decode(src, first)
                else:
                    name = _decode(src, name_field)
                _check_one(name, raw_module, resolved)
            elif c.type == "wildcard_import":
                # ``from X import *`` -- abstain (re-export semantics).
                return

    def _check_one(name: str, raw_module: str, resolved: str) -> None:
        # Step 1: legitimate via __all__ or __init__.py reexport.
        if _name_is_reexported_from(name, resolved, repo):
            return
        # Step 2: directly defined in the named module?
        if _name_is_in_module(name, resolved, repo):
            return
        # Step 3: defined elsewhere in the repo?
        expected = sorted(repo.module_of(name))
        if not expected:
            # Not in the repo at all -- the bare-name unresolved check will
            # handle it. Avoid double-flagging.
            return
        # The name lives elsewhere -- flag wrong origin.
        issues.append(
            ImportIssue(
                name=name,
                imported_from=raw_module,
                expected_modules=tuple(m for m in expected if m),
                kind="wrong_origin",
            )
        )

    def walk(n) -> None:
        if n.type == "import_from_statement":
            handle_from_import(n)
            return  # don't descend
        for c in n.children:
            walk(c)

    walk(tree.root_node)
    return issues


def check_attribute_usage(
    prediction_source: str,
    repo: RepositorySymbolTable,
    in_file_imports: Iterable[tuple[str, str]] = (),
) -> list[ImportIssue]:
    """Validate ``module.Name`` accesses in ``prediction_source``.

    ``in_file_imports`` is an iterable of ``(local_name, dotted_module)``
    pairs collected from ``import`` and ``from ... import`` statements in
    the surrounding in-file context. Only the ``import X`` form (where
    ``X`` is bound directly) is checked here; ``from X import Y`` is the
    province of :func:`check_imports`.

    Fires when ``module.Name`` is used, ``module`` was imported as a known
    repo module, and ``Name`` is not defined in that module (or re-exported
    by its ``__init__.py``).
    """
    if not prediction_source.strip():
        return []
    try:
        tree = parse(prediction_source)
    except Exception:
        return []
    src = prediction_source.encode("utf-8")
    bindings = {local: mod for local, mod in in_file_imports}
    issues: list[ImportIssue] = []

    def walk(n) -> None:
        if n.type == "attribute":
            obj = n.child_by_field_name("object")
            attr = n.child_by_field_name("attribute")
            if (
                obj is not None
                and attr is not None
                and obj.type == "identifier"
                and attr.type == "identifier"
            ):
                local = _decode(src, obj)
                attr_name = _decode(src, attr)
                module = bindings.get(local)
                if module is not None and not _is_external(module, repo):
                    if not (
                        _name_is_in_module(attr_name, module, repo)
                        or _name_is_reexported_from(attr_name, module, repo)
                    ):
                        expected = sorted(repo.module_of(attr_name))
                        if expected:
                            issues.append(
                                ImportIssue(
                                    name=attr_name,
                                    imported_from=module,
                                    expected_modules=tuple(m for m in expected if m),
                                    kind="missing_in_module",
                                )
                            )
        for c in n.children:
            walk(c)

    walk(tree.root_node)
    return issues
