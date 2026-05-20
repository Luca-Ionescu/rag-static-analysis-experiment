"""Path↔module mapping and ``__init__.py`` re-export parsing.

This module exists for Tier 3 (import validation). It does not depend on
tree-sitter at the module-name layer — that's pure string surgery — but does
use the project parser when extracting re-export bindings from
``__init__.py`` source so we benefit from the same parser configuration as
the rest of static_analysis.
"""
from __future__ import annotations

from os.path import normpath, splitext
from typing import Iterable

from .parser import parse


def path_to_module(path: str, repo_root: str = "") -> str:
    """Convert a repo-relative ``.py`` path to a dotted module name.

    ``pkg/sub/foo.py`` -> ``pkg.sub.foo``
    ``pkg/__init__.py`` -> ``pkg``
    ``foo.py`` -> ``foo``

    ``repo_root`` is stripped from the path before conversion. It must be a
    prefix of ``path`` (with or without a trailing slash); otherwise it is
    ignored and the full path is used.
    """
    p = normpath(path).replace("\\", "/")
    root = normpath(repo_root).replace("\\", "/") if repo_root else ""
    if root and root != "." and (p == root or p.startswith(root + "/")):
        p = p[len(root):].lstrip("/")
    base, ext = splitext(p)
    if ext != ".py":
        # Caller should only pass .py files; we keep going to be lenient.
        base = p
    if base.endswith("/__init__"):
        base = base[: -len("/__init__")]
    if base == "__init__":
        return ""
    return base.replace("/", ".")


def find_repo_root(files: Iterable[str]) -> str:
    """Best-effort common-prefix root for an in-memory repo.

    For CrossCodeEval-style ``{file_path: content}`` dicts the paths are
    already repo-relative, so the right answer is usually ``""``. We return
    the longest common directory prefix only when *all* files share it; this
    avoids accidentally rooting at ``"src"`` when one straggler lives at
    repo top level.
    """
    paths = [normpath(f).replace("\\", "/") for f in files]
    if not paths:
        return ""
    first = paths[0].split("/")
    common: list[str] = []
    for i, part in enumerate(first[:-1]):  # last segment is filename, skip
        if all(
            len(other.split("/")) > i and other.split("/")[i] == part
            for other in paths
        ):
            common.append(part)
        else:
            break
    return "/".join(common)


def parse_reexports(init_source: str) -> dict[str, str]:
    """Extract ``from .sub import Name`` style re-exports from an ``__init__.py``.

    Returns a mapping ``{local_name: relative_or_absolute_module}`` where
    ``module`` is the source the name was imported from (e.g. ``.core`` or
    ``pkg.core``). Star imports and conditional / runtime-dynamic re-exports
    are intentionally not handled — abstaining preserves the precision bias
    of the static-analysis gate.
    """
    reexports: dict[str, str] = {}
    if not init_source.strip():
        return reexports
    try:
        tree = parse(init_source)
    except Exception:
        return reexports
    src = init_source.encode("utf-8")

    def _decode(node) -> str:
        return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def _module_text(module_node) -> str:
        # Captures the textual module name; for relative imports tree-sitter
        # represents the leading dots as part of the module-name children.
        return _decode(module_node).strip()

    def walk(node) -> None:
        if node.type == "import_from_statement":
            module_node = node.child_by_field_name("module_name")
            if module_node is None:
                return
            module_text = _module_text(module_node)
            module_id = module_node.id
            for c in node.children:
                # tree-sitter 0.23.x returns fresh Python wrappers from
                # ``child_by_field_name``, so identity comparison via ``is``
                # is unreliable. Compare node ids instead.
                if c.id == module_id:
                    continue
                if c.type == "dotted_name":
                    # `from .x import Y` -- the imported name is Y itself.
                    for sub in c.children:
                        if sub.type == "identifier":
                            reexports[_decode(sub)] = module_text
                            break
                elif c.type == "aliased_import":
                    alias = c.child_by_field_name("alias")
                    name = c.child_by_field_name("name")
                    if alias is not None:
                        reexports[_decode(alias)] = module_text
                    elif name is not None:
                        if name.type == "dotted_name":
                            for sub in name.children:
                                if sub.type == "identifier":
                                    reexports[_decode(sub)] = module_text
                                    break
                        else:
                            reexports[_decode(name)] = module_text
            return
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return reexports


def parse_all_export(init_source: str) -> set[str] | None:
    """Best-effort parse of a top-level ``__all__ = [...]`` literal.

    Returns ``None`` if no ``__all__`` is present or if it isn't a plain
    list/tuple of string literals (e.g. computed at runtime). Returns the
    string set otherwise.
    """
    if "__all__" not in init_source:
        return None
    try:
        tree = parse(init_source)
    except Exception:
        return None
    src = init_source.encode("utf-8")

    def _decode(node) -> str:
        return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    for child in tree.root_node.children:
        if child.type != "expression_statement":
            continue
        for sub in child.children:
            if sub.type != "assignment":
                continue
            left = sub.child_by_field_name("left")
            right = sub.child_by_field_name("right")
            if left is None or right is None:
                continue
            if left.type != "identifier" or _decode(left) != "__all__":
                continue
            if right.type not in ("list", "tuple"):
                return None
            collected: set[str] = set()
            for item in right.children:
                if item.type == "string":
                    raw = _decode(item)
                    # Strip surrounding quotes (single, double, triple) without
                    # interpreting escapes — fine for the common case.
                    inner = raw.strip()
                    for q in ('"""', "'''", '"', "'"):
                        if inner.startswith(q) and inner.endswith(q):
                            inner = inner[len(q):-len(q)]
                            break
                    collected.add(inner)
            return collected
    return None
