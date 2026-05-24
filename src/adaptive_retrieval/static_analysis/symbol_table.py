"""Repository symbol table built by walking every .py file's defs.

Adds two layers on top of the original "set of names" model:

* Per-definition :class:`DefSite` records (file, dotted module name, kind,
  signature when available). This is what Tier 3 (import validation) and
  Tier 2 (signature checking) consume.
* Class-method tables so ``ClassName.method(...)`` can be resolved without
  receiver-type inference.

The legacy ``contains(name)`` and ``__contains__`` accessors are preserved
so the existing cross-file ablation path keeps working unchanged.
"""
from __future__ import annotations

import builtins as _builtins
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .modules import parse_all_export, parse_reexports, path_to_module
from .parser import parse
from .signatures import Signature, extract_signature

PYTHON_BUILTINS = set(dir(_builtins)) | {"self", "cls", "True", "False", "None"}

# Per Appendix C.7 of IMPLEMENTATION_GUIDE.md: filter common short names from
# the cross-file symbol table to avoid spurious "cross_file" classifications.
COMMON_SHORT_NAMES = {
    "result", "data", "tmp", "temp", "val", "value", "key", "item", "items",
    "elem", "obj", "ret", "res", "args", "kwargs", "name", "names", "count",
    "total", "size", "idx", "index", "lst", "arr", "out", "input", "output",
    "buf", "buffer", "msg", "err",
}


def _is_filtered_name(name: str) -> bool:
    """Names too common to be diagnostic if found cross-file."""
    return len(name) <= 1 or name in COMMON_SHORT_NAMES


@dataclass(frozen=True)
class DefSite:
    """Where a name was defined in the repository."""

    file: str            # repo-relative path (or absolute, when built from a fs root)
    module: str          # dotted module name; "" for files at the repo root
    kind: str            # "function" | "class" | "method" | "variable"
    line: int            # 0-indexed line of the def
    signature: Signature | None = None
    # For methods, the class they belong to. Empty for module-level defs.
    enclosing_class: str = ""

    def __post_init__(self) -> None:
        # Defensive: dataclass forbids mutation, but we want to assert the
        # invariants so downstream code can trust them.
        if self.kind == "method" and not self.enclosing_class:
            object.__setattr__(self, "enclosing_class", "")  # no-op


@dataclass
class ModuleInfo:
    """Per-file metadata. Tier 3 reads ``reexports`` and ``all_export``."""

    file: str
    module: str
    reexports: dict[str, str] = field(default_factory=dict)
    all_export: set[str] | None = None


class RepositorySymbolTable:
    """All names defined anywhere in the repository's .py files.

    Supports two construction modes: from a filesystem directory
    (``RepositorySymbolTable(path)``) or from an in-memory file map
    (``RepositorySymbolTable.from_files({"a.py": "..."})``). The in-memory
    path is what CrossCodeEval uses — instances ship cross-file chunks as a
    list rather than a directory tree.
    """

    def __init__(self, repo_root: str | Path | None, filter_common: bool = True):
        # Legacy storage — list of (file, kind, line) tuples per name.
        self.symbols: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
        # Richer per-definition record store.
        self._sites: dict[str, list[DefSite]] = defaultdict(list)
        # Per-class methods.
        self._class_methods: dict[str, dict[str, Signature]] = defaultdict(dict)
        # Per-module metadata.
        self._modules: dict[str, ModuleInfo] = {}
        # Reverse lookup: module name -> file path.
        self._module_to_file: dict[str, str] = {}
        # Filter flag.
        self.filter_common = filter_common
        # Track the on-disk repo root (only set when constructed from fs).
        self._fs_root: str = ""
        if repo_root is not None:
            self._fs_root = str(Path(repo_root))
            self._build(Path(repo_root))

    # ---------- construction ----------

    @classmethod
    def from_files(
        cls,
        files: dict[str, str],
        filter_common: bool = True,
    ) -> "RepositorySymbolTable":
        """Build from an in-memory ``{file_path: content}`` map."""
        inst = cls(None, filter_common=filter_common)
        # Pass 1: register module info (reexports / __all__) so cross-references
        # below resolve cleanly.
        for path, content in files.items():
            module = path_to_module(path)
            info = ModuleInfo(file=path, module=module)
            if path.endswith("__init__.py"):
                info.reexports = parse_reexports(content)
                info.all_export = parse_all_export(content)
            inst._modules[path] = info
            if module:
                inst._module_to_file[module] = path
        # Pass 2: definitions + class methods.
        for path, content in files.items():
            try:
                tree = parse(content)
            except Exception:
                continue
            src_bytes = content.encode("utf-8")
            inst._walk_defs(tree.root_node, src_bytes, path)
        return inst

    def _build(self, root: Path) -> None:
        # First pass: register all modules so per-module mapping is complete
        # before per-file def walks need it.
        files: list[tuple[Path, str]] = []
        for py_file in root.rglob("*.py"):
            try:
                src = py_file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            files.append((py_file, src))
            rel = str(py_file.relative_to(root)) if py_file.is_relative_to(root) else str(py_file)
            module = path_to_module(rel)
            info = ModuleInfo(file=str(py_file), module=module)
            if py_file.name == "__init__.py":
                info.reexports = parse_reexports(src)
                info.all_export = parse_all_export(src)
            self._modules[str(py_file)] = info
            if module:
                self._module_to_file[module] = str(py_file)
        for py_file, src in files:
            try:
                tree = parse(src)
            except Exception:
                continue
            self._walk_defs(tree.root_node, src.encode("utf-8"), str(py_file))

    # ---------- def walking ----------

    def _module_for(self, file_path: str) -> str:
        info = self._modules.get(file_path)
        if info is not None:
            return info.module
        return path_to_module(file_path)

    def _add(
        self,
        name: str,
        file_path: str,
        kind: str,
        line: int,
        signature: Signature | None = None,
        enclosing_class: str = "",
    ) -> None:
        if self.filter_common and _is_filtered_name(name):
            return
        self.symbols[name].append((file_path, kind, line))
        self._sites[name].append(
            DefSite(
                file=file_path,
                module=self._module_for(file_path),
                kind=kind,
                line=line,
                signature=signature,
                enclosing_class=enclosing_class,
            )
        )

    def _walk_defs(
        self,
        node,
        src_bytes: bytes,
        file_path: str,
        enclosing_class: str = "",
    ) -> None:
        t = node.type
        if t == "function_definition":
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                name = src_bytes[name_node.start_byte:name_node.end_byte].decode()
                line = name_node.start_point[0]
                sig = extract_signature(node, src_bytes, is_method=bool(enclosing_class))
                kind = "method" if enclosing_class else "function"
                self._add(
                    name, file_path, kind, line,
                    signature=sig, enclosing_class=enclosing_class,
                )
                if enclosing_class:
                    self._class_methods[enclosing_class][name] = sig
            # Don't recurse into function bodies — nested defs aren't part of
            # the public symbol table.
            return
        if t == "decorated_definition":
            # tree-sitter wraps decorated defs. Inspect the inner def with
            # the decorator context already attached (extract_signature peeks
            # at parent.decorator children).
            for c in node.children:
                self._walk_defs(c, src_bytes, file_path, enclosing_class)
            return
        if t == "class_definition":
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                name = src_bytes[name_node.start_byte:name_node.end_byte].decode()
                line = name_node.start_point[0]
                self._add(name, file_path, "class", line)
                body = node.child_by_field_name("body")
                if body is not None:
                    for c in body.children:
                        self._walk_defs(c, src_bytes, file_path, enclosing_class=name)
            return
        if t == "assignment":
            left = node.child_by_field_name("left")
            if left is not None and not enclosing_class:
                if left.type == "identifier":
                    name = src_bytes[left.start_byte:left.end_byte].decode()
                    self._add(name, file_path, "variable", left.start_point[0])
                elif left.type in ("pattern_list", "tuple_pattern"):
                    for sub in left.children:
                        if sub.type == "identifier":
                            name = src_bytes[sub.start_byte:sub.end_byte].decode()
                            self._add(name, file_path, "variable", sub.start_point[0])
            return
        for child in node.children:
            self._walk_defs(child, src_bytes, file_path, enclosing_class)

    # ---------- accessors ----------

    def contains(self, name: str) -> bool:
        return name in self.symbols or name in PYTHON_BUILTINS

    def sites(self, name: str) -> list[DefSite]:
        """All places ``name`` is defined in the repo."""
        return list(self._sites.get(name, ()))

    def signatures(self, name: str) -> list[Signature]:
        """All signatures for ``name`` across the repo (functions + methods)."""
        return [s.signature for s in self._sites.get(name, ()) if s.signature is not None]

    def methods_of(self, class_name: str) -> dict[str, Signature]:
        """Methods of ``class_name`` keyed by method name."""
        return dict(self._class_methods.get(class_name, {}))

    def module_of(self, name: str) -> set[str]:
        """Modules in which ``name`` is directly defined (no re-exports)."""
        return {s.module for s in self._sites.get(name, ())}

    def file_for_module(self, module: str) -> str | None:
        return self._module_to_file.get(module)

    def module_info(self, file_path: str) -> ModuleInfo | None:
        return self._modules.get(file_path)

    def modules(self) -> list[ModuleInfo]:
        return list(self._modules.values())

    def reexports_of(self, module: str) -> dict[str, str]:
        """Re-exports declared in ``module``'s ``__init__.py`` (or {} if none)."""
        path = self._module_to_file.get(module)
        if path is None:
            return {}
        info = self._modules.get(path)
        if info is None:
            return {}
        return dict(info.reexports)

    def __len__(self) -> int:
        return len(self.symbols)

    def __contains__(self, name: str) -> bool:
        return self.contains(name)
