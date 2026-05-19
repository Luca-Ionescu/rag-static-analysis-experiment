"""Repository symbol table built by walking every .py file's top-level defs."""
from __future__ import annotations

import builtins as _builtins
from collections import defaultdict
from pathlib import Path

from .parser import parse

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


class RepositorySymbolTable:
    """All names defined anywhere in the repository's .py files.

    Supports two construction modes: from a filesystem directory
    (``RepositorySymbolTable(path)``) or from an in-memory file map
    (``RepositorySymbolTable.from_files({"a.py": "..."})``). The in-memory
    path is what CrossCodeEval uses — instances ship cross-file chunks as a
    list rather than a directory tree.
    """

    def __init__(self, repo_root: str | Path | None, filter_common: bool = True):
        self.symbols: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
        self.filter_common = filter_common
        if repo_root is not None:
            self._build(Path(repo_root))

    @classmethod
    def from_files(
        cls,
        files: dict[str, str],
        filter_common: bool = True,
    ) -> "RepositorySymbolTable":
        """Build from an in-memory ``{file_path: content}`` map."""
        inst = cls(None, filter_common=filter_common)
        for path, content in files.items():
            try:
                tree = parse(content)
            except Exception:
                continue
            inst._walk_defs(tree.root_node, content.encode("utf-8"), path)
        return inst

    def _build(self, root: Path) -> None:
        for py_file in root.rglob("*.py"):
            try:
                src = py_file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            try:
                tree = parse(src)
            except Exception:
                continue
            self._walk_defs(tree.root_node, src.encode("utf-8"), str(py_file))

    def _add(self, name: str, file_path: str, kind: str, line: int) -> None:
        if self.filter_common and _is_filtered_name(name):
            return
        self.symbols[name].append((file_path, kind, line))

    def _walk_defs(self, node, src_bytes: bytes, file_path: str) -> None:
        t = node.type
        if t == "function_definition":
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                name = src_bytes[name_node.start_byte:name_node.end_byte].decode()
                self._add(name, file_path, "function", name_node.start_point[0])
        elif t == "class_definition":
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                name = src_bytes[name_node.start_byte:name_node.end_byte].decode()
                self._add(name, file_path, "class", name_node.start_point[0])
        elif t == "assignment":
            left = node.child_by_field_name("left")
            if left is not None:
                if left.type == "identifier":
                    name = src_bytes[left.start_byte:left.end_byte].decode()
                    self._add(name, file_path, "variable", left.start_point[0])
                elif left.type in ("pattern_list", "tuple_pattern"):
                    for sub in left.children:
                        if sub.type == "identifier":
                            name = src_bytes[sub.start_byte:sub.end_byte].decode()
                            self._add(name, file_path, "variable", sub.start_point[0])
        for child in node.children:
            self._walk_defs(child, src_bytes, file_path)

    def contains(self, name: str) -> bool:
        return name in self.symbols or name in PYTHON_BUILTINS

    def __len__(self) -> int:
        return len(self.symbols)

    def __contains__(self, name: str) -> bool:
        return self.contains(name)
