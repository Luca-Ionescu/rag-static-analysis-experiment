"""Tree-sitter setup for Python."""
from __future__ import annotations

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspython.language())


def get_parser() -> Parser:
    return Parser(PY_LANGUAGE)


def parse(source: str | bytes):
    """Parse Python source and return the tree. Accepts str or bytes."""
    if isinstance(source, str):
        source = source.encode("utf-8")
    return get_parser().parse(source)
