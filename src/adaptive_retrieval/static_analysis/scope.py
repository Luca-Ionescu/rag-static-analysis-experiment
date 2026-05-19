"""In-file scope analysis: which identifiers are visible at a given byte position."""
from __future__ import annotations

import builtins as _builtins

from .parser import parse

PYTHON_BUILTINS = set(dir(_builtins)) | {"self", "cls", "True", "False", "None"}


def _decode(src: bytes, node) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


class InFileScopeAnalyzer:
    """Compute the set of names visible at a byte offset within a Python source.

    The result is a coarse over-approximation suitable for the cascade's
    static-analysis gate: it accepts module-level imports + module-level
    defs + any local binding found between the enclosing function start
    and the hole. It does not model control flow.
    """

    def visible_at(self, source: str, hole_byte: int) -> set[str]:
        tree = parse(source)
        src_bytes = source.encode("utf-8")
        visible: set[str] = set(PYTHON_BUILTINS)
        visible |= self._collect_imports(tree.root_node, src_bytes)
        visible |= self._collect_module_level_names(tree.root_node, src_bytes)
        visible |= self._collect_local_names_up_to(tree.root_node, src_bytes, hole_byte)
        return visible

    # ---------- imports ----------

    def _collect_imports(self, root, src: bytes) -> set[str]:
        """`import X`, `import X.Y`, `import X as Y`, `from M import Y`,
        `from M import Y as Z`. The introduced name is X (top of dotted),
        Y, or the alias Z respectively.
        """
        names: set[str] = set()

        def add_from_dotted(dotted_node) -> None:
            # `import X.Y.Z` introduces X (the first segment).
            for c in dotted_node.children:
                if c.type == "identifier":
                    names.add(_decode(src, c))
                    return

        def handle_import_statement(node) -> None:
            for c in node.children:
                if c.type == "dotted_name":
                    add_from_dotted(c)
                elif c.type == "aliased_import":
                    alias = c.child_by_field_name("alias")
                    if alias is not None:
                        names.add(_decode(src, alias))

        def handle_import_from_statement(node) -> None:
            module_node = node.child_by_field_name("module_name")
            for c in node.children:
                if c is module_node:
                    continue
                if c.type == "dotted_name":
                    add_from_dotted(c)
                elif c.type == "aliased_import":
                    alias = c.child_by_field_name("alias")
                    if alias is not None:
                        names.add(_decode(src, alias))
                    else:
                        nm = c.child_by_field_name("name")
                        if nm is not None:
                            if nm.type == "dotted_name":
                                add_from_dotted(nm)
                            else:
                                names.add(_decode(src, nm))

        def walk(n) -> None:
            if n.type == "import_statement":
                handle_import_statement(n)
            elif n.type == "import_from_statement":
                handle_import_from_statement(n)
            for c in n.children:
                walk(c)

        walk(root)
        return names

    # ---------- module-level names ----------

    def _collect_module_level_names(self, root, src: bytes) -> set[str]:
        names: set[str] = set()
        for child in root.children:
            t = child.type
            if t == "function_definition":
                nn = child.child_by_field_name("name")
                if nn is not None:
                    names.add(_decode(src, nn))
            elif t == "class_definition":
                nn = child.child_by_field_name("name")
                if nn is not None:
                    names.add(_decode(src, nn))
            elif t == "assignment":
                self._collect_assignment_targets(child, src, names)
            elif t == "expression_statement":
                # Annotated assignments at module level: `x: int = 1` or `x: int`
                for sub in child.children:
                    if sub.type in ("assignment",):
                        self._collect_assignment_targets(sub, src, names)
        return names

    @staticmethod
    def _collect_assignment_targets(assign_node, src: bytes, names: set[str]) -> None:
        left = assign_node.child_by_field_name("left")
        if left is None:
            return
        if left.type == "identifier":
            names.add(_decode(src, left))
        elif left.type in ("pattern_list", "tuple_pattern"):
            for sub in left.children:
                if sub.type == "identifier":
                    names.add(_decode(src, sub))

    # ---------- local names in enclosing function ----------

    def _collect_local_names_up_to(self, root, src: bytes, hole_byte: int) -> set[str]:
        names: set[str] = set()
        enclosing = self._find_enclosing_callable(root, hole_byte)
        if enclosing is None:
            return names

        params = enclosing.child_by_field_name("parameters")
        if params is not None:
            for param in params.children:
                self._collect_param_names(param, src, names)

        def walk_body(n) -> None:
            if n.start_byte > hole_byte:
                return
            self._collect_binding_for_node(n, src, names)
            for c in n.children:
                walk_body(c)

        body = enclosing.child_by_field_name("body")
        if body is not None:
            walk_body(body)
        else:
            # Lambdas have a body field too; if missing, walk the whole node.
            walk_body(enclosing)
        return names

    @staticmethod
    def _find_enclosing_callable(root, hole_byte: int):
        best = None

        def walk(n) -> None:
            nonlocal best
            if n.type in ("function_definition", "lambda"):
                if n.start_byte <= hole_byte <= n.end_byte:
                    best = n
            for c in n.children:
                if c.start_byte <= hole_byte <= c.end_byte:
                    walk(c)

        walk(root)
        return best

    @staticmethod
    def _collect_param_names(param_node, src: bytes, names: set[str]) -> None:
        t = param_node.type
        if t == "identifier":
            names.add(_decode(src, param_node))
        elif t in ("default_parameter", "typed_parameter", "typed_default_parameter"):
            pn = param_node.child_by_field_name("name")
            if pn is not None:
                names.add(_decode(src, pn))
            else:
                # `typed_parameter` may not expose a `name` field; walk for first identifier.
                for c in param_node.children:
                    if c.type == "identifier":
                        names.add(_decode(src, c))
                        break
        elif t in ("list_splat_pattern", "dictionary_splat_pattern"):
            for c in param_node.children:
                if c.type == "identifier":
                    names.add(_decode(src, c))

    @staticmethod
    def _collect_binding_for_node(n, src: bytes, names: set[str]) -> None:
        t = n.type
        if t == "assignment":
            left = n.child_by_field_name("left")
            if left is None:
                return
            if left.type == "identifier":
                names.add(_decode(src, left))
            elif left.type in ("pattern_list", "tuple_pattern"):
                for sub in left.children:
                    if sub.type == "identifier":
                        names.add(_decode(src, sub))
        elif t == "for_statement":
            left = n.child_by_field_name("left")
            if left is not None:
                if left.type == "identifier":
                    names.add(_decode(src, left))
                elif left.type in ("pattern_list", "tuple_pattern"):
                    for sub in left.children:
                        if sub.type == "identifier":
                            names.add(_decode(src, sub))
        elif t == "for_in_clause":
            left = n.child_by_field_name("left")
            if left is not None:
                if left.type == "identifier":
                    names.add(_decode(src, left))
                elif left.type in ("pattern_list", "tuple_pattern"):
                    for sub in left.children:
                        if sub.type == "identifier":
                            names.add(_decode(src, sub))
        elif t == "named_expression":
            # walrus: `(x := ...)` binds x in the enclosing scope.
            target = n.child_by_field_name("name")
            if target is not None and target.type == "identifier":
                names.add(_decode(src, target))
        elif t == "as_pattern":
            # `with X as Y:` and `except E as e:` — alias is wrapped in `as_pattern_target`.
            alias = n.child_by_field_name("alias")
            if alias is None:
                return
            if alias.type == "identifier":
                names.add(_decode(src, alias))
            elif alias.type == "as_pattern_target":
                for c in alias.children:
                    if c.type == "identifier":
                        names.add(_decode(src, c))
                        break
        elif t in ("function_definition", "class_definition"):
            nn = n.child_by_field_name("name")
            if nn is not None:
                names.add(_decode(src, nn))
