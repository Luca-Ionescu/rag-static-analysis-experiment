"""Decide whether retrieval should be triggered by inspecting a prediction.

Three orthogonal static-analysis tiers, each toggleable independently:

  Tier 1 — out-of-scope identifiers (no symbol-table dependency).
    Fires when any used identifier in a *structurally significant position*
    (call target, attribute receiver, subscript value, class base, bare
    decorator, exception type, raise target) is not visible at the hole.
    The cross_file vs unresolved distinction the previous design made — a
    proxy for failure attribution that doesn't carry semantic weight — is
    deliberately collapsed: both warrant retrieval, and Tier 1 itself does
    not query the repository symbol table.

  Tier 2 — signature mismatch (uses the symbol table for callee signatures).
    For each ``call`` node in the prediction, resolve the callee to a unique
    :class:`Signature` from the repo and verify positional arity + keyword
    argument names. Catches "right name, wrong arguments" hallucinations.

  Tier 3 — wrong-origin imports (uses the symbol table for module mapping).
    For each ``from X import Y`` in the prediction, verify Y is defined in
    or re-exported by X. Catches "right name, wrong module" hallucinations.

Tiers 2 and 3 are precise structural checks on properties of the prediction
that require an indexed view of the repository — the symbol table is the
right data layer for them. Tier 1's single signal does not consult that
layer at all.

Each tier can be turned off via a constructor flag, enabling the
A1 ablation matrix (each-tier-alone, all-on, all-off).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .call_check import CallIssue, _module_bindings, check_calls
from .import_check import ImportIssue, check_attribute_usage, check_imports
from .parser import parse
from .scope import InFileScopeAnalyzer
from .symbol_table import RepositorySymbolTable


@dataclass
class StaticAnalysisResult:
    fires: bool
    # Tier 1 — no symbol table consulted.
    out_of_scope_identifiers: list[str] = field(default_factory=list)
    significant_out_of_scope: list[str] = field(default_factory=list)
    # Tier 2 — uses symbol table.
    signature_issues: list[CallIssue] = field(default_factory=list)
    # Tier 3 — uses symbol table.
    import_issues: list[ImportIssue] = field(default_factory=list)
    n_used_identifiers: int = 0


def _decode(src: bytes, node) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


class PredictionAnalyzer:
    """Three-tier static-analysis gate.

    Args:
        scope_analyzer: in-file scope resolver (no repo dependency).
        repo_symbols: repository symbol table. Used only by Tier 2 and Tier 3;
            Tier 1's decision does not query it.
        fire_on_out_of_scope: enable Tier 1.
        fire_on_signature: enable Tier 2.
        fire_on_import: enable Tier 3.
    """

    def __init__(
        self,
        scope_analyzer: InFileScopeAnalyzer,
        repo_symbols: RepositorySymbolTable,
        fire_on_out_of_scope: bool = True,
        fire_on_signature: bool = True,
        fire_on_import: bool = True,
    ):
        self.scope = scope_analyzer
        self.repo = repo_symbols
        self.fire_on_out_of_scope = fire_on_out_of_scope
        self.fire_on_signature = fire_on_signature
        self.fire_on_import = fire_on_import

    def analyze(
        self,
        prediction: str,
        x_left: str,
        x_right: str,
        importing_file: str = "",
    ) -> StaticAnalysisResult:
        if not prediction:
            return StaticAnalysisResult(fires=False)

        full_source = x_left + prediction + x_right
        hole_byte = len(x_left.encode("utf-8"))

        # Tier 1: out-of-scope check. Does NOT touch the symbol table.
        visible = self.scope.visible_at(full_source, hole_byte)
        used, significant = self._extract_used_identifiers(prediction)
        out_of_scope = sorted(n for n in used if n not in visible)
        sig_out_of_scope = sorted(n for n in significant if n not in visible)

        # Tier 2: signature check. Queries the symbol table for callee signatures.
        signature_issues = check_calls(
            prediction, x_left + x_right, self.repo, importing_file=importing_file,
        )

        # Tier 3: import validation. Queries the symbol table for module mapping
        # and per-file reexports. Both wrong-origin and attribute-usage forms.
        import_issues: list[ImportIssue] = []
        import_issues.extend(check_imports(prediction, importing_file, self.repo))
        attr_bindings = _module_bindings(x_left + x_right)
        import_issues.extend(
            check_attribute_usage(prediction, self.repo, attr_bindings.items())
        )

        fires = (
            (self.fire_on_out_of_scope and bool(sig_out_of_scope))
            or (self.fire_on_signature and bool(signature_issues))
            or (self.fire_on_import and bool(import_issues))
        )
        return StaticAnalysisResult(
            fires=fires,
            out_of_scope_identifiers=out_of_scope,
            significant_out_of_scope=sig_out_of_scope,
            signature_issues=signature_issues,
            import_issues=import_issues,
            n_used_identifiers=len(used),
        )

    # ---------- identifier extraction ----------

    def _extract_used_identifiers(self, code: str) -> tuple[set[str], set[str]]:
        """Names that are USED (not bound) in ``code``.

        Returns ``(used, significant)`` where ``significant`` is the subset of
        ``used`` whose occurrences include at least one structurally
        significant position. Bindings (assignment LHS, def names, params,
        for-vars, comprehension vars, lambda params, walrus targets,
        with/except aliases) are subtracted so e.g. ``lambda x: x*2`` does
        not flag ``x``.
        """
        tree = parse(code)
        src = code.encode("utf-8")
        used: set[str] = set()
        significant: set[str] = set()
        defined_locally: set[str] = set()
        defining_nodes: set[int] = set()

        def mark_param(param) -> None:
            t = param.type
            if t == "identifier":
                defined_locally.add(_decode(src, param))
                defining_nodes.add(param.id)
            elif t in ("default_parameter", "typed_parameter", "typed_default_parameter"):
                pn = param.child_by_field_name("name")
                if pn is not None:
                    defined_locally.add(_decode(src, pn))
                    defining_nodes.add(pn.id)
                else:
                    for c in param.children:
                        if c.type == "identifier":
                            defined_locally.add(_decode(src, c))
                            defining_nodes.add(c.id)
                            break
            elif t in ("list_splat_pattern", "dictionary_splat_pattern"):
                for c in param.children:
                    if c.type == "identifier":
                        defined_locally.add(_decode(src, c))
                        defining_nodes.add(c.id)

        def mark_pattern_list(node) -> None:
            for sub in node.children:
                if sub.type == "identifier":
                    defined_locally.add(_decode(src, sub))
                    defining_nodes.add(sub.id)

        def is_significant_position(ident_node) -> bool:
            """True iff this identifier sits in a position whose absence
            from scope would constitute a real semantic error.

            Recognised: call target, attribute receiver, subscript value,
            class base, bare decorator, exception type (including tuple and
            aliased forms), raise target.
            """
            parent = ident_node.parent
            if parent is None:
                return False
            pt = parent.type

            if pt == "call":
                fn = parent.child_by_field_name("function")
                if fn is not None and fn.id == ident_node.id:
                    return True
            elif pt == "attribute":
                obj = parent.child_by_field_name("object")
                if obj is not None and obj.id == ident_node.id:
                    return True
            elif pt == "subscript":
                val = parent.child_by_field_name("value")
                if val is not None and val.id == ident_node.id:
                    return True
            elif pt == "argument_list":
                # Class bases — argument_list parent is class_definition with @superclasses
                gp = parent.parent
                if gp is not None and gp.type == "class_definition":
                    sup = gp.child_by_field_name("superclasses")
                    if sup is not None and sup.id == parent.id:
                        return True
            elif pt == "decorator":
                # Bare decorator: @my_decorator (with call is caught via call rule).
                return True
            elif pt == "except_clause":
                return True
            elif pt == "tuple":
                gp = parent.parent
                if gp is not None and gp.type == "except_clause":
                    return True
            elif pt == "as_pattern":
                gp = parent.parent
                if gp is not None and gp.type == "except_clause":
                    return True
            elif pt == "raise_statement":
                return True
            return False

        def walk(node) -> None:
            t = node.type

            # Identifiers inside import statements are bindings, not uses;
            # Tier 3 validates them via :mod:`import_check`.
            if t in ("import_statement", "import_from_statement"):
                for sub in _all_descendant_identifiers(node):
                    defined_locally.add(_decode(src, sub))
                    defining_nodes.add(sub.id)
                return

            if t == "assignment":
                left = node.child_by_field_name("left")
                if left is not None:
                    if left.type == "identifier":
                        defined_locally.add(_decode(src, left))
                        defining_nodes.add(left.id)
                    elif left.type in ("pattern_list", "tuple_pattern"):
                        mark_pattern_list(left)
            elif t in ("function_definition", "class_definition"):
                nn = node.child_by_field_name("name")
                if nn is not None:
                    defined_locally.add(_decode(src, nn))
                    defining_nodes.add(nn.id)
                params = node.child_by_field_name("parameters")
                if params is not None:
                    for p in params.children:
                        mark_param(p)
            elif t == "lambda":
                params = node.child_by_field_name("parameters")
                if params is not None:
                    for p in params.children:
                        mark_param(p)
            elif t == "for_statement":
                left = node.child_by_field_name("left")
                if left is not None:
                    if left.type == "identifier":
                        defined_locally.add(_decode(src, left))
                        defining_nodes.add(left.id)
                    elif left.type in ("pattern_list", "tuple_pattern"):
                        mark_pattern_list(left)
            elif t == "for_in_clause":
                left = node.child_by_field_name("left")
                if left is not None:
                    if left.type == "identifier":
                        defined_locally.add(_decode(src, left))
                        defining_nodes.add(left.id)
                    elif left.type in ("pattern_list", "tuple_pattern"):
                        mark_pattern_list(left)
            elif t == "named_expression":
                target = node.child_by_field_name("name")
                if target is not None and target.type == "identifier":
                    defined_locally.add(_decode(src, target))
                    defining_nodes.add(target.id)
            elif t == "as_pattern":
                alias = node.child_by_field_name("alias")
                if alias is not None:
                    if alias.type == "identifier":
                        defined_locally.add(_decode(src, alias))
                        defining_nodes.add(alias.id)
                    elif alias.type == "as_pattern_target":
                        for c in alias.children:
                            if c.type == "identifier":
                                defined_locally.add(_decode(src, c))
                                defining_nodes.add(c.id)
                                break

            if t == "identifier" and node.id not in defining_nodes:
                parent = node.parent
                is_attribute_name = False
                if parent is not None and parent.type == "attribute":
                    attr_name = parent.child_by_field_name("attribute")
                    if attr_name is not None and attr_name.id == node.id:
                        is_attribute_name = True
                is_kwarg_name = False
                if parent is not None and parent.type == "keyword_argument":
                    name_field = parent.child_by_field_name("name")
                    if name_field is not None and name_field.id == node.id:
                        is_kwarg_name = True
                if not is_attribute_name and not is_kwarg_name:
                    name = _decode(src, node)
                    used.add(name)
                    if is_significant_position(node):
                        significant.add(name)

            for c in node.children:
                walk(c)

        walk(tree.root_node)
        used -= defined_locally
        significant -= defined_locally
        return used, significant


def _all_descendant_identifiers(node):
    """Yield every ``identifier`` node nested under ``node`` (any depth)."""
    if node.type == "identifier":
        yield node
        return
    for c in node.children:
        yield from _all_descendant_identifiers(c)
