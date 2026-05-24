"""Decide whether retrieval should be triggered by inspecting a prediction.

The trigger is a single signal: **any identifier used in a structurally
significant position that is not visible in the in-file scope at the hole**.
We deliberately do not distinguish whether such a name happens to appear
elsewhere in the repository symbol table — the action is the same either
way (retrieve), and the distinction was a proxy for failure-mode attribution
that doesn't carry strong semantic weight (a name being in our chunk-derived
symbol table doesn't mean the prediction is using it correctly).

The symbol table is still consulted to populate the loose lists below for
the RSP metric and post-hoc diagnostics, but the trigger decision doesn't
read it.

Structurally significant positions (the trigger considers only these):
    - call target            (``f`` in ``f(x)``)
    - attribute receiver     (``foo`` in ``foo.bar``)
    - subscript value        (``arr`` in ``arr[0]``)
    - class base             (``Bar`` in ``class Foo(Bar):``)
    - bare decorator         (``my_decorator`` in ``@my_decorator``)
    - exception type         (``E`` in ``except E:`` or ``except (E1, E2):``
                              or ``except E as e:``)
    - raise target           (``E`` in ``raise E`` or ``raise E from F``)

Bare identifiers in expression positions (binary-op operands, call
arguments, return values) are not significant — they're typically local
variables our scope analysis can't see, and firing on them blows up the
false-positive rate without indicating a real issue.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .parser import parse
from .scope import InFileScopeAnalyzer
from .symbol_table import RepositorySymbolTable


@dataclass
class StaticAnalysisResult:
    fires: bool
    # Loose lists — every classified identifier. Used for RSP and diagnostics.
    # Kept separately because RSP treats "in repo, not visible" as a partial
    # resolution; the cascade trigger does not.
    unresolved_identifiers: list[str] = field(default_factory=list)
    cross_file_identifiers: list[str] = field(default_factory=list)
    # The trigger signal — significant identifiers that are not visible at
    # the hole (whether they happen to be in the repo or not).
    significant_out_of_scope: list[str] = field(default_factory=list)
    n_used_identifiers: int = 0


def _decode(src: bytes, node) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


class PredictionAnalyzer:
    """Classifies the identifiers used in a prediction.

    The trigger is binary: fires iff any used identifier in a significant
    position is not visible in the in-file scope at the hole.
    """

    def __init__(
        self,
        scope_analyzer: InFileScopeAnalyzer,
        repo_symbols: RepositorySymbolTable,
    ):
        self.scope = scope_analyzer
        self.repo = repo_symbols

    def analyze(self, prediction: str, x_left: str, x_right: str) -> StaticAnalysisResult:
        if not prediction:
            return StaticAnalysisResult(fires=False)

        full_source = x_left + prediction + x_right
        hole_byte = len(x_left.encode("utf-8"))

        visible = self.scope.visible_at(full_source, hole_byte)
        used, significant = self._extract_used_identifiers(prediction)

        # Loose classification — repo lookup is consulted purely for RSP /
        # diagnostics. The trigger does not use these lists.
        cross_file: list[str] = []
        unresolved: list[str] = []
        out_of_scope: set[str] = set()
        for name in sorted(used):
            if name in visible:
                continue
            out_of_scope.add(name)
            if self.repo.contains(name):
                cross_file.append(name)
            else:
                unresolved.append(name)

        sig_out_of_scope = [n for n in sorted(significant) if n in out_of_scope]

        return StaticAnalysisResult(
            fires=bool(sig_out_of_scope),
            unresolved_identifiers=unresolved,
            cross_file_identifiers=cross_file,
            significant_out_of_scope=sig_out_of_scope,
            n_used_identifiers=len(used),
        )

    # ---------- identifier extraction ----------

    def _extract_used_identifiers(self, code: str) -> tuple[set[str], set[str]]:
        """Names that are USED (not bound) in ``code``.

        Returns ``(used, significant)`` where ``significant`` is the subset of
        ``used`` whose occurrences include at least one structurally
        significant position.

        Bindings recognised: assignment LHS (incl. tuple), function/class defs,
        for-loop vars, comprehension vars, lambda params, walrus targets,
        with/except aliases. These names are subtracted from the use set so a
        prediction like ``lambda x: x*2`` does not flag ``x``.
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
            """True iff this identifier sits in a position whose unresolved
            classification would constitute a real semantic error.

            See the module docstring for the full list of significant
            positions.
            """
            parent = ident_node.parent
            if parent is None:
                return False
            pt = parent.type

            # call target
            if pt == "call":
                fn = parent.child_by_field_name("function")
                if fn is not None and fn.id == ident_node.id:
                    return True
            # attribute receiver
            elif pt == "attribute":
                obj = parent.child_by_field_name("object")
                if obj is not None and obj.id == ident_node.id:
                    return True
            # subscript value
            elif pt == "subscript":
                val = parent.child_by_field_name("value")
                if val is not None and val.id == ident_node.id:
                    return True
            # class base (positional). `class Foo(Bar):` — Bar is a child of
            # argument_list which is @superclasses on class_definition.
            elif pt == "argument_list":
                gp = parent.parent
                if gp is not None and gp.type == "class_definition":
                    sup = gp.child_by_field_name("superclasses")
                    if sup is not None and sup.id == parent.id:
                        return True
            # bare decorator: @my_decorator (no call). Decorated form
            # @my_decorator() is caught via the call-target rule above.
            elif pt == "decorator":
                return True
            # exception type: except E:
            elif pt == "except_clause":
                return True
            # exception type in tuple: except (E1, E2):
            elif pt == "tuple":
                gp = parent.parent
                if gp is not None and gp.type == "except_clause":
                    return True
            # exception type with alias: except E as e:
            elif pt == "as_pattern":
                gp = parent.parent
                if gp is not None and gp.type == "except_clause":
                    return True
            # raise target: raise E (and raise E from F — both bare ids)
            elif pt == "raise_statement":
                return True
            return False

        def walk(node) -> None:
            t = node.type

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
