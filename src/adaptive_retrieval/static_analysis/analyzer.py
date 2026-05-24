"""Decide whether retrieval should be triggered by inspecting a prediction."""
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
    unresolved_identifiers: list[str] = field(default_factory=list)
    cross_file_identifiers: list[str] = field(default_factory=list)
    signature_issues: list[CallIssue] = field(default_factory=list)
    import_issues: list[ImportIssue] = field(default_factory=list)
    n_used_identifiers: int = 0


def _decode(src: bytes, node) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _all_descendant_identifiers(node):
    """Yield every ``identifier`` node nested under ``node`` (any depth)."""
    if node.type == "identifier":
        yield node
        return
    for c in node.children:
        yield from _all_descendant_identifiers(c)


class PredictionAnalyzer:
    """Classifies the identifiers used in a prediction.

    A name used in the prediction is one of:
      - BUILTIN:    Python built-in or visible at the hole -> ignore
      - CROSS_FILE: in repo symbol table -> diagnostic only by default
      - UNRESOLVED: nowhere -> FIRE (likely hallucination)

    Beyond bare-name resolution, the analyzer also runs two signature-level
    checks:

      - signature: every resolvable call in the prediction is checked for
        positional arity and kwarg-name correctness against the resolved
        callee's :class:`Signature`. Catches "right name, wrong arguments"
        hallucinations such as ``find_file(path, recursive, deep)`` against
        ``def find_file(path)``.
      - import: every ``from X import Y`` in the prediction is validated
        against where ``Y`` actually lives in the repository, including
        ``__init__.py`` re-exports. Catches "right name, wrong source"
        hallucinations.

    The default fires on UNRESOLVED, signature, and import. A cross-file
    identifier means the model produced a name that already exists in the
    repository, so the parametric memory got it right; firing retrieval
    there spends budget on a case the model already nailed and risks
    retrieved-context distractor effects (see CARD §4.4, FLARE §3,
    Repoformer §3). The ``fire_on_crossfile`` flag is retained so the
    A1 ablation matrix can still explore the cross-file × unresolved 2x2.
    """

    def __init__(
        self,
        scope_analyzer: InFileScopeAnalyzer,
        repo_symbols: RepositorySymbolTable,
        fire_on_crossfile: bool = False,
        fire_on_unresolved: bool = True,
        fire_on_signature: bool = True,
        fire_on_import: bool = True,
    ):
        self.scope = scope_analyzer
        self.repo = repo_symbols
        self.fire_on_crossfile = fire_on_crossfile
        self.fire_on_unresolved = fire_on_unresolved
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

        visible = self.scope.visible_at(full_source, hole_byte)
        used = self._extract_used_identifiers(prediction)

        cross_file: list[str] = []
        unresolved: list[str] = []
        for name in sorted(used):
            if name in visible:
                continue
            if self.repo.contains(name):
                cross_file.append(name)
            else:
                unresolved.append(name)

        # Tier 2: signature checking.
        signature_issues = check_calls(
            prediction, x_left + x_right, self.repo, importing_file=importing_file,
        )

        # Tier 3: import validation. Both wrong-origin and attribute-usage forms.
        import_issues: list[ImportIssue] = []
        import_issues.extend(check_imports(prediction, importing_file, self.repo))
        # Build (local_name, module) bindings from in-file `import X` lines
        # so attribute usages in the prediction can be validated.
        attr_bindings = _module_bindings(x_left + x_right)
        import_issues.extend(
            check_attribute_usage(prediction, self.repo, attr_bindings.items())
        )

        fires = (
            (self.fire_on_crossfile and bool(cross_file))
            or (self.fire_on_unresolved and bool(unresolved))
            or (self.fire_on_signature and bool(signature_issues))
            or (self.fire_on_import and bool(import_issues))
        )
        return StaticAnalysisResult(
            fires=fires,
            unresolved_identifiers=unresolved,
            cross_file_identifiers=cross_file,
            signature_issues=signature_issues,
            import_issues=import_issues,
            n_used_identifiers=len(used),
        )

    # ---------- identifier extraction ----------

    def _extract_used_identifiers(self, code: str) -> set[str]:
        """Names that are USED (not bound) in `code`.

        Bindings recognised so far: assignment LHS (incl. tuple), function/class
        defs, for-loop vars, comprehension vars, lambda params, walrus targets,
        with/except aliases. These names are subtracted from the use set so a
        prediction like `lambda x: x*2` does not flag `x`.
        """
        tree = parse(code)
        src = code.encode("utf-8")
        used: set[str] = set()
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

        def walk(node) -> None:
            t = node.type

            # Identifiers inside import statements are bindings, not uses. The
            # static-import gate validates them via :mod:`import_check`; here
            # we just suppress them so they don't double-fire as "unresolved".
            if t in ("import_statement", "import_from_statement"):
                for sub in _all_descendant_identifiers(node):
                    defining_nodes.add(sub.id)
                    defined_locally.add(_decode(src, sub))
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
                # An identifier counts as a USE unless it's the LHS of an assignment
                # / the name of a def / a parameter / a binding alias. We've already
                # registered those node ids above.
                parent = node.parent
                is_attribute_name = False
                if parent is not None and parent.type == "attribute":
                    # In `foo.bar`, only `foo` is the receiver. The attribute identifier
                    # `bar` is NOT a standalone name use (per IMPLEMENTATION_GUIDE C.6).
                    # Note: compare node ids — tree-sitter `child_by_field_name` returns
                    # a fresh Python wrapper each call, so `is` is unreliable.
                    attr_name = parent.child_by_field_name("attribute")
                    if attr_name is not None and attr_name.id == node.id:
                        is_attribute_name = True
                # Keyword arguments: `f(x=1)` — `x` here is a parameter name on the
                # callee, not a use in our scope. tree-sitter wraps it in `keyword_argument`.
                is_kwarg_name = False
                if parent is not None and parent.type == "keyword_argument":
                    name_field = parent.child_by_field_name("name")
                    if name_field is not None and name_field.id == node.id:
                        is_kwarg_name = True
                if not is_attribute_name and not is_kwarg_name:
                    used.add(_decode(src, node))

            for c in node.children:
                walk(c)

        walk(tree.root_node)
        return used - defined_locally
