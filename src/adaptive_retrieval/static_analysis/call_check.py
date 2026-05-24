"""Tier 2: detect signature mismatches in the prediction's calls.

For each ``call`` node in the prediction, resolve the callee to a single
known :class:`Signature` and verify:

* Positional argument count is within ``[min_positional, max_positional]``.
* Every keyword-argument name appears in the callee's parameter list
  (unless the callee has ``**kwargs``).

The resolver is conservative — it abstains whenever the callee can't be
pinned to a unique signature:

* Method calls on instance receivers (``self.x`` / ``obj.x``) — receiver
  type inference is Tier 1 and not implemented here.
* Calls whose callee is the result of another call, a subscript, or a
  multi-step attribute chain.
* Calls with ``*expr`` / ``**expr`` argument splats (size is unknowable).
* Calls to names with multiple repo definitions that disagree on shape.
* Calls to callables flagged ``decorated_unsafe`` by signature extraction.
"""
from __future__ import annotations

from dataclasses import dataclass

from .modules import path_to_module
from .parser import parse
from .signatures import Signature
from .symbol_table import RepositorySymbolTable


@dataclass(frozen=True)
class CallIssue:
    """A single suspicious call site."""

    callee: str               # textual callee (e.g. "foo" or "ClassName.method")
    kind: str                 # "wrong_arity" | "unknown_kwarg"
    detail: str               # human-readable explanation
    expected: str             # "1-2 positional args" or "kwargs: {x, y}"
    actual: str               # what we saw


def _decode(src: bytes, node) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _module_bindings(
    in_file_source: str,
) -> dict[str, str]:
    """Collect ``local_name -> module`` bindings from ``import`` statements.

    Limited to whole-module bindings (``import X`` / ``import X as Y``) —
    ``from X import Y`` is handled separately because it introduces ``Y``
    as the name to resolve, not a module alias.
    """
    bindings: dict[str, str] = {}
    if not in_file_source.strip():
        return bindings
    try:
        tree = parse(in_file_source)
    except Exception:
        return bindings
    src = in_file_source.encode("utf-8")

    def walk(n) -> None:
        if n.type == "import_statement":
            for c in n.children:
                if c.type == "dotted_name":
                    text = _decode(src, c).strip()
                    head = text.split(".", 1)[0]
                    bindings[head] = text
                elif c.type == "aliased_import":
                    alias = c.child_by_field_name("alias")
                    name = c.child_by_field_name("name")
                    if alias is not None and name is not None:
                        bindings[_decode(src, alias).strip()] = _decode(src, name).strip()
        for c in n.children:
            walk(c)

    walk(tree.root_node)
    return bindings


def _from_import_bindings(in_file_source: str) -> dict[str, str]:
    """Collect ``from X import Y[ as Z]`` -> local-name -> module-of-origin.

    Used to know that ``Foo`` (local in this file) came from ``pkg.core``.
    """
    bindings: dict[str, str] = {}
    if not in_file_source.strip():
        return bindings
    try:
        tree = parse(in_file_source)
    except Exception:
        return bindings
    src = in_file_source.encode("utf-8")

    def walk(n) -> None:
        if n.type == "import_from_statement":
            module_node = n.child_by_field_name("module_name")
            module = _decode(src, module_node).strip() if module_node else ""
            module_id = module_node.id if module_node is not None else -1
            for c in n.children:
                if c.id == module_id:
                    continue
                if c.type == "dotted_name":
                    first = next(
                        (sub for sub in c.children if sub.type == "identifier"),
                        None,
                    )
                    if first is not None:
                        bindings[_decode(src, first)] = module
                elif c.type == "aliased_import":
                    alias = c.child_by_field_name("alias")
                    name = c.child_by_field_name("name")
                    local = _decode(src, alias) if alias is not None else (
                        _decode(src, name) if name is not None else ""
                    )
                    if local:
                        bindings[local] = module
        for c in n.children:
            walk(c)

    walk(tree.root_node)
    return bindings


def _unique_signature(sigs: list[Signature]) -> Signature | None:
    """Collapse a list of signatures to one, or return ``None`` if they
    disagree on shape (in which case we abstain)."""
    if not sigs:
        return None
    if len(sigs) == 1:
        return sigs[0]
    # Allow duplicates that match exactly; otherwise abstain.
    first = sigs[0]
    for s in sigs[1:]:
        if (
            s.positional != first.positional
            or s.n_defaults != first.n_defaults
            or s.has_star_args != first.has_star_args
            or s.has_star_kwargs != first.has_star_kwargs
            or s.keyword_only != first.keyword_only
        ):
            return None
    return first


def _resolve_callee(
    call_node,
    src: bytes,
    repo: RepositorySymbolTable,
    module_bindings: dict[str, str],
    from_bindings: dict[str, str],
) -> tuple[str, Signature, bool] | None:
    """Resolve ``call_node``'s callee to (text, signature, is_method_call).

    Returns ``None`` if the callee is ambiguous or unsupported — the caller
    must abstain in that case.
    """
    func = call_node.child_by_field_name("function")
    if func is None:
        return None

    if func.type == "identifier":
        name = _decode(src, func)
        # ``from X import name`` — look up name in module X.
        origin = from_bindings.get(name)
        if origin:
            for site in repo.sites(name):
                if site.module == origin and site.signature is not None:
                    if site.signature.decorated_unsafe:
                        return None
                    return name, site.signature, False
        # Bare identifier: any unique signature in the repo.
        sigs = repo.signatures(name)
        sig = _unique_signature(sigs)
        if sig is None or sig.decorated_unsafe:
            return None
        return name, sig, False

    if func.type == "attribute":
        obj = func.child_by_field_name("object")
        attr = func.child_by_field_name("attribute")
        if obj is None or attr is None or attr.type != "identifier":
            return None
        attr_name = _decode(src, attr)
        # ``ClassName.method(...)`` — obj is an identifier naming a class in
        # the repo.
        if obj.type == "identifier":
            obj_name = _decode(src, obj)
            # ClassName.method on a known repo class.
            methods = repo.methods_of(obj_name)
            if attr_name in methods:
                sig = methods[attr_name]
                if sig.decorated_unsafe:
                    return None
                return f"{obj_name}.{attr_name}", sig, True
            # mod.func — obj is an imported module alias.
            module = module_bindings.get(obj_name)
            if module is not None:
                for site in repo.sites(attr_name):
                    if site.module == module and site.signature is not None:
                        if site.signature.decorated_unsafe:
                            return None
                        return f"{obj_name}.{attr_name}", site.signature, False
        # self.x / obj.x / chained — abstain.
        return None
    # Subscript / call-as-callee / lambda / etc — abstain.
    return None


def _count_args(arg_node, src: bytes) -> tuple[int, list[str], bool, bool]:
    """Return (positional_count, kwarg_names, has_star, has_dstar).

    ``has_star`` / ``has_dstar`` indicate the call uses ``*expr`` / ``**expr``
    splats. When either is present the corresponding bound check is skipped.
    """
    positional = 0
    kwargs: list[str] = []
    has_star = False
    has_dstar = False
    for c in arg_node.children:
        t = c.type
        if t in ("(", ")", ","):
            continue
        if t == "keyword_argument":
            name = c.child_by_field_name("name")
            if name is not None:
                kwargs.append(_decode(src, name))
            continue
        if t == "list_splat":
            has_star = True
            continue
        if t == "dictionary_splat":
            has_dstar = True
            continue
        if t == "parenthesized_expression":
            positional += 1
            continue
        # Default: a normal positional expression.
        positional += 1
    return positional, kwargs, has_star, has_dstar


def check_calls(
    prediction_source: str,
    in_file_source: str,
    repo: RepositorySymbolTable,
    importing_file: str = "",
) -> list[CallIssue]:
    """Check every resolvable call in ``prediction_source`` for arity/kwarg fit.

    ``in_file_source`` supplies the import context (whole-module aliases,
    ``from X import Y`` bindings). Pass ``""`` if unavailable.
    """
    if not prediction_source.strip():
        return []
    try:
        tree = parse(prediction_source)
    except Exception:
        return []
    src = prediction_source.encode("utf-8")
    module_bindings = _module_bindings(in_file_source)
    from_bindings = _from_import_bindings(in_file_source)
    # Augment the importing-file module path into module_bindings if we
    # have it -- helps relative resolution where applicable.
    if importing_file:
        _ = path_to_module(importing_file)  # currently unused, reserved
    issues: list[CallIssue] = []

    def walk(n) -> None:
        if n.type == "call":
            arg_node = n.child_by_field_name("arguments")
            resolved = _resolve_callee(n, src, repo, module_bindings, from_bindings)
            if resolved is not None and arg_node is not None:
                callee_text, sig, is_method = resolved
                pos, kwargs, has_star, has_dstar = _count_args(arg_node, src)
                # If the call uses splats we can't bound arity, but can still
                # validate kwarg names when there's no **expr splat.
                if not has_star:
                    # Adjust for class-qualified method calls (``Cls.m(receiver, ...)``):
                    # the receiver is the first positional arg, which absorbs ``self``.
                    # ``sig.min_positional`` / ``max_positional`` already strip self,
                    # so we add 1 to both bounds when it's a class-qualified method.
                    lo = sig.min_positional + (1 if (is_method and sig.has_self) else 0)
                    hi = sig.max_positional
                    if hi is not None:
                        hi += (1 if (is_method and sig.has_self) else 0)
                    if pos < lo or (hi is not None and pos > hi):
                        expected = (
                            f"{lo} positional"
                            if hi == lo
                            else f"{lo}-{hi if hi is not None else 'unbounded'} positional"
                        )
                        issues.append(
                            CallIssue(
                                callee=callee_text,
                                kind="wrong_arity",
                                detail=f"call uses {pos} positional arg(s)",
                                expected=expected,
                                actual=str(pos),
                            )
                        )
                if not has_dstar and not sig.has_star_kwargs and kwargs:
                    allowed = sig.all_keyword_names
                    bad = [k for k in kwargs if k not in allowed]
                    if bad:
                        issues.append(
                            CallIssue(
                                callee=callee_text,
                                kind="unknown_kwarg",
                                detail=f"unknown kwarg(s): {bad}",
                                expected=f"kwargs in {sorted(allowed)}",
                                actual=str(bad),
                            )
                        )
        for c in n.children:
            walk(c)

    walk(tree.root_node)
    return issues
