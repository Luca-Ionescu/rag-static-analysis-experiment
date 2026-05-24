"""Function-signature extraction from tree-sitter function-definition nodes.

The :class:`Signature` dataclass is consumed by Tier 2 (``call_check``) to
verify positional arity and keyword-argument names against the resolved
callee. The extraction is intentionally narrow:

* Decorated callables that we don't trust to preserve the signature
  (e.g. ``@click.command``) are flagged via :attr:`Signature.decorated_unsafe`
  so :mod:`call_check` can abstain. Recognised "safe" decorators
  (``staticmethod``, ``classmethod``, ``property``, ``functools.cache``,
  ``functools.lru_cache``, ``functools.wraps``) leave the signature untouched.
* ``*args`` and ``**kwargs`` are recorded but their presence widens the
  signature: with ``*args`` we can't bound positional arity above, and with
  ``**kwargs`` we accept any keyword name.
"""
from __future__ import annotations

from dataclasses import dataclass, field

_SAFE_DECORATORS = {
    "staticmethod",
    "classmethod",
    "property",
    "abstractmethod",
    "abstractproperty",
    "cache",
    "lru_cache",
    "wraps",
    "cached_property",
    # Common dotted forms — we match by trailing segment only.
    "functools.cache",
    "functools.lru_cache",
    "functools.wraps",
    "functools.cached_property",
    "abc.abstractmethod",
    "typing.overload",
    "overload",
}


@dataclass(frozen=True)
class Signature:
    """Parameter description of a Python callable.

    ``positional`` includes ``self`` / ``cls`` when present so callers can
    decide whether to account for it. ``n_defaults`` is the count of
    trailing positional parameters with default values, mirroring the
    Python convention that defaults bind from the right.
    """

    positional: tuple[str, ...] = ()
    has_self: bool = False
    n_defaults: int = 0
    has_star_args: bool = False
    has_star_kwargs: bool = False
    keyword_only: tuple[str, ...] = ()
    decorated_unsafe: bool = False
    # Reserved for future use (e.g. an "overload" cluster). Empty for now.
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def min_positional(self) -> int:
        """Lower bound on positional arg count callers must supply.

        Excludes ``self``/``cls`` since methods are called as ``obj.m(...)``
        with the implicit receiver. Callers without receiver context should
        add 1 back for class-qualified ``Cls.m(...)`` calls.
        """
        base = len(self.positional) - self.n_defaults
        return max(0, base - (1 if self.has_self else 0))

    @property
    def max_positional(self) -> int | None:
        """Upper bound on positional arg count, or ``None`` for ``*args``."""
        if self.has_star_args:
            return None
        return len(self.positional) - (1 if self.has_self else 0)

    @property
    def all_keyword_names(self) -> frozenset[str]:
        skip = 1 if self.has_self else 0
        return frozenset(self.positional[skip:]) | frozenset(self.keyword_only)


def _decode(src: bytes, node) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _decorator_name(dec_node, src: bytes) -> str:
    """Best-effort name extraction from a ``decorator`` tree-sitter node.

    Returns the textual decorator expression with leading ``@`` stripped and
    any call arguments removed. ``@functools.lru_cache(maxsize=None)`` ->
    ``"functools.lru_cache"``.
    """
    text = _decode(src, dec_node).strip()
    if text.startswith("@"):
        text = text[1:]
    if "(" in text:
        text = text.split("(", 1)[0]
    return text.strip()


def _is_safe_decorator(dec_text: str) -> bool:
    if dec_text in _SAFE_DECORATORS:
        return True
    # Match by trailing dotted segment for things like ``foo.functools.wraps``.
    tail = dec_text.rsplit(".", 1)[-1]
    return tail in _SAFE_DECORATORS


def extract_signature(func_node, src: bytes, is_method: bool = False) -> Signature:
    """Build a :class:`Signature` from a ``function_definition`` node.

    ``is_method`` indicates the function lives inside a ``class_definition``
    body. We trust the first parameter to be ``self`` / ``cls`` when it
    exists and the def isn't decorated with ``@staticmethod``.
    """
    positional: list[str] = []
    keyword_only: list[str] = []
    n_defaults = 0
    has_star_args = False
    has_star_kwargs = False
    decorated_unsafe = False
    seen_star = False  # everything after a bare ``*`` or ``*args`` is keyword-only

    # ---- decorators ----
    is_staticmethod = False
    parent = func_node.parent
    if parent is not None and parent.type == "decorated_definition":
        for child in parent.children:
            if child.type != "decorator":
                continue
            name = _decorator_name(child, src)
            if name in ("staticmethod", "abc.staticmethod"):
                is_staticmethod = True
            if not _is_safe_decorator(name):
                decorated_unsafe = True

    params = func_node.child_by_field_name("parameters")
    if params is None:
        return Signature(decorated_unsafe=decorated_unsafe)

    for child in params.children:
        t = child.type
        if t in ("(", ")", ","):
            continue
        if t == "identifier":
            name = _decode(src, child)
            if seen_star:
                keyword_only.append(name)
            else:
                positional.append(name)
        elif t in ("default_parameter", "typed_default_parameter"):
            name_node = child.child_by_field_name("name")
            if name_node is None:
                for c in child.children:
                    if c.type == "identifier":
                        name_node = c
                        break
            if name_node is None:
                continue
            name = _decode(src, name_node)
            if seen_star:
                keyword_only.append(name)
            else:
                positional.append(name)
                n_defaults += 1
        elif t == "typed_parameter":
            for c in child.children:
                if c.type == "identifier":
                    name = _decode(src, c)
                    if seen_star:
                        keyword_only.append(name)
                    else:
                        positional.append(name)
                    break
        elif t == "list_splat_pattern":
            has_star_args = True
            seen_star = True
        elif t == "dictionary_splat_pattern":
            has_star_kwargs = True
        elif t == "keyword_separator":
            # Bare ``*`` — everything after is keyword-only.
            seen_star = True
        elif t in (
            "typed_parameter_with_default",  # unlikely but be safe
        ):
            for c in child.children:
                if c.type == "identifier":
                    name = _decode(src, c)
                    if seen_star:
                        keyword_only.append(name)
                    else:
                        positional.append(name)
                        n_defaults += 1
                    break

    has_self = (
        is_method
        and not is_staticmethod
        and bool(positional)
        and positional[0] in ("self", "cls")
    )
    return Signature(
        positional=tuple(positional),
        has_self=has_self,
        n_defaults=n_defaults,
        has_star_args=has_star_args,
        has_star_kwargs=has_star_kwargs,
        keyword_only=tuple(keyword_only),
        decorated_unsafe=decorated_unsafe,
    )
