"""Canonical serialization for symproof models and proof scripts.

All SymPy expressions are serialized via ``sympy.srepr()``, which produces
a deterministic, round-trippable string representation.

``make_canonical_dict`` is the single normalization entry point: it
recursively sorts dict keys and srepr's all SymPy expressions.
"""

from __future__ import annotations

from typing import Any

import sympy


def make_canonical_dict(components: Any) -> Any:
    """Recursively sort dict keys and srepr all SymPy expressions.

    Guarantees
    ----------
    Two calls with logically identical components produce identical output
    regardless of insertion order.  This guarantee is the foundation of
    hash stability.
    """

    def _normalize(obj: Any) -> Any:
        if isinstance(obj, sympy.Basic):
            return sympy.srepr(obj)
        if isinstance(obj, dict):
            return {k: _normalize(v) for k, v in sorted(obj.items())}
        if isinstance(obj, (list, tuple)):
            return [_normalize(v) for v in obj]
        return obj

    return _normalize(components)


def canonical_srepr(expr: sympy.Basic) -> str:
    """Deterministic canonical string form of a SymPy expression.

    Uses ``sympy.srepr``, which is fully explicit and round-trippable.
    Prefer this over ``str(expr)`` whenever persistence or hashing is
    involved.
    """
    return sympy.srepr(expr)


def restore_expr(s: str) -> sympy.Basic:
    """Restore a SymPy expression from its ``canonical_srepr`` form."""
    return sympy.sympify(s)  # type: ignore[return-value]
