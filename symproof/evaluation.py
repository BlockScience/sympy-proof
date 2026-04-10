"""Explicit evaluation control for symproof.

SymPy eagerly evaluates expressions at construction time — for example,
``Symbol("x", positive=True) > 0`` becomes ``True`` immediately, losing
the structural information that the expression was a positivity constraint.

symproof inverts this default: expressions are constructed **unevaluated**
(preserving structure), and evaluation happens only at explicit gates
where ``simplify()``, ``ask()``, or ``refine()`` are called.

Usage::

    from symproof.evaluation import unevaluated, evaluation

    # Construction: preserve structure
    with unevaluated():
        expr = x > 0  # Stays StrictGreaterThan(x, 0)

    # Verification: evaluate when needed
    with evaluation():
        result = sympy.simplify(expr)  # Now evaluates to True

The ``unevaluated()`` context is applied automatically by ``AxiomSet``
construction.  Library and proof authors should use ``evaluation()``
around any ``simplify()``, ``ask()``, or ``refine()`` call.
"""

from __future__ import annotations

from contextlib import contextmanager

from sympy import evaluate as _sympy_evaluate


@contextmanager
def unevaluated():
    """Suppress SymPy's eager evaluation.

    Expressions constructed inside this context preserve their structure:
    ``Symbol("x", positive=True) > 0`` stays as ``StrictGreaterThan``
    instead of collapsing to ``True``.
    """
    with _sympy_evaluate(False):
        yield


@contextmanager
def evaluation():
    """Explicit evaluation gate.

    Use this around ``simplify()``, ``ask()``, ``refine()``, and any
    other SymPy function that needs full evaluation.  This makes every
    evaluation point visible and auditable.
    """
    with _sympy_evaluate(True):
        yield
