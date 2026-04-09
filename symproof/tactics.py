"""Basic automatic tactics for proof writing.

Helpers that assist engineers during proof construction.  They do not
replace manual proof writing but handle routine verification steps.

Each ``try_*`` function returns ``True`` if the claim holds, ``False``
if it is refuted, or ``None`` if SymPy cannot determine the result.
"""

from __future__ import annotations

import sympy

from symproof.models import Lemma, LemmaKind
from symproof.verification import _build_assumption_subs, _build_q_context


def try_simplify(
    expr: sympy.Basic,
    assumptions: dict[str, dict] | None = None,
) -> bool | None:
    """Attempt to simplify ``expr`` to ``True`` or ``False``.

    Returns ``True`` if ``simplify(expr) is sympy.true``, ``False`` if
    ``simplify(expr) is sympy.false``, or ``None`` if inconclusive.
    """
    try:
        if assumptions:
            subs = _build_assumption_subs(assumptions)
            expr = expr.subs(subs)
        result = sympy.simplify(expr)
        if result is sympy.true:
            return True
        if result is sympy.false:
            return False
        return None
    except (TypeError, ValueError, RecursionError, AttributeError):
        return None


def try_implication(
    antecedent: sympy.Basic,
    consequent: sympy.Basic,
    assumptions: dict[str, dict] | None = None,
) -> bool | None:
    """Check ``Implies(antecedent, consequent)`` via simplification.

    Returns ``True`` if the implication holds, ``False`` if refuted,
    or ``None`` if inconclusive.
    """
    impl = sympy.Implies(antecedent, consequent)
    return try_simplify(impl, assumptions)


def try_query(
    proposition: sympy.Basic,
    assumptions: dict[str, dict] | None = None,
) -> bool | None:
    """Check a proposition via ``sympy.ask()``.

    Returns ``True`` if the proposition holds under the given assumptions,
    ``False`` if refuted, or ``None`` if inconclusive.
    """
    try:
        if assumptions:
            subs = _build_assumption_subs(assumptions)
            proposition = proposition.subs(subs)
            context = _build_q_context(assumptions)
        else:
            context = sympy.Q.is_true(sympy.true)  # type: ignore[attr-defined]
        result = sympy.ask(proposition, context)
        if result is True:
            return True
        if result is False:
            return False
        return None
    except (TypeError, ValueError, RecursionError, AttributeError):
        return None


def auto_lemma(
    name: str,
    expr: sympy.Basic,
    assumptions: dict[str, dict] | None = None,
    expected: sympy.Basic | None = None,
    description: str = "",
) -> Lemma | None:
    """Try all strategies and return a ``Lemma`` if any succeeds.

    Attempts in order:
    1. EQUALITY (if ``expected`` is provided)
    2. BOOLEAN (simplify to True)
    3. QUERY (sympy.ask)

    Returns ``None`` if no strategy can verify the claim.
    """
    asm = assumptions or {}

    # Try EQUALITY if expected is provided
    if expected is not None:
        try:
            subs = _build_assumption_subs(asm) if asm else {}
            expr_sub = expr.subs(subs) if subs else expr
            expected_sub = expected.subs(subs) if subs else expected
            diff = sympy.simplify(expr_sub - expected_sub)
            if diff == sympy.Integer(0):
                return Lemma(
                    name=name,
                    kind=LemmaKind.EQUALITY,
                    expr=expr,
                    expected=expected,
                    assumptions=asm,
                    description=description,
                )
            evaluated = expr_sub.doit()
            if sympy.simplify(evaluated - expected_sub) == sympy.Integer(0):
                return Lemma(
                    name=name,
                    kind=LemmaKind.EQUALITY,
                    expr=expr,
                    expected=expected,
                    assumptions=asm,
                    description=description,
                )
        except (TypeError, ValueError, RecursionError, AttributeError):
            pass

    # Try BOOLEAN
    result = try_simplify(expr, assumptions)
    if result is True:
        return Lemma(
            name=name,
            kind=LemmaKind.BOOLEAN,
            expr=expr,
            assumptions=asm,
            description=description,
        )

    # Try QUERY
    result = try_query(expr, assumptions)
    if result is True:
        return Lemma(
            name=name,
            kind=LemmaKind.QUERY,
            expr=expr,
            assumptions=asm,
            description=description,
        )

    return None
