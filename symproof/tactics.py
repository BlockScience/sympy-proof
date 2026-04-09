"""Automatic tactics for proof writing.

Helpers that assist engineers during proof construction.  They do not
replace manual proof writing but handle routine verification steps.

**try_* functions** — return ``True``, ``False``, or ``None``
(inconclusive) for quick checks.

**auto_lemma** — tries all verification strategies and returns
a ``Lemma`` if any succeeds.

**signed_sum_lemmas** — the *signed accumulation* tactic: given N
signed terms, proves each term's sign and magnitude, then proves
the net sum has the required sign.  Domain-agnostic version of the
pattern used in DeFi rounding analysis, Lyapunov descent, and any
context where positive and negative contributions must net correctly.
"""

from __future__ import annotations

from dataclasses import dataclass

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

        # Fallback 1: refine() handles relational reasoning that
        # simplify() misses (e.g., transitivity of <=, strict => weak).
        # Skip for AppliedPredicate (Q-system) — those belong in QUERY.
        if not isinstance(expr, sympy.assumptions.assume.AppliedPredicate):
            try:
                refined = sympy.refine(expr)
                if refined is sympy.true:
                    return True
                if refined is sympy.false:
                    return False
            except (TypeError, ValueError, RecursionError, AttributeError):
                pass

        # Fallback 2: for Implies, proof by contradiction —
        # Implies(P, Q) holds iff And(P, Not(Q)) is unsatisfiable.
        if isinstance(expr, sympy.Implies):
            try:
                ante, cons = expr.args[0], expr.args[1]
                negated = sympy.simplify(sympy.And(ante, sympy.Not(cons)))
                if negated is sympy.false:
                    return True
            except (TypeError, ValueError, RecursionError, AttributeError):
                pass

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


# ---------------------------------------------------------------------------
# Signed accumulation tactic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignedTerm:
    """A single term in a signed accumulation proof.

    Represents a quantity with a declared sign (nonnegative or
    nonpositive) and an optional magnitude bound.

    Parameters
    ----------
    expr:
        The SymPy expression for this term's value.
    nonneg:
        ``True`` if this term should be >= 0, ``False`` if <= 0.
    bound:
        Optional upper bound on ``|expr|``.  If provided, a magnitude
        lemma ``|expr| < bound`` is generated.
    label:
        Human description for lemma names and descriptions.
    """

    expr: sympy.Basic
    nonneg: bool
    bound: sympy.Basic | None = None
    label: str = ""


def signed_sum_lemmas(
    terms: list[SignedTerm],
    *,
    net_nonneg: bool = True,
    name_prefix: str = "signed",
) -> list[Lemma]:
    """Prove sign, magnitude, and net direction for a list of signed terms.

    The *signed accumulation* tactic.  Given N terms where each is
    declared nonnegative or nonpositive:

    1. **Per-term sign**: ``term >= 0`` or ``term <= 0``
    2. **Per-term magnitude** (optional): ``|term| < bound``
    3. **Net direction**: ``sum(terms) >= 0`` (or ``<= 0``)

    The net-direction lemma is the payoff: it proves that the positive
    terms dominate the negative terms (or vice versa).

    Use cases:

    - **DeFi rounding**: protocol-favoring errors (positive) must
      outweigh user-favoring errors (negative)
    - **Lyapunov descent**: V-dot has positive damping terms and
      negative growth terms; net must be nonpositive
    - **Convex optimization**: gradient step has ascent and descent
      components; net must decrease the objective

    Parameters
    ----------
    terms:
        Ordered list of ``SignedTerm`` objects.
    net_nonneg:
        If ``True`` (default), the net-direction lemma proves
        ``sum >= 0``.  If ``False``, proves ``sum <= 0``.
    name_prefix:
        Prefix for generated lemma names.

    Returns
    -------
    list[Lemma]
        Per-term sign lemmas, optional magnitude lemmas, and the
        net-direction lemma.
    """
    lemmas: list[Lemma] = []

    for i, term in enumerate(terms):
        lbl = term.label or f"term_{i}"

        # Sign lemma
        if term.nonneg:
            lemmas.append(
                Lemma(
                    name=f"{name_prefix}_{i}_sign",
                    kind=LemmaKind.BOOLEAN,
                    expr=sympy.Ge(term.expr, sympy.Integer(0)),
                    depends_on=(
                        [f"{name_prefix}_{i-1}_sign"] if i > 0 else []
                    ),
                    description=f"{lbl}: >= 0",
                )
            )
        else:
            lemmas.append(
                Lemma(
                    name=f"{name_prefix}_{i}_sign",
                    kind=LemmaKind.BOOLEAN,
                    expr=sympy.Le(term.expr, sympy.Integer(0)),
                    depends_on=(
                        [f"{name_prefix}_{i-1}_sign"] if i > 0 else []
                    ),
                    description=f"{lbl}: <= 0",
                )
            )

        # Magnitude lemma (optional)
        if term.bound is not None:
            lemmas.append(
                Lemma(
                    name=f"{name_prefix}_{i}_bound",
                    kind=LemmaKind.BOOLEAN,
                    expr=sympy.Lt(sympy.Abs(term.expr), term.bound),
                    depends_on=[f"{name_prefix}_{i}_sign"],
                    description=f"{lbl}: |value| < {term.bound}",
                )
            )

    # Net direction
    total = sum(
        t.expr for t in terms
    )  # type: ignore[arg-type]
    net_expr = (
        sympy.Ge(total, sympy.Integer(0))
        if net_nonneg
        else sympy.Le(total, sympy.Integer(0))
    )
    direction = ">= 0" if net_nonneg else "<= 0"
    net_ok = sympy.simplify(net_expr) is sympy.true
    warning = "" if net_ok else (
        f" ⚠ NET {'NEGATIVE' if net_nonneg else 'POSITIVE'}"
        f" — sum violates required direction!"
    )
    lemmas.append(
        Lemma(
            name=f"{name_prefix}_net",
            kind=LemmaKind.BOOLEAN,
            expr=net_expr,
            depends_on=[
                f"{name_prefix}_{i}_sign" for i in range(len(terms))
            ],
            description=f"Net sum {direction}{warning}",
        )
    )

    return lemmas
