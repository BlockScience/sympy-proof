"""Core library proofs covering known SymPy simplification gaps.

These proofs decompose claims that SymPy's ``simplify()``, ``refine()``,
and ``ask()`` cannot handle into steps they can.
"""

from __future__ import annotations

import sympy

from symproof.builder import ProofBuilder
from symproof.bundle import seal
from symproof.models import AxiomSet, LemmaKind, ProofBundle


def max_ge_first(
    axiom_set: AxiomSet,
    a: sympy.Basic,
    b: sympy.Basic,
) -> ProofBundle:
    """Prove ``Max(a, b) >= a``.

    SymPy's simplify/refine/ask cannot verify ``Ge(Max(a,b), a)``
    symbolically.  We decompose via ``Max(0, b - a) >= 0``, which
    the Q-system handles (Max with 0 is always nonnegative).

    Parameters
    ----------
    axiom_set:
        Must contain axioms establishing ``a`` and ``b`` as real.
    a, b:
        Real-valued SymPy expressions (symbols or constants).
    """
    a_name = getattr(a, "name", str(a))
    b_name = getattr(b, "name", str(b))
    hyp = axiom_set.hypothesis(
        f"max_{a_name}_{b_name}_ge_{a_name}",
        expr=sympy.Ge(sympy.Max(a, b), a),
        description=f"Max({a_name}, {b_name}) >= {a_name}",
    )
    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name=f"max_ge_first_{a_name}_{b_name}",
            claim=f"Max({a_name}, {b_name}) >= {a_name}",
        )
        .lemma(
            "max_zero_nonneg",
            LemmaKind.QUERY,
            expr=sympy.Q.nonnegative(sympy.Max(0, b - a)),
            description="Max(0, b-a) >= 0 by definition of Max",
        )
        .build()
    )
    return seal(axiom_set, hyp, script)


def piecewise_collapse(
    axiom_set: AxiomSet,
    expr: sympy.Basic,
    condition: sympy.Basic,
    fallback: sympy.Basic,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove ``Piecewise((expr, condition), (fallback, True)) == expr``.

    SymPy's simplify does not reliably collapse Piecewise even when
    assumptions make only one branch active.  We use assumption
    substitution to make the condition decidable, allowing simplify
    to cancel the difference.

    Parameters
    ----------
    axiom_set:
        Axiom context.
    expr:
        The expression returned when ``condition`` is true.
    condition:
        A boolean expression (e.g., ``x > 0``).
    fallback:
        The expression returned when ``condition`` is false.
    assumptions:
        Symbol assumptions that make ``condition`` decidable
        (e.g., ``{"x": {"positive": True}}``).
    """
    pw = sympy.Piecewise((expr, condition), (fallback, True))
    asm = assumptions or {}

    hyp = axiom_set.hypothesis(
        "piecewise_collapse",
        expr=sympy.Eq(pw, expr),
        description="Piecewise collapses to active branch",
    )
    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="piecewise_collapse_proof",
            claim="Piecewise == expr when condition holds",
        )
        .lemma(
            "branch_collapse",
            LemmaKind.EQUALITY,
            expr=pw,
            expected=expr,
            assumptions=asm,
            description="Under assumptions, Piecewise collapses",
        )
        .build()
    )
    return seal(axiom_set, hyp, script)
