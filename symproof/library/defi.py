"""DeFi mechanism proofs — AMM fees, product invariants.

Proofs in this module address SymPy's inability to reason about
bounded-interval constraints (e.g., ``0 < f < 1 → 1-f > 0``) by
decomposing compound expressions into steps the Q-system can verify.
"""

from __future__ import annotations

import sympy

from symproof.builder import ProofBuilder
from symproof.bundle import seal
from symproof.models import AxiomSet, LemmaKind, ProofBundle


def fee_complement_positive(
    axiom_set: AxiomSet,
    fee: sympy.Symbol,
) -> ProofBundle:
    """Prove ``1 - fee > 0`` from axioms ``fee > 0`` and ``fee < 1``.

    SymPy's Q-system cannot propagate the bounded-interval constraint
    ``0 < f < 1`` through rational expressions.  This foundational
    proof establishes the positivity of the fee complement, which
    both AMM proofs depend on.

    Parameters
    ----------
    axiom_set:
        Must contain axioms establishing ``fee > 0`` and ``fee < 1``.
    fee:
        The fee rate symbol.
    """
    hyp = axiom_set.hypothesis(
        "fee_complement_positive",
        expr=1 - fee > 0,
        description="1 - fee > 0 (fee complement is positive)",
    )
    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="fee_complement_proof",
            claim="0 < fee < 1 implies 1 - fee > 0",
        )
        .lemma(
            "complement_from_bound",
            LemmaKind.BOOLEAN,
            expr=sympy.Implies(
                sympy.And(fee > 0, fee < 1),
                1 - fee > 0,
            ),
            description="Direct implication from fee < 1",
        )
        .build()
    )
    return seal(axiom_set, hyp, script)


def amm_output_positive(
    axiom_set: AxiomSet,
    Rx: sympy.Symbol,
    Ry: sympy.Symbol,
    fee: sympy.Symbol,
    dx: sympy.Symbol,
) -> ProofBundle:
    """Prove AMM swap output ``dy > 0`` for a fee-bearing pool.

    The output formula is ``dy = Ry * dx * (1-f) / (Rx + dx*(1-f))``.
    SymPy cannot verify positivity directly because it can't reason
    about ``1-f > 0`` from ``0 < f < 1``.

    Decomposition: introduce helper symbol ``g = 1-f`` (positive),
    rewrite ``dy`` in terms of ``g``, then verify positivity with
    all-positive symbols.

    Parameters
    ----------
    axiom_set:
        Must contain axioms for Rx, Ry, dx > 0 and fee in (0, 1).
    Rx, Ry:
        Reserve symbols (positive).
    fee:
        Fee rate symbol (in (0, 1)).
    dx:
        Input amount symbol (positive).
    """
    fee_bundle = fee_complement_positive(axiom_set, fee)

    g = sympy.Symbol("g", positive=True)
    dy_original = Ry * dx * (1 - fee) / (Rx + dx * (1 - fee))
    dy_with_g = Ry * dx * g / (Rx + dx * g)

    hyp = axiom_set.hypothesis(
        "amm_output_positive",
        expr=dy_original > 0,
        description="AMM swap output is positive",
    )
    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="amm_output_positive_proof",
            claim="dy > 0 for fee-bearing AMM swap",
        )
        .import_bundle(fee_bundle)
        .lemma(
            "rewrite_with_g",
            LemmaKind.EQUALITY,
            expr=dy_original,
            expected=dy_with_g.subs(g, 1 - fee),
            assumptions={
                Rx.name: {"positive": True},
                Ry.name: {"positive": True},
                dx.name: {"positive": True},
            },
            description="dy in original form equals dy(g=1-f)",
        )
        .lemma(
            "dy_g_positive",
            LemmaKind.QUERY,
            expr=sympy.Q.positive(dy_with_g),
            assumptions={
                Rx.name: {"positive": True},
                Ry.name: {"positive": True},
                dx.name: {"positive": True},
                "g": {"positive": True},
            },
            depends_on=["rewrite_with_g"],
            description="Ry*dx*g/(Rx+dx*g) > 0 with all positive",
        )
        .build()
    )
    return seal(axiom_set, hyp, script)


def amm_product_nondecreasing(
    axiom_set: AxiomSet,
    Rx: sympy.Symbol,
    Ry: sympy.Symbol,
    fee: sympy.Symbol,
    dx: sympy.Symbol,
) -> ProofBundle:
    """Prove AMM product invariant is nondecreasing after fee swap.

    The product ratio ``(Rx+dx)*(Ry-dy) / (Rx*Ry)`` simplifies to
    ``(Rx+dx) / (Rx+dx*(1-f))``, which exceeds 1 because fees make
    the numerator larger than the denominator.

    Decomposition: rewrite ratio excess as ``dx*f/(Rx+dx*g)`` where
    ``g = 1-f``, then verify positivity with all-positive symbols.

    Parameters
    ----------
    axiom_set:
        Must contain axioms for Rx, Ry, dx > 0 and fee in (0, 1).
    Rx, Ry:
        Reserve symbols (positive).
    fee:
        Fee rate symbol (in (0, 1)).
    dx:
        Input amount symbol (positive).
    """
    fee_bundle = fee_complement_positive(axiom_set, fee)

    g = sympy.Symbol("g", positive=True)

    # Ratio - 1 in original form
    ratio_original = (
        (Rx + dx) / (Rx + dx * (1 - fee)) - 1
    )
    # Same thing with g = 1-f
    excess_with_g = dx * fee / (Rx + dx * g)

    hyp = axiom_set.hypothesis(
        "amm_product_nondecreasing",
        expr=sympy.Ge(
            (Rx + dx)
            * (Ry - Ry * dx * (1 - fee) / (Rx + dx * (1 - fee))),
            Rx * Ry,
        ),
        description="Post-swap product >= pre-swap product",
    )
    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="amm_product_nondecreasing_proof",
            claim="Fee-bearing swap grows the invariant product",
        )
        .import_bundle(fee_bundle)
        .lemma(
            "ratio_excess_form",
            LemmaKind.EQUALITY,
            expr=ratio_original,
            expected=excess_with_g.subs(g, 1 - fee),
            assumptions={
                Rx.name: {"positive": True},
                dx.name: {"positive": True},
                fee.name: {"positive": True},
            },
            description="Ratio - 1 = dx*f/(Rx+dx*(1-f))",
        )
        .lemma(
            "excess_positive",
            LemmaKind.QUERY,
            expr=sympy.Q.positive(excess_with_g),
            assumptions={
                Rx.name: {"positive": True},
                dx.name: {"positive": True},
                fee.name: {"positive": True},
                "g": {"positive": True},
            },
            depends_on=["ratio_excess_form"],
            description="dx*f/(Rx+dx*g) > 0 with all positive",
        )
        .build()
    )
    return seal(axiom_set, hyp, script)
