"""Shared fixtures for symproof tests."""

from __future__ import annotations

import pytest
import sympy

from symproof.builder import ProofBuilder
from symproof.models import Axiom, AxiomSet, LemmaKind

# ---------------------------------------------------------------------------
# Common symbols
# ---------------------------------------------------------------------------

X = sympy.Symbol("x")
Y = sympy.Symbol("y")
K = sympy.Symbol("k", integer=True, nonneg=True)


# ---------------------------------------------------------------------------
# Axiom sets
# ---------------------------------------------------------------------------


@pytest.fixture
def positive_reals_axioms():
    """Axiom set declaring x > 0 and y > 0."""
    return AxiomSet(
        name="positive_reals",
        axioms=(
            Axiom(name="x_positive", expr=X > 0),
            Axiom(name="y_positive", expr=Y > 0),
        ),
    )


@pytest.fixture
def single_axiom_set():
    """Axiom set with one axiom: x > 0."""
    return AxiomSet(
        name="single",
        axioms=(Axiom(name="x_positive", expr=X > 0),),
    )


@pytest.fixture
def axiom_set_hash(positive_reals_axioms):
    return positive_reals_axioms.axiom_set_hash


# ---------------------------------------------------------------------------
# Hypotheses
# ---------------------------------------------------------------------------


@pytest.fixture
def product_positive_hypothesis(positive_reals_axioms):
    """Hypothesis: x * y > 0, bound to positive_reals axioms."""
    return positive_reals_axioms.hypothesis(
        "product_positive",
        expr=X * Y > 0,
        description="Product of positives is positive",
    )


# ---------------------------------------------------------------------------
# Proof scripts
# ---------------------------------------------------------------------------


@pytest.fixture
def product_proof_script(positive_reals_axioms, product_positive_hypothesis):
    """Proof script showing x*y > 0 via Q.positive query."""
    return (
        ProofBuilder(
            positive_reals_axioms,
            product_positive_hypothesis.name,
            name="product_positivity_proof",
            claim="x > 0 and y > 0 implies x*y > 0",
        )
        .lemma(
            "xy_positive",
            LemmaKind.QUERY,
            expr=sympy.Q.positive(X * Y),
            assumptions={"x": {"positive": True}, "y": {"positive": True}},
            description="Product of positives is positive",
        )
        .build()
    )


@pytest.fixture
def geometric_series_script(positive_reals_axioms):
    """Proof script with an EQUALITY lemma using Sum."""
    return (
        ProofBuilder(
            positive_reals_axioms,
            "series_limit",
            name="geometric_convergence",
            claim="Sum(1/2^k, k=0..inf) = 2",
        )
        .lemma(
            "geometric_series_limit",
            LemmaKind.EQUALITY,
            expr=sympy.Sum(sympy.Rational(1, 2) ** K, (K, 0, sympy.oo)),
            expected=sympy.Integer(2),
            description="Geometric series converges to 2",
        )
        .build()
    )
