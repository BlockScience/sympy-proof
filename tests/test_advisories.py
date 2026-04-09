"""Tests for the advisory warning system.

Verifies that proof results carry appropriate advisories when
verification passes through known SymPy limitations, and that
indeterminate results are clearly flagged.
"""

from __future__ import annotations

import sympy

from symproof import (
    Axiom,
    AxiomSet,
    LemmaKind,
    ProofBuilder,
    seal,
    verify_proof,
)
from symproof.models import Lemma
from symproof.verification import verify_lemma


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _axioms():
    """Minimal axiom set for testing."""
    x = sympy.Symbol("x")
    return AxiomSet(
        name="test", axioms=(Axiom(name="trivial", expr=sympy.Eq(1, 1)),)
    )


# ===================================================================
# EQUALITY advisories
# ===================================================================


class TestEqualityAdvisories:
    """EQUALITY path should flag domain-sensitive and fallback verifications."""

    def test_division_gets_advisory(self):
        """Expression with division should carry domain advisory."""
        x = sympy.Symbol("x")
        lemma = Lemma(
            name="rational_cancel",
            kind=LemmaKind.EQUALITY,
            expr=(x**2 - 1) / (x - 1),
            expected=x + 1,
        )
        result = verify_lemma(lemma)
        assert result.passed
        assert any("domain" in a.lower() for a in result.advisories), (
            f"Expected domain advisory, got: {result.advisories}"
        )

    def test_log_with_assumptions_gets_advisory(self):
        """log(x*y) == log(x)+log(y) with positive assumptions flags domain."""
        x = sympy.Symbol("x")
        y = sympy.Symbol("y")
        lemma = Lemma(
            name="log_product",
            kind=LemmaKind.EQUALITY,
            expr=sympy.log(x * y),
            expected=sympy.log(x) + sympy.log(y),
            assumptions={"x": {"positive": True}, "y": {"positive": True}},
        )
        result = verify_lemma(lemma)
        assert result.passed
        assert any("domain" in a.lower() for a in result.advisories)

    def test_sqrt_with_assumptions_gets_advisory(self):
        """sqrt(x^2) == x with positive assumption flags domain."""
        x = sympy.Symbol("x")
        lemma = Lemma(
            name="sqrt_sq",
            kind=LemmaKind.EQUALITY,
            expr=sympy.sqrt(x**2),
            expected=x,
            assumptions={"x": {"positive": True}},
        )
        result = verify_lemma(lemma)
        assert result.passed
        assert any("domain" in a.lower() for a in result.advisories)

    def test_doit_advisory_content(self):
        """The .doit() advisory text should mention symbolic evaluation."""
        from symproof.verification import _DOIT_ADVISORY

        assert "doit" in _DOIT_ADVISORY.lower()
        assert "sum" in _DOIT_ADVISORY.lower() or "product" in _DOIT_ADVISORY.lower()

    def test_simple_equality_no_advisory(self):
        """Plain polynomial equality should carry no advisories."""
        x = sympy.Symbol("x")
        lemma = Lemma(
            name="expand",
            kind=LemmaKind.EQUALITY,
            expr=(x + 1) ** 2,
            expected=x**2 + 2 * x + 1,
        )
        result = verify_lemma(lemma)
        assert result.passed
        assert result.advisories == ()

    def test_trig_identity_no_advisory(self):
        """sin^2 + cos^2 = 1 has no domain issues."""
        x = sympy.Symbol("x")
        lemma = Lemma(
            name="pythag",
            kind=LemmaKind.EQUALITY,
            expr=sympy.sin(x) ** 2 + sympy.cos(x) ** 2,
            expected=sympy.Integer(1),
        )
        result = verify_lemma(lemma)
        assert result.passed
        assert result.advisories == ()


# ===================================================================
# BOOLEAN advisories
# ===================================================================


class TestBooleanAdvisories:
    """BOOLEAN path should flag refine/negation fallbacks and indeterminate."""

    def test_refine_fallback_gets_advisory(self):
        """Implies verified via refine() should carry advisory."""
        r = sympy.Symbol("released", nonnegative=True)
        v = sympy.Symbol("vested", nonnegative=True)
        T = sympy.Symbol("total_alloc", positive=True)

        lemma = Lemma(
            name="transitivity",
            kind=LemmaKind.BOOLEAN,
            expr=sympy.Implies(
                sympy.And(sympy.Le(r, v), sympy.Le(v, T)),
                sympy.Le(r, T),
            ),
        )
        result = verify_lemma(lemma)
        assert result.passed
        # Should be verified via refine or negation, not bare simplify
        assert len(result.advisories) > 0, (
            "Expected refine/negation advisory for implication"
        )

    def test_negation_fallback_gets_advisory(self):
        """Implies verified via proof-by-contradiction should carry advisory."""
        x = sympy.Symbol("x", real=True)
        lemma = Lemma(
            name="strict_weak",
            kind=LemmaKind.BOOLEAN,
            expr=sympy.Implies(x > 0, x >= 0),
        )
        result = verify_lemma(lemma)
        assert result.passed
        assert len(result.advisories) > 0

    def test_indeterminate_boolean_gets_advisory(self):
        """Indeterminate BOOLEAN should carry INDETERMINATE advisory."""
        x = sympy.Symbol("x")
        lemma = Lemma(
            name="unknown",
            kind=LemmaKind.BOOLEAN,
            expr=x > 0,  # No assumptions — indeterminate
        )
        result = verify_lemma(lemma)
        assert not result.passed
        assert any("INDETERMINATE" in a for a in result.advisories), (
            f"Expected INDETERMINATE advisory, got: {result.advisories}"
        )

    def test_simple_true_boolean_no_advisory(self):
        """Eq(1,1) via direct simplify should have no advisory."""
        lemma = Lemma(
            name="trivial",
            kind=LemmaKind.BOOLEAN,
            expr=sympy.Eq(sympy.Integer(1), sympy.Integer(1)),
        )
        result = verify_lemma(lemma)
        assert result.passed
        assert result.advisories == ()


# ===================================================================
# QUERY advisories
# ===================================================================


class TestQueryAdvisories:
    """QUERY path should always carry Q-system advisory on pass."""

    def test_passing_query_gets_advisory(self):
        """Every passing QUERY should note Q-system heuristics."""
        x = sympy.Symbol("x")
        lemma = Lemma(
            name="pos",
            kind=LemmaKind.QUERY,
            expr=sympy.Q.positive(x),
            assumptions={"x": {"positive": True}},
        )
        result = verify_lemma(lemma)
        assert result.passed
        assert any("ask" in a.lower() for a in result.advisories)

    def test_indeterminate_query_gets_advisory(self):
        """Indeterminate QUERY should carry INDETERMINATE advisory."""
        x = sympy.Symbol("x")
        lemma = Lemma(
            name="unknown",
            kind=LemmaKind.QUERY,
            expr=sympy.Q.positive(x),
            assumptions={},
        )
        result = verify_lemma(lemma)
        assert not result.passed
        assert any("INDETERMINATE" in a for a in result.advisories), (
            f"Expected INDETERMINATE advisory, got: {result.advisories}"
        )

    def test_false_query_no_indeterminate(self):
        """Definitively false QUERY should NOT carry INDETERMINATE."""
        x = sympy.Symbol("x")
        lemma = Lemma(
            name="neg_with_pos",
            kind=LemmaKind.QUERY,
            expr=sympy.Q.negative(x),
            assumptions={"x": {"positive": True}},
        )
        result = verify_lemma(lemma)
        assert not result.passed
        assert not any("INDETERMINATE" in a for a in result.advisories)


# ===================================================================
# ProofResult advisory aggregation
# ===================================================================


class TestProofResultAdvisories:
    """ProofResult should aggregate advisories from all lemma results."""

    def test_advisories_aggregated_with_lemma_names(self):
        """ProofResult.advisories should prefix each with lemma name."""
        k = sympy.Symbol("k", integer=True, nonnegative=True)
        x = sympy.Symbol("x")
        axioms = _axioms()

        script = (
            ProofBuilder(axioms, "target", name="test", claim="test")
            .lemma(
                "series_step",
                LemmaKind.EQUALITY,
                expr=sympy.Sum(sympy.Rational(1, 2) ** k, (k, 0, sympy.oo)),
                expected=sympy.Integer(2),
            )
            .lemma(
                "query_step",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(x),
                assumptions={"x": {"positive": True}},
                depends_on=["series_step"],
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED"
        assert len(result.advisories) > 0

        # Each advisory should be prefixed with [lemma_name]
        for adv in result.advisories:
            assert adv.startswith("["), f"Advisory missing lemma prefix: {adv}"

    def test_clean_proof_no_advisories(self):
        """A proof using only plain simplify should have no advisories."""
        x = sympy.Symbol("x")
        axioms = _axioms()

        script = (
            ProofBuilder(axioms, "target", name="test", claim="test")
            .lemma(
                "expand",
                LemmaKind.EQUALITY,
                expr=(x + 1) ** 2,
                expected=x**2 + 2 * x + 1,
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED"
        assert result.advisories == ()

    def test_sealed_bundle_carries_advisories(self):
        """Sealed ProofBundle should carry advisories through."""
        x = sympy.Symbol("x")
        axioms = AxiomSet(
            name="pos", axioms=(Axiom(name="x_pos", expr=x > 0),),
        )
        h = axioms.hypothesis("q", expr=x > 0)
        script = (
            ProofBuilder(axioms, h.name, name="test", claim="x > 0")
            .lemma(
                "pos",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(x),
                assumptions={"x": {"positive": True}},
            )
            .build()
        )
        bundle = seal(axioms, h, script)
        assert len(bundle.proof_result.advisories) > 0
