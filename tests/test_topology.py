"""Tests for the point-set topology library."""

from __future__ import annotations

import pytest
import sympy

from symproof import AxiomSet, ProofStatus
from symproof.library.topology import (
    continuous_at_point,
    extreme_value,
    intermediate_value,
    verify_boundary,
    verify_closed,
    verify_compact,
    verify_open,
)

_EMPTY = AxiomSet(name="test", axioms=())
x = sympy.Symbol("x")


class TestVerifyOpen:
    def test_open_interval(self):
        bundle = verify_open(_EMPTY, sympy.Interval.open(0, 1))
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_reals_are_open(self):
        bundle = verify_open(_EMPTY, sympy.S.Reals)
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_closed_interval_rejected(self):
        with pytest.raises(ValueError, match="not open"):
            verify_open(_EMPTY, sympy.Interval(0, 1))


class TestVerifyClosed:
    def test_closed_interval(self):
        bundle = verify_closed(_EMPTY, sympy.Interval(0, 1))
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_open_interval_rejected(self):
        with pytest.raises(ValueError, match="not closed"):
            verify_closed(_EMPTY, sympy.Interval.open(0, 1))


class TestVerifyCompact:
    def test_closed_bounded_interval(self):
        bundle = verify_compact(_EMPTY, sympy.Interval(0, 1))
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_open_interval_rejected(self):
        with pytest.raises(ValueError, match="not closed"):
            verify_compact(_EMPTY, sympy.Interval.open(0, 1))

    def test_unbounded_rejected(self):
        with pytest.raises(ValueError, match="not bounded"):
            verify_compact(_EMPTY, sympy.Interval(0, sympy.oo))

    def test_heine_borel_three_lemmas(self):
        bundle = verify_compact(_EMPTY, sympy.Interval(-5, 5))
        assert len(bundle.proof.lemmas) == 3  # closed, bounded, HB


class TestVerifyBoundary:
    def test_closed_interval_boundary(self):
        bundle = verify_boundary(
            _EMPTY, sympy.Interval(0, 1), sympy.FiniteSet(0, 1),
        )
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_wrong_boundary_rejected(self):
        with pytest.raises(ValueError, match="not"):
            verify_boundary(
                _EMPTY, sympy.Interval(0, 1), sympy.FiniteSet(0),
            )


class TestContinuousAtPoint:
    def test_polynomial_continuous(self):
        bundle = continuous_at_point(_EMPTY, x**2, x, 3)
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_sin_continuous(self):
        bundle = continuous_at_point(_EMPTY, sympy.sin(x), x, 0)
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_discontinuous_rejected(self):
        # 1/x at x=0 has no limit
        with pytest.raises((ValueError, Exception)):
            continuous_at_point(_EMPTY, 1 / x, x, 0)


class TestIntermediateValue:
    def test_sqrt2_via_ivt(self):
        bundle = intermediate_value(_EMPTY, x**2 - 2, x, 1, 2, 0)
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_three_lemmas(self):
        bundle = intermediate_value(_EMPTY, x**2 - 2, x, 1, 2, 0)
        assert len(bundle.proof.lemmas) == 3  # sign change, root, in interval

    def test_no_root_rejected(self):
        # x^2 + 1 > 0 everywhere, no root in (0, 1)
        with pytest.raises(ValueError, match="No root"):
            intermediate_value(_EMPTY, x**2 + 1, x, 0, 1, 0)


class TestInferenceLemmaKind:
    """Test the INFERENCE verification strategy."""

    def test_inference_with_rule_and_deps(self):
        from symproof.models import Lemma, LemmaKind
        from symproof.verification import verify_lemma

        lem = Lemma(
            name="conclusion",
            kind=LemmaKind.INFERENCE,
            expr=sympy.S.true,
            depends_on=["premise_a", "premise_b"],
            rule="Modus ponens",
            description="A and A->B therefore B",
        )
        result = verify_lemma(lem)
        assert result.passed

    def test_inference_without_deps_fails(self):
        from symproof.models import Lemma, LemmaKind
        from symproof.verification import verify_lemma

        lem = Lemma(
            name="bad",
            kind=LemmaKind.INFERENCE,
            expr=sympy.S.true,
            depends_on=[],
            rule="Some rule",
        )
        result = verify_lemma(lem)
        assert not result.passed

    def test_inference_without_rule_fails(self):
        from symproof.models import Lemma, LemmaKind
        from symproof.verification import verify_lemma

        lem = Lemma(
            name="bad",
            kind=LemmaKind.INFERENCE,
            expr=sympy.S.true,
            depends_on=["something"],
            rule="",
        )
        result = verify_lemma(lem)
        assert not result.passed

    def test_heine_borel_uses_inference(self):
        """verify_compact produces an INFERENCE lemma for Heine-Borel."""
        from symproof.models import LemmaKind
        bundle = verify_compact(_EMPTY, sympy.Interval(0, 1))
        hb = [l for l in bundle.proof.lemmas if l.name == "heine_borel"]
        assert len(hb) == 1
        assert hb[0].kind == LemmaKind.INFERENCE
        assert hb[0].rule == "Heine-Borel theorem"


class TestPropertyLemmaKind:
    """Test the new PROPERTY verification strategy."""

    def test_property_lemma_directly(self):
        """Verify a PROPERTY lemma works at the verification level."""
        from symproof.models import Lemma, LemmaKind
        from symproof.verification import verify_lemma

        lem = Lemma(
            name="test_open",
            kind=LemmaKind.PROPERTY,
            expr=sympy.Interval.open(0, 1),
            property_name="is_open",
        )
        result = verify_lemma(lem)
        assert result.passed

    def test_property_false_fails(self):
        """PROPERTY lemma fails when property is False."""
        from symproof.models import Lemma, LemmaKind
        from symproof.verification import verify_lemma

        lem = Lemma(
            name="test_open",
            kind=LemmaKind.PROPERTY,
            expr=sympy.Interval(0, 1),  # closed, not open
            property_name="is_open",
        )
        result = verify_lemma(lem)
        assert not result.passed

    def test_property_missing_name_fails(self):
        """PROPERTY lemma without property_name fails."""
        from symproof.models import Lemma, LemmaKind
        from symproof.verification import verify_lemma

        lem = Lemma(
            name="test",
            kind=LemmaKind.PROPERTY,
            expr=sympy.Interval(0, 1),
            property_name="",
        )
        result = verify_lemma(lem)
        assert not result.passed


class TestClopen:
    """Clopen sets: both open and closed. R and EmptySet are clopen."""

    def test_reals_are_open(self):
        bundle = verify_open(_EMPTY, sympy.S.Reals)
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_reals_are_closed(self):
        bundle = verify_closed(_EMPTY, sympy.S.Reals)
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_empty_set_is_open(self):
        bundle = verify_open(_EMPTY, sympy.S.EmptySet)
        assert bundle.proof_result.status == ProofStatus.VERIFIED


class TestExtremeValue:
    def test_quadratic(self):
        bundle = extreme_value(_EMPTY, x**2, x, -1, 2)
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_cubic(self):
        bundle = extreme_value(_EMPTY, x**3 - 3 * x + 1, x, -2, 2)
        assert bundle.proof_result.status == ProofStatus.VERIFIED
