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


class TestExtremeValue:
    def test_quadratic(self):
        bundle = extreme_value(_EMPTY, x**2, x, -1, 2)
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_cubic(self):
        bundle = extreme_value(_EMPTY, x**3 - 3 * x + 1, x, -2, 2)
        assert bundle.proof_result.status == ProofStatus.VERIFIED
