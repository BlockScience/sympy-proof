"""Tests for the linear optimization library (LP/ILP/MILP)."""

from __future__ import annotations

import pytest
import sympy

from symproof import AxiomSet, ProofStatus
from symproof.library.linopt import (
    complementary_slackness,
    dual_feasible,
    feasible_point,
    integer_feasible,
    lp_optimal,
    lp_relaxation_bound,
    strong_duality,
)

_EMPTY = AxiomSet(name="test", axioms=())

# Standard LP:
#   min [1,2,0]^T x  s.t. [[1,1,1],[2,1,0]]x = [4,6], x >= 0
#   x* = (3,0,1), y* = (0, 1/2), z* = (0, 3/2, 0)
A = sympy.Matrix([[1, 1, 1], [2, 1, 0]])
b = sympy.Matrix([4, 6])
c = sympy.Matrix([1, 2, 0])
x_star = sympy.Matrix([3, 0, 1])
y_star = sympy.Matrix([0, sympy.Rational(1, 2)])
z_star = sympy.Matrix([0, sympy.Rational(3, 2), 0])


class TestFeasiblePoint:
    def test_feasible(self):
        bundle = feasible_point(_EMPTY, A, b, x_star)
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_infeasible_rejected(self):
        bad_x = sympy.Matrix([1, 1, 1])  # A*[1,1,1] = [3,3] != [4,6]
        with pytest.raises(ValueError):
            feasible_point(_EMPTY, A, b, bad_x)

    def test_lemma_count_matches_constraints_plus_bounds(self):
        bundle = feasible_point(_EMPTY, A, b, x_star)
        # 2 constraint rows + 3 nonneg bounds = 5 lemmas
        assert len(bundle.proof.lemmas) == 5


class TestDualFeasible:
    def test_feasible(self):
        bundle = dual_feasible(_EMPTY, A, c, y_star, z_star)
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_infeasible_dual_rejected(self):
        bad_z = sympy.Matrix([1, 1, 1])
        with pytest.raises(ValueError):
            dual_feasible(_EMPTY, A, c, y_star, bad_z)


class TestStrongDuality:
    def test_zero_gap(self):
        bundle = strong_duality(_EMPTY, c, b, x_star, y_star)
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_nonzero_gap_rejected(self):
        bad_y = sympy.Matrix([1, 1])
        with pytest.raises(ValueError):
            strong_duality(_EMPTY, c, b, x_star, bad_y)


class TestComplementarySlackness:
    def test_slackness_holds(self):
        bundle = complementary_slackness(_EMPTY, x_star, z_star)
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_slackness_violated(self):
        bad_z = sympy.Matrix([1, 0, 0])  # x[0]*z[0] = 3*1 != 0
        with pytest.raises(ValueError):
            complementary_slackness(_EMPTY, x_star, bad_z)


class TestIntegerFeasible:
    def test_integer_point(self):
        bundle = integer_feasible(_EMPTY, A, b, x_star)
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_noninteger_rejected(self):
        frac_x = sympy.Matrix([sympy.Rational(5, 2), sympy.Rational(1, 2), 1])
        with pytest.raises(ValueError):
            integer_feasible(_EMPTY, A, b, frac_x)


class TestLPOptimal:
    def test_composed_optimality(self):
        bundle = lp_optimal(_EMPTY, c, A, b, x_star, y_star, z_star)
        assert bundle.proof_result.status == ProofStatus.VERIFIED
        assert len(bundle.proof.imported_bundles) == 3


class TestLPRelaxationBound:
    def test_tight_bound(self):
        bundle = lp_relaxation_bound(_EMPTY, lp_value=3, ilp_value=3)
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_positive_gap(self):
        bundle = lp_relaxation_bound(
            _EMPTY, lp_value=sympy.Rational(5, 2), ilp_value=3,
        )
        assert bundle.proof_result.status == ProofStatus.VERIFIED
