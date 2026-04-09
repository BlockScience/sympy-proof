"""Tests for symproof.library.convex — convex optimization proofs."""

from __future__ import annotations

import pytest
import sympy

from symproof import Axiom, AxiomSet, ProofBuilder, LemmaKind, seal, verify_proof
from symproof.library.convex import (
    conjugate_function,
    convex_composition,
    convex_hessian,
    convex_scalar,
    convex_sum,
    gp_to_convex,
    strongly_convex,
    unique_minimizer,
)
from symproof.models import ProofStatus


X = sympy.Symbol("x", real=True)
Y = sympy.Symbol("y", real=True)
X_POS = sympy.Symbol("x", positive=True)

TRIVIAL = AxiomSet(
    name="trivial", axioms=(Axiom(name="t", expr=sympy.Eq(1, 1)),)
)
POS_X = AxiomSet(
    name="positive_x", axioms=(Axiom(name="x_pos", expr=X_POS > 0),)
)


class TestConvexScalar:
    def test_exp_convex(self):
        b = convex_scalar(TRIVIAL, sympy.exp(X), X)
        assert b.proof_result.status == ProofStatus.VERIFIED

    def test_neg_log_convex(self):
        b = convex_scalar(
            POS_X, -sympy.log(X_POS), X_POS,
            assumptions={"x": {"positive": True}},
        )
        assert b.proof_result.status == ProofStatus.VERIFIED

    def test_x_squared(self):
        b = convex_scalar(TRIVIAL, X**2, X)
        assert b.bundle_hash

    def test_x_ln_x_convex(self):
        b = convex_scalar(
            POS_X, X_POS * sympy.log(X_POS), X_POS,
            assumptions={"x": {"positive": True}},
        )
        assert b.bundle_hash


class TestConvexHessian:
    def test_quadratic_2d(self):
        b = convex_hessian(TRIVIAL, X**2 + X * Y + Y**2, [X, Y])
        assert b.bundle_hash

    def test_sum_of_squares(self):
        b = convex_hessian(TRIVIAL, X**2 + Y**2, [X, Y])
        assert b.bundle_hash

    def test_3d_diagonal(self):
        z = sympy.Symbol("z", real=True)
        a = sympy.Symbol("a", positive=True)
        b_sym = sympy.Symbol("b", positive=True)
        c = sympy.Symbol("c", positive=True)
        ax = AxiomSet(
            name="pos3",
            axioms=(
                Axiom(name="a_pos", expr=a > 0),
                Axiom(name="b_pos", expr=b_sym > 0),
                Axiom(name="c_pos", expr=c > 0),
            ),
        )
        bundle = convex_hessian(
            ax, a * X**2 + b_sym * Y**2 + c * z**2, [X, Y, z],
            assumptions={
                "a": {"positive": True},
                "b": {"positive": True},
                "c": {"positive": True},
            },
        )
        assert bundle.bundle_hash


class TestStronglyConvex:
    def test_x_squared_is_2_strongly(self):
        b = strongly_convex(TRIVIAL, X**2, [X], sympy.Integer(2))
        assert b.bundle_hash

    def test_quadratic_form(self):
        """3x^2 + 3y^2 is 6-strongly convex (min eigenval = 6)."""
        b = strongly_convex(
            TRIVIAL, 3 * X**2 + 3 * Y**2, [X, Y], sympy.Integer(6),
        )
        assert b.bundle_hash


class TestConjugateFunction:
    def test_quadratic_self_conjugate(self):
        """f(x) = x^2/2 => f*(y) = y^2/2."""
        b = conjugate_function(TRIVIAL, X**2 / 2, X, Y)
        assert b.bundle_hash

    def test_exp_conjugate(self):
        """f(x) = e^x => f*(y) = y*ln(y) - y for y > 0."""
        y_pos = sympy.Symbol("y", positive=True)
        ax = AxiomSet(
            name="pos_y",
            axioms=(Axiom(name="y_pos", expr=y_pos > 0),),
        )
        b = conjugate_function(
            ax, sympy.exp(X), X, y_pos,
            assumptions={"y": {"positive": True}},
        )
        assert b.bundle_hash


class TestConvexSum:
    def test_weighted_sum_scalar(self):
        a = sympy.Symbol("a", positive=True)
        b_sym = sympy.Symbol("b", positive=True)
        ax = AxiomSet(
            name="weights",
            axioms=(
                Axiom(name="a_pos", expr=a > 0),
                Axiom(name="b_pos", expr=b_sym > 0),
            ),
        )
        bundle = convex_sum(
            ax, [X**2, sympy.exp(X)], [a, b_sym], [X],
            assumptions={"a": {"positive": True}, "b": {"positive": True}},
        )
        assert bundle.bundle_hash


class TestConvexComposition:
    def test_exp_of_quadratic(self):
        """exp(x^2) is convex: exp is convex nondecreasing, x^2 is convex."""
        t = sympy.Symbol("t", real=True)
        b = convex_composition(TRIVIAL, sympy.exp(t), X**2, t, X)
        assert b.bundle_hash
        assert len(b.proof.lemmas) == 4

    def test_square_of_exp(self):
        """(exp(x))^2 = exp(2x) is convex: t^2 nondecreasing for t>0, exp convex."""
        t = sympy.Symbol("t", positive=True)
        ax = AxiomSet(
            name="pos_t",
            axioms=(Axiom(name="t_pos", expr=t > 0),),
        )
        b = convex_composition(
            ax, t**2, sympy.exp(X), t, X,
            assumptions={"t": {"positive": True}},
        )
        assert b.bundle_hash


class TestUniqueMinimizer:
    def test_quadratic_unique(self):
        b = unique_minimizer(TRIVIAL, X**2, [X], sympy.Integer(2))
        assert b.bundle_hash
        assert len(b.proof.imported_bundles) == 1

    def test_importable(self):
        """Unique minimizer bundle can be imported."""
        b = unique_minimizer(TRIVIAL, X**2, [X], sympy.Integer(2))
        h = TRIVIAL.hypothesis("test", expr=sympy.Eq(1, 1))
        script = (
            ProofBuilder(TRIVIAL, h.name, name="t", claim="t")
            .import_bundle(b)
            .lemma("trivial", LemmaKind.BOOLEAN, expr=sympy.Eq(1, 1))
            .build()
        )
        result = verify_proof(script)
        assert result.status == ProofStatus.VERIFIED


class TestGPToConvex:
    def test_simple_posynomial(self):
        """x^2 + 3x is a posynomial — transforms to convex."""
        y = sympy.Symbol("y", real=True)
        b = gp_to_convex(
            POS_X, [X_POS**2, 3 * X_POS], y, X_POS,
            assumptions={"x": {"positive": True}},
        )
        assert b.bundle_hash

    def test_single_monomial(self):
        """Single monomial x^3: log-transform gives 3y (affine, trivially convex)."""
        y = sympy.Symbol("y", real=True)
        b = gp_to_convex(
            POS_X, [X_POS**3], y, X_POS,
            assumptions={"x": {"positive": True}},
        )
        assert b.bundle_hash
