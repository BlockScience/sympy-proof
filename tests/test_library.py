"""Tests for symproof.library — reusable proof bundles.

Tests each library proof:
1. Seals successfully
2. Can be imported into a new proof
3. Re-verifies correctly
4. Has the right hypothesis expression
"""

from __future__ import annotations

import sympy
import pytest

from symproof import (
    Axiom,
    AxiomSet,
    LemmaKind,
    ProofBuilder,
    seal,
    verify_proof,
)
from symproof.library.core import max_ge_first, piecewise_collapse
from symproof.library.defi import (
    amm_output_positive,
    amm_product_nondecreasing,
    fee_complement_positive,
)
from symproof.models import ProofStatus

# ---------------------------------------------------------------------------
# Shared symbols and axiom sets
# ---------------------------------------------------------------------------

A = sympy.Symbol("a", real=True)
B = sympy.Symbol("b", real=True)
X = sympy.Symbol("x")
Rx = sympy.Symbol("R_x", positive=True)
Ry = sympy.Symbol("R_y", positive=True)
F = sympy.Symbol("f")
DX = sympy.Symbol("dx", positive=True)


@pytest.fixture
def real_axioms():
    return AxiomSet(
        name="reals",
        axioms=(
            Axiom(name="a_real", expr=sympy.Eq(sympy.im(A), 0)),
            Axiom(name="b_real", expr=sympy.Eq(sympy.im(B), 0)),
        ),
    )


@pytest.fixture
def positive_x_axioms():
    return AxiomSet(
        name="positive_x",
        axioms=(Axiom(name="x_pos", expr=X > 0),),
    )


@pytest.fixture
def amm_axioms():
    return AxiomSet(
        name="amm_with_fee",
        axioms=(
            Axiom(name="rx_pos", expr=Rx > 0),
            Axiom(name="ry_pos", expr=Ry > 0),
            Axiom(name="f_pos", expr=F > 0),
            Axiom(name="f_lt_1", expr=F < 1),
            Axiom(name="dx_pos", expr=DX > 0),
        ),
    )


# ===================================================================
# core.max_ge_first
# ===================================================================


class TestMaxGeFirst:
    def test_seals(self, real_axioms):
        bundle = max_ge_first(real_axioms, A, B)
        assert len(bundle.bundle_hash) == 64

    def test_hypothesis_expression(self, real_axioms):
        bundle = max_ge_first(real_axioms, A, B)
        assert bundle.hypothesis.expr == sympy.Ge(sympy.Max(A, B), A)

    def test_importable(self, real_axioms):
        """Can import into a new proof."""
        bundle = max_ge_first(real_axioms, A, B)
        h = real_axioms.hypothesis("test", expr=sympy.Ge(A, A))
        script = (
            ProofBuilder(real_axioms, h.name, name="t", claim="a >= a")
            .import_bundle(bundle)
            .lemma(
                "trivial",
                LemmaKind.BOOLEAN,
                expr=sympy.Eq(A, A),
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status == ProofStatus.VERIFIED


# ===================================================================
# core.piecewise_collapse
# ===================================================================


class TestPiecewiseCollapse:
    def test_seals(self, positive_x_axioms):
        bundle = piecewise_collapse(
            positive_x_axioms,
            X, X > 0, sympy.Integer(0),
            assumptions={"x": {"positive": True}},
        )
        assert len(bundle.bundle_hash) == 64

    def test_hypothesis_expression(self, positive_x_axioms):
        pw = sympy.Piecewise((X, X > 0), (sympy.Integer(0), True))
        bundle = piecewise_collapse(
            positive_x_axioms,
            X, X > 0, sympy.Integer(0),
            assumptions={"x": {"positive": True}},
        )
        assert bundle.hypothesis.expr == sympy.Eq(pw, X)

    def test_importable(self, positive_x_axioms):
        bundle = piecewise_collapse(
            positive_x_axioms,
            X, X > 0, sympy.Integer(0),
            assumptions={"x": {"positive": True}},
        )
        h = positive_x_axioms.hypothesis("test", expr=X > 0)
        script = (
            ProofBuilder(
                positive_x_axioms, h.name, name="t", claim="x > 0"
            )
            .import_bundle(bundle)
            .lemma(
                "pos",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(X),
                assumptions={"x": {"positive": True}},
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status == ProofStatus.VERIFIED


# ===================================================================
# defi.fee_complement_positive
# ===================================================================


class TestFeeComplementPositive:
    def test_seals(self, amm_axioms):
        bundle = fee_complement_positive(amm_axioms, F)
        assert len(bundle.bundle_hash) == 64

    def test_hypothesis(self, amm_axioms):
        bundle = fee_complement_positive(amm_axioms, F)
        assert bundle.hypothesis.expr == (1 - F > 0)

    def test_reverifies(self, amm_axioms):
        """Re-verification of the sealed proof passes."""
        bundle = fee_complement_positive(amm_axioms, F)
        result = verify_proof(bundle.proof)
        assert result.status == ProofStatus.VERIFIED


# ===================================================================
# defi.amm_output_positive
# ===================================================================


class TestAmmOutputPositive:
    def test_seals(self, amm_axioms):
        bundle = amm_output_positive(amm_axioms, Rx, Ry, F, DX)
        assert len(bundle.bundle_hash) == 64

    def test_imports_fee_complement(self, amm_axioms):
        bundle = amm_output_positive(amm_axioms, Rx, Ry, F, DX)
        assert len(bundle.proof.imported_bundles) == 1
        assert (
            bundle.proof.imported_bundles[0].hypothesis.name
            == "fee_complement_positive"
        )

    def test_reverifies(self, amm_axioms):
        bundle = amm_output_positive(amm_axioms, Rx, Ry, F, DX)
        result = verify_proof(bundle.proof)
        assert result.status == ProofStatus.VERIFIED

    def test_importable_into_larger_proof(self, amm_axioms):
        """The AMM output bundle can be imported into a higher-level proof."""
        output_bundle = amm_output_positive(amm_axioms, Rx, Ry, F, DX)
        h = amm_axioms.hypothesis("output_check", expr=DX > 0)
        script = (
            ProofBuilder(
                amm_axioms, h.name, name="check", claim="check"
            )
            .import_bundle(output_bundle)
            .lemma(
                "dx_pos",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(DX),
                assumptions={"dx": {"positive": True}},
            )
            .build()
        )
        bundle = seal(amm_axioms, h, script)
        assert len(bundle.bundle_hash) == 64


# ===================================================================
# defi.amm_product_nondecreasing
# ===================================================================


class TestAmmProductNondecreasing:
    def test_seals(self, amm_axioms):
        bundle = amm_product_nondecreasing(amm_axioms, Rx, Ry, F, DX)
        assert len(bundle.bundle_hash) == 64

    def test_imports_fee_complement(self, amm_axioms):
        bundle = amm_product_nondecreasing(amm_axioms, Rx, Ry, F, DX)
        assert len(bundle.proof.imported_bundles) == 1

    def test_reverifies(self, amm_axioms):
        bundle = amm_product_nondecreasing(amm_axioms, Rx, Ry, F, DX)
        result = verify_proof(bundle.proof)
        assert result.status == ProofStatus.VERIFIED


# ===================================================================
# Composition: import library proof into user proof
# ===================================================================


class TestLibraryComposition:
    """End-to-end: user proof imports multiple library bundles."""

    def test_amm_full_safety_proof(self, amm_axioms):
        """Compose fee_complement + output + product into one proof."""
        fee_b = fee_complement_positive(amm_axioms, F)
        output_b = amm_output_positive(amm_axioms, Rx, Ry, F, DX)
        product_b = amm_product_nondecreasing(amm_axioms, Rx, Ry, F, DX)

        h = amm_axioms.hypothesis(
            "amm_safe",
            expr=sympy.And(DX > 0, Rx > 0),
        )
        script = (
            ProofBuilder(
                amm_axioms, h.name,
                name="amm_safety",
                claim="AMM swap is safe: output > 0 and product grows",
            )
            .import_bundle(fee_b)
            .import_bundle(output_b)
            .import_bundle(product_b)
            .lemma(
                "trivial",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(DX),
                assumptions={"dx": {"positive": True}},
            )
            .build()
        )
        bundle = seal(amm_axioms, h, script)
        assert len(bundle.bundle_hash) == 64
        assert len(bundle.proof.imported_bundles) == 3
