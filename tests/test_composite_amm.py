"""Tests for ReservePairType composite — AMM constant-product proofs.

Demonstrates how the composite type API simplifies AMM proofs:
swap output formulas, invariant preservation, and hyperbolic coordinate
transforms are all generated from the type declaration.
"""

from __future__ import annotations

import sympy
from sympy import Integer, Rational, Symbol, floor, sqrt

from symproof import (
    Axiom,
    LemmaKind,
    ProofBuilder,
    seal,
    verify_proof,
)
from symproof.composite import ReservePairType, make_axiom_set


class TestReservePairConstruction:
    """Verify ReservePairType produces correct symbols and formulas."""

    def test_symbols_created(self):
        pool = ReservePairType(name="amm")
        assert pool.rx.name == "R_x"
        assert pool.ry.name == "R_y"
        assert pool.k.name == "k"
        assert pool.s.name == "s"

    def test_custom_symbol_names(self):
        pool = ReservePairType(name="pool", rx_name="A", ry_name="B")
        assert pool.rx.name == "A"
        assert pool.ry.name == "B"

    def test_swap_output_formula(self):
        pool = ReservePairType()
        dx = Symbol("dx", positive=True)
        dy = pool.swap_output(dx)
        # Should be Ry * dx / (Rx + dx)
        expected = pool.ry * dx / (pool.rx + dx)
        assert sympy.simplify(dy - expected) == 0

    def test_swap_output_int_formula(self):
        pool = ReservePairType()
        dx = Symbol("dx", positive=True)
        dy = pool.swap_output_int(dx)
        assert isinstance(dy, sympy.floor)

    def test_axioms_generated(self):
        pool = ReservePairType()
        axioms = pool.axioms()
        assert len(axioms) == 2  # rx_positive, ry_positive
        names = {a.name for a in axioms}
        assert "R_x_positive" in names
        assert "R_y_positive" in names

    def test_hyperbolic_transform(self):
        pool = ReservePairType()
        fwd, inv = pool.hyperbolic_transform()
        assert "R_x" in fwd
        assert "R_y" in fwd
        assert "k" in inv
        assert "s" in inv

    def test_make_axiom_set_with_fee(self):
        """Combine pool axioms with a fee axiom."""
        pool = ReservePairType()
        fee = Symbol("f", positive=True)
        fee_axiom = Axiom(
            name="fee_bounded",
            expr=sympy.And(fee > 0, fee < 1),
        )
        axiom_set = make_axiom_set("amm_with_fee", pool, fee_axiom)
        assert len(axiom_set.axioms) == 3


class TestAMMConcreteSwap:
    """Prove AMM swap properties with concrete reserve values."""

    def setup_method(self):
        self.pool = ReservePairType(name="amm")
        self.axioms = make_axiom_set("amm_pool", self.pool)

    def test_swap_output_concrete(self):
        """Concrete: Rx=1000, Ry=2000, dx=100 → dy = 2000*100/1100."""
        pool = self.pool
        # Substitute concrete values
        Rx_val, Ry_val, dx_val = Integer(1000), Integer(2000), Integer(100)
        dy_exact = Ry_val * dx_val / (Rx_val + dx_val)  # 200000/1100
        dy_expected = Rational(200000, 1100)

        h = self.axioms.hypothesis(
            "swap_output",
            expr=sympy.Eq(dy_exact, dy_expected),
        )
        script = (
            ProofBuilder(
                self.axioms, h.name,
                name="swap_concrete", claim="dy = 200000/1100",
            )
            .lemma(
                "dy_value",
                LemmaKind.EQUALITY,
                expr=dy_exact,
                expected=dy_expected,
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_integer_swap_preserves_product(self):
        """Floor-truncated swap preserves product (concrete values)."""
        pool = self.pool
        Rx_val, Ry_val, dx_val = Integer(1000), Integer(2000), Integer(100)
        dy_int = floor(Ry_val * dx_val / (Rx_val + dx_val))

        k_pre = Rx_val * Ry_val
        k_post = (Rx_val + dx_val) * (Ry_val - dy_int)

        h = self.axioms.hypothesis(
            "k_nondecreasing",
            expr=sympy.Ge(k_post, k_pre),
        )
        script = (
            ProofBuilder(
                self.axioms, h.name,
                name="k_preserved", claim="Floor truncation preserves product",
            )
            .lemma(
                "k_ge",
                LemmaKind.BOOLEAN,
                expr=sympy.Ge(k_post, k_pre),
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary


class TestAMMCoordinateTransform:
    """Prove AMM properties via hyperbolic coordinate transform."""

    def setup_method(self):
        self.pool = ReservePairType(name="amm")
        self.axioms = make_axiom_set("amm_pool", self.pool)

    def test_invariant_is_coordinate(self):
        """Rx * Ry = k via COORDINATE_TRANSFORM lemma from pool type."""
        pool = self.pool
        h = self.axioms.hypothesis(
            "invariant_k",
            expr=sympy.Eq(pool.rx * pool.ry, pool.k),
        )
        script = (
            ProofBuilder(
                self.axioms, h.name,
                name="invariant_proof", claim="Rx*Ry = k in hyperbolic coords",
            )
            .add_lemma(pool.invariant_is_coordinate_lemma(name="product_is_k"))
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_price_is_coordinate(self):
        """Rx / Ry = s via COORDINATE_TRANSFORM lemma from pool type."""
        pool = self.pool
        h = self.axioms.hypothesis(
            "price_s",
            expr=sympy.Eq(pool.rx / pool.ry, pool.s),
        )
        script = (
            ProofBuilder(
                self.axioms, h.name,
                name="price_proof", claim="Rx/Ry = s in hyperbolic coords",
            )
            .add_lemma(pool.price_is_coordinate_lemma(name="ratio_is_s"))
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_sealed_invariant(self):
        """Sealed proof: product invariant via hyperbolic coordinates."""
        pool = self.pool
        h = self.axioms.hypothesis(
            "invariant_sealed",
            expr=sympy.Eq(pool.rx * pool.ry, pool.k),
        )
        script = (
            ProofBuilder(
                self.axioms, h.name,
                name="sealed_invariant",
                claim="Rx*Ry = k via hyperbolic coordinate transform",
            )
            .add_lemma(pool.invariant_is_coordinate_lemma(name="product_is_k"))
            .build()
        )
        bundle = seal(self.axioms, h, script)
        assert len(bundle.bundle_hash) == 64

    def test_multi_lemma_transform_proof(self):
        """Chain: invariant is k, price is s — both via coordinate transform."""
        pool = self.pool
        h = self.axioms.hypothesis(
            "both_coordinates",
            expr=sympy.And(
                sympy.Eq(pool.rx * pool.ry, pool.k),
                sympy.Eq(pool.rx / pool.ry, pool.s),
            ),
        )
        script = (
            ProofBuilder(
                self.axioms, h.name,
                name="dual_transform",
                claim="Both (k, s) are coordinates in hyperbolic space",
            )
            .add_lemma(pool.invariant_is_coordinate_lemma(name="k_coord"))
            .add_lemma(pool.price_is_coordinate_lemma(name="s_coord"))
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary


class TestAMMSwapFormulas:
    """Verify swap formula generation with fees."""

    def test_swap_with_fee_formula(self):
        pool = ReservePairType()
        dx = Symbol("dx", positive=True)
        fee = Symbol("f", positive=True)
        dy = pool.swap_output_with_fee(dx, fee)
        # Should be Ry * dx*(1-f) / (Rx + dx*(1-f))
        net = dx * (1 - fee)
        expected = pool.ry * net / (pool.rx + net)
        assert sympy.simplify(dy - expected) == 0

    def test_post_swap_reserves(self):
        pool = ReservePairType()
        dx = Symbol("dx", positive=True)
        rx_new, ry_new = pool.post_swap_reserves(dx, integer=False)
        assert sympy.simplify(rx_new - (pool.rx + dx)) == 0

    def test_post_swap_k_integer(self):
        """post_swap_k with integer truncation should be computable."""
        pool = ReservePairType()
        Rx_val, Ry_val, dx_val = Integer(1000), Integer(2000), Integer(100)
        # Substitute concrete values
        k_post = pool.post_swap_k(dx_val, integer=True).subs(
            {pool.rx: Rx_val, pool.ry: Ry_val}
        )
        k_pre = Rx_val * Ry_val
        # Floor truncation means k_post >= k_pre
        assert k_post >= k_pre


class TestOutputLemma:
    """Verify the output_lemma generator."""

    def test_output_lemma_concrete(self):
        pool = ReservePairType()
        axioms = make_axiom_set("pool", pool)
        dx_val = Integer(100)
        Rx_val, Ry_val = Integer(1000), Integer(2000)

        # Compute expected output
        dy = Ry_val * dx_val / (Rx_val + dx_val)

        lemma = pool.output_lemma(
            dx_val,
            expected=dy,
            name="dy_check",
        )
        # The lemma expr should be the swap formula with pool symbols
        assert lemma.kind == LemmaKind.EQUALITY
        assert lemma.name == "dy_check"
