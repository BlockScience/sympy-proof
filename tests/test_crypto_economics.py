"""User-testing: crypto-economic mechanism proofs.

Exercises symproof with real invariants from token economics:
- Bitcoin supply cap (geometric series convergence)
- AMM constant-product invariant (fee-driven product growth)
- Bonding curve reserve relationships
- Vesting schedule bounds
- Quadratic voting cost constraint
"""

from __future__ import annotations

import pytest
import sympy

from symproof import (
    Axiom,
    AxiomSet,
    LemmaKind,
    ProofBuilder,
    seal,
    verify_proof,
)
from symproof.tactics import auto_lemma, try_simplify

# ---------------------------------------------------------------------------
# Shared symbols
# ---------------------------------------------------------------------------

k = sympy.Symbol("k", integer=True, nonnegative=True)
K = sympy.Symbol("K", integer=True, nonnegative=True)


# ===================================================================
# Bitcoin supply cap
# ===================================================================


class TestBitcoinSupplyCap:
    """Prove the 21M Bitcoin supply cap from the geometric halving series."""

    @pytest.fixture
    def btc_axioms(self):
        return AxiomSet(
            name="bitcoin_halving",
            axioms=(
                Axiom(
                    name="era_reward",
                    expr=sympy.Eq(
                        sympy.Function("reward")(k),
                        210000 * 50 / sympy.Integer(2) ** k,
                    ),
                    description="Block reward in era k is 210000 * 50 / 2^k",
                ),
            ),
        )

    def test_geometric_series_limit(self, btc_axioms):
        """Sum of all era rewards converges to exactly 21 million."""
        h = btc_axioms.hypothesis(
            "supply_cap",
            expr=sympy.Eq(
                sympy.Sum(210000 * 50 / sympy.Integer(2) ** k, (k, 0, sympy.oo)),
                sympy.Integer(21_000_000),
            ),
        )
        script = (
            ProofBuilder(
                btc_axioms,
                h.name,
                name="supply_cap_proof",
                claim="Total BTC supply = 21M",
            )
            .lemma(
                "geometric_series_limit",
                LemmaKind.EQUALITY,
                expr=sympy.Sum(210000 * 50 / sympy.Integer(2) ** k, (k, 0, sympy.oo)),
                expected=sympy.Integer(21_000_000),
                description="Infinite halving series sums to 21M",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_partial_sum_closed_form(self, btc_axioms):
        """Closed-form partial sum: S(K) = 21M - 10.5M / 2^K."""
        h = btc_axioms.hypothesis(
            "partial_sum",
            expr=sympy.Eq(
                sympy.Sum(210000 * 50 / sympy.Integer(2) ** k, (k, 0, K)),
                sympy.Integer(21_000_000)
                - sympy.Rational(10_500_000) / sympy.Integer(2) ** K,
            ),
        )
        script = (
            ProofBuilder(
                btc_axioms,
                h.name,
                name="partial_sum_proof",
                claim="Closed form of partial halving sum",
            )
            .lemma(
                "partial_sum_closed_form",
                LemmaKind.EQUALITY,
                expr=sympy.Sum(210000 * 50 / sympy.Integer(2) ** k, (k, 0, K)),
                expected=sympy.Integer(21_000_000)
                - sympy.Rational(10_500_000) / sympy.Integer(2) ** K,
                assumptions={"K": {"integer": True, "nonnegative": True}},
                description="S(K) = 21M - 10.5M/2^K",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_remainder_positive(self, btc_axioms):
        """Unmined supply 10.5M / 2^K is always positive."""
        h = btc_axioms.hypothesis(
            "remainder_positive",
            expr=sympy.Gt(sympy.Rational(10_500_000) / sympy.Integer(2) ** K, 0),
        )
        script = (
            ProofBuilder(
                btc_axioms,
                h.name,
                name="remainder_positive_proof",
                claim="Unmined supply is always positive",
            )
            .lemma(
                "remainder_positive",
                LemmaKind.BOOLEAN,
                expr=sympy.Gt(
                    sympy.Rational(10_500_000) / sympy.Integer(2) ** K, 0
                ),
                assumptions={"K": {"integer": True, "nonnegative": True}},
                description="10.5M / 2^K > 0 for all non-negative integer K",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_partial_sum_bounded(self, btc_axioms):
        """Every partial sum is strictly less than 21M."""
        h = btc_axioms.hypothesis(
            "partial_bounded",
            expr=sympy.Lt(
                sympy.Integer(21_000_000)
                - sympy.Rational(10_500_000) / sympy.Integer(2) ** K,
                sympy.Integer(21_000_000),
            ),
        )
        script = (
            ProofBuilder(
                btc_axioms,
                h.name,
                name="partial_bounded_proof",
                claim="S(K) < 21M for all finite K",
            )
            .lemma(
                "partial_sum_bounded",
                LemmaKind.BOOLEAN,
                expr=sympy.Lt(
                    sympy.Integer(21_000_000)
                    - sympy.Rational(10_500_000) / sympy.Integer(2) ** K,
                    sympy.Integer(21_000_000),
                ),
                assumptions={"K": {"integer": True, "nonnegative": True}},
                description="21M - 10.5M/2^K < 21M since remainder > 0",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_multi_lemma_supply_cap_seal(self, btc_axioms):
        """Full chained proof: series limit + remainder + bound, sealed."""
        h = btc_axioms.hypothesis(
            "supply_cap_full",
            expr=sympy.Eq(
                sympy.Sum(210000 * 50 / sympy.Integer(2) ** k, (k, 0, sympy.oo)),
                sympy.Integer(21_000_000),
            ),
        )
        script = (
            ProofBuilder(
                btc_axioms,
                h.name,
                name="supply_cap_chained",
                claim="Full BTC supply cap proof chain",
            )
            .lemma(
                "geometric_series_limit",
                LemmaKind.EQUALITY,
                expr=sympy.Sum(
                    210000 * 50 / sympy.Integer(2) ** k, (k, 0, sympy.oo)
                ),
                expected=sympy.Integer(21_000_000),
                description="Series converges to 21M",
            )
            .lemma(
                "partial_sum_closed_form",
                LemmaKind.EQUALITY,
                expr=sympy.Sum(210000 * 50 / sympy.Integer(2) ** k, (k, 0, K)),
                expected=sympy.Integer(21_000_000)
                - sympy.Rational(10_500_000) / sympy.Integer(2) ** K,
                assumptions={"K": {"integer": True, "nonnegative": True}},
                depends_on=["geometric_series_limit"],
                description="Closed-form partial sum",
            )
            .lemma(
                "remainder_positive",
                LemmaKind.BOOLEAN,
                expr=sympy.Gt(
                    sympy.Rational(10_500_000) / sympy.Integer(2) ** K, 0
                ),
                assumptions={"K": {"integer": True, "nonnegative": True}},
                depends_on=["partial_sum_closed_form"],
                description="Unmined supply > 0",
            )
            .build()
        )
        bundle = seal(btc_axioms, h, script)
        assert bundle.bundle_hash  # 64-char hex


# ===================================================================
# AMM constant-product invariant
# ===================================================================


class TestAMMConstantProduct:
    """Prove properties of the x*y=k AMM with fees."""

    @pytest.fixture
    def amm_axioms(self):
        Rx = sympy.Symbol("R_x", positive=True)
        Ry = sympy.Symbol("R_y", positive=True)
        fee = sympy.Symbol("f", positive=True)
        return AxiomSet(
            name="amm_constant_product",
            axioms=(
                Axiom(name="reserve_x_positive", expr=Rx > 0),
                Axiom(name="reserve_y_positive", expr=Ry > 0),
                Axiom(name="fee_in_unit_interval", expr=sympy.And(fee > 0, fee < 1)),
            ),
        )

    @pytest.mark.xfail(
        reason="SymPy Q-system cannot propagate positivity through "
        "rational expressions with bounded-interval constraints (0 < f < 1)"
    )
    def test_product_ratio_exceeds_one(self, amm_axioms):
        """Post-swap product >= pre-swap product when fee > 0."""
        Rx = sympy.Symbol("R_x", positive=True)
        Ry = sympy.Symbol("R_y", positive=True)
        fee = sympy.Symbol("f", positive=True)
        dx = sympy.Symbol("dx", positive=True)

        # AMM output: dy = Ry * dx*(1-f) / (Rx + dx*(1-f))
        net = dx * (1 - fee)
        dy = Ry * net / (Rx + net)

        # Post-swap reserves
        Rx_new = Rx + dx
        Ry_new = Ry - dy

        # Product ratio
        ratio = sympy.simplify((Rx_new * Ry_new) / (Rx * Ry))

        h = amm_axioms.hypothesis(
            "product_nondecreasing",
            expr=sympy.Ge(Rx_new * Ry_new, Rx * Ry),
        )
        script = (
            ProofBuilder(
                amm_axioms,
                h.name,
                name="amm_product_growth",
                claim="Fee-bearing swap grows the invariant product",
            )
            .lemma(
                "ratio_simplified",
                LemmaKind.EQUALITY,
                expr=ratio,
                expected=(Rx + dx) / (Rx + dx * (1 - fee)),
                assumptions={
                    "R_x": {"positive": True},
                    "R_y": {"positive": True},
                    "f": {"positive": True},
                    "dx": {"positive": True},
                },
                description="Product ratio simplifies to (Rx+dx)/(Rx+dx*(1-f))",
            )
            .lemma(
                "ratio_ge_one",
                LemmaKind.QUERY,
                expr=sympy.Q.positive((Rx + dx) / (Rx + dx * (1 - fee)) - 1),
                assumptions={
                    "R_x": {"positive": True},
                    "f": {"positive": True},
                    "dx": {"positive": True},
                },
                depends_on=["ratio_simplified"],
                description="Ratio > 1 since fee > 0",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", (
            f"{result.failure_summary}\n"
            + "\n".join(
                f"  {lr.lemma_name}: passed={lr.passed}, error={lr.error}"
                for lr in result.lemma_results
            )
        )

    @pytest.mark.xfail(
        reason="SymPy Q-system cannot propagate positivity through "
        "rational expressions with bounded-interval constraints (0 < f < 1)"
    )
    def test_output_positive(self, amm_axioms):
        """AMM output dy > 0 when reserves and input are positive."""
        Rx = sympy.Symbol("R_x", positive=True)
        Ry = sympy.Symbol("R_y", positive=True)
        fee = sympy.Symbol("f", positive=True)
        dx = sympy.Symbol("dx", positive=True)

        net = dx * (1 - fee)
        dy = Ry * net / (Rx + net)

        h = amm_axioms.hypothesis("output_positive", expr=dy > 0)
        script = (
            ProofBuilder(
                amm_axioms,
                h.name,
                name="output_positivity",
                claim="AMM output is positive given positive inputs",
            )
            .lemma(
                "dy_positive",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(dy),
                assumptions={
                    "R_x": {"positive": True},
                    "R_y": {"positive": True},
                    "f": {"positive": True},
                    "dx": {"positive": True},
                },
                description="dy > 0 from positive reserves and input",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary


# ===================================================================
# Bonding curve
# ===================================================================


class TestBondingCurve:
    """Power-law bonding curve: price = supply^n."""

    @pytest.fixture
    def curve_axioms(self):
        S = sympy.Symbol("S", positive=True)
        n = sympy.Symbol("n", positive=True)
        return AxiomSet(
            name="bonding_curve",
            axioms=(
                Axiom(name="supply_positive", expr=S > 0),
                Axiom(name="exponent_positive", expr=n > 0),
            ),
        )

    def test_reserve_is_integral_of_price(self, curve_axioms):
        """Reserve R(S) = integral of price = S^(n+1)/(n+1)."""
        S = sympy.Symbol("S", positive=True)
        n = sympy.Symbol("n", positive=True)
        t = sympy.Symbol("t")

        price = t**n
        reserve = sympy.integrate(price, (t, 0, S))

        h = curve_axioms.hypothesis(
            "reserve_integral",
            expr=sympy.Eq(reserve, S ** (n + 1) / (n + 1)),
        )
        script = (
            ProofBuilder(
                curve_axioms,
                h.name,
                name="reserve_integral_proof",
                claim="Reserve equals integral of price over supply",
            )
            .lemma(
                "integral_evaluation",
                LemmaKind.EQUALITY,
                expr=reserve,
                expected=S ** (n + 1) / (n + 1),
                assumptions={"S": {"positive": True}, "n": {"positive": True}},
                description="int_0^S t^n dt = S^(n+1)/(n+1)",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_price_monotone_increasing(self, curve_axioms):
        """Price is monotonically increasing in supply for n > 0."""
        S = sympy.Symbol("S", positive=True)
        n = sympy.Symbol("n", positive=True)

        price = S**n
        dprice = sympy.diff(price, S)

        h = curve_axioms.hypothesis(
            "price_monotone",
            expr=dprice > 0,
        )
        script = (
            ProofBuilder(
                curve_axioms,
                h.name,
                name="price_monotone_proof",
                claim="d(price)/dS > 0 for S > 0, n > 0",
            )
            .lemma(
                "derivative_positive",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(dprice),
                assumptions={"S": {"positive": True}, "n": {"positive": True}},
                description="n * S^(n-1) > 0",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary


# ===================================================================
# Vesting schedule bounds
# ===================================================================


class TestVestingSchedule:
    """Vesting invariants: released <= vested <= total_alloc."""

    @pytest.fixture
    def vest_axioms(self):
        v = sympy.Symbol("vested", nonnegative=True)
        r = sympy.Symbol("released", nonnegative=True)
        T = sympy.Symbol("total_alloc", positive=True)
        return AxiomSet(
            name="vesting_schedule",
            axioms=(
                Axiom(name="vested_le_total", expr=sympy.Le(v, T)),
                Axiom(name="released_le_vested", expr=sympy.Le(r, v)),
                Axiom(name="total_positive", expr=T > 0),
            ),
        )

    def test_released_le_total(self, vest_axioms):
        """Transitivity: released <= vested <= total implies released <= total."""
        v = sympy.Symbol("vested", nonnegative=True)
        r = sympy.Symbol("released", nonnegative=True)
        T = sympy.Symbol("total_alloc", positive=True)

        h = vest_axioms.hypothesis(
            "released_le_total",
            expr=sympy.Le(r, T),
        )
        script = (
            ProofBuilder(
                vest_axioms,
                h.name,
                name="transitivity_proof",
                claim="released <= vested <= total implies released <= total",
            )
            .lemma(
                "chain_inequality",
                LemmaKind.BOOLEAN,
                expr=sympy.Implies(
                    sympy.And(sympy.Le(r, v), sympy.Le(v, T)),
                    sympy.Le(r, T),
                ),
                description="Transitivity of <=",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary


# ===================================================================
# Quadratic voting
# ===================================================================


class TestQuadraticVoting:
    """Quadratic voting cost: n votes costs n^2 credits."""

    @pytest.fixture
    def qv_axioms(self):
        c = sympy.Symbol("credits", nonnegative=True)
        return AxiomSet(
            name="quadratic_voting",
            axioms=(Axiom(name="credits_nonneg", expr=sympy.Ge(c, 0)),),
        )

    def test_cost_superlinear(self, qv_axioms):
        """Marginal cost of nth vote > marginal cost of (n-1)th vote."""
        n = sympy.Symbol("n", positive=True, integer=True)

        marginal_cost_n = n**2 - (n - 1) ** 2  # = 2n - 1
        marginal_cost_prev = (n - 1) ** 2 - (n - 2) ** 2  # = 2n - 3

        h = qv_axioms.hypothesis(
            "cost_superlinear",
            expr=sympy.Gt(marginal_cost_n, marginal_cost_prev),
        )
        script = (
            ProofBuilder(
                qv_axioms,
                h.name,
                name="superlinear_cost_proof",
                claim="Marginal cost increases with each additional vote",
            )
            .lemma(
                "marginal_cost_diff",
                LemmaKind.EQUALITY,
                expr=sympy.simplify(marginal_cost_n - marginal_cost_prev),
                expected=sympy.Integer(2),
                description="Marginal cost increases by 2 per additional vote",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary


# ===================================================================
# Tactics: auto_lemma on economic invariants
# ===================================================================


class TestAutoLemmaCryptoEcon:
    """Exercise auto_lemma on economic expressions."""

    def test_auto_detects_geometric_series(self):
        """auto_lemma should find EQUALITY for the geometric series."""
        lemma = auto_lemma(
            "series_limit",
            sympy.Sum(sympy.Rational(1, 2) ** k, (k, 0, sympy.oo)),
            expected=sympy.Integer(2),
        )
        assert lemma is not None
        assert lemma.kind == LemmaKind.EQUALITY

    def test_auto_detects_positivity_query(self):
        """auto_lemma should find QUERY for positivity under assumptions."""
        x = sympy.Symbol("x")
        y = sympy.Symbol("y")
        lemma = auto_lemma(
            "product_positive",
            sympy.Q.positive(x * y),
            assumptions={"x": {"positive": True}, "y": {"positive": True}},
        )
        assert lemma is not None
        assert lemma.kind == LemmaKind.QUERY

    def test_auto_on_implication(self):
        """auto_lemma on a simple implication."""
        x = sympy.Symbol("x")
        lemma = auto_lemma(
            "pos_implies_nonneg",
            sympy.Implies(x > 0, x >= 0),
        )
        assert lemma is not None
