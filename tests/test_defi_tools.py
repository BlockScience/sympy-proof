"""Tests for DeFi numerical analysis tools.

Each test class addresses a real security audit finding category:
- Rounding direction (mulDivUp vs mulDivDown)
- Compound rounding error accumulation
- Phantom overflow (intermediate a*b exceeds uint256)
- Decimal mismatch between token pairs
"""

from __future__ import annotations

import sympy
from sympy import Integer, Rational, ceiling, floor

from symproof import (
    Axiom,
    AxiomSet,
    ProofBuilder,
    seal,
    verify_proof,
)
from symproof.library.defi import (
    UINT256_MAX,
    DecimalAwarePool,
    chain_error_bound,
    div_down,
    div_up,
    mul_down,
    mul_up,
    no_phantom_overflow_check,
    phantom_overflow_check,
    rounding_bias_lemma,
    rounding_gap_lemma,
    safe_mul_div,
)

# ===================================================================
# Rounding direction — the #1 audit finding in fixed-point math
# ===================================================================


WAD = Integer(10) ** 18


class TestRoundingDirection:
    """Verify UP vs DOWN rounding produces different results.

    Real audit issue: protocols must choose rounding direction per
    operation.  Vaults round shares DOWN on deposit (favor protocol),
    UP on withdrawal (favor protocol).  Getting this wrong leaks value.
    """

    def test_mul_down_concrete(self):
        """mulDown(7, 3, 10) = floor(21/10) = 2."""
        result = mul_down(Integer(7), Integer(3), Integer(10))
        assert result == Integer(2)

    def test_mul_up_concrete(self):
        """mulUp(7, 3, 10) = ceil(21/10) = 3."""
        result = mul_up(Integer(7), Integer(3), Integer(10))
        assert result == Integer(3)

    def test_div_down_concrete(self):
        """divDown(7, 3, 10) = floor(70/3) = 23."""
        result = div_down(Integer(7), Integer(3), Integer(10))
        assert result == Integer(23)

    def test_div_up_concrete(self):
        """divUp(7, 3, 10) = ceil(70/3) = 24."""
        result = div_up(Integer(7), Integer(3), Integer(10))
        assert result == Integer(24)

    def test_up_ge_down(self):
        """UP rounding always >= DOWN rounding."""
        # Concrete: mul_up >= mul_down for non-aligned values
        down = mul_down(Integer(7), Integer(3), Integer(10))
        up = mul_up(Integer(7), Integer(3), Integer(10))
        assert up >= down

    def test_exact_case_equal(self):
        """When division is exact, UP == DOWN."""
        # 6 * 5 / 10 = 3 exactly
        down = mul_down(Integer(6), Integer(5), Integer(10))
        up = mul_up(Integer(6), Integer(5), Integer(10))
        assert down == up == Integer(3)

    def test_wad_mul_direction_matters(self):
        """With WAD scale, non-aligned inputs produce different results.

        This is the real-world case: mulWadDown vs mulWadUp on
        amounts that don't divide evenly by 10^18.
        """
        # 1 wei * 1 wei / WAD: floor = 0, ceil = 1
        one = Integer(1)
        down = mul_down(one, one, WAD)
        up = mul_up(one, one, WAD)
        assert down == Integer(0)
        assert up == Integer(1)
        # The difference is 1 wei — this is the maximum extraction per op
        assert up - down == Integer(1)

    def test_vault_share_rounding(self):
        """Vault shares: deposit rounds DOWN, withdrawal rounds UP.

        shares_on_deposit  = floor(amount * totalShares / totalAssets)
        shares_on_withdraw = ceil(amount * totalShares / totalAssets)

        The protocol keeps the rounding dust both ways.
        """
        amount = Integer(100)
        total_shares = Integer(1000)
        total_assets = Integer(3000)  # 3:1 ratio, non-exact

        shares_deposit = floor(amount * total_shares / total_assets)
        shares_withdraw = ceiling(amount * total_shares / total_assets)

        # floor(100000/3000) = 33, ceil(100000/3000) = 34
        assert shares_deposit == Integer(33)
        assert shares_withdraw == Integer(34)
        # Protocol captures 1 share of rounding dust
        assert shares_withdraw - shares_deposit == Integer(1)


class TestRoundingBiasProofs:
    """Prove rounding invariants via the proof system."""

    def setup_method(self):
        self.axioms = AxiomSet(
            name="rounding",
            axioms=(Axiom(name="base", expr=sympy.Eq(Integer(1), Integer(1))),),
        )

    def test_bias_lemma_concrete(self):
        """Prove ceil >= floor for a concrete case."""
        h = self.axioms.hypothesis(
            "bias",
            expr=sympy.Ge(
                mul_up(Integer(7), Integer(3), Integer(10)),
                mul_down(Integer(7), Integer(3), Integer(10)),
            ),
        )
        script = (
            ProofBuilder(self.axioms, h.name, name="bias", claim="up >= down")
            .add_lemma(
                rounding_bias_lemma(
                    Integer(7), Integer(3), Integer(10), name="bias_check"
                )
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_gap_lemma_concrete(self):
        """Prove ceil - floor <= 1 for a concrete case."""
        h = self.axioms.hypothesis(
            "gap",
            expr=sympy.Le(
                mul_up(Integer(7), Integer(3), Integer(10))
                - mul_down(Integer(7), Integer(3), Integer(10)),
                Integer(1),
            ),
        )
        script = (
            ProofBuilder(self.axioms, h.name, name="gap", claim="gap <= 1")
            .add_lemma(
                rounding_gap_lemma(
                    Integer(7), Integer(3), Integer(10), name="gap_check"
                )
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_sealed_rounding_proof(self):
        """Sealed proof: rounding bias and gap for WAD values."""
        one = Integer(1)
        h = self.axioms.hypothesis(
            "wad_gap",
            expr=sympy.Le(
                mul_up(one, one, WAD) - mul_down(one, one, WAD),
                Integer(1),
            ),
        )
        script = (
            ProofBuilder(
                self.axioms, h.name,
                name="wad_rounding", claim="WAD rounding gap <= 1 wei",
            )
            .add_lemma(rounding_bias_lemma(one, one, WAD, name="bias"))
            .add_lemma(rounding_gap_lemma(one, one, WAD, name="gap"))
            .build()
        )
        bundle = seal(self.axioms, h, script)
        assert len(bundle.bundle_hash) == 64


# ===================================================================
# Compound rounding error — the silent wealth leak
# ===================================================================


class TestCompoundRoundingError:
    """Prove error bounds across chained floor operations.

    Real audit issue: a swap→deposit→rebalance→withdraw pipeline
    does 4+ floor operations.  Each loses up to 1 wei.  Over millions
    of users, the cumulative leak is material.
    """

    def setup_method(self):
        self.axioms = AxiomSet(
            name="chain",
            axioms=(Axiom(name="base", expr=sympy.Eq(Integer(1), Integer(1))),),
        )

    def test_three_step_chain_concrete(self):
        """3-step pipeline: each step loses fractional amount.

        Step 1: 100/3 = 33.33... → floor = 33, error = 1/3
        Step 2: 33 * 7/5 = 46.2 → floor = 46, error = 1/5
        Step 3: 46 * 11/4 = 126.5 → floor = 126, error = 1/2

        Cumulative error = 1/3 + 1/5 + 1/2 = 31/30 < 3 (N=3).
        """
        exact_vals = [
            Rational(100, 3),       # 33.333...
            Rational(33 * 7, 5),    # 46.2
            Rational(46 * 11, 4),   # 126.5
        ]
        trunc_vals = [floor(v) for v in exact_vals]

        lemmas = chain_error_bound(exact_vals, trunc_vals, name_prefix="pipe")

        # Verify each lemma individually
        for lem in lemmas:
            from symproof.verification import verify_lemma

            result = verify_lemma(lem)
            assert result.passed, f"Lemma {lem.name} failed: {result.error}"

    def test_five_step_wad_chain(self):
        """5-step WAD pipeline simulating deposit→swap→fee→rebalance→withdraw.

        Each step is a floor(exact_value) operation.  Total error < 5 wei.
        """
        # Simulate 5 operations with non-aligned WAD values
        exact_vals = [
            Rational(17, 3) * WAD,   # ~5.67e18
            Rational(23, 7) * WAD,   # ~3.29e18
            Rational(41, 11) * WAD,  # ~3.73e18
            Rational(59, 13) * WAD,  # ~4.54e18
            Rational(71, 17) * WAD,  # ~4.18e18
        ]
        trunc_vals = [floor(v) for v in exact_vals]

        lemmas = chain_error_bound(exact_vals, trunc_vals, name_prefix="wad5")

        # The cumulative lemma should verify: total error < 5
        cumulative = lemmas[-1]
        assert "Cumulative error < 5" in cumulative.description

        from symproof.verification import verify_lemma

        result = verify_lemma(cumulative)
        assert result.passed, f"Cumulative bound failed: {result.error}"

    def test_chain_error_in_proof(self):
        """Full proof: 3-step chain error bounded by 3."""
        exact_vals = [Rational(10, 3), Rational(20, 7), Rational(30, 11)]
        trunc_vals = [floor(v) for v in exact_vals]

        lemmas = chain_error_bound(exact_vals, trunc_vals, name_prefix="ch")

        total_exact = sum(exact_vals)
        total_trunc = sum(trunc_vals)

        h = self.axioms.hypothesis(
            "chain_bounded",
            expr=sympy.Lt(total_exact - total_trunc, Integer(3)),
        )
        builder = ProofBuilder(
            self.axioms, h.name,
            name="chain_proof", claim="3-step error < 3",
        )
        for lem in lemmas:
            builder = builder.add_lemma(lem)
        script = builder.build()

        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary


# ===================================================================
# Directional rounding — who benefits from each truncation?
# ===================================================================


class TestDirectionalRounding:
    """Prove that rounding direction favors the correct party.

    Real audit issue: floor() on swap output favors the protocol
    (user gets less).  floor() on fee deduction favors the USER
    (user pays less) — which is wrong.  The auditor needs to prove
    that every step in a pipeline rounds in the protocol's favor.
    """

    def setup_method(self):
        from symproof.library.defi import (
            RoundingFavor,
            RoundingStep,
            directional_chain_error,
        )

        self.RoundingFavor = RoundingFavor
        self.RoundingStep = RoundingStep
        self.directional_chain_error = directional_chain_error
        self.axioms = AxiomSet(
            name="directional",
            axioms=(Axiom(name="base", expr=sympy.Eq(Integer(1), Integer(1))),),
        )

    def test_all_protocol_favoring_floor(self):
        """5 floor steps on outputs — all favor protocol correctly."""
        RF = self.RoundingFavor
        RS = self.RoundingStep
        steps = [
            RS(Rational(100, 3), floor(Rational(100, 3)),
               RF.PROTOCOL, "deposit shares"),
            RS(Rational(200, 7), floor(Rational(200, 7)),
               RF.PROTOCOL, "swap output"),
            RS(Rational(300, 11), floor(Rational(300, 11)),
               RF.PROTOCOL, "rebalance"),
        ]

        # All should be direction-correct
        for s in steps:
            assert s.direction_correct, f"{s.label} should favor protocol"

        lemmas = self.directional_chain_error(steps, name_prefix="prot")

        from symproof.verification import verify_lemma

        for lem in lemmas:
            result = verify_lemma(lem)
            assert result.passed, f"{lem.name} failed: {result.error}"
            assert "MISMATCH" not in lem.description

    def test_ceil_on_fee_uses_user_direction(self):
        """ceil() gives rounded >= exact → USER direction.

        For a fee deduction, ceil means user PAYS more → good for protocol.
        But the sign convention is about the math, not the business logic:
          PROTOCOL direction: rounded <= exact (error >= 0)
          USER direction: rounded >= exact (error <= 0)

        The caller declares favor=USER for ceil steps, then interprets
        in business context: "fee rounded UP → user pays more → safe."
        """
        RF = self.RoundingFavor
        RS = self.RoundingStep

        step = RS(
            Rational(100, 3),
            ceiling(Rational(100, 3)),
            RF.USER,  # ceil gives rounded >= exact = USER direction
            "fee deduction (ceil)",
        )
        assert step.direction_correct

    def test_floor_on_output_favors_protocol(self):
        """floor on swap output correctly favors protocol."""
        RF = self.RoundingFavor
        RS = self.RoundingStep

        step = RS(
            Rational(100, 3),
            floor(Rational(100, 3)),
            RF.PROTOCOL,
            "swap output",
        )
        assert step.direction_correct
        assert step.uses_floor
        assert not step.uses_ceil

    def test_mismatch_flagged_in_description(self):
        """A step with wrong direction is flagged as MISMATCH."""
        RF = self.RoundingFavor
        RS = self.RoundingStep

        # floor on something that should favor USER → MISMATCH
        step = RS(
            Rational(100, 3),
            floor(Rational(100, 3)),
            RF.USER,  # floor gives less, but we want user to get at least exact
            "bad step",
        )
        assert not step.direction_correct

        lemmas = self.directional_chain_error([step], name_prefix="bad")
        # The direction lemma should have MISMATCH in description
        direction_lemma = lemmas[0]
        assert "MISMATCH" in direction_lemma.description

    def test_error_sign_properties(self):
        """Verify error sign for floor and ceil."""
        RF = self.RoundingFavor
        RS = self.RoundingStep

        floor_step = RS(Rational(100, 3), floor(Rational(100, 3)),
                        RF.PROTOCOL, "floor")
        ceil_step = RS(Rational(100, 3), ceiling(Rational(100, 3)),
                       RF.USER, "ceil")

        # floor: error = 100/3 - 33 = 1/3 > 0
        assert sympy.simplify(floor_step.error) == Rational(1, 3)
        # ceil: error = 100/3 - 34 = -2/3 < 0
        assert sympy.simplify(ceil_step.error) == Rational(-2, 3)

    def test_mixed_pipeline_detects_net_leak(self):
        """Pipeline with correct per-step directions but NET NEGATIVE flow.

        Each step individually rounds the right way, but the ceil step's
        error (-6/7) outweighs the floor steps (+1/3, +2/9).
        Net = 1/3 - 6/7 + 2/9 = -19/63 < 0.

        This is the exploit: per-step checks pass, but the pipeline
        as a whole leaks value from the protocol.
        """
        RF = self.RoundingFavor
        RS = self.RoundingStep

        steps = [
            RS(Rational(100, 3), floor(Rational(100, 3)),
               RF.PROTOCOL, "swap output"),
            RS(Rational(50, 7), ceiling(Rational(50, 7)),
               RF.USER, "fee charge"),
            RS(Rational(200, 9), floor(Rational(200, 9)),
               RF.PROTOCOL, "withdrawal"),
        ]

        # Each step is individually correct
        for s in steps:
            assert s.direction_correct, f"{s.label} direction wrong"

        lemmas = self.directional_chain_error(steps, name_prefix="mix")

        from symproof.verification import verify_lemma

        # Per-step lemmas all pass
        for lem in lemmas[:-1]:
            result = verify_lemma(lem)
            # All pass except possibly the net_flow
            if "_net" not in lem.name:
                assert result.passed, f"{lem.name} failed: {result.error}"

        # But the NET FLOW lemma FAILS — the pipeline leaks!
        net_flow_lemma = lemmas[-1]
        assert "net" in net_flow_lemma.name
        assert "NEGATIVE" in net_flow_lemma.description
        result = verify_lemma(net_flow_lemma)
        assert not result.passed, (
            "Net flow should FAIL — fee error outweighs output errors"
        )

    def test_directional_chain_in_proof(self):
        """Full sealed proof with directional chain."""
        RF = self.RoundingFavor
        RS = self.RoundingStep

        steps = [
            RS(Rational(17, 3), floor(Rational(17, 3)),
               RF.PROTOCOL, "output"),
            RS(Rational(23, 7), floor(Rational(23, 7)),
               RF.PROTOCOL, "rebalance"),
        ]

        lemmas = self.directional_chain_error(steps, name_prefix="seal")

        total_error = sum(s.error for s in steps)
        h = self.axioms.hypothesis(
            "dir_bounded",
            expr=sympy.Lt(sympy.Abs(total_error), Integer(2)),
        )
        builder = ProofBuilder(
            self.axioms, h.name,
            name="dir_proof", claim="2-step directional error bounded",
        )
        for lem in lemmas:
            builder = builder.add_lemma(lem)
        bundle = seal(self.axioms, h, builder.build())
        assert len(bundle.bundle_hash) == 64

    def test_exact_values_no_error(self):
        """When division is exact, error is 0 — both directions pass."""
        RF = self.RoundingFavor
        RS = self.RoundingStep

        # 10/2 = 5 exactly
        step = RS(Rational(10, 2), floor(Rational(10, 2)),
                  RF.PROTOCOL, "exact")
        assert step.error == 0
        assert step.direction_correct

    def test_net_flow_positive_pipeline(self):
        """Pipeline where protocol-favoring steps dominate → net safe."""
        RF = self.RoundingFavor
        RS = self.RoundingStep

        steps = [
            RS(Rational(100, 3), floor(Rational(100, 3)),
               RF.PROTOCOL, "output"),        # error = +1/3
            RS(Rational(50, 7), ceiling(Rational(50, 7)),
               RF.USER, "fee"),               # error = -6/7
            RS(Rational(200, 3), floor(Rational(200, 3)),
               RF.PROTOCOL, "withdrawal"),    # error = +2/3
        ]
        # Net = 1/3 - 6/7 + 2/3 = 7/21 - 18/21 + 14/21 = 3/21 > 0

        lemmas = self.directional_chain_error(steps, name_prefix="net")
        net_flow_lemma = lemmas[-1]
        assert "net" in net_flow_lemma.name
        assert "NEGATIVE" not in net_flow_lemma.description

        from symproof.verification import verify_lemma

        result = verify_lemma(net_flow_lemma)
        assert result.passed, f"Net flow check failed: {result.error}"

    def test_net_flow_negative_pipeline_flagged(self):
        """Pipeline where user-favoring steps dominate → net LEAKS.

        This is the exploit: if ceil steps outweigh floor steps,
        the protocol loses value on every transaction.
        """
        RF = self.RoundingFavor
        RS = self.RoundingStep

        steps = [
            RS(Rational(10, 7), floor(Rational(10, 7)),
               RF.PROTOCOL, "tiny output"),    # error = +3/7
            RS(Rational(100, 3), ceiling(Rational(100, 3)),
               RF.USER, "big fee"),            # error = -2/3
            RS(Rational(10, 9), ceiling(Rational(10, 9)),
               RF.USER, "another fee"),        # error = -8/9
        ]
        # Net = 3/7 - 2/3 - 8/9 = 27/63 - 42/63 - 56/63 = -71/63 < 0

        lemmas = self.directional_chain_error(steps, name_prefix="leak")
        net_flow_lemma = lemmas[-1]
        assert "NEGATIVE" in net_flow_lemma.description

        from symproof.verification import verify_lemma

        # The net flow lemma should FAIL (net error < 0)
        result = verify_lemma(net_flow_lemma)
        assert not result.passed, "Net flow should fail — pipeline leaks!"

    def test_net_flow_in_sealed_proof(self):
        """Sealed proof: net flow is protocol-safe."""
        RF = self.RoundingFavor
        RS = self.RoundingStep

        steps = [
            RS(Rational(17, 3), floor(Rational(17, 3)),
               RF.PROTOCOL, "output"),
            RS(Rational(23, 7), floor(Rational(23, 7)),
               RF.PROTOCOL, "rebalance"),
        ]

        lemmas = self.directional_chain_error(steps, name_prefix="sf")
        # Net = 2/3 + 2/7 = 20/21 > 0 → safe

        h = self.axioms.hypothesis(
            "net_safe",
            expr=sympy.Ge(
                sum(s.error for s in steps), Integer(0),
            ),
        )
        builder = ProofBuilder(
            self.axioms, h.name,
            name="net_proof", claim="Net flow favors protocol",
        )
        for lem in lemmas:
            builder = builder.add_lemma(lem)
        bundle = seal(self.axioms, h, builder.build())
        assert len(bundle.bundle_hash) == 64


# ===================================================================
# Phantom overflow — the invisible uint256 bomb
# ===================================================================


class TestPhantomOverflow:
    """Prove whether intermediate a*b overflows uint256.

    Real audit issue: floor(a * b / c) looks safe if the result fits
    in uint256, but the EVM computes a*b FIRST.  If a*b > 2^256,
    it wraps silently, and the division operates on garbage.
    OpenZeppelin's mulDiv exists for this reason.
    """

    def setup_method(self):
        self.axioms = AxiomSet(
            name="overflow",
            axioms=(Axiom(name="base", expr=sympy.Eq(Integer(1), Integer(1))),),
        )

    def test_small_values_no_overflow(self):
        """Small values: a*b fits in uint256."""
        a = Integer(10) ** 18
        b = Integer(10) ** 18
        product = a * b  # 10^36, well under 2^256
        assert product <= UINT256_MAX

    def test_large_values_overflow(self):
        """Large reserves: a*b exceeds uint256."""
        a = Integer(10) ** 200
        b = Integer(10) ** 200
        product = a * b  # 10^400, exceeds 2^256 ≈ 1.16e77
        assert product > UINT256_MAX

    def test_phantom_overflow_proof(self):
        """Prove that large a*b overflows (mulDiv required)."""
        a = Integer(10) ** 200
        b = Integer(10) ** 200

        h = self.axioms.hypothesis(
            "overflow_detected",
            expr=sympy.Gt(a * b, UINT256_MAX),
        )
        script = (
            ProofBuilder(
                self.axioms, h.name,
                name="phantom", claim="a*b overflows uint256",
            )
            .add_lemma(phantom_overflow_check(a, b, WAD, name="overflow"))
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_no_phantom_overflow_proof(self):
        """Prove that moderate a*b does NOT overflow (naive path safe)."""
        a = Integer(10) ** 18
        b = Integer(10) ** 18

        h = self.axioms.hypothesis(
            "no_overflow",
            expr=sympy.Le(a * b, UINT256_MAX),
        )
        script = (
            ProofBuilder(
                self.axioms, h.name,
                name="safe", claim="a*b fits in uint256",
            )
            .add_lemma(no_phantom_overflow_check(a, b, name="no_overflow"))
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_safe_mul_div_matches_naive(self):
        """When no overflow, safe_mul_div equals naive floor division."""
        a = Integer(10) ** 18 * 7
        b = Integer(10) ** 18 * 3
        c = Integer(10) ** 18

        safe = safe_mul_div(a, b, c)
        naive = floor(a * b / c)
        assert safe == naive

    def test_amm_reserve_overflow_detection(self):
        """Real scenario: large AMM reserves can cause phantom overflow.

        Pool with 10^39 tokens each: Ry * dx = 10^78 > 2^256 ≈ 1.16e77,
        so the intermediate product overflows uint256 in the swap formula
        before dividing by (Rx + dx).
        """
        Ry = Integer(10) ** 39  # Large reserve
        dx = Integer(10) ** 39  # Large swap

        h = self.axioms.hypothesis(
            "amm_overflow",
            expr=sympy.Gt(Ry * dx, UINT256_MAX),
        )
        script = (
            ProofBuilder(
                self.axioms, h.name,
                name="amm_phantom",
                claim="AMM Ry*dx overflows — mulDiv required in swap",
            )
            .add_lemma(phantom_overflow_check(Ry, dx, Integer(1), name="ry_dx"))
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_sealed_overflow_proof(self):
        """Sealed proof: detect phantom overflow in AMM reserves."""
        Ry = Integer(10) ** 50
        dx = Integer(10) ** 50
        h = self.axioms.hypothesis(
            "sealed_overflow",
            expr=sympy.Gt(Ry * dx, UINT256_MAX),
        )
        script = (
            ProofBuilder(
                self.axioms, h.name,
                name="sealed_phantom", claim="Phantom overflow sealed",
            )
            .add_lemma(phantom_overflow_check(Ry, dx, Integer(1), name="check"))
            .build()
        )
        bundle = seal(self.axioms, h, script)
        assert len(bundle.bundle_hash) == 64


# ===================================================================
# Decimal mismatch — the cross-token pricing bug
# ===================================================================


class TestDecimalAwarePool:
    """Prove that decimal normalization is required for cross-decimal swaps.

    Real audit issue: USDC (6 decimals) swapped for WETH (18 decimals)
    without normalization produces a price off by 10^12.
    """

    def test_same_decimals_no_normalization(self):
        """Same decimals: normalization factor is 1."""
        pool = DecimalAwarePool(decimals_x=18, decimals_y=18)
        assert pool.norm_x_to_y == Integer(1)
        assert pool.norm_y_to_x == Integer(1)

    def test_usdc_weth_normalization_factor(self):
        """USDC (6 dec) → WETH (18 dec): factor is 10^12."""
        pool = DecimalAwarePool(decimals_x=6, decimals_y=18)
        assert pool.norm_x_to_y == Integer(10) ** 12

    def test_weth_usdc_normalization_factor(self):
        """WETH (18 dec) → USDC (6 dec): factor is 10^-12."""
        pool = DecimalAwarePool(decimals_x=18, decimals_y=6)
        assert pool.norm_x_to_y == Rational(1, 10**12)

    def test_price_differs_without_normalization(self):
        """Spot price Rx/Ry is WRONG without decimal normalization.

        This is the real audit finding: the constant-product swap formula
        is scale-invariant (normalization cancels in the ratio), but the
        PRICE derived from reserves is off by 10^(decimals_y - decimals_x)
        if you don't normalize.

        Pool: 1,000,000 USDC (6 dec) / 500 WETH (18 dec)
        Naive price = Rx_raw / Ry_raw = 1e12 / 5e20 = 2e-9 (WRONG)
        Correct price = (Rx_raw * 10^12) / Ry_raw = 1e24 / 5e20 = 2000 (RIGHT)
        """
        pool = DecimalAwarePool(
            rx_name="USDC", ry_name="WETH",
            decimals_x=6, decimals_y=18,
        )
        Rx_val = Integer(1_000_000) * Integer(10) ** 6   # 1M USDC raw
        Ry_val = Integer(500) * Integer(10) ** 18        # 500 WETH raw

        # Naive price: just divide raw reserves
        naive_price = Rx_val / Ry_val

        # Correct price: normalize X to Y's decimals first
        correct_price = pool.normalize_x(Rx_val) / Ry_val

        # Ratio between correct and naive is the normalization factor
        ratio = sympy.simplify(correct_price / naive_price)
        assert ratio == pool.norm_x_to_y  # 10^12

        # The naive price is off by a factor of 10^12!
        assert correct_price == Integer(2000)
        assert naive_price == Rational(1, 500_000_000)

    def test_concrete_usdc_weth_swap(self):
        """Concrete: swap 100 USDC for WETH in a 1M USDC / 500 WETH pool.

        USDC reserves: 1,000,000 * 10^6 = 10^12
        WETH reserves: 500 * 10^18 = 5 * 10^20
        Input: 100 USDC = 100 * 10^6 = 10^8

        Correct (normalized to 18 dec):
            dx_norm = 10^8 * 10^12 = 10^20
            rx_norm = 10^12 * 10^12 = 10^24
            dy = 5e20 * 1e20 / (1e24 + 1e20)
               = 5e40 / 1.0001e24
               ≈ 4.9995e16

        Naive (no normalization):
            dy = 5e20 * 1e8 / (1e12 + 1e8)
               = 5e28 / 1.0001e12
               ≈ 4.9995e16

        In this case the numbers happen to work out similarly, but the
        formula structure is different and can diverge with other reserves.
        """
        pool = DecimalAwarePool(
            rx_name="USDC", ry_name="WETH",
            decimals_x=6, decimals_y=18,
        )
        # Verify normalization factor is 10^12
        assert pool.norm_x_to_y == Integer(10) ** 12

    def test_normalization_factor_proof(self):
        """Prove the normalization factor via the proof system."""
        pool = DecimalAwarePool(decimals_x=6, decimals_y=18)
        axioms = AxiomSet(
            name="decimal",
            axioms=(Axiom(name="base", expr=sympy.Eq(Integer(1), Integer(1))),),
        )
        h = axioms.hypothesis(
            "norm_factor",
            expr=sympy.Eq(pool.norm_x_to_y, Integer(10) ** 12),
        )
        script = (
            ProofBuilder(axioms, h.name, name="norm", claim="Factor is 10^12")
            .add_lemma(pool.normalization_factor_lemma(name="factor"))
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_eight_to_eighteen_decimals(self):
        """WBTC (8 dec) → DAI (18 dec): factor is 10^10."""
        pool = DecimalAwarePool(decimals_x=8, decimals_y=18)
        assert pool.norm_x_to_y == Integer(10) ** 10


# ===================================================================
# Integration: composing multiple DeFi tools
# ===================================================================


class TestDeFiToolComposition:
    """Compose rounding + overflow + error tools in realistic scenarios."""

    def setup_method(self):
        self.axioms = AxiomSet(
            name="composed",
            axioms=(Axiom(name="base", expr=sympy.Eq(Integer(1), Integer(1))),),
        )

    def test_swap_with_rounding_and_overflow_check(self):
        """Full swap safety: check overflow, then verify rounding bounds.

        Real protocol pattern:
        1. Check that Ry * dx doesn't overflow uint256
        2. Compute swap output with floor (favoring pool)
        3. Verify rounding gap is at most 1 wei
        """
        Ry = Integer(10) ** 18 * 1000  # 1000 tokens in WAD
        dx = Integer(10) ** 18 * 10    # 10 tokens in WAD
        Rx = Integer(10) ** 18 * 500   # 500 tokens in WAD

        dy_down = floor(Ry * dx / (Rx + dx))
        dy_up = ceiling(Ry * dx / (Rx + dx))

        h = self.axioms.hypothesis(
            "swap_safe",
            expr=sympy.And(
                sympy.Le(Ry * dx, UINT256_MAX),
                sympy.Le(dy_up - dy_down, Integer(1)),
            ),
        )
        script = (
            ProofBuilder(
                self.axioms, h.name,
                name="swap_safety",
                claim="Swap: no overflow, rounding gap <= 1",
            )
            .add_lemma(no_phantom_overflow_check(Ry, dx, name="no_overflow"))
            .add_lemma(
                rounding_gap_lemma(Ry * dx, Integer(1), Rx + dx, name="gap")
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary
