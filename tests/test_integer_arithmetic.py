"""Integer and fixed-point arithmetic proof patterns.

Demonstrates how to use symproof for computational systems that operate
in integer or fixed-point math — EVM/Solidity WAD arithmetic, floor
division truncation bounds, modular identities, and overflow detection.

Key insight: SymPy works in exact real arithmetic, so proofs about integer
implementations require explicit modeling of ``floor()``, ``Mod()``, and
truncation error.  The advisory system flags when these operations appear,
reminding reviewers that algebraic cancellation may not hold under truncation.

Proof patterns demonstrated:
- **WAD arithmetic**: mulWad/divWad precision and error bounds
- **Truncation error**: bounding floor(a/b) vs a/b
- **Modular arithmetic**: division algorithm, congruence identities
- **Overflow detection**: uint256 bounds as axioms
- **AMM with integer math**: constant-product swap with floor division
"""

from __future__ import annotations

import sympy
from sympy import (
    Integer,
    Mod,
    Rational,
    Symbol,
    ceiling,
    floor,
    simplify,
    symbols,
)

from symproof import (
    Axiom,
    AxiomSet,
    LemmaKind,
    ProofBuilder,
    seal,
    verify_proof,
)
from symproof.models import Lemma
from symproof.verification import (
    _has_fixed_point_pattern,
    _has_floor_ceiling,
    _has_mod,
    verify_lemma,
)


# ===================================================================
# Constants
# ===================================================================

WAD = Integer(10) ** 18  # 18-decimal fixed-point scale (Solidity standard)
RAY = Integer(10) ** 27  # 27-decimal fixed-point scale (MakerDAO)
UINT256_MAX = Integer(2) ** 256 - 1


# ===================================================================
# WAD arithmetic — fixed-point multiplication and division
# ===================================================================


class TestWADArithmetic:
    """Prove properties of WAD-scaled fixed-point arithmetic.

    In Solidity, WAD math represents decimals as integers scaled by 10^18:
        mulWad(a, b) = floor(a * b / WAD)
        divWad(a, b) = floor(a * WAD / b)

    These proofs verify algebraic properties and bound truncation errors.
    """

    def setup_method(self):
        self.a = Symbol("a", positive=True, integer=True)
        self.b = Symbol("b", positive=True, integer=True)
        self.axioms = AxiomSet(
            name="wad_arithmetic",
            axioms=(
                Axiom(name="a_positive", expr=self.a > 0),
                Axiom(name="b_positive", expr=self.b > 0),
                Axiom(
                    name="wad_scale",
                    expr=sympy.Eq(Symbol("WAD"), WAD),
                    description="WAD = 10^18",
                ),
            ),
        )

    def test_mulwad_concrete_evaluation(self):
        """mulWad(2e18, 3e18) = floor(2e18 * 3e18 / 1e18) = 6e18.

        Concrete evaluation verifies the formula produces the expected
        integer result — no truncation error when inputs are WAD-aligned.
        """
        a_val = 2 * WAD  # 2.0 in WAD
        b_val = 3 * WAD  # 3.0 in WAD
        mulwad_result = floor(a_val * b_val / WAD)

        h = self.axioms.hypothesis(
            "mulwad_exact",
            expr=sympy.Eq(mulwad_result, 6 * WAD),
        )
        script = (
            ProofBuilder(
                self.axioms,
                h.name,
                name="mulwad_concrete",
                claim="mulWad(2e18, 3e18) = 6e18 exactly",
            )
            .lemma(
                "concrete_eval",
                LemmaKind.EQUALITY,
                expr=mulwad_result,
                expected=6 * WAD,
                description="floor(2e18 * 3e18 / 1e18) = 6e18",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_mulwad_truncation_concrete(self):
        """mulWad with non-aligned inputs loses precision.

        mulWad(1, 1) = floor(1 * 1 / 1e18) = floor(1e-18) = 0
        This demonstrates the precision floor of WAD arithmetic.
        """
        mulwad_one = floor(Integer(1) * Integer(1) / WAD)

        h = self.axioms.hypothesis(
            "mulwad_precision_floor",
            expr=sympy.Eq(mulwad_one, 0),
        )
        script = (
            ProofBuilder(
                self.axioms,
                h.name,
                name="mulwad_truncation",
                claim="mulWad(1, 1) = 0 — below precision floor",
            )
            .lemma(
                "precision_floor",
                LemmaKind.EQUALITY,
                expr=mulwad_one,
                expected=Integer(0),
                description="floor(1/1e18) = 0",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_divwad_concrete_evaluation(self):
        """divWad(6e18, 3e18) = floor(6e18 * 1e18 / 3e18) = 2e18.

        Division is the inverse of multiplication when inputs are aligned.
        """
        a_val = 6 * WAD
        b_val = 3 * WAD
        divwad_result = floor(a_val * WAD / b_val)

        h = self.axioms.hypothesis(
            "divwad_inverse",
            expr=sympy.Eq(divwad_result, 2 * WAD),
        )
        script = (
            ProofBuilder(
                self.axioms,
                h.name,
                name="divwad_concrete",
                claim="divWad(6e18, 3e18) = 2e18",
            )
            .lemma(
                "concrete_div",
                LemmaKind.EQUALITY,
                expr=divwad_result,
                expected=2 * WAD,
                description="floor(6e18 * 1e18 / 3e18) = 2e18",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_mulwad_divwad_roundtrip_exact(self):
        """divWad(mulWad(a, b), b) may not equal a due to truncation.

        But for WAD-aligned values it does: divWad(mulWad(5e18, 2e18), 2e18) = 5e18.
        """
        a_val = 5 * WAD
        b_val = 2 * WAD
        mulwad = floor(a_val * b_val / WAD)
        divwad_back = floor(mulwad * WAD / b_val)

        h = self.axioms.hypothesis(
            "roundtrip_exact",
            expr=sympy.Eq(divwad_back, a_val),
        )
        script = (
            ProofBuilder(
                self.axioms,
                h.name,
                name="roundtrip_proof",
                claim="divWad(mulWad(5e18, 2e18), 2e18) = 5e18",
            )
            .lemma(
                "mul_step",
                LemmaKind.EQUALITY,
                expr=mulwad,
                expected=10 * WAD,
                description="mulWad(5e18, 2e18) = 10e18",
            )
            .lemma(
                "div_step",
                LemmaKind.EQUALITY,
                expr=divwad_back,
                expected=a_val,
                depends_on=["mul_step"],
                description="divWad(10e18, 2e18) = 5e18",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_floor_integer_identity(self):
        """floor(n) = n when n is an integer — fundamental property.

        This is what makes integer-aligned WAD values exact:
        if a*b is divisible by WAD, floor(a*b/WAD) = a*b/WAD.
        """
        n = Symbol("n", integer=True)
        h = self.axioms.hypothesis(
            "floor_integer",
            expr=sympy.Eq(floor(n), n),
        )
        script = (
            ProofBuilder(
                self.axioms,
                h.name,
                name="floor_identity",
                claim="floor(integer) = integer",
            )
            .lemma(
                "floor_id",
                LemmaKind.EQUALITY,
                expr=floor(n),
                expected=n,
                assumptions={"n": {"integer": True}},
                description="floor(n) = n for integer n",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary


# ===================================================================
# Truncation error bounds
# ===================================================================


class TestTruncationBounds:
    """Prove error bounds for floor division truncation.

    Pattern: express the "ideal" real-valued result and the "actual"
    integer result, then bound the difference.
    """

    def setup_method(self):
        self.axioms = AxiomSet(
            name="truncation",
            axioms=(
                Axiom(name="base", expr=sympy.Eq(Integer(1), Integer(1))),
            ),
        )

    def test_floor_division_concrete_error(self):
        """Concrete truncation: floor(17/5) = 3, but 17/5 = 3.4.

        Error = 17/5 - floor(17/5) = 2/5 < 1.
        """
        exact = Rational(17, 5)
        truncated = floor(exact)
        error = exact - truncated

        h = self.axioms.hypothesis(
            "concrete_error",
            expr=sympy.And(
                sympy.Eq(truncated, Integer(3)),
                sympy.Eq(error, Rational(2, 5)),
            ),
        )
        script = (
            ProofBuilder(
                self.axioms,
                h.name,
                name="floor_error_proof",
                claim="floor(17/5) = 3, error = 2/5",
            )
            .lemma(
                "truncated_value",
                LemmaKind.EQUALITY,
                expr=truncated,
                expected=Integer(3),
                description="floor(17/5) = 3",
            )
            .lemma(
                "error_value",
                LemmaKind.EQUALITY,
                expr=error,
                expected=Rational(2, 5),
                depends_on=["truncated_value"],
                description="17/5 - 3 = 2/5",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_ceiling_floor_difference(self):
        """ceiling(x) - floor(x) is 0 for integers, 1 otherwise.

        Concrete: ceiling(17/5) - floor(17/5) = 4 - 3 = 1.
        """
        val = Rational(17, 5)
        diff = ceiling(val) - floor(val)

        h = self.axioms.hypothesis(
            "ceil_floor_gap",
            expr=sympy.Eq(diff, Integer(1)),
        )
        script = (
            ProofBuilder(
                self.axioms,
                h.name,
                name="ceil_floor_proof",
                claim="ceiling(17/5) - floor(17/5) = 1",
            )
            .lemma(
                "gap_is_one",
                LemmaKind.EQUALITY,
                expr=diff,
                expected=Integer(1),
                description="ceil - floor = 1 for non-integer",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_ceiling_floor_integer_case(self):
        """For an integer, ceiling = floor = value."""
        val = Integer(10)

        h = self.axioms.hypothesis(
            "ceil_floor_integer",
            expr=sympy.And(
                sympy.Eq(floor(val), val),
                sympy.Eq(ceiling(val), val),
            ),
        )
        script = (
            ProofBuilder(
                self.axioms,
                h.name,
                name="ceil_floor_int",
                claim="ceiling(10) = floor(10) = 10",
            )
            .lemma(
                "floor_eq",
                LemmaKind.EQUALITY,
                expr=floor(val),
                expected=val,
            )
            .lemma(
                "ceil_eq",
                LemmaKind.EQUALITY,
                expr=ceiling(val),
                expected=val,
                depends_on=["floor_eq"],
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary


# ===================================================================
# Modular arithmetic
# ===================================================================


class TestModularArithmetic:
    """Prove properties of modular (remainder) arithmetic.

    Models integer operations in systems like EVM where overflow wraps
    and remainder is used for slot packing, fee splitting, etc.
    """

    def setup_method(self):
        self.axioms = AxiomSet(
            name="modular",
            axioms=(
                Axiom(name="base", expr=sympy.Eq(Integer(1), Integer(1))),
            ),
        )

    def test_mod_concrete(self):
        """Mod(17, 5) = 2 — basic modular evaluation."""
        h = self.axioms.hypothesis(
            "mod_eval",
            expr=sympy.Eq(Mod(17, 5), Integer(2)),
        )
        script = (
            ProofBuilder(
                self.axioms,
                h.name,
                name="mod_concrete",
                claim="17 mod 5 = 2",
            )
            .lemma(
                "mod_17_5",
                LemmaKind.EQUALITY,
                expr=Mod(17, 5),
                expected=Integer(2),
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_division_algorithm_concrete(self):
        """Division algorithm: a = b*q + r where q = floor(a/b), r = a mod b.

        17 = 5 * 3 + 2.
        """
        a, b = Integer(17), Integer(5)
        q = floor(a / b)
        r = Mod(a, b)
        reconstruction = b * q + r

        h = self.axioms.hypothesis(
            "div_algo",
            expr=sympy.Eq(reconstruction, a),
        )
        script = (
            ProofBuilder(
                self.axioms,
                h.name,
                name="div_algorithm",
                claim="17 = 5 * floor(17/5) + 17 mod 5",
            )
            .lemma(
                "quotient",
                LemmaKind.EQUALITY,
                expr=q,
                expected=Integer(3),
                description="floor(17/5) = 3",
            )
            .lemma(
                "remainder",
                LemmaKind.EQUALITY,
                expr=r,
                expected=Integer(2),
                depends_on=["quotient"],
                description="17 mod 5 = 2",
            )
            .lemma(
                "reconstruction",
                LemmaKind.EQUALITY,
                expr=reconstruction,
                expected=a,
                depends_on=["quotient", "remainder"],
                description="5*3 + 2 = 17",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_mod_distributive_over_addition(self):
        """(a + b) mod n = ((a mod n) + (b mod n)) mod n.

        Concrete: (13 + 17) mod 7 = (13 mod 7 + 17 mod 7) mod 7 = (6 + 3) mod 7 = 2.
        """
        a, b, n = Integer(13), Integer(17), Integer(7)
        lhs = Mod(a + b, n)
        rhs = Mod(Mod(a, n) + Mod(b, n), n)

        h = self.axioms.hypothesis(
            "mod_add",
            expr=sympy.Eq(lhs, rhs),
        )
        script = (
            ProofBuilder(
                self.axioms,
                h.name,
                name="mod_add_proof",
                claim="(13+17) mod 7 = ((13 mod 7)+(17 mod 7)) mod 7",
            )
            .lemma(
                "lhs_eval",
                LemmaKind.EQUALITY,
                expr=lhs,
                expected=Integer(2),
                description="30 mod 7 = 2",
            )
            .lemma(
                "rhs_eval",
                LemmaKind.EQUALITY,
                expr=rhs,
                expected=Integer(2),
                depends_on=["lhs_eval"],
                description="(6 + 3) mod 7 = 2",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_mod_distributive_over_multiplication(self):
        """(a * b) mod n = ((a mod n) * (b mod n)) mod n.

        Concrete: (13 * 17) mod 7 = (6 * 3) mod 7 = 18 mod 7 = 4.
        """
        a, b, n = Integer(13), Integer(17), Integer(7)
        lhs = Mod(a * b, n)
        rhs = Mod(Mod(a, n) * Mod(b, n), n)

        h = self.axioms.hypothesis(
            "mod_mul",
            expr=sympy.Eq(lhs, rhs),
        )
        script = (
            ProofBuilder(
                self.axioms,
                h.name,
                name="mod_mul_proof",
                claim="(13*17) mod 7 = ((13 mod 7)*(17 mod 7)) mod 7",
            )
            .lemma(
                "lhs_eval",
                LemmaKind.EQUALITY,
                expr=lhs,
                expected=Integer(4),
            )
            .lemma(
                "rhs_eval",
                LemmaKind.EQUALITY,
                expr=rhs,
                expected=Integer(4),
                depends_on=["lhs_eval"],
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary


# ===================================================================
# Overflow detection — uint256 bounds
# ===================================================================


class TestOverflowBounds:
    """Model uint256 overflow as axiom-bounded proofs.

    Pattern: declare max-value axioms, then prove operations stay
    within bounds for specific inputs.  This is the proof analog of
    Solidity's checked arithmetic.
    """

    def setup_method(self):
        self.a = Symbol("a", nonnegative=True, integer=True)
        self.b = Symbol("b", nonnegative=True, integer=True)
        self.axioms = AxiomSet(
            name="uint256_bounds",
            axioms=(
                Axiom(
                    name="a_bounded",
                    expr=sympy.And(self.a >= 0, self.a <= UINT256_MAX),
                    description="a is a valid uint256",
                ),
                Axiom(
                    name="b_bounded",
                    expr=sympy.And(self.b >= 0, self.b <= UINT256_MAX),
                    description="b is a valid uint256",
                ),
            ),
        )

    def test_concrete_addition_safe(self):
        """Concrete: 2^255 + 2^255 = 2^256 which EXCEEDS uint256 max.

        This is a known overflow case. Proving the sum equals 2^256
        demonstrates the value is computable but violates the bound.
        """
        a_val = Integer(2) ** 255
        b_val = Integer(2) ** 255
        total = a_val + b_val

        h = self.axioms.hypothesis(
            "overflow_demo",
            expr=sympy.Eq(total, Integer(2) ** 256),
        )
        script = (
            ProofBuilder(
                self.axioms,
                h.name,
                name="overflow_detection",
                claim="2^255 + 2^255 = 2^256 (exceeds uint256)",
            )
            .lemma(
                "sum_value",
                LemmaKind.EQUALITY,
                expr=total,
                expected=Integer(2) ** 256,
                description="Sum computes to 2^256",
            )
            .lemma(
                "exceeds_max",
                LemmaKind.BOOLEAN,
                expr=sympy.Gt(total, UINT256_MAX),
                description="2^256 > 2^256 - 1",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_concrete_addition_safe_case(self):
        """Concrete: UINT256_MAX - 1 + 1 = UINT256_MAX (safe)."""
        a_val = UINT256_MAX - 1
        b_val = Integer(1)
        total = a_val + b_val

        h = self.axioms.hypothesis(
            "safe_add",
            expr=sympy.And(
                sympy.Eq(total, UINT256_MAX),
                sympy.Le(total, UINT256_MAX),
            ),
        )
        script = (
            ProofBuilder(
                self.axioms,
                h.name,
                name="safe_addition",
                claim="(2^256 - 2) + 1 = 2^256 - 1 (within bounds)",
            )
            .lemma(
                "sum_value",
                LemmaKind.EQUALITY,
                expr=total,
                expected=UINT256_MAX,
            )
            .lemma(
                "within_bounds",
                LemmaKind.BOOLEAN,
                expr=sympy.Le(total, UINT256_MAX),
                depends_on=["sum_value"],
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_modular_wraparound(self):
        """EVM unchecked arithmetic wraps: (UINT256_MAX + 1) mod 2^256 = 0."""
        overflow_val = UINT256_MAX + 1
        wrapped = Mod(overflow_val, Integer(2) ** 256)

        h = self.axioms.hypothesis(
            "wrap_to_zero",
            expr=sympy.Eq(wrapped, Integer(0)),
        )
        script = (
            ProofBuilder(
                self.axioms,
                h.name,
                name="wraparound_proof",
                claim="uint256 overflow wraps to 0",
            )
            .lemma(
                "mod_wrap",
                LemmaKind.EQUALITY,
                expr=wrapped,
                expected=Integer(0),
                description="(2^256 - 1 + 1) mod 2^256 = 0",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary


# ===================================================================
# AMM with integer math — constant-product swap
# ===================================================================


class TestAMMIntegerSwap:
    """AMM swap formula with floor division — modeling Solidity reality.

    The real-valued formula: dy = Ry * dx / (Rx + dx)
    The integer implementation: dy = floor(Ry * dx / (Rx + dx))

    Proving properties of both and bounding the truncation error.
    """

    def setup_method(self):
        self.Rx = Integer(1_000_000) * WAD  # 1M tokens in WAD
        self.Ry = Integer(2_000_000) * WAD  # 2M tokens in WAD
        self.dx = Integer(1_000) * WAD  # 1K token input
        self.axioms = AxiomSet(
            name="amm_integer",
            axioms=(
                Axiom(name="reserves_set", expr=sympy.Eq(Integer(1), Integer(1))),
            ),
        )

    def test_integer_swap_output(self):
        """Compute AMM output with floor division.

        dy_int = floor(Ry * dx / (Rx + dx))
        """
        Rx, Ry, dx = self.Rx, self.Ry, self.dx
        dy_exact = Rational(Ry * dx, Rx + dx)
        dy_int = floor(dy_exact)

        # For these WAD-aligned values, compute expected
        # Ry * dx / (Rx + dx) = 2e24 * 1e21 / (1e24 + 1e21)
        #                     = 2e45 / 1.001e24
        #                     = 1998001998001998001998 (integer)
        expected_dy = dy_exact  # it's already a Rational

        h = self.axioms.hypothesis(
            "swap_output",
            expr=sympy.Eq(dy_int, floor(expected_dy)),
        )
        script = (
            ProofBuilder(
                self.axioms,
                h.name,
                name="amm_int_swap",
                claim="Integer AMM swap output matches floor of exact",
            )
            .lemma(
                "dy_matches",
                LemmaKind.EQUALITY,
                expr=dy_int,
                expected=floor(expected_dy),
                description="floor(Ry*dx/(Rx+dx)) computed correctly",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_product_nondecreasing_integer(self):
        """Post-swap product >= pre-swap product even with floor division.

        k_pre = Rx * Ry
        k_post = (Rx + dx) * (Ry - dy_int)

        Since dy_int = floor(exact_dy) <= exact_dy, the post-swap
        product is at least as large as it would be with exact division,
        and exact division preserves the product exactly (no fee case).
        """
        Rx, Ry, dx = self.Rx, self.Ry, self.dx
        dy_int = floor(Rational(Ry * dx, Rx + dx))

        k_pre = Rx * Ry
        k_post = (Rx + dx) * (Ry - dy_int)

        h = self.axioms.hypothesis(
            "product_nondecreasing",
            expr=sympy.Ge(k_post, k_pre),
        )
        script = (
            ProofBuilder(
                self.axioms,
                h.name,
                name="product_growth",
                claim="Floor division makes product grow (truncation favors LPs)",
            )
            .lemma(
                "k_pre_value",
                LemmaKind.EQUALITY,
                expr=k_pre,
                expected=Rx * Ry,
                description="Pre-swap product",
            )
            .lemma(
                "k_post_ge_k_pre",
                LemmaKind.BOOLEAN,
                expr=sympy.Ge(k_post, k_pre),
                depends_on=["k_pre_value"],
                description="Post-swap product >= pre-swap product",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_sealed_integer_swap(self):
        """Sealed proof: integer AMM swap preserves product invariant."""
        Rx, Ry, dx = self.Rx, self.Ry, self.dx
        dy_int = floor(Rational(Ry * dx, Rx + dx))
        k_post = (Rx + dx) * (Ry - dy_int)
        k_pre = Rx * Ry

        h = self.axioms.hypothesis(
            "integer_swap_safe",
            expr=sympy.Ge(k_post, k_pre),
        )
        script = (
            ProofBuilder(
                self.axioms,
                h.name,
                name="integer_swap_sealed",
                claim="Integer AMM swap is safe: product nondecreasing",
            )
            .lemma(
                "output_computed",
                LemmaKind.EQUALITY,
                expr=dy_int,
                expected=floor(Rational(Ry * dx, Rx + dx)),
                description="Integer swap output via floor division",
            )
            .lemma(
                "invariant_holds",
                LemmaKind.BOOLEAN,
                expr=sympy.Ge(k_post, k_pre),
                depends_on=["output_computed"],
                description="Product nondecreasing under floor truncation",
            )
            .build()
        )
        bundle = seal(self.axioms, h, script)
        assert len(bundle.bundle_hash) == 64


# ===================================================================
# Advisory detection for integer operations
# ===================================================================


class TestIntegerAdvisories:
    """Verify that integer/fixed-point operations trigger advisories."""

    def test_floor_triggers_truncation_advisory(self):
        """Symbolic floor() expression should carry truncation advisory.

        floor(x) stays unevaluated when x is real (not integer).
        We verify floor(x) == floor(x) — trivially true but the floor
        node survives in both expr and expected, triggering the advisory.
        """
        x = Symbol("x", positive=True)
        lemma = Lemma(
            name="floor_sym",
            kind=LemmaKind.EQUALITY,
            expr=floor(x),
            expected=floor(x),
        )
        result = verify_lemma(lemma)
        assert result.passed
        assert any("truncation" in a.lower() for a in result.advisories), (
            f"Expected truncation advisory, got: {result.advisories}"
        )

    def test_ceiling_triggers_truncation_advisory(self):
        """Symbolic ceiling() expression should carry truncation advisory."""
        x = Symbol("x", positive=True)
        lemma = Lemma(
            name="ceil_sym",
            kind=LemmaKind.EQUALITY,
            expr=ceiling(x),
            expected=ceiling(x),
        )
        result = verify_lemma(lemma)
        assert result.passed
        assert any("truncation" in a.lower() for a in result.advisories)

    def test_mod_triggers_advisory(self):
        """Symbolic Mod() in expr should carry modular arithmetic advisory.

        Note: Mod(17, 5) eagerly evaluates to 2 (no Mod node).
        Advisories fire on symbolic Mod where the result is not known.
        """
        x = Symbol("x", positive=True, integer=True)
        n = Symbol("n", positive=True, integer=True)
        # Mod(x, n) stays symbolic; test it equals itself
        lemma = Lemma(
            name="mod_sym",
            kind=LemmaKind.EQUALITY,
            expr=Mod(x, n),
            expected=Mod(x, n),
        )
        result = verify_lemma(lemma)
        assert result.passed
        assert any("modular" in a.lower() for a in result.advisories), (
            f"Expected modular advisory, got: {result.advisories}"
        )

    def test_fixed_point_pattern_triggers_advisory(self):
        """floor(a*b/WAD) should trigger fixed-point scaling advisory."""
        a = Symbol("a", positive=True, integer=True)
        b = Symbol("b", positive=True, integer=True)
        mulwad = floor(a * b / WAD)
        # mulwad = mulwad is trivially true and keeps the floor node
        lemma = Lemma(
            name="mulwad_sym",
            kind=LemmaKind.EQUALITY,
            expr=mulwad,
            expected=mulwad,
        )
        result = verify_lemma(lemma)
        assert result.passed
        assert any("fixed-point" in a.lower() for a in result.advisories), (
            f"Expected fixed-point advisory, got: {result.advisories}"
        )

    def test_concrete_floor_no_advisory(self):
        """Concrete floor(17/5) evaluates to 3 — no floor node remains.

        When SymPy can compute the exact result, there is no truncation
        concern, so no advisory is expected.
        """
        lemma = Lemma(
            name="concrete",
            kind=LemmaKind.EQUALITY,
            expr=floor(Rational(17, 5)),
            expected=Integer(3),
        )
        result = verify_lemma(lemma)
        assert result.passed
        # floor(17/5) = 3 at construction time — no floor node, no advisory
        assert not any("truncation" in a.lower() for a in result.advisories)

    def test_plain_integer_no_advisory(self):
        """Simple integer arithmetic without floor/Mod has no integer advisory."""
        lemma = Lemma(
            name="plain",
            kind=LemmaKind.EQUALITY,
            expr=Integer(2) + Integer(3),
            expected=Integer(5),
        )
        result = verify_lemma(lemma)
        assert result.passed
        assert not any("truncation" in a.lower() for a in result.advisories)
        assert not any("modular" in a.lower() for a in result.advisories)

    def test_proof_level_advisory_aggregation(self):
        """ProofResult should aggregate integer advisories with lemma prefixes."""
        x = Symbol("x", positive=True)  # real, so floor(x) stays symbolic
        y = Symbol("y", positive=True, integer=True)
        m = Symbol("m", positive=True, integer=True)
        axioms = AxiomSet(
            name="test",
            axioms=(Axiom(name="t", expr=sympy.Eq(Integer(1), Integer(1))),),
        )
        script = (
            ProofBuilder(axioms, "target", name="test", claim="test")
            .lemma(
                "floor_step",
                LemmaKind.EQUALITY,
                expr=floor(x),
                expected=floor(x),
            )
            .lemma(
                "mod_step",
                LemmaKind.EQUALITY,
                expr=Mod(y, m),
                expected=Mod(y, m),
                depends_on=["floor_step"],
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED"
        # Should have advisories from both lemmas
        floor_advisories = [a for a in result.advisories if "[floor_step]" in a]
        mod_advisories = [a for a in result.advisories if "[mod_step]" in a]
        assert len(floor_advisories) > 0, "Missing floor_step advisories"
        assert len(mod_advisories) > 0, "Missing mod_step advisories"


# ===================================================================
# Detection helper unit tests
# ===================================================================


class TestDetectionHelpers:
    """Unit tests for the integer operation detection functions."""

    def test_has_floor_ceiling_floor(self):
        x = Symbol("x")
        assert _has_floor_ceiling(floor(x)) is True

    def test_has_floor_ceiling_ceiling(self):
        x = Symbol("x")
        assert _has_floor_ceiling(ceiling(x)) is True

    def test_has_floor_ceiling_nested(self):
        x = Symbol("x")
        assert _has_floor_ceiling(floor(x) + 1) is True

    def test_has_floor_ceiling_absent(self):
        x = Symbol("x")
        assert _has_floor_ceiling(x + 1) is False

    def test_has_mod_present(self):
        """Symbolic Mod stays unevaluated and is detectable."""
        x = Symbol("x", integer=True)
        assert _has_mod(Mod(x, 5)) is True

    def test_has_mod_nested(self):
        x = Symbol("x", integer=True)
        assert _has_mod(Mod(x, 5) + 3) is True

    def test_has_mod_absent(self):
        x = Symbol("x")
        assert _has_mod(x * 5) is False

    def test_has_fixed_point_pattern(self):
        a = Symbol("a", positive=True, integer=True)
        b = Symbol("b", positive=True, integer=True)
        assert _has_fixed_point_pattern(floor(a * b / WAD)) is True

    def test_has_fixed_point_pattern_plain_floor(self):
        """floor(x) alone is NOT a fixed-point pattern."""
        x = Symbol("x")
        assert _has_fixed_point_pattern(floor(x)) is False

    def test_has_fixed_point_pattern_absent(self):
        x = Symbol("x")
        assert _has_fixed_point_pattern(x + 1) is False
