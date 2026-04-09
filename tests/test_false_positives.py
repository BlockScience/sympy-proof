"""False-positive prevention battery.

Every test in this file asserts that a MATHEMATICALLY FALSE claim is
correctly REJECTED by the verification engine.  A failure here means
symproof accepted an invalid proof — the most critical class of bug.

Organized by verification path:
1. EQUALITY  — false algebraic claims
2. BOOLEAN   — false logical/relational claims
3. BOOLEAN refine/negation fallbacks — false implications
4. QUERY     — false Q-system claims
5. QUERY assumption edge cases
6. Axiom-assumption contradiction (seal path)
7. EQUALITY domain hazards (known SymPy behavior)
8. COORDINATE_TRANSFORM — false transform claims
9. End-to-end seal rejection
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
from symproof.models import Lemma
from symproof.tactics import auto_lemma, try_implication, try_simplify
from symproof.verification import verify_lemma


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _must_reject(lemma: Lemma) -> None:
    """Assert that verify_lemma rejects this lemma (passed=False)."""
    result = verify_lemma(lemma)
    assert not result.passed, (
        f"FALSE POSITIVE: lemma '{lemma.name}' was accepted as passed=True "
        f"(actual_value={result.actual_value})"
    )


# ===================================================================
# 1. EQUALITY: false algebraic claims
# ===================================================================


class TestEqualityFalseClaims:
    """EQUALITY path must reject claims where expr != expected."""

    def test_one_not_two(self):
        """1 != 2."""
        _must_reject(Lemma(
            name="1_eq_2", kind=LemmaKind.EQUALITY,
            expr=sympy.Integer(1), expected=sympy.Integer(2),
        ))

    def test_x_sq_not_x(self):
        """x^2 != x in general."""
        x = sympy.Symbol("x")
        _must_reject(Lemma(
            name="x_sq_eq_x", kind=LemmaKind.EQUALITY,
            expr=x**2, expected=x,
        ))

    def test_off_by_constant(self):
        """x + 1 != x."""
        x = sympy.Symbol("x")
        _must_reject(Lemma(
            name="x_plus_1", kind=LemmaKind.EQUALITY,
            expr=x + 1, expected=x,
        ))

    def test_sin_not_cos(self):
        """sin(x) != cos(x)."""
        x = sympy.Symbol("x")
        _must_reject(Lemma(
            name="sin_eq_cos", kind=LemmaKind.EQUALITY,
            expr=sympy.sin(x), expected=sympy.cos(x),
        ))

    def test_wrong_series_value(self):
        """Sum(1/2^k, k=0..inf) != 3 (correct is 2)."""
        k = sympy.Symbol("k", integer=True, nonnegative=True)
        _must_reject(Lemma(
            name="series_eq_3", kind=LemmaKind.EQUALITY,
            expr=sympy.Sum(sympy.Rational(1, 2) ** k, (k, 0, sympy.oo)),
            expected=sympy.Integer(3),
        ))

    def test_harmonic_not_finite(self):
        """Harmonic series diverges — not equal to any finite value."""
        k = sympy.Symbol("k", integer=True, positive=True)
        _must_reject(Lemma(
            name="harmonic_eq_42", kind=LemmaKind.EQUALITY,
            expr=sympy.Sum(1 / k, (k, 1, sympy.oo)),
            expected=sympy.Integer(42),
        ))

    def test_wrong_integral(self):
        """int_0^1 x^2 dx != 1/2 (correct is 1/3)."""
        x = sympy.Symbol("x")
        _must_reject(Lemma(
            name="integral_wrong", kind=LemmaKind.EQUALITY,
            expr=sympy.Integral(x**2, (x, 0, 1)),
            expected=sympy.Rational(1, 2),
        ))

    def test_wrong_derivative(self):
        """d/dx(x^3) != 2*x^2 (correct is 3*x^2)."""
        x = sympy.Symbol("x")
        _must_reject(Lemma(
            name="deriv_wrong", kind=LemmaKind.EQUALITY,
            expr=sympy.diff(x**3, x),
            expected=2 * x**2,
        ))


# ===================================================================
# 2. BOOLEAN: false logical/relational claims
# ===================================================================


class TestBooleanFalseClaims:
    """BOOLEAN path must reject expressions that are not true."""

    def test_literal_false(self):
        """sympy.false must not verify."""
        _must_reject(Lemma(
            name="literal_false", kind=LemmaKind.BOOLEAN,
            expr=sympy.false,
        ))

    def test_x_gt_x(self):
        """x > x is never true."""
        x = sympy.Symbol("x", real=True)
        _must_reject(Lemma(
            name="x_gt_x", kind=LemmaKind.BOOLEAN,
            expr=x > x,
        ))

    def test_numeric_falsehood(self):
        """1 > 2 is false."""
        _must_reject(Lemma(
            name="1_gt_2", kind=LemmaKind.BOOLEAN,
            expr=sympy.Integer(1) > sympy.Integer(2),
        ))

    def test_contradiction(self):
        """And(x > 0, x < 0) is never true."""
        x = sympy.Symbol("x", real=True)
        _must_reject(Lemma(
            name="contradiction", kind=LemmaKind.BOOLEAN,
            expr=sympy.And(x > 0, x < 0),
        ))

    def test_wrong_sign_with_assumption(self):
        """x < 0 must fail even with x assumed positive."""
        x = sympy.Symbol("x")
        _must_reject(Lemma(
            name="neg_with_pos_asm", kind=LemmaKind.BOOLEAN,
            expr=x < 0,
            assumptions={"x": {"positive": True}},
        ))

    def test_equality_falsehood(self):
        """Eq(0, 1) is false."""
        _must_reject(Lemma(
            name="0_eq_1", kind=LemmaKind.BOOLEAN,
            expr=sympy.Eq(sympy.Integer(0), sympy.Integer(1)),
        ))

    def test_x_ne_x(self):
        """Ne(x, x) is false for all x."""
        x = sympy.Symbol("x")
        _must_reject(Lemma(
            name="x_ne_x", kind=LemmaKind.BOOLEAN,
            expr=sympy.Ne(x, x),
        ))


# ===================================================================
# 3. BOOLEAN refine/negation fallbacks: false implications
# ===================================================================


class TestBooleanFalseImplications:
    """refine() and negation-check must not accept false implications."""

    def test_positive_does_not_imply_greater_than_one(self):
        """Implies(x > 0, x > 1) is false (x=0.5 counterexample)."""
        x = sympy.Symbol("x", real=True)
        _must_reject(Lemma(
            name="pos_implies_gt1", kind=LemmaKind.BOOLEAN,
            expr=sympy.Implies(x > 0, x > 1),
            assumptions={"x": {"positive": True}},
        ))

    def test_nonneg_does_not_imply_positive(self):
        """Implies(x >= 0, x > 0) is false (x=0 counterexample)."""
        x = sympy.Symbol("x", real=True)
        _must_reject(Lemma(
            name="nonneg_implies_pos", kind=LemmaKind.BOOLEAN,
            expr=sympy.Implies(x >= 0, x > 0),
        ))

    def test_bounded_does_not_imply_unbounded(self):
        """Implies(x > 0 and y > 0, x + y > 5) is false."""
        x = sympy.Symbol("x", real=True)
        y = sympy.Symbol("y", real=True)
        _must_reject(Lemma(
            name="bounded_to_unbounded", kind=LemmaKind.BOOLEAN,
            expr=sympy.Implies(
                sympy.And(x > 0, y > 0),
                x + y > 5,
            ),
        ))

    def test_non_sequitur_chain(self):
        """Implies(x > y and y > 0, x > 1) is false."""
        x = sympy.Symbol("x", real=True)
        y = sympy.Symbol("y", real=True)
        _must_reject(Lemma(
            name="non_sequitur", kind=LemmaKind.BOOLEAN,
            expr=sympy.Implies(
                sympy.And(x > y, y > 0),
                x > 1,
            ),
        ))

    def test_reverse_inequality(self):
        """Implies(x > y, y > x) is false."""
        x = sympy.Symbol("x", real=True)
        y = sympy.Symbol("y", real=True)
        _must_reject(Lemma(
            name="reverse_ineq", kind=LemmaKind.BOOLEAN,
            expr=sympy.Implies(x > y, y > x),
        ))

    def test_false_transitivity(self):
        """Implies(x > y and y > z, z > x) is false — wrong direction."""
        x = sympy.Symbol("x", real=True)
        y = sympy.Symbol("y", real=True)
        z = sympy.Symbol("z", real=True)
        _must_reject(Lemma(
            name="false_transitivity", kind=LemmaKind.BOOLEAN,
            expr=sympy.Implies(
                sympy.And(x > y, y > z),
                z > x,
            ),
        ))


# ===================================================================
# 4. Tactics must not produce false positives
# ===================================================================


class TestTacticsFalsePositives:
    """try_simplify, try_implication, auto_lemma must not accept false claims."""

    def test_try_simplify_rejects_false(self):
        """try_simplify must not return True for a false expression."""
        x = sympy.Symbol("x", real=True)
        assert try_simplify(x > x) is not True
        assert try_simplify(sympy.false) is not True
        assert try_simplify(sympy.Integer(1) > sympy.Integer(2)) is not True

    def test_try_implication_rejects_false(self):
        """try_implication must not return True for a false implication."""
        x = sympy.Symbol("x", real=True)
        assert try_implication(x > 0, x > 1) is not True
        assert try_implication(x >= 0, x > 0) is not True

    def test_auto_lemma_rejects_false_equality(self):
        """auto_lemma must return None for a false equality."""
        x = sympy.Symbol("x")
        lemma = auto_lemma("bad", x + 1, expected=x)
        assert lemma is None

    def test_auto_lemma_rejects_false_boolean(self):
        """auto_lemma on a false boolean must return None."""
        lemma = auto_lemma("bad", sympy.false)
        assert lemma is None

    def test_auto_lemma_rejects_false_query(self):
        """auto_lemma must not produce a lemma for Q.negative with positive."""
        x = sympy.Symbol("x")
        lemma = auto_lemma(
            "bad",
            sympy.Q.negative(x),
            assumptions={"x": {"positive": True}},
        )
        assert lemma is None


# ===================================================================
# 5. QUERY: false Q-system claims
# ===================================================================


class TestQueryFalseClaims:
    """QUERY path must reject false propositions."""

    def test_negative_with_positive_assumption(self):
        """Q.negative(x) must fail when x is assumed positive."""
        x = sympy.Symbol("x")
        _must_reject(Lemma(
            name="neg_pos", kind=LemmaKind.QUERY,
            expr=sympy.Q.negative(x),
            assumptions={"x": {"positive": True}},
        ))

    def test_positive_without_assumptions(self):
        """Q.positive(x) must fail without any assumptions (indeterminate)."""
        x = sympy.Symbol("x")
        _must_reject(Lemma(
            name="pos_no_asm", kind=LemmaKind.QUERY,
            expr=sympy.Q.positive(x),
            assumptions={},
        ))

    def test_integer_with_only_real(self):
        """Q.integer(x) must fail with only real assumption (underdetermined)."""
        x = sympy.Symbol("x")
        _must_reject(Lemma(
            name="int_from_real", kind=LemmaKind.QUERY,
            expr=sympy.Q.integer(x),
            assumptions={"x": {"real": True}},
        ))

    def test_zero_with_positive(self):
        """Q.zero(x) must fail when x is assumed positive."""
        x = sympy.Symbol("x")
        _must_reject(Lemma(
            name="zero_pos", kind=LemmaKind.QUERY,
            expr=sympy.Q.zero(x),
            assumptions={"x": {"positive": True}},
        ))

    def test_negative_product_of_positives(self):
        """Q.negative(x*y) must fail when both are positive."""
        x = sympy.Symbol("x")
        y = sympy.Symbol("y")
        _must_reject(Lemma(
            name="neg_product", kind=LemmaKind.QUERY,
            expr=sympy.Q.negative(x * y),
            assumptions={"x": {"positive": True}, "y": {"positive": True}},
        ))


# ===================================================================
# 6. QUERY assumption edge cases
# ===================================================================


class TestQueryAssumptionEdgeCases:
    """Assumption handling must not create false positives."""

    def test_false_valued_assumption_dropped(self):
        """positive=False must not make Q.positive pass."""
        x = sympy.Symbol("x")
        _must_reject(Lemma(
            name="false_asm", kind=LemmaKind.QUERY,
            expr=sympy.Q.positive(x),
            assumptions={"x": {"positive": False}},
        ))

    def test_made_up_predicate_dropped(self):
        """Unknown Q predicates are silently dropped — must not cause pass."""
        x = sympy.Symbol("x")
        _must_reject(Lemma(
            name="made_up", kind=LemmaKind.QUERY,
            expr=sympy.Q.positive(x),
            assumptions={"x": {"totally_made_up_property": True}},
        ))

    def test_empty_assumptions_no_free_pass(self):
        """Empty assumptions must not allow indeterminate claims to pass."""
        x = sympy.Symbol("x")
        _must_reject(Lemma(
            name="empty_asm", kind=LemmaKind.QUERY,
            expr=sympy.Q.positive(x),
            assumptions={},
        ))

    def test_wrong_symbol_assumption(self):
        """Assumptions on symbol 'y' must not affect queries about 'x'."""
        x = sympy.Symbol("x")
        _must_reject(Lemma(
            name="wrong_sym", kind=LemmaKind.QUERY,
            expr=sympy.Q.positive(x),
            assumptions={"y": {"positive": True}},
        ))


# ===================================================================
# 7. CRITICAL: Axiom-assumption contradiction (seal path)
# ===================================================================


class TestAxiomAssumptionContradiction:
    """seal() must reject proofs where lemma assumptions contradict axioms."""

    def test_positive_contradicts_negative_axiom(self):
        """Prove x > 0 under axiom x < 0 — must be rejected."""
        x = sympy.Symbol("x")
        axioms = AxiomSet(
            name="x_negative",
            axioms=(Axiom(name="x_neg", expr=x < 0),),
        )
        h = axioms.hypothesis("x_pos", expr=x > 0)
        script = (
            ProofBuilder(axioms, h.name, name="bad", claim="x > 0")
            .lemma(
                "pos",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(x),
                assumptions={"x": {"positive": True}},
            )
            .build()
        )
        with pytest.raises(ValueError, match="contradict"):
            seal(axioms, h, script)

    def test_positive_xy_contradicts_negative_x_axiom(self):
        """Prove x*y > 0 under axiom x < 0 with positive x assumption."""
        x = sympy.Symbol("x")
        y = sympy.Symbol("y")
        axioms = AxiomSet(
            name="x_negative",
            axioms=(Axiom(name="x_neg", expr=x < 0),),
        )
        h = axioms.hypothesis("xy_pos", expr=x * y > 0)
        script = (
            ProofBuilder(axioms, h.name, name="bad", claim="x*y > 0")
            .lemma(
                "product_pos",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(x * y),
                assumptions={"x": {"positive": True}, "y": {"positive": True}},
            )
            .build()
        )
        with pytest.raises(ValueError, match="contradict"):
            seal(axioms, h, script)

    def test_zero_contradicts_positive_axiom(self):
        """Prove x == 0 under axiom x > 0 with zero assumption."""
        x = sympy.Symbol("x")
        axioms = AxiomSet(
            name="x_positive",
            axioms=(Axiom(name="x_pos", expr=x > 0),),
        )
        h = axioms.hypothesis("x_zero", expr=sympy.Eq(x, 0))
        script = (
            ProofBuilder(axioms, h.name, name="bad", claim="x == 0")
            .lemma(
                "is_zero",
                LemmaKind.QUERY,
                expr=sympy.Q.zero(x),
                assumptions={"x": {"zero": True}},
            )
            .build()
        )
        with pytest.raises(ValueError, match="contradict"):
            seal(axioms, h, script)

    def test_boolean_with_contradicting_assumptions(self):
        """BOOLEAN lemma with assumptions contradicting axioms must fail."""
        x = sympy.Symbol("x")
        axioms = AxiomSet(
            name="x_negative",
            axioms=(Axiom(name="x_neg", expr=x < 0),),
        )
        h = axioms.hypothesis("x_nonneg", expr=x >= 0)
        script = (
            ProofBuilder(axioms, h.name, name="bad", claim="x >= 0")
            .lemma(
                "nonneg",
                LemmaKind.BOOLEAN,
                expr=x >= 0,
                assumptions={"x": {"nonnegative": True}},
            )
            .build()
        )
        with pytest.raises(ValueError, match="contradict"):
            seal(axioms, h, script)

    def test_compatible_assumptions_pass(self):
        """Compatible assumptions should NOT be rejected."""
        x = sympy.Symbol("x")
        axioms = AxiomSet(
            name="x_positive",
            axioms=(Axiom(name="x_pos", expr=x > 0),),
        )
        h = axioms.hypothesis("x_pos_query", expr=x > 0)
        script = (
            ProofBuilder(axioms, h.name, name="ok", claim="x > 0")
            .lemma(
                "pos",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(x),
                assumptions={"x": {"positive": True}},
            )
            .build()
        )
        # Should seal successfully — assumptions are compatible with axioms
        bundle = seal(axioms, h, script)
        assert bundle.bundle_hash

    def test_indeterminate_assumptions_allowed(self):
        """Assumptions that are indeterminate w.r.t. axioms are allowed."""
        x = sympy.Symbol("x")
        y = sympy.Symbol("y")
        axioms = AxiomSet(
            name="x_positive",
            axioms=(Axiom(name="x_pos", expr=x > 0),),
        )
        h = axioms.hypothesis("xy_pos", expr=x * y > 0)
        script = (
            ProofBuilder(axioms, h.name, name="ok", claim="x*y > 0")
            .lemma(
                "product",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(x * y),
                # y is positive — indeterminate w.r.t. axiom (which only says x > 0)
                assumptions={"x": {"positive": True}, "y": {"positive": True}},
            )
            .build()
        )
        bundle = seal(axioms, h, script)
        assert bundle.bundle_hash

    def test_multi_axiom_contradiction(self):
        """Contradiction against any one axiom in the set is enough."""
        x = sympy.Symbol("x")
        y = sympy.Symbol("y")
        axioms = AxiomSet(
            name="mixed",
            axioms=(
                Axiom(name="x_pos", expr=x > 0),
                Axiom(name="y_neg", expr=y < 0),
            ),
        )
        h = axioms.hypothesis("y_pos", expr=y > 0)
        script = (
            ProofBuilder(axioms, h.name, name="bad", claim="y > 0")
            .lemma(
                "pos_y",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(y),
                assumptions={"y": {"positive": True}},
            )
            .build()
        )
        with pytest.raises(ValueError, match="contradict"):
            seal(axioms, h, script)


# ===================================================================
# 8. EQUALITY domain hazards (known SymPy behavior)
# ===================================================================


class TestEqualityDomainHazards:
    """Document expressions where SymPy's simplify ignores domain restrictions.

    These are known SymPy behaviors, not symproof bugs.  We test them to
    ensure we understand the boundary and to detect if SymPy changes.
    """

    @pytest.mark.xfail(
        reason="SymPy simplify() cancels (x^2-1)/(x-1) to x+1, "
        "ignoring the domain restriction at x=1. Known SymPy behavior.",
        strict=True,
    )
    def test_rational_cancellation_domain_issue(self):
        """(x^2-1)/(x-1) simplifies to x+1, ignoring x=1 singularity."""
        x = sympy.Symbol("x")
        _must_reject(Lemma(
            name="domain_cancel", kind=LemmaKind.EQUALITY,
            expr=(x**2 - 1) / (x - 1),
            expected=x + 1,
        ))

    def test_log_product_without_assumptions_rejected(self):
        """log(x*y) != log(x)+log(y) without positive assumptions."""
        x = sympy.Symbol("x")
        y = sympy.Symbol("y")
        _must_reject(Lemma(
            name="log_no_asm", kind=LemmaKind.EQUALITY,
            expr=sympy.log(x * y),
            expected=sympy.log(x) + sympy.log(y),
        ))

    def test_log_product_with_positive_is_correct(self):
        """log(x*y) == log(x)+log(y) is valid for x,y > 0."""
        x = sympy.Symbol("x")
        y = sympy.Symbol("y")
        lemma = Lemma(
            name="log_positive", kind=LemmaKind.EQUALITY,
            expr=sympy.log(x * y),
            expected=sympy.log(x) + sympy.log(y),
            assumptions={"x": {"positive": True}, "y": {"positive": True}},
        )
        result = verify_lemma(lemma)
        assert result.passed, f"Should pass with positive assumptions: {result.error}"

    def test_sqrt_sq_without_assumption_rejected(self):
        """sqrt(x^2) != x without positive assumption (could be -x)."""
        x = sympy.Symbol("x")
        _must_reject(Lemma(
            name="sqrt_sq_no_asm", kind=LemmaKind.EQUALITY,
            expr=sympy.sqrt(x**2),
            expected=x,
        ))

    def test_sqrt_sq_with_positive_is_correct(self):
        """sqrt(x^2) == x is valid for x > 0."""
        x = sympy.Symbol("x")
        lemma = Lemma(
            name="sqrt_sq_pos", kind=LemmaKind.EQUALITY,
            expr=sympy.sqrt(x**2),
            expected=x,
            assumptions={"x": {"positive": True}},
        )
        result = verify_lemma(lemma)
        assert result.passed, f"Should pass with positive assumptions: {result.error}"


# ===================================================================
# 9. COORDINATE_TRANSFORM false claims
# ===================================================================


class TestCoordinateTransformFalseClaims:
    """COORDINATE_TRANSFORM must reject invalid transforms and claims."""

    def test_wrong_expected_value(self):
        """x1^2 + x2^2 should be r^2, not r."""
        r = sympy.Symbol("r", positive=True)
        theta = sympy.Symbol("theta", real=True)
        x1 = sympy.Symbol("x_1")
        x2 = sympy.Symbol("x_2")

        _must_reject(Lemma(
            name="wrong_polar", kind=LemmaKind.COORDINATE_TRANSFORM,
            expr=x1**2 + x2**2,
            expected=r,  # WRONG — should be r**2
            transform={"x_1": r * sympy.cos(theta), "x_2": r * sympy.sin(theta)},
            inverse_transform={
                "r": sympy.sqrt(x1**2 + x2**2),
                "theta": sympy.atan2(x2, x1),
            },
        ))

    def test_missing_transform(self):
        """Missing transform field must fail."""
        x = sympy.Symbol("x")
        _must_reject(Lemma(
            name="no_transform", kind=LemmaKind.COORDINATE_TRANSFORM,
            expr=x, expected=x,
            transform=None,
            inverse_transform={"x": x},
        ))

    def test_missing_inverse_transform(self):
        """Missing inverse_transform field must fail."""
        x = sympy.Symbol("x")
        _must_reject(Lemma(
            name="no_inverse", kind=LemmaKind.COORDINATE_TRANSFORM,
            expr=x, expected=x,
            transform={"x": x},
            inverse_transform=None,
        ))

    def test_missing_expected(self):
        """Missing expected field must fail."""
        x = sympy.Symbol("x")
        _must_reject(Lemma(
            name="no_expected", kind=LemmaKind.COORDINATE_TRANSFORM,
            expr=x, expected=None,
            transform={"x": x},
            inverse_transform={"x": x},
        ))


# ===================================================================
# 10. End-to-end: false proofs cannot be sealed
# ===================================================================


class TestSealRejectsFalseProofs:
    """seal() must reject proofs that contain any false lemma."""

    def test_seal_rejects_single_false_lemma(self):
        """One false lemma in a single-lemma proof → seal rejects."""
        x = sympy.Symbol("x")
        axioms = AxiomSet(
            name="basics", axioms=(Axiom(name="x_pos", expr=x > 0),),
        )
        h = axioms.hypothesis("bad_claim", expr=sympy.Eq(x, x + 1))
        script = (
            ProofBuilder(axioms, h.name, name="bad", claim="x == x+1")
            .lemma(
                "false_eq",
                LemmaKind.EQUALITY,
                expr=x,
                expected=x + 1,
            )
            .build()
        )
        with pytest.raises(ValueError, match="VERIFIED"):
            seal(axioms, h, script)

    def test_seal_rejects_multi_lemma_with_one_false(self):
        """True lemma followed by false lemma → seal rejects."""
        x = sympy.Symbol("x")
        k = sympy.Symbol("k", integer=True, nonnegative=True)
        axioms = AxiomSet(
            name="basics", axioms=(Axiom(name="x_pos", expr=x > 0),),
        )
        h = axioms.hypothesis("multi", expr=x > 0)
        script = (
            ProofBuilder(axioms, h.name, name="mixed", claim="mixed")
            .lemma(
                "true_eq",
                LemmaKind.EQUALITY,
                expr=sympy.Sum(sympy.Rational(1, 2) ** k, (k, 0, sympy.oo)),
                expected=sympy.Integer(2),
                description="This one is true",
            )
            .lemma(
                "false_eq",
                LemmaKind.EQUALITY,
                expr=x,
                expected=x + 1,
                depends_on=["true_eq"],
                description="This one is false",
            )
            .build()
        )
        with pytest.raises(ValueError, match="VERIFIED"):
            seal(axioms, h, script)

    def test_seal_rejects_target_mismatch(self):
        """Proof targeting wrong hypothesis → seal rejects."""
        x = sympy.Symbol("x")
        axioms = AxiomSet(
            name="basics", axioms=(Axiom(name="x_pos", expr=x > 0),),
        )
        h = axioms.hypothesis("the_real_target", expr=x > 0)
        script = (
            ProofBuilder(axioms, "wrong_target", name="p", claim="x > 0")
            .lemma(
                "pos",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(x),
                assumptions={"x": {"positive": True}},
            )
            .build()
        )
        with pytest.raises(ValueError, match="target"):
            seal(axioms, h, script)

    def test_seal_rejects_axiom_hash_mismatch(self):
        """Script bound to different axiom set → seal rejects."""
        x = sympy.Symbol("x")
        ax1 = AxiomSet(name="set1", axioms=(Axiom(name="a", expr=x > 0),))
        ax2 = AxiomSet(name="set2", axioms=(Axiom(name="b", expr=x > 0),))
        h = ax1.hypothesis("h", expr=x > 0)
        script = (
            ProofBuilder(ax2, h.name, name="p", claim="x > 0")
            .lemma(
                "pos",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(x),
                assumptions={"x": {"positive": True}},
            )
            .build()
        )
        with pytest.raises(ValueError, match="axiom_set_hash"):
            seal(ax1, h, script)
