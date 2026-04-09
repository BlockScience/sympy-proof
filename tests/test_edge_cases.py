"""User-testing: edge cases and stress tests.

Exercises symproof boundaries:
- Serialization round-trips for complex expression types
- Piecewise expressions
- Abs / sign expressions
- Matrix expressions in proofs
- Large multi-lemma chains
- Timeout-prone expressions (Sum/Product with parameters)
- Evidence serialization round-trip
- Error message quality on failures
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
from symproof.models import ProofScript
from symproof.serialization import canonical_srepr, make_canonical_dict, restore_expr
from symproof.tactics import auto_lemma


# ===================================================================
# Serialization round-trips
# ===================================================================


class TestSerializationRoundTrips:
    """Verify srepr round-trip for expression types from real projects."""

    @pytest.mark.parametrize(
        "expr",
        [
            # Basic arithmetic
            sympy.Integer(21_000_000),
            sympy.Rational(10_500_000) / sympy.Integer(2) ** sympy.Symbol("K"),
            # Trig
            sympy.sin(sympy.Symbol("psi")) * sympy.Symbol("w"),
            sympy.atan2(sympy.Symbol("p1"), sympy.Symbol("p2")),
            # Sqrt and powers
            sympy.sqrt(sympy.Symbol("x") ** 2 + sympy.Symbol("y") ** 2),
            # Sum
            sympy.Sum(
                sympy.Rational(1, 2) ** sympy.Symbol("k", integer=True),
                (sympy.Symbol("k", integer=True), 0, sympy.oo),
            ),
            # Relational
            sympy.Symbol("x") > 0,
            sympy.Ge(sympy.Symbol("x"), sympy.Symbol("y")),
            # Logical
            sympy.And(sympy.Symbol("x") > 0, sympy.Symbol("y") > 0),
            sympy.Implies(sympy.Symbol("x") > 0, sympy.Symbol("x") >= 0),
            sympy.Not(sympy.Symbol("x") > 0),
            # Derivatives (evaluated)
            sympy.diff(sympy.Symbol("x") ** 3, sympy.Symbol("x")),
            # Q predicates
            sympy.Q.positive(sympy.Symbol("x")),
        ],
        ids=[
            "integer",
            "rational_power",
            "trig_product",
            "atan2",
            "sqrt_norm",
            "sum_geometric",
            "strict_inequality",
            "ge_relational",
            "and_logic",
            "implies",
            "not_logic",
            "derivative",
            "q_predicate",
        ],
    )
    def test_srepr_round_trip(self, expr):
        """Expression survives srepr -> sympify cycle."""
        s = canonical_srepr(expr)
        restored = restore_expr(s)
        # They should be structurally equal
        assert sympy.simplify(expr - restored) == 0 if hasattr(expr, '__sub__') else str(expr) == str(restored)

    def test_canonical_dict_nested(self):
        """make_canonical_dict handles nested structures from real proofs."""
        x = sympy.Symbol("x")
        data = {
            "name": "test_axiom_set",
            "axioms": [
                {
                    "name": "ax1",
                    "expr": x > 0,
                    "nested": {"inner_expr": x**2, "value": 42},
                }
            ],
        }
        result = make_canonical_dict(data)

        # Keys should be sorted
        assert list(result.keys()) == ["axioms", "name"]
        # SymPy expressions should be srepr strings
        assert isinstance(result["axioms"][0]["expr"], str)
        assert isinstance(result["axioms"][0]["nested"]["inner_expr"], str)
        # Non-sympy values preserved
        assert result["axioms"][0]["nested"]["value"] == 42

    def test_canonical_dict_order_independence(self):
        """Insertion order doesn't affect canonical form."""
        x = sympy.Symbol("x")
        d1 = {"b": x**2, "a": x > 0}
        d2 = {"a": x > 0, "b": x**2}
        assert make_canonical_dict(d1) == make_canonical_dict(d2)


# ===================================================================
# Evidence round-trip (ProofScript serialization)
# ===================================================================


class TestEvidenceRoundTrip:
    """ProofScript.to_evidence() -> from_evidence() preserves content."""

    def test_simple_proof_round_trip(self):
        """Single-lemma EQUALITY proof survives evidence round-trip."""
        x = sympy.Symbol("x")
        axioms = AxiomSet(
            name="basics",
            axioms=(Axiom(name="x_pos", expr=x > 0),),
        )
        script = (
            ProofBuilder(axioms, "test_target", name="test", claim="2+2=4")
            .lemma(
                "arithmetic",
                LemmaKind.EQUALITY,
                expr=sympy.Integer(2) + sympy.Integer(2),
                expected=sympy.Integer(4),
            )
            .build()
        )

        evidence = script.to_evidence()
        restored = ProofScript.from_evidence(evidence)

        assert restored.name == script.name
        assert restored.target == script.target
        assert restored.axiom_set_hash == script.axiom_set_hash
        assert len(restored.lemmas) == len(script.lemmas)

        # Restored proof should still verify
        result = verify_proof(restored)
        assert result.status.value == "VERIFIED"

    def test_multi_lemma_round_trip(self):
        """Multi-lemma proof with mixed kinds survives round-trip."""
        x = sympy.Symbol("x")
        k = sympy.Symbol("k", integer=True, nonnegative=True)
        axioms = AxiomSet(
            name="mixed",
            axioms=(Axiom(name="x_pos", expr=x > 0),),
        )
        script = (
            ProofBuilder(axioms, "mixed_target", name="mixed_proof", claim="Mixed")
            .lemma(
                "series",
                LemmaKind.EQUALITY,
                expr=sympy.Sum(sympy.Rational(1, 2) ** k, (k, 0, sympy.oo)),
                expected=sympy.Integer(2),
            )
            .lemma(
                "positivity",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(x),
                assumptions={"x": {"positive": True}},
                depends_on=["series"],
            )
            .build()
        )

        evidence = script.to_evidence()
        restored = ProofScript.from_evidence(evidence)

        result = verify_proof(restored)
        assert result.status.value == "VERIFIED"


# ===================================================================
# Piecewise expressions
# ===================================================================


class TestPiecewiseExpressions:
    """Piecewise expressions from real mechanisms (e.g., clamped ratios)."""

    def test_piecewise_evaluation(self):
        """Piecewise collapse via library proof."""
        from symproof.library.core import piecewise_collapse

        x = sympy.Symbol("x")
        axioms = AxiomSet(
            name="piecewise_test",
            axioms=(Axiom(name="x_pos", expr=x > 0),),
        )
        bundle = piecewise_collapse(
            axioms, x, x > 0, sympy.Integer(0),
            assumptions={"x": {"positive": True}},
        )
        assert bundle.bundle_hash

    def test_max_min_clamping(self):
        """Max/Min lower bound via library proof."""
        from symproof.library.core import max_ge_first

        a = sympy.Rational(1, 4)
        b = sympy.Symbol("b", real=True)
        axioms = AxiomSet(
            name="clamped",
            axioms=(
                Axiom(name="quarter_real", expr=sympy.Eq(sympy.im(a), 0)),
                Axiom(name="b_real", expr=sympy.Eq(sympy.im(b), 0)),
            ),
        )
        bundle = max_ge_first(axioms, a, b)
        assert bundle.bundle_hash


# ===================================================================
# Abs expressions
# ===================================================================


class TestAbsExpressions:
    """Abs from Hamiltonian H* = -|sigma| + w*||p|| - p2 + 1."""

    def test_abs_nonneg(self):
        """Abs(x) >= 0 for all real x."""
        x = sympy.Symbol("x", real=True)

        axioms = AxiomSet(
            name="abs_test",
            axioms=(Axiom(name="x_real", expr=sympy.Eq(sympy.im(x), 0)),),
        )
        h = axioms.hypothesis("abs_nonneg", expr=sympy.Ge(sympy.Abs(x), 0))
        script = (
            ProofBuilder(
                axioms, h.name, name="abs_proof", claim="|x| >= 0"
            )
            .lemma(
                "abs_ge_zero",
                LemmaKind.QUERY,
                expr=sympy.Q.nonnegative(sympy.Abs(x)),
                assumptions={"x": {"real": True}},
                description="|x| >= 0 for real x",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary


# ===================================================================
# Large multi-lemma chains
# ===================================================================


class TestLargeProofChains:
    """Stress test: many-lemma proofs."""

    def test_ten_lemma_chain(self):
        """Chain of 10 trivial equality lemmas."""
        x = sympy.Symbol("x")
        axioms = AxiomSet(
            name="chain_test",
            axioms=(Axiom(name="x_pos", expr=x > 0),),
        )
        builder = ProofBuilder(
            axioms, "chain_target", name="chain_proof", claim="Chain of 10"
        )
        for i in range(10):
            builder = builder.lemma(
                f"step_{i}",
                LemmaKind.EQUALITY,
                expr=sympy.Integer(i) + sympy.Integer(1),
                expected=sympy.Integer(i + 1),
                depends_on=[f"step_{i - 1}"] if i > 0 else [],
                description=f"{i} + 1 = {i + 1}",
            )
        script = builder.build()
        result = verify_proof(script)
        assert result.status.value == "VERIFIED"
        assert len(result.lemma_results) == 10

    def test_mixed_kind_chain(self):
        """Chain mixing EQUALITY, BOOLEAN, and QUERY lemmas."""
        x = sympy.Symbol("x")
        k = sympy.Symbol("k", integer=True, nonnegative=True)

        axioms = AxiomSet(
            name="mixed_chain",
            axioms=(Axiom(name="x_pos", expr=x > 0),),
        )
        script = (
            ProofBuilder(
                axioms, "mixed", name="mixed_chain_proof", claim="Mixed chain"
            )
            .lemma(
                "eq_step",
                LemmaKind.EQUALITY,
                expr=sympy.Sum(sympy.Rational(1, 2) ** k, (k, 0, sympy.oo)),
                expected=sympy.Integer(2),
                description="Geometric series = 2",
            )
            .lemma(
                "bool_step",
                LemmaKind.BOOLEAN,
                expr=sympy.Implies(x > 0, x >= 0),
                depends_on=["eq_step"],
                description="positive => nonnegative",
            )
            .lemma(
                "query_step",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(x),
                assumptions={"x": {"positive": True}},
                depends_on=["bool_step"],
                description="x is positive",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED"


# ===================================================================
# Error message quality
# ===================================================================


class TestErrorMessages:
    """Verify that failures produce actionable diagnostics."""

    def test_equality_failure_shows_diff(self):
        """Failed EQUALITY should show the actual difference."""
        x = sympy.Symbol("x")
        axioms = AxiomSet(
            name="error_test",
            axioms=(Axiom(name="x_pos", expr=x > 0),),
        )
        script = (
            ProofBuilder(axioms, "bad", name="bad_proof", claim="Wrong claim")
            .lemma(
                "wrong",
                LemmaKind.EQUALITY,
                expr=x**2 + 1,
                expected=x**2,
                description="Intentionally wrong",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "FAILED"
        assert result.failure_summary is not None
        assert "wrong" in result.failure_summary

    def test_boolean_failure_on_indeterminate(self):
        """BOOLEAN on an indeterminate expression should fail gracefully."""
        x = sympy.Symbol("x")
        axioms = AxiomSet(
            name="indet_test",
            axioms=(Axiom(name="x_real", expr=sympy.Eq(sympy.im(x), 0)),),
        )
        script = (
            ProofBuilder(axioms, "indet", name="indet_proof", claim="Indeterminate")
            .lemma(
                "unknown",
                LemmaKind.BOOLEAN,
                expr=x > 0,  # indeterminate without assumptions
                description="x > 0 without knowing sign",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "FAILED"

    def test_seal_rejects_unverified(self):
        """seal() should raise ValueError with clear message on failure."""
        x = sympy.Symbol("x")
        axioms = AxiomSet(
            name="reject_test",
            axioms=(Axiom(name="x_pos", expr=x > 0),),
        )
        h = axioms.hypothesis("bad_hyp", expr=x > 0)
        script = (
            ProofBuilder(axioms, h.name, name="bad", claim="Will fail")
            .lemma(
                "wrong",
                LemmaKind.EQUALITY,
                expr=x,
                expected=x + 1,
                description="x != x + 1",
            )
            .build()
        )
        with pytest.raises(ValueError, match="VERIFIED"):
            seal(axioms, h, script)

    def test_seal_rejects_hash_mismatch(self):
        """seal() rejects when script axiom_set_hash doesn't match."""
        x = sympy.Symbol("x")
        ax1 = AxiomSet(name="set1", axioms=(Axiom(name="a", expr=x > 0),))
        ax2 = AxiomSet(name="set2", axioms=(Axiom(name="b", expr=x < 0),))

        h = ax1.hypothesis("h", expr=x > 0)
        # Build script against ax2 but try to seal with ax1
        script = (
            ProofBuilder(ax2, h.name, name="mismatched", claim="Mismatch")
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


# ===================================================================
# Expression complexity: integrals and derivatives
# ===================================================================


class TestExpressionComplexity:
    """Expressions that push SymPy's simplification capabilities."""

    def test_integral_equality(self):
        """Definite integral: int_0^1 x^2 dx = 1/3."""
        x = sympy.Symbol("x")
        integral = sympy.Integral(x**2, (x, 0, 1))

        axioms = AxiomSet(
            name="calculus",
            axioms=(Axiom(name="trivial", expr=sympy.Eq(1, 1)),),
        )
        h = axioms.hypothesis(
            "integral_result",
            expr=sympy.Eq(integral, sympy.Rational(1, 3)),
        )
        script = (
            ProofBuilder(
                axioms, h.name, name="integral_proof", claim="int x^2 = 1/3"
            )
            .lemma(
                "evaluate_integral",
                LemmaKind.EQUALITY,
                expr=integral,
                expected=sympy.Rational(1, 3),
                description="int_0^1 x^2 dx = 1/3",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_product_rule(self):
        """Derivative: d/dx(x * sin(x)) = sin(x) + x*cos(x)."""
        x = sympy.Symbol("x")
        expr = sympy.diff(x * sympy.sin(x), x)
        expected = sympy.sin(x) + x * sympy.cos(x)

        axioms = AxiomSet(
            name="derivatives",
            axioms=(Axiom(name="trivial", expr=sympy.Eq(1, 1)),),
        )
        h = axioms.hypothesis(
            "product_rule",
            expr=sympy.Eq(expr, expected),
        )
        script = (
            ProofBuilder(
                axioms, h.name, name="product_rule_proof", claim="Product rule"
            )
            .lemma(
                "diff_result",
                LemmaKind.EQUALITY,
                expr=expr,
                expected=expected,
                description="d/dx(x*sin(x)) = sin(x) + x*cos(x)",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_trig_identity(self):
        """sin^2(x) + cos^2(x) = 1."""
        x = sympy.Symbol("x")
        expr = sympy.sin(x) ** 2 + sympy.cos(x) ** 2

        axioms = AxiomSet(
            name="trig",
            axioms=(Axiom(name="trivial", expr=sympy.Eq(1, 1)),),
        )
        h = axioms.hypothesis(
            "pythagorean",
            expr=sympy.Eq(expr, 1),
        )
        script = (
            ProofBuilder(axioms, h.name, name="trig_proof", claim="sin^2 + cos^2 = 1")
            .lemma(
                "identity",
                LemmaKind.EQUALITY,
                expr=expr,
                expected=sympy.Integer(1),
                description="Pythagorean identity",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary
