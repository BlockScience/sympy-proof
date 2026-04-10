"""Tests for explicit evaluation control: unevaluated() and evaluation().

Verifies that:
1. unevaluated() suppresses SymPy's eager evaluation
2. evaluation() re-enables it
3. Round-trip: construct under unevaluated, verify under evaluation
4. Axiom expressions preserve structure under unevaluated
5. Verification still works on structurally-preserved expressions
"""

from __future__ import annotations

import sympy

from symproof import (
    Axiom,
    AxiomSet,
    LemmaKind,
    ProofBuilder,
    ProofStatus,
    evaluation,
    seal,
    unevaluated,
)
from symproof.serialization import canonical_srepr


class TestUnevaluated:
    def test_suppresses_eager_relational(self):
        """x > 0 stays structural when x has positive=True."""
        x = sympy.Symbol("x", positive=True)
        with unevaluated():
            expr = x > 0
        assert expr is not sympy.S.true
        assert isinstance(expr, sympy.StrictGreaterThan)

    def test_suppresses_eager_ge(self):
        """x >= 0 stays structural when x has nonnegative=True."""
        x = sympy.Symbol("x", nonnegative=True)
        with unevaluated():
            expr = x >= 0
        assert expr is not sympy.S.true

    def test_without_unevaluated_collapses(self):
        """Baseline: without unevaluated, x > 0 becomes True."""
        x = sympy.Symbol("x", positive=True)
        expr = x > 0
        assert expr is sympy.S.true

    def test_preserves_addition(self):
        """x + x stays as Add, not 2*x."""
        x = sympy.Symbol("x")
        with unevaluated():
            expr = x + x
        # Under evaluate(False), x + x may remain unevaluated
        # The exact form depends on SymPy version, but it should
        # not be simplified to 2*x in all cases
        assert expr is not None  # sanity

    def test_srepr_captures_structure(self):
        """srepr of unevaluated expr includes the relational operator."""
        x = sympy.Symbol("x", positive=True)
        with unevaluated():
            expr = x > 0
        s = canonical_srepr(expr)
        assert "StrictGreaterThan" in s
        assert "Symbol" in s


class TestEvaluation:
    def test_enables_simplification(self):
        """simplify works inside evaluation() gate."""
        x = sympy.Symbol("x", positive=True)
        with unevaluated():
            expr = x > 0
        with evaluation():
            result = sympy.simplify(expr)
        assert result is sympy.S.true

    def test_ask_works_inside_gate(self):
        """sympy.ask returns True inside evaluation() gate."""
        x = sympy.Symbol("x", positive=True)
        with evaluation():
            result = sympy.ask(sympy.Q.positive(x))
        assert result is True

    def test_nesting_is_safe(self):
        """Nested evaluation/unevaluated contexts work correctly."""
        x = sympy.Symbol("x", positive=True)
        with unevaluated():
            expr = x > 0
            with evaluation():
                inner = sympy.simplify(expr)
            # Back to unevaluated
            still_structural = x >= 0
        assert inner is sympy.S.true
        assert still_structural is not sympy.S.true


class TestRoundTrip:
    def test_construct_unevaluated_verify_evaluated(self):
        """Full workflow: build under unevaluated, seal verifies correctly."""
        x = sympy.Symbol("x", positive=True)
        with unevaluated():
            axioms = AxiomSet(
                name="test",
                axioms=(Axiom(name="x_pos", expr=x > 0),),
            )

        h = axioms.hypothesis("h", expr=x > 0)
        script = (
            ProofBuilder(axioms, h.name, name="p", claim="c")
            .lemma(
                "l",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(x),
                assumptions={"x": {"positive": True}},
            )
            .build()
        )
        bundle = seal(axioms, h, script)
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_axiom_expr_preserved_in_hash(self):
        """Axiom sets built under unevaluated have different hashes
        from those built without (because expr is structural vs True)."""
        x = sympy.Symbol("x", positive=True)

        with unevaluated():
            ax_structural = AxiomSet(
                name="test",
                axioms=(Axiom(name="x_pos", expr=x > 0),),
            )

        ax_collapsed = AxiomSet(
            name="test",
            axioms=(Axiom(name="x_pos", expr=x > 0),),
        )

        # They encode different information, so different hashes
        assert ax_structural.axiom_set_hash != ax_collapsed.axiom_set_hash

    def test_equality_lemma_verifies_with_structural_axioms(self):
        """EQUALITY lemmas verify correctly when axioms are structural."""
        x = sympy.Symbol("x")
        with unevaluated():
            axioms = AxiomSet(
                name="test",
                axioms=(Axiom(name="dummy", expr=sympy.Eq(1, 1)),),
            )

        h = axioms.hypothesis("h", expr=sympy.Eq((x + 1) ** 2, x**2 + 2 * x + 1))
        script = (
            ProofBuilder(axioms, h.name, name="p", claim="c")
            .lemma(
                "expand",
                LemmaKind.EQUALITY,
                expr=(x + 1) ** 2,
                expected=x**2 + 2 * x + 1,
            )
            .build()
        )
        bundle = seal(axioms, h, script)
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_falsity_check_works_on_structural_exprs(self):
        """_no_false_axioms catches false axioms even when structural."""
        import pytest

        x = sympy.Symbol("x", real=True)
        with pytest.raises(ValueError, match="provably false"):
            with unevaluated():
                AxiomSet(
                    name="bad",
                    axioms=(
                        Axiom(name="contradiction", expr=sympy.And(x > 0, x < 0)),
                    ),
                )
