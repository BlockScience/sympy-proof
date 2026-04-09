"""Tests for symproof.tactics."""

from __future__ import annotations

import sympy

from symproof import LemmaKind, auto_lemma, try_implication, try_query, try_simplify


class TestTrySimplify:
    def test_true_expression(self):
        assert try_simplify(sympy.true) is True

    def test_false_expression(self):
        assert try_simplify(sympy.false) is False

    def test_with_assumptions(self):
        x = sympy.Symbol("x")
        asm = {"x": {"positive": True, "real": True}}
        result = try_simplify(x > 0, assumptions=asm)
        assert result is True

    def test_inconclusive(self):
        x = sympy.Symbol("x")
        result = try_simplify(x > 0)
        assert result is None


class TestTryImplication:
    def test_simple_implication(self):
        result = try_implication(sympy.true, sympy.true)
        assert result is True

    def test_inconclusive_implication(self):
        x, y = sympy.symbols("x y")
        result = try_implication(x > 0, y > 0)
        assert result is None


class TestTryQuery:
    def test_positive_query(self):
        x = sympy.Symbol("x")
        result = try_query(
            sympy.Q.positive(x),
            assumptions={"x": {"positive": True}},
        )
        assert result is True

    def test_fails_without_assumptions(self):
        x = sympy.Symbol("x")
        result = try_query(sympy.Q.positive(x))
        assert result is None


class TestAutoLemma:
    def test_equality_auto_detected(self):
        lem = auto_lemma(
            "sum",
            expr=sympy.Integer(2) + sympy.Integer(3),
            expected=sympy.Integer(5),
        )
        assert lem is not None
        assert lem.kind == LemmaKind.EQUALITY

    def test_boolean_auto_detected(self):
        lem = auto_lemma("taut", expr=sympy.true)
        assert lem is not None
        assert lem.kind == LemmaKind.BOOLEAN

    def test_query_auto_detected(self):
        x = sympy.Symbol("x")
        lem = auto_lemma(
            "pos",
            expr=sympy.Q.positive(x),
            assumptions={"x": {"positive": True}},
        )
        assert lem is not None
        assert lem.kind == LemmaKind.QUERY

    def test_returns_none_when_inconclusive(self):
        x = sympy.Symbol("x")
        lem = auto_lemma("unknown", expr=x > 0)
        assert lem is None


# ===================================================================
# Signed accumulation tactic
# ===================================================================


class TestSignedSumLemmas:
    """Core signed accumulation tactic — domain-agnostic."""

    def test_all_nonneg_net_nonneg(self):
        """All positive terms → net >= 0."""
        from symproof.tactics import SignedTerm, signed_sum_lemmas
        from symproof.verification import verify_lemma

        terms = [
            SignedTerm(expr=sympy.Rational(1, 3), nonneg=True, label="a"),
            SignedTerm(expr=sympy.Rational(2, 5), nonneg=True, label="b"),
        ]
        lemmas = signed_sum_lemmas(terms, net_nonneg=True, name_prefix="t")
        net_lemma = lemmas[-1]
        assert "net" in net_lemma.name
        result = verify_lemma(net_lemma)
        assert result.passed

    def test_mixed_signs_net_positive(self):
        """Positive terms dominate negative → net >= 0."""
        from symproof.tactics import SignedTerm, signed_sum_lemmas
        from symproof.verification import verify_lemma

        terms = [
            SignedTerm(expr=sympy.Rational(3, 4), nonneg=True, label="gain"),
            SignedTerm(expr=sympy.Rational(-1, 4), nonneg=False, label="loss"),
        ]
        # net = 3/4 - 1/4 = 1/2 >= 0
        lemmas = signed_sum_lemmas(terms, net_nonneg=True, name_prefix="m")
        for lem in lemmas:
            result = verify_lemma(lem)
            assert result.passed, f"{lem.name} failed: {result.error}"

    def test_net_negative_fails(self):
        """Negative terms dominate → net >= 0 FAILS."""
        from symproof.tactics import SignedTerm, signed_sum_lemmas
        from symproof.verification import verify_lemma

        terms = [
            SignedTerm(expr=sympy.Rational(1, 7), nonneg=True, label="small"),
            SignedTerm(expr=sympy.Rational(-5, 7), nonneg=False, label="big"),
        ]
        # net = 1/7 - 5/7 = -4/7 < 0
        lemmas = signed_sum_lemmas(terms, net_nonneg=True, name_prefix="f")
        net_lemma = lemmas[-1]
        assert "NEGATIVE" in net_lemma.description
        result = verify_lemma(net_lemma)
        assert not result.passed

    def test_net_nonpositive_mode(self):
        """Lyapunov-style: all terms negative → net <= 0."""
        from symproof.tactics import SignedTerm, signed_sum_lemmas
        from symproof.verification import verify_lemma

        terms = [
            SignedTerm(expr=sympy.Rational(-1, 3), nonneg=False, label="damp1"),
            SignedTerm(expr=sympy.Rational(-1, 5), nonneg=False, label="damp2"),
        ]
        lemmas = signed_sum_lemmas(
            terms, net_nonneg=False, name_prefix="lyap",
        )
        net_lemma = lemmas[-1]
        result = verify_lemma(net_lemma)
        assert result.passed

    def test_net_nonpositive_fails_when_positive(self):
        """Net is positive but we wanted <= 0 → FAILS."""
        from symproof.tactics import SignedTerm, signed_sum_lemmas
        from symproof.verification import verify_lemma

        terms = [
            SignedTerm(expr=sympy.Rational(3, 4), nonneg=True, label="growth"),
            SignedTerm(expr=sympy.Rational(-1, 4), nonneg=False, label="damp"),
        ]
        # net = 1/2 > 0, but we want <= 0
        lemmas = signed_sum_lemmas(
            terms, net_nonneg=False, name_prefix="bad",
        )
        net_lemma = lemmas[-1]
        assert "POSITIVE" in net_lemma.description
        result = verify_lemma(net_lemma)
        assert not result.passed

    def test_magnitude_bounds(self):
        """Magnitude lemmas generated when bound is provided."""
        from symproof.tactics import SignedTerm, signed_sum_lemmas
        from symproof.verification import verify_lemma

        terms = [
            SignedTerm(
                expr=sympy.Rational(1, 3), nonneg=True,
                bound=sympy.Integer(1), label="bounded",
            ),
        ]
        lemmas = signed_sum_lemmas(terms, name_prefix="b")
        bound_lemmas = [l for l in lemmas if "bound" in l.name]
        assert len(bound_lemmas) == 1
        result = verify_lemma(bound_lemmas[0])
        assert result.passed

    def test_no_magnitude_without_bound(self):
        """No magnitude lemma when bound is None."""
        from symproof.tactics import SignedTerm, signed_sum_lemmas

        terms = [
            SignedTerm(expr=sympy.Rational(1, 3), nonneg=True, label="no_bound"),
        ]
        lemmas = signed_sum_lemmas(terms, name_prefix="nb")
        bound_lemmas = [l for l in lemmas if "bound" in l.name]
        assert len(bound_lemmas) == 0

    def test_sealed_proof_with_signed_sum(self):
        """Full sealed proof using signed accumulation."""
        from symproof import Axiom, AxiomSet, ProofBuilder, seal
        from symproof.tactics import SignedTerm, signed_sum_lemmas

        axioms = AxiomSet(
            name="test",
            axioms=(Axiom(name="base", expr=sympy.Eq(1, 1)),),
        )
        terms = [
            SignedTerm(expr=sympy.Rational(2, 3), nonneg=True, label="a"),
            SignedTerm(expr=sympy.Rational(-1, 5), nonneg=False, label="b"),
        ]
        lemmas = signed_sum_lemmas(terms, net_nonneg=True, name_prefix="s")

        total = sum(t.expr for t in terms)
        h = axioms.hypothesis("net_ok", expr=sympy.Ge(total, 0))
        builder = ProofBuilder(
            axioms, h.name, name="signed_proof", claim="net >= 0",
        )
        for lem in lemmas:
            builder = builder.add_lemma(lem)
        bundle = seal(axioms, h, builder.build())
        assert len(bundle.bundle_hash) == 64
