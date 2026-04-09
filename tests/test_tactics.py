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
