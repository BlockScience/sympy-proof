"""Tests for symproof.serialization."""

from __future__ import annotations

import sympy

from symproof import canonical_srepr, make_canonical_dict, restore_expr


class TestMakeCanonicalDict:
    def test_order_independent(self):
        x, y = sympy.symbols("x y")
        d1 = make_canonical_dict({"b": x + y, "a": x**2})
        d2 = make_canonical_dict({"a": x**2, "b": x + y})
        assert d1 == d2

    def test_nested_dict_order_independent(self):
        x = sympy.Symbol("x")
        d1 = make_canonical_dict({"outer": {"b": x, "a": x + 1}})
        d2 = make_canonical_dict({"outer": {"a": x + 1, "b": x}})
        assert d1 == d2

    def test_sympy_expressions_become_strings(self):
        x = sympy.Symbol("x")
        d = make_canonical_dict({"expr": x**2 + 1})
        assert isinstance(d["expr"], str)

    def test_sympy_expr_round_trips(self):
        x = sympy.Symbol("x")
        expr = x**2 + 2 * x + 1
        d = make_canonical_dict({"e": expr})
        restored = sympy.sympify(d["e"])
        assert sympy.simplify(restored - expr) == 0

    def test_list_values_preserved_in_order(self):
        x, y = sympy.symbols("x y")
        d = make_canonical_dict({"items": [x, y]})
        assert d["items"][0] == sympy.srepr(x)
        assert d["items"][1] == sympy.srepr(y)

    def test_scalar_values_pass_through(self):
        d = make_canonical_dict({"n": 42, "s": "hello", "b": True})
        assert d["n"] == 42
        assert d["s"] == "hello"
        assert d["b"] is True


class TestCanonicalSrepr:
    def test_deterministic(self):
        x, y = sympy.symbols("x y")
        expr = x**2 + 2 * x * y + y**2
        assert canonical_srepr(expr) == canonical_srepr(expr)

    def test_round_trip(self):
        x, y = sympy.symbols("x y")
        expr = x**2 + 2 * x * y + y**2
        s = canonical_srepr(expr)
        restored = restore_expr(s)
        assert sympy.simplify(restored - expr) == 0

    def test_different_expressions_different_reprs(self):
        x = sympy.Symbol("x")
        assert canonical_srepr(x) != canonical_srepr(x + 1)

    def test_sympy_true_round_trips(self):
        s = canonical_srepr(sympy.true)
        assert restore_expr(s) == sympy.true

    def test_boolean_expr_round_trips(self):
        x = sympy.Symbol("x")
        expr = sympy.Ge(x, 0)
        s = canonical_srepr(expr)
        restored = restore_expr(s)
        assert canonical_srepr(restored) == canonical_srepr(expr)
