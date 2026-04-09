"""Tests for FixedPointType composite — WAD arithmetic proofs.

Demonstrates how the composite type API reduces boilerplate for
integer-scaled fixed-point proofs compared to manual construction.
"""

from __future__ import annotations

import sympy
from sympy import Integer, Rational, Symbol, floor

from symproof import (
    Axiom,
    LemmaKind,
    ProofBuilder,
    seal,
    verify_proof,
)
from symproof.composite import FixedPointType, make_axiom_set


WAD_SCALE = 10**18


class TestFixedPointTypeConstruction:
    """Verify FixedPointType produces correct expressions and axioms."""

    def test_from_int(self):
        wad = FixedPointType(name="wad", scale=WAD_SCALE)
        assert wad.from_int(2) == 2 * Integer(WAD_SCALE)

    def test_mul_expression(self):
        wad = FixedPointType(name="wad", scale=WAD_SCALE)
        a = Symbol("a", positive=True, integer=True)
        b = Symbol("b", positive=True, integer=True)
        result = wad.mul(a, b)
        assert isinstance(result, sympy.floor)

    def test_div_expression(self):
        wad = FixedPointType(name="wad", scale=WAD_SCALE)
        a = Symbol("a", positive=True, integer=True)
        b = Symbol("b", positive=True, integer=True)
        result = wad.div(a, b)
        assert isinstance(result, sympy.floor)

    def test_encode_decode(self):
        wad = FixedPointType(name="wad", scale=WAD_SCALE)
        assert wad.encode(Integer(5)) == 5 * Integer(WAD_SCALE)
        # decode(encode(5)) should give back 5
        round_trip = sympy.simplify(wad.decode(wad.encode(Integer(5))) - 5)
        assert round_trip == 0

    def test_axioms_generated(self):
        wad = FixedPointType(name="wad", scale=WAD_SCALE)
        axioms = wad.axioms()
        assert len(axioms) >= 1
        assert any("scale" in a.name for a in axioms)

    def test_axioms_with_prefix(self):
        wad = FixedPointType(name="wad", scale=WAD_SCALE)
        axioms = wad.axioms(prefix="pool1_")
        assert all(a.name.startswith("pool1_") for a in axioms)

    def test_make_axiom_set(self):
        wad = FixedPointType(name="wad", scale=WAD_SCALE)
        extra = Axiom(name="extra", expr=sympy.Eq(Integer(1), Integer(1)))
        axiom_set = make_axiom_set("test", wad, extra)
        names = {a.name for a in axiom_set.axioms}
        assert "wad_scale" in names
        assert "extra" in names


class TestWADConcreteProofs:
    """Prove WAD arithmetic properties using the composite type API."""

    def setup_method(self):
        self.wad = FixedPointType(name="wad", scale=WAD_SCALE)
        self.axioms = make_axiom_set("wad_arithmetic", self.wad)

    def test_mulwad_concrete(self):
        """mulWad(2e18, 3e18) = 6e18."""
        wad = self.wad
        a = wad.from_int(2)
        b = wad.from_int(3)
        expected = wad.from_int(6)

        h = self.axioms.hypothesis(
            "mulwad_exact",
            expr=sympy.Eq(wad.mul(a, b), expected),
        )
        script = (
            ProofBuilder(self.axioms, h.name, name="mulwad", claim="2 * 3 = 6 in WAD")
            .add_lemma(wad.mul_lemma(a, b, expected, name="mul_eval"))
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_divwad_concrete(self):
        """divWad(6e18, 3e18) = 2e18."""
        wad = self.wad
        a = wad.from_int(6)
        b = wad.from_int(3)
        expected = wad.from_int(2)

        h = self.axioms.hypothesis(
            "divwad_exact",
            expr=sympy.Eq(wad.div(a, b), expected),
        )
        script = (
            ProofBuilder(self.axioms, h.name, name="divwad", claim="6 / 3 = 2 in WAD")
            .add_lemma(wad.div_lemma(a, b, expected, name="div_eval"))
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_mulwad_truncation(self):
        """mulWad(1, 1) = 0 — below precision floor."""
        wad = self.wad
        h = self.axioms.hypothesis(
            "mulwad_floor",
            expr=sympy.Eq(wad.mul(Integer(1), Integer(1)), Integer(0)),
        )
        script = (
            ProofBuilder(
                self.axioms, h.name, name="trunc", claim="mulWad(1,1) = 0"
            )
            .add_lemma(
                wad.mul_lemma(Integer(1), Integer(1), Integer(0), name="floor_eval")
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_roundtrip_exact(self):
        """divWad(mulWad(5e18, 2e18), 2e18) = 5e18."""
        wad = self.wad
        a = wad.from_int(5)
        b = wad.from_int(2)
        product = wad.mul(a, b)
        back = wad.div(product, b)

        h = self.axioms.hypothesis(
            "roundtrip",
            expr=sympy.Eq(back, a),
        )
        script = (
            ProofBuilder(
                self.axioms, h.name,
                name="roundtrip", claim="divWad(mulWad(5, 2), 2) = 5 in WAD",
            )
            .add_lemma(wad.mul_lemma(a, b, wad.from_int(10), name="mul_step"))
            .add_lemma(wad.div_lemma(product, b, a, name="div_step"))
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_to_int(self):
        """to_int(3.5 * WAD) = floor(3.5) = 3."""
        wad = self.wad
        raw = Rational(7, 2) * wad.scale  # 3.5 in WAD
        truncated = wad.to_int(raw)

        h = self.axioms.hypothesis(
            "to_int_trunc",
            expr=sympy.Eq(truncated, Integer(3)),
        )
        script = (
            ProofBuilder(
                self.axioms, h.name,
                name="to_int", claim="to_int(3.5 WAD) = 3",
            )
            .lemma(
                "trunc",
                LemmaKind.EQUALITY,
                expr=truncated,
                expected=Integer(3),
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_sealed_mulwad(self):
        """Sealed proof: mulWad(4, 5) = 20 in WAD."""
        wad = self.wad
        a = wad.from_int(4)
        b = wad.from_int(5)
        expected = wad.from_int(20)

        h = self.axioms.hypothesis(
            "mulwad_sealed",
            expr=sympy.Eq(wad.mul(a, b), expected),
        )
        script = (
            ProofBuilder(
                self.axioms, h.name,
                name="mulwad_sealed", claim="4 * 5 = 20 in WAD",
            )
            .add_lemma(wad.mul_lemma(a, b, expected, name="mul_eval"))
            .build()
        )
        bundle = seal(self.axioms, h, script)
        assert len(bundle.bundle_hash) == 64


class TestRayType:
    """Verify FixedPointType works for RAY (27 decimals) too."""

    def test_ray_mul(self):
        ray = FixedPointType(name="ray", scale=10**27)
        a = ray.from_int(3)
        b = ray.from_int(7)
        result = ray.mul(a, b)
        assert result == ray.from_int(21)

    def test_ray_axioms(self):
        ray = FixedPointType(name="ray", scale=10**27)
        axioms = ray.axioms()
        assert axioms[0].name == "ray_scale"


class TestMakeAxiomSetEdgeCases:
    """Edge cases for make_axiom_set helper."""

    def test_collision_raises(self):
        """Duplicate axiom names should raise ValueError."""
        wad = FixedPointType(name="wad", scale=WAD_SCALE)
        # This axiom has the same name as one generated by wad
        dupe = Axiom(name="wad_scale", expr=sympy.Eq(Integer(1), Integer(1)))
        try:
            make_axiom_set("test", wad, dupe)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "collision" in str(e).lower()

    def test_tuple_source(self):
        """Tuple of axioms should be accepted."""
        ax1 = Axiom(name="a1", expr=sympy.Eq(Integer(1), Integer(1)))
        ax2 = Axiom(name="a2", expr=sympy.Eq(Integer(2), Integer(2)))
        axiom_set = make_axiom_set("test", (ax1, ax2))
        assert len(axiom_set.axioms) == 2

    def test_bad_source_type(self):
        """Non-supported source type should raise TypeError."""
        try:
            make_axiom_set("test", "bad")
            assert False, "Should have raised TypeError"
        except TypeError:
            pass
