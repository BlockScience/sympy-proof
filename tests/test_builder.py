"""Tests for symproof.builder."""

from __future__ import annotations

import pytest
import sympy

from symproof import LemmaKind, ProofBuilder


class TestProofBuilder:
    def test_chainable_api(self, positive_reals_axioms):
        script = (
            ProofBuilder(
                positive_reals_axioms,
                "some_hypothesis",
                name="test",
                claim="test claim",
            )
            .lemma(
                "l1",
                LemmaKind.EQUALITY,
                expr=sympy.Integer(1) + sympy.Integer(1),
                expected=sympy.Integer(2),
            )
            .lemma("l2", LemmaKind.BOOLEAN, expr=sympy.true)
            .build()
        )
        assert len(script.lemmas) == 2

    def test_empty_builder_raises(self, axiom_set_hash):
        with pytest.raises(ValueError):
            ProofBuilder(axiom_set_hash, "h", name="name", claim="claim").build()

    def test_script_has_correct_metadata(self, positive_reals_axioms):
        script = (
            ProofBuilder(
                positive_reals_axioms,
                "my_hypothesis",
                name="my_proof",
                claim="my claim",
            )
            .lemma("l1", LemmaKind.BOOLEAN, expr=sympy.true)
            .build()
        )
        assert script.name == "my_proof"
        assert script.target == "my_hypothesis"
        assert script.axiom_set_hash == positive_reals_axioms.axiom_set_hash
        assert script.claim == "my claim"

    def test_accepts_hash_string(self, axiom_set_hash):
        script = (
            ProofBuilder(axiom_set_hash, "h", name="p", claim="c")
            .lemma("l1", LemmaKind.BOOLEAN, expr=sympy.true)
            .build()
        )
        assert script.axiom_set_hash == axiom_set_hash

    def test_rejects_invalid_type(self):
        with pytest.raises(TypeError):
            ProofBuilder(42, "h", name="p", claim="c")  # type: ignore[arg-type]

    def test_lemma_with_depends_on(self, axiom_set_hash):
        script = (
            ProofBuilder(axiom_set_hash, "h", name="p", claim="c")
            .lemma("l1", LemmaKind.BOOLEAN, expr=sympy.true)
            .lemma("l2", LemmaKind.BOOLEAN, expr=sympy.true, depends_on=["l1"])
            .build()
        )
        assert script.lemmas[1].depends_on == ["l1"]

    def test_lemma_with_assumptions(self, axiom_set_hash):
        x = sympy.Symbol("x")
        script = (
            ProofBuilder(axiom_set_hash, "h", name="p", claim="c")
            .lemma(
                "l1",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(x),
                assumptions={"x": {"positive": True}},
            )
            .build()
        )
        assert script.lemmas[0].assumptions == {"x": {"positive": True}}
