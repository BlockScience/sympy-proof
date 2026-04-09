"""Tests for symproof.models."""

from __future__ import annotations

import pytest
import sympy
from pydantic import ValidationError

from symproof.models import (
    Axiom,
    AxiomSet,
    Hypothesis,
    Lemma,
    LemmaKind,
    ProofBundle,
    ProofResult,
    ProofStatus,
)


class TestAxiom:
    def test_creation(self):
        x = sympy.Symbol("x")
        a = Axiom(name="x_pos", expr=x > 0, description="x is positive")
        assert a.name == "x_pos"
        assert a.description == "x is positive"

    def test_frozen(self):
        x = sympy.Symbol("x")
        a = Axiom(name="x_pos", expr=x > 0)
        with pytest.raises(ValidationError):
            a.name = "changed"  # type: ignore[misc]


class TestAxiomSet:
    def test_creation(self):
        x = sympy.Symbol("x")
        s = AxiomSet(
            name="test",
            axioms=(Axiom(name="a1", expr=x > 0),),
        )
        assert len(s.axioms) == 1

    def test_duplicate_names_rejected(self):
        x = sympy.Symbol("x")
        with pytest.raises(ValueError, match="Duplicate"):
            AxiomSet(
                name="test",
                axioms=(
                    Axiom(name="same", expr=x > 0),
                    Axiom(name="same", expr=x > 1),
                ),
            )

    def test_canonical_dict_deterministic(self):
        x, y = sympy.symbols("x y")
        s = AxiomSet(
            name="test",
            axioms=(
                Axiom(name="a1", expr=x > 0),
                Axiom(name="a2", expr=y > 0),
            ),
        )
        assert s.canonical_dict() == s.canonical_dict()

    def test_hypothesis_factory(self, positive_reals_axioms):
        x = sympy.Symbol("x")
        h = positive_reals_axioms.hypothesis("h1", expr=x**2 > 0)
        assert h.axiom_set_hash == positive_reals_axioms.axiom_set_hash
        assert h.name == "h1"

    def test_frozen(self):
        x = sympy.Symbol("x")
        s = AxiomSet(
            name="test",
            axioms=(Axiom(name="a1", expr=x > 0),),
        )
        with pytest.raises(ValidationError):
            s.name = "changed"  # type: ignore[misc]


class TestHypothesis:
    def test_requires_axiom_set_hash(self):
        x = sympy.Symbol("x")
        h = Hypothesis(name="h1", expr=x > 0, axiom_set_hash="a" * 64)
        assert h.axiom_set_hash == "a" * 64

    def test_negate(self, positive_reals_axioms):
        x = sympy.Symbol("x")
        h = positive_reals_axioms.hypothesis("h1", expr=x > 0)
        neg = h.negate()
        assert neg.name == "not_h1"
        assert neg.axiom_set_hash == h.axiom_set_hash

    def test_negate_preserves_axiom_binding(self, positive_reals_axioms):
        x = sympy.Symbol("x")
        h = positive_reals_axioms.hypothesis("h1", expr=x > 0)
        neg = h.negate()
        assert neg.axiom_set_hash == positive_reals_axioms.axiom_set_hash

    def test_frozen(self, positive_reals_axioms):
        x = sympy.Symbol("x")
        h = positive_reals_axioms.hypothesis("h1", expr=x > 0)
        with pytest.raises(ValidationError):
            h.name = "changed"  # type: ignore[misc]


class TestLemma:
    def test_creation(self):
        lem = Lemma(
            name="l1",
            kind=LemmaKind.BOOLEAN,
            expr=sympy.true,
        )
        assert lem.name == "l1"
        assert lem.kind == LemmaKind.BOOLEAN

    def test_frozen(self):
        lem = Lemma(name="l1", kind=LemmaKind.BOOLEAN, expr=sympy.true)
        with pytest.raises(ValidationError):
            lem.name = "changed"  # type: ignore[misc]


class TestProofBundle:
    def test_rejects_unverified(
        self,
        positive_reals_axioms,
        product_positive_hypothesis,
        product_proof_script,
    ):
        result = ProofResult(status=ProofStatus.FAILED, failure_summary="test")
        with pytest.raises(ValueError, match="VERIFIED"):
            ProofBundle(
                axiom_set=positive_reals_axioms,
                hypothesis=product_positive_hypothesis,
                proof=product_proof_script,
                proof_result=result,
                bundle_hash="a" * 64,
            )
