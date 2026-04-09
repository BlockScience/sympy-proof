"""Tests for symproof.bundle (seal, disprove, check_consistency)."""

from __future__ import annotations

import pytest
import sympy

from symproof import (
    Axiom,
    AxiomSet,
    LemmaKind,
    ProofBuilder,
    ProofStatus,
    disprove,
    seal,
)
from symproof.bundle import check_consistency


class TestSeal:
    def test_seal_produces_bundle(
        self,
        positive_reals_axioms,
        product_positive_hypothesis,
        product_proof_script,
    ):
        bundle = seal(
            positive_reals_axioms,
            product_positive_hypothesis,
            product_proof_script,
        )
        assert bundle.proof_result.status == ProofStatus.VERIFIED
        assert len(bundle.bundle_hash) == 64

    def test_seal_bundle_hash_deterministic(
        self,
        positive_reals_axioms,
        product_positive_hypothesis,
        product_proof_script,
    ):
        b1 = seal(
            positive_reals_axioms,
            product_positive_hypothesis,
            product_proof_script,
        )
        b2 = seal(
            positive_reals_axioms,
            product_positive_hypothesis,
            product_proof_script,
        )
        assert b1.bundle_hash == b2.bundle_hash

    def test_seal_rejects_wrong_axiom_hash(
        self, positive_reals_axioms, product_positive_hypothesis
    ):
        x = sympy.Symbol("x")
        wrong_axioms = AxiomSet(
            name="wrong",
            axioms=(Axiom(name="a1", expr=x > 1),),
        )
        script = (
            ProofBuilder(
                positive_reals_axioms,
                product_positive_hypothesis.name,
                name="p",
                claim="c",
            )
            .lemma("l1", LemmaKind.BOOLEAN, expr=sympy.true)
            .build()
        )
        with pytest.raises(ValueError, match="axiom_set_hash"):
            seal(wrong_axioms, product_positive_hypothesis, script)

    def test_seal_rejects_wrong_target(
        self, positive_reals_axioms, product_positive_hypothesis
    ):
        script = (
            ProofBuilder(
                positive_reals_axioms,
                "wrong_target",
                name="p",
                claim="c",
            )
            .lemma("l1", LemmaKind.BOOLEAN, expr=sympy.true)
            .build()
        )
        with pytest.raises(ValueError, match="target"):
            seal(
                positive_reals_axioms,
                product_positive_hypothesis,
                script,
            )

    def test_seal_rejects_unverified_proof(
        self, positive_reals_axioms, product_positive_hypothesis
    ):
        script = (
            ProofBuilder(
                positive_reals_axioms,
                product_positive_hypothesis.name,
                name="bad",
                claim="bad proof",
            )
            .lemma(
                "wrong",
                LemmaKind.EQUALITY,
                expr=sympy.Integer(2),
                expected=sympy.Integer(5),
            )
            .build()
        )
        with pytest.raises(ValueError, match="VERIFIED"):
            seal(
                positive_reals_axioms,
                product_positive_hypothesis,
                script,
            )

    def test_seal_rejects_hypothesis_axiom_mismatch(self, positive_reals_axioms):
        x = sympy.Symbol("x")
        other_axioms = AxiomSet(
            name="other",
            axioms=(Axiom(name="a1", expr=x > 0),),
        )
        h = other_axioms.hypothesis("h1", expr=x > 0)
        script = (
            ProofBuilder(
                positive_reals_axioms,
                h.name,
                name="p",
                claim="c",
            )
            .lemma("l1", LemmaKind.BOOLEAN, expr=sympy.true)
            .build()
        )
        with pytest.raises(ValueError, match="axiom_set_hash"):
            seal(positive_reals_axioms, h, script)


class TestDisprove:
    def test_disprove_produces_disproof(self):
        p = sympy.Symbol("p")
        axioms = AxiomSet(
            name="context",
            axioms=(Axiom(name="p_negative", expr=p < 0),),
        )
        h = axioms.hypothesis("p_positive", expr=p > 0)
        not_h = h.negate()

        script = (
            ProofBuilder(
                axioms,
                not_h.name,
                name="disproof_script",
                claim="p < 0 implies not p > 0",
            )
            .lemma(
                "p_not_positive",
                LemmaKind.QUERY,
                expr=~sympy.Q.positive(p),
                assumptions={"p": {"negative": True}},
            )
            .build()
        )
        negation_bundle = seal(axioms, not_h, script)
        result = disprove(h, negation_bundle)
        assert len(result.disproof_hash) == 64

    def test_disprove_rejects_axiom_mismatch(self):
        p = sympy.Symbol("p")
        axioms1 = AxiomSet(
            name="a1",
            axioms=(Axiom(name="p_neg", expr=p < 0),),
        )
        axioms2 = AxiomSet(
            name="a2",
            axioms=(Axiom(name="p_neg", expr=p < -1),),
        )

        h1 = axioms1.hypothesis("h", expr=p > 0)
        not_h2 = axioms2.hypothesis("not_h", expr=~(p > 0))

        script = (
            ProofBuilder(axioms2, not_h2.name, name="p", claim="c")
            .lemma(
                "l1",
                LemmaKind.QUERY,
                expr=~sympy.Q.positive(p),
                assumptions={"p": {"negative": True}},
            )
            .build()
        )
        bundle = seal(axioms2, not_h2, script)

        with pytest.raises(ValueError, match="Axiom set mismatch"):
            disprove(h1, bundle)


class TestCheckConsistency:
    def test_no_error_on_compatible_bundles(
        self,
        positive_reals_axioms,
        product_positive_hypothesis,
        product_proof_script,
    ):
        bundle = seal(
            positive_reals_axioms,
            product_positive_hypothesis,
            product_proof_script,
        )
        # Same bundle twice — no contradiction
        check_consistency(bundle, bundle)

    def test_different_axiom_sets_no_contradiction(self):
        x = sympy.Symbol("x")
        axioms1 = AxiomSet(
            name="a1",
            axioms=(Axiom(name="x_pos", expr=x > 0),),
        )
        axioms2 = AxiomSet(
            name="a2",
            axioms=(Axiom(name="x_neg", expr=x < 0),),
        )

        h1 = axioms1.hypothesis("h1", expr=x > 0)
        h2 = axioms2.hypothesis("h2", expr=x < 0)

        s1 = (
            ProofBuilder(axioms1, h1.name, name="p1", claim="c1")
            .lemma(
                "l1",
                LemmaKind.BOOLEAN,
                expr=x > 0,
                assumptions={"x": {"positive": True, "real": True}},
            )
            .build()
        )
        s2 = (
            ProofBuilder(axioms2, h2.name, name="p2", claim="c2")
            .lemma(
                "l1",
                LemmaKind.BOOLEAN,
                expr=x < 0,
                assumptions={"x": {"negative": True, "real": True}},
            )
            .build()
        )
        b1 = seal(axioms1, h1, s1)
        b2 = seal(axioms2, h2, s2)
        # Different axiom sets — no contradiction possible
        check_consistency(b1, b2)
