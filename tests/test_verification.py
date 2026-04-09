"""Tests for symproof.verification."""

from __future__ import annotations

import sympy

from symproof import (
    LemmaKind,
    ProofBuilder,
    ProofScript,
    ProofStatus,
    hash_proof,
    verify_lemma,
    verify_proof,
)
from symproof.models import Lemma


class TestVerifyLemma:
    def test_equality_via_doit(self):
        k = sympy.Symbol("k", integer=True, nonneg=True)
        lemma = Lemma(
            name="geometric",
            kind=LemmaKind.EQUALITY,
            expr=sympy.Sum(sympy.Rational(1, 2) ** k, (k, 0, sympy.oo)),
            expected=sympy.Integer(2),
        )
        result = verify_lemma(lemma)
        assert result.passed

    def test_equality_via_simplify(self):
        lemma = Lemma(
            name="arithmetic",
            kind=LemmaKind.EQUALITY,
            expr=sympy.Integer(2) + sympy.Integer(2),
            expected=sympy.Integer(4),
        )
        result = verify_lemma(lemma)
        assert result.passed

    def test_equality_fails_wrong_expected(self):
        lemma = Lemma(
            name="wrong",
            kind=LemmaKind.EQUALITY,
            expr=sympy.Integer(2) + sympy.Integer(2),
            expected=sympy.Integer(5),
        )
        result = verify_lemma(lemma)
        assert not result.passed

    def test_equality_missing_expected_fails(self):
        lemma = Lemma(
            name="no_expected",
            kind=LemmaKind.EQUALITY,
            expr=sympy.Integer(4),
        )
        result = verify_lemma(lemma)
        assert not result.passed
        assert result.error is not None

    def test_boolean_true(self):
        lemma = Lemma(
            name="tautology",
            kind=LemmaKind.BOOLEAN,
            expr=sympy.true,
        )
        result = verify_lemma(lemma)
        assert result.passed

    def test_boolean_false(self):
        lemma = Lemma(
            name="contradiction",
            kind=LemmaKind.BOOLEAN,
            expr=sympy.false,
        )
        result = verify_lemma(lemma)
        assert not result.passed

    def test_boolean_with_assumptions(self):
        x = sympy.Symbol("x")
        lemma = Lemma(
            name="pos_check",
            kind=LemmaKind.BOOLEAN,
            expr=x > 0,
            assumptions={"x": {"positive": True, "real": True}},
        )
        result = verify_lemma(lemma)
        assert result.passed

    def test_query_positive(self):
        x = sympy.Symbol("x")
        lemma = Lemma(
            name="pos_query",
            kind=LemmaKind.QUERY,
            expr=sympy.Q.positive(x),
            assumptions={"x": {"positive": True}},
        )
        result = verify_lemma(lemma)
        assert result.passed

    def test_query_fails_without_assumptions(self):
        x = sympy.Symbol("x")
        lemma = Lemma(
            name="pos_query_no_asm",
            kind=LemmaKind.QUERY,
            expr=sympy.Q.positive(x),
            assumptions={},
        )
        result = verify_lemma(lemma)
        assert not result.passed


class TestVerifyProof:
    def test_verified_on_valid_script(self, product_proof_script):
        result = verify_proof(product_proof_script)
        assert result.status == ProofStatus.VERIFIED

    def test_proof_hash_returned_on_verified(self, product_proof_script):
        result = verify_proof(product_proof_script)
        assert result.proof_hash is not None
        assert len(result.proof_hash) == 64

    def test_lemma_results_populated(self, product_proof_script):
        result = verify_proof(product_proof_script)
        assert len(result.lemma_results) == len(product_proof_script.lemmas)
        assert all(lr.passed for lr in result.lemma_results)

    def test_fails_on_failing_lemma(self, axiom_set_hash):
        script = (
            ProofBuilder(axiom_set_hash, "some_target", name="bad", claim="bad proof")
            .lemma(
                "wrong_arithmetic",
                LemmaKind.EQUALITY,
                expr=sympy.Integer(2) + sympy.Integer(2),
                expected=sympy.Integer(5),
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status == ProofStatus.FAILED
        assert "wrong_arithmetic" in result.failure_summary


class TestEvidenceRoundTrip:
    def test_to_evidence_from_evidence_hash_stable(self, product_proof_script):
        evidence = product_proof_script.to_evidence()
        restored = ProofScript.from_evidence(evidence)
        assert hash_proof(restored) == hash_proof(product_proof_script)

    def test_evidence_is_json_compatible(self, product_proof_script):
        import json

        evidence = product_proof_script.to_evidence()
        serialized = json.dumps(evidence)
        assert isinstance(serialized, str)

    def test_restored_script_verifies(self, product_proof_script):
        evidence = product_proof_script.to_evidence()
        restored = ProofScript.from_evidence(evidence)
        result = verify_proof(restored)
        assert result.status == ProofStatus.VERIFIED

    def test_lemma_expressions_restored(self, product_proof_script):
        evidence = product_proof_script.to_evidence()
        restored = ProofScript.from_evidence(evidence)
        orig_expr = product_proof_script.lemmas[0].expr
        restored_expr = restored.lemmas[0].expr
        assert orig_expr == restored_expr
