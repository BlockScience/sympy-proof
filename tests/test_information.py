"""Tests for the Shannon information theory library."""

from __future__ import annotations

import pytest
import sympy
from sympy import Rational as R

from symproof import AxiomSet, ProofStatus
from symproof.library.information import (
    binary_entropy_func,
    binary_symmetric_channel,
    entropy,
    joint_entropy,
    kl_divergence,
    mutual_information,
)

_EMPTY = AxiomSet(name="test", axioms=())


class TestEntropy:
    def test_fair_coin(self):
        bundle = entropy(_EMPTY, [R(1, 2), R(1, 2)])
        assert bundle.proof_result.status == ProofStatus.VERIFIED
        assert "1 bits" in bundle.hypothesis.description

    def test_biased_ternary(self):
        bundle = entropy(_EMPTY, [R(1, 4), R(1, 2), R(1, 4)])
        assert bundle.proof_result.status == ProofStatus.VERIFIED
        assert "3/2" in bundle.hypothesis.description

    def test_deterministic_zero_entropy(self):
        bundle = entropy(_EMPTY, [R(1, 1)])
        assert bundle.proof_result.status == ProofStatus.VERIFIED
        assert "0 bits" in bundle.hypothesis.description

    def test_bad_probs_rejected(self):
        with pytest.raises(ValueError, match="not 1"):
            entropy(_EMPTY, [R(1, 3), R(1, 3)])


class TestJointEntropy:
    def test_uniform_2x2(self):
        bundle = joint_entropy(_EMPTY, [[R(1, 4)] * 2] * 2)
        assert bundle.proof_result.status == ProofStatus.VERIFIED
        assert "2" in bundle.hypothesis.description  # 2 bits


class TestMutualInformation:
    def test_independent_is_zero(self):
        bundle = mutual_information(_EMPTY, [[R(1, 4)] * 2] * 2)
        assert bundle.proof_result.status == ProofStatus.VERIFIED
        assert "I(X;Y) = 0" in bundle.hypothesis.description

    def test_perfectly_correlated(self):
        bundle = mutual_information(_EMPTY, [[R(1, 2), 0], [0, R(1, 2)]])
        assert bundle.proof_result.status == ProofStatus.VERIFIED
        assert "I(X;Y) = 1" in bundle.hypothesis.description


class TestKLDivergence:
    def test_same_distribution_is_zero(self):
        bundle = kl_divergence(_EMPTY, [R(1, 2), R(1, 2)], [R(1, 2), R(1, 2)])
        assert bundle.proof_result.status == ProofStatus.VERIFIED
        assert "0 bits" in bundle.hypothesis.description

    def test_gibbs_inequality_verified(self):
        bundle = kl_divergence(
            _EMPTY, [R(1, 4), R(1, 2), R(1, 4)], [R(1, 3), R(1, 3), R(1, 3)],
        )
        assert bundle.proof_result.status == ProofStatus.VERIFIED
        # Gibbs' inequality lemma should pass
        gibbs = [lr for lr in bundle.proof_result.lemma_results if "gibbs" in lr.lemma_name]
        assert len(gibbs) == 1
        assert gibbs[0].passed

    def test_zero_q_rejected(self):
        with pytest.raises(ValueError, match="zero probability"):
            kl_divergence(_EMPTY, [R(1, 2), R(1, 2)], [R(1, 1), 0])


class TestBinaryEntropy:
    def test_at_quarter(self):
        bundle = binary_entropy_func(_EMPTY, R(1, 4))
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_max_at_half_is_one(self):
        bundle = binary_entropy_func(_EMPTY, R(1, 4))
        # Should verify H(1/2) = 1
        max_lemma = [lr for lr in bundle.proof_result.lemma_results if "max" in lr.lemma_name]
        assert len(max_lemma) == 1
        assert max_lemma[0].passed


class TestBSC:
    def test_capacity(self):
        bundle = binary_symmetric_channel(_EMPTY, R(1, 10))
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_low_noise_channel(self):
        """p=1/100 channel has capacity close to 1 bit."""
        bundle = binary_symmetric_channel(_EMPTY, R(1, 100))
        assert bundle.proof_result.status == ProofStatus.VERIFIED
