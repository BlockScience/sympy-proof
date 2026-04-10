"""Tests for the boolean circuit library."""

from __future__ import annotations

import pytest
import sympy
from sympy import And, Not, Or, Xor, symbols

from symproof import AxiomSet, ProofStatus
from symproof.library.circuits import (
    boolean_entropy,
    circuit_equivalence,
    circuit_output,
    circuit_satisfies,
    gate_truth_table,
    r1cs_witness_check,
)

_EMPTY = AxiomSet(name="test", axioms=())
a, b, c = symbols("a b c")


class TestGateTruthTable:
    def test_xor(self):
        bundle = gate_truth_table(
            _EMPTY, Xor(a, b), [a, b], [False, True, True, False],
        )
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_and(self):
        bundle = gate_truth_table(
            _EMPTY, And(a, b), [a, b], [False, False, False, True],
        )
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_or(self):
        bundle = gate_truth_table(
            _EMPTY, Or(a, b), [a, b], [False, True, True, True],
        )
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_wrong_table_rejected(self):
        with pytest.raises(ValueError):
            gate_truth_table(
                _EMPTY, And(a, b), [a, b], [True, True, True, True],
            )

    def test_three_variable_gate(self):
        # Majority gate: output True if 2+ inputs are True
        majority = Or(And(a, b), And(a, c), And(b, c))
        expected = [False, False, False, True, False, True, True, True]
        bundle = gate_truth_table(_EMPTY, majority, [a, b, c], expected)
        assert bundle.proof_result.status == ProofStatus.VERIFIED


class TestCircuitEquivalence:
    def test_xor_decomposition(self):
        bundle = circuit_equivalence(
            _EMPTY,
            Xor(a, b),
            Or(And(a, Not(b)), And(Not(a), b)),
            [a, b],
        )
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_de_morgan(self):
        """De Morgan's law: Not(And(a, b)) == Or(Not(a), Not(b))."""
        bundle = circuit_equivalence(
            _EMPTY, Not(And(a, b)), Or(Not(a), Not(b)), [a, b],
        )
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_inequivalent_rejected(self):
        with pytest.raises(ValueError, match="differ"):
            circuit_equivalence(_EMPTY, And(a, b), Or(a, b), [a, b])


class TestCircuitOutput:
    def test_correct_output(self):
        bundle = circuit_output(_EMPTY, And(a, b), {a: True, b: True}, True)
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_wrong_output_rejected(self):
        with pytest.raises(ValueError, match="expected"):
            circuit_output(_EMPTY, And(a, b), {a: True, b: False}, True)


class TestCircuitSatisfies:
    def test_satisfying_assignment(self):
        bundle = circuit_satisfies(_EMPTY, Or(a, b), {a: False, b: True})
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_unsatisfying_rejected(self):
        with pytest.raises(ValueError, match="does not satisfy"):
            circuit_satisfies(_EMPTY, And(a, b), {a: True, b: False})


class TestR1CSWitnessCheck:
    def test_simple_multiplication(self):
        """R1CS: a * b = c with witness [1, 3, 7, 21]."""
        A = sympy.Matrix([[0, 1, 0, 0]])
        B = sympy.Matrix([[0, 0, 1, 0]])
        C = sympy.Matrix([[0, 0, 0, 1]])
        w = sympy.Matrix([1, 3, 7, 21])
        bundle = r1cs_witness_check(_EMPTY, A, B, C, w)
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_invalid_witness_rejected(self):
        """Wrong witness: 3 * 7 != 20."""
        A = sympy.Matrix([[0, 1, 0, 0]])
        B = sympy.Matrix([[0, 0, 1, 0]])
        C = sympy.Matrix([[0, 0, 0, 1]])
        w = sympy.Matrix([1, 3, 7, 20])
        with pytest.raises(ValueError, match="violated"):
            r1cs_witness_check(_EMPTY, A, B, C, w)

    def test_multiple_constraints(self):
        """Two constraints: a*b=c and c*1=c (identity check)."""
        A = sympy.Matrix([[0, 1, 0, 0], [0, 0, 0, 1]])
        B = sympy.Matrix([[0, 0, 1, 0], [1, 0, 0, 0]])
        C = sympy.Matrix([[0, 0, 0, 1], [0, 0, 0, 1]])
        w = sympy.Matrix([1, 5, 4, 20])
        bundle = r1cs_witness_check(_EMPTY, A, B, C, w)
        assert bundle.proof_result.status == ProofStatus.VERIFIED


class TestBooleanEntropy:
    def test_xor_max_entropy(self):
        bundle = boolean_entropy(_EMPTY, Xor(a, b), [a, b])
        assert bundle.proof_result.status == ProofStatus.VERIFIED
        # XOR has entropy = 1 bit
        assert "1" in bundle.hypothesis.description

    def test_and_lower_entropy(self):
        bundle = boolean_entropy(_EMPTY, And(a, b), [a, b])
        assert bundle.proof_result.status == ProofStatus.VERIFIED
        # AND has entropy < 1 bit
        assert "0.811" in bundle.hypothesis.description or "True: 1/4" in bundle.hypothesis.description

    def test_constant_zero_entropy(self):
        """A constant function has zero entropy."""
        bundle = boolean_entropy(_EMPTY, And(a, Not(a)), [a])
        assert bundle.proof_result.status == ProofStatus.VERIFIED
        assert "0" in bundle.hypothesis.description
