#!/usr/bin/env python3
"""ZK witness satisfaction and output entropy.

Scenario
--------
A simple R1CS circuit encoding "prover knows a, b such that a * b = c"
where c is the public output.

R1CS format: for each constraint i, (A[i,:] . w) * (B[i,:] . w) = C[i,:] . w
where w is the witness vector [1, a, b, c] (1 is the constant wire).

Constraints:
  1. a * b = c     →  A=[0,1,0,0], B=[0,0,1,0], C=[0,0,0,1]

We verify: the witness [1, 3, 7, 21] satisfies the circuit (3 * 7 = 21).

We also compute the output entropy of the XOR gate vs AND gate to
quantify information leakage — staging toward Shannon theory.

Run: uv run python -m symproof.library.examples.circuits.03_zk_witness
"""

import sympy
from sympy import Xor, And, symbols

from symproof import AxiomSet
from symproof.library.circuits import boolean_entropy, r1cs_witness_check

axioms = AxiomSet(name="zk_circuit", axioms=())

# R1CS: a * b = c
# Witness: [1, a, b, c] = [1, 3, 7, 21]
A = sympy.Matrix([[0, 1, 0, 0]])  # selects a
B = sympy.Matrix([[0, 0, 1, 0]])  # selects b
C = sympy.Matrix([[0, 0, 0, 1]])  # selects c
witness = sympy.Matrix([1, 3, 7, 21])

r1cs_bundle = r1cs_witness_check(axioms, A, B, C, witness)
print("R1CS witness check: a * b = c")
print(f"  Witness: [1, a=3, b=7, c=21]")
print(f"  Constraint: 3 * 7 = 21")
print(f"  Status: {r1cs_bundle.proof_result.status.value}")
print(f"  Hash:   {r1cs_bundle.bundle_hash[:24]}...")

# Output entropy comparison
a, b = symbols("a b")

print("\nOutput entropy (information leakage)")

xor_bundle = boolean_entropy(axioms, Xor(a, b), [a, b])
print(f"  XOR(a,b): {xor_bundle.hypothesis.description}")

and_bundle = boolean_entropy(axioms, And(a, b), [a, b])
print(f"  AND(a,b): {and_bundle.hypothesis.description}")

print("\n  XOR has maximum entropy (1 bit) — no information leaked.")
print("  AND has lower entropy (0.811 bits) — output is biased toward False,")
print("  revealing information about the inputs.")
