#!/usr/bin/env python3
"""Gate verification: truth tables and specific outputs.

Scenario
--------
Verify basic boolean gates against their truth tables.  This is the
foundation for circuit verification — if individual gates are wrong,
the whole circuit is wrong.

What this proves
----------------
1. XOR gate matches its 4-row truth table
2. AND gate produces False for inputs (1, 0)

Run: uv run python -m symproof.library.examples.circuits.01_gates
"""

from sympy import And, Xor, symbols

from symproof import AxiomSet
from symproof.library.circuits import circuit_output, gate_truth_table

a, b = symbols("a b")
axioms = AxiomSet(name="boolean_gates", axioms=())

# XOR truth table
xor_bundle = gate_truth_table(
    axioms,
    Xor(a, b),
    [a, b],
    [False, True, True, False],  # 00→0, 01→1, 10→1, 11→0
)
print("XOR truth table verification")
print(f"  Status: {xor_bundle.proof_result.status.value}")
print(f"  Hash:   {xor_bundle.bundle_hash[:24]}...")
print(f"  Rows verified: {len(xor_bundle.proof.lemmas)}")

# AND gate for specific input
and_bundle = circuit_output(
    axioms,
    And(a, b),
    {a: True, b: False},
    False,
)
print("\nAND(1, 0) = 0")
print(f"  Status: {and_bundle.proof_result.status.value}")
print(f"  Hash:   {and_bundle.bundle_hash[:24]}...")
