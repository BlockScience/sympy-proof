#!/usr/bin/env python3
"""Circuit equivalence: XOR equals its AND/OR/NOT decomposition.

Scenario
--------
A common circuit optimization question: is the XOR gate functionally
equivalent to (a AND NOT b) OR (NOT a AND b)?  This matters for
hardware synthesis where XOR may not be a primitive gate.

What this proves
----------------
Xor(a, b) == Or(And(a, Not(b)), And(Not(a), b)) for all inputs.

Run: uv run python -m symproof.library.examples.circuits.02_equivalence
"""

from sympy import And, Not, Or, Xor, symbols

from symproof import AxiomSet
from symproof.library.circuits import circuit_equivalence

a, b = symbols("a b")
axioms = AxiomSet(name="gate_equivalence", axioms=())

# XOR decomposition
xor_native = Xor(a, b)
xor_decomposed = Or(And(a, Not(b)), And(Not(a), b))

bundle = circuit_equivalence(axioms, xor_native, xor_decomposed, [a, b])

print("Circuit equivalence: XOR decomposition")
print("  A: Xor(a, b)")
print("  B: Or(And(a, Not(b)), And(Not(a), b))")
print(f"  Equivalent: {bundle.proof_result.status.value}")
print(f"  Hash: {bundle.bundle_hash[:24]}...")
