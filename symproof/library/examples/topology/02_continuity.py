#!/usr/bin/env python3
"""Continuity and the intermediate value theorem.

Scenario
--------
1. Prove sin(x) is continuous at x = pi/4 via limits.
2. Apply IVT: x^2 - 2 has a root in (1, 2) because f(1) = -1 < 0
   and f(2) = 2 > 0.  The root is sqrt(2).

The IVT is the 1D special case of the Brouwer fixed point theorem:
a continuous function that changes sign must cross zero.

Run: uv run python -m symproof.library.examples.topology.02_continuity
"""

import sympy

from symproof import AxiomSet
from symproof.library.topology import continuous_at_point, intermediate_value

axioms = AxiomSet(name="analysis", axioms=())
x = sympy.Symbol("x")

# Continuity of sin at pi/4
cont_bundle = continuous_at_point(axioms, sympy.sin(x), x, sympy.pi / 4)
print("sin(x) is continuous at x = pi/4")
print("  lim sin(x) as x -> pi/4 = sin(pi/4) = sqrt(2)/2")
print(f"  Status: {cont_bundle.proof_result.status.value}")
print(f"  Hash:   {cont_bundle.bundle_hash[:24]}...")

# IVT: x^2 - 2 = 0 has a root in (1, 2)
ivt_bundle = intermediate_value(axioms, x**2 - 2, x, 1, 2, 0)
print("\nIVT: x^2 - 2 = 0 has a root in (1, 2)")
print("  f(1) = -1 < 0, f(2) = 2 > 0 (sign change)")
print("  Root: x = sqrt(2)")
print(f"  Status: {ivt_bundle.proof_result.status.value}")
print(f"  Hash:   {ivt_bundle.bundle_hash[:24]}...")
for lr in ivt_bundle.proof_result.lemma_results:
    mark = "pass" if lr.passed else "FAIL"
    print(f"    [{mark}] {lr.lemma_name}")
