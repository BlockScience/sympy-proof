#!/usr/bin/env python3
"""Extreme value theorem: continuous f on [a,b] attains its max and min.

Scenario
--------
f(x) = x^3 - 3x + 1 on [-2, 2].

Critical points: f'(x) = 3x^2 - 3 = 0 at x = +/-1.
Candidates: x = -2, -1, 1, 2.
Values:     f(-2) = -1, f(-1) = 3, f(1) = -1, f(2) = 3.
Max = 3 at x = -1 and x = 2.  Min = -1 at x = -2 and x = 1.

The EVT guarantees these extrema exist because f is continuous
and [-2, 2] is compact (closed + bounded).

Run: uv run python -m symproof.library.examples.topology.03_extreme_value
"""

import sympy

from symproof import AxiomSet
from symproof.library.topology import extreme_value, verify_compact

axioms = AxiomSet(name="evt_demo", axioms=())
x = sympy.Symbol("x")

# First prove the domain is compact
domain = sympy.Interval(-2, 2)
compact_bundle = verify_compact(axioms, domain)
print(f"Domain [-2, 2] is compact: {compact_bundle.proof_result.status.value}")

# Then apply EVT
f = x**3 - 3 * x + 1
evt_bundle = extreme_value(axioms, f, x, -2, 2)
print("\nEVT: f(x) = x^3 - 3x + 1 on [-2, 2]")
print(f"  Status: {evt_bundle.proof_result.status.value}")
print(f"  Hash:   {evt_bundle.bundle_hash[:24]}...")
print(f"  {evt_bundle.hypothesis.description}")
for lr in evt_bundle.proof_result.lemma_results:
    mark = "pass" if lr.passed else "FAIL"
    print(f"    [{mark}] {lr.lemma_name}")
