#!/usr/bin/env python3
"""Verify LP feasibility: does a candidate point satisfy the constraints?

Scenario
--------
A production planning LP:
    min  x1 + 2*x2         (minimize cost)
    s.t. x1 + x2 + x3 = 4  (resource 1)
         2*x1 + x2     = 6  (resource 2)
         x >= 0

A solver returns x* = (3, 0, 1).  We verify it's feasible.

Run: uv run python -m symproof.library.examples.linopt.01_feasibility
"""

import sympy

from symproof import AxiomSet
from symproof.library.linopt import feasible_point

# No domain axioms needed — pure linear algebra.
axioms = AxiomSet(name="production_lp", axioms=())

A = sympy.Matrix([[1, 1, 1], [2, 1, 0]])
b = sympy.Matrix([4, 6])
x_star = sympy.Matrix([3, 0, 1])

bundle = feasible_point(axioms, A, b, x_star)

print("LP feasibility check")
print(f"  A = {A.tolist()}")
print(f"  b = {b.T.tolist()[0]}")
print(f"  x* = {x_star.T.tolist()[0]}")
print(f"  Status: {bundle.proof_result.status.value}")
print(f"  Hash:   {bundle.bundle_hash[:24]}...")
for lr in bundle.proof_result.lemma_results:
    mark = "pass" if lr.passed else "FAIL"
    print(f"    [{mark}] {lr.lemma_name}")
