#!/usr/bin/env python3
"""ILP feasibility and LP relaxation bound.

Scenario
--------
A scheduling ILP (variables must be integer):
    min  3*x1 + 2*x2
    s.t. x1 + x2 >= 4   (rewrite as -x1 - x2 + s = -4, s >= 0)
         x1, x2 >= 0, integer

LP relaxation optimal: x_LP = (0, 4), value = 8
ILP optimal:           x_IP = (0, 4), value = 8  (tight in this case)

Alternative ILP with gap:
    LP relaxation value = 7.5,  ILP value = 8

Run: uv run python -m symproof.library.examples.linopt.03_integer
"""

import sympy

from symproof import AxiomSet
from symproof.library.linopt import integer_feasible, lp_relaxation_bound

axioms = AxiomSet(name="scheduling_ilp", axioms=())

# Simple ILP: x1 + x2 = 4, x >= 0, integer
A = sympy.Matrix([[1, 1]])
b = sympy.Matrix([4])
x_star = sympy.Matrix([1, 3])

bundle = integer_feasible(axioms, A, b, x_star)
print("ILP feasibility")
print(f"  x* = {x_star.T.tolist()[0]} (integer)")
print(f"  Ax* = b: {(A * x_star)[0,0]} = {b[0]}")
print(f"  Status: {bundle.proof_result.status.value}")
print(f"  Hash:   {bundle.bundle_hash[:24]}...")
for lr in bundle.proof_result.lemma_results:
    mark = "pass" if lr.passed else "FAIL"
    print(f"    [{mark}] {lr.lemma_name}")

# LP relaxation bound
print("\nLP relaxation bound")
bound_bundle = lp_relaxation_bound(
    axioms,
    lp_value=sympy.Rational(15, 2),  # 7.5
    ilp_value=sympy.Integer(8),
)
print(f"  LP relaxation value: 7.5")
print(f"  ILP optimal value:   8")
print(f"  Gap: 8 - 7.5 = 0.5 >= 0")
print(f"  Status: {bound_bundle.proof_result.status.value}")
print(f"  Hash:   {bound_bundle.bundle_hash[:24]}...")
