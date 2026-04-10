#!/usr/bin/env python3
"""Full LP optimality: primal + dual feasibility + strong duality.

Scenario
--------
Same production LP:
    min  [1, 2, 0]^T x
    s.t. [[1,1,1],[2,1,0]] x = [4, 6], x >= 0

Primal optimal:  x* = (3, 0, 1), objective = 3
Dual optimal:    y* = (0, 1/2), z* = (0, 3/2, 0)

We compose three proofs: primal feasibility + dual feasibility + strong
duality => x* is LP optimal.

Run: uv run python -m symproof.library.examples.linopt.02_optimality
"""

import sympy

from symproof import AxiomSet
from symproof.library.linopt import lp_optimal

axioms = AxiomSet(name="production_lp", axioms=())

c = sympy.Matrix([1, 2, 0])
A = sympy.Matrix([[1, 1, 1], [2, 1, 0]])
b = sympy.Matrix([4, 6])
x_star = sympy.Matrix([3, 0, 1])
y_star = sympy.Matrix([0, sympy.Rational(1, 2)])
z_star = sympy.Matrix([0, sympy.Rational(3, 2), 0])

bundle = lp_optimal(axioms, c, A, b, x_star, y_star, z_star)

print("LP optimality proof (composed)")
print(f"  Objective value: c^T x* = {(c.T * x_star)[0,0]}")
print(f"  Dual value:      b^T y* = {(b.T * y_star)[0,0]}")
print(f"  Status: {bundle.proof_result.status.value}")
print(f"  Hash:   {bundle.bundle_hash[:24]}...")
print(f"  Imported: {len(bundle.proof.imported_bundles)} sub-proofs")
for imp in bundle.proof.imported_bundles:
    print(f"    - {imp.hypothesis.name}: {imp.proof_result.status.value}")
