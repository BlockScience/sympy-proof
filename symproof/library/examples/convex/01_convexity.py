#!/usr/bin/env python3
"""Proving convexity: scalar, Hessian, and strong convexity.

Scenario
--------
You're formulating an optimization problem and need to verify that
your objective function is convex before handing it to a solver.
This is especially important when:
  - You're writing a custom objective (not a standard CVXPY atom)
  - You need to certify the formulation for a regulated application
  - You're debugging "solver says infeasible" — is the problem actually convex?

What this proves
----------------
- Scalar: f''(x) >= 0 (convex on the real line or a convex subset)
- Hessian PSD: all leading principal minors >= 0 (multivariate convexity)
- Strong convexity: Hessian eigenvalues >= m > 0 (implies unique minimum)

What this does NOT prove
------------------------
- Convexity of the CONSTRAINT set (you need separate feasibility proofs)
- That the solver will find the optimum efficiently (depends on conditioning)
- Global optimality for non-convex problems (convexity is a precondition)
- Numerical stability of the solver on this specific instance

What to do next
---------------
1. Verify constraint convexity separately
2. Check problem conditioning (strong convexity param / Lipschitz constant)
3. Solve with CVXPY/Mosek/Gurobi
4. Validate solution against physical constraints

Run: uv run python -m symproof.library.examples.convex.01_convexity
"""

import sympy

from symproof import Axiom, AxiomSet
from symproof.library.convex import convex_hessian, convex_scalar, strongly_convex

x = sympy.Symbol("x", real=True)
y = sympy.Symbol("y", real=True)

trivial = AxiomSet(
    name="convexity_proofs",
    axioms=(Axiom(name="defined", expr=sympy.Eq(1, 1)),),
)

# ─── Scalar convexity: common functions ──────────────────────

print("Scalar convexity (f''(x) >= 0):")

for name, f, ax, asm in [
    ("exp(x)", sympy.exp(x), trivial, {}),
    ("x^2", x**2, trivial, {}),
    ("x^4", x**4, trivial, {}),
    ("-log(x)",
     -sympy.log(sympy.Symbol("x", positive=True)),
     AxiomSet(name="p", axioms=(
         Axiom(name="x_pos", expr=sympy.Symbol("x", positive=True) > 0),
     )),
     {"x": {"positive": True}}),
]:
    bundle = convex_scalar(ax, f, next(iter(f.free_symbols)), assumptions=asm)
    print(f"  {name:12s}  {bundle.proof_result.status.value}")


# ─── Hessian PSD: multivariate quadratic ────────────────────

print("\nHessian PSD (Sylvester's criterion):")

f_2d = x**2 + x * y + y**2  # Eigenvalues: 1 and 3
bundle_hess = convex_hessian(trivial, f_2d, [x, y])
print(f"  x^2 + xy + y^2:  {bundle_hess.proof_result.status.value}")
print("  Hessian = [[2, 1], [1, 2]], eigenvalues = 1, 3")
print("  Minors: det([2]) = 2 >= 0, det([[2,1],[1,2]]) = 3 >= 0")


# ─── Strong convexity: implies unique minimizer ─────────────

print("\nStrong convexity (Hessian - m*I is PSD):")

# x^2 + y^2: Hessian = 2I, so 2-strongly convex
bundle_strong = strongly_convex(
    trivial, x**2 + y**2, [x, y], sympy.Integer(2),
)
print(f"  x^2 + y^2 is 2-strongly convex: "
      f"{bundle_strong.proof_result.status.value}")
print("  Implication: unique global minimizer exists")
print("  Implication: gradient descent converges linearly")
print(f"  Hash: {bundle_strong.bundle_hash[:24]}...")
