#!/usr/bin/env python3
"""Lyapunov stability: auto-construct V(x) = x^T P x for a damped system.

Scenario
--------
You have a linearized model of a vibration isolation mount (mass on
spring with viscous damper).  You need a Lyapunov function to certify
stability — for instance, as part of a structural dynamics review or
to satisfy a requirement that says "demonstrate asymptotic stability
by analytical means."

What this proves
----------------
A quadratic Lyapunov function V(x) = x^T P x exists such that
dV/dt < 0 for all x ≠ 0.  This is a SUFFICIENT condition for
asymptotic stability of the equilibrium x = 0.

The library:
  1. Solves A^T P + P A = -Q for symmetric P  (Q defaults to I)
  2. Verifies A^T P + P A + Q = 0 entry-by-entry
  3. Proves P is positive definite via Sylvester's criterion

What this does NOT prove
------------------------
- Stability of the nonlinear system (only valid near equilibrium)
- Size of the region of attraction (requires nonlinear analysis)
- Performance (settling time, overshoot — Lyapunov gives no bounds)
- Robustness to parameter changes (P was constructed for specific A)

What to do next
---------------
1. Check eigenvalues of A numerically for specific parameter values
2. Estimate region of attraction for the nonlinear model
3. Simulate time response to verify settling time meets requirements
4. Sensitivity analysis: how does stability margin change with k, c?

Run: uv run python -m symproof.library.examples.control.02_lyapunov
"""

import sympy
from symproof import Axiom, AxiomSet
from symproof.library.control import lyapunov_from_system

# ─── Damped oscillator: m·x'' + c·x' + k·x = 0 ─────────────
#
# State-space: d/dt [x, x'] = A [x, x']
#   A = [[0, 1], [-k/m, -c/m]]
#
# We normalize m = 1 here.  For a real system you'd substitute
# your actual parameters (e.g., k=100 N/m, c=5 N·s/m).

k = sympy.Symbol("k", positive=True)   # stiffness [N/m]
c = sympy.Symbol("c", positive=True)   # damping [N·s/m]

A = sympy.Matrix([
    [0, 1],
    [-k, -c],
])

axioms = AxiomSet(
    name="vibration_isolator",
    axioms=(
        Axiom(name="stiffness_positive", expr=k > 0),
        Axiom(name="damping_positive", expr=c > 0),
    ),
)

bundle = lyapunov_from_system(axioms, A)

print("Vibration isolator: A = [[0, 1], [-k, -c]]")
print(f"  Status: {bundle.proof_result.status.value}")
print(f"  Proof chain ({len(bundle.proof.lemmas)} lemmas):")
for lr in bundle.proof_result.lemma_results:
    tag = "PASS" if lr.passed else "FAIL"
    print(f"    [{tag}] {lr.lemma_name}")

# Show the constructed Lyapunov function
print(f"\n  Constructed P (from solving A^T P + PA = -I):")
# Extract P from the proof by re-solving (the bundle stores the proof,
# not the intermediate P matrix — by design, the proof is the artifact)
p_00 = sympy.Symbol("p_00")
p_01 = sympy.Symbol("p_01")
p_11 = sympy.Symbol("p_11")
P_sym = sympy.Matrix([[p_00, p_01], [p_01, p_11]])
sol = sympy.solve(
    [eq for eq in (A.T * P_sym + P_sym * A + sympy.eye(2))],
    [p_00, p_01, p_11],
)
if sol:
    P_solved = P_sym.subs(sol)
    print(f"    P = {P_solved}")
    print(f"    V(x) = x^T · P · x is a Lyapunov function")

print(f"\n  Hash: {bundle.bundle_hash[:24]}...")
print()
print("  This proves: asymptotic stability of x=0 for the linear model.")
print("  This does NOT prove: nonlinear stability, region of attraction,")
print("    settling time bounds, or robustness to parameter variation.")
print("  Next: eigenvalue analysis, time-domain simulation, sensitivity.")
