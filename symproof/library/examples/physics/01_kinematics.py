#!/usr/bin/env python3
"""Kinematics: derive velocity and acceleration from position.

Scenario
--------
A ball is thrown upward with initial velocity v0.  Under constant
gravitational acceleration g (downward), its position is:

    x(t) = v0*t - (1/2)*g*t^2

Differentiation gives velocity and confirms acceleration is constant.
The rotational analog (spinning wheel with constant angular acceleration)
uses the same mathematical structure.

What this proves
----------------
1. v(t) = dx/dt = v0 - g*t  (velocity from position)
2. a(t) = dv/dt = -g         (acceleration is constant)
3. omega(t) = dtheta/dt       (rotational analog)

Run: uv run python -m symproof.library.examples.physics.01_kinematics
"""

import sympy

from symproof import AxiomSet
from symproof.library.physics import constant_acceleration, rotational_kinematic

t = sympy.Symbol("t")

# ─── Linear kinematics: ball thrown upward ──────────────────────
v0 = sympy.Symbol("v_0", positive=True)  # positive = upward
g = sympy.Symbol("g", positive=True)     # positive = magnitude (not direction!)

# from_symbols() auto-generates axioms from symbol assumptions.
# v0 > 0 and g > 0 become formal axioms — even "obvious" physics
# facts must be stated explicitly, because the sign of g is a real
# source of errors (g is a positive magnitude; -g is the acceleration).
axioms = AxiomSet.from_symbols("projectile", v0, g, t)

bundle = constant_acceleration(axioms, x0=0, v0=v0, a=-g, t=t)

print("Projectile motion (constant acceleration)")
print(f"  x(t) = v0*t - (1/2)*g*t^2")
print(f"  v(t) = v0 - g*t")
print(f"  a(t) = -g")
print(f"  Status: {bundle.proof_result.status.value}")
print(f"  Hash:   {bundle.bundle_hash[:24]}...")
for lr in bundle.proof_result.lemma_results:
    mark = "pass" if lr.passed else "FAIL"
    print(f"    [{mark}] {lr.lemma_name}")

# ─── Rotational analog: spinning wheel ─────────────────────────
omega0 = sympy.Symbol("omega_0")
alpha = sympy.Symbol("alpha")

# No domain constraints needed — pure algebra.
# Empty axiom set works when the proof is just differentiation.
rot_axioms = AxiomSet(name="spinning_wheel", axioms=())

rot_bundle = rotational_kinematic(rot_axioms, theta0=0, omega0=omega0, alpha=alpha, t=t)

print("\nSpinning wheel (constant angular acceleration)")
print(f"  theta(t) = omega0*t + (1/2)*alpha*t^2")
print(f"  omega(t) = omega0 + alpha*t")
print(f"  Status: {rot_bundle.proof_result.status.value}")
print(f"  Hash:   {rot_bundle.bundle_hash[:24]}...")
