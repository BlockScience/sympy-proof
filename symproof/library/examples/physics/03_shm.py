#!/usr/bin/env python3
"""Simple harmonic motion: solution verification and energy conservation.

Scenario
--------
A mass m on a spring with constant k oscillates as x(t) = A*cos(omega*t + phi)
where omega = sqrt(k/m).  Two proofs:

1. x(t) satisfies the SHM differential equation: x'' + omega^2 * x = 0
2. Total energy E = (1/2)*m*v^2 + (1/2)*k*x^2 is constant (dE/dt = 0)

What this proves
----------------
Pure calculus: second derivatives, chain rule, trigonometric identities.

Run: uv run python -m symproof.library.examples.physics.03_shm
"""

import sympy

from symproof import Axiom, AxiomSet
from symproof.library.physics import shm_energy_conservation, shm_solution_verify

t = sympy.Symbol("t")
A = sympy.Symbol("A")
omega = sympy.Symbol("omega")
phi = sympy.Symbol("phi")
m = sympy.Symbol("m", positive=True)
k = sympy.Symbol("k", positive=True)

axioms = AxiomSet(
    name="spring_mass",
    axioms=(
        Axiom(name="omega_relation", expr=sympy.Eq(omega**2, k / m),
              description="omega = sqrt(k/m) — natural frequency of the oscillator"),
    ),
)

# 1. Verify solution satisfies the ODE
ode_bundle = shm_solution_verify(axioms, A=A, omega=omega, phi=phi, t=t)
print("SHM solution verification")
print("  x(t) = A*cos(omega*t + phi)")
print("  x'' + omega^2*x = 0")
print(f"  Status: {ode_bundle.proof_result.status.value}")
print(f"  Hash:   {ode_bundle.bundle_hash[:24]}...")
for lr in ode_bundle.proof_result.lemma_results:
    mark = "pass" if lr.passed else "FAIL"
    print(f"    [{mark}] {lr.lemma_name}")

# 2. Energy conservation
energy_bundle = shm_energy_conservation(
    axioms, m=m, k=k, A=A, omega=omega, phi=phi, t=t,
)
print("\nSHM energy conservation")
print("  E = (1/2)*m*v^2 + (1/2)*k*x^2")
print("  dE/dt = 0 (using omega^2 = k/m)")
print(f"  Status: {energy_bundle.proof_result.status.value}")
print(f"  Hash:   {energy_bundle.bundle_hash[:24]}...")
