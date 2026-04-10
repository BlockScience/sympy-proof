#!/usr/bin/env python3
"""Work-energy and impulse-momentum theorems.

Scenario
--------
A block of mass m starts at rest and is pushed by a constant force F = m*a
for time t.  The work done by the force equals the change in kinetic energy,
and the impulse equals the change in momentum.

What this proves
----------------
1. W = F*x = (1/2)*m*v^2 - 0 = delta(KE)  (work-energy theorem)
2. J = F*t = m*v - 0 = delta(p)             (impulse-momentum theorem)

Run: uv run python -m symproof.library.examples.physics.02_energy
"""

import sympy

from symproof import Axiom, AxiomSet
from symproof.library.physics import impulse_momentum, work_energy_theorem

t = sympy.Symbol("t")
F = sympy.Symbol("F")
m = sympy.Symbol("m")
a = sympy.Symbol("a")

axioms = AxiomSet(
    name="constant_force",
    axioms=(
        Axiom(name="newton_second_law", expr=sympy.Eq(F, m * a),
              description="F = ma (Newton's second law)"),
    ),
)

# Work-energy theorem
we_bundle = work_energy_theorem(axioms, F=F, m=m, v0=0, a=a, t=t)
print("Work-energy theorem (block pushed from rest)")
print("  W = F*x = (1/2)*m*v^2 = delta(KE)")
print(f"  Status: {we_bundle.proof_result.status.value}")
print(f"  Hash:   {we_bundle.bundle_hash[:24]}...")

# Impulse-momentum theorem
im_bundle = impulse_momentum(axioms, F=F, m=m, v0=0, a=a, t=t)
print("\nImpulse-momentum theorem")
print("  J = F*t = m*v = delta(p)")
print(f"  Status: {im_bundle.proof_result.status.value}")
print(f"  Hash:   {im_bundle.bundle_hash[:24]}...")
