#!/usr/bin/env python3
"""Gravitational potential from the inverse-square force law.

Scenario
--------
Newton's law of gravitation: F(r) = -GMm/r^2 (attractive).
The gravitational potential energy is U(r) = -GMm/r.

This proves: F(r) = -dU/dr — the force is the negative gradient
of the potential.

What this proves
----------------
Integration/differentiation: -d/dr(-GMm/r) = -GMm/r^2 = F(r).

Run: uv run python -m symproof.library.examples.physics.04_gravitation
"""

import sympy

from symproof import Axiom, AxiomSet, unevaluated
from symproof.library.physics import gravitational_potential_from_force

G = sympy.Symbol("G", positive=True)
M = sympy.Symbol("M", positive=True)
m = sympy.Symbol("m", positive=True)
r = sympy.Symbol("r", positive=True)

with unevaluated():
    axioms = AxiomSet(
        name="newtonian_gravity",
        axioms=(
            Axiom(name="G_positive", expr=G > 0,
                  description="Gravitational constant G > 0"),
            Axiom(name="M_positive", expr=M > 0,
                  description="Central mass M > 0"),
            Axiom(name="m_positive", expr=m > 0,
                  description="Test mass m > 0"),
            Axiom(name="r_positive", expr=r > 0,
                  description="Radial distance r > 0"),
        ),
    )

bundle = gravitational_potential_from_force(axioms, G=G, M=M, m=m, r=r)

print("Gravitational potential from force law")
print(f"  F(r) = -GMm/r^2")
print(f"  U(r) = -GMm/r")
print(f"  F = -dU/dr (verified)")
print(f"  Status: {bundle.proof_result.status.value}")
print(f"  Hash:   {bundle.bundle_hash[:24]}...")
