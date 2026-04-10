#!/usr/bin/env python3
"""Control engineering: prove closed-loop stability.

A real engineering workflow: given a plant and a controller, prove
the closed-loop system is stable. The library extracts the
characteristic polynomial and checks the Routh-Hurwitz conditions.

Run: uv run python examples/05_control_engineering.py
"""

import sympy

from symproof import Axiom, AxiomSet, seal, unevaluated
from symproof.library.control import (
    closed_loop_stability,
    lyapunov_from_system,
    quadratic_invariant,
)

s = sympy.Symbol("s")

# ─── Example 1: PD controller on a double integrator ───
#
# Plant:      G(s) = 1/s^2          (free mass)
# Controller: C(s) = Kp + Kd*s      (PD controller)
# Closed-loop char poly: s^2 + Kd*s + Kp
# Hurwitz iff Kd > 0 and Kp > 0 — which our axioms guarantee.

Kp = sympy.Symbol("Kp", positive=True)
Kd = sympy.Symbol("Kd", positive=True)

# Use unevaluated() to keep axiom expressions structural.
# Without it, Kp > 0 would collapse to True (Kp has positive=True).
with unevaluated():
    pd_axioms = AxiomSet(
        name="pd_controller",
        axioms=(
            Axiom(name="Kp_pos", expr=Kp > 0),
            Axiom(name="Kd_pos", expr=Kd > 0),
        ),
    )

stability = closed_loop_stability(
    pd_axioms,
    plant_num=sympy.Integer(1),
    plant_den=s**2,
    ctrl_num=Kp + Kd * s,
    ctrl_den=sympy.Integer(1),
    s=s,
    assumptions={"Kp": {"positive": True}, "Kd": {"positive": True}},
)

print("Example 1: PD on double integrator")
print(f"  Status: {stability.proof_result.status.value}")
print(f"  Hash:   {stability.bundle_hash[:24]}...")

# ─── Example 2: Lyapunov stability for a damped oscillator ───
#
# System: dx/dt = Ax where A = [[0, 1], [-k, -c]]
# The library CONSTRUCTS a Lyapunov function P by solving A^T P + PA = -I,
# then proves P is positive definite.  You just provide A.

k = sympy.Symbol("k", positive=True)
c = sympy.Symbol("c", positive=True)

A = sympy.Matrix([[0, 1], [-k, -c]])

with unevaluated():
    lyap_axioms = AxiomSet(
        name="oscillator",
        axioms=(
            Axiom(name="k_pos", expr=k > 0),
            Axiom(name="c_pos", expr=c > 0),
        ),
    )

lyap_bundle = lyapunov_from_system(lyap_axioms, A)

print("\nExample 2: Lyapunov stability for damped oscillator")
print(f"  Status: {lyap_bundle.proof_result.status.value}")
print(f"  Lemmas: {len(lyap_bundle.proof.lemmas)} "
      f"(4 equation entries + 2 positive-definiteness checks)")
print(f"  Hash:   {lyap_bundle.bundle_hash[:24]}...")

# ─── Example 3: Conservation law (from optimal control) ───
#
# In the Homicidal Chauffeur pursuit-evasion game, the costate
# norm ||p||^2 = p1^2 + p2^2 is conserved along trajectories:
#   dp1/dt = -p2*phi,  dp2/dt = p1*phi
#   d/dt(||p||^2) = 2*p1*(-p2*phi) + 2*p2*(p1*phi) = 0

p1 = sympy.Symbol("p_1", real=True)
p2 = sympy.Symbol("p_2", real=True)
phi = sympy.Symbol("phi", real=True)

hc_axioms = AxiomSet(
    name="costate_dynamics",
    axioms=(Axiom(name="dynamics_defined", expr=sympy.Eq(1, 1)),),
)

conservation = quadratic_invariant(
    hc_axioms,
    state_symbols=[p1, p2],
    state_dots=[-p2 * phi, p1 * phi],
    V=p1**2 + p2**2,
)

print("\nExample 3: Costate norm conservation")
print(f"  Status: {conservation.proof_result.status.value}")
print(f"  Hash:   {conservation.bundle_hash[:24]}...")
print(f"  Claim:  dV/dt = 0 where V = ||p||^2")
