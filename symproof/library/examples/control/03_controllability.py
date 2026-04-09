#!/usr/bin/env python3
"""Controllability and observability for a spacecraft attitude system.

Scenario
--------
You're checking whether your ADCS (Attitude Determination and Control
System) can reach any attitude from any initial condition (controllable),
and whether your sensor suite can reconstruct the full state
(observable).  This matters for Kalman filter design — an unobservable
mode means your filter will diverge for that state.

What this proves
----------------
- Controllability: the controllability Gramian C·C^T is nonsingular,
  meaning the actuator can steer the system to any state.
- Observability: the observability Gramian O^T·O is nonsingular,
  meaning the sensor can distinguish all states.

These are NECESSARY conditions for most control/estimation designs.

What this does NOT prove
------------------------
- That the system is PRACTICALLY controllable (actuator torque limits
  may make some maneuvers infeasible in finite time)
- Stabilizability or detectability (weaker conditions that suffice
  when you only need to stabilize unstable modes)
- That your Kalman filter will converge fast enough (observability
  says "eventually" — it doesn't bound convergence rate)
- Controllability/observability of the discrete-time implementation

What to do next
---------------
1. Compute controllability/observability Gramians numerically
   to check conditioning (near-singular = practically hard)
2. Design state feedback / LQR using the controllable system
3. Design Kalman filter / observer using the observable system
4. Verify with simulation that estimation errors converge

Run: uv run python -m symproof.library.examples.control.03_controllability
"""

import sympy
from symproof import Axiom, AxiomSet
from symproof.library.control import controllability_rank, observability_rank

# ─── Spacecraft single-axis: x = [theta, omega] ─────────────
#
#   State: theta (angle), omega (angular rate)
#   Input: u (reaction wheel torque)
#   Plant: d/dt [theta, omega] = [[0,1],[0,0]] [theta,omega] + [[0],[1/J]] u
#
#   Output options:
#     Star tracker: measures theta       → C = [[1, 0]]
#     Gyroscope:    measures omega       → C = [[0, 1]]
#     Both:         measures both        → C = [[1,0],[0,1]]

A = sympy.Matrix([[0, 1], [0, 0]])     # double integrator (free rigid body)
B = sympy.Matrix([[0], [1]])            # normalized (J = 1 for simplicity)

axioms = AxiomSet(
    name="spacecraft_adcs",
    axioms=(Axiom(name="dynamics", expr=sympy.Eq(1, 1)),),
)

# ─── Controllability: can the wheel steer to any attitude? ───
ctrl = controllability_rank(axioms, A, B)
print("Spacecraft ADCS controllability:")
print(f"  Controllability matrix: [B | AB] = [[0,1],[1,0]]")
print(f"  det(C·C^T) ≠ 0?  {ctrl.proof_result.status.value}")

# ─── Observability with star tracker only (theta) ───────────
C_star = sympy.Matrix([[1, 0]])
obs_star = observability_rank(axioms, A, C_star)
print(f"\nStar tracker only (measures theta):")
print(f"  Observability matrix: [C; CA] = [[1,0],[0,1]]")
print(f"  Observable?  {obs_star.proof_result.status.value}")
print(f"  → Kalman filter CAN estimate omega from theta measurements")

# ─── Observability with gyro only (omega) ───────────────────
C_gyro = sympy.Matrix([[0, 1]])
print(f"\nGyroscope only (measures omega):")
try:
    obs_gyro = observability_rank(axioms, A, C_gyro)
    print(f"  Observable?  {obs_gyro.proof_result.status.value}")
except ValueError:
    print(f"  NOT observable (seal rejected — det = 0)")
    print(f"  → Kalman filter CANNOT estimate theta from gyro alone")
    print(f"  → Need star tracker or combined sensor suite")

# ─── Observability with both sensors ────────────────────────
C_both = sympy.Matrix([[1, 0], [0, 1]])
obs_both = observability_rank(axioms, A, C_both)
print(f"\nBoth sensors (star tracker + gyro):")
print(f"  Observable?  {obs_both.proof_result.status.value}")
print(f"  → Full state estimation possible (and well-conditioned)")

print()
print("Summary: controllable with reaction wheel, observable with")
print("star tracker (alone or combined), NOT observable with gyro alone.")
print("This informs sensor suite selection for the ADCS design.")
