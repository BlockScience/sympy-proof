#!/usr/bin/env python3
"""Closed-loop stability for a satellite attitude controller.

Scenario
--------
You're designing a single-axis attitude controller for a small satellite.
The plant is a rigid body (moment of inertia J) with a reaction wheel
actuator.  You've chosen a PD controller and need to prove the closed
loop is asymptotically stable for your design review.

What this proves
----------------
The closed-loop characteristic polynomial has all roots in the left
half-plane (Routh-Hurwitz conditions satisfied).  This guarantees
asymptotic stability of the LINEAR, CONTINUOUS-TIME, NOMINAL model.

What this does NOT prove
------------------------
- Robustness to parameter uncertainty (J may vary ±20% on orbit)
- Stability of the discrete-time implementation (sample rate matters)
- Actuator saturation effects (reaction wheel has finite torque)
- Sensor noise rejection (star tracker + gyro noise model)
- Nonlinear dynamics (large-angle slews violate linearization)

What to do next
---------------
1. Bode/Nyquist analysis for gain and phase margins
2. Monte Carlo simulation with parameter uncertainty
3. Discretize the controller (Tustin/ZOH) and re-check stability
4. Hardware-in-the-loop testing with the flight processor

Run: uv run python -m symproof.library.examples.control.01_stability
"""

import sympy
from symproof import Axiom, AxiomSet
from symproof.library.control import closed_loop_stability

s = sympy.Symbol("s")

# ─── System: PD controller on rigid body ─────────────────────
#
#   Plant:      G(s) = 1/(J·s²)       (rigid body, single axis)
#   Controller: C(s) = Kp + Kd·s      (proportional-derivative)
#
#   Closed-loop char poly: J·s² + Kd·s + Kp
#   Hurwitz iff J > 0, Kd > 0, Kp > 0

J = sympy.Symbol("J", positive=True)      # moment of inertia [kg·m²]
Kp = sympy.Symbol("Kp", positive=True)    # proportional gain [N·m/rad]
Kd = sympy.Symbol("Kd", positive=True)    # derivative gain [N·m·s/rad]

axioms = AxiomSet(
    name="satellite_attitude_pd",
    axioms=(
        Axiom(name="inertia_positive", expr=J > 0),
        Axiom(name="proportional_gain_positive", expr=Kp > 0),
        Axiom(name="derivative_gain_positive", expr=Kd > 0),
    ),
)

bundle = closed_loop_stability(
    axioms,
    plant_num=sympy.Integer(1),
    plant_den=J * s**2,
    ctrl_num=Kp + Kd * s,
    ctrl_den=sympy.Integer(1),
    s=s,
    assumptions={
        "J": {"positive": True},
        "Kp": {"positive": True},
        "Kd": {"positive": True},
    },
)

print("Satellite single-axis attitude control (PD on rigid body)")
print(f"  Plant:      G(s) = 1/(J·s²)")
print(f"  Controller: C(s) = Kp + Kd·s")
print(f"  CL poly:    J·s² + Kd·s + Kp")
print(f"  Status:     {bundle.proof_result.status.value}")
print(f"  Hash:       {bundle.bundle_hash[:24]}...")
print()
print("  This proves: nominal linear continuous-time stability.")
print("  This does NOT prove: robustness, discrete-time stability,")
print("    actuator saturation handling, or sensor noise rejection.")
print("  Next: Bode margins, Monte Carlo, discretization analysis.")
