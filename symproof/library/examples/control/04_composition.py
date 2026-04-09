#!/usr/bin/env python3
"""Composition: full verification of a vibration isolator design.

Scenario
--------
You've designed a vibration isolation mount for a payload (mass m on
spring k with damper c).  The design review requires evidence that:
  1. The open-loop plant is controllable (actuator can affect all states)
  2. The plant is observable (sensors can reconstruct all states)
  3. The closed-loop system is Lyapunov stable

Each property is proved independently with its own sealed bundle, then
all three are composed into a single proof with one deterministic hash.
This hash is your design evidence artifact — traceable, reproducible,
and re-verifiable by any reviewer.

What this proves
----------------
Controllability + observability + Lyapunov stability of the LINEAR,
CONTINUOUS-TIME, NOMINAL model.  These are the analytical prerequisites
before you move to simulation and testing.

What this does NOT prove (and what to do about it)
--------------------------------------------------
- Robustness to ±20% mass uncertainty → do mu-analysis or Monte Carlo
- Discrete controller stability → discretize and re-check
- Actuator/sensor limitations → include saturation in simulation
- Nonlinear large-displacement behavior → nonlinear sim + test
- Fatigue/thermal effects on k, c → environmental testing

When would you use this?
------------------------
- Preliminary Design Review (PDR): "we have analytical stability evidence"
- Requirements tracing: map this proof hash to a stability requirement
- Regression: re-run after parameter changes to confirm stability held

Run: uv run python -m symproof.library.examples.control.04_composition
"""

import sympy

from symproof import Axiom, AxiomSet, LemmaKind, ProofBuilder, seal
from symproof.library.control import (
    controllability_rank,
    lyapunov_from_system,
    observability_rank,
)

# ─── Vibration isolator model ────────────────────────────────
#
#   m·x'' + c·x' + k·x = u     (force input from actuator)
#   y = x                        (displacement sensor)
#
# State-space (normalized to m=1 for symbolic tractability):
#   A = [[0, 1], [-k, -c]]
#   B = [[0], [1]]
#   C = [[1, 0]]

k = sympy.Symbol("k", positive=True)   # stiffness
c = sympy.Symbol("c", positive=True)   # damping

A = sympy.Matrix([[0, 1], [-k, -c]])
B = sympy.Matrix([[0], [1]])
C = sympy.Matrix([[1, 0]])

axioms = AxiomSet(
    name="vibration_isolator",
    axioms=(
        Axiom(name="stiffness_positive", expr=k > 0),
        Axiom(name="damping_positive", expr=c > 0),
    ),
)

# ─── Prove each property ─────────────────────────────────────

print("Vibration isolator: m·x'' + c·x' + k·x = u, y = x")
print()

print("1. Controllability (can the actuator affect all states?)")
ctrl = controllability_rank(axioms, A, B)
print(f"   {ctrl.proof_result.status.value}")

print("2. Observability (can the sensor reconstruct all states?)")
obs = observability_rank(axioms, A, C)
print(f"   {obs.proof_result.status.value}")

print("3. Lyapunov stability (is the equilibrium asymptotically stable?)")
lyap = lyapunov_from_system(axioms, A)
print(f"   {lyap.proof_result.status.value}")

# ─── Compose into one artifact ───────────────────────────────

print()
print("4. Compose into single design evidence bundle:")

hypothesis = axioms.hypothesis(
    "isolator_verified",
    expr=sympy.And(k > 0, c > 0),
    description="Isolator is controllable, observable, and Lyapunov stable",
)

script = (
    ProofBuilder(
        axioms, hypothesis.name,
        name="isolator_verification",
        claim="Vibration isolator: controllable + observable + stable",
    )
    .import_bundle(ctrl)
    .import_bundle(obs)
    .import_bundle(lyap)
    .lemma(
        "parameters_physical",
        LemmaKind.QUERY,
        expr=sympy.Q.positive(k * c),
        assumptions={"k": {"positive": True}, "c": {"positive": True}},
        description="Physical parameters are positive",
    )
    .build()
)

bundle = seal(axioms, hypothesis, script)

print(f"   {bundle.proof_result.status.value}")
print(f"   Hash: {bundle.bundle_hash}")
print(f"   Imported: {len(script.imported_bundles)} sub-proofs")
n_import = sum(
    1 for lr in bundle.proof_result.lemma_results
    if lr.lemma_name.startswith("import:")
)
n_local = len(bundle.proof_result.lemma_results) - n_import
print(f"   Verified: {n_import} import(s) + {n_local} local lemma(s)")

print()
print("This hash is your design evidence artifact.")
print("Trace it to your stability requirement in the V&V matrix.")
print("Any reviewer can re-run this script and get the same hash.")
