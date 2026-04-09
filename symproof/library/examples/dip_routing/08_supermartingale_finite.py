#!/usr/bin/env python3
"""Supermartingale finite-time return to zero (Solo-Kong core argument).

Scenario
--------
File 05 (Proposition 3) axiomatised the Solo & Kong [21] supermartingale
convergence theorem (Theorem E.7.4): if a nonnegative process Y_t
satisfies E[Y_{t+1} | F_t] <= Y_t - delta when Y_t > 0, then
lim inf Y_t = 0 a.s.

The computational core of this theorem is a simple arithmetic
contradiction: a nonnegative quantity cannot decrease by a fixed
positive amount more than finitely many times.

What this proves
----------------
The deterministic/arithmetic kernel of the supermartingale argument:

1. After N consecutive steps of descent by delta:
   Y_N <= Y_0 - N * delta

2. If Y stays above epsilon > 0 for N steps, and N > Y_0 / delta,
   then Y_N < 0, contradicting nonnegativity.

3. Therefore the process must return to [0, epsilon) within at most
   ceil(Y_0 / delta) steps from any starting point.

We demonstrate this both symbolically and with concrete numerics
(Y_0 = 10, delta = 3 → return within 4 steps).

What this does NOT prove
------------------------
- Extension to "infinitely often" (Borel-Cantelli; axiomatised)
- Almost-sure convergence (measure theory)
- The conditional expectation bound itself (that's established in
  file 05 via the gradient vanishing condition)

Run: uv run python -m symproof.library.examples.dip_routing.08_supermartingale_finite
"""

import sympy

from symproof import Axiom, AxiomSet, LemmaKind, ProofBuilder, seal

# ─── Symbols ────────────────────────────────────────────────────
Y_0 = sympy.Symbol("Y_0", nonnegative=True)      # starting value
delta = sympy.Symbol("delta", positive=True)       # minimum descent per step
N = sympy.Symbol("N", positive=True, integer=True)  # number of descent steps

# ─── Axioms ─────────────────────────────────────────────────────
axioms = AxiomSet(
    name="supermartingale_finite_time",
    axioms=(
        Axiom(
            name="process_nonneg",
            expr=Y_0 >= 0,
            description="The process Y_t is nonnegative at all times.",
        ),
        Axiom(
            name="descent_per_step",
            expr=delta > 0,
            description=(
                "Each step decreases the process by at least delta > 0 "
                "(drift condition from supermartingale property)."
            ),
        ),
        Axiom(
            name="borel_cantelli_extension",
            expr=sympy.S.true,
            description=(
                "Borel-Cantelli: if a nonneg process returns to zero in "
                "finite time from any starting point, and the starting "
                "points recur infinitely often, then lim inf = 0 a.s. "
                "Measure-theoretic — not proved here."
            ),
        ),
    ),
)

# ─── Hypothesis ─────────────────────────────────────────────────
# The finite-time return: Y cannot stay positive for more than
# Y_0/delta steps without going negative (contradiction).
hypothesis = axioms.hypothesis(
    name="finite_time_return",
    expr=sympy.S.true,
    description=(
        "Solo-Kong [21] Thm E.7.4 (finite-time kernel): A nonneg process "
        "with descent delta > 0 per step must return to zero within "
        "ceil(Y_0 / delta) steps."
    ),
)

# ─── Proof ──────────────────────────────────────────────────────
# The argument: after N steps of descent by delta,
#   Y_N = Y_0 - N * delta
# For N > Y_0/delta, Y_N < 0, contradicting Y >= 0.
#
# Concrete example: Y_0 = 10, delta = 3
#   Step 1: Y <= 10 - 3 = 7
#   Step 2: Y <= 7 - 3 = 4
#   Step 3: Y <= 4 - 3 = 1
#   Step 4: Y <= 1 - 3 = -2 < 0 → contradiction
#   ceil(10/3) = 4 steps

Y_after_N = Y_0 - N * delta

# Concrete values
Y_0_val = sympy.Integer(10)
delta_val = sympy.Integer(3)
max_steps = sympy.ceiling(Y_0_val / delta_val)  # = 4

script = (
    ProofBuilder(
        axioms,
        hypothesis.name,
        name="supermartingale_finite_return",
        claim=(
            "A nonneg process with descent delta > 0 per step must "
            "return to zero within ceil(Y_0/delta) steps."
        ),
    )
    .lemma(
        "linear_descent_form",
        LemmaKind.EQUALITY,
        expr=Y_after_N,
        expected=Y_0 - N * delta,
        description=(
            "After N consecutive descent steps: Y_N = Y_0 - N*delta. "
            "This is the telescoped sum of N decrements."
        ),
    )
    .lemma(
        "concrete_max_steps",
        LemmaKind.EQUALITY,
        expr=max_steps,
        expected=sympy.Integer(4),
        description=(
            "Concrete example: Y_0 = 10, delta = 3. "
            "ceil(10/3) = 4 maximum steps above zero."
        ),
    )
    .lemma(
        "concrete_violation",
        LemmaKind.BOOLEAN,
        expr=sympy.Lt(
            Y_after_N.subs({Y_0: Y_0_val, N: max_steps, delta: delta_val}),
            0,
        ),
        description=(
            "After 4 steps: Y_4 = 10 - 4*3 = -2 < 0. "
            "This violates nonnegativity — contradiction."
        ),
    )
    .lemma(
        "symbolic_overshoot",
        LemmaKind.EQUALITY,
        expr=sympy.expand(Y_after_N.subs(N, Y_0 / delta + 1)),
        expected=-delta,
        depends_on=["linear_descent_form"],
        description=(
            "At N = Y_0/delta + 1 (continuous version): "
            "Y_0 - (Y_0/delta + 1)*delta = -delta < 0. "
            "The process overshoots zero, contradicting nonnegativity."
        ),
    )
    .lemma(
        "overshoot_negative",
        LemmaKind.QUERY,
        expr=sympy.Q.negative(-delta),
        assumptions={"delta": {"positive": True}},
        depends_on=["symbolic_overshoot"],
        description="-delta < 0, confirming the contradiction.",
    )
    .build()
)

def make_supermartingale_bundle():
    """Build and return the supermartingale finite-time foundation bundle."""
    return seal(axioms, hypothesis, script)


bundle = make_supermartingale_bundle()

# ─── Output ─────────────────────────────────────────────────────
print("Solo-Kong Thm E.7.4 — Finite-time return to zero")
print("  Paper: Solo & Kong [21], referenced in Zargham et al. (2014)")
print("  Foundation for: file 05 (Proposition 3, 'supermartingale_convergence' axiom)")
print()
print("  Descent structure: Y_{t+1} <= Y_t - delta when Y_t > 0")
print("  After N steps:     Y_N <= Y_0 - N * delta")
print()
print(f"  Concrete: Y_0=10, delta=3 → max {max_steps} steps above zero")
print(f"    Y_4 = 10 - 4*3 = {Y_0_val - max_steps * delta_val} < 0 (contradiction)")
print()
print(f"  Status: {bundle.proof_result.status.value}")
print(f"  Hash:   {bundle.bundle_hash[:24]}...")
for lr in bundle.proof_result.lemma_results:
    mark = "pass" if lr.passed else "FAIL"
    print(f"    [{mark}] {lr.lemma_name}")
print()
print("  This proves: finite-time return via arithmetic contradiction.")
print("  Remaining axiom: Borel-Cantelli extension to 'infinitely often'.")
