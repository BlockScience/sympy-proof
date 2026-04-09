#!/usr/bin/env python3
"""Danskin's envelope theorem applied to the DIP quadratic objective.

Scenario
--------
File 03 (Proposition 1) axiomatised Danskin's theorem as an external
result to establish dual function differentiability.  This file proves
the computational content of that theorem for the paper's concrete
quadratic objective (eq. 35):

    f(r, lambda) = -(1/2) r^2 + beta * r + lambda * r

where lambda (the dual variable / priority) plays the role of the
parameter theta.  The max-value function is:

    V(lambda) = max_{r >= 0} f(r, lambda)

Danskin's theorem states: dV/dlambda = df/dlambda|_{r = r*(lambda)},
i.e., the gradient of the dual function equals the optimal routing rate.

What this proves
----------------
Using the library-level envelope_theorem():
1. FOC: df/dr = 0 at r* = beta + lambda (unique maximiser)
2. Strict concavity: d^2f/dr^2 = -1 < 0
3. Hessian invertibility (IFT): d^2f/dr^2 != 0
4. Envelope identity: dV/dlambda = r* = beta + lambda

This narrows the `danskin_theorem` axiom from file 03 to just the
topological assumption that the maximiser exists on a compact domain.

What this does NOT prove
------------------------
- Maximiser existence on compact domain (topology; axiomatised)
- The general-case Danskin's theorem for arbitrary f

Run: uv run python -m symproof.library.examples.dip_routing.06_danskin_concrete
"""

import sympy

from symproof import Axiom, AxiomSet
from symproof.library.envelope import envelope_theorem

# ─── Symbols ────────────────────────────────────────────────────
r = sympy.Symbol("r", nonnegative=True)       # routing rate
lam = sympy.Symbol("lambda")                   # dual variable (priority)
beta = sympy.Symbol("beta", positive=True)     # linear reward coefficient

# ─── Concrete objective (eq. 35 from the paper) ────────────────
# f(r, lambda) = -(1/2) r^2 + beta * r + lambda * r
# The lambda * r term is the Lagrangian coupling.
f = -sympy.Rational(1, 2) * r**2 + beta * r + lam * r

# ─── Axioms ─────────────────────────────────────────────────────
axioms = AxiomSet(
    name="danskin_dip_quadratic",
    axioms=(
        Axiom(
            name="beta_positive",
            expr=beta > 0,
            description="Linear reward coefficient beta > 0.",
        ),
        Axiom(
            name="maximiser_exists",
            expr=sympy.S.true,
            description=(
                "The maximiser r*(lambda) exists on the compact feasible "
                "set [0, C_max]. Topological assumption — not proved here."
            ),
        ),
    ),
)

def make_danskin_bundle():
    """Build and return the Danskin envelope theorem bundle for the DIP quadratic."""
    return envelope_theorem(axioms, f, r, lam)


# ─── Seal ───────────────────────────────────────────────────────
bundle = make_danskin_bundle()

# ─── Derived quantities for display ────────────────────────────
r_star = sympy.solve(sympy.diff(f, r), r)[0]  # beta + lambda
V = sympy.simplify(f.subs(r, r_star))

# ─── Output ─────────────────────────────────────────────────────
print("Danskin's Envelope Theorem — DIP quadratic objective")
print("  Paper: Zargham, Ribeiro & Jadbabaie, Globecom 2014, eq. 35")
print("  Foundation for: file 03 (Proposition 1, 'danskin_theorem' axiom)")
print()
print(f"  f(r, lambda) = -(1/2)r^2 + beta*r + lambda*r")
print(f"  r*(lambda)   = {r_star}")
print(f"  V(lambda)    = {V}")
print(f"  dV/dlambda   = {sympy.diff(V, lam)}  (= r*, the envelope identity)")
print()
print(f"  Status: {bundle.proof_result.status.value}")
print(f"  Hash:   {bundle.bundle_hash[:24]}...")
for lr in bundle.proof_result.lemma_results:
    mark = "pass" if lr.passed else "FAIL"
    print(f"    [{mark}] {lr.lemma_name}")
print()
print("  This proves: FOC, strict concavity, IFT, envelope identity.")
print("  Remaining axiom: maximiser existence on compact domain.")
