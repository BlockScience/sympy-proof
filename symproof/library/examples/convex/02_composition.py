#!/usr/bin/env python3
"""Composition rules: building convex functions from simpler ones.

Scenario
--------
Your objective function isn't a textbook example — it's a composition
like exp(||Ax - b||^2) or log(sum(exp(c_i^T x))).  You need to prove
it's convex to justify using a convex solver.

CVXPY does this automatically via DCP (Disciplined Convex Programming)
rules.  symproof lets you prove these rules explicitly, which matters
when you're using a custom solver, a non-standard formulation, or need
the proof as a traceable artifact.

What this proves
----------------
- DCP composition: f convex nondecreasing + g convex => f(g(x)) convex
- Nonneg weighted sum: alpha*f + beta*g is convex if f, g are convex
- Conjugate convexity: f*(y) is always convex (Fenchel conjugate)

What this does NOT prove
------------------------
- That the DCP rules cover ALL convex functions (they don't — DCP is
  sufficient but not necessary)
- Convexity of the full problem (objective + constraints together)
- That the composition is numerically well-conditioned

Run: uv run python -m symproof.library.examples.convex.02_composition
"""

import sympy

from symproof import Axiom, AxiomSet
from symproof.library.convex import (
    conjugate_function,
    convex_composition,
    convex_sum,
)

x = sympy.Symbol("x", real=True)
t = sympy.Symbol("t", real=True)

trivial = AxiomSet(
    name="composition_proofs",
    axioms=(Axiom(name="defined", expr=sympy.Eq(1, 1)),),
)

# ─── DCP composition: exp(x^2) ──────────────────────────────
#
#  f(t) = exp(t)  — convex and nondecreasing
#  g(x) = x^2     — convex
#  h(x) = f(g(x)) = exp(x^2)  — convex by DCP rule
#
#  The library proves: f'' >= 0, f' >= 0, g'' >= 0, h'' >= 0

print("DCP composition: exp(x^2)")
bundle_dcp = convex_composition(
    trivial, sympy.exp(t), x**2, t, x,
)
print(f"  Status: {bundle_dcp.proof_result.status.value}")
print("  Proof chain:")
for lr in bundle_dcp.proof_result.lemma_results:
    print(f"    [{lr.lemma_name}] {lr.passed}")
print(f"  Hash: {bundle_dcp.bundle_hash[:24]}...")


# ─── Nonneg weighted sum: a*x^2 + b*exp(x) ─────────────────
#
#  Both x^2 and exp(x) are convex. Their nonneg weighted sum
#  is convex.  This is the most basic composition rule.

print("\nWeighted sum: a*x^2 + b*exp(x)")
a = sympy.Symbol("a", positive=True)
b = sympy.Symbol("b", positive=True)

sum_axioms = AxiomSet(
    name="weighted_sum",
    axioms=(
        Axiom(name="a_pos", expr=a > 0),
        Axiom(name="b_pos", expr=b > 0),
    ),
)
bundle_sum = convex_sum(
    sum_axioms, [x**2, sympy.exp(x)], [a, b], [x],
    assumptions={"a": {"positive": True}, "b": {"positive": True}},
)
print(f"  Status: {bundle_sum.proof_result.status.value}")


# ─── Conjugate function: f(x) = x^2/2 => f*(y) = y^2/2 ─────
#
#  The Fenchel conjugate f*(y) = sup_x(xy - f(x)) is always convex.
#  For f = x^2/2, the conjugate is y^2/2 (self-conjugate).
#
#  Useful for: dual problem formulation, Bregman divergences,
#  mirror descent analysis.

print("\nConjugate function: f(x) = x^2/2")
y = sympy.Symbol("y", real=True)
bundle_conj = conjugate_function(trivial, x**2 / 2, x, y)
print("  f*(y) should be y^2/2")
print(f"  Status: {bundle_conj.proof_result.status.value}")
print(f"  Hash: {bundle_conj.bundle_hash[:24]}...")

print()
print("These composition proofs are the building blocks for certifying")
print("complex objective functions before passing them to a solver.")
