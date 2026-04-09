#!/usr/bin/env python3
"""Proving series and sum identities.

Shows how symproof handles infinite series (via SymPy's .doit())
and closed-form sums with symbolic parameters.

Run: uv run python examples/02_series_and_sums.py
"""

import sympy
from symproof import Axiom, AxiomSet, ProofBuilder, LemmaKind, seal

k = sympy.Symbol("k", integer=True, nonnegative=True)
n = sympy.Symbol("n", integer=True, positive=True)

axioms = AxiomSet(
    name="series_axioms",
    axioms=(Axiom(name="pi_positive", expr=sympy.pi > 0),),
)

# --- Geometric series: Sum(1/2^k, k=0..inf) = 2 ---

h1 = axioms.hypothesis(
    "geometric_series",
    expr=sympy.Eq(
        sympy.Sum(sympy.Rational(1, 2)**k, (k, 0, sympy.oo)),
        sympy.Integer(2),
    ),
)
s1 = (
    ProofBuilder(axioms, h1.name, name="geo", claim="Sum(1/2^k) = 2")
    .lemma("evaluate", LemmaKind.EQUALITY,
           expr=sympy.Sum(sympy.Rational(1, 2)**k, (k, 0, sympy.oo)),
           expected=sympy.Integer(2))
    .build()
)
b1 = seal(axioms, h1, s1)
print(f"Geometric series:  {b1.bundle_hash[:20]}...")

# --- Basel problem: Sum(1/k^2, k=1..inf) = pi^2/6 ---

k_pos = sympy.Symbol("k", integer=True, positive=True)
h2 = axioms.hypothesis(
    "basel_problem",
    expr=sympy.Eq(
        sympy.Sum(1/k_pos**2, (k_pos, 1, sympy.oo)),
        sympy.pi**2 / 6,
    ),
)
s2 = (
    ProofBuilder(axioms, h2.name, name="basel", claim="Sum(1/k^2) = pi^2/6")
    .lemma("evaluate", LemmaKind.EQUALITY,
           expr=sympy.Sum(1/k_pos**2, (k_pos, 1, sympy.oo)),
           expected=sympy.pi**2 / 6)
    .build()
)
b2 = seal(axioms, h2, s2)
print(f"Basel problem:     {b2.bundle_hash[:20]}...")

# --- Gauss's sum: Sum(k, k=1..n) = n(n+1)/2 ---
# This one is symbolic in n — it works for ALL positive integers n.

h3 = axioms.hypothesis(
    "gauss_sum",
    expr=sympy.Eq(sympy.Sum(k, (k, 1, n)), n*(n+1)/2),
)
s3 = (
    ProofBuilder(axioms, h3.name, name="gauss", claim="1+2+...+n = n(n+1)/2")
    .lemma("closed_form", LemmaKind.EQUALITY,
           expr=sympy.Sum(k, (k, 1, n)),
           expected=n*(n+1)/2,
           assumptions={"n": {"integer": True, "positive": True}})
    .build()
)
b3 = seal(axioms, h3, s3)
print(f"Gauss's sum:       {b3.bundle_hash[:20]}...")

# Note the advisories — series proofs go through SymPy's .doit()
# fallback, which the advisory system flags for human review.
for adv in b1.proof_result.advisories:
    print(f"\n  Advisory: {adv}")
