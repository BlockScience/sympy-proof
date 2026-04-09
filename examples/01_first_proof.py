#!/usr/bin/env python3
"""Your first symproof proof: e^(i*pi) + 1 = 0.

This is the simplest possible proof — one axiom set, one hypothesis,
one lemma, sealed into a reproducible bundle.

Run: uv run python examples/01_first_proof.py
"""

import sympy
from symproof import Axiom, AxiomSet, ProofBuilder, LemmaKind, seal

# Step 1: Declare your axioms — the truths you accept without proof.
axioms = AxiomSet(
    name="complex_analysis",
    axioms=(
        Axiom(name="i_squared", expr=sympy.Eq(sympy.I**2, -1)),
    ),
)

# Step 2: State what you want to prove, bound to those axioms.
hypothesis = axioms.hypothesis(
    "euler_identity",
    expr=sympy.Eq(sympy.exp(sympy.I * sympy.pi) + 1, 0),
)

# Step 3: Build a proof — a chain of lemmas that establish the claim.
script = (
    ProofBuilder(
        axioms, hypothesis.name,
        name="euler_proof",
        claim="Euler's identity: e^(i*pi) + 1 = 0",
    )
    .lemma(
        "evaluate",
        LemmaKind.EQUALITY,           # check: simplify(expr - expected) == 0
        expr=sympy.exp(sympy.I * sympy.pi) + 1,
        expected=sympy.Integer(0),
        description="SymPy evaluates e^(i*pi) to -1, so e^(i*pi)+1 = 0",
    )
    .build()
)

# Step 4: Seal the proof. This re-verifies every lemma and produces
# a deterministic SHA-256 hash. Same inputs always give the same hash.
bundle = seal(axioms, hypothesis, script)

print(f"Proved: {bundle.hypothesis.description or hypothesis.name}")
print(f"Hash:   {bundle.bundle_hash}")
print(f"Status: {bundle.proof_result.status.value}")

# The hash is your receipt. Anyone with the same axioms and proof
# script will get the exact same hash — that's deterministic proof.
