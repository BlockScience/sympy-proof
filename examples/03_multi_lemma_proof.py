#!/usr/bin/env python3
"""Multi-lemma proof: the AM-GM inequality.

Not everything can be proved in one step. This example shows how to
decompose a non-trivial claim into a chain of lemmas that SymPy can
verify individually.

    Claim: (a+b)/2 >= sqrt(a*b)  for a, b > 0

    Strategy:
        1. (a-b)^2 >= 0            (squares are nonneg — Q-system)
        2. (a-b)^2 = a^2-2ab+b^2   (expand — equality)
        3. (a+b)^2 - 4ab = (a-b)^2  (rearrange — equality)

    Chain logic: step 3 says (a+b)^2 >= 4ab, so (a+b)/2 >= sqrt(ab).

Run: uv run python examples/03_multi_lemma_proof.py
"""

import sympy
from symproof import Axiom, AxiomSet, ProofBuilder, LemmaKind, seal

a = sympy.Symbol("a", positive=True)
b = sympy.Symbol("b", positive=True)

axioms = AxiomSet(
    name="positive_reals",
    axioms=(
        Axiom(name="a_positive", expr=a > 0),
        Axiom(name="b_positive", expr=b > 0),
    ),
)

hypothesis = axioms.hypothesis(
    "am_gm",
    expr=sympy.Ge((a + b) / 2, sympy.sqrt(a * b)),
)

script = (
    ProofBuilder(
        axioms, hypothesis.name,
        name="am_gm_proof",
        claim="(a+b)/2 >= sqrt(a*b)",
    )
    # Lemma 1: the key insight — any square is nonnegative
    .lemma(
        "square_nonneg",
        LemmaKind.QUERY,
        expr=sympy.Q.nonnegative((a - b)**2),
        assumptions={"a": {"positive": True}, "b": {"positive": True}},
        description="(a-b)^2 >= 0",
    )
    # Lemma 2: expand the square
    .lemma(
        "expand_square",
        LemmaKind.EQUALITY,
        expr=(a - b)**2,
        expected=a**2 - 2*a*b + b**2,
        depends_on=["square_nonneg"],
        description="(a-b)^2 = a^2 - 2ab + b^2",
    )
    # Lemma 3: connect to AM-GM form
    .lemma(
        "rearrange",
        LemmaKind.EQUALITY,
        expr=(a + b)**2 - 4*a*b,
        expected=(a - b)**2,
        depends_on=["expand_square"],
        description="(a+b)^2 - 4ab = (a-b)^2 >= 0  =>  (a+b)^2 >= 4ab",
    )
    .build()
)

bundle = seal(axioms, hypothesis, script)

print("AM-GM inequality proved!")
print(f"  Lemma chain: {[l.name for l in script.lemmas]}")
print(f"  Hash: {bundle.bundle_hash[:24]}...")

# This is the pattern for any non-trivial proof: break the claim
# into steps where each step is something simplify() or ask() handles.
