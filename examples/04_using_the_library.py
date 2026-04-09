#!/usr/bin/env python3
"""Using the proof library: import pre-built proofs as building blocks.

symproof ships with a library of reusable proofs for common results
that are either tedious to re-derive or require workarounds for SymPy
limitations. You import a sealed bundle and build on top of it.

Run: uv run python examples/04_using_the_library.py
"""

import sympy
from symproof import Axiom, AxiomSet, ProofBuilder, LemmaKind, seal
from symproof.library.defi import fee_complement_positive, amm_output_positive

# --- Scenario: prove an AMM swap produces positive output ---
#
# SymPy can't prove this directly because it can't reason about
# bounded intervals: 0 < fee < 1 implies 1-fee > 0.  The library
# decomposes this into steps SymPy *can* verify.

Rx = sympy.Symbol("R_x", positive=True)
Ry = sympy.Symbol("R_y", positive=True)
fee = sympy.Symbol("f")
dx = sympy.Symbol("dx", positive=True)

axioms = AxiomSet(
    name="amm_pool",
    axioms=(
        Axiom(name="rx_pos", expr=Rx > 0),
        Axiom(name="ry_pos", expr=Ry > 0),
        Axiom(name="fee_pos", expr=fee > 0),
        Axiom(name="fee_lt_1", expr=fee < 1),
        Axiom(name="dx_pos", expr=dx > 0),
    ),
)

# Step 1: Use the library to prove the fee complement is positive.
# This is a one-liner — the library handles the decomposition internally.
fee_bundle = fee_complement_positive(axioms, fee)
print(f"Library proof: 1-fee > 0     hash={fee_bundle.bundle_hash[:16]}...")

# Step 2: Use the library to prove swap output is positive.
# This internally imports the fee_complement proof and builds on it.
output_bundle = amm_output_positive(axioms, Rx, Ry, fee, dx)
print(f"Library proof: dy > 0        hash={output_bundle.bundle_hash[:16]}...")

# Step 3: Import the library proof into YOUR proof.
# Now you can build higher-level claims on top of proven foundations.
hypothesis = axioms.hypothesis(
    "pool_safe",
    expr=sympy.And(Rx > 0, dx > 0),
    description="Pool is safe: positive reserves and output",
)

my_script = (
    ProofBuilder(
        axioms, hypothesis.name,
        name="pool_safety_proof",
        claim="AMM pool produces positive output",
    )
    .import_bundle(output_bundle)    # re-verified automatically
    .lemma(
        "reserves_positive",
        LemmaKind.QUERY,
        expr=sympy.Q.positive(Rx),
        assumptions={"R_x": {"positive": True}},
        description="Reserves are positive",
    )
    .build()
)

my_bundle = seal(axioms, hypothesis, my_script)
print(f"\nMy proof:          hash={my_bundle.bundle_hash[:16]}...")
print(f"  Imported: {len(my_script.imported_bundles)} bundle(s)")
print(f"  Lemmas:   {len(my_script.lemmas)} local + "
      f"{len(my_bundle.proof_result.lemma_results) - len(my_script.lemmas)} import")

# When you seal, the imported bundle is RE-VERIFIED from scratch.
# For exploratory work, you can skip re-verification:
#   verify_proof(my_script, trust_imports=True)
# But seal() always re-verifies — no shortcuts for production proofs.
