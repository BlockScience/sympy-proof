#!/usr/bin/env python3
"""Channel capacity and KL divergence.

Scenario
--------
1. Binary symmetric channel (BSC) with crossover probability p = 1/10.
   Capacity C = 1 - H(p) bits per channel use.

2. KL divergence between a biased distribution [1/4, 1/2, 1/4] and
   uniform [1/3, 1/3, 1/3].  Verifies Gibbs' inequality: D(P||Q) >= 0.

Run: uv run python -m symproof.library.examples.information.03_channels
"""

from sympy import Rational

from symproof import AxiomSet
from symproof.library.information import binary_symmetric_channel, kl_divergence

axioms = AxiomSet(name="channel_theory", axioms=())
R = Rational

# BSC capacity
bsc_bundle = binary_symmetric_channel(axioms, R(1, 10))
print("Binary symmetric channel (p = 1/10)")
print(f"  {bsc_bundle.hypothesis.description}")
print(f"  Status: {bsc_bundle.proof_result.status.value}")

# KL divergence
kl_bundle = kl_divergence(
    axioms,
    [R(1, 4), R(1, 2), R(1, 4)],
    [R(1, 3), R(1, 3), R(1, 3)],
)
print("\nKL divergence: P=[1/4,1/2,1/4] vs Q=[1/3,1/3,1/3]")
print(f"  {kl_bundle.hypothesis.description}")
print(f"  Status: {kl_bundle.proof_result.status.value}")
for lr in kl_bundle.proof_result.lemma_results:
    mark = "pass" if lr.passed else "FAIL"
    print(f"    [{mark}] {lr.lemma_name}")
