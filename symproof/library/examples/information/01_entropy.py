#!/usr/bin/env python3
"""Shannon entropy of discrete distributions.

Scenario
--------
Compute the entropy of:
1. A fair coin: H = 1 bit (maximum uncertainty)
2. A biased die: [1/4, 1/2, 1/4] → H = 3/2 bits
3. Binary entropy function H(p) at p = 1/4

All values are exact — no floating-point approximations.

Run: uv run python -m symproof.library.examples.information.01_entropy
"""

from sympy import Rational

from symproof import AxiomSet
from symproof.library.information import binary_entropy_func, entropy

axioms = AxiomSet(name="information_theory", axioms=())
R = Rational

# Fair coin
coin_bundle = entropy(axioms, [R(1, 2), R(1, 2)])
print("Fair coin: P = [1/2, 1/2]")
print(f"  {coin_bundle.hypothesis.description}")
print(f"  Status: {coin_bundle.proof_result.status.value}")

# Biased source
biased_bundle = entropy(axioms, [R(1, 4), R(1, 2), R(1, 4)])
print("\nBiased source: P = [1/4, 1/2, 1/4]")
print(f"  {biased_bundle.hypothesis.description}")
print(f"  Status: {biased_bundle.proof_result.status.value}")

# Binary entropy function
bef_bundle = binary_entropy_func(axioms, R(1, 4))
print("\nBinary entropy H(1/4):")
print(f"  {bef_bundle.hypothesis.description}")
print(f"  Status: {bef_bundle.proof_result.status.value}")
