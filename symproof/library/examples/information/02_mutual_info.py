#!/usr/bin/env python3
"""Mutual information: independent vs correlated sources.

Scenario
--------
1. Independent sources: P(X,Y) = P(X)*P(Y), so I(X;Y) = 0.
2. Perfectly correlated: X = Y, so I(X;Y) = H(X) = 1 bit.

I(X;Y) quantifies how much knowing Y reduces uncertainty about X.
For independent sources, knowing Y tells you nothing. For perfectly
correlated sources, knowing Y tells you everything.

Run: uv run python -m symproof.library.examples.information.02_mutual_info
"""

from sympy import Rational

from symproof import AxiomSet
from symproof.library.information import mutual_information

axioms = AxiomSet(name="mutual_info_demo", axioms=())
R = Rational

# Independent: uniform joint distribution
independent = mutual_information(
    axioms,
    [[R(1, 4), R(1, 4)],
     [R(1, 4), R(1, 4)]],
)
print("Independent sources: P(X,Y) = 1/4 for all (x,y)")
print(f"  {independent.hypothesis.description}")
print(f"  Status: {independent.proof_result.status.value}")

# Perfectly correlated: X = Y
correlated = mutual_information(
    axioms,
    [[R(1, 2), 0],
     [0, R(1, 2)]],
)
print("\nPerfectly correlated: P(X=Y) = 1")
print(f"  {correlated.hypothesis.description}")
print(f"  Status: {correlated.proof_result.status.value}")
