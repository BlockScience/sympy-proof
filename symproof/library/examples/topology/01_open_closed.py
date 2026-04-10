#!/usr/bin/env python3
"""Open, closed, and compact sets; boundary computation.

Scenario
--------
Verify fundamental topological properties of intervals in R:
- (0, 1) is open
- [0, 1] is closed
- [0, 1] is compact (Heine-Borel: closed + bounded)
- boundary([0, 1]) = {0, 1}

These are prerequisites for the intermediate and extreme value
theorems, and ultimately for Brouwer's fixed point theorem.

Run: uv run python -m symproof.library.examples.topology.01_open_closed
"""

import sympy

from symproof import AxiomSet
from symproof.library.topology import (
    verify_boundary,
    verify_closed,
    verify_compact,
    verify_open,
)

# No domain axioms — pure set theory.
axioms = AxiomSet(name="real_line_topology", axioms=())

# Open set
open_bundle = verify_open(axioms, sympy.Interval.open(0, 1))
print("(0, 1) is open")
print(f"  Status: {open_bundle.proof_result.status.value}")
print(f"  Hash:   {open_bundle.bundle_hash[:24]}...")

# Closed set
closed_bundle = verify_closed(axioms, sympy.Interval(0, 1))
print("\n[0, 1] is closed")
print(f"  Status: {closed_bundle.proof_result.status.value}")
print(f"  Hash:   {closed_bundle.bundle_hash[:24]}...")

# Compact set (Heine-Borel)
compact_bundle = verify_compact(axioms, sympy.Interval(0, 1))
print("\n[0, 1] is compact (closed + bounded)")
print(f"  Status: {compact_bundle.proof_result.status.value}")
print(f"  Hash:   {compact_bundle.bundle_hash[:24]}...")

# Boundary
boundary_bundle = verify_boundary(
    axioms, sympy.Interval(0, 1), sympy.FiniteSet(0, 1),
)
print("\nboundary([0, 1]) = {0, 1}")
print(f"  Status: {boundary_bundle.proof_result.status.value}")
print(f"  Hash:   {boundary_bundle.bundle_hash[:24]}...")
