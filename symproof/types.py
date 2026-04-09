"""Semantic type aliases for SymPy expressions used throughout symproof."""

from __future__ import annotations

import sympy

SympyExpr = sympy.Basic
"""Any SymPy expression (general)."""

SympyBoolean = sympy.Basic
"""A SymPy expression expected to evaluate to a boolean (e.g. relational, logical)."""
