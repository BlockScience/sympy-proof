"""Convex optimization examples for symproof.library.convex.

Graduated examples showing how to certify optimization problem
formulations before solving them.  Each file is standalone::

    uv run python -m symproof.library.examples.convex.01_convexity
    uv run python -m symproof.library.examples.convex.02_composition
    uv run python -m symproof.library.examples.convex.03_portfolio

Examples
--------
01_convexity
    Prove a function is convex (scalar, Hessian, strong convexity)
02_composition
    DCP composition rules, conjugate functions, weighted sums
03_portfolio
    End-to-end: certify a Markowitz portfolio problem formulation
"""
