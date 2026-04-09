"""Runnable examples for symproof.library submodules.

Each subdirectory contains domain-specific examples. Run any
example as a module::

    uv run python -m symproof.library.examples.control.01_stability
    uv run python -m symproof.library.examples.convex.01_convexity

Domains
-------
control/
    Stability (Routh-Hurwitz, Lyapunov), controllability,
    observability, and composition of multiple properties.

convex/
    Convexity certification (scalar, Hessian, strong convexity),
    DCP composition rules, and end-to-end portfolio optimization.
"""
