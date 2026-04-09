"""Runnable examples for symproof.library submodules.

Each subdirectory contains domain-specific examples. Run any
example as a module::

    uv run python -m symproof.library.examples.control.01_stability
    uv run python -m symproof.library.examples.convex.01_loss_function
    uv run python -m symproof.library.examples.defi.01_amm_swap_audit

Domains
-------
control/
    Stability (Routh-Hurwitz, Lyapunov), controllability,
    observability, and composition of multiple properties.

convex/
    Convexity certification (scalar, Hessian, strong convexity),
    DCP composition rules, and end-to-end portfolio optimization.

defi/
    AMM swap audit: rounding direction, phantom overflow,
    directional error chains, decimal mismatch, composed safety.

dip_routing/
    Reproves main results of Zargham, Ribeiro & Jadbabaie (2014)
    "Discounted Integral Priority Routing For Data Networks."
    Heavy ball equivalence, bounded gradient, Lagrangian structure,
    dual convergence, and queue stability.
"""
