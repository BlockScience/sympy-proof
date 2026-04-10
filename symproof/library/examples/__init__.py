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

physics/
    High school physics with calculus: kinematics, energy,
    simple harmonic motion, gravitation.

linopt/
    Linear/integer optimization: LP feasibility, duality,
    optimality certificates, ILP integrality.

topology/
    Point-set topology: open/closed sets, compactness,
    continuity, intermediate and extreme value theorems.

circuits/
    Boolean circuits with ZK applications: gate verification,
    circuit equivalence, R1CS witness, information leakage.

information/
    Shannon information theory: entropy, mutual information,
    KL divergence, channel capacity.

dip_routing/
    Reproves main results of Zargham, Ribeiro & Jadbabaie (2014)
    "Discounted Integral Priority Routing For Data Networks."
    Heavy ball equivalence, bounded gradient, Lagrangian structure,
    dual convergence, and queue stability.
"""
