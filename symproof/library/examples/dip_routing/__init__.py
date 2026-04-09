"""DIP routing examples — computational proofs from Zargham, Ribeiro & Jadbabaie (2014).

Reproves the main results of "Discounted Integral Priority Routing For
Data Networks" (Globecom 2014) using symproof's deterministic proof
framework.  Each file is standalone and runnable::

    uv run python -m symproof.library.examples.dip_routing.01_heavy_ball_equivalence

Examples
--------
01_heavy_ball_equivalence
    Lemma 1: DIP priority update is algebraically equivalent to heavy
    ball momentum applied to soft backpressure.
02_bounded_gradient
    Lemma 2: The stochastic gradient is uniformly bounded.
03_lagrangian_structure
    Proposition 1: Dual function differentiability and reverse
    waterfilling Lagrangian maximizers.
04_dual_convergence
    Proposition 2: Stochastic heavy ball dual variables converge a.s.
05_queue_stability
    Proposition 3 + Corollary 1: All queues empty infinitely often
    with probability one.
"""
