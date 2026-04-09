"""Convex optimization examples for symproof.library.convex.

Graduated examples with realistic scenarios, failure cases, and
explicit limitations.  Each file is standalone::

    uv run python -m symproof.library.examples.convex.01_loss_function
    uv run python -m symproof.library.examples.convex.02_regularization
    uv run python -m symproof.library.examples.convex.03_portfolio

Examples
--------
01_loss_function
    Certify ML loss functions (squared error, cross-entropy, log-barrier).
    Failure case: cubic loss correctly rejected.
02_regularization
    Ridge regression: regularization strengthens convexity.
    Shows lambda's mathematical effect on convergence and uniqueness.
03_portfolio
    Markowitz portfolio certification with uniqueness proof.
    Failure case: perfectly correlated assets breaks strict convexity.
"""
