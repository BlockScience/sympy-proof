"""DeFi mechanism examples for symproof.library.defi.

Graduated examples modeling real security audit workflows.
Each file is standalone and runnable::

    uv run python -m symproof.library.examples.defi.01_amm_swap_audit

Examples
--------
01_amm_swap_audit
    Full AMM audit walkthrough: rounding direction, phantom overflow,
    directional error accumulation, decimal mismatch, and composed
    safety bundles.  Models the thought process of a security auditor
    reviewing a Uniswap-style constant-product pool.
"""
