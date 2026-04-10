"""Linear optimization examples for symproof.library.linopt.

Validates LP, ILP, and MILP solutions using symproof.
Each file is standalone and runnable::

    uv run python -m symproof.library.examples.linopt.01_feasibility

Examples
--------
01_feasibility
    Verify a solution satisfies Ax = b, x >= 0.
02_optimality
    Full LP optimality: primal + dual feasibility + strong duality.
03_integer
    ILP feasibility and LP relaxation bound.
"""
