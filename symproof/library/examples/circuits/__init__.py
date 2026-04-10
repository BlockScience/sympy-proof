"""Boolean circuit examples for symproof.library.circuits.

Gate verification, circuit equivalence, and ZK witness satisfaction.
Each file is standalone and runnable::

    uv run python -m symproof.library.examples.circuits.01_gates

Examples
--------
01_gates
    Truth table verification, gate output checking.
02_equivalence
    Prove XOR equals its decomposition into AND/OR/NOT.
03_zk_witness
    R1CS witness satisfaction for a simple ZK circuit; output entropy.
04_information_leakage
    Cross-domain composition: circuit correctness + Shannon entropy
    proves whether a gate leaks information about its inputs.
"""
