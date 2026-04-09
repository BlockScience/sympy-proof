"""symproof.library — Reusable proof bundles for known SymPy gaps.

Organized like scipy: a ``core`` module ships with the package covering
general SymPy limitations, with topic submodules for domain-specific
proofs (``defi``, and eventually ``control``, ``crypto``, etc.).

Each entry is a **function** that accepts an ``AxiomSet`` (plus relevant
symbols) and returns a sealed ``ProofBundle``.  Import the bundle into
your proof via ``ProofBuilder.import_bundle()``.

Example::

    from symproof.library.defi import fee_complement_positive

    bundle = fee_complement_positive(my_axioms, fee_symbol=f)
    script = (
        ProofBuilder(my_axioms, h.name, name="p", claim="...")
        .import_bundle(bundle)
        .lemma(...)
        .build()
    )
"""

from symproof.library.control import (
    controllability_rank,
    hurwitz_second_order,
    hurwitz_third_order,
    lyapunov_stability,
    observability_rank,
    quadratic_invariant,
)
from symproof.library.core import max_ge_first, piecewise_collapse
from symproof.library.defi import (
    DecimalAwarePool,
    amm_output_positive,
    amm_product_nondecreasing,
    chain_error_bound,
    div_down,
    div_up,
    fee_complement_positive,
    mul_down,
    mul_up,
    no_phantom_overflow_check,
    phantom_overflow_check,
    rounding_bias_lemma,
    rounding_gap_lemma,
    safe_mul_div,
)

__all__ = [
    "DecimalAwarePool",
    "amm_output_positive",
    "amm_product_nondecreasing",
    "chain_error_bound",
    "controllability_rank",
    "div_down",
    "div_up",
    "fee_complement_positive",
    "hurwitz_second_order",
    "hurwitz_third_order",
    "lyapunov_stability",
    "max_ge_first",
    "mul_down",
    "mul_up",
    "no_phantom_overflow_check",
    "observability_rank",
    "phantom_overflow_check",
    "piecewise_collapse",
    "quadratic_invariant",
    "rounding_bias_lemma",
    "rounding_gap_lemma",
    "safe_mul_div",
]
