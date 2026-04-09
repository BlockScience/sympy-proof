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
    closed_loop_stability,
    controllability_rank,
    gain_margin,
    hurwitz_second_order,
    hurwitz_third_order,
    lyapunov_from_system,
    lyapunov_stability,
    observability_rank,
    quadratic_invariant,
)
from symproof.library.convex import (
    conjugate_function,
    convex_composition,
    convex_hessian,
    convex_scalar,
    convex_sum,
    gp_to_convex,
    strongly_convex,
    unique_minimizer,
)
from symproof.library.core import max_ge_first, piecewise_collapse
from symproof.library.envelope import envelope_theorem
from symproof.library.defi import (
    DecimalAwarePool,
    RoundingFavor,
    RoundingStep,
    amm_output_positive,
    amm_product_nondecreasing,
    chain_error_bound,
    directional_chain_error,
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
    "RoundingFavor",
    "RoundingStep",
    "amm_output_positive",
    "amm_product_nondecreasing",
    "chain_error_bound",
    "closed_loop_stability",
    "conjugate_function",
    "controllability_rank",
    "convex_composition",
    "convex_hessian",
    "convex_scalar",
    "convex_sum",
    "directional_chain_error",
    "envelope_theorem",
    "div_down",
    "div_up",
    "fee_complement_positive",
    "gain_margin",
    "gp_to_convex",
    "hurwitz_second_order",
    "hurwitz_third_order",
    "lyapunov_from_system",
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
    "strongly_convex",
    "unique_minimizer",
]
