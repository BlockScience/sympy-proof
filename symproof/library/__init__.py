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
from symproof.library.circuits import (
    boolean_entropy,
    circuit_equivalence,
    circuit_output,
    circuit_satisfies,
    gate_truth_table,
    r1cs_witness_check,
)
from symproof.library.core import max_ge_first, piecewise_collapse
from symproof.library.topology import (
    continuous_at_point,
    extreme_value,
    intermediate_value,
    verify_boundary,
    verify_closed,
    verify_compact,
    verify_open,
)
from symproof.library.linopt import (
    complementary_slackness,
    dual_feasible,
    feasible_point,
    integer_feasible,
    lp_optimal,
    lp_relaxation_bound,
    strong_duality,
)
from symproof.library.physics import (
    constant_acceleration,
    gravitational_potential_from_force,
    impulse_momentum,
    rotational_kinematic,
    shm_energy_conservation,
    shm_solution_verify,
    work_energy_theorem,
)
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
    "boolean_entropy",
    "circuit_equivalence",
    "circuit_output",
    "circuit_satisfies",
    "closed_loop_stability",
    "complementary_slackness",
    "conjugate_function",
    "controllability_rank",
    "convex_composition",
    "convex_hessian",
    "convex_scalar",
    "convex_sum",
    "directional_chain_error",
    "envelope_theorem",
    "feasible_point",
    "div_down",
    "dual_feasible",
    "div_up",
    "fee_complement_positive",
    "gain_margin",
    "gate_truth_table",
    "gravitational_potential_from_force",
    "gp_to_convex",
    "hurwitz_second_order",
    "impulse_momentum",
    "integer_feasible",
    "lp_optimal",
    "lp_relaxation_bound",
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
    "constant_acceleration",
    "rotational_kinematic",
    "r1cs_witness_check",
    "rounding_bias_lemma",
    "rounding_gap_lemma",
    "safe_mul_div",
    "shm_energy_conservation",
    "shm_solution_verify",
    "strong_duality",
    "strongly_convex",
    "continuous_at_point",
    "extreme_value",
    "intermediate_value",
    "unique_minimizer",
    "verify_boundary",
    "verify_closed",
    "verify_compact",
    "verify_open",
    "work_energy_theorem",
]
