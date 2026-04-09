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

from symproof.library.core import max_ge_first, piecewise_collapse
from symproof.library.defi import (
    amm_output_positive,
    amm_product_nondecreasing,
    fee_complement_positive,
)

__all__ = [
    "amm_output_positive",
    "amm_product_nondecreasing",
    "fee_complement_positive",
    "max_ge_first",
    "piecewise_collapse",
]
