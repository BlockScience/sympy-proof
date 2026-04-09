"""symproof — Deterministic proof writing with SymPy."""

__version__ = "0.1.0"

from symproof.builder import ProofBuilder
from symproof.bundle import ContradictionError, check_consistency, disprove, seal
from symproof.composite import (
    CompositeType,
    FixedPointType,
    ReservePairType,
    make_axiom_set,
)
from symproof.hashing import hash_axiom_set, hash_bundle, hash_disproof, hash_proof
from symproof.models import (
    Axiom,
    AxiomSet,
    Citation,
    Disproof,
    Hypothesis,
    Lemma,
    LemmaKind,
    LemmaResult,
    ProofBundle,
    ProofResult,
    ProofScript,
    ProofStatus,
)
from symproof.serialization import canonical_srepr, make_canonical_dict, restore_expr
from symproof.tactics import (
    SignedTerm,
    auto_lemma,
    signed_sum_lemmas,
    try_implication,
    try_query,
    try_simplify,
)
from symproof.verification import verify_lemma, verify_proof

__all__ = [
    "Axiom",
    "AxiomSet",
    "Citation",
    "CompositeType",
    "ContradictionError",
    "Disproof",
    "FixedPointType",
    "Hypothesis",
    "Lemma",
    "LemmaKind",
    "LemmaResult",
    "ProofBuilder",
    "ProofBundle",
    "ProofResult",
    "ProofScript",
    "ProofStatus",
    "ReservePairType",
    "SignedTerm",
    "auto_lemma",
    "canonical_srepr",
    "check_consistency",
    "disprove",
    "hash_axiom_set",
    "hash_bundle",
    "hash_disproof",
    "hash_proof",
    "make_axiom_set",
    "make_canonical_dict",
    "restore_expr",
    "seal",
    "signed_sum_lemmas",
    "try_implication",
    "try_query",
    "try_simplify",
    "verify_lemma",
    "verify_proof",
]
