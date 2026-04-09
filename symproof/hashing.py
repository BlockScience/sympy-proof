"""Deterministic SHA-256 hashing for axiom set identity and proof binding.

Pure functions — no state.

Hash scope
----------
``hash_axiom_set``
    Hashes the axiom set's declared components via ``canonical_dict()``.
    Two axiom sets with identical declarations produce identical hashes
    regardless of construction order.

``hash_proof``
    Hashes the lemma chain content (names, kinds, srepr'd expressions,
    expected values, assumptions) together with the ``axiom_set_hash``.
    Binds the proof to a specific axiom context.

``hash_bundle``
    Hashes the triple (axiom_set_hash, hypothesis canonical form, proof_hash)
    to produce the final bundle identity.

The evidence chain
------------------

    AxiomSet.canonical_dict()
        -> hash_axiom_set(axiom_set) -> axiom_set_hash
              -> hash_proof(script) -> proof_hash
                    -> hash_bundle(...) -> bundle_hash
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import sympy

from symproof.serialization import make_canonical_dict

if TYPE_CHECKING:
    from symproof.models import AxiomSet, Hypothesis, ProofScript


def _serialize_for_hash(data: dict) -> str:
    """JSON-serialize a dict deterministically for hashing."""
    return json.dumps(data, sort_keys=True, default=str)


def hash_axiom_set(axiom_set: AxiomSet) -> str:
    """Deterministic SHA-256 hash of an axiom set's declared components.

    Returns
    -------
    str
        64-character SHA-256 hex digest.
    """
    data = axiom_set.canonical_dict()
    serialized = _serialize_for_hash(data)
    return hashlib.sha256(serialized.encode()).hexdigest()


def hash_proof(script: ProofScript) -> str:
    """Deterministic SHA-256 hash binding a proof script to its axiom context.

    Hashes lemma content together with ``axiom_set_hash``.  If the axiom set
    changes, the proof hash changes even if lemma content is identical.

    Returns
    -------
    str
        64-character SHA-256 hex digest.
    """
    lemma_records: list[dict[str, Any]] = []
    for lemma in script.lemmas:
        lemma_records.append(
            {
                "name": lemma.name,
                "kind": lemma.kind.value,
                "expr": sympy.srepr(lemma.expr),
                "expected": (
                    sympy.srepr(lemma.expected) if lemma.expected is not None else None
                ),
                "assumptions": lemma.assumptions,
                "depends_on": sorted(lemma.depends_on),
            }
        )
    data: dict[str, Any] = {
        "axiom_set_hash": script.axiom_set_hash,
        "target": script.target,
        "lemmas": lemma_records,
    }
    serialized = _serialize_for_hash(data)
    return hashlib.sha256(serialized.encode()).hexdigest()


def hash_bundle(
    axiom_set_hash: str,
    hypothesis: Hypothesis,
    proof_hash: str,
) -> str:
    """Deterministic SHA-256 hash binding the (axioms, hypothesis, proof) triple.

    Returns
    -------
    str
        64-character SHA-256 hex digest.
    """
    hypothesis_canonical = make_canonical_dict(
        {
            "name": hypothesis.name,
            "expr": hypothesis.expr,
            "axiom_set_hash": hypothesis.axiom_set_hash,
        }
    )
    data: dict[str, Any] = {
        "axiom_set_hash": axiom_set_hash,
        "hypothesis": hypothesis_canonical,
        "proof_hash": proof_hash,
    }
    serialized = _serialize_for_hash(data)
    return hashlib.sha256(serialized.encode()).hexdigest()


def hash_disproof(
    hypothesis: Hypothesis,
    negation_bundle_hash: str,
) -> str:
    """Deterministic SHA-256 hash binding a disproof to its components.

    Returns
    -------
    str
        64-character SHA-256 hex digest.
    """
    hypothesis_canonical = make_canonical_dict(
        {
            "name": hypothesis.name,
            "expr": hypothesis.expr,
            "axiom_set_hash": hypothesis.axiom_set_hash,
        }
    )
    data: dict[str, Any] = {
        "hypothesis": hypothesis_canonical,
        "negation_bundle_hash": negation_bundle_hash,
    }
    serialized = _serialize_for_hash(data)
    return hashlib.sha256(serialized.encode()).hexdigest()
