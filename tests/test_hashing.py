"""Tests for symproof.hashing."""

from __future__ import annotations

import hashlib
import json

import sympy

from symproof import hash_axiom_set, hash_proof
from symproof.hashing import _serialize_for_hash


class TestHashAxiomSet:
    def test_returns_64_char_hex(self, positive_reals_axioms):
        h = hash_axiom_set(positive_reals_axioms)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic_same_set(self, positive_reals_axioms):
        h1 = hash_axiom_set(positive_reals_axioms)
        h2 = hash_axiom_set(positive_reals_axioms)
        assert h1 == h2

    def test_different_sets_different_hashes(
        self, positive_reals_axioms, single_axiom_set
    ):
        h1 = hash_axiom_set(positive_reals_axioms)
        h2 = hash_axiom_set(single_axiom_set)
        assert h1 != h2

    def test_is_sha256_of_canonical_dict(self, positive_reals_axioms):
        data = positive_reals_axioms.canonical_dict()
        serialized = json.dumps(data, sort_keys=True, default=str).encode()
        expected = hashlib.sha256(serialized).hexdigest()
        assert hash_axiom_set(positive_reals_axioms) == expected

    def test_property_matches_function(self, positive_reals_axioms):
        prop = positive_reals_axioms.axiom_set_hash
        func = hash_axiom_set(positive_reals_axioms)
        assert prop == func


class TestHashProof:
    def test_returns_64_char_hex(self, product_proof_script):
        h = hash_proof(product_proof_script)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self, product_proof_script):
        h1 = hash_proof(product_proof_script)
        h2 = hash_proof(product_proof_script)
        assert h1 == h2

    def test_uses_srepr_for_expressions(self, product_proof_script):
        script = product_proof_script
        lemma = script.lemmas[0]
        data = {
            "axiom_set_hash": script.axiom_set_hash,
            "target": script.target,
            "lemmas": [
                {
                    "name": lemma.name,
                    "kind": lemma.kind.value,
                    "expr": sympy.srepr(lemma.expr),
                    "expected": None,
                    "assumptions": lemma.assumptions,
                    "depends_on": sorted(lemma.depends_on),
                }
            ],
        }
        serialized = json.dumps(data, sort_keys=True, default=str).encode()
        expected = hashlib.sha256(serialized).hexdigest()
        assert hash_proof(script) == expected


class TestSerializeForHash:
    def test_sort_keys(self):
        d = {"b": 1, "a": 2}
        s1 = _serialize_for_hash(d)
        d2 = {"a": 2, "b": 1}
        s2 = _serialize_for_hash(d2)
        assert s1 == s2

    def test_non_json_values_stringified(self):
        d = {"expr": sympy.Symbol("x") ** 2}
        s = _serialize_for_hash(d)
        assert isinstance(s, str)
