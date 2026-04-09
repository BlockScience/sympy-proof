"""Tests for proof composition via bundle imports.

Covers:
1. Import a valid bundle, build new proof, seal
2. Import bundle with different axiom set → rejected
3. trust_imports=True skips re-verification
4. seal() always re-verifies regardless
5. Proof hash stability and sensitivity to imports
6. Multi-level composition
7. Evidence round-trip preserves imported bundles
8. Imported bundle with failing proof → caught on re-verify
9. Advisories propagated from imported bundles
"""

from __future__ import annotations

import pytest
import sympy

from symproof import (
    Axiom,
    AxiomSet,
    LemmaKind,
    ProofBuilder,
    ProofScript,
    seal,
    verify_proof,
)
from symproof.hashing import hash_proof
from symproof.models import Lemma, ProofBundle, ProofResult, ProofStatus

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

X = sympy.Symbol("x")
Y = sympy.Symbol("y")
K = sympy.Symbol("k", integer=True, nonnegative=True)


@pytest.fixture
def axioms():
    return AxiomSet(
        name="positive_reals",
        axioms=(
            Axiom(name="x_positive", expr=X > 0),
            Axiom(name="y_positive", expr=Y > 0),
        ),
    )


@pytest.fixture
def x_pos_bundle(axioms):
    """Sealed proof that x > 0."""
    h = axioms.hypothesis("x_positive_claim", expr=X > 0)
    script = (
        ProofBuilder(axioms, h.name, name="x_pos_proof", claim="x > 0")
        .lemma(
            "x_pos",
            LemmaKind.QUERY,
            expr=sympy.Q.positive(X),
            assumptions={"x": {"positive": True}},
        )
        .build()
    )
    return seal(axioms, h, script)


@pytest.fixture
def series_bundle(axioms):
    """Sealed proof that Sum(1/2^k, k=0..inf) = 2."""
    h = axioms.hypothesis(
        "series_limit",
        expr=sympy.Eq(
            sympy.Sum(sympy.Rational(1, 2) ** K, (K, 0, sympy.oo)),
            sympy.Integer(2),
        ),
    )
    script = (
        ProofBuilder(axioms, h.name, name="series_proof", claim="geo series = 2")
        .lemma(
            "series",
            LemmaKind.EQUALITY,
            expr=sympy.Sum(sympy.Rational(1, 2) ** K, (K, 0, sympy.oo)),
            expected=sympy.Integer(2),
        )
        .build()
    )
    return seal(axioms, h, script)


# ===================================================================
# 1. Basic import and seal
# ===================================================================


class TestBasicImport:
    """Import a valid bundle, build a new proof on top, seal it."""

    def test_import_and_seal(self, axioms, x_pos_bundle):
        """New proof importing x>0 bundle seals successfully."""
        h = axioms.hypothesis("xy_positive", expr=X * Y > 0)
        script = (
            ProofBuilder(axioms, h.name, name="xy_proof", claim="x*y > 0")
            .import_bundle(x_pos_bundle)
            .lemma(
                "product",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(X * Y),
                assumptions={"x": {"positive": True}, "y": {"positive": True}},
            )
            .build()
        )
        bundle = seal(axioms, h, script)
        assert len(bundle.bundle_hash) == 64
        assert len(script.imported_bundles) == 1

    def test_import_multiple_bundles(self, axioms, x_pos_bundle, series_bundle):
        """Multiple bundles can be imported into one proof."""
        h = axioms.hypothesis("combined", expr=X > 0)
        script = (
            ProofBuilder(axioms, h.name, name="combined", claim="combined")
            .import_bundle(x_pos_bundle)
            .import_bundle(series_bundle)
            .lemma(
                "trivial",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(X),
                assumptions={"x": {"positive": True}},
            )
            .build()
        )
        bundle = seal(axioms, h, script)
        assert len(script.imported_bundles) == 2

    def test_import_results_in_lemma_results(self, axioms, x_pos_bundle):
        """Imported bundles appear as import: entries in lemma_results."""
        h = axioms.hypothesis("test", expr=X > 0)
        script = (
            ProofBuilder(axioms, h.name, name="test", claim="test")
            .import_bundle(x_pos_bundle)
            .lemma(
                "local",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(X),
                assumptions={"x": {"positive": True}},
            )
            .build()
        )
        result = verify_proof(script)
        names = [lr.lemma_name for lr in result.lemma_results]
        assert names[0].startswith("import:")
        assert names[1] == "local"


# ===================================================================
# 2. Axiom set mismatch rejection
# ===================================================================


class TestAxiomMismatch:
    """Imported bundles must share the same axiom set."""

    def test_builder_rejects_mismatch(self, x_pos_bundle):
        """ProofBuilder.import_bundle rejects different axiom set."""
        other_axioms = AxiomSet(
            name="other",
            axioms=(Axiom(name="y_neg", expr=Y < 0),),
        )
        h = other_axioms.hypothesis("test", expr=Y < 0)
        builder = ProofBuilder(
            other_axioms, h.name, name="test", claim="test"
        )
        with pytest.raises(ValueError, match="different axiom set"):
            builder.import_bundle(x_pos_bundle)

    def test_seal_rejects_mismatch(self, axioms, x_pos_bundle):
        """seal() rejects if imported bundle has wrong axiom hash."""
        other_axioms = AxiomSet(
            name="other",
            axioms=(Axiom(name="z_pos", expr=sympy.Symbol("z") > 0),),
        )
        h = other_axioms.hypothesis("test", expr=sympy.Symbol("z") > 0)
        # Manually construct a script with mismatched import
        script = ProofScript(
            name="bad",
            target=h.name,
            axiom_set_hash=other_axioms.axiom_set_hash,
            claim="bad",
            lemmas=(
                Lemma(
                    name="z",
                    kind=LemmaKind.QUERY,
                    expr=sympy.Q.positive(sympy.Symbol("z")),
                    assumptions={"z": {"positive": True}},
                ),
            ),
            imported_bundles=(x_pos_bundle,),
        )
        with pytest.raises(ValueError, match="does not match"):
            seal(other_axioms, h, script)


# ===================================================================
# 3. trust_imports flag
# ===================================================================


class TestTrustImports:
    """trust_imports=True skips re-verification in verify_proof."""

    def test_trust_skips_reverification(self, axioms, x_pos_bundle):
        """With trust_imports=True, imported bundles are not re-verified."""
        h = axioms.hypothesis("test", expr=X > 0)
        script = (
            ProofBuilder(axioms, h.name, name="test", claim="test")
            .import_bundle(x_pos_bundle)
            .lemma(
                "local",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(X),
                assumptions={"x": {"positive": True}},
            )
            .build()
        )
        result = verify_proof(script, trust_imports=True)
        assert result.status == ProofStatus.VERIFIED

        # The import result should carry a TRUSTED advisory
        import_lr = result.lemma_results[0]
        assert import_lr.lemma_name.startswith("import:")
        assert any("TRUSTED" in a for a in import_lr.advisories)

    def test_default_reverifies(self, axioms, x_pos_bundle):
        """Default (trust_imports=False) re-verifies imports."""
        h = axioms.hypothesis("test", expr=X > 0)
        script = (
            ProofBuilder(axioms, h.name, name="test", claim="test")
            .import_bundle(x_pos_bundle)
            .lemma(
                "local",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(X),
                assumptions={"x": {"positive": True}},
            )
            .build()
        )
        result = verify_proof(script, trust_imports=False)
        assert result.status == ProofStatus.VERIFIED

        import_lr = result.lemma_results[0]
        assert not any("TRUSTED" in a for a in import_lr.advisories)


# ===================================================================
# 4. seal() always re-verifies
# ===================================================================


class TestSealAlwaysReverifies:
    """seal() must always re-verify imported bundles."""

    def test_seal_reverifies_imports(self, axioms, x_pos_bundle):
        """seal() successfully re-verifies a valid imported bundle."""
        h = axioms.hypothesis("test", expr=X > 0)
        script = (
            ProofBuilder(axioms, h.name, name="test", claim="test")
            .import_bundle(x_pos_bundle)
            .lemma(
                "local",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(X),
                assumptions={"x": {"positive": True}},
            )
            .build()
        )
        bundle = seal(axioms, h, script)
        # No TRUSTED advisory — seal always re-verifies
        import_lr = bundle.proof_result.lemma_results[0]
        assert not any("TRUSTED" in a for a in import_lr.advisories)


# ===================================================================
# 5. Hash stability and sensitivity
# ===================================================================


class TestHashBehavior:
    """Proof hash must change when imports change."""

    def test_hash_changes_with_import(self, axioms, x_pos_bundle):
        """Adding an import changes the proof hash."""
        h = axioms.hypothesis("test", expr=X > 0)

        script_no_import = (
            ProofBuilder(axioms, h.name, name="test", claim="test")
            .lemma(
                "local",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(X),
                assumptions={"x": {"positive": True}},
            )
            .build()
        )
        script_with_import = (
            ProofBuilder(axioms, h.name, name="test", claim="test")
            .import_bundle(x_pos_bundle)
            .lemma(
                "local",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(X),
                assumptions={"x": {"positive": True}},
            )
            .build()
        )

        h1 = hash_proof(script_no_import)
        h2 = hash_proof(script_with_import)
        assert h1 != h2

    def test_hash_stable_same_imports(self, axioms, x_pos_bundle):
        """Same imports produce same proof hash."""
        h = axioms.hypothesis("test", expr=X > 0)
        script1 = (
            ProofBuilder(axioms, h.name, name="test", claim="test")
            .import_bundle(x_pos_bundle)
            .lemma(
                "local",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(X),
                assumptions={"x": {"positive": True}},
            )
            .build()
        )
        script2 = (
            ProofBuilder(axioms, h.name, name="test", claim="test")
            .import_bundle(x_pos_bundle)
            .lemma(
                "local",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(X),
                assumptions={"x": {"positive": True}},
            )
            .build()
        )
        assert hash_proof(script1) == hash_proof(script2)

    def test_hash_differs_different_imports(
        self, axioms, x_pos_bundle, series_bundle
    ):
        """Different imported bundles produce different hashes."""
        h = axioms.hypothesis("test", expr=X > 0)
        s1 = (
            ProofBuilder(axioms, h.name, name="test", claim="test")
            .import_bundle(x_pos_bundle)
            .lemma(
                "l",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(X),
                assumptions={"x": {"positive": True}},
            )
            .build()
        )
        s2 = (
            ProofBuilder(axioms, h.name, name="test", claim="test")
            .import_bundle(series_bundle)
            .lemma(
                "l",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(X),
                assumptions={"x": {"positive": True}},
            )
            .build()
        )
        assert hash_proof(s1) != hash_proof(s2)


# ===================================================================
# 6. Multi-level composition
# ===================================================================


class TestMultiLevelComposition:
    """Import a bundle that itself imports bundles."""

    def test_two_level_composition(self, axioms, x_pos_bundle):
        """Bundle B2 imports B1, Bundle B3 imports B2."""
        # B2: imports x_pos_bundle, proves y > 0
        h2 = axioms.hypothesis("y_positive_claim", expr=Y > 0)
        s2 = (
            ProofBuilder(axioms, h2.name, name="y_proof", claim="y > 0")
            .import_bundle(x_pos_bundle)
            .lemma(
                "y_pos",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(Y),
                assumptions={"y": {"positive": True}},
            )
            .build()
        )
        b2 = seal(axioms, h2, s2)
        assert len(b2.proof.imported_bundles) == 1

        # B3: imports B2 (which internally imports B1)
        h3 = axioms.hypothesis("product_positive", expr=X * Y > 0)
        s3 = (
            ProofBuilder(axioms, h3.name, name="prod_proof", claim="x*y > 0")
            .import_bundle(b2)
            .lemma(
                "product",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(X * Y),
                assumptions={
                    "x": {"positive": True},
                    "y": {"positive": True},
                },
            )
            .build()
        )
        b3 = seal(axioms, h3, s3)
        assert len(b3.bundle_hash) == 64


# ===================================================================
# 7. Evidence round-trip
# ===================================================================


class TestEvidenceRoundTrip:
    """to_evidence/from_evidence preserves imported bundles."""

    def test_proof_script_round_trip(self, axioms, x_pos_bundle):
        """ProofScript with imports survives evidence round-trip."""
        h = axioms.hypothesis("test", expr=X > 0)
        script = (
            ProofBuilder(axioms, h.name, name="test", claim="test")
            .import_bundle(x_pos_bundle)
            .lemma(
                "local",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(X),
                assumptions={"x": {"positive": True}},
            )
            .build()
        )

        evidence = script.to_evidence()
        restored = ProofScript.from_evidence(evidence)

        assert restored.name == script.name
        assert restored.axiom_set_hash == script.axiom_set_hash
        assert len(restored.imported_bundles) == 1
        assert (
            restored.imported_bundles[0].bundle_hash
            == x_pos_bundle.bundle_hash
        )

        # Restored script should still verify
        result = verify_proof(restored)
        assert result.status == ProofStatus.VERIFIED

    def test_bundle_round_trip(self, axioms, x_pos_bundle):
        """ProofBundle.to_evidence/from_evidence round-trips."""
        evidence = x_pos_bundle.to_evidence()
        restored = ProofBundle.from_evidence(evidence)
        assert restored.bundle_hash == x_pos_bundle.bundle_hash
        assert restored.hypothesis.name == x_pos_bundle.hypothesis.name


# ===================================================================
# 8. Imported bundle with failing proof
# ===================================================================


class TestFailingImport:
    """Re-verification must catch imports whose proofs fail."""

    def test_tampered_import_caught(self, axioms):
        """A manually constructed 'bundle' with a bad proof fails."""
        # Create a script with a false lemma
        bad_script = ProofScript(
            name="bad",
            target="bad_target",
            axiom_set_hash=axioms.axiom_set_hash,
            claim="false claim",
            lemmas=(
                Lemma(
                    name="false_eq",
                    kind=LemmaKind.EQUALITY,
                    expr=sympy.Integer(1),
                    expected=sympy.Integer(2),
                ),
            ),
        )
        # Force-construct a ProofBundle (bypass seal)
        bad_bundle = ProofBundle(
            axiom_set=axioms,
            hypothesis=axioms.hypothesis("bad_target", expr=X > 0),
            proof=bad_script,
            proof_result=ProofResult(
                status=ProofStatus.VERIFIED,
                proof_hash="fake",
            ),
            bundle_hash="fake_hash",
        )

        h = axioms.hypothesis("test", expr=X > 0)
        script = (
            ProofBuilder(axioms, h.name, name="test", claim="test")
            .import_bundle(bad_bundle)
            .lemma(
                "local",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(X),
                assumptions={"x": {"positive": True}},
            )
            .build()
        )

        # verify_proof with default trust_imports=False should fail
        result = verify_proof(script)
        assert result.status == ProofStatus.FAILED
        assert "re-verification" in result.failure_summary.lower()

        # seal should also fail
        with pytest.raises(ValueError):
            seal(axioms, h, script)


# ===================================================================
# 9. Advisories from imported bundles
# ===================================================================


class TestImportAdvisories:
    """Advisories from imported bundle verification propagate."""

    def test_import_advisories_in_proof_result(
        self, axioms, x_pos_bundle
    ):
        """Advisories from import re-verification appear in result."""
        h = axioms.hypothesis("test", expr=X > 0)
        script = (
            ProofBuilder(axioms, h.name, name="test", claim="test")
            .import_bundle(x_pos_bundle)
            .lemma(
                "local",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(X),
                assumptions={"x": {"positive": True}},
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status == ProofStatus.VERIFIED
        # Should have advisories from both import and local lemma
        assert len(result.advisories) > 0


# ===================================================================
# 10. No imports = backward compatible
# ===================================================================


class TestBackwardCompatibility:
    """Proofs without imports work exactly as before."""

    def test_empty_imports_default(self, axioms):
        """ProofScript with no imports has empty imported_bundles."""
        h = axioms.hypothesis("test", expr=X > 0)
        script = (
            ProofBuilder(axioms, h.name, name="test", claim="test")
            .lemma(
                "pos",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(X),
                assumptions={"x": {"positive": True}},
            )
            .build()
        )
        assert script.imported_bundles == ()

    def test_hash_unchanged_no_imports(self, axioms):
        """Proof hash for scripts without imports is unchanged."""
        h = axioms.hypothesis("test", expr=X > 0)
        script = (
            ProofBuilder(axioms, h.name, name="test", claim="test")
            .lemma(
                "pos",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(X),
                assumptions={"x": {"positive": True}},
            )
            .build()
        )
        # Imported_bundle_hashes should be empty list in hash input
        proof_hash = hash_proof(script)
        assert len(proof_hash) == 64
