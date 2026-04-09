"""Tests for LaTeX export of proof bundles.

Renders actual proof bundles and verifies the LaTeX output contains
expected structural fragments.  No PDF compilation — just string checks.
"""

from __future__ import annotations

import sympy

from symproof import (
    Axiom,
    AxiomSet,
    LemmaKind,
    ProofBuilder,
    seal,
)
from symproof.export import latex_bundle, latex_document, latex_lemma, latex_proof
from symproof.models import Lemma
from symproof.verification import verify_lemma

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _simple_bundle():
    """A minimal sealed bundle for testing."""
    x = sympy.Symbol("x")
    axioms = AxiomSet(
        name="test_axioms",
        axioms=(Axiom(name="x_positive", expr=x > 0),),
    )
    h = axioms.hypothesis("product_claim", expr=x > 0)
    script = (
        ProofBuilder(axioms, h.name, name="simple_proof", claim="x is positive")
        .lemma(
            "x_pos",
            LemmaKind.QUERY,
            expr=sympy.Q.positive(x),
            assumptions={"x": {"positive": True}},
            description="x is positive by assumption",
        )
        .build()
    )
    return seal(axioms, h, script)


def _multi_lemma_bundle():
    """A bundle with multiple lemmas and dependencies."""
    x = sympy.Symbol("x", positive=True)
    axioms = AxiomSet(
        name="algebra",
        axioms=(
            Axiom(name="x_pos", expr=x > 0),
        ),
    )
    h = axioms.hypothesis("expand_check", expr=sympy.Eq((x + 1) ** 2, x**2 + 2*x + 1))
    script = (
        ProofBuilder(axioms, h.name, name="expand_proof", claim="(x+1)^2 expansion")
        .lemma(
            "expand",
            LemmaKind.EQUALITY,
            expr=(x + 1) ** 2,
            expected=x**2 + 2*x + 1,
            description="Expand the square",
        )
        .lemma(
            "positive_check",
            LemmaKind.QUERY,
            expr=sympy.Q.positive(x),
            assumptions={"x": {"positive": True}},
            depends_on=["expand"],
            description="x is still positive",
        )
        .build()
    )
    return seal(axioms, h, script)


def _imported_bundle():
    """A bundle that imports another bundle."""
    x = sympy.Symbol("x")
    axioms = AxiomSet(
        name="composed",
        axioms=(Axiom(name="x_pos", expr=x > 0),),
    )
    # First: a sub-proof
    h_sub = axioms.hypothesis("sub_claim", expr=x > 0)
    sub_script = (
        ProofBuilder(axioms, h_sub.name, name="sub_proof", claim="x > 0")
        .lemma("q", LemmaKind.QUERY, expr=sympy.Q.positive(x),
               assumptions={"x": {"positive": True}})
        .build()
    )
    sub_bundle = seal(axioms, h_sub, sub_script)

    # Second: a proof that imports the sub-proof
    h_main = axioms.hypothesis("main_claim", expr=x > 0)
    main_script = (
        ProofBuilder(axioms, h_main.name, name="main_proof", claim="composed claim")
        .import_bundle(sub_bundle)
        .lemma("anchor", LemmaKind.QUERY, expr=sympy.Q.positive(x),
               assumptions={"x": {"positive": True}})
        .build()
    )
    return seal(axioms, h_main, main_script)


# ---------------------------------------------------------------------------
# Tests: latex_lemma
# ---------------------------------------------------------------------------


class TestLatexLemma:
    def test_basic_rendering(self):
        x = sympy.Symbol("x")
        lemma = Lemma(
            name="test_lem",
            kind=LemmaKind.EQUALITY,
            expr=x**2 + 1,
            expected=x**2 + 1,
            description="trivial identity",
        )
        tex = latex_lemma(lemma)
        assert "test\\_lem" in tex
        assert "equality" in tex
        assert "trivial identity" in tex

    def test_with_result(self):
        x = sympy.Symbol("x")
        lemma = Lemma(
            name="pos",
            kind=LemmaKind.QUERY,
            expr=sympy.Q.positive(x),
            assumptions={"x": {"positive": True}},
        )
        result = verify_lemma(lemma)
        tex = latex_lemma(lemma, result=result)
        assert r"\textsc{passed}" in tex

    def test_with_index(self):
        lemma = Lemma(
            name="step",
            kind=LemmaKind.BOOLEAN,
            expr=sympy.Eq(1, 1),
        )
        tex = latex_lemma(lemma, index=3)
        assert "Lemma 3" in tex

    def test_assumptions_rendered(self):
        x = sympy.Symbol("x")
        lemma = Lemma(
            name="asm_test",
            kind=LemmaKind.QUERY,
            expr=sympy.Q.positive(x),
            assumptions={"x": {"positive": True, "integer": True}},
        )
        tex = latex_lemma(lemma)
        assert "> 0" in tex
        assert r"\mathbb{Z}" in tex

    def test_equality_shows_expected(self):
        x = sympy.Symbol("x")
        lemma = Lemma(
            name="eq_test",
            kind=LemmaKind.EQUALITY,
            expr=x**2,
            expected=x * x,
        )
        tex = latex_lemma(lemma)
        assert "=" in tex

    def test_coordinate_transform_shows_maps(self):
        x = sympy.Symbol("x")
        r = sympy.Symbol("r", positive=True)
        theta = sympy.Symbol("theta")
        lemma = Lemma(
            name="ct",
            kind=LemmaKind.COORDINATE_TRANSFORM,
            expr=x,
            expected=r * sympy.cos(theta),
            transform={"x": r * sympy.cos(theta)},
            inverse_transform={"r": sympy.sqrt(x**2)},
        )
        tex = latex_lemma(lemma)
        assert "Forward" in tex
        assert "Inverse" in tex
        assert r"\mapsto" in tex

    def test_depends_on_rendered(self):
        lemma = Lemma(
            name="dep",
            kind=LemmaKind.BOOLEAN,
            expr=sympy.Eq(1, 1),
            depends_on=["step_1", "step_2"],
        )
        tex = latex_lemma(lemma)
        assert "step\\_1" in tex
        assert "step\\_2" in tex


# ---------------------------------------------------------------------------
# Tests: latex_proof
# ---------------------------------------------------------------------------


class TestLatexProof:
    def test_basic_structure(self):
        bundle = _simple_bundle()
        tex = latex_proof(bundle.proof, result=bundle.proof_result)
        assert "Lemma Chain" in tex
        assert r"\begin{description}" in tex
        assert r"\end{description}" in tex

    def test_lemma_names_present(self):
        bundle = _multi_lemma_bundle()
        tex = latex_proof(bundle.proof, result=bundle.proof_result)
        assert "expand" in tex
        assert "positive\\_check" in tex

    def test_imported_bundles_section(self):
        bundle = _imported_bundle()
        tex = latex_proof(bundle.proof, result=bundle.proof_result)
        assert "Imported Proofs" in tex
        assert "sub\\_claim" in tex


# ---------------------------------------------------------------------------
# Tests: latex_bundle
# ---------------------------------------------------------------------------


class TestLatexBundle:
    def test_has_section_header(self):
        bundle = _simple_bundle()
        tex = latex_bundle(bundle)
        assert r"\section*{Proof:" in tex

    def test_has_claim(self):
        bundle = _simple_bundle()
        tex = latex_bundle(bundle)
        assert "x is positive" in tex

    def test_has_status(self):
        bundle = _simple_bundle()
        tex = latex_bundle(bundle)
        assert r"\textsc{verified}" in tex

    def test_has_hash(self):
        bundle = _simple_bundle()
        tex = latex_bundle(bundle)
        assert bundle.bundle_hash[:16] in tex
        assert bundle.bundle_hash in tex  # full hash in footer

    def test_has_axioms(self):
        bundle = _simple_bundle()
        tex = latex_bundle(bundle)
        assert "Axioms:" in tex
        assert "x\\_positive" in tex

    def test_has_hypothesis(self):
        bundle = _simple_bundle()
        tex = latex_bundle(bundle)
        assert "Hypothesis" in tex

    def test_has_proof_section(self):
        bundle = _simple_bundle()
        tex = latex_bundle(bundle)
        assert r"\subsection*{Proof}" in tex

    def test_no_advisories_section_when_empty(self):
        """Simple proof has no advisories — section should be absent."""
        bundle = _multi_lemma_bundle()
        tex = latex_bundle(bundle)
        # This particular proof might or might not have advisories;
        # check that the section only appears if there are advisories
        if not bundle.proof_result.advisories:
            assert "Advisories" not in tex

    def test_advisories_section_when_present(self):
        """A QUERY proof always has advisories — section should appear."""
        bundle = _simple_bundle()
        tex = latex_bundle(bundle)
        if bundle.proof_result.advisories:
            assert "Advisories" in tex

    def test_imported_bundles_rendered(self):
        bundle = _imported_bundle()
        tex = latex_bundle(bundle)
        assert "Imported Proofs" in tex

    def test_hash_footer(self):
        bundle = _simple_bundle()
        tex = latex_bundle(bundle)
        assert "bundle\\_hash:" in tex


# ---------------------------------------------------------------------------
# Tests: latex_document
# ---------------------------------------------------------------------------


class TestLatexDocument:
    def test_has_documentclass(self):
        bundle = _simple_bundle()
        tex = latex_document(bundle)
        assert r"\documentclass" in tex

    def test_has_amsmath(self):
        bundle = _simple_bundle()
        tex = latex_document(bundle)
        assert "amsmath" in tex

    def test_has_begin_document(self):
        bundle = _simple_bundle()
        tex = latex_document(bundle)
        assert r"\begin{document}" in tex
        assert r"\end{document}" in tex

    def test_has_title(self):
        bundle = _simple_bundle()
        tex = latex_document(bundle)
        assert r"\title{" in tex
        assert "simple\\_proof" in tex

    def test_contains_bundle_content(self):
        bundle = _simple_bundle()
        tex = latex_document(bundle)
        # Should contain all the bundle content too
        assert r"\section*{Proof:" in tex
        assert bundle.bundle_hash in tex
