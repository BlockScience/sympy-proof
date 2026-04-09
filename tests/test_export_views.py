"""Tests for LaTeX and graph export views across all example bundles.

For each proof domain, builds representative bundles and verifies:
1. LaTeX rendering produces non-empty string with expected markers
2. Graph DAG extraction produces correct node/edge counts
3. DOT output is valid (has digraph header/footer)
4. Mermaid output is valid (has graph BT header)
5. JSON output parses and follows node-link schema
6. Imported bundles appear in all views (recursion works)
"""

from __future__ import annotations

import json

import pytest
import sympy

from symproof import Axiom, AxiomSet, LemmaKind, ProofBuilder, seal
from symproof.export.graph import (
    proof_dag,
    proof_dag_dot,
    proof_dag_json,
    proof_dag_mermaid,
)
from symproof.export.latex import latex_bundle, latex_document, latex_proof


# ---------------------------------------------------------------------------
# Fixtures: one bundle per domain
# ---------------------------------------------------------------------------

X = sympy.Symbol("x", real=True)
K = sympy.Symbol("k", integer=True, nonnegative=True)


@pytest.fixture
def euler_bundle():
    """Simplest proof: Euler's identity."""
    axioms = AxiomSet(
        name="complex",
        axioms=(Axiom(name="i_sq", expr=sympy.Eq(sympy.I**2, -1)),),
    )
    h = axioms.hypothesis(
        "euler", expr=sympy.Eq(sympy.exp(sympy.I * sympy.pi) + 1, 0),
    )
    script = (
        ProofBuilder(axioms, h.name, name="euler_proof", claim="e^(ipi)+1=0")
        .lemma(
            "eval", LemmaKind.EQUALITY,
            expr=sympy.exp(sympy.I * sympy.pi) + 1,
            expected=sympy.Integer(0),
        )
        .build()
    )
    return seal(axioms, h, script)


@pytest.fixture
def amgm_bundle():
    """Multi-lemma: AM-GM inequality."""
    a = sympy.Symbol("a", positive=True)
    b = sympy.Symbol("b", positive=True)
    axioms = AxiomSet(
        name="pos_reals",
        axioms=(
            Axiom(name="a_pos", expr=a > 0),
            Axiom(name="b_pos", expr=b > 0),
        ),
    )
    h = axioms.hypothesis(
        "am_gm", expr=sympy.Ge((a + b) / 2, sympy.sqrt(a * b)),
    )
    script = (
        ProofBuilder(axioms, h.name, name="amgm", claim="AM >= GM")
        .lemma(
            "sq_nonneg", LemmaKind.QUERY,
            expr=sympy.Q.nonnegative((a - b)**2),
            assumptions={"a": {"positive": True}, "b": {"positive": True}},
        )
        .lemma(
            "expand", LemmaKind.EQUALITY,
            expr=(a - b)**2, expected=a**2 - 2 * a * b + b**2,
            depends_on=["sq_nonneg"],
        )
        .lemma(
            "rearrange", LemmaKind.EQUALITY,
            expr=(a + b)**2 - 4 * a * b, expected=(a - b)**2,
            depends_on=["expand"],
        )
        .build()
    )
    return seal(axioms, h, script)


@pytest.fixture
def composed_bundle():
    """Composed proof: imports a sub-proof."""
    from symproof.library.control import (
        controllability_rank,
        lyapunov_from_system,
    )

    k = sympy.Symbol("k", positive=True)
    c = sympy.Symbol("c", positive=True)
    A = sympy.Matrix([[0, 1], [-k, -c]])
    B = sympy.Matrix([[0], [1]])

    axioms = AxiomSet(
        name="oscillator",
        axioms=(
            Axiom(name="k_pos", expr=k > 0),
            Axiom(name="c_pos", expr=c > 0),
        ),
    )
    ctrl = controllability_rank(axioms, A, B)
    lyap = lyapunov_from_system(axioms, A)

    h = axioms.hypothesis("verified", expr=sympy.And(k > 0, c > 0))
    script = (
        ProofBuilder(axioms, h.name, name="full", claim="ctrl+lyap")
        .import_bundle(ctrl)
        .import_bundle(lyap)
        .lemma(
            "params", LemmaKind.QUERY,
            expr=sympy.Q.positive(k * c),
            assumptions={"k": {"positive": True}, "c": {"positive": True}},
        )
        .build()
    )
    return seal(axioms, h, script)


@pytest.fixture
def convex_bundle():
    """Convex optimization: strong convexity + unique minimizer."""
    from symproof.library.convex import unique_minimizer

    axioms = AxiomSet(
        name="convex",
        axioms=(Axiom(name="t", expr=sympy.Eq(1, 1)),),
    )
    return unique_minimizer(
        axioms, X**2, [X], sympy.Integer(2),
    )


# ---------------------------------------------------------------------------
# LaTeX tests
# ---------------------------------------------------------------------------


class TestLatexBundle:
    """latex_bundle produces well-formed LaTeX for all bundle types."""

    def test_simple(self, euler_bundle):
        tex = latex_bundle(euler_bundle)
        assert r"\section*{Proof:" in tex
        assert "euler" in tex.lower()
        assert euler_bundle.bundle_hash[:16] in tex

    def test_multi_lemma(self, amgm_bundle):
        tex = latex_bundle(amgm_bundle)
        assert "sq_nonneg" in tex or "sq\\_nonneg" in tex
        assert "expand" in tex
        assert "rearrange" in tex
        assert r"\begin{description}" in tex

    def test_composed_shows_imports(self, composed_bundle):
        tex = latex_bundle(composed_bundle)
        assert "Imported" in tex
        assert "controllable" in tex.lower() or "ctrl" in tex.lower()

    def test_convex_with_imports(self, convex_bundle):
        tex = latex_bundle(convex_bundle)
        assert "strongly" in tex.lower() or "strong" in tex.lower()

    def test_advisories_section(self, amgm_bundle):
        tex = latex_bundle(amgm_bundle)
        # AM-GM has QUERY lemma → advisory
        if amgm_bundle.proof_result.advisories:
            assert "Advisor" in tex


class TestLatexDocument:
    """latex_document wraps in a compilable preamble."""

    def test_has_documentclass(self, euler_bundle):
        tex = latex_document(euler_bundle)
        assert r"\documentclass" in tex
        assert r"\begin{document}" in tex
        assert r"\end{document}" in tex

    def test_compiles_structure(self, composed_bundle):
        tex = latex_document(composed_bundle)
        assert r"\usepackage{amsmath" in tex


class TestLatexProof:
    """latex_proof renders just the proof chain."""

    def test_lemma_chain(self, amgm_bundle):
        tex = latex_proof(
            amgm_bundle.proof, result=amgm_bundle.proof_result,
        )
        assert r"\begin{description}" in tex
        assert "Lemma 1" in tex or "Lemma 2" in tex


# ---------------------------------------------------------------------------
# Graph DAG tests
# ---------------------------------------------------------------------------


class TestProofDag:
    """proof_dag extracts correct structure."""

    def test_simple_counts(self, euler_bundle):
        dag = proof_dag(euler_bundle)
        bundles = [n for n in dag["nodes"] if n["type"] == "bundle"]
        lemmas = [n for n in dag["nodes"] if n["type"] == "lemma"]
        assert len(bundles) == 1
        assert len(lemmas) == 1

    def test_multi_lemma_edges(self, amgm_bundle):
        dag = proof_dag(amgm_bundle)
        deps = [e for e in dag["edges"] if e["type"] == "depends_on"]
        assert len(deps) == 2  # expand→sq_nonneg, rearrange→expand

    def test_composed_has_imports(self, composed_bundle):
        dag = proof_dag(composed_bundle)
        imports = [e for e in dag["edges"] if e["type"] == "imports"]
        assert len(imports) >= 2  # ctrl + lyap

    def test_compact_mode(self, composed_bundle):
        full = proof_dag(composed_bundle, expand_lemmas=True)
        compact = proof_dag(composed_bundle, expand_lemmas=False)
        assert len(compact["nodes"]) < len(full["nodes"])

    def test_deduplication(self):
        """Shared import appears once in the DAG."""
        from symproof.library.defi import (
            amm_output_positive,
            amm_product_nondecreasing,
        )

        Rx = sympy.Symbol("R_x", positive=True)
        Ry = sympy.Symbol("R_y", positive=True)
        f = sympy.Symbol("f")
        dx = sympy.Symbol("dx", positive=True)
        axioms = AxiomSet(
            name="amm",
            axioms=(
                Axiom(name="rx", expr=Rx > 0),
                Axiom(name="ry", expr=Ry > 0),
                Axiom(name="f_pos", expr=f > 0),
                Axiom(name="f_lt1", expr=f < 1),
                Axiom(name="dx", expr=dx > 0),
            ),
        )
        b1 = amm_output_positive(axioms, Rx, Ry, f, dx)
        b2 = amm_product_nondecreasing(axioms, Rx, Ry, f, dx)

        dag = proof_dag(b1, b2, expand_lemmas=False)
        bundle_ids = [n["id"] for n in dag["nodes"]]
        # fee_complement appears once despite both importing it
        assert len(bundle_ids) == len(set(bundle_ids))

    def test_convex_nested_import(self, convex_bundle):
        dag = proof_dag(convex_bundle)
        imports = [e for e in dag["edges"] if e["type"] == "imports"]
        assert len(imports) >= 1  # unique_minimizer imports strongly_convex


class TestProofDagDot:
    """DOT output is well-formed."""

    def test_has_digraph(self, euler_bundle):
        dot = proof_dag_dot(euler_bundle)
        assert dot.startswith("digraph")
        assert dot.strip().endswith("}")

    def test_has_legend(self, euler_bundle):
        dot = proof_dag_dot(euler_bundle)
        assert "cluster_legend" in dot
        assert "Legend" in dot

    def test_composed_has_import_edges(self, composed_bundle):
        dot = proof_dag_dot(composed_bundle)
        assert 'label="imports"' in dot

    def test_advisory_marker(self, amgm_bundle):
        dot = proof_dag_dot(amgm_bundle)
        # QUERY lemma gets advisory → ⚠ in label
        if any(
            lr.advisories
            for lr in amgm_bundle.proof_result.lemma_results
        ):
            assert "⚠" in dot


class TestProofDagMermaid:
    """Mermaid output is well-formed."""

    def test_has_header(self, euler_bundle):
        mmd = proof_dag_mermaid(euler_bundle)
        assert mmd.startswith("graph BT")

    def test_composed(self, composed_bundle):
        mmd = proof_dag_mermaid(composed_bundle)
        assert "imports" in mmd


class TestProofDagJson:
    """JSON output follows node-link schema."""

    def test_parses(self, euler_bundle):
        s = proof_dag_json(euler_bundle)
        data = json.loads(s)
        assert data["directed"] is True
        assert data["multigraph"] is False
        assert "nodes" in data
        assert "links" in data

    def test_schema_version(self, euler_bundle):
        data = json.loads(proof_dag_json(euler_bundle))
        assert data["graph"]["schema_version"] == 1

    def test_tool_hints(self, euler_bundle):
        data = json.loads(proof_dag_json(euler_bundle))
        hints = data["graph"]["tool_hints"]
        assert "d3" in hints
        assert "networkx" in hints
        assert "neo4j" in hints

    def test_all_nodes_have_id(self, composed_bundle):
        data = json.loads(proof_dag_json(composed_bundle))
        for node in data["nodes"]:
            assert "id" in node
            assert "type" in node

    def test_all_links_have_source_target(self, composed_bundle):
        data = json.loads(proof_dag_json(composed_bundle))
        for link in data["links"]:
            assert "source" in link
            assert "target" in link
            assert "type" in link

    def test_compact_mode(self, composed_bundle):
        full = json.loads(
            proof_dag_json(composed_bundle, expand_lemmas=True),
        )
        compact = json.loads(
            proof_dag_json(composed_bundle, expand_lemmas=False),
        )
        assert len(compact["nodes"]) < len(full["nodes"])


# ---------------------------------------------------------------------------
# Cross-domain: all views on same bundle
# ---------------------------------------------------------------------------


class TestAllViewsConsistent:
    """All export formats agree on structure for the same bundle."""

    def test_node_count_matches(self, composed_bundle):
        dag = proof_dag(composed_bundle, expand_lemmas=True)
        data = json.loads(
            proof_dag_json(composed_bundle, expand_lemmas=True),
        )
        assert len(dag["nodes"]) == len(data["nodes"])
        assert len(dag["edges"]) == len(data["links"])

    def test_latex_and_graph_both_show_hash(self, euler_bundle):
        tex = latex_bundle(euler_bundle)
        dag = proof_dag(euler_bundle)
        bundle_node = [
            n for n in dag["nodes"] if n["type"] == "bundle"
        ][0]
        # Both should reference the same hash
        assert euler_bundle.bundle_hash[:16] in tex
        assert bundle_node["hash"] == euler_bundle.bundle_hash
