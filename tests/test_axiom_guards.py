"""Tests for axiom guards: falsity blocking, citations, consistency, assumption reporting."""

from __future__ import annotations

import pytest
import sympy

from symproof import (
    Axiom,
    AxiomSet,
    Citation,
    LemmaKind,
    ProofBuilder,
    ProofStatus,
    seal,
)


# ---------------------------------------------------------------------------
# Falsity blocking (AxiomSet construction)
# ---------------------------------------------------------------------------


class TestFalsityBlocking:
    def test_provably_false_axiom_rejected(self):
        """And(x > 0, x < 0) simplifies to False → ValueError."""
        x = sympy.Symbol("x", real=True)
        with pytest.raises(ValueError, match="provably false"):
            AxiomSet(
                name="bad",
                axioms=(Axiom(name="contradiction", expr=sympy.And(x > 0, x < 0)),),
            )

    def test_false_literal_rejected(self):
        """expr=sympy.false is rejected."""
        with pytest.raises(ValueError, match="provably false"):
            AxiomSet(
                name="bad",
                axioms=(Axiom(name="false", expr=sympy.S.false),),
            )

    def test_true_axiom_skipped(self):
        """expr=True (external results) passes — cannot check."""
        ax = AxiomSet(
            name="ok",
            axioms=(Axiom(name="ext", expr=sympy.S.true),),
        )
        assert len(ax.axioms) == 1

    def test_indeterminate_axiom_allowed(self):
        """Cannot determine truth value — allowed through."""
        x = sympy.Symbol("x")
        ax = AxiomSet(
            name="ok",
            axioms=(Axiom(name="maybe", expr=sympy.Gt(x, 0, evaluate=False)),),
        )
        assert len(ax.axioms) == 1


# ---------------------------------------------------------------------------
# Pairwise consistency (seal time)
# ---------------------------------------------------------------------------


class TestPairwiseConsistency:
    def test_contradictory_pair_rejected_at_seal(self):
        """Two axioms that contradict → ValueError at seal()."""
        x = sympy.Symbol("x")
        ax = AxiomSet(
            name="bad",
            axioms=(
                Axiom(name="pos", expr=sympy.Gt(x, 5, evaluate=False)),
                Axiom(name="neg", expr=sympy.Lt(x, -5, evaluate=False)),
            ),
        )
        h = ax.hypothesis("h", expr=sympy.S.true)
        script = (
            ProofBuilder(ax, h.name, name="p", claim="c")
            .lemma("trivial", LemmaKind.BOOLEAN, expr=sympy.S.true)
            .build()
        )
        with pytest.raises(ValueError, match="contradictory"):
            seal(ax, h, script)

    def test_consistency_check_skippable(self):
        """check_consistency=False bypasses the pairwise check."""
        x = sympy.Symbol("x")
        ax = AxiomSet(
            name="bad",
            axioms=(
                Axiom(name="pos", expr=sympy.Gt(x, 5, evaluate=False)),
                Axiom(name="neg", expr=sympy.Lt(x, -5, evaluate=False)),
            ),
        )
        h = ax.hypothesis("h", expr=sympy.S.true)
        script = (
            ProofBuilder(ax, h.name, name="p", claim="c")
            .lemma("trivial", LemmaKind.BOOLEAN, expr=sympy.S.true)
            .build()
        )
        # Passes with consistency check disabled
        bundle = seal(ax, h, script, check_consistency=False)
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_compatible_axioms_pass(self):
        """Non-contradictory axioms pass the consistency check."""
        x = sympy.Symbol("x")
        ax = AxiomSet(
            name="ok",
            axioms=(
                Axiom(name="pos", expr=sympy.Gt(x, 0, evaluate=False)),
                Axiom(name="bounded", expr=sympy.Lt(x, 100, evaluate=False)),
            ),
        )
        h = ax.hypothesis("h", expr=sympy.S.true)
        script = (
            ProofBuilder(ax, h.name, name="p", claim="c")
            .lemma("trivial", LemmaKind.BOOLEAN, expr=sympy.S.true)
            .build()
        )
        bundle = seal(ax, h, script)
        assert bundle.proof_result.status == ProofStatus.VERIFIED


# ---------------------------------------------------------------------------
# Citation requirement
# ---------------------------------------------------------------------------


class TestCitationRequirement:
    def test_inherited_without_citation_rejected(self):
        with pytest.raises(Exception, match="citation"):
            Axiom(name="a", expr=sympy.S.true, inherited=True)

    def test_inherited_with_citation_accepted(self):
        a = Axiom(
            name="a",
            expr=sympy.S.true,
            inherited=True,
            citation=Citation(source="Flam 2004, Theorem 2"),
        )
        assert a.citation.source == "Flam 2004, Theorem 2"
        assert a.citation.bundle_hash == ""

    def test_non_inherited_without_citation_ok(self):
        """Regular axioms don't need citations."""
        x = sympy.Symbol("x")
        a = Axiom(name="a", expr=sympy.Gt(x, 0, evaluate=False))
        assert a.citation is None

    def test_citation_with_bundle_hash(self):
        """Citation can optionally include a bundle hash for future linking."""
        a = Axiom(
            name="a",
            expr=sympy.S.true,
            inherited=True,
            citation=Citation(
                source="Queue stability proof",
                bundle_hash="ab" * 32,
            ),
        )
        assert a.citation.bundle_hash == "ab" * 32


# ---------------------------------------------------------------------------
# Noisy assumption reporting
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Load-bearing assumption accounting
# ---------------------------------------------------------------------------


class TestLoadBearing:
    def test_equality_with_uncovered_assumption_rejected(self):
        """EQUALITY lemma using symbol with assumptions but no axiom → error."""
        x = sympy.Symbol("x", positive=True)
        ax = AxiomSet(name="s", axioms=(Axiom(name="d", expr=sympy.S.true),))
        h = ax.hypothesis("h", expr=sympy.S.true)
        script = (
            ProofBuilder(ax, h.name, name="p", claim="c")
            .lemma("uses_x", LemmaKind.EQUALITY, expr=x + 1, expected=x + 1)
            .build()
        )
        with pytest.raises(ValueError, match="load-bearing"):
            seal(ax, h, script)

    def test_covered_assumption_passes(self):
        """Symbol with assumption that HAS a matching axiom → passes."""
        x = sympy.Symbol("x", positive=True)
        ax = AxiomSet(
            name="s",
            axioms=(Axiom(name="x_pos", expr=x > 0),),
        )
        h = ax.hypothesis("h", expr=x > 0)
        script = (
            ProofBuilder(ax, h.name, name="p", claim="c")
            .lemma("uses_x", LemmaKind.EQUALITY, expr=x + 1, expected=x + 1)
            .build()
        )
        bundle = seal(ax, h, script)
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_bare_symbol_no_issue(self):
        """Symbol without assumptions → no load-bearing issue."""
        x = sympy.Symbol("x")
        ax = AxiomSet(name="s", axioms=(Axiom(name="d", expr=sympy.S.true),))
        h = ax.hypothesis("h", expr=sympy.S.true)
        script = (
            ProofBuilder(ax, h.name, name="p", claim="c")
            .lemma("uses_x", LemmaKind.EQUALITY, expr=x + 1, expected=x + 1)
            .build()
        )
        bundle = seal(ax, h, script)
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_query_with_lemma_assumptions_not_flagged(self):
        """QUERY lemma provides its own assumptions dict → symbol assumption is redundant."""
        x = sympy.Symbol("x", positive=True)
        ax = AxiomSet(
            name="s",
            axioms=(Axiom(name="x_pos", expr=x > 0),),
        )
        h = ax.hypothesis("h", expr=x > 0)
        script = (
            ProofBuilder(ax, h.name, name="p", claim="c")
            .lemma(
                "q",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(x),
                assumptions={"x": {"positive": True}},
            )
            .build()
        )
        bundle = seal(ax, h, script)
        assert bundle.proof_result.status == ProofStatus.VERIFIED


class TestNoisyAssumptions:
    def test_seal_reports_assumptions(self):
        """seal() result includes [ASSUMPTIONS] advisories."""
        x = sympy.Symbol("x", positive=True)
        ax = AxiomSet(
            name="test",
            axioms=(Axiom(name="x_pos", expr=x > 0),),
        )
        h = ax.hypothesis("h", expr=x > 0)
        script = (
            ProofBuilder(ax, h.name, name="p", claim="c")
            .lemma(
                "l",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(x),
                assumptions={"x": {"positive": True}},
            )
            .build()
        )
        bundle = seal(ax, h, script)
        assumption_advs = [
            a for a in bundle.proof_result.advisories if "[ASSUMPTIONS]" in a
        ]
        assert len(assumption_advs) > 0

    def test_inherited_axioms_called_out(self):
        """Inherited axioms get specific callout in advisories."""
        x = sympy.Symbol("x", positive=True)
        ax = AxiomSet(
            name="test",
            axioms=(
                Axiom(name="x_pos", expr=x > 0),
                Axiom(
                    name="ext",
                    expr=sympy.S.true,
                    inherited=True,
                    citation=Citation(source="Test source"),
                ),
            ),
        )
        h = ax.hypothesis("h", expr=x > 0)
        script = (
            ProofBuilder(ax, h.name, name="p", claim="c")
            .lemma(
                "l",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(x),
                assumptions={"x": {"positive": True}},
            )
            .build()
        )
        bundle = seal(ax, h, script)
        inherited_advs = [
            a for a in bundle.proof_result.advisories if "INHERITED" in a
        ]
        assert len(inherited_advs) > 0
        assert any("Test source" in a for a in inherited_advs)

    def test_external_axioms_flagged(self):
        """Non-inherited True axioms get a 'consider backing' advisory."""
        ax = AxiomSet(
            name="test",
            axioms=(Axiom(name="external_thm", expr=sympy.S.true),),
        )
        h = ax.hypothesis("h", expr=sympy.S.true)
        script = (
            ProofBuilder(ax, h.name, name="p", claim="c")
            .lemma("trivial", LemmaKind.BOOLEAN, expr=sympy.S.true)
            .build()
        )
        bundle = seal(ax, h, script)
        external_advs = [
            a for a in bundle.proof_result.advisories if "EXTERNAL" in a
        ]
        assert len(external_advs) > 0
        assert any("foundation" in a.lower() for a in external_advs)
