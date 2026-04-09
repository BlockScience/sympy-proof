"""Tests for hidden axiom detection and foundation enforcement in seal()."""

from __future__ import annotations

import pytest
import sympy

from symproof import (
    Axiom,
    AxiomSet,
    LemmaKind,
    ProofBuilder,
    ProofStatus,
    seal,
)


# ---------------------------------------------------------------------------
# Helpers — build minimal sealed bundles for testing
# ---------------------------------------------------------------------------

def _make_bundle(axiom_set, hypothesis_name="h", hypothesis_expr=sympy.S.true):
    """Build a trivially-verified sealed bundle under the given axiom set."""
    h = axiom_set.hypothesis(hypothesis_name, expr=hypothesis_expr)
    script = (
        ProofBuilder(axiom_set, h.name, name="p", claim="test")
        .lemma("trivial", LemmaKind.BOOLEAN, expr=sympy.S.true)
        .build()
    )
    return seal(axiom_set, h, script)


# ---------------------------------------------------------------------------
# Axiom.inherited field
# ---------------------------------------------------------------------------

class TestInherited:
    def test_default_not_inherited(self):
        a = Axiom(name="a", expr=sympy.S.true)
        assert a.inherited is False

    def test_explicit_inherited(self):
        a = Axiom(name="a", expr=sympy.S.true, inherited=True)
        assert a.inherited is True

    def test_inherited_in_canonical_dict(self):
        ax1 = AxiomSet(name="s", axioms=(
            Axiom(name="a", expr=sympy.S.true),
        ))
        ax2 = AxiomSet(name="s", axioms=(
            Axiom(name="a", expr=sympy.S.true, inherited=True),
        ))
        # Different inherited values → different hashes
        assert ax1.axiom_set_hash != ax2.axiom_set_hash

    def test_inherited_does_not_affect_verification(self):
        """Proofs verify the same regardless of inherited flag."""
        x = sympy.Symbol("x", positive=True)
        for inherited in (False, True):
            ax = AxiomSet(name="s", axioms=(
                Axiom(name="x_pos", expr=x > 0, inherited=inherited),
            ))
            h = ax.hypothesis("h", expr=x > 0)
            script = (
                ProofBuilder(ax, h.name, name="p", claim="c")
                .lemma("l", LemmaKind.QUERY, expr=sympy.Q.positive(x),
                       assumptions={"x": {"positive": True}})
                .build()
            )
            bundle = seal(ax, h, script)
            assert bundle.proof_result.status == ProofStatus.VERIFIED


# ---------------------------------------------------------------------------
# AxiomSet helpers
# ---------------------------------------------------------------------------

class TestAxiomSetHelpers:
    def test_true_axioms(self):
        ax = AxiomSet(name="s", axioms=(
            Axiom(name="real", expr=sympy.Symbol("x") > 0),
            Axiom(name="external", expr=sympy.S.true),
        ))
        true_ax = ax.true_axioms()
        assert len(true_ax) == 1
        assert true_ax[0].name == "external"

    def test_get_axiom_found(self):
        ax = AxiomSet(name="s", axioms=(
            Axiom(name="a1", expr=sympy.S.true),
        ))
        assert ax.get_axiom("a1") is not None
        assert ax.get_axiom("a1").name == "a1"

    def test_get_axiom_not_found(self):
        ax = AxiomSet(name="s", axioms=(
            Axiom(name="a1", expr=sympy.S.true),
        ))
        assert ax.get_axiom("missing") is None


# ---------------------------------------------------------------------------
# Foundation enforcement in seal()
# ---------------------------------------------------------------------------

class TestFoundationEnforcement:
    def test_seal_without_foundations_works(self):
        """No foundations → no check (backwards compatible)."""
        x = sympy.Symbol("x", positive=True)
        ax = AxiomSet(name="s", axioms=(
            Axiom(name="x_pos", expr=x > 0),
            Axiom(name="external", expr=sympy.S.true),
        ))
        bundle = _make_bundle(ax)
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_foundation_detects_hidden_axioms(self):
        """Foundation has an axiom not in downstream → ValueError."""
        gamma = sympy.Symbol("gamma")
        v = sympy.Symbol("v")

        # Downstream: only has gamma > 0 and an external theorem
        downstream_ax = AxiomSet(name="downstream", axioms=(
            Axiom(name="gradient_bounded", expr=sympy.Gt(gamma, 0, evaluate=False)),
            Axiom(name="flam_theorem", expr=sympy.S.true),
        ))

        # Foundation: has gamma > 0 AND v >= 0 (hidden!)
        foundation_ax = AxiomSet(name="foundation", axioms=(
            Axiom(name="gradient_bounded", expr=sympy.Gt(gamma, 0, evaluate=False)),
            Axiom(name="lyapunov_nonneg", expr=sympy.Ge(v, 0, evaluate=False)),
        ))
        foundation_bundle = _make_bundle(foundation_ax)

        h = downstream_ax.hypothesis("h", expr=sympy.S.true)
        script = (
            ProofBuilder(downstream_ax, h.name, name="p", claim="c")
            .lemma("trivial", LemmaKind.BOOLEAN, expr=sympy.S.true)
            .build()
        )
        with pytest.raises(ValueError, match="hidden axioms"):
            seal(downstream_ax, h, script,
                 foundations=[(foundation_bundle, "flam_theorem")])

    def test_foundation_passes_when_all_covered(self):
        """All foundation axioms present in downstream → passes."""
        gamma = sympy.Symbol("gamma", positive=True)
        v = sympy.Symbol("v", nonnegative=True)

        downstream_ax = AxiomSet(name="downstream", axioms=(
            Axiom(name="gradient_bounded", expr=gamma > 0),
            Axiom(name="flam_theorem", expr=sympy.S.true),
            Axiom(name="lyapunov_nonneg", expr=v >= 0, inherited=True),
        ))

        foundation_ax = AxiomSet(name="foundation", axioms=(
            Axiom(name="gradient_bounded", expr=gamma > 0),
            Axiom(name="lyapunov_nonneg", expr=v >= 0),
        ))
        foundation_bundle = _make_bundle(foundation_ax)

        h = downstream_ax.hypothesis("h", expr=sympy.S.true)
        script = (
            ProofBuilder(downstream_ax, h.name, name="p", claim="c")
            .lemma("trivial", LemmaKind.BOOLEAN, expr=sympy.S.true)
            .build()
        )
        bundle = seal(downstream_ax, h, script,
                      foundations=[(foundation_bundle, "flam_theorem")])
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_foundation_matches_by_expression(self):
        """Different name but same expression → matched."""
        gamma = sympy.Symbol("gamma", positive=True)

        downstream_ax = AxiomSet(name="downstream", axioms=(
            Axiom(name="grad_bound", expr=gamma > 0),
            Axiom(name="theorem", expr=sympy.S.true),
        ))

        foundation_ax = AxiomSet(name="foundation", axioms=(
            Axiom(name="gradient_bounded", expr=gamma > 0),  # different name!
        ))
        foundation_bundle = _make_bundle(foundation_ax)

        h = downstream_ax.hypothesis("h", expr=sympy.S.true)
        script = (
            ProofBuilder(downstream_ax, h.name, name="p", claim="c")
            .lemma("trivial", LemmaKind.BOOLEAN, expr=sympy.S.true)
            .build()
        )
        bundle = seal(downstream_ax, h, script,
                      foundations=[(foundation_bundle, "theorem")])
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_foundation_justified_axiom_not_found(self):
        """justified_axiom_name not in axiom set → ValueError."""
        ax = AxiomSet(name="s", axioms=(
            Axiom(name="a", expr=sympy.S.true),
        ))
        foundation_bundle = _make_bundle(ax)

        h = ax.hypothesis("h", expr=sympy.S.true)
        script = (
            ProofBuilder(ax, h.name, name="p", claim="c")
            .lemma("trivial", LemmaKind.BOOLEAN, expr=sympy.S.true)
            .build()
        )
        with pytest.raises(ValueError, match="does not exist"):
            seal(ax, h, script,
                 foundations=[(foundation_bundle, "nonexistent")])

    def test_multiple_foundations(self):
        """Multiple foundations, each covering different axioms."""
        g = sympy.Symbol("g", positive=True)
        v = sympy.Symbol("v", nonnegative=True)

        downstream_ax = AxiomSet(name="downstream", axioms=(
            Axiom(name="bound", expr=g > 0),
            Axiom(name="theorem_a", expr=sympy.S.true),
            Axiom(name="nonneg", expr=v >= 0, inherited=True),
            Axiom(name="theorem_b", expr=sympy.S.true),
        ))

        found_a_ax = AxiomSet(name="fa", axioms=(
            Axiom(name="bound", expr=g > 0),
            Axiom(name="nonneg", expr=v >= 0),
        ))
        found_b_ax = AxiomSet(name="fb", axioms=(
            Axiom(name="bound", expr=g > 0),
        ))

        fa_bundle = _make_bundle(found_a_ax)
        fb_bundle = _make_bundle(found_b_ax)

        h = downstream_ax.hypothesis("h", expr=sympy.S.true)
        script = (
            ProofBuilder(downstream_ax, h.name, name="p", claim="c")
            .lemma("trivial", LemmaKind.BOOLEAN, expr=sympy.S.true)
            .build()
        )
        bundle = seal(downstream_ax, h, script, foundations=[
            (fa_bundle, "theorem_a"),
            (fb_bundle, "theorem_b"),
        ])
        assert bundle.proof_result.status == ProofStatus.VERIFIED
