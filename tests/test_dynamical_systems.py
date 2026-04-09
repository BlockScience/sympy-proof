"""User-testing: dynamical systems and inductive safety proofs.

Exercises symproof with patterns from gds-core:
- Withdrawal/deposit state transitions with predicate guards
- Inductive invariant preservation (balance non-negativity)
- Implication-based proof obligations
- Compositional disproof
- Conservation laws in stock-flow systems
"""

from __future__ import annotations

import pytest
import sympy

from symproof import (
    Axiom,
    AxiomSet,
    LemmaKind,
    ProofBuilder,
    disprove,
    seal,
    verify_proof,
)
from symproof.bundle import check_consistency, ContradictionError
from symproof.tactics import auto_lemma, try_implication, try_simplify

# ---------------------------------------------------------------------------
# Shared symbols
# ---------------------------------------------------------------------------

x_prev = sympy.Symbol("x_prev")  # pre-state balance
u = sympy.Symbol("u")  # input (deposit/withdraw amount)


# ===================================================================
# Withdrawal safety (predicate-guarded transition)
# ===================================================================


class TestWithdrawalSafety:
    """Withdrawal: x_new = x_prev - u, guarded by u < x_prev."""

    @pytest.fixture
    def balance_axioms(self):
        return AxiomSet(
            name="balance_nonneg",
            axioms=(
                Axiom(name="balance_nonneg", expr=sympy.Ge(x_prev, 0)),
                Axiom(name="withdraw_nonneg", expr=sympy.Ge(u, 0)),
            ),
        )

    def test_predicate_implies_post_nonneg(self, balance_axioms):
        """x_prev >= 0 AND u < x_prev => x_prev - u > 0."""
        h = balance_axioms.hypothesis(
            "withdraw_safe",
            expr=sympy.Implies(
                sympy.And(sympy.Ge(x_prev, 0), sympy.Lt(u, x_prev)),
                sympy.Gt(x_prev - u, 0),
            ),
        )
        script = (
            ProofBuilder(
                balance_axioms,
                h.name,
                name="withdraw_safety_proof",
                claim="Predicate guard ensures post-state > 0",
            )
            .lemma(
                "guarded_implication",
                LemmaKind.BOOLEAN,
                expr=sympy.Implies(
                    sympy.And(sympy.Ge(x_prev, 0), sympy.Lt(u, x_prev)),
                    sympy.Gt(x_prev - u, 0),
                ),
                description="Standard implication: guard => postcondition",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_deposit_preserves_nonneg(self, balance_axioms):
        """x_prev >= 0 AND u >= 0 => x_prev + u >= 0 (no guard needed)."""
        h = balance_axioms.hypothesis(
            "deposit_safe",
            expr=sympy.Implies(
                sympy.And(sympy.Ge(x_prev, 0), sympy.Ge(u, 0)),
                sympy.Ge(x_prev + u, 0),
            ),
        )
        script = (
            ProofBuilder(
                balance_axioms,
                h.name,
                name="deposit_safety_proof",
                claim="Deposit always preserves non-negativity",
            )
            .lemma(
                "sum_nonneg",
                LemmaKind.BOOLEAN,
                expr=sympy.Implies(
                    sympy.And(sympy.Ge(x_prev, 0), sympy.Ge(u, 0)),
                    sympy.Ge(x_prev + u, 0),
                ),
                assumptions={
                    "x_prev": {"nonnegative": True},
                    "u": {"nonnegative": True},
                },
                description="Sum of non-negatives is non-negative",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary


# ===================================================================
# Inductive safety: invariant preservation
# ===================================================================


class TestInductiveSafety:
    """Inductive proofs: I(x) AND transition => I(x')."""

    @pytest.fixture
    def inductive_axioms(self):
        return AxiomSet(
            name="inductive_safety",
            axioms=(
                Axiom(
                    name="invariant_holds",
                    expr=sympy.Ge(x_prev, 0),
                    description="Balance is non-negative (invariant)",
                ),
                Axiom(
                    name="input_nonneg",
                    expr=sympy.Ge(u, 0),
                    description="Input is non-negative",
                ),
            ),
        )

    def test_inductive_step_deposit(self, inductive_axioms):
        """Inductive step for deposit: I(x) => I(x + u)."""
        x_new = x_prev + u

        h = inductive_axioms.hypothesis(
            "deposit_inductive",
            expr=sympy.Ge(x_new, 0),
        )
        script = (
            ProofBuilder(
                inductive_axioms,
                h.name,
                name="deposit_inductive_proof",
                claim="Deposit preserves non-negativity inductively",
            )
            .lemma(
                "invariant_preserved",
                LemmaKind.QUERY,
                expr=sympy.Q.nonnegative(x_prev + u),
                assumptions={
                    "x_prev": {"nonnegative": True},
                    "u": {"nonnegative": True},
                },
                description="x_prev >= 0 and u >= 0 => x_prev + u >= 0",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary


# ===================================================================
# Conservation laws in stock-flow systems
# ===================================================================


class TestConservationLaws:
    """Total quantity conservation under paired flows."""

    @pytest.fixture
    def conservation_axioms(self):
        A = sympy.Symbol("A", nonnegative=True)
        B = sympy.Symbol("B", nonnegative=True)
        flow = sympy.Symbol("flow", nonnegative=True)
        return AxiomSet(
            name="two_stock_conservation",
            axioms=(
                Axiom(name="stock_a_nonneg", expr=sympy.Ge(A, 0)),
                Axiom(name="stock_b_nonneg", expr=sympy.Ge(B, 0)),
                Axiom(name="flow_nonneg", expr=sympy.Ge(flow, 0)),
            ),
        )

    def test_total_conserved(self, conservation_axioms):
        """Transfer from A to B: (A - flow) + (B + flow) = A + B."""
        A = sympy.Symbol("A", nonnegative=True)
        B = sympy.Symbol("B", nonnegative=True)
        flow = sympy.Symbol("flow", nonnegative=True)

        total_pre = A + B
        total_post = (A - flow) + (B + flow)

        h = conservation_axioms.hypothesis(
            "total_conserved",
            expr=sympy.Eq(total_post, total_pre),
        )
        script = (
            ProofBuilder(
                conservation_axioms,
                h.name,
                name="conservation_proof",
                claim="Transfer conserves total: (A-f) + (B+f) = A + B",
            )
            .lemma(
                "cancellation",
                LemmaKind.EQUALITY,
                expr=total_post,
                expected=total_pre,
                description="Flow cancels: -flow + flow = 0",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_three_stock_conservation(self, conservation_axioms):
        """Three-stock system: A -> B -> C, net flows cancel."""
        A = sympy.Symbol("A")
        B = sympy.Symbol("B")
        C = sympy.Symbol("C")
        f1 = sympy.Symbol("f1")
        f2 = sympy.Symbol("f2")

        total_pre = A + B + C
        total_post = (A - f1) + (B + f1 - f2) + (C + f2)

        axioms = AxiomSet(
            name="three_stock",
            axioms=(Axiom(name="total_exists", expr=sympy.Ge(A + B + C, 0)),),
        )
        h = axioms.hypothesis(
            "three_stock_conserved",
            expr=sympy.Eq(total_post, total_pre),
        )
        script = (
            ProofBuilder(
                axioms,
                h.name,
                name="three_stock_proof",
                claim="Cascaded transfer conserves total",
            )
            .lemma(
                "cancel",
                LemmaKind.EQUALITY,
                expr=sympy.expand(total_post),
                expected=total_pre,
                description="f1 and f2 both cancel",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary


# ===================================================================
# Compositional disproof
# ===================================================================


class TestCompositionalDisproof:
    """Prove ~H and compose into a Disproof of H."""

    def test_disprove_false_claim(self):
        """Disprove: x + y = x (false when y != 0)."""
        x = sympy.Symbol("x", real=True)
        y = sympy.Symbol("y", positive=True)

        axioms = AxiomSet(
            name="nonzero_y",
            axioms=(Axiom(name="y_positive", expr=y > 0),),
        )

        # The false claim
        h = axioms.hypothesis(
            "false_equality",
            expr=sympy.Eq(x + y, x),
        )

        # Negate: x + y != x, i.e., NOT(x + y = x)
        h_neg = h.negate()

        # Prove the negation: simplify(x + y - x) = y != 0
        script = (
            ProofBuilder(
                axioms,
                h_neg.name,
                name="negation_proof",
                claim="x + y != x since y > 0",
            )
            .lemma(
                "difference_is_y",
                LemmaKind.EQUALITY,
                expr=sympy.simplify(x + y - x),
                expected=y,
                description="x + y - x simplifies to y",
            )
            .lemma(
                "y_nonzero",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(y),
                assumptions={"y": {"positive": True}},
                depends_on=["difference_is_y"],
                description="y > 0 so difference is nonzero",
            )
            .build()
        )

        neg_bundle = seal(axioms, h_neg, script)
        d = disprove(h, neg_bundle)
        assert d.disproof_hash
        assert d.hypothesis.name == "false_equality"


# ===================================================================
# Consistency checking
# ===================================================================


class TestConsistencyChecking:
    """check_consistency detects contradictions."""

    def test_no_contradiction_different_axioms(self):
        """Bundles under different axiom sets cannot contradict."""
        x = sympy.Symbol("x")

        ax1 = AxiomSet(
            name="positive_x",
            axioms=(Axiom(name="x_pos", expr=x > 0),),
        )
        ax2 = AxiomSet(
            name="negative_x",
            axioms=(Axiom(name="x_neg", expr=x < 0),),
        )

        h1 = ax1.hypothesis("claim_pos", expr=x > 0)
        h2 = ax2.hypothesis("claim_neg", expr=x < 0)

        s1 = (
            ProofBuilder(ax1, h1.name, name="p1", claim="x > 0")
            .lemma(
                "pos",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(x),
                assumptions={"x": {"positive": True}},
            )
            .build()
        )
        s2 = (
            ProofBuilder(ax2, h2.name, name="p2", claim="x < 0")
            .lemma(
                "neg",
                LemmaKind.QUERY,
                expr=sympy.Q.negative(x),
                assumptions={"x": {"negative": True}},
            )
            .build()
        )

        b1 = seal(ax1, h1, s1)
        b2 = seal(ax2, h2, s2)

        # Should NOT raise — different axiom sets
        check_consistency(b1, b2)


# ===================================================================
# Tactics on dynamical system expressions
# ===================================================================


class TestTacticsDynamicalSystems:
    """Exercise tactics helpers on system expressions."""

    def test_try_implication_deposit(self):
        """try_implication on deposit safety."""
        result = try_implication(
            sympy.And(sympy.Ge(x_prev, 0), sympy.Ge(u, 0)),
            sympy.Ge(x_prev + u, 0),
            assumptions={
                "x_prev": {"nonnegative": True},
                "u": {"nonnegative": True},
            },
        )
        assert result is True

    def test_try_simplify_conservation(self):
        """try_simplify on A + B == (A - f) + (B + f)."""
        A = sympy.Symbol("A")
        B = sympy.Symbol("B")
        f = sympy.Symbol("f")

        expr = sympy.Eq((A - f) + (B + f), A + B)
        result = try_simplify(expr)
        assert result is True

    def test_auto_lemma_on_conservation(self):
        """auto_lemma should find EQUALITY for conservation identity."""
        A = sympy.Symbol("A")
        B = sympy.Symbol("B")
        f = sympy.Symbol("f")

        lemma = auto_lemma(
            "conservation",
            (A - f) + (B + f),
            expected=A + B,
        )
        assert lemma is not None
        assert lemma.kind == LemmaKind.EQUALITY

    def test_auto_lemma_on_positivity(self):
        """auto_lemma should detect positivity via QUERY."""
        x = sympy.Symbol("x")
        lemma = auto_lemma(
            "x_positive",
            sympy.Q.positive(x),
            assumptions={"x": {"positive": True}},
        )
        assert lemma is not None
        assert lemma.kind == LemmaKind.QUERY
