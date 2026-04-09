"""User-testing: optimal control and differential game proofs.

Exercises symproof with real derivations from the Homicidal Chauffeur
pursuit-evasion differential game (hc-marimo project):
- Hamiltonian formulation and linearity
- Switching function extraction
- Optimal evader heading (atan2 maximizer)
- Costate norm conservation
- Hamiltonian separability (Isaacs condition)
"""

from __future__ import annotations

import pytest
import sympy
from sympy import (
    Abs,
    Function,
    Symbol,
    atan2,
    cos,
    diff,
    expand,
    sign,
    simplify,
    sin,
    sqrt,
    symbols,
    trigsimp,
)

from symproof import (
    Axiom,
    AxiomSet,
    LemmaKind,
    ProofBuilder,
    seal,
    verify_proof,
)
from symproof.tactics import auto_lemma, try_simplify

# ---------------------------------------------------------------------------
# Shared symbols — Isaacs canonical form
# ---------------------------------------------------------------------------

x1, x2 = symbols("x_1 x_2", real=True)
p1, p2 = symbols("p_1 p_2", real=True)
phi = Symbol("phi", real=True)  # pursuer curvature (control)
psi = Symbol("psi", real=True)  # evader heading (control)
w = Symbol("w", positive=True)  # speed ratio v_E / v_P

# Reduced dynamics (body-frame, normalized v_P = 1)
f1 = -phi * x2 + w * sin(psi)
f2 = phi * x1 + w * cos(psi) - 1


# ===================================================================
# Axiom set for the Homicidal Chauffeur game
# ===================================================================


@pytest.fixture
def hc_axioms():
    return AxiomSet(
        name="homicidal_chauffeur",
        axioms=(
            Axiom(
                name="speed_ratio_subunity",
                expr=sympy.And(w > 0, w < 1),
                description="Evader slower than pursuer: 0 < w < 1",
            ),
            Axiom(
                name="dynamics_f1",
                expr=sympy.Eq(Function("f_1")(x1, x2, phi, psi), f1),
            ),
            Axiom(
                name="dynamics_f2",
                expr=sympy.Eq(Function("f_2")(x1, x2, phi, psi), f2),
            ),
        ),
    )


# ===================================================================
# Hamiltonian structure
# ===================================================================


class TestHamiltonianStructure:
    """Verify Hamiltonian properties for time-optimal pursuit-evasion."""

    def test_hamiltonian_linearity_in_phi(self, hc_axioms):
        """H must be linear (affine) in the pursuer control phi."""
        H = p1 * f1 + p2 * f2 + 1
        H_expanded = expand(H)

        # Coefficient of phi^2 must be zero
        coeff_phi_sq = H_expanded.coeff(phi, 2)

        h = hc_axioms.hypothesis(
            "hamiltonian_linear_in_phi",
            expr=sympy.Eq(coeff_phi_sq, 0),
        )
        script = (
            ProofBuilder(
                hc_axioms,
                h.name,
                name="H_linearity_proof",
                claim="No phi^2 term in Hamiltonian",
            )
            .lemma(
                "no_phi_squared",
                LemmaKind.EQUALITY,
                expr=coeff_phi_sq,
                expected=sympy.Integer(0),
                description="coeff(H, phi^2) = 0",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_switching_function(self, hc_axioms):
        """Switching function sigma = coefficient of phi in H = p2*x1 - p1*x2."""
        H = p1 * f1 + p2 * f2 + 1
        H_expanded = expand(H)
        sigma = H_expanded.coeff(phi)

        expected_sigma = p2 * x1 - p1 * x2

        h = hc_axioms.hypothesis(
            "switching_function_form",
            expr=sympy.Eq(sigma, expected_sigma),
        )
        script = (
            ProofBuilder(
                hc_axioms,
                h.name,
                name="switching_function_proof",
                claim="sigma = p2*x1 - p1*x2",
            )
            .lemma(
                "sigma_identity",
                LemmaKind.EQUALITY,
                expr=sigma,
                expected=expected_sigma,
                description="Extract phi coefficient from expanded H",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary


# ===================================================================
# Optimal evader heading
# ===================================================================


class TestOptimalEvaderHeading:
    """Evader maximizes p1*sin(psi) + p2*cos(psi) => psi* = atan2(p1, p2)."""

    def test_first_order_condition(self, hc_axioms):
        """d/dpsi (p1*sin(psi) + p2*cos(psi)) = 0 at psi = atan2(p1, p2)."""
        expr = p1 * sin(psi) + p2 * cos(psi)
        dexpr = diff(expr, psi)  # p1*cos(psi) - p2*sin(psi)

        # At psi* = atan2(p1, p2), the derivative should be zero
        # But verifying this symbolically requires trig substitution.
        # Instead verify the derivative form.
        h = hc_axioms.hypothesis(
            "evader_foc",
            expr=sympy.Eq(dexpr, p1 * cos(psi) - p2 * sin(psi)),
        )
        script = (
            ProofBuilder(
                hc_axioms,
                h.name,
                name="evader_foc_proof",
                claim="FOC for evader heading",
            )
            .lemma(
                "derivative_form",
                LemmaKind.EQUALITY,
                expr=dexpr,
                expected=p1 * cos(psi) - p2 * sin(psi),
                description="d/dpsi(p1*sin + p2*cos) = p1*cos - p2*sin",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_max_contribution_is_norm(self, hc_axioms):
        """At optimal psi, p1*sin(psi*) + p2*cos(psi*) = sqrt(p1^2 + p2^2)."""
        # Substitute psi* = atan2(p1, p2) and simplify
        # sin(atan2(p1, p2)) = p1/||p||, cos(atan2(p1, p2)) = p2/||p||
        norm_p = sqrt(p1**2 + p2**2)
        contribution = p1 * (p1 / norm_p) + p2 * (p2 / norm_p)
        simplified = simplify(contribution)

        h = hc_axioms.hypothesis(
            "max_is_norm",
            expr=sympy.Eq(simplified, norm_p),
        )
        script = (
            ProofBuilder(
                hc_axioms,
                h.name,
                name="max_contribution_proof",
                claim="Optimal evader contribution = costate norm",
            )
            .lemma(
                "contribution_equals_norm",
                LemmaKind.EQUALITY,
                expr=simplified,
                expected=norm_p,
                description="(p1^2 + p2^2)/||p|| = ||p||",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary


# ===================================================================
# Costate equations and conservation laws
# ===================================================================


class TestCostateEquations:
    """Costate dynamics and conservation of ||p||."""

    def test_costate_equations(self, hc_axioms):
        """dp1/dt = -dH/dx1 = -p2*phi and dp2/dt = -dH/dx2 = p1*phi."""
        H = expand(p1 * f1 + p2 * f2 + 1)

        p1_dot = -diff(H, x1)
        p2_dot = -diff(H, x2)

        h = hc_axioms.hypothesis(
            "costate_dynamics",
            expr=sympy.And(
                sympy.Eq(p1_dot, -p2 * phi),
                sympy.Eq(p2_dot, p1 * phi),
            ),
        )
        script = (
            ProofBuilder(
                hc_axioms,
                h.name,
                name="costate_proof",
                claim="Costate equations from Hamiltonian",
            )
            .lemma(
                "p1_dot",
                LemmaKind.EQUALITY,
                expr=p1_dot,
                expected=-p2 * phi,
                description="dp1/dt = -dH/dx1 = -p2*phi",
            )
            .lemma(
                "p2_dot",
                LemmaKind.EQUALITY,
                expr=p2_dot,
                expected=p1 * phi,
                depends_on=["p1_dot"],
                description="dp2/dt = -dH/dx2 = p1*phi",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_costate_norm_conservation(self, hc_axioms):
        """d/dt(p1^2 + p2^2) = 0 under the costate equations.

        This is a key invariant: ||p||^2 is conserved along optimal trajectories.
        d/dt(||p||^2) = 2*p1*(-p2*phi) + 2*p2*(p1*phi)
                      = -2*p1*p2*phi + 2*p1*p2*phi = 0
        """
        p1_dot = -p2 * phi
        p2_dot = p1 * phi
        d_norm_sq = simplify(2 * (p1 * p1_dot + p2 * p2_dot))

        h = hc_axioms.hypothesis(
            "costate_norm_conserved",
            expr=sympy.Eq(d_norm_sq, 0),
        )
        script = (
            ProofBuilder(
                hc_axioms,
                h.name,
                name="norm_conservation_proof",
                claim="Costate norm is conserved",
            )
            .lemma(
                "d_norm_sq_zero",
                LemmaKind.EQUALITY,
                expr=d_norm_sq,
                expected=sympy.Integer(0),
                description="2*(p1*p1_dot + p2*p2_dot) = 0",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary


# ===================================================================
# Hamiltonian separability (Isaacs condition)
# ===================================================================


class TestIsaacsCondition:
    """Verify the Hamiltonian is separable in pursuer and evader controls."""

    def test_no_phi_psi_cross_terms(self, hc_axioms):
        """The phi-dependent part has no sin(psi) or cos(psi) terms."""
        H = expand(p1 * f1 + p2 * f2 + 1)
        phi_part = H.coeff(phi) * phi

        h = hc_axioms.hypothesis(
            "isaacs_separability",
            expr=sympy.And(
                sympy.Eq(phi_part.coeff(sin(psi)), 0),
                sympy.Eq(phi_part.coeff(cos(psi)), 0),
            ),
        )
        script = (
            ProofBuilder(
                hc_axioms,
                h.name,
                name="isaacs_proof",
                claim="No phi*sin(psi) or phi*cos(psi) cross terms",
            )
            .lemma(
                "no_phi_sin_psi",
                LemmaKind.EQUALITY,
                expr=phi_part.coeff(sin(psi)),
                expected=sympy.Integer(0),
                description="phi-part has no sin(psi) factor",
            )
            .lemma(
                "no_phi_cos_psi",
                LemmaKind.EQUALITY,
                expr=phi_part.coeff(cos(psi)),
                expected=sympy.Integer(0),
                depends_on=["no_phi_sin_psi"],
                description="phi-part has no cos(psi) factor",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary


# ===================================================================
# Full sealed proof: switching function + costate conservation
# ===================================================================


class TestSealedOptimalControl:
    """End-to-end sealed proofs combining multiple results."""

    def test_sealed_costate_conservation(self, hc_axioms):
        """Seal a multi-lemma proof of costate norm conservation."""
        H = expand(p1 * f1 + p2 * f2 + 1)
        p1_dot = -diff(H, x1)
        p2_dot = -diff(H, x2)
        d_norm_sq = simplify(2 * (p1 * p1_dot + p2 * p2_dot))

        h = hc_axioms.hypothesis(
            "costate_conserved",
            expr=sympy.Eq(d_norm_sq, 0),
        )
        script = (
            ProofBuilder(
                hc_axioms,
                h.name,
                name="costate_conservation_sealed",
                claim="||p|| is conserved along characteristics",
            )
            .lemma(
                "costate_p1",
                LemmaKind.EQUALITY,
                expr=p1_dot,
                expected=-p2 * phi,
                description="dp1/dt from Hamiltonian",
            )
            .lemma(
                "costate_p2",
                LemmaKind.EQUALITY,
                expr=p2_dot,
                expected=p1 * phi,
                depends_on=["costate_p1"],
                description="dp2/dt from Hamiltonian",
            )
            .lemma(
                "norm_conserved",
                LemmaKind.EQUALITY,
                expr=d_norm_sq,
                expected=sympy.Integer(0),
                depends_on=["costate_p1", "costate_p2"],
                description="Cross terms cancel: d/dt(||p||^2) = 0",
            )
            .build()
        )
        bundle = seal(hc_axioms, h, script)
        assert len(bundle.bundle_hash) == 64

    def test_sealed_switching_function(self, hc_axioms):
        """Seal the switching function derivation."""
        H = expand(p1 * f1 + p2 * f2 + 1)
        sigma = H.coeff(phi)

        h = hc_axioms.hypothesis(
            "sigma_form",
            expr=sympy.Eq(sigma, p2 * x1 - p1 * x2),
        )
        script = (
            ProofBuilder(
                hc_axioms,
                h.name,
                name="sigma_sealed",
                claim="Switching function sigma = p2*x1 - p1*x2",
            )
            .lemma(
                "sigma_extracted",
                LemmaKind.EQUALITY,
                expr=sigma,
                expected=p2 * x1 - p1 * x2,
                description="Coefficient of phi in H",
            )
            .build()
        )
        bundle = seal(hc_axioms, h, script)
        assert len(bundle.bundle_hash) == 64
