"""Coordinate transformation proof strategy demonstrations.

The ``COORDINATE_TRANSFORM`` strategy follows a universal pattern:

1. Define forward transform T: old → new coordinates
2. Define inverse transform T⁻¹: new → old coordinates
3. Verify round-trip: T⁻¹(T(x)) = x  (automatic in verification)
4. Prove the claim in the transformed coordinate space
5. The verified round-trip guarantees equivalence in the original space

Demonstrated on:
- **Homicidal Chauffeur (HC)**: Cartesian → polar body-frame coordinates.
  The costate norm and switching function become trivially expressed in
  polar coordinates where p₁ = ‖p‖cos(α), p₂ = ‖p‖sin(α).
- **AMM constant-product**: Reserve space → hyperbolic (k, s) coordinates.
  The invariant k = Rx·Ry is a coordinate axis in the transformed space,
  making conservation and fee-growth arguments direct.
"""

from __future__ import annotations

import pytest
import sympy
from sympy import (
    Symbol,
    atan2,
    cos,
    expand,
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


# ===================================================================
# Homicidal Chauffeur — polar costate coordinates
# ===================================================================

# Shared symbols (same as test_optimal_control.py)
x1, x2 = symbols("x_1 x_2", real=True)
p1, p2 = symbols("p_1 p_2", real=True)
phi = Symbol("phi", real=True)
psi = Symbol("psi", real=True)
w = Symbol("w", positive=True)

# Body-frame dynamics
f1 = -phi * x2 + w * sin(psi)
f2 = phi * x1 + w * cos(psi) - 1

# Polar costate coordinates: p1 = rho*cos(alpha), p2 = rho*sin(alpha)
rho = Symbol("rho", positive=True)  # costate norm ||p||
alpha = Symbol("alpha", real=True)  # costate angle


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
                expr=sympy.Eq(sympy.Function("f_1")(x1, x2, phi, psi), f1),
            ),
            Axiom(
                name="dynamics_f2",
                expr=sympy.Eq(sympy.Function("f_2")(x1, x2, phi, psi), f2),
            ),
        ),
    )


class TestHCPolarTransform:
    """Prove HC properties via polar costate coordinate transformation.

    Transform: (p1, p2) → (rho, alpha) where
        p1 = rho*cos(alpha),  p2 = rho*sin(alpha)
    Inverse:
        rho = sqrt(p1² + p2²),  alpha = atan2(p2, p1)

    In polar costate coords the switching function and norm conservation
    have simpler forms.
    """

    # Forward: old symbols → expressions in new coordinates
    polar_forward = {
        "p_1": rho * cos(alpha),
        "p_2": rho * sin(alpha),
    }

    # Inverse: new symbols → expressions in old coordinates
    polar_inverse = {
        "rho": sqrt(p1**2 + p2**2),
        "alpha": atan2(p2, p1),
    }

    def test_switching_function_in_polar(self, hc_axioms):
        """Switching function sigma = p2*x1 - p1*x2 becomes rho*sin(alpha)*x1 - rho*cos(alpha)*x2.

        In polar costate coords this factors as rho*(x1*sin(alpha) - x2*cos(alpha)),
        showing that sigma = 0 iff the position vector is aligned with the costate.
        """
        sigma = p2 * x1 - p1 * x2
        sigma_polar_expected = rho * (x1 * sin(alpha) - x2 * cos(alpha))

        h = hc_axioms.hypothesis(
            "sigma_polar_form",
            expr=sympy.Eq(sigma, sigma_polar_expected),
            description="Switching function factors as rho * angular term",
        )
        script = (
            ProofBuilder(
                hc_axioms,
                h.name,
                name="sigma_polar_proof",
                claim="sigma = rho*(x1*sin(alpha) - x2*cos(alpha)) "
                "in polar costate coords",
            )
            .lemma(
                "sigma_polar",
                LemmaKind.COORDINATE_TRANSFORM,
                expr=sigma,
                expected=sigma_polar_expected,
                transform=self.polar_forward,
                inverse_transform=self.polar_inverse,
                description="Apply polar costate transform to switching function",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_costate_norm_dot_in_polar(self, hc_axioms):
        """d/dt(rho²) = 0 in polar coordinates.

        The costate equations dp1/dt = -p2*phi, dp2/dt = p1*phi give:
            d/dt(p1² + p2²) = 2*p1*(-p2*phi) + 2*p2*(p1*phi) = 0

        In polar coords this is immediate: d/dt(rho²) = 0 because the
        cross terms cancel identically regardless of alpha.
        """
        # Build the norm derivative in Cartesian
        p1_dot = -p2 * phi
        p2_dot = p1 * phi
        d_norm_sq = 2 * (p1 * p1_dot + p2 * p2_dot)

        h = hc_axioms.hypothesis(
            "norm_conserved_polar",
            expr=sympy.Eq(d_norm_sq, 0),
            description="Costate norm conservation via polar transform",
        )
        script = (
            ProofBuilder(
                hc_axioms,
                h.name,
                name="norm_polar_proof",
                claim="d/dt(||p||^2) = 0 verified through polar costate coords",
            )
            .lemma(
                "d_rho_sq_zero",
                LemmaKind.COORDINATE_TRANSFORM,
                expr=d_norm_sq,
                expected=sympy.Integer(0),
                transform=self.polar_forward,
                inverse_transform=self.polar_inverse,
                description="Transform norm derivative to polar — "
                "cross terms cancel as rho²*(cos·sin - sin·cos)*phi = 0",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_hamiltonian_in_polar(self, hc_axioms):
        """Hamiltonian in polar costate coordinates.

        H = p1*f1 + p2*f2 + 1 transforms to:
        H = rho*[cos(alpha)*(-phi*x2 + w*sin(psi))
                + sin(alpha)*(phi*x1 + w*cos(psi) - 1)] + 1
          = rho*[phi*(x1*sin(alpha) - x2*cos(alpha))
                + w*(cos(alpha)*sin(psi) + sin(alpha)*cos(psi))
                - sin(alpha)] + 1
          = rho*[phi*sigma/rho + w*sin(psi + alpha) - sin(alpha)] + 1

        Verify the expanded polar form.
        """
        H = p1 * f1 + p2 * f2 + 1
        H_polar_expected = (
            rho * (
                phi * (x1 * sin(alpha) - x2 * cos(alpha))
                + w * sin(psi + alpha)
                - sin(alpha)
            )
            + 1
        )

        h = hc_axioms.hypothesis(
            "hamiltonian_polar",
            expr=sympy.Eq(H, H_polar_expected),
            description="Hamiltonian in polar costate form",
        )
        script = (
            ProofBuilder(
                hc_axioms,
                h.name,
                name="H_polar_proof",
                claim="H expressed in polar costate coords",
            )
            .lemma(
                "H_polar",
                LemmaKind.COORDINATE_TRANSFORM,
                expr=H,
                expected=H_polar_expected,
                transform=self.polar_forward,
                inverse_transform=self.polar_inverse,
                description="Transform Hamiltonian to polar costate form",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_sealed_polar_costate_conservation(self, hc_axioms):
        """End-to-end sealed proof: costate norm conservation via polar transform."""
        p1_dot = -p2 * phi
        p2_dot = p1 * phi
        d_norm_sq = 2 * (p1 * p1_dot + p2 * p2_dot)

        h = hc_axioms.hypothesis(
            "costate_conserved_polar",
            expr=sympy.Eq(d_norm_sq, 0),
        )
        script = (
            ProofBuilder(
                hc_axioms,
                h.name,
                name="polar_conservation_sealed",
                claim="||p|| conserved — polar coordinate transform proof",
            )
            .lemma(
                "costate_p1",
                LemmaKind.EQUALITY,
                expr=-sympy.diff(expand(p1 * f1 + p2 * f2 + 1), x1),
                expected=-p2 * phi,
                description="dp1/dt from Hamiltonian",
            )
            .lemma(
                "costate_p2",
                LemmaKind.EQUALITY,
                expr=-sympy.diff(expand(p1 * f1 + p2 * f2 + 1), x2),
                expected=p1 * phi,
                depends_on=["costate_p1"],
                description="dp2/dt from Hamiltonian",
            )
            .lemma(
                "norm_conserved_polar",
                LemmaKind.COORDINATE_TRANSFORM,
                expr=d_norm_sq,
                expected=sympy.Integer(0),
                transform=self.polar_forward,
                inverse_transform=self.polar_inverse,
                depends_on=["costate_p1", "costate_p2"],
                description="Transform to polar — cross terms cancel",
            )
            .build()
        )
        bundle = seal(hc_axioms, h, script)
        assert len(bundle.bundle_hash) == 64


# ===================================================================
# AMM constant-product — hyperbolic coordinates
# ===================================================================

# Reserve symbols
Rx = Symbol("R_x", positive=True)
Ry = Symbol("R_y", positive=True)
fee = Symbol("f", positive=True)
dx = Symbol("dx", positive=True)

# Hyperbolic coordinates: k = Rx*Ry (invariant), s = Rx/Ry (price)
k = Symbol("k", positive=True)
s = Symbol("s", positive=True)


@pytest.fixture
def amm_axioms():
    return AxiomSet(
        name="amm_constant_product",
        axioms=(
            Axiom(name="reserve_x_positive", expr=Rx > 0),
            Axiom(name="reserve_y_positive", expr=Ry > 0),
            Axiom(name="fee_in_unit_interval", expr=sympy.And(fee > 0, fee < 1)),
        ),
    )


class TestAMMHyperbolicTransform:
    """Prove AMM properties via hyperbolic coordinate transformation.

    Transform: (Rx, Ry) → (k, s) where
        k = Rx * Ry   (the constant-product invariant)
        s = Rx / Ry   (the marginal price ratio)
    Inverse:
        Rx = sqrt(k * s)
        Ry = sqrt(k / s)

    In (k, s) space the invariant surface is simply the k-axis,
    making conservation and monotonicity arguments direct.
    """

    # Forward: old → new
    hyp_forward = {
        "R_x": sqrt(k * s),
        "R_y": sqrt(k / s),
    }

    # Inverse: new → old
    hyp_inverse = {
        "k": Rx * Ry,
        "s": Rx / Ry,
    }

    def test_invariant_is_coordinate(self, amm_axioms):
        """In hyperbolic coords, k = Rx*Ry is literally a coordinate.

        The constant-product invariant Rx*Ry = k becomes the trivial
        identity k = k after transformation — the deepest justification
        for why (k, s) are the natural coordinates for AMMs.
        """
        product = Rx * Ry

        h = amm_axioms.hypothesis(
            "invariant_is_k",
            expr=sympy.Eq(product, k),
            description="Product invariant = k coordinate",
        )
        script = (
            ProofBuilder(
                amm_axioms,
                h.name,
                name="invariant_coordinate_proof",
                claim="Rx*Ry = k is a tautology in hyperbolic coords",
            )
            .lemma(
                "product_is_k",
                LemmaKind.COORDINATE_TRANSFORM,
                expr=product,
                expected=k,
                transform=self.hyp_forward,
                inverse_transform=self.hyp_inverse,
                assumptions={
                    "R_x": {"positive": True},
                    "R_y": {"positive": True},
                    "k": {"positive": True},
                    "s": {"positive": True},
                },
                description="Rx*Ry → sqrt(k*s)*sqrt(k/s) = k",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_price_ratio_is_coordinate(self, amm_axioms):
        """The marginal price Rx/Ry = s in hyperbolic coords.

        Marginal price in a constant-product AMM is the reserve ratio.
        In (k, s) coords this is literally the s coordinate.
        """
        price = Rx / Ry

        h = amm_axioms.hypothesis(
            "price_is_s",
            expr=sympy.Eq(price, s),
            description="Price ratio = s coordinate",
        )
        script = (
            ProofBuilder(
                amm_axioms,
                h.name,
                name="price_coordinate_proof",
                claim="Rx/Ry = s is a tautology in hyperbolic coords",
            )
            .lemma(
                "ratio_is_s",
                LemmaKind.COORDINATE_TRANSFORM,
                expr=price,
                expected=s,
                transform=self.hyp_forward,
                inverse_transform=self.hyp_inverse,
                assumptions={
                    "R_x": {"positive": True},
                    "R_y": {"positive": True},
                    "k": {"positive": True},
                    "s": {"positive": True},
                },
                description="Rx/Ry → sqrt(k*s)/sqrt(k/s) = s",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_swap_output_in_hyperbolic(self, amm_axioms):
        """AMM output formula expressed in (k, s) coordinates.

        dy = Ry * dx*(1-f) / (Rx + dx*(1-f))

        In hyperbolic coords (substituting Rx = sqrt(k*s), Ry = sqrt(k/s)):
        the output becomes a function of (k, s, dx, f) where k appears
        only as a scaling factor — swaps move along s while k captures
        fee accumulation.
        """
        net = dx * (1 - fee)
        dy = Ry * net / (Rx + net)

        # The same formula in hyperbolic coordinates
        dy_hyp = sqrt(k / s) * net / (sqrt(k * s) + net)

        h = amm_axioms.hypothesis(
            "swap_output_hyperbolic",
            expr=sympy.Eq(dy, dy_hyp),
            description="AMM output in hyperbolic coordinates",
        )
        script = (
            ProofBuilder(
                amm_axioms,
                h.name,
                name="swap_hyperbolic_proof",
                claim="dy formula is equivalent in (k, s) coords",
            )
            .lemma(
                "dy_transform",
                LemmaKind.COORDINATE_TRANSFORM,
                expr=dy,
                expected=dy_hyp,
                transform=self.hyp_forward,
                inverse_transform=self.hyp_inverse,
                assumptions={
                    "R_x": {"positive": True},
                    "R_y": {"positive": True},
                    "k": {"positive": True},
                    "s": {"positive": True},
                    "f": {"positive": True},
                    "dx": {"positive": True},
                },
                description="Substitute Rx=sqrt(k*s), Ry=sqrt(k/s) into dy formula",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_sealed_invariant_proof(self, amm_axioms):
        """Sealed proof: the constant-product invariant is a coordinate in (k,s) space."""
        product = Rx * Ry

        h = amm_axioms.hypothesis(
            "invariant_k_sealed",
            expr=sympy.Eq(product, k),
        )
        script = (
            ProofBuilder(
                amm_axioms,
                h.name,
                name="invariant_sealed",
                claim="Rx*Ry = k via hyperbolic coordinate transform",
            )
            .lemma(
                "product_is_k",
                LemmaKind.COORDINATE_TRANSFORM,
                expr=product,
                expected=k,
                transform=self.hyp_forward,
                inverse_transform=self.hyp_inverse,
                assumptions={
                    "R_x": {"positive": True},
                    "R_y": {"positive": True},
                    "k": {"positive": True},
                    "s": {"positive": True},
                },
                description="Rx*Ry = sqrt(k*s)*sqrt(k/s) = k",
            )
            .build()
        )
        bundle = seal(amm_axioms, h, script)
        assert len(bundle.bundle_hash) == 64


# ===================================================================
# Strategy edge cases
# ===================================================================


class TestCoordinateTransformEdgeCases:
    """Verify the strategy rejects bad inputs correctly."""

    def test_missing_transform_fails(self):
        """COORDINATE_TRANSFORM without transform dict should fail."""
        axioms = AxiomSet(
            name="test",
            axioms=(Axiom(name="trivial", expr=sympy.true),),
        )
        h = axioms.hypothesis("test_h", expr=sympy.Eq(x1, x1))
        script = (
            ProofBuilder(axioms, h.name, name="bad", claim="missing transform")
            .lemma(
                "no_transform",
                LemmaKind.COORDINATE_TRANSFORM,
                expr=x1,
                expected=x1,
                # transform and inverse_transform intentionally omitted
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "FAILED"
        assert "requires both" in result.failure_summary

    def test_missing_expected_fails(self):
        """COORDINATE_TRANSFORM without expected should fail."""
        axioms = AxiomSet(
            name="test",
            axioms=(Axiom(name="trivial", expr=sympy.true),),
        )
        h = axioms.hypothesis("test_h", expr=sympy.Eq(x1, x1))
        script = (
            ProofBuilder(axioms, h.name, name="bad", claim="missing expected")
            .lemma(
                "no_expected",
                LemmaKind.COORDINATE_TRANSFORM,
                expr=x1,
                # expected intentionally omitted
                transform={"x_1": rho * cos(alpha)},
                inverse_transform={"rho": sqrt(x1**2), "alpha": sympy.Integer(0)},
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "FAILED"
        assert "requires 'expected'" in result.failure_summary

    def test_bad_roundtrip_fails(self):
        """A non-invertible transform should fail the round-trip check."""
        axioms = AxiomSet(
            name="test",
            axioms=(Axiom(name="trivial", expr=sympy.true),),
        )
        h = axioms.hypothesis("test_h", expr=sympy.Eq(x1, x1))
        # Forward: x1 → rho, but inverse says rho → x1 + 1 (wrong!)
        script = (
            ProofBuilder(axioms, h.name, name="bad", claim="bad roundtrip")
            .lemma(
                "bad_rt",
                LemmaKind.COORDINATE_TRANSFORM,
                expr=x1,
                expected=x1,
                transform={"x_1": rho},
                inverse_transform={"rho": x1 + 1},
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "FAILED"
        assert "Round-trip failed" in result.failure_summary


# ===================================================================
# Serialization round-trip
# ===================================================================


class TestCoordinateTransformSerialization:
    """Verify evidence serialization handles transform fields."""

    def test_evidence_roundtrip(self):
        axioms = AxiomSet(
            name="test",
            axioms=(Axiom(name="trivial", expr=sympy.true),),
        )
        script = (
            ProofBuilder(axioms, "test_h", name="ser_test", claim="roundtrip")
            .lemma(
                "ct_lemma",
                LemmaKind.COORDINATE_TRANSFORM,
                expr=Rx * Ry,
                expected=k,
                transform={"R_x": sqrt(k * s), "R_y": sqrt(k / s)},
                inverse_transform={"k": Rx * Ry, "s": Rx / Ry},
                description="test serialization",
            )
            .build()
        )
        evidence = script.to_evidence()
        restored = type(script).from_evidence(evidence)

        assert restored.lemmas[0].kind == LemmaKind.COORDINATE_TRANSFORM
        assert restored.lemmas[0].transform is not None
        assert restored.lemmas[0].inverse_transform is not None
        assert len(restored.lemmas[0].transform) == 2
        assert len(restored.lemmas[0].inverse_transform) == 2
