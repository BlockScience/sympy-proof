"""User-testing: control system stability proofs.

Exercises symproof with real derivations from the ADCS lifecycle demo:
- Characteristic polynomial for PD-controlled rigid body
- Routh-Hurwitz stability conditions
- Steady-state pointing error bounds
- Sensitivity analysis (error decreases with gain)
- Inertia tensor composition (parallel axis theorem)
"""

from __future__ import annotations

import pytest
import sympy

from symproof import (
    Axiom,
    AxiomSet,
    LemmaKind,
    ProofBuilder,
    seal,
    verify_proof,
)
from symproof.tactics import auto_lemma, try_query

# ---------------------------------------------------------------------------
# Shared symbols
# ---------------------------------------------------------------------------

Jxx = sympy.Symbol("J_xx", positive=True)
Jyy = sympy.Symbol("J_yy", positive=True)
Jzz = sympy.Symbol("J_zz", positive=True)
Kp = sympy.Symbol("K_p", positive=True)
Kd = sympy.Symbol("K_d", positive=True)
s = sympy.Symbol("s")
n = sympy.Symbol("n", positive=True)  # orbital rate


# ===================================================================
# Axiom set: PD-controlled single-axis rigid body
# ===================================================================


@pytest.fixture
def pd_axioms():
    return AxiomSet(
        name="pd_single_axis",
        axioms=(
            Axiom(name="inertia_positive", expr=Jxx > 0),
            Axiom(name="proportional_gain_positive", expr=Kp > 0),
            Axiom(name="derivative_gain_positive", expr=Kd > 0),
        ),
    )


# ===================================================================
# Characteristic polynomial
# ===================================================================


class TestCharacteristicPolynomial:
    """PD controller on rigid body: J*s^2 + Kd*s + Kp/2 = 0."""

    def test_polynomial_form(self, pd_axioms):
        """Characteristic polynomial has the expected form."""
        poly = Jxx * s**2 + Kd * s + Kp / 2

        h = pd_axioms.hypothesis(
            "char_poly_form",
            expr=sympy.Eq(poly, Jxx * s**2 + Kd * s + Kp / 2),
        )
        script = (
            ProofBuilder(
                pd_axioms,
                h.name,
                name="char_poly_proof",
                claim="Characteristic polynomial is J*s^2 + Kd*s + Kp/2",
            )
            .lemma(
                "poly_identity",
                LemmaKind.EQUALITY,
                expr=poly,
                expected=Jxx * s**2 + Kd * s + Kp / 2,
                description="Direct identity",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_discriminant(self, pd_axioms):
        """Discriminant = Kd^2 - 2*J*Kp (used for eigenvalue classification)."""
        discriminant = Kd**2 - 2 * Jxx * Kp

        h = pd_axioms.hypothesis(
            "discriminant_form",
            expr=sympy.Eq(discriminant, Kd**2 - 2 * Jxx * Kp),
        )
        script = (
            ProofBuilder(
                pd_axioms,
                h.name,
                name="discriminant_proof",
                claim="Discriminant of char poly",
            )
            .lemma(
                "disc_value",
                LemmaKind.EQUALITY,
                expr=discriminant,
                expected=Kd**2 - 2 * Jxx * Kp,
                description="b^2 - 4ac with a=J, b=Kd, c=Kp/2",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary


# ===================================================================
# Routh-Hurwitz stability
# ===================================================================


class TestRouthHurwitz:
    """All Routh rows positive => stable closed loop."""

    def test_routh_row0_positive(self, pd_axioms):
        """Row 0 coefficient (leading) = J_xx > 0."""
        h = pd_axioms.hypothesis(
            "routh_row0", expr=Jxx > 0,
        )
        script = (
            ProofBuilder(
                pd_axioms,
                h.name,
                name="routh_row0_proof",
                claim="Routh row 0: J_xx > 0",
            )
            .lemma(
                "routh_row0_positive",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(Jxx),
                assumptions={"J_xx": {"positive": True}},
                description="J_xx is positive by axiom",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_routh_row1_positive(self, pd_axioms):
        """Row 1 coefficient = K_d > 0."""
        h = pd_axioms.hypothesis("routh_row1", expr=Kd > 0)
        script = (
            ProofBuilder(
                pd_axioms,
                h.name,
                name="routh_row1_proof",
                claim="Routh row 1: K_d > 0",
            )
            .lemma(
                "routh_row1_positive",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(Kd),
                assumptions={"K_d": {"positive": True}},
                description="K_d is positive by axiom",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_routh_row2_positive(self, pd_axioms):
        """Row 2 coefficient = K_p / 2 > 0."""
        h = pd_axioms.hypothesis("routh_row2", expr=Kp / 2 > 0)
        script = (
            ProofBuilder(
                pd_axioms,
                h.name,
                name="routh_row2_proof",
                claim="Routh row 2: K_p / 2 > 0",
            )
            .lemma(
                "routh_row2_positive",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(Kp / 2),
                assumptions={"K_p": {"positive": True}},
                description="K_p/2 positive since K_p positive",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_sealed_routh_hurwitz(self, pd_axioms):
        """Sealed proof: all three Routh rows positive."""
        h = pd_axioms.hypothesis(
            "closed_loop_stable",
            expr=sympy.And(Jxx > 0, Kd > 0, Kp / 2 > 0),
        )
        script = (
            ProofBuilder(
                pd_axioms,
                h.name,
                name="routh_hurwitz_full",
                claim="All Routh rows positive => asymptotically stable",
            )
            .lemma(
                "row0",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(Jxx),
                assumptions={"J_xx": {"positive": True}},
            )
            .lemma(
                "row1",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(Kd),
                assumptions={"K_d": {"positive": True}},
                depends_on=["row0"],
            )
            .lemma(
                "row2",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(Kp / 2),
                assumptions={"K_p": {"positive": True}},
                depends_on=["row1"],
            )
            .build()
        )
        bundle = seal(pd_axioms, h, script)
        assert len(bundle.bundle_hash) == 64


# ===================================================================
# Steady-state pointing error
# ===================================================================


class TestPointingError:
    """Steady-state error theta_ss = 2 * tau_gg / Kp."""

    @pytest.fixture
    def error_axioms(self):
        tau_gg = sympy.Symbol("tau_gg", positive=True)
        return AxiomSet(
            name="pointing_error",
            axioms=(
                Axiom(name="gain_positive", expr=Kp > 0),
                Axiom(name="disturbance_positive", expr=tau_gg > 0),
            ),
        )

    def test_error_formula(self, error_axioms):
        """theta_ss = 2 * tau_gg / Kp is the steady-state error."""
        tau_gg = sympy.Symbol("tau_gg", positive=True)
        theta_ss = 2 * tau_gg / Kp

        h = error_axioms.hypothesis(
            "error_formula",
            expr=sympy.Eq(theta_ss, 2 * tau_gg / Kp),
        )
        script = (
            ProofBuilder(
                error_axioms,
                h.name,
                name="error_formula_proof",
                claim="Steady-state error = 2*tau_gg / Kp",
            )
            .lemma(
                "theta_ss_form",
                LemmaKind.EQUALITY,
                expr=theta_ss,
                expected=2 * tau_gg / Kp,
                description="Direct computation",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_error_decreases_with_gain(self, error_axioms):
        """d(theta_ss)/d(Kp) < 0: more gain => less error."""
        tau_gg = sympy.Symbol("tau_gg", positive=True)
        theta_ss = 2 * tau_gg / Kp
        d_theta = sympy.diff(theta_ss, Kp)

        h = error_axioms.hypothesis(
            "error_decreases",
            expr=d_theta < 0,
        )
        script = (
            ProofBuilder(
                error_axioms,
                h.name,
                name="error_sensitivity_proof",
                claim="Error decreases as proportional gain increases",
            )
            .lemma(
                "derivative_form",
                LemmaKind.EQUALITY,
                expr=d_theta,
                expected=-2 * tau_gg / Kp**2,
                description="d/dKp(2*tau/Kp) = -2*tau/Kp^2",
            )
            .lemma(
                "derivative_negative",
                LemmaKind.QUERY,
                expr=sympy.Q.negative(d_theta),
                assumptions={"tau_gg": {"positive": True}, "K_p": {"positive": True}},
                depends_on=["derivative_form"],
                description="-2*tau/Kp^2 < 0 since tau, Kp > 0",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary


# ===================================================================
# Gravity gradient torque
# ===================================================================


class TestGravityGradient:
    """Gravity gradient disturbance torque properties."""

    @pytest.fixture
    def gg_axioms(self):
        delta_J = sympy.Symbol("Delta_J", positive=True)
        theta_max = sympy.Rational(1, 10) * sympy.pi / 180
        return AxiomSet(
            name="gravity_gradient",
            axioms=(
                Axiom(name="orbital_rate_positive", expr=n > 0),
                Axiom(name="inertia_asymmetry_positive", expr=delta_J > 0),
            ),
        )

    def test_torque_scales_with_asymmetry(self, gg_axioms):
        """d(tau_gg)/d(Delta_J) > 0: more asymmetry => more torque."""
        delta_J = sympy.Symbol("Delta_J", positive=True)
        theta_max = sympy.Rational(1, 10) * sympy.pi / 180
        tau_gg = 3 * n**2 * delta_J * theta_max

        d_tau = sympy.diff(tau_gg, delta_J)

        h = gg_axioms.hypothesis(
            "torque_scales",
            expr=d_tau > 0,
        )
        script = (
            ProofBuilder(
                gg_axioms,
                h.name,
                name="gg_scaling_proof",
                claim="GG torque increases with inertia asymmetry",
            )
            .lemma(
                "derivative_positive",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(d_tau),
                assumptions={"n": {"positive": True}},
                description="3*n^2*theta_max > 0",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary


# ===================================================================
# Peak wheel momentum
# ===================================================================


class TestWheelMomentum:
    """Peak reaction wheel momentum: h_peak = Kd * theta_0 * omega_n."""

    @pytest.fixture
    def wheel_axioms(self):
        theta_0 = sympy.Symbol("theta_0", positive=True)
        return AxiomSet(
            name="wheel_momentum",
            axioms=(
                Axiom(name="inertia_positive", expr=Jxx > 0),
                Axiom(name="gains_positive", expr=sympy.And(Kp > 0, Kd > 0)),
                Axiom(name="initial_error_positive", expr=theta_0 > 0),
            ),
        )

    def test_momentum_scales_with_kd(self, wheel_axioms):
        """d(h_peak)/d(Kd) > 0: more derivative gain => more peak momentum."""
        theta_0 = sympy.Symbol("theta_0", positive=True)
        omega_n = sympy.sqrt(Kp / (2 * Jxx))
        h_peak = Kd * theta_0 * omega_n

        d_h = sympy.diff(h_peak, Kd)

        h = wheel_axioms.hypothesis("momentum_scales_kd", expr=d_h > 0)
        script = (
            ProofBuilder(
                wheel_axioms,
                h.name,
                name="momentum_kd_proof",
                claim="Peak momentum increases with Kd",
            )
            .lemma(
                "d_h_positive",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(d_h),
                assumptions={
                    "theta_0": {"positive": True},
                    "K_p": {"positive": True},
                    "J_xx": {"positive": True},
                },
                description="theta_0 * sqrt(Kp/(2*Jxx)) > 0",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary


# ===================================================================
# Inertia tensor composition
# ===================================================================


class TestInertiaTensor:
    """Parallel axis theorem and composite inertia."""

    def test_parallel_axis_shift(self):
        """Parallel axis: I_shifted = I_cm + m*d^2."""
        m = sympy.Symbol("m", positive=True)
        d = sympy.Symbol("d", real=True)
        I_cm = sympy.Symbol("I_cm", positive=True)

        I_shifted = I_cm + m * d**2

        axioms = AxiomSet(
            name="parallel_axis",
            axioms=(
                Axiom(name="mass_positive", expr=m > 0),
                Axiom(name="base_inertia_positive", expr=I_cm > 0),
            ),
        )
        h = axioms.hypothesis(
            "shifted_ge_cm",
            expr=sympy.Ge(I_shifted, I_cm),
        )
        script = (
            ProofBuilder(
                axioms,
                h.name,
                name="parallel_axis_proof",
                claim="Shifted inertia >= CM inertia",
            )
            .lemma(
                "shift_nonneg",
                LemmaKind.QUERY,
                expr=sympy.Q.nonnegative(m * d**2),
                assumptions={"m": {"positive": True}, "d": {"real": True}},
                description="m*d^2 >= 0 since m > 0 and d^2 >= 0",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary

    def test_box_inertia_symmetry(self):
        """Box inertia: Ixx = m*(Ly^2+Lz^2)/12 is symmetric in Ly, Lz."""
        m = sympy.Symbol("m", positive=True)
        Ly = sympy.Symbol("L_y", positive=True)
        Lz = sympy.Symbol("L_z", positive=True)

        Ixx_orig = m * (Ly**2 + Lz**2) / 12
        Ixx_swapped = m * (Lz**2 + Ly**2) / 12

        axioms = AxiomSet(
            name="box_inertia",
            axioms=(Axiom(name="mass_positive", expr=m > 0),),
        )
        h = axioms.hypothesis(
            "inertia_symmetric",
            expr=sympy.Eq(Ixx_orig, Ixx_swapped),
        )
        script = (
            ProofBuilder(
                axioms,
                h.name,
                name="symmetry_proof",
                claim="Box Ixx symmetric in Ly and Lz",
            )
            .lemma(
                "commutativity",
                LemmaKind.EQUALITY,
                expr=Ixx_orig,
                expected=Ixx_swapped,
                description="Ly^2 + Lz^2 = Lz^2 + Ly^2",
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status.value == "VERIFIED", result.failure_summary
