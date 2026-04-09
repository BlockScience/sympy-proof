"""Tests for symproof.library.control — control systems proof library.

Exercises: Hurwitz stability (2nd/3rd order), Lyapunov equation,
controllability, observability, quadratic invariants.
"""

from __future__ import annotations

import sympy

from symproof import (
    Axiom,
    AxiomSet,
    LemmaKind,
    ProofBuilder,
    seal,
    verify_proof,
)
from symproof.library.control import (
    controllability_rank,
    hurwitz_second_order,
    hurwitz_third_order,
    lyapunov_stability,
    observability_rank,
    quadratic_invariant,
)
from symproof.models import ProofStatus


# ===================================================================
# Hurwitz stability
# ===================================================================


class TestHurwitzSecondOrder:
    """Second-order Routh-Hurwitz: a2*s^2 + a1*s + a0 Hurwitz iff all > 0."""

    def test_pd_controller(self):
        """PD-controlled rigid body: J*s^2 + Kd*s + Kp/2."""
        J = sympy.Symbol("J", positive=True)
        Kd = sympy.Symbol("Kd", positive=True)
        Kp = sympy.Symbol("Kp", positive=True)

        axioms = AxiomSet(
            name="pd_system",
            axioms=(
                Axiom(name="J_pos", expr=J > 0),
                Axiom(name="Kd_pos", expr=Kd > 0),
                Axiom(name="Kp_pos", expr=Kp > 0),
            ),
        )
        bundle = hurwitz_second_order(
            axioms, a2=J, a1=Kd, a0=Kp / 2,
            assumptions={
                "J": {"positive": True},
                "Kd": {"positive": True},
                "Kp": {"positive": True},
            },
        )
        assert len(bundle.bundle_hash) == 64
        assert len(bundle.proof.lemmas) == 3

    def test_mass_spring_damper(self):
        """m*s^2 + c*s + k — classic MSD system."""
        m = sympy.Symbol("m", positive=True)
        c = sympy.Symbol("c", positive=True)
        k = sympy.Symbol("k", positive=True)

        axioms = AxiomSet(
            name="msd",
            axioms=(
                Axiom(name="m_pos", expr=m > 0),
                Axiom(name="c_pos", expr=c > 0),
                Axiom(name="k_pos", expr=k > 0),
            ),
        )
        bundle = hurwitz_second_order(
            axioms, a2=m, a1=c, a0=k,
            assumptions={
                "m": {"positive": True},
                "c": {"positive": True},
                "k": {"positive": True},
            },
        )
        assert bundle.bundle_hash

    def test_importable(self):
        """Can import Hurwitz proof into a larger proof."""
        J = sympy.Symbol("J", positive=True)
        Kd = sympy.Symbol("Kd", positive=True)
        Kp = sympy.Symbol("Kp", positive=True)

        axioms = AxiomSet(
            name="pd",
            axioms=(
                Axiom(name="J_pos", expr=J > 0),
                Axiom(name="Kd_pos", expr=Kd > 0),
                Axiom(name="Kp_pos", expr=Kp > 0),
            ),
        )
        stability = hurwitz_second_order(
            axioms, J, Kd, Kp / 2,
            assumptions={
                "J": {"positive": True},
                "Kd": {"positive": True},
                "Kp": {"positive": True},
            },
        )
        h = axioms.hypothesis("safe", expr=J > 0)
        script = (
            ProofBuilder(axioms, h.name, name="safe", claim="safe")
            .import_bundle(stability)
            .lemma(
                "j",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(J),
                assumptions={"J": {"positive": True}},
            )
            .build()
        )
        bundle = seal(axioms, h, script)
        assert bundle.bundle_hash


class TestHurwitzThirdOrder:
    """Third-order Routh-Hurwitz with cross-term condition."""

    def test_third_order_concrete(self):
        """Concrete numeric coefficients: s^3 + 3s^2 + 3s + 1."""
        # (s+1)^3 — all roots at -1, clearly stable
        # Routh: a3=1>0, a2=3>0, cross=3*3-1*1=8>0, a0=1>0
        axioms = AxiomSet(
            name="third_order",
            axioms=(Axiom(name="sys", expr=sympy.Eq(1, 1)),),
        )
        bundle = hurwitz_third_order(
            axioms,
            a3=sympy.Integer(1),
            a2=sympy.Integer(3),
            a1=sympy.Integer(3),
            a0=sympy.Integer(1),
        )
        assert len(bundle.proof.lemmas) == 4

    def test_third_order_another_concrete(self):
        """s^3 + 6s^2 + 11s + 6 = (s+1)(s+2)(s+3) — all roots negative."""
        # Routh: a3=1, a2=6, cross=11*6-6*1=60>0, a0=6
        axioms = AxiomSet(
            name="third_order_2",
            axioms=(Axiom(name="sys", expr=sympy.Eq(1, 1)),),
        )
        bundle = hurwitz_third_order(
            axioms,
            a3=sympy.Integer(1),
            a2=sympy.Integer(6),
            a1=sympy.Integer(11),
            a0=sympy.Integer(6),
        )
        assert len(bundle.proof.lemmas) == 4


# ===================================================================
# Lyapunov stability
# ===================================================================


class TestLyapunovStability:
    """Lyapunov equation A^T P + P A + Q = 0."""

    def test_scalar_system(self):
        """1x1 system: A = [-a], P = [1], Q = [2a]."""
        a = sympy.Symbol("a", positive=True)
        A = sympy.Matrix([[-a]])
        P = sympy.Matrix([[1]])
        Q = sympy.Matrix([[2 * a]])

        axioms = AxiomSet(
            name="scalar_lyapunov",
            axioms=(Axiom(name="a_pos", expr=a > 0),),
        )
        bundle = lyapunov_stability(axioms, A, P, Q)
        assert bundle.bundle_hash

    def test_2x2_damped_oscillator(self):
        """2x2 damped oscillator: A = [[0,1],[-k,-c]]."""
        k = sympy.Symbol("k", positive=True)
        c = sympy.Symbol("c", positive=True)
        A = sympy.Matrix([[0, 1], [-k, -c]])
        # P = [[k + c^2/2, c/2], [c/2, 1]]  (solves Lyapunov for Q = I)
        P = sympy.Matrix([
            [k + c**2 / 2, c / 2],
            [c / 2, 1],
        ])
        Q = -(A.T * P + P * A)
        # Q should simplify to identity-like
        Q_simplified = sympy.simplify(Q)

        axioms = AxiomSet(
            name="oscillator",
            axioms=(
                Axiom(name="k_pos", expr=k > 0),
                Axiom(name="c_pos", expr=c > 0),
            ),
        )
        bundle = lyapunov_stability(axioms, A, P, Q_simplified)
        assert bundle.bundle_hash
        # Should have 4 lemmas (2x2 matrix, one per entry)
        assert len(bundle.proof.lemmas) == 4


# ===================================================================
# Controllability
# ===================================================================


class TestControllabilityRank:
    """Controllability via Gramian determinant."""

    def test_double_integrator(self):
        """Double integrator [[0,1],[0,0]] with B = [[0],[1]] is controllable."""
        A = sympy.Matrix([[0, 1], [0, 0]])
        B = sympy.Matrix([[0], [1]])

        axioms = AxiomSet(
            name="double_integrator",
            axioms=(Axiom(name="system_defined", expr=sympy.Eq(1, 1)),),
        )
        bundle = controllability_rank(axioms, A, B)
        assert bundle.bundle_hash

    def test_single_input_single_state(self):
        """Scalar system A=[-1], B=[1] is trivially controllable."""
        A = sympy.Matrix([[-1]])
        B = sympy.Matrix([[1]])

        axioms = AxiomSet(
            name="scalar",
            axioms=(Axiom(name="sys", expr=sympy.Eq(1, 1)),),
        )
        bundle = controllability_rank(axioms, A, B)
        assert bundle.bundle_hash


# ===================================================================
# Observability
# ===================================================================


class TestObservabilityRank:
    """Observability via Gramian determinant."""

    def test_position_measurement(self):
        """Double integrator with position measurement is observable."""
        A = sympy.Matrix([[0, 1], [0, 0]])
        C = sympy.Matrix([[1, 0]])

        axioms = AxiomSet(
            name="pos_measurement",
            axioms=(Axiom(name="system_defined", expr=sympy.Eq(1, 1)),),
        )
        bundle = observability_rank(axioms, A, C)
        assert bundle.bundle_hash

    def test_full_state_measurement(self):
        """Double integrator with full-state measurement is observable."""
        A = sympy.Matrix([[0, 1], [0, 0]])
        C = sympy.Matrix([[1, 0], [0, 1]])  # Full state

        axioms = AxiomSet(
            name="full_measurement",
            axioms=(Axiom(name="system_defined", expr=sympy.Eq(1, 1)),),
        )
        bundle = observability_rank(axioms, A, C)
        assert bundle.bundle_hash


# ===================================================================
# Quadratic invariant (conservation law)
# ===================================================================


class TestQuadraticInvariant:
    """dV/dt = 0 along ODE trajectories."""

    def test_costate_norm_conservation(self):
        """||p||^2 conserved under dp1/dt=-p2*phi, dp2/dt=p1*phi."""
        p1 = sympy.Symbol("p_1", real=True)
        p2 = sympy.Symbol("p_2", real=True)
        phi = sympy.Symbol("phi", real=True)

        V = p1**2 + p2**2
        p1_dot = -p2 * phi
        p2_dot = p1 * phi

        axioms = AxiomSet(
            name="costate_dynamics",
            axioms=(
                Axiom(
                    name="dynamics",
                    expr=sympy.Eq(1, 1),
                ),
            ),
        )
        bundle = quadratic_invariant(
            axioms,
            state_symbols=[p1, p2],
            state_dots=[p1_dot, p2_dot],
            V=V,
        )
        assert bundle.bundle_hash

    def test_harmonic_oscillator_energy(self):
        """E = x^2 + v^2 conserved under dx/dt=v, dv/dt=-x."""
        x = sympy.Symbol("x", real=True)
        v = sympy.Symbol("v", real=True)

        V = x**2 + v**2

        axioms = AxiomSet(
            name="harmonic",
            axioms=(Axiom(name="system", expr=sympy.Eq(1, 1)),),
        )
        bundle = quadratic_invariant(
            axioms,
            state_symbols=[x, v],
            state_dots=[v, -x],
            V=V,
        )
        assert bundle.bundle_hash

    def test_importable_into_proof(self):
        """Conservation proof can be imported into a larger proof."""
        p1 = sympy.Symbol("p_1", real=True)
        p2 = sympy.Symbol("p_2", real=True)
        phi = sympy.Symbol("phi", real=True)

        axioms = AxiomSet(
            name="costate",
            axioms=(Axiom(name="sys", expr=sympy.Eq(1, 1)),),
        )
        conservation = quadratic_invariant(
            axioms,
            [p1, p2],
            [-p2 * phi, p1 * phi],
            p1**2 + p2**2,
        )
        h = axioms.hypothesis("test", expr=sympy.Eq(1, 1))
        script = (
            ProofBuilder(axioms, h.name, name="t", claim="t")
            .import_bundle(conservation)
            .lemma(
                "trivial",
                LemmaKind.BOOLEAN,
                expr=sympy.Eq(1, 1),
            )
            .build()
        )
        result = verify_proof(script)
        assert result.status == ProofStatus.VERIFIED


# ===================================================================
# Cross-domain composition
# ===================================================================


class TestCrossDomainComposition:
    """Compose stability + controllability in one proof."""

    def test_stable_and_controllable(self):
        """Import Hurwitz stability + controllability for double integrator."""
        m = sympy.Symbol("m", positive=True)
        c = sympy.Symbol("c", positive=True)
        k = sympy.Symbol("k", positive=True)

        # Mass-spring-damper: m*x'' + c*x' + k*x = u
        A = sympy.Matrix([[0, 1], [-k / m, -c / m]])
        B = sympy.Matrix([[0], [1 / m]])

        axioms = AxiomSet(
            name="msd_ctrl",
            axioms=(
                Axiom(name="m_pos", expr=m > 0),
                Axiom(name="c_pos", expr=c > 0),
                Axiom(name="k_pos", expr=k > 0),
            ),
        )

        stability = hurwitz_second_order(
            axioms, a2=m, a1=c, a0=k,
            assumptions={
                "m": {"positive": True},
                "c": {"positive": True},
                "k": {"positive": True},
            },
        )
        ctrl = controllability_rank(axioms, A, B)

        h = axioms.hypothesis("safe_ctrl", expr=m > 0)
        script = (
            ProofBuilder(axioms, h.name, name="combo", claim="combo")
            .import_bundle(stability)
            .import_bundle(ctrl)
            .lemma(
                "m_pos",
                LemmaKind.QUERY,
                expr=sympy.Q.positive(m),
                assumptions={"m": {"positive": True}},
            )
            .build()
        )
        bundle = seal(axioms, h, script)
        assert len(bundle.proof.imported_bundles) == 2
