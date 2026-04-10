"""Tests for the physics domain library (AP Physics C level)."""

from __future__ import annotations

import sympy

from symproof import Axiom, AxiomSet, ProofStatus
from symproof.library.physics import (
    constant_acceleration,
    gravitational_potential_from_force,
    impulse_momentum,
    rotational_kinematic,
    shm_energy_conservation,
    shm_solution_verify,
    work_energy_theorem,
)

t = sympy.Symbol("t")


def _dummy_axioms(name="test"):
    return AxiomSet(name=name, axioms=(
        Axiom(name="defined", expr=sympy.Eq(1, 1)),
    ))


class TestConstantAcceleration:
    def test_seals(self):
        v0, a = sympy.symbols("v0 a")
        bundle = constant_acceleration(_dummy_axioms(), x0=0, v0=v0, a=a, t=t)
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_with_numeric_params(self):
        bundle = constant_acceleration(
            _dummy_axioms(), x0=0, v0=sympy.Integer(10), a=sympy.Rational(-10, 1), t=t,
        )
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_nonzero_initial_position(self):
        v0, a, x0 = sympy.symbols("v0 a x0")
        bundle = constant_acceleration(_dummy_axioms(), x0=x0, v0=v0, a=a, t=t)
        assert bundle.proof_result.status == ProofStatus.VERIFIED


class TestRotationalKinematic:
    def test_seals(self):
        theta0, omega0, alpha = sympy.symbols("theta0 omega0 alpha")
        bundle = rotational_kinematic(_dummy_axioms(), theta0, omega0, alpha, t)
        assert bundle.proof_result.status == ProofStatus.VERIFIED


class TestSHMSolutionVerify:
    def test_seals(self):
        A, omega, phi = sympy.symbols("A omega phi")
        bundle = shm_solution_verify(_dummy_axioms(), A, omega, phi, t)
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_two_lemmas(self):
        A, omega, phi = sympy.symbols("A omega phi")
        bundle = shm_solution_verify(_dummy_axioms(), A, omega, phi, t)
        assert len(bundle.proof.lemmas) == 2


class TestSHMEnergyConservation:
    def test_seals(self):
        m = sympy.Symbol("m", positive=True)
        k = sympy.Symbol("k", positive=True)
        A, omega, phi = sympy.symbols("A omega phi")
        bundle = shm_energy_conservation(_dummy_axioms(), m, k, A, omega, phi, t)
        assert bundle.proof_result.status == ProofStatus.VERIFIED


class TestWorkEnergyTheorem:
    def test_seals(self):
        F, m, v0, a = sympy.symbols("F m v0 a")
        bundle = work_energy_theorem(_dummy_axioms(), F, m, v0, a, t)
        assert bundle.proof_result.status == ProofStatus.VERIFIED

    def test_from_rest(self):
        F, m, a = sympy.symbols("F m a")
        bundle = work_energy_theorem(_dummy_axioms(), F, m, v0=0, a=a, t=t)
        assert bundle.proof_result.status == ProofStatus.VERIFIED


class TestImpulseMomentum:
    def test_seals(self):
        F, m, v0, a = sympy.symbols("F m v0 a")
        bundle = impulse_momentum(_dummy_axioms(), F, m, v0, a, t)
        assert bundle.proof_result.status == ProofStatus.VERIFIED


class TestGravitationalPotential:
    def test_seals(self):
        G, M, m_g = sympy.symbols("G M m_g", positive=True)
        r = sympy.Symbol("r", positive=True)
        bundle = gravitational_potential_from_force(_dummy_axioms(), G, M, m_g, r)
        assert bundle.proof_result.status == ProofStatus.VERIFIED
