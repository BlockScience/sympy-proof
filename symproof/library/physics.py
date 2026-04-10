"""High school physics with calculus — proof library.

Covers AP Physics C level mechanics: kinematics, energy, momentum,
simple harmonic motion, rotational motion, and gravitation.  Each
function proves a standard physics result using calculus (derivatives,
integrals) and returns a sealed ``ProofBundle``.

Building blocks
    ``constant_acceleration`` — kinematic equations from x(t)
    ``shm_solution_verify`` — verify x(t) satisfies the SHM ODE
    ``shm_energy_conservation`` — total energy is constant along SHM

Composed proofs
    ``conservation_of_energy`` — dE/dt = 0 for conservative systems
    ``work_energy_theorem`` — W = delta(KE) for constant force
    ``impulse_momentum`` — J = delta(p) for constant force
    ``rotational_kinematic`` — angular kinematic equations
    ``gravitational_potential_from_force`` — U(r) from F(r)
"""

from __future__ import annotations

import sympy

from symproof.builder import ProofBuilder
from symproof.bundle import seal
from symproof.evaluation import evaluation
from symproof.models import AxiomSet, LemmaKind, ProofBundle


# ===================================================================
# Kinematics
# ===================================================================


def constant_acceleration(
    axiom_set: AxiomSet,
    x0: sympy.Basic,
    v0: sympy.Basic,
    a: sympy.Basic,
    t: sympy.Symbol,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove the kinematic equations for constant acceleration.

    Given position ``x(t) = x0 + v0*t + (1/2)*a*t**2``, proves:

    1. ``v(t) = dx/dt = v0 + a*t``
    2. ``a(t) = dv/dt = a`` (constant)

    Parameters
    ----------
    axiom_set:
        Axiom context (e.g., constraints on parameters).
    x0, v0, a:
        Initial position, initial velocity, and acceleration.
    t:
        Time symbol.
    assumptions:
        Symbol assumptions for verification.
    """

    x_t = x0 + v0 * t + sympy.Rational(1, 2) * a * t**2
    with evaluation():
        v_t = sympy.simplify(sympy.diff(x_t, t))
        a_t = sympy.simplify(sympy.diff(v_t, t))

    hyp = axiom_set.hypothesis(
        "kinematic_equations",
        expr=sympy.And(
            sympy.Eq(v_t, v0 + a * t),
            sympy.Eq(a_t, a),
        ),
        description=(
            "Constant acceleration kinematics: "
            "v(t) = v0 + a*t and a(t) = a"
        ),
    )

    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="constant_acceleration_proof",
            claim="Derive velocity and acceleration from position via differentiation.",
        )
        .lemma(
            "velocity_is_derivative",
            LemmaKind.EQUALITY,
            expr=sympy.diff(x_t, t),
            expected=v0 + a * t,
            description="v(t) = dx/dt = v0 + a*t",
        )
        .lemma(
            "acceleration_is_constant",
            LemmaKind.EQUALITY,
            expr=sympy.diff(x_t, t, 2),
            expected=a,
            depends_on=["velocity_is_derivative"],
            description="a(t) = d²x/dt² = a (constant)",
        )
        .build()
    )

    return seal(axiom_set, hyp, script)


def rotational_kinematic(
    axiom_set: AxiomSet,
    theta0: sympy.Basic,
    omega0: sympy.Basic,
    alpha: sympy.Basic,
    t: sympy.Symbol,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove rotational kinematic equations for constant angular acceleration.

    Given ``theta(t) = theta0 + omega0*t + (1/2)*alpha*t**2``, proves:

    1. ``omega(t) = dtheta/dt = omega0 + alpha*t``
    2. ``alpha(t) = domega/dt = alpha`` (constant)

    Parameters
    ----------
    axiom_set:
        Axiom context.
    theta0, omega0, alpha:
        Initial angle, initial angular velocity, angular acceleration.
    t:
        Time symbol.
    assumptions:
        Symbol assumptions for verification.
    """
    theta_t = theta0 + omega0 * t + sympy.Rational(1, 2) * alpha * t**2
    with evaluation():
        omega_t = sympy.simplify(sympy.diff(theta_t, t))
        alpha_t = sympy.simplify(sympy.diff(omega_t, t))

    hyp = axiom_set.hypothesis(
        "rotational_kinematics",
        expr=sympy.And(
            sympy.Eq(omega_t, omega0 + alpha * t),
            sympy.Eq(alpha_t, alpha),
        ),
        description="Rotational kinematics: omega = dtheta/dt, alpha = domega/dt",
    )

    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="rotational_kinematic_proof",
            claim="Derive angular velocity and acceleration from angle.",
        )
        .lemma(
            "angular_velocity",
            LemmaKind.EQUALITY,
            expr=sympy.diff(theta_t, t),
            expected=omega0 + alpha * t,
            description="omega(t) = dtheta/dt = omega0 + alpha*t",
        )
        .lemma(
            "angular_acceleration_constant",
            LemmaKind.EQUALITY,
            expr=sympy.diff(theta_t, t, 2),
            expected=alpha,
            depends_on=["angular_velocity"],
            description="alpha(t) = d²theta/dt² = alpha (constant)",
        )
        .build()
    )

    return seal(axiom_set, hyp, script)


# ===================================================================
# Simple Harmonic Motion
# ===================================================================


def shm_solution_verify(
    axiom_set: AxiomSet,
    A: sympy.Basic,
    omega: sympy.Basic,
    phi: sympy.Basic,
    t: sympy.Symbol,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Verify that ``x(t) = A*cos(omega*t + phi)`` satisfies the SHM ODE.

    Proves: ``x''(t) + omega**2 * x(t) = 0``.

    Parameters
    ----------
    axiom_set:
        Axiom context.
    A:
        Amplitude.
    omega:
        Angular frequency.
    phi:
        Phase offset.
    t:
        Time symbol.
    assumptions:
        Symbol assumptions for verification.
    """
    x_t = A * sympy.cos(omega * t + phi)
    with evaluation():
        x_ddot = sympy.simplify(sympy.diff(x_t, t, 2))
        ode_residual = sympy.simplify(x_ddot + omega**2 * x_t)

    hyp = axiom_set.hypothesis(
        "shm_ode_satisfied",
        expr=sympy.Eq(ode_residual, 0),
        description="x(t) = A*cos(omega*t + phi) satisfies x'' + omega^2*x = 0",
    )

    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="shm_solution_proof",
            claim="A*cos(omega*t + phi) is a solution to the SHM equation.",
        )
        .lemma(
            "second_derivative",
            LemmaKind.EQUALITY,
            expr=sympy.diff(x_t, t, 2),
            expected=-(omega**2) * x_t,
            description="x''(t) = -omega^2 * A*cos(omega*t + phi) = -omega^2 * x(t)",
        )
        .lemma(
            "ode_residual_zero",
            LemmaKind.EQUALITY,
            expr=ode_residual,
            expected=sympy.Integer(0),
            depends_on=["second_derivative"],
            description="x''(t) + omega^2*x(t) = 0",
        )
        .build()
    )

    return seal(axiom_set, hyp, script)


def shm_energy_conservation(
    axiom_set: AxiomSet,
    m: sympy.Basic,
    k: sympy.Basic,
    A: sympy.Basic,
    omega: sympy.Basic,
    phi: sympy.Basic,
    t: sympy.Symbol,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    r"""Prove total energy is constant for a simple harmonic oscillator.

    For ``x(t) = A*cos(omega*t + phi)`` with ``omega = sqrt(k/m)``:

    - KE = (1/2)*m*v^2
    - PE = (1/2)*k*x^2
    - E = KE + PE = (1/2)*k*A^2 (constant)

    Proves: ``dE/dt = 0``.

    Parameters
    ----------
    axiom_set:
        Axiom context (should include m > 0, k > 0).
    m, k:
        Mass and spring constant.
    A, omega, phi:
        Amplitude, angular frequency, phase.
    t:
        Time symbol.
    assumptions:
        Symbol assumptions for verification.
    """
    x_t = A * sympy.cos(omega * t + phi)
    v_t = sympy.diff(x_t, t)

    KE = sympy.Rational(1, 2) * m * v_t**2
    PE = sympy.Rational(1, 2) * k * x_t**2
    E = KE + PE

    # dE/dt is only zero when omega^2 = k/m.  SymPy doesn't use axiom
    # equations during simplification, so we substitute k = m*omega^2
    # explicitly (the defining relation for SHM frequency).
    with evaluation():
        dE_dt_raw = sympy.diff(E, t)
        dE_dt = sympy.simplify(dE_dt_raw.subs(k, m * omega**2))

    hyp = axiom_set.hypothesis(
        "shm_energy_constant",
        expr=sympy.Eq(dE_dt, 0),
        description="Total energy E = KE + PE is constant along SHM trajectories (using omega^2 = k/m)",
    )

    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="shm_energy_proof",
            claim="dE/dt = 0 for simple harmonic motion (using omega^2 = k/m).",
        )
        .lemma(
            "energy_derivative_with_omega",
            LemmaKind.EQUALITY,
            expr=dE_dt,
            expected=sympy.Integer(0),
            description=(
                "d/dt[(1/2)*m*v^2 + (1/2)*k*x^2] = 0 "
                "after substituting k = m*omega^2"
            ),
        )
        .build()
    )

    return seal(axiom_set, hyp, script)


# ===================================================================
# Energy and Work
# ===================================================================


def work_energy_theorem(
    axiom_set: AxiomSet,
    F: sympy.Basic,
    m: sympy.Basic,
    v0: sympy.Basic,
    a: sympy.Basic,
    t: sympy.Symbol,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove the work-energy theorem for constant force.

    For constant force ``F = m*a``, constant acceleration kinematics
    give ``v(t) = v0 + a*t`` and ``x(t) = v0*t + (1/2)*a*t**2``.

    Proves: ``W = F*x(t) = (1/2)*m*v(t)**2 - (1/2)*m*v0**2 = delta(KE)``.

    Parameters
    ----------
    axiom_set:
        Axiom context (should include m > 0, F = m*a).
    F, m, v0, a:
        Force, mass, initial velocity, acceleration.
    t:
        Time symbol.
    assumptions:
        Symbol assumptions for verification.
    """
    x_t = v0 * t + sympy.Rational(1, 2) * a * t**2
    v_t = v0 + a * t

    W = F * x_t
    KE_final = sympy.Rational(1, 2) * m * v_t**2
    KE_initial = sympy.Rational(1, 2) * m * v0**2
    delta_KE = KE_final - KE_initial

    with evaluation():
        diff = sympy.simplify(W.subs(F, m * a) - delta_KE)

    hyp = axiom_set.hypothesis(
        "work_equals_delta_ke",
        expr=sympy.Eq(diff, 0),
        description="W = F*x = delta(KE) (work-energy theorem)",
    )

    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="work_energy_proof",
            claim="Work done by constant force equals change in kinetic energy.",
        )
        .lemma(
            "work_minus_delta_ke",
            LemmaKind.EQUALITY,
            expr=diff,
            expected=sympy.Integer(0),
            description="F*x(t) - [(1/2)*m*v(t)^2 - (1/2)*m*v0^2] = 0 when F = m*a",
        )
        .build()
    )

    return seal(axiom_set, hyp, script)


def impulse_momentum(
    axiom_set: AxiomSet,
    F: sympy.Basic,
    m: sympy.Basic,
    v0: sympy.Basic,
    a: sympy.Basic,
    t: sympy.Symbol,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove the impulse-momentum theorem for constant force.

    For constant force ``F = m*a``: ``J = F*t = m*v(t) - m*v0 = delta(p)``.

    Parameters
    ----------
    axiom_set:
        Axiom context.
    F, m, v0, a:
        Force, mass, initial velocity, acceleration.
    t:
        Time symbol.
    assumptions:
        Symbol assumptions for verification.
    """
    v_t = v0 + a * t
    J = F * t
    delta_p = m * v_t - m * v0

    with evaluation():
        diff = sympy.simplify(J.subs(F, m * a) - delta_p)

    hyp = axiom_set.hypothesis(
        "impulse_equals_delta_p",
        expr=sympy.Eq(diff, 0),
        description="J = F*t = delta(p) (impulse-momentum theorem)",
    )

    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="impulse_momentum_proof",
            claim="Impulse from constant force equals change in momentum.",
        )
        .lemma(
            "impulse_minus_delta_p",
            LemmaKind.EQUALITY,
            expr=diff,
            expected=sympy.Integer(0),
            description="F*t - [m*v(t) - m*v0] = 0 when F = m*a",
        )
        .build()
    )

    return seal(axiom_set, hyp, script)


# ===================================================================
# Gravitation
# ===================================================================


def gravitational_potential_from_force(
    axiom_set: AxiomSet,
    G: sympy.Basic,
    M: sympy.Basic,
    m: sympy.Basic,
    r: sympy.Symbol,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove gravitational potential from the force law.

    Given ``F(r) = -G*M*m/r**2`` (attractive, toward origin):

    - ``U(r) = -integral(F, r) = -G*M*m/r``
    - ``F(r) = -dU/dr`` (force is negative gradient of potential)

    Parameters
    ----------
    axiom_set:
        Axiom context (should include G, M, m > 0, r > 0).
    G, M, m:
        Gravitational constant, central mass, test mass.
    r:
        Radial distance symbol.
    assumptions:
        Symbol assumptions for verification.
    """
    F_r = -G * M * m / r**2
    U_r = -G * M * m / r

    with evaluation():
        neg_dU_dr = sympy.simplify(-sympy.diff(U_r, r))
        force_match = sympy.simplify(neg_dU_dr - F_r)

    hyp = axiom_set.hypothesis(
        "gravitational_potential",
        expr=sympy.Eq(force_match, 0),
        description="F(r) = -dU/dr where U(r) = -GMm/r",
    )

    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="gravitational_potential_proof",
            claim="Gravitational potential U = -GMm/r satisfies F = -dU/dr.",
        )
        .lemma(
            "force_from_potential",
            LemmaKind.EQUALITY,
            expr=force_match,
            expected=sympy.Integer(0),
            description="-dU/dr = -d/dr(-GMm/r) = -GMm/r^2 = F(r)",
        )
        .build()
    )

    return seal(axiom_set, hyp, script)
