"""Control systems proof library — stability, controllability, observability.

Reusable proof bundles for classical control theory results.
Each function accepts an ``AxiomSet`` (plus system matrices/symbols),
builds a decomposed proof, seals it, and returns a ``ProofBundle``.

Categories
----------
Stability
    Routh-Hurwitz (2nd/3rd order), Lyapunov equation
Controllability & observability
    Rank conditions via minor determinants
Conservation laws
    Quadratic invariants along trajectories (Lyapunov-like)
"""

from __future__ import annotations

import sympy

from symproof.builder import ProofBuilder
from symproof.bundle import seal
from symproof.models import AxiomSet, LemmaKind, ProofBundle

# ===================================================================
# Stability: Hurwitz (polynomial coefficient conditions)
# ===================================================================


def hurwitz_second_order(
    axiom_set: AxiomSet,
    a2: sympy.Basic,
    a1: sympy.Basic,
    a0: sympy.Basic,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove a second-order polynomial is Hurwitz (all roots Re < 0).

    For ``a2*s^2 + a1*s + a0``, the Routh-Hurwitz conditions are
    simply ``a2 > 0``, ``a1 > 0``, ``a0 > 0``.

    Parameters
    ----------
    axiom_set:
        Must contain axioms establishing positivity of all coefficients.
    a2, a1, a0:
        Polynomial coefficients (highest degree first).
    assumptions:
        Symbol assumptions for the QUERY lemmas.
    """
    asm = assumptions or {}
    hyp = axiom_set.hypothesis(
        "hurwitz_second_order",
        expr=sympy.And(a2 > 0, a1 > 0, a0 > 0),
        description="All Routh rows positive => stable 2nd-order system",
    )
    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="hurwitz_2nd_order_proof",
            claim="a2*s^2 + a1*s + a0 is Hurwitz",
        )
        .lemma(
            "routh_row0",
            LemmaKind.QUERY,
            expr=sympy.Q.positive(a2),
            assumptions=asm,
            description="Leading coefficient positive",
        )
        .lemma(
            "routh_row1",
            LemmaKind.QUERY,
            expr=sympy.Q.positive(a1),
            assumptions=asm,
            depends_on=["routh_row0"],
            description="Damping coefficient positive",
        )
        .lemma(
            "routh_row2",
            LemmaKind.QUERY,
            expr=sympy.Q.positive(a0),
            assumptions=asm,
            depends_on=["routh_row1"],
            description="Stiffness coefficient positive",
        )
        .build()
    )
    return seal(axiom_set, hyp, script)


def hurwitz_third_order(
    axiom_set: AxiomSet,
    a3: sympy.Basic,
    a2: sympy.Basic,
    a1: sympy.Basic,
    a0: sympy.Basic,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove a third-order polynomial is Hurwitz.

    For ``a3*s^3 + a2*s^2 + a1*s + a0``, the Routh conditions are:
    ``a3 > 0``, ``a2 > 0``, ``a1*a2 - a0*a3 > 0``, ``a0 > 0``.

    Parameters
    ----------
    axiom_set:
        Must contain axioms establishing the Routh conditions.
    a3, a2, a1, a0:
        Polynomial coefficients (highest degree first).
    assumptions:
        Symbol assumptions for the QUERY lemmas.
    """
    asm = assumptions or {}
    routh_cross = a1 * a2 - a0 * a3

    hyp = axiom_set.hypothesis(
        "hurwitz_third_order",
        expr=sympy.And(a3 > 0, a2 > 0, routh_cross > 0, a0 > 0),
        description="All Routh rows positive => stable 3rd-order system",
    )
    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="hurwitz_3rd_order_proof",
            claim="a3*s^3 + a2*s^2 + a1*s + a0 is Hurwitz",
        )
        .lemma(
            "routh_row0",
            LemmaKind.QUERY,
            expr=sympy.Q.positive(a3),
            assumptions=asm,
            description="Leading coefficient positive",
        )
        .lemma(
            "routh_row1",
            LemmaKind.QUERY,
            expr=sympy.Q.positive(a2),
            assumptions=asm,
            depends_on=["routh_row0"],
            description="Second coefficient positive",
        )
        .lemma(
            "routh_cross_term",
            LemmaKind.QUERY,
            expr=sympy.Q.positive(routh_cross),
            assumptions=asm,
            depends_on=["routh_row1"],
            description="Routh cross-term a1*a2 - a0*a3 > 0",
        )
        .lemma(
            "routh_row3",
            LemmaKind.QUERY,
            expr=sympy.Q.positive(a0),
            assumptions=asm,
            depends_on=["routh_cross_term"],
            description="Constant coefficient positive",
        )
        .build()
    )
    return seal(axiom_set, hyp, script)


# ===================================================================
# Stability: Lyapunov equation
# ===================================================================


def lyapunov_stability(
    axiom_set: AxiomSet,
    A: sympy.Matrix,
    P: sympy.Matrix,
    Q: sympy.Matrix,
) -> ProofBundle:
    r"""Prove stability via the Lyapunov equation ``A^T P + P A + Q = 0``.

    If ``P`` is positive definite and ``Q`` is positive semi-definite,
    then ``V(x) = x^T P x`` is a Lyapunov function and the system
    ``dx/dt = Ax`` is stable.

    Verification steps:
    1. Compute the Lyapunov residual ``L = A^T P + P A + Q``
    2. Verify every entry of ``L`` is zero

    Parameters
    ----------
    axiom_set:
        Axiom context.
    A:
        System matrix (n x n).
    P:
        Candidate Lyapunov matrix (n x n, must be positive definite).
    Q:
        Dissipation matrix (n x n, positive semi-definite).
    """
    L = A.T * P + P * A + Q
    n = A.shape[0]

    hyp = axiom_set.hypothesis(
        "lyapunov_equation_satisfied",
        expr=sympy.Eq(sympy.trace(L * L.T), 0),
        description="A^T P + P A + Q = 0 (Lyapunov equation)",
    )

    builder = ProofBuilder(
        axiom_set,
        hyp.name,
        name="lyapunov_stability_proof",
        claim="Lyapunov equation A^T P + P A + Q = 0 holds",
    )

    # Verify each entry of the residual matrix is zero
    for i in range(n):
        for j in range(n):
            entry = sympy.simplify(L[i, j])
            builder = builder.lemma(
                f"L_{i}{j}_zero",
                LemmaKind.EQUALITY,
                expr=entry,
                expected=sympy.Integer(0),
                depends_on=(
                    [f"L_{i}{j - 1}_zero"] if j > 0
                    else [f"L_{i - 1}{n - 1}_zero"] if i > 0
                    else []
                ),
                description=f"Residual entry ({i},{j}) = 0",
            )

    script = builder.build()
    return seal(axiom_set, hyp, script)


# ===================================================================
# Controllability
# ===================================================================


def controllability_rank(
    axiom_set: AxiomSet,
    A: sympy.Matrix,
    B: sympy.Matrix,
) -> ProofBundle:
    """Prove the system ``(A, B)`` is controllable.

    The controllability matrix ``C = [B | AB | A^2 B | ... | A^{n-1}B]``
    must have full row rank (rank = n).  For symbolic matrices, we
    verify this by computing ``det(C * C^T) != 0`` (Gramian is
    nonsingular).

    Parameters
    ----------
    axiom_set:
        Axiom context.
    A:
        System matrix (n x n).
    B:
        Input matrix (n x m).
    """
    n = A.shape[0]

    # Build controllability matrix [B, AB, A^2 B, ..., A^(n-1) B]
    blocks = [B]
    Ak = sympy.eye(n)
    for _ in range(n - 1):
        Ak = Ak * A
        blocks.append(Ak * B)
    C_mat = sympy.Matrix.hstack(*blocks)

    # Gramian: C * C^T (n x n, full rank iff nonsingular)
    gramian = sympy.simplify(C_mat * C_mat.T)
    gram_det = sympy.simplify(gramian.det())

    hyp = axiom_set.hypothesis(
        "controllable",
        expr=sympy.Ne(gram_det, 0),
        description="Controllability Gramian is nonsingular",
    )
    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="controllability_proof",
            claim="det(C * C^T) != 0 => full rank => controllable",
        )
        .lemma(
            "gramian_det_nonzero",
            LemmaKind.BOOLEAN,
            expr=sympy.Ne(gram_det, 0),
            description=f"det(Gramian) = {gram_det} != 0",
        )
        .build()
    )
    return seal(axiom_set, hyp, script)


# ===================================================================
# Observability
# ===================================================================


def observability_rank(
    axiom_set: AxiomSet,
    A: sympy.Matrix,
    C: sympy.Matrix,
) -> ProofBundle:
    """Prove the system ``(A, C)`` is observable.

    The observability matrix ``O = [C; CA; CA^2; ...; CA^{n-1}]``
    must have full column rank (rank = n).  We verify via
    ``det(O^T * O) != 0``.

    Parameters
    ----------
    axiom_set:
        Axiom context.
    A:
        System matrix (n x n).
    C:
        Output matrix (p x n).
    """
    n = A.shape[0]

    blocks = [C]
    Ak = sympy.eye(n)
    for _ in range(n - 1):
        Ak = Ak * A
        blocks.append(C * Ak)
    O_mat = sympy.Matrix.vstack(*blocks)

    gramian = sympy.simplify(O_mat.T * O_mat)
    gram_det = sympy.simplify(gramian.det())

    hyp = axiom_set.hypothesis(
        "observable",
        expr=sympy.Ne(gram_det, 0),
        description="Observability Gramian is nonsingular",
    )
    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="observability_proof",
            claim="det(O^T * O) != 0 => full rank => observable",
        )
        .lemma(
            "gramian_det_nonzero",
            LemmaKind.BOOLEAN,
            expr=sympy.Ne(gram_det, 0),
            description=f"det(Gramian) = {gram_det} != 0",
        )
        .build()
    )
    return seal(axiom_set, hyp, script)


# ===================================================================
# Conservation laws (quadratic invariants)
# ===================================================================


def quadratic_invariant(
    axiom_set: AxiomSet,
    state_symbols: list[sympy.Symbol],
    state_dots: list[sympy.Basic],
    V: sympy.Basic,
) -> ProofBundle:
    r"""Prove ``dV/dt = 0`` along trajectories of the ODE system.

    Given state equations ``dx_i/dt = f_i(x)`` and a candidate
    invariant ``V(x)``, computes the Lie derivative

    .. math::
        \dot{V} = \sum_i \frac{\partial V}{\partial x_i} f_i(x)

    and verifies it simplifies to zero.

    Parameters
    ----------
    axiom_set:
        Axiom context.
    state_symbols:
        Ordered list of state variables ``[x1, x2, ...]``.
    state_dots:
        Time derivatives ``[dx1/dt, dx2/dt, ...]`` as expressions.
    V:
        Candidate invariant expression.
    """
    # Compute Lie derivative: sum of (dV/dx_i) * (dx_i/dt)
    V_dot = sum(
        sympy.diff(V, xi) * fi
        for xi, fi in zip(state_symbols, state_dots, strict=True)
    )
    V_dot_simplified = sympy.simplify(V_dot)

    hyp = axiom_set.hypothesis(
        "quadratic_invariant",
        expr=sympy.Eq(V_dot_simplified, 0),
        description="dV/dt = 0 along trajectories",
    )
    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="invariant_conservation_proof",
            claim="V is conserved: dV/dt = 0",
        )
        .lemma(
            "lie_derivative_zero",
            LemmaKind.EQUALITY,
            expr=V_dot_simplified,
            expected=sympy.Integer(0),
            description="Lie derivative simplifies to 0",
        )
        .build()
    )
    return seal(axiom_set, hyp, script)
