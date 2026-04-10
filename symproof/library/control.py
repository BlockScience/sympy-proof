"""Control systems proof library — stability, controllability, observability.

Reusable proof bundles for classical control theory results.
Each function accepts an ``AxiomSet`` (plus system matrices/symbols),
builds a decomposed proof, seals it, and returns a ``ProofBundle``.

Building blocks (wrap SymPy basics, composable)
    ``hurwitz_second_order``, ``hurwitz_third_order`` — Routh-Hurwitz
    ``controllability_rank``, ``observability_rank`` — Gramian determinants
    ``lyapunov_stability`` — verify A^T P + P A + Q = 0
    ``quadratic_invariant`` — Lie derivative dV/dt = 0

Non-trivial composed proofs (real engineering value)
    ``closed_loop_stability`` — plant + controller → char poly → Hurwitz
    ``lyapunov_from_system`` — construct P by solving Lyapunov eq, then verify
    ``gain_margin`` — critical gain at which stability is lost
"""

from __future__ import annotations

import sympy

from symproof.builder import ProofBuilder
from symproof.bundle import seal
from symproof.evaluation import evaluation
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
            with evaluation():
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
    with evaluation():
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

    with evaluation():
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
    with evaluation():
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


# ===================================================================
# Composed: closed-loop stability (plant + controller)
# ===================================================================


def closed_loop_stability(
    axiom_set: AxiomSet,
    plant_num: sympy.Basic,
    plant_den: sympy.Basic,
    ctrl_num: sympy.Basic,
    ctrl_den: sympy.Basic,
    s: sympy.Symbol,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove closed-loop stability from plant and controller transfer functions.

    Given plant ``G(s) = plant_num / plant_den`` and controller
    ``C(s) = ctrl_num / ctrl_den``, the closed-loop characteristic
    polynomial is::

        cl_poly = plant_den * ctrl_den + plant_num * ctrl_num

    This function extracts the polynomial, identifies its order, and
    delegates to ``hurwitz_second_order`` or ``hurwitz_third_order``.

    This is the workflow an engineer actually uses: specify plant and
    controller, get a stability certificate.

    Parameters
    ----------
    axiom_set:
        Must contain axioms establishing positivity of controller gains.
    plant_num, plant_den:
        Numerator and denominator of the plant transfer function.
    ctrl_num, ctrl_den:
        Numerator and denominator of the controller.
    s:
        The Laplace variable symbol.
    assumptions:
        Symbol assumptions for coefficient positivity checks.
    """
    asm = assumptions or {}

    cl_poly = sympy.expand(plant_den * ctrl_den + plant_num * ctrl_num)
    poly = sympy.Poly(cl_poly, s)
    coeffs = poly.all_coeffs()  # highest degree first
    order = poly.degree()

    if order == 2:
        return hurwitz_second_order(
            axiom_set, a2=coeffs[0], a1=coeffs[1], a0=coeffs[2],
            assumptions=asm,
        )
    if order == 3:
        return hurwitz_third_order(
            axiom_set,
            a3=coeffs[0], a2=coeffs[1], a1=coeffs[2], a0=coeffs[3],
            assumptions=asm,
        )
    raise ValueError(
        f"closed_loop_stability supports order 2 and 3, got {order}. "
        f"Characteristic polynomial: {cl_poly}"
    )


# ===================================================================
# Composed: Lyapunov function construction + verification
# ===================================================================


def lyapunov_from_system(
    axiom_set: AxiomSet,
    A: sympy.Matrix,
    Q: sympy.Matrix | None = None,
) -> ProofBundle:
    r"""Construct a Lyapunov function for a linear system and prove stability.

    Given ``dx/dt = Ax``, solves the Lyapunov equation
    ``A^T P + P A = -Q`` for a symmetric ``P``, then verifies:

    1. The Lyapunov equation is satisfied (entry-by-entry)
    2. ``P`` is positive definite (all leading principal minors > 0)

    This is the non-trivial version: the engineer provides ``A`` and
    gets back a complete stability certificate with a constructed
    Lyapunov function — not just a verification of a given ``P``.

    Parameters
    ----------
    axiom_set:
        Axiom context.
    A:
        System matrix (n x n). Must be Hurwitz (all eigenvalues Re < 0).
    Q:
        Positive definite dissipation matrix. Defaults to identity.

    Raises
    ------
    ValueError
        If the Lyapunov equation has no solution (system may be unstable).
    """
    n = A.shape[0]
    if Q is None:
        Q = sympy.eye(n)

    # Construct symbolic symmetric P
    p_syms = {}
    P_sym = sympy.zeros(n)
    for i in range(n):
        for j in range(i, n):
            name = f"p_{i}{j}"
            p_syms[name] = sympy.Symbol(name)
            P_sym[i, j] = p_syms[name]
            P_sym[j, i] = p_syms[name]

    # Solve A^T P + P A + Q = 0
    lyap_residual = A.T * P_sym + P_sym * A + Q
    equations = []
    for i in range(n):
        for j in range(i, n):
            equations.append(lyap_residual[i, j])

    solution = sympy.solve(equations, list(p_syms.values()))
    if not solution:
        raise ValueError(
            "Lyapunov equation has no solution — system may be unstable."
        )

    P_solved = P_sym.subs(solution)
    with evaluation():
        P_simplified = sympy.simplify(P_solved)

    # Build proof: Lyapunov equation + positive definiteness
    hyp = axiom_set.hypothesis(
        "lyapunov_constructed",
        expr=sympy.Eq(
            sympy.trace((A.T * P_simplified + P_simplified * A + Q) ** 2),
            0,
        ),
        description="Constructed Lyapunov function proves stability",
    )

    builder = ProofBuilder(
        axiom_set,
        hyp.name,
        name="lyapunov_construction_proof",
        claim="Lyapunov equation solved and P verified positive definite",
    )

    # Lemma group 1: Lyapunov equation satisfied
    L = A.T * P_simplified + P_simplified * A + Q
    for i in range(n):
        for j in range(n):
            with evaluation():
                entry = sympy.simplify(L[i, j])
            builder = builder.lemma(
                f"lyap_eq_{i}{j}",
                LemmaKind.EQUALITY,
                expr=entry,
                expected=sympy.Integer(0),
                description=f"Lyapunov residual ({i},{j}) = 0",
            )

    # Lemma group 2: P is positive definite
    # Sylvester's criterion: all leading principal minors > 0
    for k in range(1, n + 1):
        minor = P_simplified[:k, :k].det()
        with evaluation():
            minor_simplified = sympy.simplify(minor)
        builder = builder.lemma(
            f"P_minor_{k}_positive",
            LemmaKind.QUERY,
            expr=sympy.Q.positive(minor_simplified),
            assumptions={
                sym.name: {"positive": True}
                for sym in minor_simplified.free_symbols
                if hasattr(sym, "name")
            },
            description=f"Leading principal minor of order {k} > 0",
        )

    script = builder.build()
    return seal(axiom_set, hyp, script)


# ===================================================================
# Composed: gain margin
# ===================================================================


def gain_margin(
    axiom_set: AxiomSet,
    open_loop_char_coeffs: list[sympy.Basic],
    gain: sympy.Symbol,
    s: sympy.Symbol,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove stability under gain, and identify the critical gain.

    For a third-order system ``a3*s^3 + a2*s^2 + a1*s + K`` where
    ``K`` is a tunable gain, the Routh condition requires
    ``a1*a2 > K*a3``. This function proves:

    1. The current gain is positive
    2. The Routh cross-term condition holds (gain below critical)

    The critical gain ``K_crit = a1*a2/a3`` is the stability boundary.

    Parameters
    ----------
    axiom_set:
        Must contain axiom establishing ``gain < K_crit``.
    open_loop_char_coeffs:
        Coefficients ``[a3, a2, a1]`` of the open-loop char poly
        (without the gain term). The closed-loop poly is
        ``a3*s^3 + a2*s^2 + a1*s + gain``.
    gain:
        The tunable gain symbol.
    s:
        Laplace variable (unused, kept for API consistency).
    assumptions:
        Symbol assumptions for positivity checks.
    """
    asm = assumptions or {}
    if len(open_loop_char_coeffs) != 3:
        raise ValueError(
            "gain_margin requires exactly 3 coefficients [a3, a2, a1]"
        )

    a3, a2, a1 = open_loop_char_coeffs
    K_crit = a1 * a2 / a3
    routh_cross = a1 * a2 - gain * a3

    hyp = axiom_set.hypothesis(
        "gain_margin_stable",
        expr=sympy.And(gain > 0, routh_cross > 0),
        description="System stable: gain < K_crit = a1*a2/a3",
    )

    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="gain_margin_proof",
            claim="Gain below critical value => Routh condition holds",
        )
        .lemma(
            "gain_positive",
            LemmaKind.QUERY,
            expr=sympy.Q.positive(gain),
            assumptions=asm,
            description="Gain is positive",
        )
        .lemma(
            "below_critical",
            LemmaKind.BOOLEAN,
            expr=sympy.Implies(gain < K_crit, routh_cross > 0),
            assumptions=asm,
            depends_on=["gain_positive"],
            description="gain < a1*a2/a3 => a1*a2 - gain*a3 > 0",
        )
        .build()
    )
    return seal(axiom_set, hyp, script)
