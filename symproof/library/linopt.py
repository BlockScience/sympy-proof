"""Linear optimization proof library — LP, ILP, MILP validation.

Validate solutions from solvers (Gurobi, PuLP, CPLEX, etc.).
symproof does not solve — it proves that a given solution has the
claimed properties: feasibility, optimality, duality, integrality.

Building blocks
    ``feasible_point`` — Ax = b, x >= 0
    ``dual_feasible`` — A^T y + z = c, z >= 0
    ``strong_duality`` — c^T x* = b^T y*
    ``complementary_slackness`` — x_i * z_i = 0 for all i
    ``integer_feasible`` — feasible + all entries integer

Composed proofs
    ``lp_optimal`` — primal + dual feasibility + strong duality
    ``lp_relaxation_bound`` — LP value bounds ILP value
"""

from __future__ import annotations

import sympy

from symproof.builder import ProofBuilder
from symproof.bundle import seal
from symproof.evaluation import evaluation
from symproof.models import AxiomSet, LemmaKind, ProofBundle


# ===================================================================
# Building blocks
# ===================================================================


def feasible_point(
    axiom_set: AxiomSet,
    A: sympy.Matrix,
    b: sympy.Matrix,
    x_star: sympy.Matrix,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove a point is feasible for an LP: Ax* = b and x* >= 0.

    Verifies each constraint row and each nonnegativity bound
    entry-by-entry.

    Parameters
    ----------
    axiom_set:
        Axiom context.
    A:
        Constraint matrix (m x n).
    b:
        Right-hand side (m x 1).
    x_star:
        Candidate feasible point (n x 1).
    assumptions:
        Symbol assumptions for nonnegativity checks.
    """
    asm = assumptions or {}
    n = x_star.rows

    with evaluation():
        residual = sympy.simplify(A * x_star - b)

    hyp = axiom_set.hypothesis(
        "primal_feasible",
        expr=sympy.S.true,
        description=f"x* is feasible: Ax* = b and x* >= 0 ({n} variables)",
    )

    builder = ProofBuilder(
        axiom_set,
        hyp.name,
        name="feasibility_proof",
        claim="Verify Ax* = b entry-by-entry and x* >= 0 component-wise.",
    )

    # Equality constraints: each row of A*x - b = 0
    for i in range(residual.rows):
        builder = builder.lemma(
            f"constraint_{i}",
            LemmaKind.EQUALITY,
            expr=residual[i],
            expected=sympy.Integer(0),
            description=f"Row {i}: (Ax* - b)[{i}] = 0",
        )

    # Nonnegativity: each x_i >= 0
    for j in range(n):
        val = x_star[j]
        if val.is_number:
            builder = builder.lemma(
                f"x_{j}_nonneg",
                LemmaKind.BOOLEAN,
                expr=sympy.Ge(val, 0),
                depends_on=[f"constraint_{residual.rows - 1}"],
                description=f"x*[{j}] = {val} >= 0",
            )
        else:
            builder = builder.lemma(
                f"x_{j}_nonneg",
                LemmaKind.QUERY,
                expr=sympy.Q.nonnegative(val),
                assumptions=asm,
                depends_on=[f"constraint_{residual.rows - 1}"],
                description=f"x*[{j}] >= 0",
            )

    return seal(axiom_set, hyp, builder.build())


def dual_feasible(
    axiom_set: AxiomSet,
    A: sympy.Matrix,
    c: sympy.Matrix,
    y_star: sympy.Matrix,
    z_star: sympy.Matrix,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove dual feasibility: A^T y* + z* = c and z* >= 0.

    Parameters
    ----------
    axiom_set:
        Axiom context.
    A:
        Constraint matrix (m x n).
    c:
        Objective coefficients (n x 1).
    y_star:
        Dual variables for equality constraints (m x 1).
    z_star:
        Dual variables for nonnegativity bounds (n x 1).
    assumptions:
        Symbol assumptions for checks.
    """
    asm = assumptions or {}

    with evaluation():
        dual_residual = sympy.simplify(A.T * y_star + z_star - c)

    hyp = axiom_set.hypothesis(
        "dual_feasible",
        expr=sympy.S.true,
        description="Dual feasible: A^T y* + z* = c and z* >= 0",
    )

    builder = ProofBuilder(
        axiom_set,
        hyp.name,
        name="dual_feasibility_proof",
        claim="Verify A^T y* + z* = c and z* >= 0.",
    )

    for i in range(dual_residual.rows):
        builder = builder.lemma(
            f"dual_constraint_{i}",
            LemmaKind.EQUALITY,
            expr=dual_residual[i],
            expected=sympy.Integer(0),
            description=f"(A^T y* + z* - c)[{i}] = 0",
        )

    for j in range(z_star.rows):
        val = z_star[j]
        if val.is_number:
            builder = builder.lemma(
                f"z_{j}_nonneg",
                LemmaKind.BOOLEAN,
                expr=sympy.Ge(val, 0),
                description=f"z*[{j}] = {val} >= 0",
            )
        else:
            builder = builder.lemma(
                f"z_{j}_nonneg",
                LemmaKind.QUERY,
                expr=sympy.Q.nonnegative(val),
                assumptions=asm,
                description=f"z*[{j}] >= 0",
            )

    return seal(axiom_set, hyp, builder.build())


def strong_duality(
    axiom_set: AxiomSet,
    c: sympy.Matrix,
    b: sympy.Matrix,
    x_star: sympy.Matrix,
    y_star: sympy.Matrix,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove strong duality: primal value c^T x* = dual value b^T y*.

    Parameters
    ----------
    axiom_set:
        Axiom context.
    c:
        Objective coefficients (n x 1).
    b:
        RHS of equality constraints (m x 1).
    x_star:
        Primal optimal (n x 1).
    y_star:
        Dual optimal (m x 1).
    assumptions:
        Symbol assumptions.
    """
    with evaluation():
        primal_val = sympy.simplify((c.T * x_star)[0, 0])
        dual_val = sympy.simplify((b.T * y_star)[0, 0])
        gap = sympy.simplify(primal_val - dual_val)

    hyp = axiom_set.hypothesis(
        "strong_duality",
        expr=sympy.Eq(gap, 0),
        description=f"Strong duality: c^T x* = b^T y* (gap = 0)",
    )

    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="strong_duality_proof",
            claim="Primal objective equals dual objective (zero duality gap).",
        )
        .lemma(
            "primal_value",
            LemmaKind.EQUALITY,
            expr=primal_val,
            expected=primal_val,
            description=f"Primal value: c^T x* = {primal_val}",
        )
        .lemma(
            "duality_gap_zero",
            LemmaKind.EQUALITY,
            expr=gap,
            expected=sympy.Integer(0),
            depends_on=["primal_value"],
            description=f"c^T x* - b^T y* = 0",
        )
        .build()
    )

    return seal(axiom_set, hyp, script)


def complementary_slackness(
    axiom_set: AxiomSet,
    x_star: sympy.Matrix,
    z_star: sympy.Matrix,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove complementary slackness: x_i * z_i = 0 for all i.

    Parameters
    ----------
    axiom_set:
        Axiom context.
    x_star:
        Primal solution (n x 1).
    z_star:
        Dual bound variables (n x 1).
    assumptions:
        Symbol assumptions.
    """
    n = x_star.rows

    hyp = axiom_set.hypothesis(
        "complementary_slackness",
        expr=sympy.S.true,
        description="Complementary slackness: x_i * z_i = 0 for all i",
    )

    builder = ProofBuilder(
        axiom_set,
        hyp.name,
        name="complementary_slackness_proof",
        claim="Each primal-dual product x_i * z_i = 0.",
    )

    for i in range(n):
        with evaluation():
            product = sympy.simplify(x_star[i] * z_star[i])
        builder = builder.lemma(
            f"cs_{i}",
            LemmaKind.EQUALITY,
            expr=product,
            expected=sympy.Integer(0),
            description=f"x*[{i}] * z*[{i}] = {x_star[i]} * {z_star[i]} = 0",
        )

    return seal(axiom_set, hyp, builder.build())


def integer_feasible(
    axiom_set: AxiomSet,
    A: sympy.Matrix,
    b: sympy.Matrix,
    x_star: sympy.Matrix,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove a point is feasible for an ILP: Ax* = b, x* >= 0, x* integer.

    Parameters
    ----------
    axiom_set:
        Axiom context.
    A:
        Constraint matrix (m x n).
    b:
        Right-hand side (m x 1).
    x_star:
        Candidate integer feasible point (n x 1).
    assumptions:
        Symbol assumptions.
    """
    asm = assumptions or {}
    n = x_star.rows

    with evaluation():
        residual = sympy.simplify(A * x_star - b)

    hyp = axiom_set.hypothesis(
        "integer_feasible",
        expr=sympy.S.true,
        description="x* is integer-feasible: Ax* = b, x* >= 0, x* integer",
    )

    builder = ProofBuilder(
        axiom_set,
        hyp.name,
        name="integer_feasibility_proof",
        claim="Verify Ax* = b, x* >= 0, and all entries are integer.",
    )

    # Equality constraints
    for i in range(residual.rows):
        builder = builder.lemma(
            f"constraint_{i}",
            LemmaKind.EQUALITY,
            expr=residual[i],
            expected=sympy.Integer(0),
            description=f"(Ax* - b)[{i}] = 0",
        )

    # Nonnegativity + integrality
    for j in range(n):
        val = x_star[j]
        if val.is_number:
            builder = builder.lemma(
                f"x_{j}_nonneg",
                LemmaKind.BOOLEAN,
                expr=sympy.Ge(val, 0),
                description=f"x*[{j}] = {val} >= 0",
            )
            builder = builder.lemma(
                f"x_{j}_integer",
                LemmaKind.BOOLEAN,
                expr=sympy.Eq(val, sympy.floor(val)),
                depends_on=[f"x_{j}_nonneg"],
                description=f"x*[{j}] = {val} is integer",
            )
        else:
            builder = builder.lemma(
                f"x_{j}_nonneg",
                LemmaKind.QUERY,
                expr=sympy.Q.nonnegative(val),
                assumptions=asm,
                description=f"x*[{j}] >= 0",
            )
            builder = builder.lemma(
                f"x_{j}_integer",
                LemmaKind.QUERY,
                expr=sympy.Q.integer(val),
                assumptions=asm,
                depends_on=[f"x_{j}_nonneg"],
                description=f"x*[{j}] is integer",
            )

    return seal(axiom_set, hyp, builder.build())


# ===================================================================
# Composed proofs
# ===================================================================


def lp_optimal(
    axiom_set: AxiomSet,
    c: sympy.Matrix,
    A: sympy.Matrix,
    b: sympy.Matrix,
    x_star: sympy.Matrix,
    y_star: sympy.Matrix,
    z_star: sympy.Matrix,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove LP optimality: primal feasible + dual feasible + strong duality.

    Composes three building-block proofs into one sealed bundle that
    establishes x* is optimal.

    Parameters
    ----------
    axiom_set:
        Axiom context.
    c:
        Objective coefficients (n x 1).
    A:
        Constraint matrix (m x n).
    b:
        RHS (m x 1).
    x_star:
        Primal optimal (n x 1).
    y_star:
        Dual optimal for equalities (m x 1).
    z_star:
        Dual optimal for bounds (n x 1).
    assumptions:
        Symbol assumptions.
    """
    asm = assumptions or {}

    # Build sub-proofs
    primal_bundle = feasible_point(axiom_set, A, b, x_star, assumptions=asm)
    dual_bundle = dual_feasible(axiom_set, A, c, y_star, z_star, assumptions=asm)
    duality_bundle = strong_duality(axiom_set, c, b, x_star, y_star, assumptions=asm)

    with evaluation():
        obj_val = sympy.simplify((c.T * x_star)[0, 0])

    hyp = axiom_set.hypothesis(
        "lp_optimal",
        expr=sympy.S.true,
        description=(
            f"x* is optimal for LP with objective value {obj_val}: "
            f"primal feasible, dual feasible, zero duality gap"
        ),
    )

    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="lp_optimality_proof",
            claim=f"x* is LP-optimal with value {obj_val}.",
        )
        .import_bundle(primal_bundle)
        .import_bundle(dual_bundle)
        .import_bundle(duality_bundle)
        .lemma(
            "optimality_conclusion",
            LemmaKind.BOOLEAN,
            expr=sympy.S.true,
            description=(
                "Primal feasible + dual feasible + zero duality gap "
                "=> x* is optimal (LP strong duality theorem)."
            ),
        )
        .build()
    )

    return seal(axiom_set, hyp, script)


def lp_relaxation_bound(
    axiom_set: AxiomSet,
    lp_value: sympy.Basic,
    ilp_value: sympy.Basic,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove LP relaxation bounds the ILP: lp_value <= ilp_value (for min).

    The LP relaxation of an ILP has a larger feasible set (no integrality
    constraint), so its optimal value is a lower bound.

    Parameters
    ----------
    axiom_set:
        Axiom context.
    lp_value:
        Optimal value of the LP relaxation.
    ilp_value:
        Optimal value of the ILP.
    assumptions:
        Symbol assumptions.
    """
    with evaluation():
        gap = sympy.simplify(ilp_value - lp_value)

    hyp = axiom_set.hypothesis(
        "relaxation_bound",
        expr=sympy.Ge(gap, 0),
        description=f"LP relaxation bound: {lp_value} <= {ilp_value}",
    )

    asm = assumptions or {}

    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="relaxation_bound_proof",
            claim="LP relaxation value <= ILP optimal value.",
        )
        .lemma(
            "gap_nonneg",
            LemmaKind.BOOLEAN,
            expr=sympy.Ge(gap, 0),
            description=f"ILP value - LP value = {gap} >= 0",
        )
        .build()
    )

    return seal(axiom_set, hyp, script)
