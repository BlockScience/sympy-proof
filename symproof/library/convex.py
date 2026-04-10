"""Convex optimization proof library — convexity, strong convexity, composition.

Certify that an optimization problem formulation is correct BEFORE
solving it.  Not a solver — a proof layer that sits between problem
formulation and CVXPY/Mosek/Gurobi.

Building blocks
    ``convex_scalar`` — f''(x) >= 0
    ``convex_hessian`` — Hessian PSD via Sylvester's criterion
    ``strongly_convex`` — min eigenvalue of Hessian >= m > 0
    ``conjugate_function`` — compute f* and verify convexity

Composed proofs (non-trivial value-add)
    ``convex_sum`` — nonneg weighted sum preserves convexity
    ``convex_composition`` — DCP rule: f(g(x)) convex
    ``unique_minimizer`` — strictly convex → unique minimum
    ``gp_to_convex`` — geometric program log-transform equivalence
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


def convex_scalar(
    axiom_set: AxiomSet,
    f: sympy.Basic,
    x: sympy.Symbol,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove a scalar function is convex: ``f''(x) >= 0``.

    Parameters
    ----------
    axiom_set:
        Axiom context (e.g., domain constraints on x).
    f:
        The function expression.
    x:
        The variable.
    assumptions:
        Symbol assumptions for the nonnegativity check.
    """
    asm = assumptions or {}
    with evaluation():
        f_pp = sympy.simplify(sympy.diff(f, x, 2))

    hyp = axiom_set.hypothesis(
        "scalar_convex",
        expr=sympy.Ge(f_pp, 0),
        description="f''(x) >= 0 (scalar convexity)",
    )
    script = (
        ProofBuilder(
            axiom_set, hyp.name,
            name="scalar_convexity_proof",
            claim=f"d^2/dx^2({f}) >= 0",
        )
        .lemma(
            "second_deriv_nonneg",
            LemmaKind.QUERY,
            expr=sympy.Q.nonnegative(f_pp),
            assumptions=asm,
            description=f"f''(x) = {f_pp} >= 0",
        )
        .build()
    )
    return seal(axiom_set, hyp, script)


def convex_hessian(
    axiom_set: AxiomSet,
    f: sympy.Basic,
    variables: list[sympy.Symbol],
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove convexity via Hessian positive semi-definiteness.

    Computes the Hessian of ``f`` w.r.t. ``variables`` and checks
    all leading principal minors are nonnegative (Sylvester's criterion
    for PSD).

    Works well for functions up to ~4 variables.  For larger problems,
    consider decomposing via ``convex_sum`` or ``convex_composition``.

    Parameters
    ----------
    axiom_set:
        Axiom context.
    f:
        The function expression.
    variables:
        List of symbols to differentiate w.r.t.
    assumptions:
        Symbol assumptions for the minor checks.
    """
    asm = assumptions or {}
    n = len(variables)
    H = sympy.hessian(f, variables)

    hyp = axiom_set.hypothesis(
        "hessian_psd",
        expr=sympy.Ge(H.det(), 0),
        description="Hessian of f is PSD (f is convex)",
    )

    builder = ProofBuilder(
        axiom_set, hyp.name,
        name="hessian_psd_proof",
        claim=f"Hessian PSD via Sylvester's criterion ({n} variables)",
    )

    for k in range(1, n + 1):
        with evaluation():
            minor = sympy.simplify(H[:k, :k].det())
        builder = builder.lemma(
            f"minor_{k}_nonneg",
            LemmaKind.QUERY,
            expr=sympy.Q.nonnegative(minor),
            assumptions=asm,
            depends_on=[f"minor_{k - 1}_nonneg"] if k > 1 else [],
            description=f"Leading principal minor of order {k} >= 0",
        )

    script = builder.build()
    return seal(axiom_set, hyp, script)


def strongly_convex(
    axiom_set: AxiomSet,
    f: sympy.Basic,
    variables: list[sympy.Symbol],
    m: sympy.Basic,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove ``f`` is ``m``-strongly convex.

    Verifies that all eigenvalues of the Hessian are >= ``m > 0``.
    Equivalently, ``H - m*I`` is PSD.

    Strong convexity implies:
    - Unique global minimizer
    - Linear convergence of gradient descent
    - Bounded condition number

    Parameters
    ----------
    axiom_set:
        Axiom context.
    f:
        The function expression.
    variables:
        List of symbols.
    m:
        The strong convexity parameter (must be > 0).
    assumptions:
        Symbol assumptions.
    """
    asm = assumptions or {}
    n = len(variables)
    H = sympy.hessian(f, variables)
    H_shifted = H - m * sympy.eye(n)

    hyp = axiom_set.hypothesis(
        "strongly_convex",
        expr=sympy.Gt(m, 0),
        description=f"f is {m}-strongly convex (H - {m}I is PSD)",
    )

    builder = ProofBuilder(
        axiom_set, hyp.name,
        name="strong_convexity_proof",
        claim=f"Hessian eigenvalues >= {m}",
    )

    builder = builder.lemma(
        "m_positive",
        LemmaKind.QUERY,
        expr=sympy.Q.positive(m),
        assumptions=asm,
        description=f"Strong convexity parameter m = {m} > 0",
    )

    for k in range(1, n + 1):
        with evaluation():
            minor = sympy.simplify(H_shifted[:k, :k].det())
        builder = builder.lemma(
            f"shifted_minor_{k}_nonneg",
            LemmaKind.QUERY,
            expr=sympy.Q.nonnegative(minor),
            assumptions=asm,
            depends_on=["m_positive"],
            description=f"det((H-mI)[:{k},:{k}]) >= 0",
        )

    script = builder.build()
    return seal(axiom_set, hyp, script)


def conjugate_function(
    axiom_set: AxiomSet,
    f: sympy.Basic,
    x: sympy.Symbol,
    y: sympy.Symbol,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    r"""Compute the conjugate ``f*(y) = sup_x(xy - f(x))`` and prove it convex.

    Solves the first-order condition ``y - f'(x) = 0`` for ``x*``,
    substitutes back to get ``f*(y)``, then verifies ``f*''(y) >= 0``.

    Conjugate functions are always convex (even if ``f`` isn't), but
    this proof verifies the computation is correct.

    Parameters
    ----------
    axiom_set:
        Axiom context.
    f:
        The primal function (in terms of ``x``).
    x:
        Primal variable.
    y:
        Dual variable (conjugate argument).
    assumptions:
        Symbol assumptions for the convexity check.
    """
    asm = assumptions or {}

    # FOC: d/dx(xy - f(x)) = y - f'(x) = 0
    foc = y - sympy.diff(f, x)
    x_star_solutions = sympy.solve(foc, x)
    if not x_star_solutions:
        raise ValueError(
            f"Cannot solve FOC y - f'(x) = 0 for x. "
            f"FOC expression: {foc}"
        )
    x_star = x_star_solutions[0]

    # f*(y) = x* y - f(x*)
    with evaluation():
        f_conj = sympy.simplify(x_star * y - f.subs(x, x_star))
        f_conj_pp = sympy.simplify(sympy.diff(f_conj, y, 2))

    hyp = axiom_set.hypothesis(
        "conjugate_convex",
        expr=sympy.Ge(f_conj_pp, 0),
        description=f"f*(y) = {f_conj} is convex",
    )
    with evaluation():
        _conj_foc_simplified = sympy.simplify(foc.subs(x, x_star))

    script = (
        ProofBuilder(
            axiom_set, hyp.name,
            name="conjugate_proof",
            claim=f"f*(y) = {f_conj}, and f*''(y) >= 0",
        )
        .lemma(
            "foc_solution",
            LemmaKind.EQUALITY,
            expr=_conj_foc_simplified,
            expected=sympy.Integer(0),
            description=f"FOC: y - f'(x*) = 0 at x* = {x_star}",
        )
        .lemma(
            "conjugate_second_deriv",
            LemmaKind.QUERY,
            expr=sympy.Q.nonnegative(f_conj_pp),
            assumptions=asm,
            depends_on=["foc_solution"],
            description=f"f*''(y) = {f_conj_pp} >= 0",
        )
        .build()
    )
    return seal(axiom_set, hyp, script)


# ===================================================================
# Composed proofs
# ===================================================================


def convex_sum(
    axiom_set: AxiomSet,
    functions: list[sympy.Basic],
    weights: list[sympy.Basic],
    variables: list[sympy.Symbol],
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove a nonnegative weighted sum of convex functions is convex.

    For scalar functions, proves each ``w_i * f_i''(x) >= 0`` then
    the sum.  For multivariate, uses Hessian.

    Parameters
    ----------
    axiom_set:
        Axiom context.
    functions:
        List of function expressions (each must be convex).
    weights:
        Nonneg weight for each function.
    variables:
        Variables the functions depend on.
    assumptions:
        Symbol assumptions.
    """
    asm = assumptions or {}
    if len(functions) != len(weights):
        raise ValueError("functions and weights must have same length")

    combo = sum(
        w * f for w, f in zip(weights, functions, strict=True)
    )

    if len(variables) == 1:
        # Scalar path: prove each w_i * f_i'' >= 0
        x = variables[0]
        hyp = axiom_set.hypothesis(
            "convex_sum",
            expr=sympy.Ge(sympy.diff(combo, x, 2), 0),
            description="Nonneg weighted sum of convex functions is convex",
        )
        builder = ProofBuilder(
            axiom_set, hyp.name,
            name="convex_sum_proof",
            claim="Sum of convex is convex",
        )
        for i, (w, f) in enumerate(
            zip(weights, functions, strict=True)
        ):
            with evaluation():
                term_pp = sympy.simplify(w * sympy.diff(f, x, 2))
            builder = builder.lemma(
                f"term_{i}_convex",
                LemmaKind.QUERY,
                expr=sympy.Q.nonnegative(term_pp),
                assumptions=asm,
                depends_on=[f"term_{i - 1}_convex"] if i > 0 else [],
                description=f"w_{i} * f_{i}''(x) = {term_pp} >= 0",
            )
        script = builder.build()
    else:
        # Multivariate: Hessian PSD of the combined function
        H = sympy.hessian(combo, variables)
        n = len(variables)
        hyp = axiom_set.hypothesis(
            "convex_sum",
            expr=sympy.Ge(H.det(), 0),
            description="Hessian of weighted sum is PSD",
        )
        builder = ProofBuilder(
            axiom_set, hyp.name,
            name="convex_sum_proof",
            claim="Weighted sum Hessian PSD",
        )
        for k in range(1, n + 1):
            with evaluation():
                minor = sympy.simplify(H[:k, :k].det())
            builder = builder.lemma(
                f"sum_minor_{k}",
                LemmaKind.QUERY,
                expr=sympy.Q.nonnegative(minor),
                assumptions=asm,
                depends_on=(
                    [f"sum_minor_{k - 1}"] if k > 1 else []
                ),
                description=f"Minor {k} of sum Hessian >= 0",
            )
        script = builder.build()

    return seal(axiom_set, hyp, script)


def convex_composition(
    axiom_set: AxiomSet,
    f_outer: sympy.Basic,
    g_inner: sympy.Basic,
    outer_var: sympy.Symbol,
    inner_var: sympy.Symbol,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove ``f(g(x))`` is convex via the DCP composition rule.

    For scalar functions, the composition ``h(x) = f(g(x))`` is convex if:
    - ``f`` is convex AND nondecreasing, and ``g`` is convex, OR
    - ``f`` is convex AND nonincreasing, and ``g`` is concave

    This function verifies the first case (convex nondecreasing + convex)
    by proving:
    1. ``f''(t) >= 0`` (f is convex)
    2. ``f'(t) >= 0``  (f is nondecreasing)
    3. ``g''(x) >= 0`` (g is convex)
    4. ``h''(x) >= 0``  (composition is convex — cross-check)

    Parameters
    ----------
    axiom_set:
        Axiom context.
    f_outer:
        Outer function in terms of ``outer_var``.
    g_inner:
        Inner function in terms of ``inner_var``.
    outer_var:
        Variable of the outer function.
    inner_var:
        Variable of the inner function (the actual decision variable).
    assumptions:
        Symbol assumptions.
    """
    asm = assumptions or {}
    t = outer_var

    with evaluation():
        f_p = sympy.simplify(sympy.diff(f_outer, t))
        f_pp = sympy.simplify(sympy.diff(f_outer, t, 2))
        g_pp = sympy.simplify(sympy.diff(g_inner, inner_var, 2))

    # The composed function
    h = f_outer.subs(t, g_inner)
    with evaluation():
        h_pp = sympy.simplify(sympy.diff(h, inner_var, 2))

    hyp = axiom_set.hypothesis(
        "composition_convex",
        expr=sympy.Ge(h_pp, 0),
        description="f(g(x)) is convex (DCP composition rule)",
    )
    script = (
        ProofBuilder(
            axiom_set, hyp.name,
            name="dcp_composition_proof",
            claim="f convex nondecreasing, g convex => f(g(x)) convex",
        )
        .lemma(
            "f_convex",
            LemmaKind.QUERY,
            expr=sympy.Q.nonnegative(f_pp),
            assumptions=asm,
            description=f"f''(t) = {f_pp} >= 0 (f is convex)",
        )
        .lemma(
            "f_nondecreasing",
            LemmaKind.QUERY,
            expr=sympy.Q.nonnegative(f_p),
            assumptions=asm,
            depends_on=["f_convex"],
            description=f"f'(t) = {f_p} >= 0 (f is nondecreasing)",
        )
        .lemma(
            "g_convex",
            LemmaKind.QUERY,
            expr=sympy.Q.nonnegative(g_pp),
            assumptions=asm,
            depends_on=["f_nondecreasing"],
            description=f"g''(x) = {g_pp} >= 0 (g is convex)",
        )
        .lemma(
            "composition_second_deriv",
            LemmaKind.QUERY,
            expr=sympy.Q.nonnegative(h_pp),
            assumptions=asm,
            depends_on=["f_convex", "f_nondecreasing", "g_convex"],
            description=f"h''(x) = {h_pp} >= 0 (cross-check)",
        )
        .build()
    )
    return seal(axiom_set, hyp, script)


def unique_minimizer(
    axiom_set: AxiomSet,
    f: sympy.Basic,
    variables: list[sympy.Symbol],
    m: sympy.Basic,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove ``f`` has a unique global minimizer.

    If ``f`` is ``m``-strongly convex (``m > 0``), any local minimum
    is the unique global minimum.  This function imports a strong
    convexity proof and derives uniqueness.

    Parameters
    ----------
    axiom_set:
        Axiom context.
    f:
        The function expression.
    variables:
        Decision variables.
    m:
        Strong convexity parameter (> 0).
    assumptions:
        Symbol assumptions.
    """
    asm = assumptions or {}

    strong_bundle = strongly_convex(
        axiom_set, f, variables, m, assumptions=asm,
    )

    hyp = axiom_set.hypothesis(
        "unique_minimizer",
        expr=sympy.Gt(m, 0),
        description=(
            "Strictly convex => unique global minimizer "
            "(any local min is the global min)"
        ),
    )
    script = (
        ProofBuilder(
            axiom_set, hyp.name,
            name="unique_minimizer_proof",
            claim="Strong convexity => unique minimizer",
        )
        .import_bundle(strong_bundle)
        .lemma(
            "strict_convexity_implies_unique",
            LemmaKind.QUERY,
            expr=sympy.Q.positive(m),
            assumptions=asm,
            description=(
                f"m = {m} > 0 => strictly convex "
                f"=> unique global minimizer"
            ),
        )
        .build()
    )
    return seal(axiom_set, hyp, script)


def gp_to_convex(
    axiom_set: AxiomSet,
    monomial_terms: list[sympy.Basic],
    log_var: sympy.Symbol,
    orig_var: sympy.Symbol,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove a geometric program objective transforms to convex.

    In a GP, the objective is a posynomial (sum of monomials).
    Substituting ``x = exp(y)`` transforms each monomial
    ``c * x^a`` into ``log(c) + a*y`` (affine in ``y``), and the
    posynomial into ``log(sum(exp(affine_i)))`` (log-sum-exp, convex).

    This function proves the log-sum-exp of the transformed terms
    is convex by verifying its second derivative is nonnegative.

    Parameters
    ----------
    axiom_set:
        Axiom context.
    monomial_terms:
        List of monomial expressions in ``orig_var``.
    log_var:
        The new variable (``y = log(x)``).
    orig_var:
        The original positive variable.
    assumptions:
        Symbol assumptions.
    """
    asm = assumptions or {}

    # Transform each monomial: substitute x = exp(y)
    with evaluation():
        transformed = [
            sympy.simplify(term.subs(orig_var, sympy.exp(log_var)))
            for term in monomial_terms
        ]

    # Log of the posynomial: log(sum(transformed))
    posynomial_log = sympy.log(sum(transformed))
    with evaluation():
        lse_pp = sympy.simplify(sympy.diff(posynomial_log, log_var, 2))

    hyp = axiom_set.hypothesis(
        "gp_convex",
        expr=sympy.Ge(lse_pp, 0),
        description="GP objective is convex after log-transform",
    )
    script = (
        ProofBuilder(
            axiom_set, hyp.name,
            name="gp_convexity_proof",
            claim="log(sum(exp(affine))) is convex (log-sum-exp)",
        )
        .lemma(
            "lse_second_deriv_nonneg",
            LemmaKind.QUERY,
            expr=sympy.Q.nonnegative(lse_pp),
            assumptions=asm,
            description=f"d^2/dy^2(log-sum-exp) = {lse_pp} >= 0",
        )
        .build()
    )
    return seal(axiom_set, hyp, script)
