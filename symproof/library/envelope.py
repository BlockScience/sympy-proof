"""Danskin's envelope theorem — differentiability of max-value functions.

For a function ``f(x, theta)`` that is strongly concave in ``x``, the
max-value function ``V(theta) = max_x f(x, theta)`` is differentiable
and its gradient is simply the partial derivative of ``f`` with respect
to ``theta`` evaluated at the unique maximiser ``x*(theta)``:

    dV/dtheta = (partial f / partial theta)|_{x = x*(theta)}

The "envelope" name comes from the fact that the gradient of the maximum
ignores how the maximiser shifts — only the direct effect of ``theta``
on ``f`` matters, because the first-order condition zeroes out the
indirect effect through ``x*``.

Building blocks
    ``envelope_theorem`` — solve FOC, verify envelope identity
"""

from __future__ import annotations

import sympy

from symproof.builder import ProofBuilder
from symproof.bundle import seal
from symproof.models import AxiomSet, LemmaKind, ProofBundle


def envelope_theorem(
    axiom_set: AxiomSet,
    f: sympy.Basic,
    x: sympy.Symbol,
    theta: sympy.Symbol,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    r"""Prove Danskin's envelope theorem for a concrete ``f(x, theta)``.

    Given ``f`` strongly concave in ``x``, computes the unique maximiser
    ``x*(theta)`` via the first-order condition, constructs the max-value
    function ``V(theta) = f(x*(theta), theta)``, and verifies the
    envelope identity::

        dV/dtheta = (partial f / partial theta)|_{x = x*(theta)}

    The proof chain:

    1. Solve FOC ``df/dx = 0`` for ``x*``
    2. Verify ``f`` is strictly concave in ``x`` (``f_xx < 0``)
    3. Verify Hessian invertibility (``f_xx != 0``, IFT condition)
    4. Construct ``V(theta)`` and differentiate
    5. Verify envelope identity: ``dV/dtheta = df/dtheta|_{x*}``

    Parameters
    ----------
    axiom_set:
        Axiom context (e.g., parameter constraints).
    f:
        The function ``f(x, theta)`` — must be strongly concave in ``x``
        so that the FOC has a unique solution.
    x:
        The optimisation variable.
    theta:
        The parameter with respect to which the envelope is computed.
    assumptions:
        Symbol assumptions for positivity / inequality checks.

    Returns
    -------
    ProofBundle
        Sealed proof of the envelope identity.

    Raises
    ------
    ValueError
        If the FOC cannot be solved symbolically.
    """
    asm = assumptions or {}

    # ── Step 1: FOC and maximiser ──────────────────────────────
    f_x = sympy.diff(f, x)
    x_star_solutions = sympy.solve(f_x, x)
    if not x_star_solutions:
        raise ValueError(
            f"Cannot solve FOC df/dx = 0 for {x}. "
            f"FOC expression: {f_x}"
        )
    x_star = x_star_solutions[0]

    # ── Step 2: Concavity ──────────────────────────────────────
    f_xx = sympy.simplify(sympy.diff(f, x, 2))

    # ── Step 3: Max-value function and envelope ────────────────
    V = sympy.simplify(f.subs(x, x_star))
    dV_dtheta = sympy.simplify(sympy.diff(V, theta))
    partial_f_theta_at_star = sympy.simplify(
        sympy.diff(f, theta).subs(x, x_star)
    )

    # ── Hypothesis ─────────────────────────────────────────────
    hyp = axiom_set.hypothesis(
        "envelope_identity",
        expr=sympy.Eq(dV_dtheta, partial_f_theta_at_star),
        description=(
            f"Danskin's envelope theorem: "
            f"dV/d{theta} = (df/d{theta})|_{{x={x_star}}}"
        ),
    )

    # ── Proof script ───────────────────────────────────────────
    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="envelope_theorem",
            claim=(
                f"For f strongly concave in {x}, the max-value function "
                f"V({theta}) = f({x_star}, {theta}) satisfies "
                f"dV/d{theta} = df/d{theta}|_{{x={x_star}}} (envelope theorem)."
            ),
        )
        .lemma(
            "foc_solution",
            LemmaKind.EQUALITY,
            expr=sympy.simplify(f_x.subs(x, x_star)),
            expected=sympy.Integer(0),
            description=f"FOC: df/d{x} = 0 at {x} = {x_star}.",
        )
        .lemma(
            "strict_concavity",
            LemmaKind.BOOLEAN,
            expr=sympy.Lt(f_xx, 0),
            assumptions=asm,
            depends_on=["foc_solution"],
            description=f"f is strictly concave in {x}: d²f/d{x}² = {f_xx} < 0.",
        )
        .lemma(
            "hessian_invertible",
            LemmaKind.BOOLEAN,
            expr=sympy.Ne(f_xx, 0),
            assumptions=asm,
            depends_on=["strict_concavity"],
            description=(
                f"d²f/d{x}² = {f_xx} != 0 — the Hessian is invertible, "
                f"so by the implicit function theorem {x}*({theta}) is smooth."
            ),
        )
        .lemma(
            "envelope_identity_verified",
            LemmaKind.EQUALITY,
            expr=dV_dtheta - partial_f_theta_at_star,
            expected=sympy.Integer(0),
            depends_on=["hessian_invertible"],
            description=(
                f"dV/d{theta} - (df/d{theta})|_{{x*}} = 0. "
                f"The envelope identity holds."
            ),
        )
        .build()
    )

    return seal(axiom_set, hyp, script)
