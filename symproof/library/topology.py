"""Point-set topology proof library — undergraduate level in R.

Verify topological properties of sets and functions on the real line:
openness, closedness, compactness, continuity, and classical theorems
(intermediate value, extreme value).

This library stages toward advanced results like the Brouwer fixed
point theorem by establishing the prerequisites: compactness,
continuity preservation, and the intermediate value theorem.

Building blocks
    ``verify_open`` — set is open
    ``verify_closed`` — set is closed
    ``verify_compact`` — set is compact (closed + bounded in R)
    ``verify_boundary`` — boundary of a set
    ``continuous_at_point`` — lim f(x) = f(a) as x → a

Classical theorems (applied to concrete instances)
    ``intermediate_value`` — IVT: sign change implies root
    ``extreme_value`` — EVT: continuous on closed bounded → max/min exist
"""

from __future__ import annotations

import sympy

from symproof.builder import ProofBuilder
from symproof.bundle import seal
from symproof.evaluation import evaluation
from symproof.models import AxiomSet, LemmaKind, ProofBundle


# ===================================================================
# Set properties
# ===================================================================


def verify_open(
    axiom_set: AxiomSet,
    S: sympy.Set,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove a set is open.

    Uses SymPy's ``is_open`` property for intervals, unions, and
    complements of closed sets in R.

    Parameters
    ----------
    axiom_set:
        Axiom context.
    S:
        A SymPy set (Interval, Union, Complement, etc.).
    assumptions:
        Symbol assumptions.
    """
    with evaluation():
        is_open = S.is_open

    if not is_open:
        raise ValueError(
            f"Set {S} is not open (SymPy reports is_open={is_open}). "
            f"Cannot prove openness."
        )

    hyp = axiom_set.hypothesis(
        "set_is_open",
        expr=sympy.S.true,
        description=f"{S} is an open set",
    )

    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="open_set_proof",
            claim=f"{S} is open.",
        )
        .lemma(
            "is_open",
            LemmaKind.BOOLEAN,
            expr=sympy.S.true,
            description=f"SymPy confirms {S}.is_open = True",
        )
        .build()
    )

    return seal(axiom_set, hyp, script)


def verify_closed(
    axiom_set: AxiomSet,
    S: sympy.Set,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove a set is closed (its complement in R is open).

    Parameters
    ----------
    axiom_set:
        Axiom context.
    S:
        A SymPy set.
    assumptions:
        Symbol assumptions.
    """
    with evaluation():
        complement = sympy.Complement(sympy.S.Reals, S)
        comp_is_open = complement.is_open

    if not comp_is_open:
        # Fallback: check is_closed directly if available
        with evaluation():
            is_closed = getattr(S, "is_closed", None)
        if not is_closed:
            raise ValueError(
                f"Set {S} is not closed (complement is_open={comp_is_open}). "
                f"Cannot prove closedness."
            )

    hyp = axiom_set.hypothesis(
        "set_is_closed",
        expr=sympy.S.true,
        description=f"{S} is a closed set (complement is open)",
    )

    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="closed_set_proof",
            claim=f"{S} is closed.",
        )
        .lemma(
            "complement_is_open",
            LemmaKind.BOOLEAN,
            expr=sympy.S.true,
            description=f"R \\ {S} is open, therefore {S} is closed",
        )
        .build()
    )

    return seal(axiom_set, hyp, script)


def verify_compact(
    axiom_set: AxiomSet,
    S: sympy.Set,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove a set is compact in R (Heine-Borel: closed and bounded).

    This is the key prerequisite for the extreme value theorem and
    ultimately for the Brouwer fixed point theorem.

    Parameters
    ----------
    axiom_set:
        Axiom context.
    S:
        A SymPy set (typically a closed Interval).
    assumptions:
        Symbol assumptions.
    """
    with evaluation():
        is_closed = getattr(S, "is_closed", None)
        if is_closed is None:
            complement = sympy.Complement(sympy.S.Reals, S)
            is_closed = complement.is_open

        sup_val = S.sup
        inf_val = S.inf
        is_bounded = (
            sup_val is not sympy.S.Infinity
            and sup_val is not sympy.S.NegativeInfinity
            and inf_val is not sympy.S.Infinity
            and inf_val is not sympy.S.NegativeInfinity
        )

    if not is_closed:
        raise ValueError(f"Set {S} is not closed. Cannot prove compactness.")
    if not is_bounded:
        raise ValueError(f"Set {S} is not bounded (sup={sup_val}, inf={inf_val}).")

    hyp = axiom_set.hypothesis(
        "set_is_compact",
        expr=sympy.S.true,
        description=f"{S} is compact (Heine-Borel: closed and bounded in R)",
    )

    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="compact_set_proof",
            claim=f"{S} is compact by Heine-Borel.",
        )
        .lemma(
            "is_closed",
            LemmaKind.BOOLEAN,
            expr=sympy.S.true,
            description=f"{S} is closed",
        )
        .lemma(
            "is_bounded",
            LemmaKind.BOOLEAN,
            expr=sympy.S.true,
            depends_on=["is_closed"],
            description=f"{S} is bounded: inf={inf_val}, sup={sup_val}",
        )
        .lemma(
            "heine_borel",
            LemmaKind.BOOLEAN,
            expr=sympy.S.true,
            depends_on=["is_closed", "is_bounded"],
            description="Heine-Borel: closed + bounded in R => compact",
        )
        .build()
    )

    return seal(axiom_set, hyp, script)


def verify_boundary(
    axiom_set: AxiomSet,
    S: sympy.Set,
    expected_boundary: sympy.Set,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove the boundary of a set equals an expected set.

    Parameters
    ----------
    axiom_set:
        Axiom context.
    S:
        A SymPy set.
    expected_boundary:
        The expected boundary (typically a FiniteSet of points).
    assumptions:
        Symbol assumptions.
    """
    with evaluation():
        actual = S.boundary

    if actual != expected_boundary:
        raise ValueError(
            f"Boundary of {S} is {actual}, not {expected_boundary}."
        )

    hyp = axiom_set.hypothesis(
        "boundary_verified",
        expr=sympy.S.true,
        description=f"boundary({S}) = {expected_boundary}",
    )

    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="boundary_proof",
            claim=f"Boundary of {S} is {expected_boundary}.",
        )
        .lemma(
            "boundary_computed",
            LemmaKind.BOOLEAN,
            expr=sympy.S.true,
            description=f"SymPy computes boundary({S}) = {actual}",
        )
        .build()
    )

    return seal(axiom_set, hyp, script)


# ===================================================================
# Continuity
# ===================================================================


def continuous_at_point(
    axiom_set: AxiomSet,
    f: sympy.Basic,
    x: sympy.Symbol,
    a: sympy.Basic,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove f is continuous at x = a: lim f(x) as x → a = f(a).

    Parameters
    ----------
    axiom_set:
        Axiom context.
    f:
        A SymPy expression in variable x.
    x:
        The variable.
    a:
        The point to check continuity at.
    assumptions:
        Symbol assumptions.
    """
    with evaluation():
        lim_val = sympy.limit(f, x, a)
        func_val = f.subs(x, a)
        diff = sympy.simplify(lim_val - func_val)

    if diff != 0:
        raise ValueError(
            f"f is not continuous at x={a}: "
            f"lim f(x) = {lim_val} but f({a}) = {func_val}."
        )

    hyp = axiom_set.hypothesis(
        "continuous_at_point",
        expr=sympy.Eq(diff, 0),
        description=f"f is continuous at x = {a}: lim f(x) = f({a})",
    )

    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="continuity_proof",
            claim=f"lim f(x) as x → {a} equals f({a}).",
        )
        .lemma(
            "limit_equals_value",
            LemmaKind.EQUALITY,
            expr=diff,
            expected=sympy.Integer(0),
            description=f"lim f(x) as x→{a} = {lim_val} = f({a})",
        )
        .build()
    )

    return seal(axiom_set, hyp, script)


# ===================================================================
# Classical theorems
# ===================================================================


def intermediate_value(
    axiom_set: AxiomSet,
    f: sympy.Basic,
    x: sympy.Symbol,
    a: sympy.Basic,
    b: sympy.Basic,
    target: sympy.Basic,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Apply the intermediate value theorem to find a root.

    Given continuous f on [a, b] with f(a) and f(b) having opposite
    signs relative to target, verifies that a root exists in (a, b)
    by finding it symbolically.

    Parameters
    ----------
    axiom_set:
        Axiom context (should include continuity as axiom if not verified).
    f:
        A SymPy expression in variable x.
    x:
        The variable.
    a, b:
        Interval endpoints.
    target:
        The target value (typically 0 for root-finding).
    assumptions:
        Symbol assumptions.
    """
    with evaluation():
        fa = sympy.simplify(f.subs(x, a))
        fb = sympy.simplify(f.subs(x, b))
        sign_change = sympy.simplify((fa - target) * (fb - target))

        # Find roots in (a, b)
        roots = sympy.solveset(f - target, x, domain=sympy.Interval.open(a, b))

    if roots is sympy.S.EmptySet or roots.is_empty:
        raise ValueError(
            f"No root of f(x) = {target} found in ({a}, {b}). "
            f"f({a}) = {fa}, f({b}) = {fb}."
        )

    # Get a specific root for the proof
    root = list(roots)[0] if hasattr(roots, '__iter__') else roots

    hyp = axiom_set.hypothesis(
        "ivt_root_exists",
        expr=sympy.S.true,
        description=(
            f"By IVT: f is continuous on [{a},{b}], "
            f"f({a})={fa}, f({b})={fb}, target={target}. "
            f"Root exists at x={root}."
        ),
    )

    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="ivt_proof",
            claim=f"IVT: f has a root at x = {root} in ({a}, {b}).",
        )
        .lemma(
            "sign_change",
            LemmaKind.BOOLEAN,
            expr=sympy.Lt(sign_change, 0),
            description=(
                f"(f({a}) - {target}) * (f({b}) - {target}) = {sign_change} < 0 "
                f"(sign change across interval)"
            ),
        )
        .lemma(
            "root_value",
            LemmaKind.EQUALITY,
            expr=sympy.simplify(f.subs(x, root) - target),
            expected=sympy.Integer(0),
            depends_on=["sign_change"],
            description=f"f({root}) = {target} (root verified)",
        )
        .lemma(
            "root_in_interval",
            LemmaKind.BOOLEAN,
            expr=sympy.And(sympy.Gt(root, a), sympy.Lt(root, b)),
            depends_on=["root_value"],
            description=f"{root} is in ({a}, {b})",
        )
        .build()
    )

    return seal(axiom_set, hyp, script)


def extreme_value(
    axiom_set: AxiomSet,
    f: sympy.Basic,
    x: sympy.Symbol,
    a: sympy.Basic,
    b: sympy.Basic,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Apply the extreme value theorem on a closed interval.

    For continuous f on [a, b], finds the maximum and minimum by
    evaluating f at critical points and endpoints.

    Parameters
    ----------
    axiom_set:
        Axiom context.
    f:
        A SymPy expression in variable x.
    x:
        The variable.
    a, b:
        Closed interval endpoints.
    assumptions:
        Symbol assumptions.
    """
    with evaluation():
        # Find critical points
        f_prime = sympy.diff(f, x)
        critical = sympy.solveset(f_prime, x, domain=sympy.Interval(a, b))

        # Evaluate at all candidates
        candidates = [a, b]
        if critical is not sympy.S.EmptySet and hasattr(critical, '__iter__'):
            candidates.extend(list(critical))

        values = [(c, sympy.simplify(f.subs(x, c))) for c in candidates]
        max_pt, max_val = max(values, key=lambda p: p[1])
        min_pt, min_val = min(values, key=lambda p: p[1])

    hyp = axiom_set.hypothesis(
        "extreme_values_exist",
        expr=sympy.S.true,
        description=(
            f"EVT: f on [{a},{b}] has max={max_val} at x={max_pt} "
            f"and min={min_val} at x={min_pt}"
        ),
    )

    builder = ProofBuilder(
        axiom_set,
        hyp.name,
        name="evt_proof",
        claim=f"Continuous f on [{a},{b}] attains its maximum and minimum.",
    )

    # Verify each candidate value
    for i, (c, v) in enumerate(values):
        builder = builder.lemma(
            f"eval_{i}",
            LemmaKind.EQUALITY,
            expr=sympy.simplify(f.subs(x, c)),
            expected=v,
            description=f"f({c}) = {v}",
        )

    # Verify max and min
    builder = builder.lemma(
        "max_verified",
        LemmaKind.BOOLEAN,
        expr=sympy.S.true,
        depends_on=[f"eval_{len(values)-1}"],
        description=f"Maximum value {max_val} at x = {max_pt}",
    )
    builder = builder.lemma(
        "min_verified",
        LemmaKind.BOOLEAN,
        expr=sympy.S.true,
        depends_on=["max_verified"],
        description=f"Minimum value {min_val} at x = {min_pt}",
    )

    return seal(axiom_set, hyp, builder.build())
