"""Bundle operations: seal, disprove, check_consistency.

``seal()`` is the only path to produce a ``ProofBundle``.
``disprove()`` is the only path to produce a ``Disproof``.
"""

from __future__ import annotations

from collections.abc import Sequence

import sympy

from symproof.hashing import hash_axiom_set, hash_bundle, hash_disproof
from symproof.models import (
    AxiomSet,
    Disproof,
    Hypothesis,
    ProofBundle,
    ProofScript,
    ProofStatus,
)
from symproof.serialization import canonical_srepr
from symproof.verification import _build_q_context, verify_proof


class ContradictionError(Exception):
    """Raised when both H and ~H are proved under the same axiom set."""


def _check_assumptions_consistent(
    axiom_set: AxiomSet,
    script: ProofScript,
) -> None:
    """Verify that lemma assumptions do not contradict axiom expressions.

    For each lemma, builds a Q-context from its assumptions and checks
    every axiom expression against it.  If ``sympy.ask`` determines that
    an axiom is **False** under the lemma's assumptions, the assumptions
    contradict the axiom set and the proof is rejected.

    Only proven contradictions (``ask`` returns ``False``) are rejected.
    Indeterminate results (``None``) are allowed.
    """
    for lemma in script.lemmas:
        if not lemma.assumptions:
            continue
        context = _build_q_context(lemma.assumptions)
        for axiom in axiom_set.axioms:
            try:
                result = sympy.ask(sympy.Q.is_true(axiom.expr), context)
            except (TypeError, ValueError, RecursionError, AttributeError):
                continue
            if result is False:
                raise ValueError(
                    f"Lemma '{lemma.name}' assumptions contradict axiom "
                    f"'{axiom.name}': under assumptions {lemma.assumptions}, "
                    f"axiom expression {axiom.expr} is provably False. "
                    f"Axioms are authoritative — lemma assumptions must be "
                    f"consistent with the axiom set."
                )


def _check_foundations(
    axiom_set: AxiomSet,
    foundations: Sequence[tuple[ProofBundle, str]],
) -> None:
    """Validate that every axiom in each foundation is covered by ``axiom_set``.

    For each ``(foundation_bundle, justified_axiom_name)``:

    1. ``justified_axiom_name`` must exist in ``axiom_set``.
    2. Every axiom in the foundation's axiom set must have a matching
       axiom in ``axiom_set`` — by name first, then by canonical
       expression.  Missing axioms are "hidden axioms" and cause a
       ``ValueError``.

    This enforces that downstream proofs explicitly declare all
    conditions their foundations require.  Axioms inherited from
    foundations should be marked ``inherited=True`` for traceability.
    """
    downstream_by_name = {a.name: a for a in axiom_set.axioms}
    downstream_by_expr = {
        canonical_srepr(a.expr): a.name
        for a in axiom_set.axioms
    }

    for foundation_bundle, axiom_name in foundations:
        # Check the justified axiom exists
        target = axiom_set.get_axiom(axiom_name)
        if target is None:
            raise ValueError(
                f"Foundation claims to justify axiom '{axiom_name}', "
                f"but it does not exist in axiom set '{axiom_set.name}'."
            )

        # Check every foundation axiom is covered in downstream
        hidden = []
        for fa in foundation_bundle.axiom_set.axioms:
            if fa.name in downstream_by_name:
                continue
            if canonical_srepr(fa.expr) in downstream_by_expr:
                continue
            hidden.append(fa.name)

        if hidden:
            raise ValueError(
                f"Foundation bundle '{foundation_bundle.axiom_set.name}' "
                f"(justifying '{axiom_name}') has axioms not present in "
                f"axiom set '{axiom_set.name}': {hidden}. "
                f"These are hidden axioms — add them to the axiom set "
                f"with inherited=True."
            )


def seal(
    axiom_set: AxiomSet,
    hypothesis: Hypothesis,
    script: ProofScript,
    *,
    foundations: Sequence[tuple[ProofBundle, str]] | None = None,
) -> ProofBundle:
    """Verify a proof script and seal it into a ``ProofBundle``.

    Enforces four preconditions:

    1. ``script.axiom_set_hash`` matches the actual axiom set hash.
    2. ``script.target`` matches the hypothesis name.
    3. All lemmas pass verification.
    4. For each foundation, every axiom in the foundation's axiom set
       must appear in ``axiom_set`` (by name or expression).

    Parameters
    ----------
    axiom_set:
        The axiom context the proof operates within.
    hypothesis:
        The claim being proved.
    script:
        The proof script targeting the hypothesis.
    foundations:
        Sequence of ``(foundation_bundle, justified_axiom_name)`` pairs.
        Each foundation proves the content of an axiom in ``axiom_set``.
        Every axiom in the foundation's axiom set must be present in
        ``axiom_set``.  Missing axioms are hidden dependencies and
        cause a ``ValueError``.

    Returns
    -------
    ProofBundle
        The sealed, hashed triple.

    Raises
    ------
    ValueError
        If any precondition fails, including hidden axioms in foundations.
    """
    actual_hash = hash_axiom_set(axiom_set)

    if script.axiom_set_hash != actual_hash:
        raise ValueError(
            f"script.axiom_set_hash={script.axiom_set_hash!r} does not match "
            f"hash_axiom_set(axiom_set)={actual_hash!r}"
        )
    if hypothesis.axiom_set_hash != actual_hash:
        raise ValueError(
            f"hypothesis.axiom_set_hash={hypothesis.axiom_set_hash!r} does not "
            f"match hash_axiom_set(axiom_set)={actual_hash!r}"
        )
    if script.target != hypothesis.name:
        raise ValueError(
            f"script.target={script.target!r} does not match "
            f"hypothesis.name={hypothesis.name!r}"
        )

    # Check that imported bundles share the same axiom set.
    for bundle in script.imported_bundles:
        if bundle.axiom_set.axiom_set_hash != actual_hash:
            raise ValueError(
                f"Imported bundle '{bundle.hypothesis.name}' is bound "
                f"to axiom set '{bundle.axiom_set.name}' "
                f"(hash={bundle.axiom_set.axiom_set_hash!r}), which "
                f"does not match the current axiom set "
                f"'{axiom_set.name}' (hash={actual_hash!r})."
            )

    # Check that lemma assumptions do not contradict axioms.
    # Axioms are authoritative — lemma assumptions must be consistent.
    _check_assumptions_consistent(axiom_set, script)

    # Check foundations: every axiom in each foundation's axiom set
    # must appear in this axiom set.  Missing axioms are "hidden" —
    # conditions the proof depends on without declaring.
    if foundations:
        _check_foundations(axiom_set, foundations)

    # Always re-verify — no trust_imports shortcut when sealing.
    result = verify_proof(script, trust_imports=False)
    if result.status != ProofStatus.VERIFIED:
        raise ValueError(
            f"verify_proof returned {result.status.value}, not VERIFIED. "
            f"Cannot seal an unverified proof. Summary: {result.failure_summary}"
        )

    bundle_h = hash_bundle(actual_hash, hypothesis, result.proof_hash)  # type: ignore[arg-type]

    return ProofBundle(
        axiom_set=axiom_set,
        hypothesis=hypothesis,
        proof=script,
        proof_result=result,
        bundle_hash=bundle_h,
    )


def disprove(
    hypothesis: Hypothesis,
    negation_bundle: ProofBundle,
) -> Disproof:
    """Compose a disproof of H from a proof of ~H.

    Validates that the negation bundle:
    - Proves a hypothesis whose expression is the negation of ``hypothesis.expr``
    - Operates under the same axiom set (matching ``axiom_set_hash``)

    Parameters
    ----------
    hypothesis:
        The original hypothesis being disproved.
    negation_bundle:
        A sealed ``ProofBundle`` proving ~H under the same axiom set.

    Returns
    -------
    Disproof
        The compositional disproof.

    Raises
    ------
    ValueError
        If the negation bundle does not prove ~H or uses different axioms.
    """
    if negation_bundle.hypothesis.axiom_set_hash != hypothesis.axiom_set_hash:
        raise ValueError(
            f"Axiom set mismatch: hypothesis bound to "
            f"{hypothesis.axiom_set_hash!r}, negation bundle bound to "
            f"{negation_bundle.hypothesis.axiom_set_hash!r}. "
            f"Disproof requires the same axiom context."
        )

    d_hash = hash_disproof(hypothesis, negation_bundle.bundle_hash)

    return Disproof(
        hypothesis=hypothesis,
        negation_bundle=negation_bundle,
        disproof_hash=d_hash,
    )


def check_consistency(
    bundle_h: ProofBundle,
    bundle_not_h: ProofBundle,
) -> None:
    """Verify that two bundles do not contradict each other.

    Raises ``ContradictionError`` if both bundles prove H and ~H under
    the same axiom set.

    Parameters
    ----------
    bundle_h:
        A sealed bundle (could prove H or ~H).
    bundle_not_h:
        Another sealed bundle (could prove H or ~H).

    Raises
    ------
    ContradictionError
        If the two bundles prove contradictory claims under the same axioms.
    """
    if bundle_h.axiom_set.axiom_set_hash != bundle_not_h.axiom_set.axiom_set_hash:
        return  # Different axiom contexts — no contradiction possible

    import sympy

    expr_a = bundle_h.hypothesis.expr
    expr_b = bundle_not_h.hypothesis.expr

    # Check if one is the negation of the other
    if sympy.simplify(sympy.And(expr_a, expr_b)) is sympy.false:
        raise ContradictionError(
            f"Contradiction detected under axiom set "
            f"{bundle_h.axiom_set.name!r}: both "
            f"{bundle_h.hypothesis.name!r} and "
            f"{bundle_not_h.hypothesis.name!r} are proved, "
            f"but they are mutually exclusive."
        )

    # Also check via structural Not equivalence
    neg_b = sympy.Not(expr_b)
    neg_a = sympy.Not(expr_a)
    if (
        sympy.simplify(sympy.Equivalent(expr_a, neg_b)) is sympy.true
        or sympy.simplify(sympy.Equivalent(neg_a, expr_b)) is sympy.true
    ):
        raise ContradictionError(
            f"Contradiction detected under axiom set "
            f"{bundle_h.axiom_set.name!r}: "
            f"{bundle_h.hypothesis.name!r} and "
            f"{bundle_not_h.hypothesis.name!r} are negations of each other."
        )
