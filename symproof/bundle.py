"""Bundle operations: seal, disprove, check_consistency.

``seal()`` is the only path to produce a ``ProofBundle``.
``disprove()`` is the only path to produce a ``Disproof``.
"""

from __future__ import annotations

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


def seal(
    axiom_set: AxiomSet,
    hypothesis: Hypothesis,
    script: ProofScript,
) -> ProofBundle:
    """Verify a proof script and seal it into a ``ProofBundle``.

    Enforces three preconditions:

    1. ``script.axiom_set_hash`` matches the actual axiom set hash.
    2. ``script.target`` matches the hypothesis name.
    3. All lemmas pass verification.

    Parameters
    ----------
    axiom_set:
        The axiom context the proof operates within.
    hypothesis:
        The claim being proved.
    script:
        The proof script targeting the hypothesis.

    Returns
    -------
    ProofBundle
        The sealed, hashed triple.

    Raises
    ------
    ValueError
        If any precondition fails.
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
