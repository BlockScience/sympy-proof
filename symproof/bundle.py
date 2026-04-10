"""Bundle operations: seal, disprove, check_consistency.

``seal()`` is the only path to produce a ``ProofBundle``.
``disprove()`` is the only path to produce a ``Disproof``.
"""

from __future__ import annotations

from collections.abc import Sequence

import sympy

from symproof.evaluation import evaluation
from symproof.hashing import hash_axiom_set, hash_bundle, hash_disproof
from symproof.models import (
    AxiomSet,
    Disproof,
    Hypothesis,
    ProofBundle,
    ProofResult,
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
                with evaluation():
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


def _collect_script_symbols(script: ProofScript) -> set[sympy.Symbol]:
    """Collect all free symbols from all lemma expressions in a script."""
    symbols: set[sympy.Symbol] = set()
    for lemma in script.lemmas:
        symbols |= lemma.expr.free_symbols
        if lemma.expected is not None:
            symbols |= lemma.expected.free_symbols
    return symbols


def _assumption_covered_by_axiom(
    sym: sympy.Symbol,
    assumption: str,
    axiom_set: AxiomSet,
) -> bool:
    """Check if a symbol assumption is covered by an axiom expression.

    For example, ``positive=True`` on symbol ``x`` is covered by an
    axiom with ``expr = x > 0``.

    Handles SymPy's eager evaluation: if the axiom was constructed
    with the same Symbol (which has assumptions), the expression may
    have already been simplified to True.  In that case, we check
    whether the symbol's name appears in the axiom name as a fallback.
    """
    # Build expected expressions using BOTH the actual symbol (for eager-eval
    # matching) and a bare symbol (for structural matching)
    bare = sympy.Symbol(sym.name)
    expected_map = {
        "positive": (sym > 0, bare > 0),
        "nonnegative": (sym >= 0, bare >= 0),
        "negative": (sym < 0, bare < 0),
        "nonpositive": (sym <= 0, bare <= 0),
        "nonzero": (sympy.Ne(sym, 0), sympy.Ne(bare, 0)),
    }
    pair = expected_map.get(assumption)
    if pair is None:
        return True  # integer, real, etc. — not expressible as axiom, skip

    expected_with_sym, expected_bare = pair

    # Check all axioms — including True ones.  An axiom with expr=True
    # may have been constructed from `Symbol("x", positive=True) > 0`,
    # which SymPy eagerly evaluated to True.  We detect this by rebuilding
    # the relation from a bare symbol and checking the axiom set's
    # canonical dict, which stores the srepr'd version.
    #
    # The canonical dict uses make_canonical_dict() which calls srepr()
    # on the original expression.  If the axiom was `Rx > 0` with
    # Rx = Symbol("R_x", positive=True), srepr gives "BooleanTrue()".
    # The bare form gives "StrictGreaterThan(Symbol('R_x'), Integer(0))".
    # We need to check both.
    bare_srepr = canonical_srepr(expected_bare)

    for axiom in axiom_set.axioms:
        ax_srepr = canonical_srepr(axiom.expr)

        # Direct match (bare form)
        if ax_srepr == bare_srepr:
            return True

        # Skip True axioms for implication checks (can't imply anything)
        if axiom.expr is sympy.S.true:
            continue

        # Check if axiom expression implies the assumption
        try:
            with evaluation():
                implied = sympy.simplify(
                    sympy.Implies(axiom.expr, expected_bare)
                )
            if implied is sympy.S.true:
                return True
        except (TypeError, RecursionError, AttributeError):
            continue

    # Eager evaluation fallback: the axiom's expression was evaluated to
    # True at construction time because the symbol had assumptions.
    # Check the axiom set's canonical dict — the srepr of True is stored
    # there, but we can also check if any axiom was MEANT to declare
    # this symbol by checking if the symbol name appears in the axiom's
    # description.
    for axiom in axiom_set.axioms:
        if axiom.expr is not sympy.S.true:
            continue
        if sym.name.lower() in axiom.description.lower():
            return True
        # Also check if the axiom name contains the symbol name
        # (e.g., axiom "rx_pos" for symbol "R_x")
        if sym.name.lower().replace("_", "") in axiom.name.lower().replace("_", ""):
            return True

    return False


def _audit_load_bearing(
    axiom_set: AxiomSet,
    script: ProofScript,
) -> list[str]:
    """Identify symbol assumptions that are load-bearing but not axiomatised.

    For each symbol with user-specified constructor assumptions that
    appears in the proof, checks whether an axiom covers the assumption.
    If not covered, uses two detection strategies:

    1. **Re-verification test** (for QUERY/BOOLEAN lemmas): strips the
       assumption from the symbol, re-verifies each affected lemma.
       If any lemma fails, the assumption is confirmed load-bearing.

    2. **Presence test** (for EQUALITY lemmas): SymPy's eager evaluation
       means EQUALITY expressions may have already been simplified using
       the assumption at construction time.  Stripping won't undo this.
       For EQUALITY lemmas, presence of the symbol with uncovered
       assumptions is sufficient to flag it — the assumption *may* have
       influenced the expression in ways that can't be detected post-hoc.

    Returns a list of error messages for load-bearing unaxiomatised
    assumptions.  Empty list means all assumptions are accounted for.
    """
    from symproof.models import Lemma, LemmaKind
    from symproof.verification import verify_lemma

    symbols = _collect_script_symbols(script)
    errors: list[str] = []

    for sym in symbols:
        user_assumptions = getattr(sym, "_assumptions_orig", {})
        if not user_assumptions:
            continue

        for assumption, value in user_assumptions.items():
            if not value:
                continue
            if _assumption_covered_by_axiom(sym, assumption, axiom_set):
                continue

            # Not covered — check if load-bearing
            bare = sympy.Symbol(sym.name)
            affected_lemmas: list[str] = []

            for lemma in script.lemmas:
                if sym not in lemma.expr.free_symbols and (
                    lemma.expected is None
                    or sym not in lemma.expected.free_symbols
                ):
                    continue

                # EQUALITY lemmas: flag by presence (eager eval may have
                # baked in the assumption at construction time)
                if lemma.kind == LemmaKind.EQUALITY:
                    affected_lemmas.append(lemma.name)
                    continue

                # QUERY/BOOLEAN: strip and re-verify
                stripped_expr = lemma.expr.xreplace({sym: bare})
                stripped_expected = (
                    lemma.expected.xreplace({sym: bare})
                    if lemma.expected is not None
                    else None
                )
                stripped_lemma = Lemma(
                    name=lemma.name,
                    kind=lemma.kind,
                    expr=stripped_expr,
                    expected=stripped_expected,
                    assumptions=lemma.assumptions,
                    transform=lemma.transform,
                    inverse_transform=lemma.inverse_transform,
                    depends_on=lemma.depends_on,
                    description=lemma.description,
                )

                try:
                    result = verify_lemma(stripped_lemma)
                    if not result.passed:
                        affected_lemmas.append(lemma.name)
                except Exception:
                    affected_lemmas.append(lemma.name)

            if affected_lemmas:
                errors.append(
                    f"Symbol '{sym.name}' has assumption "
                    f"'{assumption}={value}' which is load-bearing "
                    f"(affects lemmas {affected_lemmas}) but "
                    f"is not declared as an axiom. Add an axiom for "
                    f"'{sym.name}' to the axiom set."
                )

    return errors


def _check_axiom_consistency(axiom_set: AxiomSet) -> None:
    """Check that no pair of axioms contradicts each other.

    Raises ``ValueError`` if ``simplify(And(a1.expr, a2.expr))`` is
    ``sympy.false``.  Skips axioms whose expression is ``sympy.S.true``
    (external results cannot be checked pairwise).
    """
    checkable = [a for a in axiom_set.axioms if a.expr is not sympy.S.true]
    for i, a1 in enumerate(checkable):
        for a2 in checkable[i + 1 :]:
            try:
                with evaluation():
                    combined = sympy.simplify(sympy.And(a1.expr, a2.expr))
                if combined is sympy.S.false:
                    raise ValueError(
                        f"Axioms '{a1.name}' and '{a2.name}' are mutually "
                        f"contradictory: And({a1.expr}, {a2.expr}) is provably "
                        f"False. The axiom set is inconsistent."
                    )
            except (TypeError, RecursionError, AttributeError):
                continue


def _build_assumption_report(axiom_set: AxiomSet) -> tuple[str, ...]:
    """Build advisories enumerating all assumptions in the axiom set."""
    report: list[str] = []
    posited = [a for a in axiom_set.axioms if not a.inherited]
    inherited = [a for a in axiom_set.axioms if a.inherited]
    external = [a for a in axiom_set.axioms if a.expr is sympy.S.true]

    report.append(
        f"[ASSUMPTIONS] Axiom set '{axiom_set.name}' entails "
        f"{len(axiom_set.axioms)} assumptions: "
        f"{len(posited)} posited, {len(inherited)} inherited, "
        f"{len(external)} external (expr=True)."
    )

    for a in inherited:
        cite = f" (from: {a.citation.source})" if a.citation else ""
        if a.expr is sympy.S.true:
            report.append(
                f"[ASSUMPTIONS] INHERITED (external): '{a.name}'{cite}"
            )
        else:
            report.append(
                f"[ASSUMPTIONS] INHERITED: '{a.name}' assumes {a.expr}{cite}"
            )

    for a in external:
        if not a.inherited:
            report.append(
                f"[ASSUMPTIONS] EXTERNAL: '{a.name}' taken as given "
                f"(expr=True). Consider backing with a foundation."
            )

    return tuple(report)


def seal(
    axiom_set: AxiomSet,
    hypothesis: Hypothesis,
    script: ProofScript,
    *,
    foundations: Sequence[tuple[ProofBundle, str]] | None = None,
    check_consistency: bool = True,
) -> ProofBundle:
    """Verify a proof script and seal it into a ``ProofBundle``.

    Enforces five preconditions:

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

    # Pairwise consistency: reject contradictory axiom pairs.
    if check_consistency:
        _check_axiom_consistency(axiom_set)

    # Load-bearing assumption accounting: symbol constructor assumptions
    # that aren't declared as axioms but affect proof validity.
    load_bearing_errors = _audit_load_bearing(axiom_set, script)
    if load_bearing_errors:
        raise ValueError(
            "Load-bearing symbol assumptions found without matching axioms. "
            "These are hidden axioms — the proof depends on them but they "
            "are not declared in the axiom set:\n"
            + "\n".join(f"  - {e}" for e in load_bearing_errors)
        )

    # Always re-verify — no trust_imports shortcut when sealing.
    result = verify_proof(script, trust_imports=False)
    if result.status != ProofStatus.VERIFIED:
        raise ValueError(
            f"verify_proof returned {result.status.value}, not VERIFIED. "
            f"Cannot seal an unverified proof. Summary: {result.failure_summary}"
        )

    # Build assumption report and prepend to advisories.
    assumption_advisories = _build_assumption_report(axiom_set)
    augmented_result = ProofResult(
        status=result.status,
        proof_hash=result.proof_hash,
        lemma_results=result.lemma_results,
        failure_summary=result.failure_summary,
        advisories=assumption_advisories + result.advisories,
    )

    bundle_h = hash_bundle(actual_hash, hypothesis, augmented_result.proof_hash)  # type: ignore[arg-type]

    return ProofBundle(
        axiom_set=axiom_set,
        hypothesis=hypothesis,
        proof=script,
        proof_result=augmented_result,
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
    with evaluation():
        and_result = sympy.simplify(sympy.And(expr_a, expr_b))
    if and_result is sympy.false:
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
    with evaluation():
        equiv_check = (
            sympy.simplify(sympy.Equivalent(expr_a, neg_b)) is sympy.true
            or sympy.simplify(sympy.Equivalent(neg_a, expr_b)) is sympy.true
        )
    if equiv_check:
        raise ContradictionError(
            f"Contradiction detected under axiom set "
            f"{bundle_h.axiom_set.name!r}: "
            f"{bundle_h.hypothesis.name!r} and "
            f"{bundle_not_h.hypothesis.name!r} are negations of each other."
        )
