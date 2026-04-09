"""Proof verification engine.

Three-way dispatch on ``LemmaKind``:

- ``EQUALITY``  — ``simplify(expr - expected) == 0`` with ``.doit()`` fallback
- ``BOOLEAN``   — ``simplify(expr) is sympy.true``
- ``QUERY``     — ``sympy.ask(expr, context)`` returns ``True``

``verify_proof`` checks mathematical validity only (do the lemmas hold?).
Context binding (does ``axiom_set_hash`` match the actual axiom set?) is
enforced in ``seal()``.
"""

from __future__ import annotations

import sympy

from symproof.hashing import hash_proof
from symproof.models import (
    Lemma,
    LemmaKind,
    LemmaResult,
    ProofResult,
    ProofScript,
    ProofStatus,
)

# ---------------------------------------------------------------------------
# Assumption helpers
# ---------------------------------------------------------------------------


def _make_assumed_symbol(name: str, assumptions: dict) -> sympy.Symbol:
    """Create a SymPy Symbol with the given assumption kwargs."""
    return sympy.Symbol(name, **assumptions)


def _build_assumption_subs(
    assumptions: dict[str, dict],
) -> dict[sympy.Symbol, sympy.Symbol]:
    """Map plain symbols to assumption-enhanced symbols for simplification."""
    return {
        sympy.Symbol(name): _make_assumed_symbol(name, asm)
        for name, asm in assumptions.items()
    }


def _build_q_context(assumptions: dict[str, dict]) -> sympy.Basic:
    """Build a SymPy Q assumption context from the assumptions dict."""
    facts = []
    for name, asm in assumptions.items():
        sym = sympy.Symbol(name)
        for prop, val in asm.items():
            if val:
                q_pred = getattr(sympy.Q, prop, None)
                if q_pred is not None:
                    facts.append(q_pred(sym))
    if not facts:
        return sympy.Q.is_true(sympy.true)  # type: ignore[attr-defined]
    return sympy.And(*facts)


# ---------------------------------------------------------------------------
# Lemma verification
# ---------------------------------------------------------------------------


def verify_lemma(lemma: Lemma) -> LemmaResult:
    """Verify a single lemma independently.

    Dispatches on ``lemma.kind``:

    EQUALITY
        ``sympy.simplify(expr - expected) == 0``, falling back to
        ``expr.doit() == expected`` for sum/product expressions.
    BOOLEAN
        ``sympy.simplify(expr) is sympy.true``.
    QUERY
        ``sympy.ask(expr, context)`` is ``True`` under stated assumptions.
    """
    try:
        assumption_subs = _build_assumption_subs(lemma.assumptions)
        expr_with_assumptions = lemma.expr.subs(assumption_subs)

        if lemma.kind == LemmaKind.EQUALITY:
            if lemma.expected is None:
                return LemmaResult(
                    lemma_name=lemma.name,
                    passed=False,
                    error="EQUALITY lemma requires 'expected' to be set",
                )
            expected_with_assumptions = lemma.expected.subs(assumption_subs)
            diff = sympy.simplify(expr_with_assumptions - expected_with_assumptions)
            if diff == sympy.Integer(0):
                return LemmaResult(
                    lemma_name=lemma.name, passed=True, actual_value=sympy.Integer(0)
                )
            # Fallback: try .doit() for Sum/Product expressions
            evaluated = expr_with_assumptions.doit()
            diff = evaluated - expected_with_assumptions
            if sympy.simplify(diff) == sympy.Integer(0):
                return LemmaResult(
                    lemma_name=lemma.name, passed=True, actual_value=evaluated
                )
            return LemmaResult(
                lemma_name=lemma.name,
                passed=False,
                actual_value=diff,
                error=f"simplify(expr - expected) = {diff}, not 0",
            )

        if lemma.kind == LemmaKind.BOOLEAN:
            result = sympy.simplify(expr_with_assumptions)
            passed = result is sympy.true

            if not passed and not isinstance(
                expr_with_assumptions, sympy.assumptions.assume.AppliedPredicate
            ):
                # Fallback 1: refine() handles implications and relational
                # reasoning that simplify() misses (e.g., transitivity of <=).
                # Skip for AppliedPredicate (Q-system) — those belong in QUERY.
                try:
                    refined = sympy.refine(expr_with_assumptions)
                    if refined is sympy.true:
                        passed = True
                        result = refined
                except (TypeError, ValueError, RecursionError, AttributeError):
                    pass

            if not passed and isinstance(expr_with_assumptions, sympy.Implies):
                # Fallback 2: proof by contradiction — Implies(P, Q) holds
                # iff And(P, Not(Q)) is unsatisfiable.
                try:
                    ante = expr_with_assumptions.args[0]
                    cons = expr_with_assumptions.args[1]
                    negated = sympy.simplify(sympy.And(ante, sympy.Not(cons)))
                    if negated is sympy.false:
                        passed = True
                        result = sympy.true
                except (TypeError, ValueError, RecursionError, AttributeError):
                    pass

            return LemmaResult(
                lemma_name=lemma.name, passed=passed, actual_value=result
            )

        if lemma.kind == LemmaKind.QUERY:
            context = _build_q_context(lemma.assumptions)
            ask_result = sympy.ask(expr_with_assumptions, context)
            passed = ask_result is True
            return LemmaResult(
                lemma_name=lemma.name,
                passed=passed,
                actual_value=sympy.true if passed else sympy.false,
            )

        if lemma.kind == LemmaKind.COORDINATE_TRANSFORM:
            if lemma.transform is None or lemma.inverse_transform is None:
                return LemmaResult(
                    lemma_name=lemma.name,
                    passed=False,
                    error="COORDINATE_TRANSFORM requires both 'transform' "
                    "and 'inverse_transform'",
                )
            if lemma.expected is None:
                return LemmaResult(
                    lemma_name=lemma.name,
                    passed=False,
                    error="COORDINATE_TRANSFORM requires 'expected' to be set",
                )

            # Collect all free symbols from transform expressions so we
            # match by *name*, respecting any SymPy assumptions already
            # attached to the symbols the caller used.
            all_fwd_syms = {}
            for fwd_expr in lemma.transform.values():
                for sym in fwd_expr.free_symbols:
                    all_fwd_syms[sym.name] = sym
            all_inv_syms = {}
            for inv_expr in lemma.inverse_transform.values():
                for sym in inv_expr.free_symbols:
                    all_inv_syms[sym.name] = sym

            # Build forward substitution: old_symbol → new_expr
            forward_subs = {}
            for name, fwd_expr in lemma.transform.items():
                # Find the actual symbol (with assumptions) from expr or
                # inverse_transform values; fall back to bare Symbol.
                orig_sym = all_inv_syms.get(name) or sympy.Symbol(name)
                forward_subs[orig_sym] = fwd_expr

            # Build inverse substitution: new_symbol → old_expr
            inverse_subs = {}
            for name, inv_expr in lemma.inverse_transform.items():
                new_sym = all_fwd_syms.get(name) or sympy.Symbol(name)
                inverse_subs[new_sym] = inv_expr

            # Step 1: verify round-trip  inverse(forward(s)) == s
            for orig_name, fwd_expr in lemma.transform.items():
                orig_sym = all_inv_syms.get(orig_name) or sympy.Symbol(orig_name)
                composed = fwd_expr.subs(inverse_subs)
                diff_rt = sympy.simplify(composed - orig_sym)
                if diff_rt != sympy.Integer(0):
                    # Fallback: trigsimp for polar/spherical round-trips
                    diff_rt = sympy.trigsimp(composed - orig_sym)
                if diff_rt != sympy.Integer(0):
                    # Fallback: powsimp + radsimp for sqrt-based round-trips
                    diff_rt = sympy.simplify(
                        sympy.powsimp(composed, force=True) - orig_sym
                    )
                if diff_rt != sympy.Integer(0):
                    return LemmaResult(
                        lemma_name=lemma.name,
                        passed=False,
                        actual_value=diff_rt,
                        error=f"Round-trip failed for {orig_name}: "
                        f"inverse(forward({orig_name})) - {orig_name} "
                        f"= {diff_rt}, not 0",
                    )

            # Step 2: apply forward transform to expr, simplify in new coords
            transformed_expr = expr_with_assumptions.subs(forward_subs)
            expected_with_assumptions = lemma.expected.subs(assumption_subs)
            transformed_expected = expected_with_assumptions.subs(forward_subs)

            diff = sympy.simplify(transformed_expr - transformed_expected)
            if diff == sympy.Integer(0):
                return LemmaResult(
                    lemma_name=lemma.name,
                    passed=True,
                    actual_value=sympy.Integer(0),
                )

            # Fallback: trigsimp for polar/spherical transforms
            diff = sympy.trigsimp(transformed_expr - transformed_expected)
            if diff == sympy.Integer(0):
                return LemmaResult(
                    lemma_name=lemma.name,
                    passed=True,
                    actual_value=sympy.Integer(0),
                )

            return LemmaResult(
                lemma_name=lemma.name,
                passed=False,
                actual_value=diff,
                error=f"Transformed simplify(expr - expected) = {diff}, not 0",
            )

        return LemmaResult(  # pragma: no cover
            lemma_name=lemma.name,
            passed=False,
            error=f"Unknown LemmaKind: {lemma.kind}",
        )

    except (TypeError, ValueError, RecursionError, AttributeError) as exc:
        return LemmaResult(lemma_name=lemma.name, passed=False, error=str(exc))


# ---------------------------------------------------------------------------
# Proof script verification
# ---------------------------------------------------------------------------


def verify_proof(script: ProofScript) -> ProofResult:
    """Verify a complete proof script.

    Each lemma is verified independently.  A single failing lemma fails
    the entire proof.  This function checks mathematical validity only —
    context binding is enforced in ``seal()``.
    """
    lemma_results: list[LemmaResult] = []
    for lemma in script.lemmas:
        result = verify_lemma(lemma)
        lemma_results.append(result)
        if not result.passed:
            return ProofResult(
                status=ProofStatus.FAILED,
                proof_hash=hash_proof(script),
                lemma_results=tuple(lemma_results),
                failure_summary=f"Lemma '{lemma.name}' failed: {result.error}",
            )

    return ProofResult(
        status=ProofStatus.VERIFIED,
        proof_hash=hash_proof(script),
        lemma_results=tuple(lemma_results),
    )
