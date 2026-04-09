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
