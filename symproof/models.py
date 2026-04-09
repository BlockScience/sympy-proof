"""Core data models for symproof.

All models are frozen Pydantic BaseModels with ``arbitrary_types_allowed``
for SymPy expression fields.

Key design principle: **no hypothesis without axioms**.  A ``Hypothesis``
is always bound to an ``AxiomSet`` via ``axiom_set_hash``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

import sympy
from pydantic import BaseModel, ConfigDict, model_validator

from symproof.serialization import make_canonical_dict
from symproof.types import SympyBoolean, SympyExpr  # noqa: TC001

# ---------------------------------------------------------------------------
# Axiom
# ---------------------------------------------------------------------------


class Axiom(BaseModel):
    """An accepted truth — a SymPy boolean expression taken as given."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    name: str
    expr: SympyBoolean
    description: str = ""
    inherited: bool = False
    """True if this axiom was inherited from a foundation proof rather
    than posited directly.  Inherited axioms represent conditions that
    the proof chain forced — they were not chosen by the proof author
    but are required by an external theorem the proof depends on."""


# ---------------------------------------------------------------------------
# AxiomSet
# ---------------------------------------------------------------------------


class AxiomSet(BaseModel):
    """Named, immutable collection of axioms.

    The hashable "context" all reasoning operates within.  Every hypothesis
    and proof script is bound to an axiom set via its hash.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    name: str
    axioms: tuple[Axiom, ...]

    @model_validator(mode="after")
    def _unique_axiom_names(self) -> Self:
        names = [a.name for a in self.axioms]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"Duplicate axiom names in AxiomSet: {dupes}")
        return self

    def canonical_dict(self) -> dict:
        """Sorted, srepr'd canonical form for hashing."""
        return make_canonical_dict(
            {
                "name": self.name,
                "axioms": [
                    {
                        "name": a.name,
                        "expr": a.expr,
                        "description": a.description,
                        "inherited": a.inherited,
                    }
                    for a in self.axioms
                ],
            }
        )

    def true_axioms(self) -> tuple[Axiom, ...]:
        """Return axioms whose expression is ``sympy.S.true`` (external results)."""
        return tuple(a for a in self.axioms if a.expr is sympy.S.true)

    def get_axiom(self, name: str) -> Axiom | None:
        """Look up an axiom by name, or return None."""
        for a in self.axioms:
            if a.name == name:
                return a
        return None

    @property
    def axiom_set_hash(self) -> str:
        """SHA-256 hex digest of the canonical form."""
        from symproof.hashing import hash_axiom_set

        return hash_axiom_set(self)

    def hypothesis(
        self,
        name: str,
        expr: SympyBoolean,
        description: str = "",
    ) -> Hypothesis:
        """Create a hypothesis bound to this axiom set."""
        return Hypothesis(
            name=name,
            expr=expr,
            axiom_set_hash=self.axiom_set_hash,
            description=description,
        )


# ---------------------------------------------------------------------------
# Hypothesis
# ---------------------------------------------------------------------------


class Hypothesis(BaseModel):
    """A claim to prove or disprove, always bound to an axiom set.

    The axiom set gives meaning to the symbols and provides the reasoning
    context.  A hypothesis cannot be constructed without ``axiom_set_hash``.

    The preferred construction path is ``axiom_set.hypothesis(...)``.
    Direct construction is supported for deserialization.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    name: str
    expr: SympyBoolean
    axiom_set_hash: str
    description: str = ""

    def negate(self) -> Hypothesis:
        """Return a new hypothesis with the negated expression, same axiom binding."""
        return Hypothesis(
            name=f"not_{self.name}",
            expr=sympy.Not(self.expr),
            axiom_set_hash=self.axiom_set_hash,
            description=f"Negation of: {self.description}" if self.description else "",
        )


# ---------------------------------------------------------------------------
# Lemma types
# ---------------------------------------------------------------------------


class LemmaKind(StrEnum):
    """The four verification strategies SymPy handles reliably."""

    EQUALITY = "equality"
    """``simplify(expr - expected) == 0`` or ``expr.doit() == expected``."""

    BOOLEAN = "boolean"
    """``simplify(expr)`` is ``sympy.true``."""

    QUERY = "query"
    """``sympy.ask(expr, assumption_context)`` is ``True``."""

    COORDINATE_TRANSFORM = "coordinate_transform"
    """Transform → prove in new coordinates → verify round-trip.

    Requires ``transform`` (forward map) and ``inverse_transform`` (back map).
    Verification:
    1. Round-trip: ``inverse(forward(s)) == s`` for each transformed symbol.
    2. Apply forward transform to ``expr``.
    3. ``simplify(transformed_expr - expected) == 0`` in new coordinates.
    """


class Lemma(BaseModel):
    """A single verifiable SymPy claim within a proof script."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    name: str
    kind: LemmaKind
    expr: SympyExpr
    expected: SympyExpr | None = None
    """Required for EQUALITY lemmas; the value ``expr`` must equal."""

    assumptions: dict[str, dict] = {}
    """symbol_name -> SymPy assumption kwargs, e.g. ``{"x": {"positive": True}}``."""

    transform: dict[str, SympyExpr] | None = None
    """Forward coordinate map: old_symbol_name -> new_expr.

    Required for ``COORDINATE_TRANSFORM`` lemmas.  E.g. for Cartesian→polar:
    ``{"x_1": r*cos(theta), "x_2": r*sin(theta)}``.
    """

    inverse_transform: dict[str, SympyExpr] | None = None
    """Inverse coordinate map: new_symbol_name -> old_expr.

    Required for ``COORDINATE_TRANSFORM`` lemmas.  E.g. for polar→Cartesian:
    ``{"r": sqrt(x_1**2 + x_2**2), "theta": atan2(x_2, x_1)}``.
    """

    depends_on: list[str] = []
    """Names of prior lemmas whose results this lemma logically depends on."""

    description: str = ""


# ---------------------------------------------------------------------------
# Proof status & results
# ---------------------------------------------------------------------------


class ProofStatus(StrEnum):
    """Overall proof verification status."""

    UNCHECKED = "UNCHECKED"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"


class LemmaResult(BaseModel):
    """Verification output for a single lemma."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    lemma_name: str
    passed: bool
    actual_value: SympyExpr | None = None
    error: str | None = None
    advisories: tuple[str, ...] = ()
    """Warnings about known SymPy limitations that apply to this result.

    Present on both passing and failing lemmas.  A non-empty list signals
    that a human reviewer should inspect this proof step.
    """


class ProofResult(BaseModel):
    """Verification output for a complete proof script."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    status: ProofStatus
    proof_hash: str | None = None
    lemma_results: tuple[LemmaResult, ...] = ()
    failure_summary: str | None = None
    advisories: tuple[str, ...] = ()
    """Aggregated advisories from all lemma results."""


# ---------------------------------------------------------------------------
# ProofScript
# ---------------------------------------------------------------------------


class ProofScript(BaseModel):
    """An ordered chain of lemmas targeting one hypothesis, bound to an axiom set."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    name: str
    target: str
    """Hypothesis name this proof targets."""

    axiom_set_hash: str
    """SHA-256 hex digest of the axiom set this proof is bound to."""

    claim: str
    """Human-readable statement of what the proof chain establishes."""

    lemmas: tuple[Lemma, ...]

    imported_bundles: tuple[ProofBundle, ...] = ()
    """Sealed bundles whose hypothesis results are used as premises.

    Each imported bundle must share the same ``axiom_set_hash``.
    When verifying, imported bundles are re-verified by default
    (see ``trust_imports`` on ``verify_proof``).
    """

    def to_evidence(self) -> dict[str, Any]:
        """Serialize as a JSON-compatible evidence record.

        Each lemma expression is serialized with ``sympy.srepr`` for
        deterministic round-trip.
        """
        return {
            "name": self.name,
            "target": self.target,
            "axiom_set_hash": self.axiom_set_hash,
            "claim": self.claim,
            "imported_bundles": [
                b.to_evidence() for b in self.imported_bundles
            ],
            "lemmas": [
                {
                    "name": lem.name,
                    "kind": lem.kind.value,
                    "expr": sympy.srepr(lem.expr),
                    "expected": (
                        sympy.srepr(lem.expected) if lem.expected is not None else None
                    ),
                    "assumptions": lem.assumptions,
                    "transform": (
                        {k: sympy.srepr(v) for k, v in lem.transform.items()}
                        if lem.transform is not None
                        else None
                    ),
                    "inverse_transform": (
                        {k: sympy.srepr(v) for k, v in lem.inverse_transform.items()}
                        if lem.inverse_transform is not None
                        else None
                    ),
                    "depends_on": lem.depends_on,
                    "description": lem.description,
                }
                for lem in self.lemmas
            ],
        }

    @classmethod
    def from_evidence(cls, data: dict[str, Any]) -> ProofScript:
        """Restore a ``ProofScript`` from an evidence dict.

        Inverse of ``to_evidence()``.
        """
        lemmas = []
        for ld in data["lemmas"]:
            raw_transform = ld.get("transform")
            raw_inverse = ld.get("inverse_transform")
            lemmas.append(
                Lemma(
                    name=ld["name"],
                    kind=LemmaKind(ld["kind"]),
                    expr=sympy.sympify(ld["expr"]),
                    expected=(
                        sympy.sympify(ld["expected"])
                        if ld.get("expected") is not None
                        else None
                    ),
                    assumptions=ld.get("assumptions", {}),
                    transform=(
                        {k: sympy.sympify(v) for k, v in raw_transform.items()}
                        if raw_transform is not None
                        else None
                    ),
                    inverse_transform=(
                        {k: sympy.sympify(v) for k, v in raw_inverse.items()}
                        if raw_inverse is not None
                        else None
                    ),
                    depends_on=ld.get("depends_on", []),
                    description=ld.get("description", ""),
                )
            )
        imported = tuple(
            ProofBundle.from_evidence(bd)
            for bd in data.get("imported_bundles", [])
        )
        return cls(
            name=data["name"],
            target=data["target"],
            axiom_set_hash=data["axiom_set_hash"],
            claim=data["claim"],
            lemmas=tuple(lemmas),
            imported_bundles=imported,
        )


# ---------------------------------------------------------------------------
# ProofBundle
# ---------------------------------------------------------------------------


class ProofBundle(BaseModel):
    """The sealed, verified triple: (axioms, hypothesis, proof).

    Produced exclusively by ``seal()``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    axiom_set: AxiomSet
    hypothesis: Hypothesis
    proof: ProofScript
    proof_result: ProofResult
    bundle_hash: str

    @model_validator(mode="after")
    def _must_be_verified(self) -> Self:
        if self.proof_result.status != ProofStatus.VERIFIED:
            raise ValueError(
                f"ProofBundle requires VERIFIED status, got {self.proof_result.status}"
            )
        return self

    def to_evidence(self) -> dict[str, Any]:
        """Serialize the full bundle as a JSON-compatible evidence record."""
        return {
            "axiom_set": {
                "name": self.axiom_set.name,
                "axioms": [
                    {
                        "name": a.name,
                        "expr": sympy.srepr(a.expr),
                        "description": a.description,
                    }
                    for a in self.axiom_set.axioms
                ],
            },
            "hypothesis": {
                "name": self.hypothesis.name,
                "expr": sympy.srepr(self.hypothesis.expr),
                "axiom_set_hash": self.hypothesis.axiom_set_hash,
                "description": self.hypothesis.description,
            },
            "proof": self.proof.to_evidence(),
            "proof_result": {
                "status": self.proof_result.status.value,
                "proof_hash": self.proof_result.proof_hash,
            },
            "bundle_hash": self.bundle_hash,
        }

    @classmethod
    def from_evidence(cls, data: dict[str, Any]) -> ProofBundle:
        """Restore a ``ProofBundle`` from an evidence dict."""
        ax_data = data["axiom_set"]
        axiom_set = AxiomSet(
            name=ax_data["name"],
            axioms=tuple(
                Axiom(
                    name=a["name"],
                    expr=sympy.sympify(a["expr"]),
                    description=a.get("description", ""),
                )
                for a in ax_data["axioms"]
            ),
        )
        hyp_data = data["hypothesis"]
        hypothesis = Hypothesis(
            name=hyp_data["name"],
            expr=sympy.sympify(hyp_data["expr"]),
            axiom_set_hash=hyp_data["axiom_set_hash"],
            description=hyp_data.get("description", ""),
        )
        proof = ProofScript.from_evidence(data["proof"])
        pr_data = data["proof_result"]
        proof_result = ProofResult(
            status=ProofStatus(pr_data["status"]),
            proof_hash=pr_data.get("proof_hash"),
        )
        return cls(
            axiom_set=axiom_set,
            hypothesis=hypothesis,
            proof=proof,
            proof_result=proof_result,
            bundle_hash=data["bundle_hash"],
        )


# ---------------------------------------------------------------------------
# Disproof
# ---------------------------------------------------------------------------


class Disproof(BaseModel):
    """Compositional disproof of H under A.

    Holds a ``ProofBundle`` that proves ~H under the same axiom set.
    Produced exclusively by ``disprove()``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    hypothesis: Hypothesis
    """The original hypothesis being disproved."""

    negation_bundle: ProofBundle
    """The sealed bundle proving ~H under the same axiom set."""

    disproof_hash: str
    """SHA-256 binding H and the negation bundle."""
