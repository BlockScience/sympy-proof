"""Chainable builder for constructing proof scripts.

Usage::

    import sympy
    from symproof import AxiomSet, Axiom, ProofBuilder, LemmaKind

    x = sympy.Symbol("x")
    axioms = AxiomSet(name="basics", axioms=(
        Axiom(name="x_positive", expr=x > 0),
    ))

    script = (
        ProofBuilder(axioms, "some_hypothesis",
                     name="my_proof", claim="x > 0 implies x**2 > 0")
        .lemma("x_sq_positive", LemmaKind.QUERY,
               expr=sympy.Q.positive(x**2),
               assumptions={"x": {"positive": True}})
        .build()
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from symproof.models import AxiomSet, Lemma, LemmaKind, ProofScript

if TYPE_CHECKING:
    import sympy


class ProofBuilder:
    """Chainable builder for constructing auxiliary proof scripts."""

    def __init__(
        self,
        axiom_set: AxiomSet | str,
        target: str,
        *,
        name: str,
        claim: str,
    ) -> None:
        """Initialize the proof builder.

        Parameters
        ----------
        axiom_set:
            An ``AxiomSet`` object (extracts hash automatically) or a raw
            SHA-256 hash string (for deserialized contexts).
        target:
            The hypothesis name this proof targets.
        name:
            Name for the proof script.
        claim:
            Human-readable statement of what the proof establishes.
        """
        if isinstance(axiom_set, AxiomSet):
            self._axiom_set_hash = axiom_set.axiom_set_hash
        elif isinstance(axiom_set, str):
            self._axiom_set_hash = axiom_set
        else:
            raise TypeError(
                f"axiom_set must be AxiomSet or str, got {type(axiom_set).__name__}"
            )
        self._target = target
        self._name = name
        self._claim = claim
        self._lemmas: list[Lemma] = []

    def lemma(
        self,
        name: str,
        kind: LemmaKind,
        expr: sympy.Basic,
        expected: sympy.Basic | None = None,
        assumptions: dict[str, dict] | None = None,
        depends_on: list[str] | None = None,
        description: str = "",
        transform: dict[str, sympy.Basic] | None = None,
        inverse_transform: dict[str, sympy.Basic] | None = None,
    ) -> ProofBuilder:
        """Add a lemma to the proof chain and return self for chaining."""
        self._lemmas.append(
            Lemma(
                name=name,
                kind=kind,
                expr=expr,
                expected=expected,
                assumptions=assumptions or {},
                depends_on=depends_on or [],
                description=description,
                transform=transform,
                inverse_transform=inverse_transform,
            )
        )
        return self

    def build(self) -> ProofScript:
        """Finalize and return the ``ProofScript``.

        Does not verify the proof — call ``verify_proof()`` separately.

        Raises
        ------
        ValueError
            If no lemmas were added.
        """
        if not self._lemmas:
            raise ValueError("A ProofScript must contain at least one lemma.")
        return ProofScript(
            name=self._name,
            target=self._target,
            axiom_set_hash=self._axiom_set_hash,
            claim=self._claim,
            lemmas=tuple(self._lemmas),
        )
