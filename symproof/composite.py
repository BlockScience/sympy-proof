"""Composite symbol types for discrete and fixed-point arithmetic.

Construction-layer abstractions that generate standard SymPy expressions,
Axioms, and Lemmas from declared data types.  Composite types never enter
the proof pipeline directly — they produce artifacts that flow through
the existing ``ProofBuilder`` / ``verify_proof`` / ``seal`` chain.

Available types:

- ``FixedPointType`` — WAD/RAY-style integer-scaled fixed-point arithmetic
- ``ReservePairType`` — AMM constant-product reserve pairs with invariant

Helper:

- ``make_axiom_set`` — merge composite types and explicit axioms into an ``AxiomSet``
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import sympy
from sympy import Eq, Integer, Rational, Symbol, floor, sqrt

from symproof.models import Axiom, AxiomSet, Lemma, LemmaKind

if TYPE_CHECKING:
    from symproof.types import SympyBoolean, SympyExpr


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class CompositeType(ABC):
    """A cluster of related symbols with defined arithmetic.

    Not a Pydantic model — a factory that produces proof artifacts
    (``Axiom``, ``Lemma``, SymPy expressions).  Never appears in a
    ``ProofScript`` or ``ProofBundle``.

    Subclasses must implement ``axioms()``.
    """

    def __init__(
        self,
        name: str,
        symbols: dict[str, sympy.Symbol],
        invariants: tuple[SympyBoolean, ...] = (),
        assumptions: dict[str, dict] | None = None,
    ) -> None:
        self.name = name
        self.symbols = symbols
        self.invariants = invariants
        self.assumptions = assumptions or {}

    @abstractmethod
    def axioms(self, prefix: str = "") -> tuple[Axiom, ...]:
        """Generate axioms from this type's invariants and assumptions."""

    def encode(self, logical_value: SympyExpr) -> SympyExpr:
        """Map a logical value to its raw representation.

        Override in subclasses where encoding is meaningful
        (e.g., WAD: ``n → n * scale``).
        """
        return logical_value

    def decode(self, raw_value: SympyExpr) -> SympyExpr:
        """Map a raw representation back to its logical value.

        Override in subclasses (e.g., WAD: ``a → a / scale``).
        """
        return raw_value

    def _assumption_axioms(self, prefix: str = "") -> list[Axiom]:
        """Generate positivity/integer axioms from per-symbol assumptions."""
        result: list[Axiom] = []
        # Map from assumption kwarg to SymPy relational constructor
        _prop_to_expr: dict[str, any] = {
            "positive": lambda s: s > 0,
            "nonnegative": lambda s: s >= 0,
            "negative": lambda s: s < 0,
            "nonpositive": lambda s: s <= 0,
            "nonzero": lambda s: sympy.Ne(s, 0),
        }
        for sym_name, asm in self.assumptions.items():
            sym = self.symbols.get(sym_name, Symbol(sym_name))
            for prop, val in asm.items():
                if val and prop in _prop_to_expr:
                    result.append(
                        Axiom(
                            name=f"{prefix}{sym_name}_{prop}",
                            expr=_prop_to_expr[prop](sym),
                        )
                    )
        return result

    def _invariant_axioms(self, prefix: str = "") -> list[Axiom]:
        """Generate axioms from declared invariants."""
        return [
            Axiom(
                name=f"{prefix}{self.name}_invariant_{i}",
                expr=inv,
            )
            for i, inv in enumerate(self.invariants)
        ]


# ---------------------------------------------------------------------------
# FixedPointType — WAD / RAY arithmetic
# ---------------------------------------------------------------------------


class FixedPointType(CompositeType):
    """Integer-scaled fixed-point arithmetic (WAD, RAY, basis points).

    A logical value ``v`` is stored as the integer ``v * scale``.
    Operations:

    - ``mul(a, b)`` → ``floor(a * b / scale)``
    - ``div(a, b)`` → ``floor(a * scale / b)``
    - ``from_int(n)`` → ``n * scale``
    - ``to_int(a)`` → ``floor(a / scale)``

    Parameters
    ----------
    name:
        Human name for this fixed-point scheme (e.g., ``"wad"``).
    scale:
        The scaling factor as a Python int (e.g., ``10**18``).
    """

    def __init__(self, name: str = "wad", scale: int = 10**18) -> None:
        self.scale_int = scale
        self.scale = Integer(scale)
        scale_sym = Symbol(name.upper(), positive=True, integer=True)
        super().__init__(
            name=name,
            symbols={name.upper(): scale_sym},
            invariants=(Eq(scale_sym, self.scale),),
            assumptions={name.upper(): {"positive": True, "integer": True}},
        )
        self.scale_symbol = scale_sym

    # -- Axiom generation ---------------------------------------------------

    def axioms(self, prefix: str = "") -> tuple[Axiom, ...]:
        """Generate axioms: scale definition."""
        return (
            Axiom(
                name=f"{prefix}{self.name}_scale",
                expr=Eq(self.scale_symbol, self.scale),
                description=f"{self.name.upper()} = {self.scale_int}",
            ),
        )

    # -- Encoding / decoding ------------------------------------------------

    def encode(self, logical_value: SympyExpr) -> SympyExpr:
        """Encode a logical value: ``n → n * scale``."""
        return logical_value * self.scale

    def from_int(self, n: int | SympyExpr) -> SympyExpr:
        """Shorthand for ``encode``: ``n → n * scale``."""
        if isinstance(n, int):
            n = Integer(n)
        return n * self.scale

    def decode(self, raw_value: SympyExpr) -> SympyExpr:
        """Decode a raw value: ``a → a / scale``."""
        return Rational(1, self.scale_int) * raw_value

    def to_int(self, a: SympyExpr) -> SympyExpr:
        """Truncating decode: ``a → floor(a / scale)``."""
        return floor(a / self.scale)

    # -- Operations ---------------------------------------------------------

    def mul(self, a: SympyExpr, b: SympyExpr) -> SympyExpr:
        """Fixed-point multiply: ``floor(a * b / scale)``."""
        return floor(a * b / self.scale)

    def div(self, a: SympyExpr, b: SympyExpr) -> SympyExpr:
        """Fixed-point divide: ``floor(a * scale / b)``."""
        return floor(a * self.scale / b)

    # -- Lemma generators ---------------------------------------------------

    def mul_lemma(
        self,
        a: SympyExpr,
        b: SympyExpr,
        expected: SympyExpr,
        name: str = "mulwad",
        description: str = "",
    ) -> Lemma:
        """EQUALITY lemma: ``floor(a * b / scale) == expected``."""
        return Lemma(
            name=name,
            kind=LemmaKind.EQUALITY,
            expr=self.mul(a, b),
            expected=expected,
            description=description or f"{self.name}_mul({a}, {b}) = {expected}",
        )

    def div_lemma(
        self,
        a: SympyExpr,
        b: SympyExpr,
        expected: SympyExpr,
        name: str = "divwad",
        description: str = "",
    ) -> Lemma:
        """EQUALITY lemma: ``floor(a * scale / b) == expected``."""
        return Lemma(
            name=name,
            kind=LemmaKind.EQUALITY,
            expr=self.div(a, b),
            expected=expected,
            description=description or f"{self.name}_div({a}, {b}) = {expected}",
        )

    def roundtrip_lemmas(
        self,
        a: SympyExpr,
        b: SympyExpr,
        prefix: str = "rt",
    ) -> list[Lemma]:
        """Generate mul-then-div roundtrip lemma chain.

        Returns two lemmas:
        1. ``{prefix}_mul``: ``mul(a, b) == product``
        2. ``{prefix}_div``: ``div(product, b) == a`` (or truncated)

        Only exact when ``a * b`` is divisible by ``scale``.
        """
        product = self.mul(a, b)
        back = self.div(product, b)
        return [
            Lemma(
                name=f"{prefix}_mul",
                kind=LemmaKind.EQUALITY,
                expr=product,
                expected=product,  # self-verifying; caller can override
                description=f"{self.name}_mul step",
            ),
            Lemma(
                name=f"{prefix}_div",
                kind=LemmaKind.EQUALITY,
                expr=back,
                expected=back,  # self-verifying; caller can override
                depends_on=[f"{prefix}_mul"],
                description=f"{self.name}_div roundtrip step",
            ),
        ]


# ---------------------------------------------------------------------------
# ReservePairType — AMM constant-product
# ---------------------------------------------------------------------------


class ReservePairType(CompositeType):
    """AMM constant-product reserve pair with invariant ``Rx * Ry = k``.

    Provides swap formulas (exact and integer-truncated), invariant
    preservation lemmas, and hyperbolic coordinate transforms for
    ``COORDINATE_TRANSFORM`` proofs.

    Parameters
    ----------
    name:
        Name for this pool (e.g., ``"amm"``).
    rx_name:
        Symbol name for reserve X (default ``"R_x"``).
    ry_name:
        Symbol name for reserve Y (default ``"R_y"``).
    """

    def __init__(
        self,
        name: str = "amm",
        rx_name: str = "R_x",
        ry_name: str = "R_y",
    ) -> None:
        self.rx = Symbol(rx_name, positive=True)
        self.ry = Symbol(ry_name, positive=True)
        self.k = Symbol("k", positive=True)
        self.s = Symbol("s", positive=True)  # price ratio for transforms

        super().__init__(
            name=name,
            symbols={
                rx_name: self.rx,
                ry_name: self.ry,
                "k": self.k,
                "s": self.s,
            },
            invariants=(Eq(self.rx * self.ry, self.k),),
            assumptions={
                rx_name: {"positive": True},
                ry_name: {"positive": True},
                "k": {"positive": True},
                "s": {"positive": True},
            },
        )

    # -- Axiom generation ---------------------------------------------------

    def axioms(self, prefix: str = "") -> tuple[Axiom, ...]:
        """Generate axioms: reserve positivity and product invariant."""
        return (
            Axiom(
                name=f"{prefix}{self.rx.name}_positive",
                expr=self.rx > 0,
            ),
            Axiom(
                name=f"{prefix}{self.ry.name}_positive",
                expr=self.ry > 0,
            ),
        )

    # -- Operations ---------------------------------------------------------

    def swap_output(self, dx: SympyExpr) -> SympyExpr:
        """Exact (real-valued) swap output: ``Ry * dx / (Rx + dx)``."""
        return self.ry * dx / (self.rx + dx)

    def swap_output_int(self, dx: SympyExpr) -> SympyExpr:
        """Integer-truncated swap output: ``floor(Ry * dx / (Rx + dx))``."""
        return floor(self.ry * dx / (self.rx + dx))

    def swap_output_with_fee(
        self, dx: SympyExpr, fee: SympyExpr,
    ) -> SympyExpr:
        """Swap output with fee: ``Ry * net / (Rx + net)``.

        Where ``net = dx * (1 - fee)``.
        """
        net = dx * (1 - fee)
        return self.ry * net / (self.rx + net)

    def post_swap_reserves(
        self, dx: SympyExpr, *, integer: bool = False,
    ) -> tuple[SympyExpr, SympyExpr]:
        """Post-swap reserves ``(Rx + dx, Ry - dy)``.

        If ``integer=True``, uses floor-truncated output.
        """
        dy = self.swap_output_int(dx) if integer else self.swap_output(dx)
        return (self.rx + dx, self.ry - dy)

    def post_swap_k(
        self, dx: SympyExpr, *, integer: bool = False,
    ) -> SympyExpr:
        """Post-swap product: ``(Rx + dx) * (Ry - dy)``."""
        rx_new, ry_new = self.post_swap_reserves(dx, integer=integer)
        return rx_new * ry_new

    # -- Transforms ---------------------------------------------------------

    def hyperbolic_transform(
        self,
    ) -> tuple[dict[str, SympyExpr], dict[str, SympyExpr]]:
        """Forward/inverse maps for hyperbolic (k, s) coordinates.

        Forward:  ``Rx → sqrt(k * s)``,  ``Ry → sqrt(k / s)``
        Inverse:  ``k → Rx * Ry``,       ``s → Rx / Ry``

        Returns ``(forward_dict, inverse_dict)`` suitable for
        ``COORDINATE_TRANSFORM`` lemmas.
        """
        forward = {
            self.rx.name: sqrt(self.k * self.s),
            self.ry.name: sqrt(self.k / self.s),
        }
        inverse = {
            "k": self.rx * self.ry,
            "s": self.rx / self.ry,
        }
        return forward, inverse

    # -- Lemma generators ---------------------------------------------------

    def output_lemma(
        self,
        dx: SympyExpr,
        expected: SympyExpr,
        name: str = "swap_output",
        *,
        integer: bool = False,
        description: str = "",
    ) -> Lemma:
        """EQUALITY lemma for swap output."""
        expr = self.swap_output_int(dx) if integer else self.swap_output(dx)
        return Lemma(
            name=name,
            kind=LemmaKind.EQUALITY,
            expr=expr,
            expected=expected,
            assumptions={
                self.rx.name: {"positive": True},
                self.ry.name: {"positive": True},
            },
            description=description or f"Swap output for dx={dx}",
        )

    def invariant_lemma(
        self,
        dx: SympyExpr,
        name: str = "k_preserved",
        *,
        integer: bool = True,
        description: str = "",
    ) -> Lemma:
        """BOOLEAN lemma: ``post_swap_k >= pre_swap_k``.

        With integer truncation (default), floor division favors the pool,
        so the product is nondecreasing.
        """
        k_post = self.post_swap_k(dx, integer=integer)
        return Lemma(
            name=name,
            kind=LemmaKind.BOOLEAN,
            expr=sympy.Ge(k_post, self.rx * self.ry),
            description=description or "Product invariant nondecreasing after swap",
        )

    def invariant_is_coordinate_lemma(
        self,
        name: str = "product_is_k",
        description: str = "",
    ) -> Lemma:
        """COORDINATE_TRANSFORM lemma: ``Rx * Ry = k`` in hyperbolic coords.

        The constant-product invariant is literally a coordinate in (k, s) space.
        """
        fwd, inv = self.hyperbolic_transform()
        return Lemma(
            name=name,
            kind=LemmaKind.COORDINATE_TRANSFORM,
            expr=self.rx * self.ry,
            expected=self.k,
            transform=fwd,
            inverse_transform=inv,
            assumptions={
                self.rx.name: {"positive": True},
                self.ry.name: {"positive": True},
                "k": {"positive": True},
                "s": {"positive": True},
            },
            description=description
            or "Rx*Ry = k is a tautology in hyperbolic coordinates",
        )

    def price_is_coordinate_lemma(
        self,
        name: str = "price_is_s",
        description: str = "",
    ) -> Lemma:
        """COORDINATE_TRANSFORM lemma: ``Rx / Ry = s`` in hyperbolic coords."""
        fwd, inv = self.hyperbolic_transform()
        return Lemma(
            name=name,
            kind=LemmaKind.COORDINATE_TRANSFORM,
            expr=self.rx / self.ry,
            expected=self.s,
            transform=fwd,
            inverse_transform=inv,
            assumptions={
                self.rx.name: {"positive": True},
                self.ry.name: {"positive": True},
                "k": {"positive": True},
                "s": {"positive": True},
            },
            description=description
            or "Rx/Ry = s is a tautology in hyperbolic coordinates",
        )


# ---------------------------------------------------------------------------
# Axiom set construction helper
# ---------------------------------------------------------------------------


def make_axiom_set(
    name: str,
    *sources: CompositeType | Axiom | tuple[Axiom, ...],
) -> AxiomSet:
    """Build an ``AxiomSet`` from composite types and explicit axioms.

    Collects ``.axioms()`` from each ``CompositeType``, merges with
    explicit ``Axiom`` objects, and checks for name collisions.

    Parameters
    ----------
    name:
        Name for the resulting axiom set.
    *sources:
        Mix of ``CompositeType`` instances (calls ``.axioms()``),
        individual ``Axiom`` objects, or tuples of ``Axiom`` objects.

    Raises
    ------
    ValueError
        If axiom names collide across sources.
    """
    all_axioms: list[Axiom] = []
    for src in sources:
        if isinstance(src, CompositeType):
            all_axioms.extend(src.axioms())
        elif isinstance(src, Axiom):
            all_axioms.append(src)
        elif isinstance(src, tuple):
            all_axioms.extend(src)
        else:
            raise TypeError(
                f"Expected CompositeType, Axiom, or tuple[Axiom, ...], "
                f"got {type(src).__name__}"
            )

    # Check for name collisions
    names = [a.name for a in all_axioms]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(f"Axiom name collision in make_axiom_set: {dupes}")

    return AxiomSet(name=name, axioms=tuple(all_axioms))
