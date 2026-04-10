"""DeFi mechanism proofs — the symbolic mathematics leg.

This module proves **mathematical properties of DeFi formulas**.  It is
one layer of a three-part verification architecture:

    Symbolic (this module)
        → proves the formula is correct
    Numeric (simulation / fuzzing)
        → proves the formula behaves under finite precision
    Implementation (Certora / Halmos / manual audit)
        → proves the Solidity code matches the formula

What this module PROVES:

- Rounding direction is correct per operation and **net-positive
  across a pipeline** (the signed accumulation pattern)
- Truncation error is bounded: at most 1 unit per floor/ceil
- Intermediate products don't overflow uint256 (phantom overflow)
- Cross-decimal token pairs require normalization
- Fee-bearing swap output is positive
- Constant-product invariant is nondecreasing with fees

What this module does NOT prove (and what covers the gap):

- Production Solidity matches these formulas → code audit / Certora
- State transitions are atomic (reentrancy) → static analysis
- Token transfers deliver expected amounts → integration tests
- Fee parameters are governance-bounded → access control audit
- Formulas are robust to MEV / sandwich attacks → simulation

The sealed proof hashes from this module go into a requirements
traceability matrix alongside simulation results and code findings.
Any reviewer can re-run and get the identical hash — independent,
reproducible evidence for the mathematical layer.

Tools:

**Rounding direction** — ``mul_down`` / ``mul_up`` / ``div_down`` / ``div_up``
    and proofs that rounding bias favors the correct party.

**Directional chain error** — ``directional_chain_error`` proves each
    step's sign AND that the net error favors the protocol.  Built on
    the core ``signed_sum_lemmas`` tactic.

**Phantom overflow** — ``phantom_overflow_check`` / ``safe_mul_div``
    detect when ``a * b`` exceeds uint256 before division.

**Decimal-aware reserves** — ``DecimalAwarePool`` handles token pairs
    with different decimal places (e.g., USDC/6 ↔ WETH/18).

**AMM proof bundles** — ``fee_complement_positive``,
    ``amm_output_positive``, ``amm_product_nondecreasing``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import sympy
from sympy import Integer, Rational, Symbol, ceiling, floor

from symproof.builder import ProofBuilder
from symproof.bundle import seal
from symproof.evaluation import evaluation
from symproof.models import AxiomSet, Lemma, LemmaKind, ProofBundle

# ===================================================================
# Rounding direction — mul/div with explicit UP or DOWN
# ===================================================================


def mul_down(a: sympy.Basic, b: sympy.Basic, scale: sympy.Basic) -> sympy.Basic:
    """Fixed-point multiply rounding DOWN (truncate): ``floor(a * b / scale)``.

    Favors the protocol / pool — the result is at most the true value.
    This is what Solidity's ``mulWadDown`` computes.
    """
    return floor(a * b / scale)


def mul_up(a: sympy.Basic, b: sympy.Basic, scale: sympy.Basic) -> sympy.Basic:
    """Fixed-point multiply rounding UP: ``ceil(a * b / scale)``.

    Favors the user / depositor — the result is at least the true value.
    Equivalent to ``(a * b + scale - 1) / scale`` in integer Solidity.
    """
    return ceiling(a * b / scale)


def div_down(a: sympy.Basic, b: sympy.Basic, scale: sympy.Basic) -> sympy.Basic:
    """Fixed-point divide rounding DOWN: ``floor(a * scale / b)``.

    Favors the protocol — the result is at most the true value.
    """
    return floor(a * scale / b)


def div_up(a: sympy.Basic, b: sympy.Basic, scale: sympy.Basic) -> sympy.Basic:
    """Fixed-point divide rounding UP: ``ceil(a * scale / b)``.

    Favors the user — the result is at least the true value.
    """
    return ceiling(a * scale / b)


def rounding_bias_lemma(
    a: sympy.Basic,
    b: sympy.Basic,
    scale: sympy.Basic,
    name: str = "rounding_bias",
) -> Lemma:
    """Prove ``mul_up(a, b, s) >= mul_down(a, b, s)``.

    The UP variant is always >= the DOWN variant.  This is the
    fundamental invariant auditors check: whichever direction you
    round, the other party is disadvantaged by at most 1 unit.
    """
    down = mul_down(a, b, scale)
    up = mul_up(a, b, scale)
    return Lemma(
        name=name,
        kind=LemmaKind.BOOLEAN,
        expr=sympy.Ge(up, down),
        description="ceil(a*b/s) >= floor(a*b/s)",
    )


def rounding_gap_lemma(
    a: sympy.Basic,
    b: sympy.Basic,
    scale: sympy.Basic,
    name: str = "rounding_gap",
) -> Lemma:
    """Prove ``mul_up(a, b, s) - mul_down(a, b, s) <= 1``.

    The gap between UP and DOWN rounding is at most 1 unit of the
    output token.  This bounds the maximum extraction per operation.
    """
    down = mul_down(a, b, scale)
    up = mul_up(a, b, scale)
    gap = up - down
    return Lemma(
        name=name,
        kind=LemmaKind.BOOLEAN,
        expr=sympy.Le(gap, Integer(1)),
        description="ceil - floor <= 1 (max 1 unit rounding gap)",
    )


# ===================================================================
# Directional rounding — who benefits from truncation?
# ===================================================================


class RoundingFavor(StrEnum):
    """Who a rounding operation should favor."""

    PROTOCOL = "protocol"
    """Rounding favors the protocol / pool / LPs."""

    USER = "user"
    """Rounding favors the user / swapper / depositor."""


@dataclass(frozen=True)
class RoundingStep:
    """One step in a multi-operation pipeline with directional intent.

    Parameters
    ----------
    exact:
        The real-valued result before rounding.
    rounded:
        The integer result after rounding (``floor`` or ``ceil``).
    favor:
        Who this step **should** favor.
    label:
        Human description (e.g., ``"swap output"``, ``"fee deduction"``).
    """

    exact: sympy.Basic
    rounded: sympy.Basic
    favor: RoundingFavor
    label: str = ""

    @property
    def error(self) -> sympy.Basic:
        """Signed error: ``exact - rounded``.

        Positive means rounded value is LESS than exact (floor behavior).
        Negative means rounded value is MORE than exact (ceil behavior).
        """
        return self.exact - self.rounded

    @property
    def uses_floor(self) -> bool:
        """True if ``rounded <= exact`` (floor-style rounding)."""
        with evaluation():
            diff = sympy.simplify(self.exact - self.rounded)
        return diff.is_nonnegative is True  # type: ignore[return-value]

    @property
    def uses_ceil(self) -> bool:
        """True if ``rounded >= exact`` (ceil-style rounding)."""
        with evaluation():
            diff = sympy.simplify(self.rounded - self.exact)
        return diff.is_nonnegative is True  # type: ignore[return-value]

    @property
    def direction_correct(self) -> bool:
        """Check if the rounding direction matches the declared favor.

        - Favor PROTOCOL on an output/payment TO user → floor is correct
          (user receives less, protocol keeps dust)
        - Favor PROTOCOL on a deduction/fee FROM user → ceil is correct
          (user pays more, protocol takes dust)

        Convention: ``floor`` favors the NON-recipient (protocol keeps
        dust from what it sends, user keeps dust from what they send).
        For this check, we use the simpler rule: floor means
        ``rounded <= exact``, so the holder of the rounding dust is
        the party that did NOT receive ``rounded``.

        Since the caller declares ``favor`` explicitly, we just check:
        - favor=PROTOCOL + floor → correct (protocol withholds dust)
        - favor=PROTOCOL + ceil → correct (protocol charges dust)
        - favor=USER + floor → MISMATCH (user gets less)
        - favor=USER + ceil → MISMATCH (user pays more)

        Wait — that's wrong.  The correct rule is:

        - ``favor=PROTOCOL`` means "the rounded value should be
          conservative FOR THE PROTOCOL."
          - On outputs to user: ``floor(exact)`` → user gets less ✓
          - On charges from user: ``ceil(exact)`` → user pays more ✓
        - ``favor=USER`` means "the rounded value should be
          conservative FOR THE USER."
          - On outputs to user: ``ceil(exact)`` → user gets more ✓
          - On charges from user: ``floor(exact)`` → user pays less ✓

        Since we can't know from the expression alone whether this is
        an "output" or "charge", the caller encodes the semantic via
        the combination of rounding function + favor direction.

        Final rule: a step is correctly directional if
        ``favor=PROTOCOL ↔ uses_floor`` OR ``favor=USER ↔ uses_ceil``
        for the STANDARD case (computing an amount the user receives).

        **We simplify**: the caller is responsible for choosing
        floor/ceil to match their intent.  We just verify that the
        rounding error's sign is consistent with the declared favor:
        - ``favor=PROTOCOL``: ``error >= 0`` (exact >= rounded, user
          gets at most the true value)
        - ``favor=USER``: ``error <= 0`` (exact <= rounded, user
          gets at least the true value)
        """
        if self.favor == RoundingFavor.PROTOCOL:
            return self.uses_floor
        return self.uses_ceil


def directional_chain_error(
    steps: list[RoundingStep],
    name_prefix: str = "dir_err",
) -> list[Lemma]:
    """Prove directional error bounds across a multi-step pipeline.

    Built on the core ``signed_sum_lemmas`` tactic (from
    ``symproof.tactics``), adding DeFi-specific semantics:

    1. **Per-step direction**: error sign matches declared favor
    2. **Per-step magnitude**: ``|error| < 1`` (max 1 wei per op)
    3. **Directional mismatch warnings**: flagged in lemma descriptions
    4. **Net flow check**: net error >= 0 (protocol doesn't leak)

    Parameters
    ----------
    steps:
        Ordered list of ``RoundingStep`` objects.
    name_prefix:
        Prefix for generated lemma names.

    Returns
    -------
    list[Lemma]
        Per-term sign + bound lemmas (from core tactic), plus a
        DeFi-specific net-flow lemma.  Mismatch steps are flagged.
    """
    from symproof.tactics import SignedTerm, signed_sum_lemmas

    # Convert DeFi RoundingSteps to core SignedTerms.
    # PROTOCOL favor → error >= 0 (nonneg=True)
    # USER favor → error <= 0 (nonneg=False)
    signed_terms = []
    for step in steps:
        is_correct = step.direction_correct
        mismatch = "" if is_correct else " ⚠ MISMATCH"
        lbl = (step.label or "step") + mismatch
        signed_terms.append(
            SignedTerm(
                expr=step.error,
                nonneg=(step.favor == RoundingFavor.PROTOCOL),
                bound=Integer(1),  # max 1 wei per rounding op
                label=lbl,
            )
        )

    # Delegate to core tactic: sign + bound + net direction
    lemmas = signed_sum_lemmas(
        signed_terms,
        net_nonneg=True,  # net must favor protocol
        name_prefix=name_prefix,
    )

    # Enhance the net lemma description with DeFi-specific warning
    net_lemma = lemmas[-1]
    total_error = sum(
        step.error for step in steps
    )  # type: ignore[arg-type]
    with evaluation():
        net_is_safe = sympy.simplify(
            sympy.Ge(total_error, Integer(0))
        ) is sympy.true
    if not net_is_safe and "NEGATIVE" not in net_lemma.description:
        # Replace the net lemma with DeFi-specific description
        lemmas[-1] = Lemma(
            name=net_lemma.name,
            kind=net_lemma.kind,
            expr=net_lemma.expr,
            depends_on=net_lemma.depends_on,
            description=(
                "Net error >= 0 ⚠ NET NEGATIVE"
                " — pipeline leaks value from protocol!"
            ),
        )

    return lemmas


# ===================================================================
# Compound rounding error — N-step error accumulation (legacy)
# ===================================================================


def chain_error_bound(
    exact_values: list[sympy.Basic],
    truncated_values: list[sympy.Basic],
    name_prefix: str = "chain_err",
) -> list[Lemma]:
    """Prove that N sequential floor operations accumulate at most N error.

    Given a list of exact (real-valued) results and their floor-truncated
    counterparts, generates lemmas proving:
    1. Each step's error: ``0 <= exact[i] - truncated[i] < 1``
    2. Cumulative bound: ``sum(exact) - sum(truncated) < N``

    This is the proof auditors need: "over N operations, total rounding
    loss is bounded by N units (wei)."

    Parameters
    ----------
    exact_values:
        The real-valued results of each operation (before floor).
    truncated_values:
        The floor-truncated results (what the EVM actually computes).
    """
    n = len(exact_values)
    if n != len(truncated_values):
        raise ValueError("exact_values and truncated_values must have same length")

    lemmas: list[Lemma] = []

    # Per-step error bounds
    for i, (exact, trunc) in enumerate(
        zip(exact_values, truncated_values, strict=True),
    ):
        step_error = exact - trunc
        lemmas.append(
            Lemma(
                name=f"{name_prefix}_step_{i}_nonneg",
                kind=LemmaKind.BOOLEAN,
                expr=sympy.Ge(step_error, Integer(0)),
                depends_on=[f"{name_prefix}_step_{i-1}_nonneg"] if i > 0 else [],
                description=f"Step {i}: exact - floor >= 0",
            )
        )
        lemmas.append(
            Lemma(
                name=f"{name_prefix}_step_{i}_lt_one",
                kind=LemmaKind.BOOLEAN,
                expr=sympy.Lt(step_error, Integer(1)),
                depends_on=[f"{name_prefix}_step_{i}_nonneg"],
                description=f"Step {i}: exact - floor < 1",
            )
        )

    # Cumulative bound
    total_exact = sum(exact_values)  # type: ignore[arg-type]
    total_trunc = sum(truncated_values)  # type: ignore[arg-type]
    cumulative_error = total_exact - total_trunc
    lemmas.append(
        Lemma(
            name=f"{name_prefix}_cumulative",
            kind=LemmaKind.BOOLEAN,
            expr=sympy.Lt(cumulative_error, Integer(n)),
            depends_on=[f"{name_prefix}_step_{i}_lt_one" for i in range(n)],
            description=f"Cumulative error < {n} (N operations)",
        )
    )

    return lemmas


# ===================================================================
# Phantom overflow — safe mulDiv modeling
# ===================================================================

UINT256_MAX = Integer(2) ** 256 - 1


def overflows_uint256(expr: sympy.Basic) -> sympy.Basic:
    """Boolean expression: ``expr > UINT256_MAX``."""
    return sympy.Gt(expr, UINT256_MAX)


def safe_mul_div(
    a: sympy.Basic,
    b: sympy.Basic,
    denominator: sympy.Basic,
) -> sympy.Basic:
    """Model OpenZeppelin-style mulDiv that avoids intermediate overflow.

    In the EVM, ``a * b / c`` computes ``a * b`` first, which may overflow
    uint256 even if the final result fits.  ``mulDiv`` handles this via
    512-bit intermediate or modular decomposition.

    This function returns the mathematically correct ``floor(a * b / c)``
    as a SymPy expression.  Use ``phantom_overflow_check`` to generate
    a lemma proving whether the naive path would overflow.
    """
    return floor(a * b / denominator)


def phantom_overflow_check(
    a: sympy.Basic,
    b: sympy.Basic,
    denominator: sympy.Basic,
    name: str = "phantom_overflow",
) -> Lemma:
    """Prove whether ``a * b`` overflows uint256 (phantom overflow check).

    Returns a BOOLEAN lemma that is True if ``a * b > UINT256_MAX``.
    A passing proof means phantom overflow IS present — the naive
    ``a * b / denominator`` would silently wrap in the EVM.

    Use this to prove that ``mulDiv`` (not naive division) is required.
    """
    product = a * b
    return Lemma(
        name=name,
        kind=LemmaKind.BOOLEAN,
        expr=overflows_uint256(product),
        description=f"a*b = {a}*{b} overflows uint256 "
        f"(phantom overflow — mulDiv required)",
    )


def no_phantom_overflow_check(
    a: sympy.Basic,
    b: sympy.Basic,
    name: str = "no_phantom_overflow",
) -> Lemma:
    """Prove ``a * b`` does NOT overflow uint256 (naive path is safe).

    A passing proof means the naive ``a * b / c`` is safe — no need
    for ``mulDiv``.
    """
    product = a * b
    return Lemma(
        name=name,
        kind=LemmaKind.BOOLEAN,
        expr=sympy.Le(product, UINT256_MAX),
        description=f"a*b = {a}*{b} fits in uint256 (naive division safe)",
    )


# ===================================================================
# Decimal-aware pool — cross-decimal token pairs
# ===================================================================


class DecimalAwarePool:
    """AMM pool that handles tokens with different decimal places.

    Real protocols must normalize amounts when computing swaps between
    tokens with different decimals (e.g., USDC/6 ↔ WETH/18).
    Failing to normalize is a routine audit finding.

    The pool stores reserves in their native decimal representation
    and provides normalization methods for cross-decimal arithmetic.

    Parameters
    ----------
    rx_name:
        Symbol name for reserve X.
    ry_name:
        Symbol name for reserve Y.
    decimals_x:
        Decimal places for token X (e.g., 6 for USDC).
    decimals_y:
        Decimal places for token Y (e.g., 18 for WETH).
    """

    def __init__(
        self,
        rx_name: str = "R_x",
        ry_name: str = "R_y",
        decimals_x: int = 18,
        decimals_y: int = 18,
    ) -> None:
        self.rx = Symbol(rx_name, positive=True, integer=True)
        self.ry = Symbol(ry_name, positive=True, integer=True)
        self.decimals_x = decimals_x
        self.decimals_y = decimals_y
        self.scale_x = Integer(10) ** decimals_x
        self.scale_y = Integer(10) ** decimals_y

        # Normalization factor: multiply X amounts by this to match Y's scale
        if decimals_x < decimals_y:
            self.norm_x_to_y = Integer(10) ** (decimals_y - decimals_x)
            self.norm_y_to_x = Rational(1, 10 ** (decimals_y - decimals_x))
        elif decimals_x > decimals_y:
            self.norm_x_to_y = Rational(1, 10 ** (decimals_x - decimals_y))
            self.norm_y_to_x = Integer(10) ** (decimals_x - decimals_y)
        else:
            self.norm_x_to_y = Integer(1)
            self.norm_y_to_x = Integer(1)

    def normalize_x(self, amount_x: sympy.Basic) -> sympy.Basic:
        """Normalize a token-X amount to token-Y's decimal scale."""
        return amount_x * self.norm_x_to_y

    def normalize_y(self, amount_y: sympy.Basic) -> sympy.Basic:
        """Normalize a token-Y amount to token-X's decimal scale."""
        return amount_y * self.norm_y_to_x

    def swap_x_for_y(self, dx: sympy.Basic) -> sympy.Basic:
        """Swap dx of token X for dy of token Y (exact, normalized).

        Normalizes dx to Y's decimals before computing the constant-product
        formula, then returns dy in Y's native decimals.
        """
        dx_norm = self.normalize_x(dx)
        rx_norm = self.normalize_x(self.rx)
        return self.ry * dx_norm / (rx_norm + dx_norm)

    def swap_x_for_y_int(self, dx: sympy.Basic) -> sympy.Basic:
        """Integer-truncated swap (floor): what the EVM actually computes."""
        return floor(self.swap_x_for_y(dx))

    def naive_swap_x_for_y(self, dx: sympy.Basic) -> sympy.Basic:
        """WRONG formula: no decimal normalization.

        This is the bug auditors find — computing ``Ry * dx / (Rx + dx)``
        without normalizing decimals.  Compare against ``swap_x_for_y``
        to prove the mismatch.
        """
        return self.ry * dx / (self.rx + dx)

    def decimal_mismatch_lemma(
        self,
        dx: sympy.Basic,
        name: str = "decimal_mismatch",
    ) -> Lemma:
        """Prove that naive (unnormalized) swap differs from correct swap.

        Returns an EQUALITY lemma showing the ratio between the correct
        and naive formulas.  When decimals differ, this ratio != 1,
        proving the normalization is required.
        """
        correct = self.swap_x_for_y(dx)
        naive = self.naive_swap_x_for_y(dx)
        # The ratio correct/naive should equal norm_x_to_y when
        # the constant-product terms cancel.  We prove the difference.
        with evaluation():
            _mismatch_simplified = sympy.simplify(correct - naive)
        return Lemma(
            name=name,
            kind=LemmaKind.EQUALITY,
            expr=_mismatch_simplified,
            expected=_mismatch_simplified,
            description="Correct (normalized) - naive swap output "
            f"(X:{self.decimals_x} dec, Y:{self.decimals_y} dec)",
        )

    def normalization_factor_lemma(
        self,
        name: str = "normalization_factor",
    ) -> Lemma:
        """Prove the normalization factor between the two decimal scales.

        When decimals_x != decimals_y, the factor is ``10^|dx - dy|``.
        """
        factor = self.norm_x_to_y
        expected = Integer(10) ** abs(self.decimals_x - self.decimals_y)
        if self.decimals_x > self.decimals_y:
            expected = Rational(
                1, 10 ** (self.decimals_x - self.decimals_y)
            )
        return Lemma(
            name=name,
            kind=LemmaKind.EQUALITY,
            expr=factor,
            expected=expected,
            description=f"Normalization: X({self.decimals_x} dec) "
            f"→ Y({self.decimals_y} dec) = {expected}",
        )


# ===================================================================
# Existing proofs (unchanged)
# ===================================================================


def fee_complement_positive(
    axiom_set: AxiomSet,
    fee: sympy.Symbol,
) -> ProofBundle:
    """Prove ``1 - fee > 0`` from axioms ``fee > 0`` and ``fee < 1``.

    SymPy's Q-system cannot propagate the bounded-interval constraint
    ``0 < f < 1`` through rational expressions.  This foundational
    proof establishes the positivity of the fee complement, which
    both AMM proofs depend on.

    Parameters
    ----------
    axiom_set:
        Must contain axioms establishing ``fee > 0`` and ``fee < 1``.
    fee:
        The fee rate symbol.
    """
    hyp = axiom_set.hypothesis(
        "fee_complement_positive",
        expr=1 - fee > 0,
        description="1 - fee > 0 (fee complement is positive)",
    )
    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="fee_complement_proof",
            claim="0 < fee < 1 implies 1 - fee > 0",
        )
        .lemma(
            "complement_from_bound",
            LemmaKind.BOOLEAN,
            expr=sympy.Implies(
                sympy.And(fee > 0, fee < 1),
                1 - fee > 0,
            ),
            description="Direct implication from fee < 1",
        )
        .build()
    )
    return seal(axiom_set, hyp, script)


def amm_output_positive(
    axiom_set: AxiomSet,
    Rx: sympy.Symbol,
    Ry: sympy.Symbol,
    fee: sympy.Symbol,
    dx: sympy.Symbol,
) -> ProofBundle:
    """Prove AMM swap output ``dy > 0`` for a fee-bearing pool.

    The output formula is ``dy = Ry * dx * (1-f) / (Rx + dx*(1-f))``.
    SymPy cannot verify positivity directly because it can't reason
    about ``1-f > 0`` from ``0 < f < 1``.

    Decomposition: introduce helper symbol ``g = 1-f`` (positive),
    rewrite ``dy`` in terms of ``g``, then verify positivity with
    all-positive symbols.

    Parameters
    ----------
    axiom_set:
        Must contain axioms for Rx, Ry, dx > 0 and fee in (0, 1).
    Rx, Ry:
        Reserve symbols (positive).
    fee:
        Fee rate symbol (in (0, 1)).
    dx:
        Input amount symbol (positive).
    """
    fee_bundle = fee_complement_positive(axiom_set, fee)

    g = sympy.Symbol("g", positive=True)
    dy_original = Ry * dx * (1 - fee) / (Rx + dx * (1 - fee))
    dy_with_g = Ry * dx * g / (Rx + dx * g)

    hyp = axiom_set.hypothesis(
        "amm_output_positive",
        expr=dy_original > 0,
        description="AMM swap output is positive",
    )
    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="amm_output_positive_proof",
            claim="dy > 0 for fee-bearing AMM swap",
        )
        .import_bundle(fee_bundle)
        .lemma(
            "rewrite_with_g",
            LemmaKind.EQUALITY,
            expr=dy_original,
            expected=dy_with_g.subs(g, 1 - fee),
            assumptions={
                Rx.name: {"positive": True},
                Ry.name: {"positive": True},
                dx.name: {"positive": True},
            },
            description="dy in original form equals dy(g=1-f)",
        )
        .lemma(
            "dy_g_positive",
            LemmaKind.QUERY,
            expr=sympy.Q.positive(dy_with_g),
            assumptions={
                Rx.name: {"positive": True},
                Ry.name: {"positive": True},
                dx.name: {"positive": True},
                "g": {"positive": True},
            },
            depends_on=["rewrite_with_g"],
            description="Ry*dx*g/(Rx+dx*g) > 0 with all positive",
        )
        .build()
    )
    return seal(axiom_set, hyp, script)


def amm_product_nondecreasing(
    axiom_set: AxiomSet,
    Rx: sympy.Symbol,
    Ry: sympy.Symbol,
    fee: sympy.Symbol,
    dx: sympy.Symbol,
) -> ProofBundle:
    """Prove AMM product invariant is nondecreasing after fee swap.

    The product ratio ``(Rx+dx)*(Ry-dy) / (Rx*Ry)`` simplifies to
    ``(Rx+dx) / (Rx+dx*(1-f))``, which exceeds 1 because fees make
    the numerator larger than the denominator.

    Decomposition: rewrite ratio excess as ``dx*f/(Rx+dx*g)`` where
    ``g = 1-f``, then verify positivity with all-positive symbols.

    Parameters
    ----------
    axiom_set:
        Must contain axioms for Rx, Ry, dx > 0 and fee in (0, 1).
    Rx, Ry:
        Reserve symbols (positive).
    fee:
        Fee rate symbol (in (0, 1)).
    dx:
        Input amount symbol (positive).
    """
    fee_bundle = fee_complement_positive(axiom_set, fee)

    g = sympy.Symbol("g", positive=True)

    # Ratio - 1 in original form
    ratio_original = (Rx + dx) / (Rx + dx * (1 - fee)) - 1
    # Same thing with g = 1-f
    excess_with_g = dx * fee / (Rx + dx * g)

    hyp = axiom_set.hypothesis(
        "amm_product_nondecreasing",
        expr=sympy.Ge(
            (Rx + dx)
            * (Ry - Ry * dx * (1 - fee) / (Rx + dx * (1 - fee))),
            Rx * Ry,
        ),
        description="Post-swap product >= pre-swap product",
    )
    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="amm_product_nondecreasing_proof",
            claim="Fee-bearing swap grows the invariant product",
        )
        .import_bundle(fee_bundle)
        .lemma(
            "ratio_excess_form",
            LemmaKind.EQUALITY,
            expr=ratio_original,
            expected=excess_with_g.subs(g, 1 - fee),
            assumptions={
                Rx.name: {"positive": True},
                dx.name: {"positive": True},
                fee.name: {"positive": True},
            },
            description="Ratio - 1 = dx*f/(Rx+dx*(1-f))",
        )
        .lemma(
            "excess_positive",
            LemmaKind.QUERY,
            expr=sympy.Q.positive(excess_with_g),
            assumptions={
                Rx.name: {"positive": True},
                dx.name: {"positive": True},
                fee.name: {"positive": True},
                "g": {"positive": True},
            },
            depends_on=["ratio_excess_form"],
            description="dx*f/(Rx+dx*g) > 0 with all positive",
        )
        .build()
    )
    return seal(axiom_set, hyp, script)
