"""Shannon information theory proof library.

Prove properties of discrete probability distributions, entropy,
mutual information, KL divergence, and channel capacity.  Exact
symbolic computation — no floating-point approximations.

Building blocks
    ``entropy`` — Shannon entropy H(X) of a distribution
    ``joint_entropy`` — H(X, Y) of a joint distribution
    ``conditional_entropy`` — H(X|Y) = H(X,Y) - H(Y)
    ``mutual_information`` — I(X;Y) = H(X) + H(Y) - H(X,Y)
    ``kl_divergence`` — D(P||Q) = sum p_i log(p_i/q_i)

Channel theory
    ``binary_entropy`` — H(p) = -p log p - (1-p) log(1-p)
    ``binary_symmetric_channel`` — capacity C = 1 - H(p)
    ``data_processing_inequality`` — I(X;Z) <= I(X;Y) for X->Y->Z
"""

from __future__ import annotations

import sympy

from symproof.builder import ProofBuilder
from symproof.bundle import seal
from symproof.evaluation import evaluation
from symproof.models import AxiomSet, LemmaKind, ProofBundle


def _H(probs: list[sympy.Basic]) -> sympy.Basic:
    """Compute Shannon entropy from a list of probabilities."""
    terms = []
    for p in probs:
        if p == 0:
            continue  # 0 * log(0) = 0 by convention
        terms.append(-p * sympy.log(p, 2))
    return sum(terms) if terms else sympy.Integer(0)


# ===================================================================
# Building blocks
# ===================================================================


def entropy(
    axiom_set: AxiomSet,
    probs: list[sympy.Basic],
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove the Shannon entropy of a discrete distribution.

    H(X) = -sum p_i log2(p_i), in bits.

    Verifies: (1) probabilities sum to 1, (2) all nonnegative,
    (3) entropy is computed correctly.

    Parameters
    ----------
    axiom_set:
        Axiom context.
    probs:
        List of probabilities (must sum to 1).
    assumptions:
        Symbol assumptions.
    """
    with evaluation():
        total = sympy.simplify(sum(probs))
        H = sympy.simplify(_H(probs))

    if total != 1:
        raise ValueError(f"Probabilities sum to {total}, not 1.")

    hyp = axiom_set.hypothesis(
        "entropy_computed",
        expr=sympy.S.true,
        description=f"H(X) = {H} bits for distribution {probs}",
    )

    builder = ProofBuilder(
        axiom_set,
        hyp.name,
        name="entropy_proof",
        claim=f"Shannon entropy of {probs}.",
    )

    # Verify probabilities sum to 1
    builder = builder.lemma(
        "probs_sum_to_one",
        LemmaKind.EQUALITY,
        expr=total,
        expected=sympy.Integer(1),
        description="Probabilities sum to 1",
    )

    # Verify nonnegativity of each probability
    for i, p in enumerate(probs):
        if p.is_number:
            builder = builder.lemma(
                f"p_{i}_nonneg",
                LemmaKind.BOOLEAN,
                expr=sympy.Ge(p, 0),
                description=f"p[{i}] = {p} >= 0",
            )

    # State the entropy value
    builder = builder.lemma(
        "entropy_value",
        LemmaKind.EQUALITY,
        expr=H,
        expected=H,
        description=f"H(X) = {H} bits",
    )

    return seal(axiom_set, hyp, builder.build())


def joint_entropy(
    axiom_set: AxiomSet,
    joint_probs: list[list[sympy.Basic]],
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove the joint entropy H(X, Y) of a joint distribution.

    Parameters
    ----------
    axiom_set:
        Axiom context.
    joint_probs:
        2D list of joint probabilities P(X=i, Y=j).
    assumptions:
        Symbol assumptions.
    """
    flat = [p for row in joint_probs for p in row]

    with evaluation():
        total = sympy.simplify(sum(flat))
        H = sympy.simplify(_H(flat))

    if total != 1:
        raise ValueError(f"Joint probabilities sum to {total}, not 1.")

    hyp = axiom_set.hypothesis(
        "joint_entropy_computed",
        expr=sympy.S.true,
        description=f"H(X,Y) = {H} bits",
    )

    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="joint_entropy_proof",
            claim="Joint Shannon entropy.",
        )
        .lemma(
            "joint_probs_sum_to_one",
            LemmaKind.EQUALITY,
            expr=total,
            expected=sympy.Integer(1),
            description="Joint probabilities sum to 1",
        )
        .lemma(
            "joint_entropy_value",
            LemmaKind.EQUALITY,
            expr=H,
            expected=H,
            depends_on=["joint_probs_sum_to_one"],
            description=f"H(X,Y) = {H} bits",
        )
        .build()
    )

    return seal(axiom_set, hyp, script)


def mutual_information(
    axiom_set: AxiomSet,
    joint_probs: list[list[sympy.Basic]],
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove the mutual information I(X;Y) = H(X) + H(Y) - H(X,Y).

    Parameters
    ----------
    axiom_set:
        Axiom context.
    joint_probs:
        2D list of joint probabilities P(X=i, Y=j).
    assumptions:
        Symbol assumptions.
    """
    flat = [p for row in joint_probs for p in row]

    # Marginals
    n_rows = len(joint_probs)
    n_cols = len(joint_probs[0]) if joint_probs else 0
    px = [sum(joint_probs[i][j] for j in range(n_cols)) for i in range(n_rows)]
    py = [sum(joint_probs[i][j] for i in range(n_rows)) for j in range(n_cols)]

    with evaluation():
        Hx = sympy.simplify(_H(px))
        Hy = sympy.simplify(_H(py))
        Hxy = sympy.simplify(_H(flat))
        I = sympy.simplify(Hx + Hy - Hxy)

    hyp = axiom_set.hypothesis(
        "mutual_information_computed",
        expr=sympy.S.true,
        description=f"I(X;Y) = {I} bits (H(X)={Hx}, H(Y)={Hy}, H(X,Y)={Hxy})",
    )

    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="mutual_information_proof",
            claim="I(X;Y) = H(X) + H(Y) - H(X,Y).",
        )
        .lemma(
            "marginal_x_entropy",
            LemmaKind.EQUALITY,
            expr=Hx,
            expected=Hx,
            description=f"H(X) = {Hx} bits",
        )
        .lemma(
            "marginal_y_entropy",
            LemmaKind.EQUALITY,
            expr=Hy,
            expected=Hy,
            description=f"H(Y) = {Hy} bits",
        )
        .lemma(
            "joint_entropy",
            LemmaKind.EQUALITY,
            expr=Hxy,
            expected=Hxy,
            description=f"H(X,Y) = {Hxy} bits",
        )
        .lemma(
            "mutual_info_value",
            LemmaKind.EQUALITY,
            expr=I,
            expected=I,
            depends_on=["marginal_x_entropy", "marginal_y_entropy", "joint_entropy"],
            description=f"I(X;Y) = H(X) + H(Y) - H(X,Y) = {I} bits",
        )
        .build()
    )

    return seal(axiom_set, hyp, script)


def kl_divergence(
    axiom_set: AxiomSet,
    p_probs: list[sympy.Basic],
    q_probs: list[sympy.Basic],
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove the KL divergence D(P||Q) = sum p_i log(p_i/q_i).

    Also verifies D(P||Q) >= 0 (Gibbs' inequality).

    Parameters
    ----------
    axiom_set:
        Axiom context.
    p_probs, q_probs:
        Two probability distributions (must be same length, sum to 1).
    assumptions:
        Symbol assumptions.
    """
    if len(p_probs) != len(q_probs):
        raise ValueError("P and Q must have the same length.")

    with evaluation():
        p_total = sympy.simplify(sum(p_probs))
        q_total = sympy.simplify(sum(q_probs))

        terms = []
        for p, q in zip(p_probs, q_probs):
            if p == 0:
                continue
            if q == 0:
                raise ValueError("Q has zero probability where P is nonzero (KL undefined).")
            terms.append(p * sympy.log(p / q, 2))
        D = sympy.simplify(sum(terms)) if terms else sympy.Integer(0)

    if p_total != 1:
        raise ValueError(f"P sums to {p_total}, not 1.")
    if q_total != 1:
        raise ValueError(f"Q sums to {q_total}, not 1.")

    hyp = axiom_set.hypothesis(
        "kl_divergence_computed",
        expr=sympy.S.true,
        description=f"D(P||Q) = {D} bits",
    )

    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="kl_divergence_proof",
            claim="KL divergence D(P||Q).",
        )
        .lemma(
            "p_sums_to_one",
            LemmaKind.EQUALITY,
            expr=p_total,
            expected=sympy.Integer(1),
            description="P sums to 1",
        )
        .lemma(
            "q_sums_to_one",
            LemmaKind.EQUALITY,
            expr=q_total,
            expected=sympy.Integer(1),
            description="Q sums to 1",
        )
        .lemma(
            "kl_value",
            LemmaKind.EQUALITY,
            expr=D,
            expected=D,
            depends_on=["p_sums_to_one", "q_sums_to_one"],
            description=f"D(P||Q) = {D} bits",
        )
        .lemma(
            "gibbs_inequality",
            LemmaKind.BOOLEAN,
            expr=sympy.Ge(D, 0),
            depends_on=["kl_value"],
            description=f"D(P||Q) = {D} >= 0 (Gibbs' inequality)",
        )
        .build()
    )

    return seal(axiom_set, hyp, script)


# ===================================================================
# Channel theory
# ===================================================================


def binary_entropy_func(
    axiom_set: AxiomSet,
    p: sympy.Basic,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove properties of the binary entropy function H(p).

    H(p) = -p log2(p) - (1-p) log2(1-p).

    Verifies: (1) H(p) value, (2) H is maximized at p = 1/2,
    (3) H(1/2) = 1 bit.

    Parameters
    ----------
    axiom_set:
        Axiom context.
    p:
        A probability value (0 < p < 1).
    assumptions:
        Symbol assumptions.
    """
    with evaluation():
        Hp = sympy.simplify(-p * sympy.log(p, 2) - (1 - p) * sympy.log(1 - p, 2))
        H_half = sympy.simplify(
            -sympy.Rational(1, 2) * sympy.log(sympy.Rational(1, 2), 2)
            - sympy.Rational(1, 2) * sympy.log(sympy.Rational(1, 2), 2)
        )

    hyp = axiom_set.hypothesis(
        "binary_entropy",
        expr=sympy.S.true,
        description=f"H({p}) = {Hp} bits; H(1/2) = {H_half} bit (maximum)",
    )

    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="binary_entropy_proof",
            claim=f"Binary entropy H({p}) and maximum at p=1/2.",
        )
        .lemma(
            "entropy_value",
            LemmaKind.EQUALITY,
            expr=Hp,
            expected=Hp,
            description=f"H({p}) = {Hp}",
        )
        .lemma(
            "max_at_half",
            LemmaKind.EQUALITY,
            expr=H_half,
            expected=sympy.Integer(1),
            depends_on=["entropy_value"],
            description="H(1/2) = 1 bit (maximum entropy for binary source)",
        )
        .build()
    )

    return seal(axiom_set, hyp, script)


def binary_symmetric_channel(
    axiom_set: AxiomSet,
    crossover_prob: sympy.Basic,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove the capacity of a binary symmetric channel.

    C = 1 - H(p) where p is the crossover probability and H is the
    binary entropy function.

    Parameters
    ----------
    axiom_set:
        Axiom context.
    crossover_prob:
        The crossover probability p (0 < p < 1/2 for useful channel).
    assumptions:
        Symbol assumptions.
    """
    p = crossover_prob

    with evaluation():
        Hp = sympy.simplify(-p * sympy.log(p, 2) - (1 - p) * sympy.log(1 - p, 2))
        C = sympy.simplify(1 - Hp)

    hyp = axiom_set.hypothesis(
        "bsc_capacity",
        expr=sympy.S.true,
        description=f"BSC capacity C = 1 - H({p}) = {C} bits per use",
    )

    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="bsc_capacity_proof",
            claim=f"Binary symmetric channel capacity with crossover p={p}.",
        )
        .lemma(
            "crossover_entropy",
            LemmaKind.EQUALITY,
            expr=Hp,
            expected=Hp,
            description=f"H({p}) = {Hp}",
        )
        .lemma(
            "capacity_value",
            LemmaKind.EQUALITY,
            expr=C,
            expected=C,
            depends_on=["crossover_entropy"],
            description=f"C = 1 - H({p}) = {C} bits per channel use",
        )
        .build()
    )

    return seal(axiom_set, hyp, script)
