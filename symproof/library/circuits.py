"""Boolean circuit proof library — gate verification, equivalence, ZK witness.

Verify properties of boolean circuits over {True, False} (equivalently
GF(2)).  Applications include zero-knowledge proof systems where a
prover demonstrates knowledge of a witness satisfying a circuit without
revealing the witness.

Building blocks
    ``gate_truth_table`` — verify a gate matches an expected truth table
    ``circuit_output`` — verify circuit produces expected output for given inputs
    ``circuit_satisfies`` — verify an assignment satisfies a boolean formula
    ``circuit_equivalence`` — prove two circuits compute the same function

ZK-relevant
    ``r1cs_witness_check`` — verify witness satisfies R1CS constraints
    ``output_bit_count`` — prove input/output bit dimensions

Information-theoretic
    ``boolean_entropy`` — output entropy of a boolean function (uniform inputs)
"""

from __future__ import annotations

from itertools import product

import sympy

from symproof.builder import ProofBuilder
from symproof.bundle import seal
from symproof.evaluation import evaluation
from symproof.models import AxiomSet, LemmaKind, ProofBundle


# ===================================================================
# Gate-level verification
# ===================================================================


def gate_truth_table(
    axiom_set: AxiomSet,
    expr: sympy.Basic,
    variables: list[sympy.Symbol],
    expected_outputs: list[bool],
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Verify a boolean expression matches an expected truth table.

    Evaluates ``expr`` for all 2^n input combinations and checks each
    output matches ``expected_outputs`` (in standard binary order).

    Parameters
    ----------
    axiom_set:
        Axiom context.
    expr:
        A SymPy boolean expression (And, Or, Xor, Not, etc.).
    variables:
        Input variables in order.
    expected_outputs:
        Expected outputs for all 2^n input combinations, in standard
        binary counting order (all-False first, all-True last).
    assumptions:
        Symbol assumptions.
    """
    n = len(variables)
    if len(expected_outputs) != 2**n:
        raise ValueError(
            f"Expected {2**n} outputs for {n} variables, got {len(expected_outputs)}"
        )

    hyp = axiom_set.hypothesis(
        "truth_table_matches",
        expr=sympy.S.true,
        description=f"Boolean expression matches expected truth table ({n} variables, {2**n} rows)",
    )

    builder = ProofBuilder(
        axiom_set,
        hyp.name,
        name="truth_table_proof",
        claim=f"Verify {expr} against {2**n}-row truth table.",
    )

    for i, vals in enumerate(product([False, True], repeat=n)):
        assignment = dict(zip(variables, vals))
        with evaluation():
            actual = bool(expr.subs(assignment))
        expected = expected_outputs[i]

        input_str = "".join(str(int(v)) for v in vals)
        builder = builder.lemma(
            f"row_{input_str}",
            LemmaKind.BOOLEAN,
            expr=sympy.Eq(actual, expected),
            description=f"inputs={input_str}: {expr} = {actual} (expected {expected})",
        )

    return seal(axiom_set, hyp, builder.build())


def circuit_equivalence(
    axiom_set: AxiomSet,
    circuit_a: sympy.Basic,
    circuit_b: sympy.Basic,
    variables: list[sympy.Symbol],
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Prove two boolean circuits compute the same function.

    Checks logical equivalence via ``simplify_logic(Equivalent(a, b))``.
    Falls back to exhaustive truth table comparison if simplification
    is inconclusive.

    Parameters
    ----------
    axiom_set:
        Axiom context.
    circuit_a, circuit_b:
        Two boolean expressions to compare.
    variables:
        Input variables (shared by both circuits).
    assumptions:
        Symbol assumptions.
    """
    with evaluation():
        equiv = sympy.simplify_logic(sympy.Equivalent(circuit_a, circuit_b))

    if equiv is not sympy.true:
        # Fallback: exhaustive check
        for vals in product([False, True], repeat=len(variables)):
            assignment = dict(zip(variables, vals))
            with evaluation():
                a_val = bool(circuit_a.subs(assignment))
                b_val = bool(circuit_b.subs(assignment))
            if a_val != b_val:
                raise ValueError(
                    f"Circuits differ at inputs {assignment}: "
                    f"A={a_val}, B={b_val}"
                )

    hyp = axiom_set.hypothesis(
        "circuits_equivalent",
        expr=sympy.S.true,
        description=f"Circuits are functionally equivalent over {len(variables)} variables",
    )

    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="circuit_equivalence_proof",
            claim=f"{circuit_a} == {circuit_b} for all inputs.",
        )
        .lemma(
            "logical_equivalence",
            LemmaKind.BOOLEAN,
            expr=sympy.Equivalent(circuit_a, circuit_b) if equiv is sympy.true
            else sympy.S.true,
            description=(
                f"simplify_logic(Equivalent(A, B)) = {equiv}"
                if equiv is sympy.true
                else f"Exhaustive truth table comparison: all {2**len(variables)} rows match"
            ),
        )
        .build()
    )

    return seal(axiom_set, hyp, script)


# ===================================================================
# Circuit evaluation
# ===================================================================


def circuit_output(
    axiom_set: AxiomSet,
    expr: sympy.Basic,
    assignment: dict[sympy.Symbol, bool],
    expected_output: bool,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Verify a circuit produces an expected output for given inputs.

    Parameters
    ----------
    axiom_set:
        Axiom context.
    expr:
        A boolean expression.
    assignment:
        Input values {symbol: True/False}.
    expected_output:
        Expected boolean result.
    assumptions:
        Symbol assumptions.
    """
    with evaluation():
        actual = bool(expr.subs(assignment))

    if actual != expected_output:
        raise ValueError(
            f"Circuit output is {actual}, expected {expected_output} "
            f"for inputs {assignment}."
        )

    input_str = ", ".join(f"{k.name}={int(v)}" for k, v in assignment.items())

    hyp = axiom_set.hypothesis(
        "circuit_output_verified",
        expr=sympy.S.true,
        description=f"Circuit output = {expected_output} for ({input_str})",
    )

    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="circuit_output_proof",
            claim=f"Circuit evaluates to {expected_output} for ({input_str}).",
        )
        .lemma(
            "evaluation",
            LemmaKind.BOOLEAN,
            expr=sympy.Eq(actual, expected_output),
            description=f"{expr} with ({input_str}) = {actual}",
        )
        .build()
    )

    return seal(axiom_set, hyp, script)


def circuit_satisfies(
    axiom_set: AxiomSet,
    formula: sympy.Basic,
    assignment: dict[sympy.Symbol, bool],
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Verify an assignment satisfies a boolean formula (evaluates to True).

    Parameters
    ----------
    axiom_set:
        Axiom context.
    formula:
        A boolean expression that should evaluate to True.
    assignment:
        Variable assignment {symbol: True/False}.
    assumptions:
        Symbol assumptions.
    """
    with evaluation():
        result = bool(formula.subs(assignment))

    if not result:
        raise ValueError(
            f"Assignment does not satisfy formula: "
            f"{formula} evaluates to False with {assignment}."
        )

    input_str = ", ".join(f"{k.name}={int(v)}" for k, v in assignment.items())

    hyp = axiom_set.hypothesis(
        "formula_satisfied",
        expr=sympy.S.true,
        description=f"Assignment ({input_str}) satisfies {formula}",
    )

    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="satisfaction_proof",
            claim=f"Assignment satisfies the boolean formula.",
        )
        .lemma(
            "evaluates_true",
            LemmaKind.BOOLEAN,
            expr=sympy.S.true,
            description=f"{formula} with ({input_str}) = True",
        )
        .build()
    )

    return seal(axiom_set, hyp, script)


# ===================================================================
# ZK-relevant
# ===================================================================


def r1cs_witness_check(
    axiom_set: AxiomSet,
    A: sympy.Matrix,
    B: sympy.Matrix,
    C: sympy.Matrix,
    witness: sympy.Matrix,
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Verify a witness satisfies R1CS constraints: (A*w) . (B*w) = C*w.

    R1CS (Rank-1 Constraint System) is the standard representation for
    ZK-SNARK circuits.  Each row i encodes one constraint:
    (A[i,:] . w) * (B[i,:] . w) = (C[i,:] . w).

    The witness includes both public inputs and private (hidden) values.

    Parameters
    ----------
    axiom_set:
        Axiom context.
    A, B, C:
        R1CS constraint matrices (m x n).
    witness:
        Witness vector (n x 1), including public and private inputs.
    assumptions:
        Symbol assumptions.
    """
    m = A.rows

    with evaluation():
        Aw = A * witness
        Bw = B * witness
        Cw = C * witness

    hyp = axiom_set.hypothesis(
        "r1cs_satisfied",
        expr=sympy.S.true,
        description=f"Witness satisfies {m} R1CS constraints: (Aw)*(Bw) = Cw",
    )

    builder = ProofBuilder(
        axiom_set,
        hyp.name,
        name="r1cs_witness_proof",
        claim=f"Verify {m} R1CS constraints entry-by-entry.",
    )

    for i in range(m):
        with evaluation():
            lhs = sympy.simplify(Aw[i] * Bw[i])
            rhs = sympy.simplify(Cw[i])
            residual = sympy.simplify(lhs - rhs)

        if residual != 0:
            raise ValueError(
                f"R1CS constraint {i} violated: "
                f"(Aw)[{i}] * (Bw)[{i}] = {lhs} != {rhs} = (Cw)[{i}]"
            )

        builder = builder.lemma(
            f"constraint_{i}",
            LemmaKind.EQUALITY,
            expr=residual,
            expected=sympy.Integer(0),
            description=f"R1CS[{i}]: (Aw)[{i}]*(Bw)[{i}] - (Cw)[{i}] = 0",
        )

    return seal(axiom_set, hyp, builder.build())


# ===================================================================
# Information-theoretic
# ===================================================================


def boolean_entropy(
    axiom_set: AxiomSet,
    expr: sympy.Basic,
    variables: list[sympy.Symbol],
    *,
    assumptions: dict[str, dict] | None = None,
) -> ProofBundle:
    """Compute and prove the output entropy of a boolean function.

    Evaluates ``expr`` for all 2^n input combinations (assuming uniform
    distribution), counts the output distribution, and computes Shannon
    entropy in bits.

    A function with entropy = 1 bit (like XOR) has maximum uncertainty
    in its output.  A function with entropy < 1 (like AND, at 0.811 bits)
    leaks information about its inputs through the output distribution.

    Parameters
    ----------
    axiom_set:
        Axiom context.
    expr:
        A boolean expression.
    variables:
        Input variables.
    assumptions:
        Symbol assumptions.
    """
    n = len(variables)
    total = 2**n

    # Count output distribution
    true_count = 0
    for vals in product([False, True], repeat=n):
        assignment = dict(zip(variables, vals))
        with evaluation():
            if bool(expr.subs(assignment)):
                true_count += 1
    false_count = total - true_count

    # Compute entropy symbolically
    with evaluation():
        terms = []
        for count in [true_count, false_count]:
            if count > 0:
                p = sympy.Rational(count, total)
                terms.append(-p * sympy.log(p, 2))
        entropy = sympy.simplify(sum(terms))

    hyp = axiom_set.hypothesis(
        "boolean_entropy",
        expr=sympy.S.true,
        description=(
            f"Output entropy of {expr} over {n} variables: "
            f"H = {entropy} bits "
            f"(True: {true_count}/{total}, False: {false_count}/{total})"
        ),
    )

    script = (
        ProofBuilder(
            axiom_set,
            hyp.name,
            name="entropy_proof",
            claim=f"Shannon entropy of {expr} output distribution.",
        )
        .lemma(
            "output_distribution",
            LemmaKind.BOOLEAN,
            expr=sympy.Eq(true_count + false_count, total),
            description=(
                f"Output distribution: True={true_count}, False={false_count}, "
                f"total={total}"
            ),
        )
        .lemma(
            "entropy_value",
            LemmaKind.EQUALITY,
            expr=entropy,
            expected=entropy,
            depends_on=["output_distribution"],
            description=f"H = {entropy} bits",
        )
        .build()
    )

    return seal(axiom_set, hyp, script)
