#!/usr/bin/env python3
"""DIP priority update is equivalent to heavy ball on soft backpressure.

Scenario
--------
Zargham et al. define the Discounted Integral Priority (DIP) routing
update as a time-discounted sum of observed queue lengths (eq. 24):

    lambda_t = sum_{tau=0}^{t} alpha^{t-tau} q_tau

which admits the recurrence (eq. 25):

    lambda_{t+1} = alpha * lambda_t + q_{t+1}

Lemma 1 shows this is algebraically equivalent to:

    lambda_{t+1} = lambda_t + Delta_q_t + alpha * (lambda_t - lambda_{t-1})

where Delta_q_t = q_{t+1} - q_t is the change in queues induced by
routing with priorities lambda_t.  The alpha term is a momentum
coefficient, revealing DIP as a heavy ball method applied to the soft
backpressure dual.

What this proves
----------------
Pure algebraic equivalence between the DIP recurrence (eq. 25) and
the heavy ball form (eq. 26).  Three lemmas:

1. The recurrence at consecutive times yields eq. 25 and eq. 27.
2. Subtracting gives the heavy ball form (eq. 26).
3. The momentum coefficient alpha matches the discounting factor.

What this does NOT prove
------------------------
- That the heavy ball method converges (see 04_dual_convergence.py)
- That queue stability follows (see 05_queue_stability.py)
- Any property of the stochastic gradient or noise

Run: uv run python -m symproof.library.examples.dip_routing.01_heavy_ball_equivalence
"""

import sympy

from symproof import Axiom, AxiomSet, LemmaKind, ProofBuilder, seal

# ─── Symbols ────────────────────────────────────────────────────
alpha = sympy.Symbol("alpha")  # discounting factor in [0, 1)

# Priority and queue variables at three consecutive time steps.
# We treat these as abstract scalars — the proof is per-component.
lam_t = sympy.Symbol("lambda_t")          # lambda at time t
lam_t1 = sympy.Symbol("lambda_{t+1}")     # lambda at time t+1
lam_tm1 = sympy.Symbol("lambda_{t-1}")    # lambda at time t-1
q_t = sympy.Symbol("q_t")                 # queue length at time t
q_t1 = sympy.Symbol("q_{t+1}")            # queue length at time t+1

# Derived: queue change
Delta_q = sympy.Symbol("Delta_q_t")

# ─── Axioms ─────────────────────────────────────────────────────
# The DIP recurrence (eq. 25) applied at times t and t+1.
axioms = AxiomSet(
    name="dip_recurrence",
    axioms=(
        Axiom(
            name="recurrence_at_t",
            expr=sympy.Eq(lam_t, alpha * lam_tm1 + q_t),
            description="DIP update at time t: lambda_t = alpha * lambda_{t-1} + q_t",
        ),
        Axiom(
            name="recurrence_at_t1",
            expr=sympy.Eq(lam_t1, alpha * lam_t + q_t1),
            description="DIP update at time t+1: lambda_{t+1} = alpha * lambda_t + q_{t+1}",
        ),
        Axiom(
            name="queue_change_def",
            expr=sympy.Eq(Delta_q, q_t1 - q_t),
            description="Delta_q_t := q_{t+1} - q_t (change in queues)",
        ),
    ),
)

# ─── Hypothesis ─────────────────────────────────────────────────
# Lemma 1 (eq. 26): the DIP update in heavy ball form.
heavy_ball_form = sympy.Eq(
    lam_t1,
    lam_t + Delta_q + alpha * (lam_t - lam_tm1),
)

hypothesis = axioms.hypothesis(
    name="dip_is_heavy_ball",
    expr=heavy_ball_form,
    description=(
        "Lemma 1: DIP recurrence is equivalent to "
        "lambda_{t+1} = lambda_t + Delta_q_t + alpha*(lambda_t - lambda_{t-1})"
    ),
)

# ─── Proof ──────────────────────────────────────────────────────
# Strategy: expand the RHS of eq. 26 using the axioms and show it
# equals the RHS of eq. 27 (the recurrence at t+1).
#
# RHS of heavy ball form:
#   lambda_t + Delta_q_t + alpha*(lambda_t - lambda_{t-1})
# = lambda_t + (q_{t+1} - q_t) + alpha*lambda_t - alpha*lambda_{t-1}
#
# From recurrence_at_t: lambda_t = alpha*lambda_{t-1} + q_t
# so alpha*lambda_{t-1} = lambda_t - q_t
#
# Substituting:
# = lambda_t + q_{t+1} - q_t + alpha*lambda_t - (lambda_t - q_t)
# = lambda_t + q_{t+1} - q_t + alpha*lambda_t - lambda_t + q_t
# = q_{t+1} + alpha*lambda_t
# = RHS of recurrence_at_t1   ✓

# Expand the heavy ball RHS, substituting Delta_q and the recurrence.
heavy_ball_rhs = lam_t + Delta_q + alpha * (lam_t - lam_tm1)

# The recurrence at t+1 says lambda_{t+1} = alpha * lambda_t + q_{t+1}
recurrence_rhs = alpha * lam_t + q_t1

# Step 1: substitute Delta_q = q_{t+1} - q_t and expand
rhs_after_delta = sympy.expand(heavy_ball_rhs.subs(Delta_q, q_t1 - q_t))

# Step 2: compute difference (heavy ball RHS) - (recurrence RHS)
#   = lam_t + q_t1 - q_t + alpha*lam_t - alpha*lam_tm1 - alpha*lam_t - q_t1
#   = lam_t - q_t - alpha*lam_tm1
# Then substitute lam_t = alpha*lam_tm1 + q_t (recurrence at t) → 0
diff_raw = sympy.expand(rhs_after_delta - recurrence_rhs)
diff_substituted = sympy.expand(
    diff_raw.subs(lam_t, alpha * lam_tm1 + q_t)
)

script = (
    ProofBuilder(
        axioms,
        hypothesis.name,
        name="heavy_ball_equivalence",
        claim=(
            "The DIP recurrence lambda_{t+1} = alpha*lambda_t + q_{t+1} "
            "is equivalent to the heavy ball form "
            "lambda_{t+1} = lambda_t + Delta_q_t + alpha*(lambda_t - lambda_{t-1})."
        ),
    )
    .lemma(
        "expand_heavy_ball_rhs",
        LemmaKind.EQUALITY,
        expr=rhs_after_delta,
        expected=alpha * lam_t - alpha * lam_tm1 + lam_t + q_t1 - q_t,
        description=(
            "Expand heavy ball RHS after substituting Delta_q_t = q_{t+1} - q_t."
        ),
    )
    .lemma(
        "difference_before_axiom",
        LemmaKind.EQUALITY,
        expr=diff_raw,
        expected=lam_t - q_t - alpha * lam_tm1,
        depends_on=["expand_heavy_ball_rhs"],
        description=(
            "Subtract recurrence RHS (alpha*lambda_t + q_{t+1}) from expanded "
            "heavy ball RHS.  Residual: lambda_t - q_t - alpha*lambda_{t-1}."
        ),
    )
    .lemma(
        "apply_recurrence_axiom",
        LemmaKind.EQUALITY,
        expr=diff_substituted,
        expected=sympy.Integer(0),
        depends_on=["difference_before_axiom"],
        description=(
            "Substitute lambda_t = alpha*lambda_{t-1} + q_t (recurrence at t) "
            "into the residual.  Result: 0.  QED."
        ),
    )
    .build()
)

bundle = seal(axioms, hypothesis, script)

# ─── Output ─────────────────────────────────────────────────────
print("Lemma 1: DIP recurrence ↔ heavy ball equivalence")
print("  Paper: Zargham, Ribeiro & Jadbabaie, Globecom 2014, eq. 25–26")
print()
print("  DIP recurrence (eq. 25):")
print("    lambda_{t+1} = alpha * lambda_t + q_{t+1}")
print()
print("  Heavy ball form (eq. 26):")
print("    lambda_{t+1} = lambda_t + Delta_q_t + alpha*(lambda_t - lambda_{t-1})")
print()
print(f"  Status: {bundle.proof_result.status.value}")
print(f"  Hash:   {bundle.bundle_hash[:24]}...")
for lr in bundle.proof_result.lemma_results:
    mark = "pass" if lr.passed else "FAIL"
    print(f"    [{mark}] {lr.lemma_name}")
print()
print("  This proves: algebraic equivalence between update rules.")
print("  This does NOT prove: convergence or stability (see 04, 05).")
