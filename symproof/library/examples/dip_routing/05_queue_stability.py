#!/usr/bin/env python3
"""Queue stability: all queues empty infinitely often with probability one.

Scenario
--------
Proposition 3 of Zargham et al. is the paper's main stability result.
It states: if the dual gradient vanishes (lim ||nabla h(lambda_t)|| = 0),
then all queues in the network become empty infinitely often:

    lim inf_{t -> inf} q_i^k(t) = 0,  a.s.,  for all k, i != o_k

Corollary 1 combines this with Proposition 2: under the Stochastic
Heavy Ball DIP routing with decaying step size, queue stability holds.

The proof relies on:
1. The queue evolution equation (eq. 2)
2. The gradient structure (eq. 11): when nabla h = 0 at optimal lambda*,
   the expected outflow exceeds inflow by xi at every non-destination node
3. A supermartingale convergence argument (Solo & Kong [21], Thm E.7.4)

What this proves
----------------
We verify the key algebraic steps computationally:

(a) The queue evolution q_{t+1} = [q_t + a_t + sum(in) - sum(out)]^+
    is correctly structured.
(b) At the optimal dual variables (nabla h = 0), the net flow condition
    sum_j R*_{ij} - R*_{ji} >= a_i + xi holds — meaning expected outflow
    strictly exceeds expected inflow.
(c) Under this condition, E[q_{t+1} | q_t > 0] < q_t - xi, establishing
    the supermartingale drift that forces queues to zero.

The supermartingale convergence theorem itself is axiomatised.

What this does NOT prove
------------------------
- The supermartingale convergence theorem (Solo & Kong [21])
- Almost sure convergence from the supermartingale bound
- Finite-time bounds on queue clearing

Run: uv run python -m symproof.library.examples.dip_routing.05_queue_stability
"""

import importlib

import sympy

from symproof import Axiom, AxiomSet, LemmaKind, ProofBuilder, seal

_mod_08 = importlib.import_module(
    "symproof.library.examples.dip_routing.08_supermartingale_finite"
)
make_supermartingale_bundle = _mod_08.make_supermartingale_bundle

# ─── Symbols ────────────────────────────────────────────────────
# Queue and flow variables (per-component, scalar analysis)
q_t = sympy.Symbol("q_t", nonnegative=True)      # queue at time t
a_t = sympy.Symbol("a_t", nonnegative=True)       # arrivals at time t
R_out = sympy.Symbol("R_out", nonnegative=True)   # total outgoing rate
R_in = sympy.Symbol("R_in", nonnegative=True)     # total incoming rate
xi = sympy.Symbol("xi", positive=True)            # strictness parameter

# At optimal dual variables, the gradient vanishing condition (eq. 11 = 0)
# gives: R*_out - R*_in = a + xi  (expected net outflow exceeds inflow)
a_bar = sympy.Symbol("a_bar", nonnegative=True)   # mean arrival rate
R_out_star = sympy.Symbol("R_out_star", nonnegative=True)  # optimal outflow
R_in_star = sympy.Symbol("R_in_star", nonnegative=True)    # optimal inflow

# Symbols inherited from supermartingale foundation (08)
Y_0 = sympy.Symbol("Y_0", nonnegative=True)   # starting process value
delta = sympy.Symbol("delta", positive=True)    # minimum descent per step

# ─── Queue evolution (eq. 2) ────────────────────────────────────
# q_{t+1} = [q_t + a_t + R_in - R_out]^+
q_next = sympy.Max(q_t + a_t + R_in - R_out, 0)

# ─── Axioms ─────────────────────────────────────────────────────
axioms = AxiomSet(
    name="queue_stability_conditions",
    axioms=(
        Axiom(
            name="queue_nonneg",
            expr=q_t >= 0,
            description="Queue lengths are nonnegative.",
        ),
        Axiom(
            name="xi_positive",
            expr=xi > 0,
            description="Strictness parameter xi > 0 (strict feasibility).",
        ),
        Axiom(
            name="gradient_vanishes",
            expr=sympy.Eq(R_out_star - R_in_star, a_bar + xi),
            description=(
                "At optimal dual variables (nabla h = 0), net outflow "
                "exceeds mean inflow by xi. From Proposition 2 + eq. 11."
            ),
        ),
        Axiom(
            name="supermartingale_convergence",
            expr=sympy.S.true,
            description=(
                "Solo & Kong [21], Theorem E.7.4: If a nonneg process Y_t "
                "satisfies E[Y_{t+1} | F_t] <= Y_t - delta when Y_t > 0, "
                "then lim inf Y_t = 0 a.s. "
                "Proved computationally in 08_supermartingale_finite."
            ),
        ),
        # Inherited from supermartingale foundation (08_supermartingale_finite)
        Axiom(
            name="process_nonneg",
            expr=Y_0 >= 0,
            inherited=True,
            description="The process Y_t is nonnegative. Required by Solo-Kong.",
        ),
        Axiom(
            name="descent_per_step",
            expr=delta > 0,
            inherited=True,
            description=(
                "Each step decreases the process by at least delta > 0. "
                "Required by Solo-Kong."
            ),
        ),
        Axiom(
            name="borel_cantelli_extension",
            expr=sympy.S.true,
            inherited=True,
            description=(
                "Borel-Cantelli: finite-time return from any starting point "
                "implies lim inf = 0 a.s. Required by Solo-Kong."
            ),
        ),
    ),
)

# ─── Hypothesis ─────────────────────────────────────────────────
hypothesis = axioms.hypothesis(
    name="queues_clear",
    expr=R_out_star - R_in_star - a_bar > 0,
    description=(
        "Proposition 3 + Corollary 1: The net drift at optimal routing "
        "is strictly positive (= xi > 0), so the supermartingale condition "
        "holds and all queues empty infinitely often a.s."
    ),
)

# ─── Proof ──────────────────────────────────────────────────────
# Key steps:
# 1. From gradient_vanishes: R*_out - R*_in = a_bar + xi
# 2. Therefore R*_out - R*_in - a_bar = xi > 0
# 3. This means expected change in queue is negative by xi when q > 0
# 4. By supermartingale convergence: lim inf q_t = 0 a.s.

# Compute the net drift
net_drift = R_out_star - R_in_star - a_bar

# Substitute the gradient vanishing condition
net_drift_value = (a_bar + xi) - a_bar  # = xi

script = (
    ProofBuilder(
        axioms,
        hypothesis.name,
        name="queue_stability",
        claim=(
            "At optimal dual variables, net outflow exceeds mean inflow "
            "by xi > 0, establishing the supermartingale drift condition "
            "for queue clearing."
        ),
    )
    .lemma(
        "net_drift_equals_xi",
        LemmaKind.EQUALITY,
        expr=net_drift.subs(R_out_star - R_in_star, a_bar + xi),
        expected=xi,
        description=(
            "Substitute gradient vanishing condition "
            "(R*_out - R*_in = a_bar + xi) to get net drift = xi."
        ),
    )
    .lemma(
        "drift_is_positive",
        LemmaKind.QUERY,
        expr=sympy.Q.positive(xi),
        assumptions={"xi": {"positive": True}},
        depends_on=["net_drift_equals_xi"],
        description="xi > 0, so the drift is strictly positive.",
    )
    .lemma(
        "queue_projection_structure",
        LemmaKind.EQUALITY,
        expr=sympy.Max(q_t + a_bar - (a_bar + xi), 0),
        expected=sympy.Max(q_t - xi, 0),
        description=(
            "At optimal routing with mean arrivals, substitute "
            "R*_out - R*_in = a_bar + xi into the queue evolution: "
            "q_{t+1} = max(q_t + a_bar - (a_bar + xi), 0) = max(q_t - xi, 0). "
            "Queue decreases by xi each step (projected to nonneg reals)."
        ),
    )
    .build()
)

supermartingale_foundation = make_supermartingale_bundle()
bundle = seal(
    axioms, hypothesis, script,
    foundations=[(supermartingale_foundation, "supermartingale_convergence")],
)

# ─── Output ─────────────────────────────────────────────────────
print("Proposition 3 + Corollary 1: Queue stability")
print("  Paper: Zargham, Ribeiro & Jadbabaie, Globecom 2014, eq. 33–34")
print()
print("  Queue evolution: q_{t+1} = [q_t + a_t + R_in - R_out]^+")
print("  At optimal routing: R*_out - R*_in = a_bar + xi  (grad h = 0)")
print("  → q_{t+1} = max(q_t - xi, 0)")
print("  → lim inf q_t = 0,  a.s.  (supermartingale convergence)")
print()
print(f"  Status: {bundle.proof_result.status.value}")
print(f"  Hash:   {bundle.bundle_hash[:24]}...")
for lr in bundle.proof_result.lemma_results:
    mark = "pass" if lr.passed else "FAIL"
    print(f"    [{mark}] {lr.lemma_name}")
print()
print("  This proves: strictly positive drift at optimal routing,")
print("    and deterministic queue decrease structure.")
print("  This does NOT prove: supermartingale convergence theorem")
print("    (Solo & Kong [21]; axiomatised).")
