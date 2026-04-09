---
name: symproof-base
description: Background knowledge for symproof — deterministic proof writing with SymPy. Apply when working in this repo.
user-invocable: false
---

# symproof internals

## Core philosophy: loosely coupled proofs, not monoliths

symproof is designed for **collective coverage** — many small, focused proofs that each address one property of a system. This is more powerful and more scalable than complex proofs that try to establish everything at once.

A single system should have MANY independent proofs:
- Stability proof (does the system converge?)
- Controllability proof (can we steer it?)
- Observability proof (can we measure it?)
- Invariant proof (is this quantity conserved?)
- Convexity proof (is the optimization well-posed?)

Each proof is sealed independently with its own hash. The hashes go into a **requirements traceability matrix** — each proof hash maps to a specific requirement. A reviewer can verify any proof independently without understanding the others.

Composition via `import_bundle` is for when one property LOGICALLY DEPENDS on another (e.g., uniqueness depends on strict convexity). It is NOT for bundling unrelated properties together. If stability and controllability are independent properties, prove them separately.

**symproof covers the symbolic analysis layer only.** It does not replace simulation (Monte Carlo, HIL) or code verification (static analysis, formal methods). A complete V&V program needs all three layers, with symproof's proof hashes linking the symbolic layer to the requirements alongside simulation results and audit findings.

## Evidence chain

```
AxiomSet → axiom_set_hash
  → Hypothesis (carries axiom_set_hash)
  → ProofScript (carries axiom_set_hash + imported_bundles + lemmas)
    → verify_proof(script) → ProofResult
      → seal(axiom_set, hypothesis, script) → ProofBundle(bundle_hash)
```

`seal()` is the ONLY path to a ProofBundle. It enforces:
1. Axiom hash matches
2. Hypothesis name matches script target
3. Lemma assumptions don't contradict axioms
4. Imported bundles share the same axiom set
5. All lemmas pass verification (imports re-verified, no trust shortcut)

## Verification strategies (LemmaKind)

| Kind | Checks | Use when |
|------|--------|----------|
| EQUALITY | `simplify(expr - expected) == 0`, .doit() fallback | Algebraic identities, series, closed forms |
| BOOLEAN | `simplify → refine → proof-by-contradiction` | Implications, inequalities, relational |
| QUERY | `sympy.ask(expr, Q-context)` | Positivity, type predicates (irrational, integer) |
| COORDINATE_TRANSFORM | Round-trip + transform + simplify/trigsimp | Polar, hyperbolic, body-frame transforms |

## Advisory system

Results carry `advisories: tuple[str, ...]` when verification passes through known SymPy limitations. Every advisory is a flag for human review — it doesn't mean the proof is wrong, it means an engineer should inspect this step.

## Composition (import_bundle)

Use `import_bundle` ONLY when there is a logical dependency between proofs. The imported bundle's proof is re-verified at seal time.

```python
# YES: uniqueness depends on strong convexity
unique = unique_minimizer(axioms, f, vars, m)  # internally imports strongly_convex

# NO: don't bundle unrelated properties
# Instead: prove stability and controllability SEPARATELY, trace both to requirements
```

## Library catalog

### core — SymPy gap workarounds
- `max_ge_first(ax, a, b)` — Max(a,b) >= a
- `piecewise_collapse(ax, expr, cond, fb, assumptions)` — Piecewise branch collapse

### control — stability, controllability, observability
- `hurwitz_second_order / hurwitz_third_order` — Routh-Hurwitz
- `closed_loop_stability(ax, G_n, G_d, C_n, C_d, s)` — plant+controller → Hurwitz
- `lyapunov_stability(ax, A, P, Q)` — verify given Lyapunov equation
- `lyapunov_from_system(ax, A)` — CONSTRUCT P and prove PD
- `gain_margin(ax, coeffs, K, s)` — gain below critical
- `controllability_rank(ax, A, B)` — Gramian det ≠ 0
- `observability_rank(ax, A, C)` — Gramian det ≠ 0
- `quadratic_invariant(ax, states, dots, V)` — dV/dt = 0

### convex — optimization problem certification
- `convex_scalar / convex_hessian / strongly_convex` — convexity proofs
- `conjugate_function` — compute and verify Fenchel conjugate
- `convex_sum / convex_composition` — DCP composition rules
- `unique_minimizer` — strictly convex → unique (imports strongly_convex)
- `gp_to_convex` — geometric program log-transform

### defi — DeFi mechanism analysis
- `fee_complement_positive` — 1-f > 0 from bounded interval
- `amm_output_positive / amm_product_nondecreasing` — AMM properties
- `mul_down / mul_up / div_down / div_up` — directed rounding
- `rounding_bias_lemma / rounding_gap_lemma / chain_error_bound` — error analysis
- `phantom_overflow_check / no_phantom_overflow_check` — uint256 safety

## Known SymPy limitations

1. **Bounded intervals**: Q-system can't reason about `0 < f < 1 → 1-f > 0`. Workaround: helper symbol `g = 1-f`.
2. **Domain-ignoring simplify**: cancels across singularities. Advisory system flags this.
3. **Piecewise/Max/Min**: not simplified under assumptions. Library workarounds available.
