---
name: symproof-base
description: Background knowledge for symproof — deterministic proof writing with SymPy. Apply when working in this repo.
user-invocable: false
---

# symproof internals

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

Results carry `advisories: tuple[str, ...]` when verification passes through:
- Domain-ignoring simplification (division, log, sqrt in EQUALITY)
- `.doit()` fallback (Sum/Product evaluated, not directly simplified)
- `refine()` or negation-check fallback (BOOLEAN)
- Q-system heuristics (every passing QUERY)
- INDETERMINATE (SymPy can't determine truth — not an error, needs different strategy)

## Composition

```python
ProofBuilder(axioms, hyp.name, name="...", claim="...")
    .import_bundle(prior_bundle)    # re-verified at seal time
    .lemma("step", LemmaKind.QUERY, expr=..., assumptions={...})
    .build()
```

`verify_proof(script, trust_imports=True)` skips import re-verification for exploration. `seal()` always re-verifies.

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
- `convex_scalar(ax, f, x)` — f''(x) >= 0
- `convex_hessian(ax, f, vars)` — Hessian PSD via Sylvester
- `strongly_convex(ax, f, vars, m)` — H - mI PSD
- `conjugate_function(ax, f, x, y)` — compute f*, verify convex
- `convex_sum(ax, funcs, weights, vars)` — weighted sum
- `convex_composition(ax, f, g, t, x)` — DCP rule
- `unique_minimizer(ax, f, vars, m)` — strictly convex → unique
- `gp_to_convex(ax, monomials, y, x)` — GP log-transform

### defi — DeFi mechanism analysis
- `fee_complement_positive(ax, fee)` — 1-f > 0 from 0<f<1
- `amm_output_positive(ax, Rx, Ry, fee, dx)` — swap output > 0
- `amm_product_nondecreasing(ax, Rx, Ry, fee, dx)` — product grows
- `mul_down / mul_up / div_down / div_up` — directed rounding ops
- `rounding_bias_lemma / rounding_gap_lemma` — ceil >= floor, gap <= 1
- `chain_error_bound` — cumulative rounding < N
- `phantom_overflow_check / no_phantom_overflow_check` — uint256 safety

## Known SymPy limitations

1. **Bounded intervals**: Q-system can't reason about `0 < f < 1 → 1-f > 0`. Workaround: introduce helper symbol `g = 1-f`, prove `g > 0` separately.
2. **Domain-ignoring simplify**: `simplify((x²-1)/(x-1))` cancels to `x+1` ignoring singularity at x=1. Advisory system flags this.
3. **Piecewise under assumptions**: `simplify` doesn't collapse branches. Workaround: use assumption substitution (library `piecewise_collapse`).
4. **Max/Min properties**: `Max(a,b) >= a` not verifiable directly. Workaround: library `max_ge_first`.

## Key constraints

- All SymPy serialization uses `sympy.srepr()`, never `str()`
- Absolute imports only
- All models are frozen Pydantic BaseModels
- `seal()` is the only path to ProofBundle
- `disprove()` is the only path to Disproof
- Axioms are authoritative — lemma assumptions must not contradict them
