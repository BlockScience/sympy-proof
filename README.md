# symproof

Deterministic proof writing with SymPy.

Declare axioms, bind hypotheses, build lemma chains, and seal reproducible hashed proof bundles. Every sealed proof gets a deterministic SHA-256 hash — same inputs, same hash, every time.

## Verification vs. Validation

The most expensive failures in engineering happen when you **correctly implement the wrong thing**.

- Terra/Luna: code correctly implemented a flawed peg mechanism. $40B lost.
- Beanstalk: governance math was correct Solidity. The economic model was exploitable.
- Mango Markets: oracle math was implemented exactly as specified. The specification was wrong.

Mature fields (aerospace, medical devices, nuclear) solved this decades ago with an explicit split:

| | Question | Failure mode if skipped |
|---|---|---|
| **Verification** | Did we build the thing right? (code matches spec) | Implementation bugs |
| **Validation** | Did we build the right thing? (spec has the properties we need) | Correct code, wrong behavior |

Current smart contract tools (Certora, Halmos, Slither) are **verification** tools — they prove the code matches a specification. But if the specification is flawed, correct code still gets exploited.

**symproof is primarily a validation tool.** It works at the formula level, not the bytecode level. The question "does this AMM formula actually preserve the invariant under adversarial fee configurations?" is a math question. The question "does rounding error net in the protocol's favor across a 5-step pipeline?" is a math question. These are questions about whether the *design* is correct — before any code is written.

### The three-layer architecture

| Layer | Role | Proves | Tool |
|---|---|---|---|
| **Validation** (symproof) | Is the math right? | Formulas have the behavioral properties designers expect | SymPy + deterministic hashing |
| **Simulation** | Does it work in practice? | Properties hold under finite precision, noise, adversarial inputs | numpy, MATLAB, Monte Carlo, fuzzing |
| **Verification** | Is the code right? | Production code matches the validated model | Certora, Halmos, Slither, manual audit |

symproof covers the first layer. It proves that the *formula* has the properties you think it has — not that the *code* implements the formula correctly, and not that the formula behaves well under conditions you haven't modeled.

The innovation is **binding all three layers via cryptographically traceable evidence trees.** Each sealed proof bundle gets a deterministic SHA-256 hash. That hash goes into a requirements traceability matrix alongside simulation results and code audit findings. Any reviewer can re-run the proof and get the identical hash — independent, reproducible evidence.

The [satellite ADCS example](symproof/library/examples/control/04_composition.py) demonstrates the full pattern: three independent proofs (controllability, observability, Lyapunov stability) composed into a single sealed bundle, with an explicit hash mapped to a stability requirement in the V&V matrix.

### What symproof proves (validation)

- The AMM formula actually preserves the product invariant with fees
- Rounding error is bounded AND nets in the protocol's favor
- Intermediate products don't overflow machine word sizes
- Cross-decimal token pricing requires normalization (and by how much)
- A Lyapunov function exists (the linear model is actually stable)
- Coordinate transforms are invertible and preserve properties

### What symproof does NOT prove

- Production code matches the proven formula → **verification** (Certora, Halmos)
- The formula behaves under adversarial parameter ranges → **simulation** (fuzzing, Monte Carlo)
- State transitions are atomic (no reentrancy) → **static analysis** (Slither)
- Token transfers deliver the expected amount → **integration tests**
- Governance parameters stay within proven bounds → **access control audit**

These gaps are covered by the other two layers. symproof's proofs are **necessary but not sufficient** — they establish that the design is mathematically sound before simulation tests it under stress and code analysis verifies the implementation.

### Why this matters

An economist designs an AMM. An auditor verifies the Solidity. But nobody proves the economist's math actually has the properties the protocol claims. This is the gap where the largest DeFi exploits live — and it's the gap symproof fills.

The goal: an open-source verification and validation stack where symbolic proofs, simulation results, and code audit findings are all independently reproducible, cryptographically linked, and traceable to requirements.

## Features

- **Reproducible proofs** — Every proof is hashed. Share the hash as a receipt; anyone can re-verify.
- **Composable** — Import sealed proofs as building blocks. Prove `A`, prove `B`, then import both into a proof of `C`.
- **Advisory system** — When verification passes through known SymPy limitations (domain-ignoring simplification, heuristic Q-system), the result flags it for human review.
- **False-positive protection** — `seal()` rejects proofs where lemma assumptions contradict axioms. Axioms are authoritative.

## Install

```bash
uv add symproof
```

## Quick Start

```python
import sympy
from symproof import Axiom, AxiomSet, ProofBuilder, LemmaKind, seal

x, y = sympy.symbols("x y")

axioms = AxiomSet(name="positive_reals", axioms=(
    Axiom(name="x_positive", expr=x > 0),
    Axiom(name="y_positive", expr=y > 0),
))

h = axioms.hypothesis("product_positive", expr=x * y > 0)

script = (
    ProofBuilder(axioms, h.name, name="pos_product", claim="x*y > 0")
    .lemma("xy_positive", LemmaKind.QUERY,
           expr=sympy.Q.positive(x * y),
           assumptions={"x": {"positive": True}, "y": {"positive": True}})
    .build()
)

bundle = seal(axioms, h, script)
print(bundle.bundle_hash)  # deterministic 64-char SHA-256
```

## Examples

Graduated examples in [`examples/`](examples/), each self-contained and runnable:

| File | What it teaches |
|------|----------------|
| [`01_first_proof.py`](examples/01_first_proof.py) | Simplest proof: Euler's identity e^(ipi)+1=0 |
| [`02_series_and_sums.py`](examples/02_series_and_sums.py) | Infinite series, closed-form sums, advisory system |
| [`03_multi_lemma_proof.py`](examples/03_multi_lemma_proof.py) | Decomposing AM-GM into a chain of lemmas |
| [`04_using_the_library.py`](examples/04_using_the_library.py) | Importing pre-built proofs, composition |
| [`05_control_engineering.py`](examples/05_control_engineering.py) | Plant + controller stability, Lyapunov construction |

Run any example:

```bash
uv run python examples/01_first_proof.py
```

## Proof Library

Pre-built, reusable proof bundles organized by domain. Import them into your proofs via `ProofBuilder.import_bundle()`.

### Core (`symproof.library.core`)

| Function | Proves |
|----------|--------|
| `max_ge_first(ax, a, b)` | Max(a, b) >= a |
| `piecewise_collapse(ax, expr, cond, fb)` | Piecewise collapses to active branch |

### Control Systems (`symproof.library.control`)

| Function | Proves |
|----------|--------|
| `hurwitz_second_order(ax, a2, a1, a0)` | 2nd-order polynomial is Hurwitz stable |
| `hurwitz_third_order(ax, a3, a2, a1, a0)` | 3rd-order Routh-Hurwitz |
| `closed_loop_stability(ax, G_n, G_d, C_n, C_d, s)` | Plant + controller closed-loop is stable |
| `lyapunov_stability(ax, A, P, Q)` | Verify Lyapunov equation A^T P + PA + Q = 0 |
| `lyapunov_from_system(ax, A)` | **Construct** P and prove stability |
| `gain_margin(ax, coeffs, K, s)` | Gain below critical => stable |
| `controllability_rank(ax, A, B)` | System (A,B) is controllable |
| `observability_rank(ax, A, C)` | System (A,C) is observable |
| `quadratic_invariant(ax, states, dots, V)` | dV/dt = 0 along trajectories |

### DeFi (`symproof.library.defi`)

Tools for the numerical analysis issues Solidity auditors actually face.

| Function / Class | Proves |
|------------------|--------|
| `mul_down`, `mul_up`, `div_down`, `div_up` | Floor/ceil fixed-point ops with explicit rounding direction |
| `rounding_bias_lemma(a, b, s)` | ceil >= floor (UP always >= DOWN) |
| `rounding_gap_lemma(a, b, s)` | ceil - floor <= 1 (max 1 unit extraction per op) |
| `directional_chain_error(steps)` | Per-step sign + net flow direction across a pipeline |
| `chain_error_bound(exact, trunc)` | Cumulative rounding error < N over N operations |
| `phantom_overflow_check(a, b, d)` | a*b overflows uint256 (mulDiv required) |
| `no_phantom_overflow_check(a, b)` | a*b fits in uint256 (naive path safe) |
| `DecimalAwarePool(rx, ry, dec_x, dec_y)` | Cross-decimal normalization (e.g., USDC/6 ↔ WETH/18) |
| `fee_complement_positive(ax, fee)` | 1 - fee > 0 from 0 < fee < 1 |
| `amm_output_positive(ax, Rx, Ry, fee, dx)` | AMM swap output > 0 |
| `amm_product_nondecreasing(ax, Rx, Ry, fee, dx)` | Product invariant grows with fees |

Run the walkthrough: `uv run python -m symproof.library.examples.amm_swap_audit`

## Verification Strategies

symproof dispatches on `LemmaKind`:

| Kind | How it verifies | Use when |
|------|----------------|----------|
| `EQUALITY` | `simplify(expr - expected) == 0` | Algebraic identities, series |
| `BOOLEAN` | `simplify(expr) is True`, with `refine()` and proof-by-contradiction fallbacks | Implications, inequalities |
| `QUERY` | `sympy.ask(expr, context)` | Positivity, type queries (irrational, integer, etc.) |
| `COORDINATE_TRANSFORM` | Round-trip + transform + simplify | Polar/hyperbolic coordinate proofs |

## Development

```bash
uv sync
uv run python -m pytest tests/ -v
uv run ruff check symproof/
```
