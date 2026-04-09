# symproof

Deterministic proof writing with SymPy.

Declare axioms, bind hypotheses, build lemma chains, and seal reproducible hashed proof bundles. Every sealed proof gets a deterministic SHA-256 hash — same inputs, same hash, every time.

## Why symproof?

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

| Function | Proves |
|----------|--------|
| `fee_complement_positive(ax, fee)` | 1 - fee > 0 from 0 < fee < 1 |
| `amm_output_positive(ax, Rx, Ry, fee, dx)` | AMM swap output > 0 |
| `amm_product_nondecreasing(ax, Rx, Ry, fee, dx)` | Product invariant grows with fees |

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
