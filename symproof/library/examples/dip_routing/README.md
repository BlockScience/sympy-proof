# Discounted Integral Priority Routing — Computational Proofs

Reproves the main results of:

> **M. Zargham, A. Ribeiro, A. Jadbabaie.**
> "Discounted Integral Priority Routing For Data Networks."
> *IEEE Globecom 2014 — Next Generation Networking Symposium.*
> <https://www.seas.upenn.edu/~aribeiro/preprints/c_2014_zargham_etal.pdf>

The paper introduces Discounted Integral Priority (DIP) routing, which
sets routing priorities to a time-discounted integral of observed queue
lengths.  Unlike standard backpressure methods that merely stabilise
queues, DIP actively drives them toward zero.  The key insight is that
DIP is equivalent to the heavy ball method applied to the dual of a soft
backpressure optimisation, which unlocks convergence guarantees from
stochastic optimisation theory.

## Results reproved

| File | Paper result | What is proved |
|------|-------------|----------------|
| `01_heavy_ball_equivalence.py` | Lemma 1 (eq. 25 ↔ 26) | DIP update = heavy ball on soft backpressure (algebra) |
| `02_bounded_gradient.py` | Lemma 2 (eq. 30) | Stochastic gradient uniformly bounded by capacity + arrival rates |
| `03_lagrangian_structure.py` | Proposition 1 (eq. 18–19) | Dual differentiability, reverse waterfilling maximisers |
| `04_dual_convergence.py` | Proposition 2 (eq. 31) | Dual variables converge to optimum a.s. under decaying step |
| `05_queue_stability.py` | Proposition 3 + Corollary 1 (eq. 33–34) | All queues empty infinitely often with probability one |

## Run

```bash
# Individual examples
uv run python -m symproof.library.examples.dip_routing.01_heavy_ball_equivalence
uv run python -m symproof.library.examples.dip_routing.02_bounded_gradient

# All examples
for f in 01 02 03 04 05; do
  uv run python -m symproof.library.examples.dip_routing.${f}_*
done
```

## Scope

These proofs cover the **algebraic and structural** content of the
paper's results.  Specifically:

- Algebraic equivalence between update rules (Lemma 1)
- Structural boundedness from finite network parameters (Lemma 2)
- Convex-analytic Lagrangian properties (Proposition 1)
- Convergence framing via decaying step size conditions (Proposition 2)
- Queue stability as a consequence of dual convergence (Proposition 3)

They do **not** cover:

- Full measure-theoretic supermartingale arguments (require probability
  theory beyond SymPy's scope — these are framed as axioms)
- Numerical simulation of the algorithm on specific network topologies
- Comparison with BP, SBP, or ABP (empirical, not analytical)
- Extension to time-varying arrival rates or non-stationary networks

## Approach

Where a result is purely algebraic (Lemma 1), the proof is
self-contained.  Where a result depends on external theorems — e.g.,
Flam's stochastic heavy ball convergence (Proposition 2) or Solo &
Kong's supermartingale theorem (Proposition 3) — the external result is
stated as an axiom, and the paper's contribution (applying it to the DIP
setting) is proved computationally.
