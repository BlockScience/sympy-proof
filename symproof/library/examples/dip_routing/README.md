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
| `01_heavy_ball_equivalence.py` | Lemma 1 (eq. 25 - 26) | DIP update = heavy ball on soft backpressure (algebra) |
| `02_bounded_gradient.py` | Lemma 2 (eq. 30) | Stochastic gradient uniformly bounded by capacity + arrival rates |
| `03_lagrangian_structure.py` | Proposition 1 (eq. 18-19) | Dual differentiability, reverse waterfilling maximisers |
| `04_dual_convergence.py` | Proposition 2 (eq. 31) | Dual variables converge to optimum a.s. under decaying step |
| `05_queue_stability.py` | Proposition 3 + Corollary 1 (eq. 33-34) | All queues empty infinitely often with probability one |
| `06_danskin_concrete.py` | Danskin's theorem (concrete) | Envelope identity for the DIP quadratic objective |
| `07_flam_convergence.py` | Flam [19], Theorem 2 (algebraic kernel) | Lyapunov descent structure + telescoping for stochastic heavy ball |
| `08_supermartingale_finite.py` | Solo-Kong [21], Thm E.7.4 (finite-time core) | Arithmetic contradiction: bounded descent forces return to zero |

## Run

```bash
# Individual examples
uv run python -m symproof.library.examples.dip_routing.01_heavy_ball_equivalence
uv run python -m symproof.library.examples.dip_routing.06_danskin_concrete

# All examples
for f in 01 02 03 04 05 06 07 08; do
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

## Foundation proofs and hidden axiom discovery (06-08)

Files 01-05 initially axiomatised three external theorems as
`expr=True` — placeholder axioms with no mathematical content.
Files 06-08 then proved the computational cores of those theorems.

**This process revealed hidden axioms.** When we built the foundation
proofs (06-08), each required its own axiom set with conditions that
the downstream proofs (03-05) had not declared.  For example:

- **File 04** axiomatised `flam_theorem = True` (3 axioms total)
- **File 07** proved the algebraic content of Flam's theorem — but
  under an axiom set with 4 axioms, including `lyapunov_nonneg`,
  `concavity_descent`, and `robbins_siegmund`, none of which appeared
  in file 04

This meant file 04's proof was implicitly relying on conditions it
never declared.  The proof was *correct* but *incomplete* in its
accounting of what it assumed.

### How symproof caught this

symproof's `seal()` function now accepts a `foundations` parameter.
When provided, it checks that **every axiom in the foundation's axiom
set exists in the downstream proof's axiom set**.  Missing axioms are
flagged as hidden dependencies and `seal()` refuses to proceed:

```python
# This raises ValueError — lyapunov_nonneg, concavity_descent,
# robbins_siegmund are not in file 04's axiom set
bundle = seal(axioms, hypothesis, script,
              foundations=[(flam_bundle, "flam_theorem")])
```

### How we fixed it

Each downstream proof (03, 04, 05) now explicitly declares all axioms
its foundation requires, marked with `inherited=True`:

```python
Axiom(
    name="lyapunov_nonneg",
    expr=V_t >= 0,
    inherited=True,  # came from Flam foundation, not posited by us
    description="Required by Flam's convergence argument.",
)
```

The `inherited` flag is semantically meaningful: it distinguishes
axioms the proof author chose (posited) from axioms the proof chain
forced (inherited).  Both are required for the proof to be sound, but
inherited axioms trace back to a specific foundation rather than a
modelling decision.

### Complete axiom accounting

After the fix, the full axiom picture for each downstream proof:

| Downstream | Posited axioms | Inherited axioms (from foundation) |
|------------|---------------|-------------------------------------|
| 03 (Prop 1) | `beta_positive`, `rate_nonneg`, `danskin_theorem` | `maximiser_exists` (from 06) |
| 04 (Prop 2) | `gradient_bounded`, `alpha_in_unit`, `flam_theorem` | `lyapunov_nonneg`, `concavity_descent`, `robbins_siegmund` (from 07) |
| 05 (Prop 3) | `queue_nonneg`, `xi_positive`, `gradient_vanishes`, `supermartingale_convergence` | `process_nonneg`, `descent_per_step`, `borel_cantelli_extension` (from 08) |

### What remains irreducibly axiomatic

After exhausting what SymPy can verify, these conditions cannot be
proved computationally and remain as honest axioms:

- **Maximiser existence** on compact domain (topology)
- **Concavity descent inequality** for abstract h (integral calculus)
- **Robbins-Siegmund lemma** (stochastic supermartingale convergence)
- **Borel-Cantelli extension** from finite-time return to "infinitely often"

These are well-established results from convex analysis and probability
theory.  The contribution of the computational proof chain is to make
explicit exactly where these results enter and what they require.

Further development of symproof domain libraries for topology, convex analysis and probability theory could allow us to extend these proof chains deeper to reveal a more fundamental axiom base.

## Approach

Where a result is purely algebraic (Lemma 1), the proof is
self-contained.  Where a result depends on external theorems — e.g.,
Flam's stochastic heavy ball convergence (Proposition 2) or Solo &
Kong's supermartingale theorem (Proposition 3) — the external result is
stated as an axiom, the foundation proof (06-08) verifies as much as
possible computationally, and the downstream proof (03-05) inherits the
foundation's remaining assumptions via `seal(foundations=...)`.

This creates a transparent chain of evidence: every assumption is either
proved, inherited from a named foundation, or declared as an irreducible
axiom.  Nothing is hidden.
