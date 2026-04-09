---
name: construct-proof
description: Construct a proof for a framed problem — build lemma chains, import library proofs, seal into a bundle. Use when an engineer needs to prove a stated hypothesis.
---

# Construct a proof

You are helping an **engineer** who has a framed problem (axioms + hypothesis) and needs to construct a proof that seals into a deterministic `ProofBundle`.

**You build the proof. You do NOT question whether the hypothesis is worth proving — that was the framer's job.**

## Argument

The framed problem or file to prove: $ARGUMENTS

## Workflow

### Step 1: Load the framed problem

Read the axioms and hypothesis. Understand:
- What symbols are involved and their assumptions
- What the hypothesis claims
- Any suggested proof strategy from the framer

### Step 2: Check the library first

Before writing custom lemmas, check if the library already has what you need:

```python
# Control systems
from symproof.library.control import (
    closed_loop_stability,    # plant + controller → Hurwitz
    lyapunov_from_system,     # construct Lyapunov function
    controllability_rank,     # Gramian nonsingularity
    observability_rank,       # Gramian nonsingularity
    hurwitz_second_order,     # Routh-Hurwitz (2nd order)
    gain_margin,              # critical gain analysis
    quadratic_invariant,      # dV/dt = 0
)

# Convex optimization
from symproof.library.convex import (
    convex_scalar,            # f'' >= 0
    convex_hessian,           # Hessian PSD
    strongly_convex,          # H - mI PSD
    unique_minimizer,         # strictly convex → unique
    convex_composition,       # DCP rule
)

# DeFi
from symproof.library.defi import (
    fee_complement_positive,  # 1-f > 0 from bounded interval
    amm_output_positive,      # swap output > 0
    amm_product_nondecreasing, # product invariant grows
)

# Core (SymPy gap workarounds)
from symproof.library.core import (
    max_ge_first,             # Max(a,b) >= a
    piecewise_collapse,       # Piecewise branch under assumptions
)
```

If a library function covers the claim, use it directly and compose.

### Step 3: Try the direct approach

Use tactics to check if the claim is directly provable:

```python
from symproof.tactics import auto_lemma, try_simplify, try_query

# Does auto_lemma find a strategy?
lemma = auto_lemma("claim", hypothesis.expr, assumptions={...})
if lemma:
    print(f"Direct proof found via {lemma.kind}")
```

### Step 4: Decompose if direct proof fails

When SymPy can't verify the claim in one step, decompose. Common patterns:

**Pattern A: Helper symbol substitution** (for bounded intervals)
```python
# SymPy can't reason about 0 < f < 1 → 1-f > 0
# Introduce g = 1-f declared positive, prove separately
g = sympy.Symbol("g", positive=True)
# Rewrite expression in terms of g, then Q.positive works
```

**Pattern B: Algebraic decomposition** (for inequalities)
```python
# Prove (a+b)/2 >= sqrt(ab) via:
# 1. (a-b)^2 >= 0  (Q.nonnegative — squares are nonneg)
# 2. Expand and rearrange to AM-GM form
```

**Pattern C: Import and extend** (compose library proofs)
```python
# Import an existing result, then add new lemmas on top
builder = ProofBuilder(axioms, hyp.name, name="...", claim="...")
    .import_bundle(library_bundle)
    .lemma("new_step", LemmaKind.QUERY, ...)
```

**Pattern D: Case analysis** (for Piecewise/Max/Min)
```python
# Use library workarounds for Max/Min/Piecewise
from symproof.library.core import max_ge_first
```

### Step 5: Build and seal

```python
from symproof import ProofBuilder, LemmaKind, seal

script = (
    ProofBuilder(axioms, hypothesis.name, name="...", claim="...")
    .import_bundle(...)  # if using library
    .lemma("step_1", LemmaKind.EQUALITY, expr=..., expected=...,
           assumptions={...}, description="...")
    .lemma("step_2", LemmaKind.QUERY, expr=...,
           assumptions={...}, depends_on=["step_1"], description="...")
    .build()
)

bundle = seal(axioms, hypothesis, script)
```

### Step 6: Check advisories

After sealing, inspect the result:

```python
print(f"Status: {bundle.proof_result.status.value}")
print(f"Hash: {bundle.bundle_hash}")

for adv in bundle.proof_result.advisories:
    print(f"  Advisory: {adv}")
```

Advisories flag steps where SymPy's guarantees are weaker. Report them to the reviewer.

### Step 7: Save the proof

Write a Python file that:
1. Imports the framed problem
2. Constructs and seals the proof
3. Prints the hash and advisories
4. Is runnable standalone: `uv run python proof_<name>.py`

## When you get stuck

If no decomposition works:
1. Check if the axioms are strong enough — maybe the framer missed a constraint
2. Check if the hypothesis is actually true — try numeric counterexamples
3. Check if this is a known SymPy limitation — consult the library for workarounds
4. If truly stuck, report back to the framer: "This claim may need a stronger axiom set" or "SymPy cannot verify this class of expression"

## What you do NOT do

- Do NOT question whether the hypothesis is worth proving
- Do NOT weaken the hypothesis to make it easier to prove
- Do NOT change the axioms without consulting the framer
- Do NOT skip advisories — they are part of the deliverable
