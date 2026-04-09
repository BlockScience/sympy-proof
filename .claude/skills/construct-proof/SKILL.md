---
name: construct-proof
description: Construct a proof for a framed hypothesis — build lemma chains, import library proofs, seal into a bundle. Use when an engineer needs to prove a specific property.
---

# Construct a proof

You are helping an **engineer** who has a framed problem (axioms + hypotheses) and needs to construct a proof for ONE specific hypothesis. The framing was done by a domain expert; the proof is your job.

**Prove one property at a time. Each sealed bundle maps to one requirement.**

## Key principle: focused proofs, not monoliths

The framer gave you multiple independent hypotheses. Prove each one separately:

```
framed_satellite.py:
├── h_stability    → prove → seal → hash_stability    → REQ-STAB
├── h_controllable → prove → seal → hash_controllable → REQ-CTRL
├── h_observable   → prove → seal → hash_observable   → REQ-OBS
└── h_lyapunov     → prove → seal → hash_lyapunov     → REQ-LYAP
```

Each proof is simple, reviewable, and independently re-verifiable. If the system changes and stability breaks, you re-prove h_stability without touching the controllability proof.

Use `import_bundle` ONLY when one hypothesis genuinely depends on another (e.g., the proof of uniqueness imports the proof of strict convexity). If they're independent, prove them independently.

## Argument

The specific hypothesis to prove: $ARGUMENTS

## Workflow

### Step 1: Load the framed problem

Read the axioms and the specific hypothesis you're proving. Understand:
- What symbols are involved and their assumptions
- What the hypothesis claims
- Whether it depends on another hypothesis (check the traceability map)

### Step 2: Check the library

Before writing custom lemmas, check if a library function covers this directly:

**Control:** `closed_loop_stability`, `lyapunov_from_system`, `controllability_rank`, `observability_rank`, `hurwitz_second_order`, `gain_margin`, `quadratic_invariant`

**Convex:** `convex_scalar`, `convex_hessian`, `strongly_convex`, `unique_minimizer`, `convex_composition`, `conjugate_function`

**DeFi:** `fee_complement_positive`, `amm_output_positive`, `amm_product_nondecreasing`

**Core:** `max_ge_first`, `piecewise_collapse`

If a library function covers the claim, use it. That's what the library is for.

### Step 3: Try direct proof

```python
from symproof.tactics import auto_lemma
lemma = auto_lemma("claim", hypothesis.expr, assumptions={...})
```

If `auto_lemma` returns a Lemma, you have a one-step proof. Build and seal.

### Step 4: Decompose if needed

Common decomposition patterns:

**Helper symbol** (for bounded intervals):
```python
g = sympy.Symbol("g", positive=True)  # g = 1-f
# Rewrite in terms of g, Q.positive works on all-positive symbols
```

**Algebraic chain** (for inequalities):
```python
# Prove (a-b)^2 >= 0, then expand, then rearrange to target form
```

**Import and extend** (when this property depends on another):
```python
# Only if the framer flagged a dependency
builder.import_bundle(prior_bundle).lemma("new_step", ...)
```

### Step 5: Build and seal

```python
script = (
    ProofBuilder(axioms, hypothesis.name, name="...", claim="...")
    .lemma("step_1", LemmaKind.EQUALITY, expr=..., expected=...,
           description="why this step matters")
    .build()
)
bundle = seal(axioms, hypothesis, script)
```

### Step 6: Report

Print:
- The hypothesis name and description
- Status (VERIFIED)
- Bundle hash (this goes into the traceability matrix)
- Advisories (each is a review item for the auditor)
- Which requirement this maps to

```python
print(f"Requirement: REQ-STAB")
print(f"Hypothesis:  {hypothesis.name}")
print(f"Status:      {bundle.proof_result.status.value}")
print(f"Hash:        {bundle.bundle_hash}")
print(f"Advisories:  {len(bundle.proof_result.advisories)}")
```

### Step 7: Save

Write a Python file that imports the framed problem, constructs and seals the proof, and prints the traceability record. Runnable standalone.

## When you get stuck

1. Check if the axioms are strong enough — the framer may need to add a constraint
2. Try numeric spot-checks — if the hypothesis is false for some parameter values, no proof exists
3. Check known SymPy limitations — the library may have a workaround
4. Report back: "This hypothesis requires axiom X to be provable" or "SymPy cannot verify this expression class"

## What you do NOT do

- Do NOT combine multiple independent hypotheses into one proof
- Do NOT weaken the hypothesis to make it easier
- Do NOT change the axioms without consulting the framer
- Do NOT skip advisories — they're part of the deliverable for the auditor
