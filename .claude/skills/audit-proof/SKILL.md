---
name: audit-proof
description: Audit a sealed proof bundle — verify correctness, interrogate assumptions, find gaps. Use when reviewing proofs for soundness and completeness.
---

# Audit a proof

You are helping a **reviewer** who has received a sealed `ProofBundle` and needs to determine whether the proof is sound, complete, and honest about its limitations.

**Your job is adversarial. You are looking for reasons the proof might be wrong, incomplete, or misleading — not confirming it's correct.**

## Argument

The proof bundle, file, or claim to audit: $ARGUMENTS

## Workflow

### Step 1: Load and re-verify

```python
from symproof import verify_proof

# Re-verify from scratch (never trust the stored status)
result = verify_proof(bundle.proof, trust_imports=False)
assert result.status.value == "VERIFIED", f"FAILED: {result.failure_summary}"
```

If re-verification fails, stop. The proof is broken.

### Step 2: Examine the axiom set

For EACH axiom, ask:
- **Is this actually true in the system being modeled?** An axiom that says `fee > 0` is only valid if the protocol enforces positive fees.
- **Is this too strong?** Does the axiom assume more than necessary? (e.g., `x > 0` when `x >= 0` would suffice)
- **Is this too weak?** Are there constraints the framer forgot? Missing axioms mean the proof holds in a larger space than intended.
- **Are there implicit assumptions?** SymPy symbol assumptions (positive=True on the Symbol itself) are separate from Axiom expressions. Check both.

```python
print("Axiom set:", bundle.axiom_set.name)
for ax in bundle.axiom_set.axioms:
    print(f"  {ax.name}: {ax.expr}")
```

### Step 3: Examine the hypothesis

- **Does the hypothesis match what the stakeholder actually needs?** A proof that "dy > 0" doesn't prove "dy >= minimum_viable_output".
- **Is the hypothesis expression correct?** Verify the SymPy expression matches the mathematical statement in the documentation.
- **What does this hypothesis NOT cover?** List explicitly.

### Step 4: Walk through each lemma

For EACH lemma in the proof chain:

```python
for lr in bundle.proof_result.lemma_results:
    print(f"[{lr.lemma_name}] passed={lr.passed}")
    if lr.advisories:
        for adv in lr.advisories:
            print(f"  ADVISORY: {adv}")
```

For each lemma, ask:
- **What verification strategy was used?** (EQUALITY / BOOLEAN / QUERY / COORDINATE_TRANSFORM)
- **Are there advisories?** Each advisory is a flag for human review:
  - "domain" advisory → check for division-by-zero or branch-cut issues
  - "doit" advisory → verify the closed-form evaluation is correct
  - "refine" / "negation" advisory → the proof used a fallback; is the fallback sound here?
  - "QUERY" advisory → Q-system uses heuristics; is the assumption context complete?
  - "INDETERMINATE" → SymPy couldn't determine truth; the proof has a gap
- **Does `depends_on` reflect actual logical dependence?** Or are dependencies missing?

### Step 5: Check for false-positive vectors

Known ways a proof can be technically VERIFIED but misleading:

1. **Domain-ignoring simplification**: `simplify((x²-1)/(x-1))` returns `x+1`, ignoring the singularity at `x=1`. If the proof involves rational expressions, check whether domain restrictions matter.

2. **Assumption override**: Lemma assumptions like `{"x": {"positive": True}}` are checked against axioms by `seal()`, but verify the CHECK actually caught all conflicts.

3. **Vacuous truth**: An `Implies(P, Q)` is true when P is false. If the antecedent is never satisfied in the real system, the implication proves nothing useful.

4. **Symbolic vs numeric**: A proof that `f(x) >= 0` symbolically doesn't mean `f(0.0001)` won't underflow to negative in floating-point.

### Step 6: Attempt the negation

Try to prove the OPPOSITE of the hypothesis. If you can, the axiom set is inconsistent.

```python
from symproof import disprove

h_neg = bundle.hypothesis.negate()
# Try to construct a proof of ~H under the same axioms
# If seal() succeeds for both H and ~H → contradiction in axioms
```

Even if you can't prove ~H, check whether specific parameter values violate the hypothesis:
```python
# Numeric spot-check: does the claim hold for edge cases?
expr = bundle.hypothesis.expr
print(expr.subs({x: 0}))       # boundary
print(expr.subs({x: 1e-10}))   # near-zero
print(expr.subs({x: 1e10}))    # large
```

### Step 7: Assess completeness

Ask:
- **What properties does the system need that this proof DOESN'T cover?**
- **What parameters are symbolic that should have concrete bounds?**
- **What failure modes of the real system are not modeled?**
  - For control: actuator saturation, sensor noise, discretization
  - For DeFi: reentrancy, oracle manipulation, gas limits
  - For optimization: numerical conditioning, constraint feasibility

### Step 8: Write the audit report

Structure:

```
## Audit report: <proof name>

### Verdict: PASS / PASS WITH ADVISORIES / FAIL

### Axiom review
- <axiom>: <assessment>

### Hypothesis review
- Matches stated claim: yes/no
- Scope gaps: <what's not covered>

### Lemma chain review
- <lemma>: <strategy> — <assessment>
- Advisories requiring attention: <list>

### False-positive risk: LOW / MEDIUM / HIGH
- <specific risks identified>

### Completeness gaps
- <what should be proven next>

### Recommendation
- <accept / accept with caveats / reject with reason>
```

## What you do NOT do

- Do NOT assume the proof is correct because it says VERIFIED
- Do NOT skip advisories — every advisory is a review item
- Do NOT accept vacuous truths without flagging them
- Do NOT confuse "SymPy can prove it" with "it's true" — SymPy has known false-positive vectors
- Do NOT sign off without checking the axiom set against the real system
