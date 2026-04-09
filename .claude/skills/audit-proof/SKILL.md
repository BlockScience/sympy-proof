---
name: audit-proof
description: Audit proof bundles — verify correctness, assess coverage, find gaps. Use when reviewing proofs for a system's requirements.
---

# Audit proofs

You are helping a **reviewer** who has received one or more sealed `ProofBundle` objects and needs to determine whether the proof collection is sound, complete, and honest about its limitations.

**Your job is adversarial. You look for gaps — both within individual proofs and across the collection.**

## Two levels of audit

### Level 1: Individual proof soundness
Is THIS proof correct? Are the axioms true? Are the advisories addressed?

### Level 2: Collective coverage
Does the COLLECTION of proofs cover all the properties the system needs? What's missing? What falls in the gap between symbolic proof and real-world behavior?

Level 2 is where most failures live. A system can have five individually correct proofs and still fail because the sixth property — the one nobody proved — is the one that breaks.

## Argument

The proof bundle(s), file(s), or system to audit: $ARGUMENTS

## Workflow

### Step 1: Enumerate what needs to be true

Before looking at any proofs, ask: **what properties does this system need?**

For a control system:
- Stability (open-loop and closed-loop)
- Controllability
- Observability
- Performance bounds (settling time, overshoot, steady-state error)
- Robustness (gain margin, phase margin, parameter sensitivity)

For a DeFi mechanism:
- Invariant preservation
- Rounding direction correctness
- Overflow safety
- Fee/slippage bounds
- Economic incentive alignment

For an optimization problem:
- Convexity of objective
- Feasibility of constraints
- Uniqueness of solution
- Conditioning (strong convexity parameter)

**Write this list first.** Then check which items have proofs and which don't.

### Step 2: Map proofs to requirements

```
Property needed        | Proof exists? | Bundle hash         | Status
-----------------------|---------------|---------------------|--------
Closed-loop stability  | YES           | a3f2...             | VERIFIED
Controllability        | YES           | 7c51...             | VERIFIED
Observability          | YES           | fc89...             | VERIFIED
Pointing error < 0.1°  | NO            | —                   | UNPROVEN
Robustness to ±20% J   | NO (not symbolic) | —              | NEEDS SIM
Discrete-time stability | NO            | —                   | UNPROVEN
```

Gaps are more important than passes. Flag every UNPROVEN and NEEDS SIM entry.

### Step 3: Audit each proof individually

For each sealed bundle, re-verify and interrogate:

```python
result = verify_proof(bundle.proof, trust_imports=False)
assert result.status.value == "VERIFIED"
```

For each proof, check:

**Axioms:**
- Is each axiom actually true in the real system?
- Are any axioms too strong (assuming more than needed)?
- Are any missing (the proof holds in a wider space than intended)?

**Hypothesis:**
- Does it match the stated requirement?
- Is the SymPy expression correct?

**Lemma chain:**
- What advisories exist? Each is a review item.
- Are dependencies (`depends_on`, `import_bundle`) logically correct?

**False-positive vectors:**
- Domain-ignoring simplification (rational expressions with singularities)
- Vacuous truth (implication P→Q is trivially true when P is never satisfied)
- Symbolic vs numeric divergence (symbolic proof doesn't guarantee floating-point behavior)

### Step 4: Assess the collection

This is the critical step. Ask:

**Coverage:** What percentage of needed properties have proofs? 100% symbolic coverage is rare — flag what needs simulation or testing instead.

**Independence:** Are the proofs genuinely independent, or do they share hidden assumptions? Two proofs using the same axiom set are coupled through those axioms — if an axiom is wrong, both proofs fall.

**Layer completeness:** symproof covers symbolic analysis only. For each proven property, ask:
- Does it also need simulation? (robustness, noise, finite precision)
- Does it also need code verification? (implementation correctness)
- Does it also need testing? (integration, hardware-in-the-loop)

**Consistency:** If multiple proofs share an axiom set, run `check_consistency` on any bundles that prove related claims. Contradictory proofs under the same axioms indicate a bug in the axiom set.

### Step 5: Write the audit report

```
## Audit report: <system name>

### Coverage summary
- Properties needed: N
- Symbolically proven: M (with hashes)
- Unproven (needs proof): K
- Out of scope for symbolic analysis (needs sim/test): L

### Requirements traceability
| Requirement | Property | Proof hash | Verdict |
|---|---|---|---|
| REQ-STAB | Closed-loop stability | a3f2... | PASS |
| REQ-CTRL | Controllability | 7c51... | PASS WITH 1 ADVISORY |
| REQ-PERF | Pointing error | — | NO PROOF |
| REQ-ROB  | Robustness | — | NEEDS SIMULATION |

### Individual proof assessments
(per-proof details: axiom review, advisory review, false-positive risk)

### Collection-level findings
- Axiom consistency: <assessment>
- Coverage gaps: <what's missing>
- Layer gaps: <what needs sim/test beyond symbolic proof>

### Recommendation
- Accept / Accept with caveats / Reject
- Specific actions needed before sign-off
```

## What you do NOT do

- Do NOT assume a proof is correct because it says VERIFIED
- Do NOT assess individual proofs without first enumerating what needs to be true
- Do NOT sign off on "symbolic coverage" as "full coverage" — flag what needs simulation and testing
- Do NOT skip the collection-level assessment — individual correctness doesn't imply system correctness
