---
name: frame-proof
description: Frame a proof problem — define axioms and hypothesis without constructing the proof. Use when a domain expert wants to state what needs to be proven.
---

# Frame a proof problem

You are helping a **domain expert** who knows WHAT they want to prove but isn't going to construct the proof themselves. Your job is to translate their domain knowledge into a well-formed symproof problem: an `AxiomSet` and one or more `Hypothesis` objects.

**You produce the problem statement. You do NOT produce the proof.**

The output of this workflow is a Python file that a prover can pick up and work with.

## Argument

The user's proof request: $ARGUMENTS

## Workflow

### Step 1: Understand the domain

Ask the user:
- **What system or problem are you working with?** (e.g., control system, optimization problem, AMM mechanism, physical model)
- **What are the key parameters/symbols?** Get names, types (real, positive, integer), and physical meaning
- **What do you already know is true?** These become axioms — the accepted truths you don't need to prove

Don't assume. Domain experts know their systems; your job is to capture that knowledge precisely in SymPy.

### Step 2: Define symbols

Declare SymPy symbols with appropriate assumptions:
```python
import sympy
Kp = sympy.Symbol("Kp", positive=True)   # proportional gain [N·m/rad]
```

Always include:
- Physical units in comments
- SymPy assumptions that match the domain constraints (positive, real, integer, nonnegative)
- Descriptive names matching the domain convention

### Step 3: Build the axiom set

Axioms are what the user ACCEPTS AS TRUE. Ask explicitly:
- "Is this always positive?"
- "Is this bounded? By what?"
- "What constraints come from physics / the problem definition?"

```python
from symproof import Axiom, AxiomSet

axioms = AxiomSet(
    name="descriptive_name",
    axioms=(
        Axiom(name="gain_positive", expr=Kp > 0),
        # ...
    ),
)
```

Rules:
- Each axiom gets a descriptive name
- Don't include things that need to be PROVEN as axioms
- If a constraint requires two separate conditions (e.g., `0 < f < 1`), use two axioms (`f > 0` and `f < 1`), not `And(f > 0, f < 1)` — separate axioms compose better

### Step 4: State the hypothesis

Ask: "What exactly do you want to prove?" Then formalize it:

```python
hypothesis = axioms.hypothesis(
    "descriptive_claim_name",
    expr=<sympy boolean expression>,
    description="Plain English description of the claim",
)
```

The hypothesis MUST be bound to the axiom set via `axioms.hypothesis()`.

### Step 5: Document what's NOT covered

Every framed problem should include a comment block:

```python
# What this proof would establish:
#   <what the hypothesis means in domain terms>
#
# What this proof would NOT establish:
#   <what's out of scope — robustness, implementation correctness, etc.>
#
# Suggested proof strategy:
#   <if the framer has intuition about how to prove it, capture it>
```

### Step 6: Save the framed problem

Write a Python file (e.g., `framed_<name>.py`) containing:
1. Symbol declarations with comments
2. AxiomSet
3. Hypothesis (or hypotheses)
4. Scope documentation

The file should be importable — a prover will `from framed_<name> import axioms, hypothesis`.

## What you do NOT do

- Do NOT construct lemmas
- Do NOT call `ProofBuilder`
- Do NOT call `seal()`
- Do NOT import from `symproof.library`
- Do NOT try to prove anything

Your output is ONLY the problem statement. The prover picks it up from here.

## Example output structure

```python
"""Framed proof: AMM swap output is positive.

Framer: <domain expert>
Date: <date>
Domain: DeFi / AMM constant-product

What this proof would establish:
    The swap output dy is strictly positive for any positive input dx,
    given positive reserves and a fee rate in (0, 1).

What this proof would NOT establish:
    - Integer truncation effects (Solidity uses floor division)
    - That the output exceeds gas costs (economic viability)
    - Reentrancy safety of the swap function
"""

import sympy
from symproof import Axiom, AxiomSet

Rx = sympy.Symbol("R_x", positive=True)   # reserve of token X [wei]
Ry = sympy.Symbol("R_y", positive=True)   # reserve of token Y [wei]
fee = sympy.Symbol("f")                    # fee rate (dimensionless)
dx = sympy.Symbol("dx", positive=True)    # input amount [wei]

axioms = AxiomSet(
    name="amm_constant_product",
    axioms=(
        Axiom(name="reserve_x_positive", expr=Rx > 0),
        Axiom(name="reserve_y_positive", expr=Ry > 0),
        Axiom(name="fee_positive", expr=fee > 0),
        Axiom(name="fee_below_one", expr=fee < 1),
        Axiom(name="input_positive", expr=dx > 0),
    ),
)

dy = Ry * dx * (1 - fee) / (Rx + dx * (1 - fee))

hypothesis = axioms.hypothesis(
    "swap_output_positive",
    expr=dy > 0,
    description="AMM swap output is strictly positive",
)
```
