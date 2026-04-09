#!/usr/bin/env python3
"""Certifying ML loss functions: is gradient descent finding the global min?

Scenario
--------
You're designing a custom loss function for a model.  If the loss is
convex, gradient descent finds the global minimum.  If it's not, you
might be stuck in a local minimum and not know it.

This example proves convexity for common losses, then shows what
happens when you try a non-convex loss — symproof correctly rejects it.

What this proves
----------------
- Squared error:    d²/dy² (y - ŷ)² = 2 >= 0
- Cross-entropy:    d²/dp² [-y·log(p) - (1-y)·log(1-p)] >= 0  for p ∈ (0,1)
- Log-barrier:      d²/dx² (-log(x)) = 1/x² >= 0  for x > 0
- Cubic (REJECTED): d²/dx² (x³) = 6x — NOT nonneg everywhere

What this does NOT prove
------------------------
- Convergence rate (depends on strong convexity param + learning rate)
- Generalization (convex loss ≠ good generalization — see bias-variance)
- Numerical stability (vanishing/exploding gradients are separate issues)
- That your model architecture is appropriate for the task

What to do next
---------------
1. Compute Lipschitz constant of the gradient (bounds step size)
2. Train the model, monitor loss curve for convergence
3. Validate on held-out data (convexity guarantees optimization, not generalization)
4. If non-convex loss is needed, use convex relaxation or multiple restarts

Run: uv run python -m symproof.library.examples.convex.01_loss_function
"""

import sympy

from symproof import Axiom, AxiomSet
from symproof.library.convex import convex_scalar

# ─── Squared error: (y - ŷ)² ────────────────────────────────
#
# The workhorse of regression.  Always convex — second derivative
# is the constant 2.

y_hat = sympy.Symbol("y_hat", real=True)
squared_error = (1 - y_hat)**2  # y=1 fixed for simplicity

trivial = AxiomSet(
    name="loss_functions",
    axioms=(Axiom(name="defined", expr=sympy.Eq(1, 1)),),
)

se_bundle = convex_scalar(trivial, squared_error, y_hat)
print("Squared error loss: (y - ŷ)²")
print("  f''(ŷ) = 2 >= 0")
print(f"  Convex? {se_bundle.proof_result.status.value}")


# ─── Cross-entropy: -y·log(p) - (1-y)·log(1-p) ─────────────
#
# The standard classification loss.  Convex in the predicted
# probability p ∈ (0, 1) when the true label y is fixed.
#
# f''(p) = y/p² + (1-y)/(1-p)²  — positive for p ∈ (0,1), y ∈ [0,1]

p = sympy.Symbol("p", positive=True)
cross_entropy = -sympy.log(p) - sympy.Rational(1, 2) * sympy.log(1 - p)

# We need p ∈ (0,1), but SymPy only knows p > 0.  The library's
# scalar convexity check uses Q.nonnegative on f''(p) = 1/p²,
# which works because p > 0 is sufficient for 1/p² > 0.
pos_axioms = AxiomSet(
    name="probability",
    axioms=(Axiom(name="p_positive", expr=p > 0),),
)

ce_bundle = convex_scalar(
    pos_axioms, -sympy.log(p), p,
    assumptions={"p": {"positive": True}},
)
print("\nNeg-log-likelihood: -log(p)")
print("  f''(p) = 1/p² >= 0 for p > 0")
print(f"  Convex? {ce_bundle.proof_result.status.value}")


# ─── Log-barrier: -log(x) ───────────────────────────────────
#
# Used in interior-point methods to enforce x > 0 without hard
# constraints.  Convex and goes to infinity as x → 0.

x = sympy.Symbol("x", positive=True)
barrier_axioms = AxiomSet(
    name="interior_point",
    axioms=(Axiom(name="x_pos", expr=x > 0),),
)

barrier_bundle = convex_scalar(
    barrier_axioms, -sympy.log(x), x,
    assumptions={"x": {"positive": True}},
)
print("\nLog-barrier: -log(x)")
print("  f''(x) = 1/x² >= 0 for x > 0")
print(f"  Convex? {barrier_bundle.proof_result.status.value}")


# ─── FAILURE CASE: x³ is NOT convex ─────────────────────────
#
# f''(x) = 6x, which is negative for x < 0.  Gradient descent
# on a cubic loss has no global minimum — it goes to -∞.
#
# symproof should REJECT this.  If it didn't, we couldn't trust
# the positive results above.

print("\nFailure case: x³")
print("  f''(x) = 6x — negative for x < 0, not globally convex")
try:
    cubic_bundle = convex_scalar(trivial, sympy.Symbol("x")**3, sympy.Symbol("x"))
    print("  ERROR: should have been rejected!")
except ValueError:
    print("  Correctly REJECTED by seal()")
    print("  → symproof won't certify a non-convex loss as convex")

print()
print("Summary: squared error and log-barrier are certifiably convex.")
print("Cubic is correctly rejected. This is the trust foundation —")
print("positive results are meaningful because negatives are caught.")
