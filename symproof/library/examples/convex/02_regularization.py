#!/usr/bin/env python3
"""Regularization strengthens convexity: why lambda matters mathematically.

Scenario
--------
You're tuning regularization strength for ridge regression.  You know
larger lambda = "simpler model" intuitively, but what does it mean
mathematically?  This example proves that regularization increases the
strong convexity parameter, which:
  - Guarantees a unique minimum (no ambiguity in the solution)
  - Speeds up gradient descent convergence (linear rate improves)
  - Improves numerical conditioning (solver is more stable)

Then we show what happens when you use a bad loss — symproof catches it.

What this proves
----------------
- Ridge loss is strongly convex with parameter 2·(1 + λ)
- Removing the regularizer (λ=0) is still convex but weaker (param = 2)
- Adding regularization strictly increases the strong convexity parameter

What this does NOT prove
------------------------
- That the regularized solution generalizes better (statistical question)
- Optimal value of lambda (use cross-validation)
- Convergence of your specific optimizer implementation
- That L2 is the right regularizer vs L1, elastic net, etc.

What to do next
---------------
1. Cross-validate to choose lambda
2. Solve with sklearn or CVXPY
3. Check test set performance (convexity ≠ generalization)
4. If L1 needed, note that |x| is convex but not smooth — different analysis

Run: uv run python -m symproof.library.examples.convex.02_regularization
"""

import sympy

from symproof import Axiom, AxiomSet
from symproof.library.convex import convex_hessian, strongly_convex

x1 = sympy.Symbol("x1", real=True)
x2 = sympy.Symbol("x2", real=True)
lam = sympy.Symbol("lambda", positive=True)

# ─── Ridge regression: ||x - target||² + λ||x||² ───────────
#
# Simplified 2D: f(x) = (x1-1)² + (x2-2)² + λ(x1² + x2²)
# Hessian = diag(2+2λ, 2+2λ) → eigenvalues both = 2(1+λ)
# Strong convexity parameter = 2(1+λ)

ridge = (x1 - 1)**2 + (x2 - 2)**2 + lam * (x1**2 + x2**2)

axioms = AxiomSet(
    name="ridge_regression",
    axioms=(Axiom(name="lambda_positive", expr=lam > 0),),
)

m_ridge = 2 * (1 + lam)

print("Ridge regression: f(x) = ||x - target||² + λ·||x||²")
print("  Hessian eigenvalues: 2·(1 + λ)")
print("  Strong convexity parameter: m = 2·(1 + λ)")

bundle_ridge = strongly_convex(
    axioms, ridge, [x1, x2], m_ridge,
    assumptions={"lambda": {"positive": True}},
)
print(f"  {m_ridge}-strongly convex? {bundle_ridge.proof_result.status.value}")
print("  → Unique global minimizer guaranteed")
print("  → Gradient descent converges linearly with rate ~ 1 - 2m/L")


# ─── Without regularization: still convex, but weaker ───────
#
# Set λ = 0: f(x) = ||x - target||²
# Hessian = diag(2, 2) → eigenvalues = 2
# Strong convexity parameter drops from 2(1+λ) to 2

no_reg = (x1 - 1)**2 + (x2 - 2)**2

trivial = AxiomSet(
    name="no_reg", axioms=(Axiom(name="defined", expr=sympy.Eq(1, 1)),),
)

print("\nWithout regularization (λ = 0):")
bundle_no_reg = strongly_convex(
    trivial, no_reg, [x1, x2], sympy.Integer(2),
)
print(f"  2-strongly convex? {bundle_no_reg.proof_result.status.value}")
print("  Still has unique minimum, but:")
print("  - Weaker conditioning (solver may be slower)")
print("  - No regularization bias toward simpler solutions")


# ─── FAILURE CASE: bad loss is correctly rejected ───────────
#
# What if someone uses x1³ + x2² as a loss?
# Hessian = [[6·x1, 0], [0, 2]] — det = 12·x1
# Negative when x1 < 0 → NOT convex.

print("\nFailure case: f(x) = x1³ + x2² (not convex)")
bad_loss = x1**3 + x2**2
try:
    bad_bundle = convex_hessian(trivial, bad_loss, [x1, x2])
    print("  ERROR: should have been rejected!")
except ValueError:
    print("  Correctly REJECTED — Hessian minor can be negative")
    print("  → Gradient descent on this loss may find a saddle point,")
    print("    not a minimum. Use a convex surrogate or different approach.")

print()
print("Key insight: regularization doesn't just 'prevent overfitting' —")
print("it mathematically strengthens the optimization landscape.")
print("The strong convexity parameter 2·(1+λ) directly controls")
print("convergence speed and solution uniqueness.")
