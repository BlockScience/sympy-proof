#!/usr/bin/env python3
"""End-to-end: certify a Markowitz portfolio optimization formulation.

Scenario
--------
You're building a portfolio optimizer that minimizes variance subject
to a target return.  Before deploying it (or submitting it for risk
review), you need to certify that the problem formulation is convex —
otherwise the solver may find a local minimum that isn't globally
optimal, leading to suboptimal portfolio allocation.

The Markowitz problem (2-asset simplified):
    minimize    w1^2 * sigma1^2 + 2*rho*sigma1*sigma2*w1*w2 + w2^2 * sigma2^2
    subject to  w1 + w2 = 1  (budget)
                w1*mu1 + w2*mu2 >= r_target  (return)
                w1, w2 >= 0  (no shorting)

The objective is a quadratic form w^T Sigma w where Sigma is the
covariance matrix.  This is convex iff Sigma is PSD.

What this proves
----------------
1. The covariance matrix is PSD (Hessian of objective)
2. The objective is strictly convex if Sigma is PD (unique optimal portfolio)
3. Uniqueness of the optimal portfolio (from strong convexity)

What this does NOT prove
------------------------
- That the covariance estimate Sigma is accurate (estimation error)
- That the expected returns mu are correctly estimated
- Transaction costs, liquidity constraints, or integer constraints
- Out-of-sample performance (this is in-sample formulation only)
- Constraint feasibility (return target may be unreachable)

What to do next
---------------
1. Estimate Sigma and mu from historical data (or a factor model)
2. Solve with CVXPY: cp.Problem(cp.Minimize(w @ Sigma @ w), constraints)
3. Backtest the portfolio on out-of-sample data
4. Sensitivity analysis: how does the optimal w change with Sigma?

Run: uv run python -m symproof.library.examples.convex.03_portfolio
"""

import sympy

from symproof import Axiom, AxiomSet, LemmaKind, ProofBuilder, seal
from symproof.library.convex import convex_hessian, strongly_convex, unique_minimizer

# ─── 2-asset Markowitz model ────────────────────────────────

w1 = sympy.Symbol("w1", real=True)   # weight of asset 1
w2 = sympy.Symbol("w2", real=True)   # weight of asset 2

# Volatilities and correlation
sigma1 = sympy.Symbol("sigma1", positive=True)  # vol of asset 1
sigma2 = sympy.Symbol("sigma2", positive=True)  # vol of asset 2
rho = sympy.Symbol("rho", real=True)            # correlation

# Portfolio variance: w^T Sigma w
# Sigma = [[sigma1^2,            rho*sigma1*sigma2],
#          [rho*sigma1*sigma2,   sigma2^2         ]]
portfolio_var = (
    w1**2 * sigma1**2
    + 2 * rho * sigma1 * sigma2 * w1 * w2
    + w2**2 * sigma2**2
)

# For the covariance matrix to be PSD, we need |rho| < 1.
# For PD (strict convexity), we need |rho| < 1 strictly,
# which means the two assets are not perfectly correlated.

# We'll work with concrete values for a provable example:
# sigma1 = 0.2 (20% annual vol), sigma2 = 0.3, rho = 0.5
concrete_var = portfolio_var.subs({
    sigma1: sympy.Rational(1, 5),
    sigma2: sympy.Rational(3, 10),
    rho: sympy.Rational(1, 2),
})

print("Markowitz portfolio optimization (2 assets)")
print("  sigma1 = 20%, sigma2 = 30%, rho = 0.5")
print(f"  Objective: {sympy.expand(concrete_var)}")

axioms = AxiomSet(
    name="portfolio",
    axioms=(Axiom(name="formulation", expr=sympy.Eq(1, 1)),),
)

# ─── Step 1: Prove convexity (Hessian PSD) ──────────────────

print("\n1. Convexity: Hessian of portfolio variance is PSD")
hess_bundle = convex_hessian(axioms, concrete_var, [w1, w2])
print(f"   {hess_bundle.proof_result.status.value}")

# The Hessian is 2*Sigma:
H = sympy.hessian(concrete_var, [w1, w2])
print(f"   Hessian = {H}")
print(f"   Eigenvalues: {H.eigenvals()}")

# ─── Step 2: Strong convexity → unique portfolio ────────────

# Minimum eigenvalue of 2*Sigma determines strong convexity parameter
eigvals = list(H.eigenvals().keys())
m = min(eigvals)
print(f"\n2. Strong convexity: min eigenvalue = {m}")

strong_bundle = strongly_convex(axioms, concrete_var, [w1, w2], m)
print(f"   {m}-strongly convex: {strong_bundle.proof_result.status.value}")

# ─── Step 3: Uniqueness of optimal portfolio ────────────────

print(f"\n3. Unique minimizer: {m}-strongly convex => unique optimal portfolio")
unique_bundle = unique_minimizer(axioms, concrete_var, [w1, w2], m)
print(f"   {unique_bundle.proof_result.status.value}")

# ─── Step 4: Compose into design evidence ───────────────────

print("\n4. Compose into single certification bundle:")
hypothesis = axioms.hypothesis(
    "portfolio_formulation_certified",
    expr=sympy.Gt(m, 0),
    description="Markowitz formulation is strictly convex with unique optimum",
)
script = (
    ProofBuilder(
        axioms, hypothesis.name,
        name="portfolio_certification",
        claim="Convex + strongly convex + unique minimizer",
    )
    .import_bundle(hess_bundle)
    .import_bundle(unique_bundle)
    .lemma(
        "strict_convexity",
        LemmaKind.QUERY,
        expr=sympy.Q.positive(m),
        description=f"Strong convexity parameter m = {m} > 0",
    )
    .build()
)
bundle = seal(axioms, hypothesis, script)
print(f"   {bundle.proof_result.status.value}")
print(f"   Hash: {bundle.bundle_hash}")

print()
print("This certifies: the Markowitz objective is strictly convex,")
print("so any local minimum found by the solver IS the global minimum,")
print("and the optimal portfolio weights are unique.")
print()
print("This does NOT certify: accuracy of sigma/rho estimates,")
print("constraint feasibility, or out-of-sample performance.")
print("Next: solve in CVXPY, backtest, sensitivity analysis.")
