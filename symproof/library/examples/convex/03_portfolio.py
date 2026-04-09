#!/usr/bin/env python3
"""Markowitz portfolio: certify formulation, catch degenerate cases.

Scenario
--------
You're building a portfolio optimizer.  Before deploying (or submitting
for risk review), you need to certify:
  1. The objective is convex (solver finds global optimum)
  2. It's strictly convex (optimal portfolio is unique)
  3. The formulation doesn't degenerate (e.g., perfectly correlated assets)

This example proves all three, then shows what breaks when assets are
perfectly correlated — the strong convexity proof correctly fails.

Scaling limits
--------------
Symbolic Hessian PSD via Sylvester's criterion works for 2-4 assets.
For real portfolios (50-500 assets), use numerical eigenvalue checks
instead: ``np.linalg.eigvalsh(Sigma).min() > 0``.  symproof is for
certifying the FORMULATION STRUCTURE, not for 500x500 symbolic matrices.

What this proves
----------------
- Covariance matrix Sigma is PSD → objective is convex
- Sigma is PD (when |rho| < 1) → objective is strictly convex → unique portfolio
- When rho = 1 (perfect correlation), strict convexity FAILS (correctly)

What this does NOT prove
------------------------
- Accuracy of Sigma and mu estimates (estimation error is the real risk)
- Constraint feasibility (target return may be unreachable)
- Out-of-sample performance (in-sample optimal ≠ out-of-sample optimal)
- Transaction costs, liquidity, or integer share constraints

What to do next
---------------
1. Estimate Sigma robustly (shrinkage, factor model)
2. Solve: ``cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), constraints)``
3. Backtest on out-of-sample data
4. Stress test: re-run with perturbed Sigma to check sensitivity

Run: uv run python -m symproof.library.examples.convex.03_portfolio
"""

import sympy

from symproof import Axiom, AxiomSet, LemmaKind, ProofBuilder, seal
from symproof.library.convex import convex_hessian, strongly_convex, unique_minimizer

w1 = sympy.Symbol("w1", real=True)
w2 = sympy.Symbol("w2", real=True)

# ─── Concrete 2-asset portfolio: sigma1=20%, sigma2=30%, rho=0.5 ─

portfolio_var = (
    w1**2 * sympy.Rational(1, 25)         # sigma1^2 = 0.04
    + 2 * sympy.Rational(1, 2) * sympy.Rational(1, 5) * sympy.Rational(3, 10) * w1 * w2
    + w2**2 * sympy.Rational(9, 100)      # sigma2^2 = 0.09
)
# Simplifies to: w1²/25 + 3·w1·w2/50 + 9·w2²/100

axioms = AxiomSet(
    name="portfolio",
    axioms=(Axiom(name="formulation", expr=sympy.Eq(1, 1)),),
)

print("Markowitz portfolio (2 assets, sigma1=20%, sigma2=30%, rho=0.5)")
print(f"  Objective: {sympy.expand(portfolio_var)}")

# Step 1: Hessian PSD → convex
H = sympy.hessian(portfolio_var, [w1, w2])
print(f"  Hessian: {H}")

hess_bundle = convex_hessian(axioms, portfolio_var, [w1, w2])
print(f"\n1. Convex (Hessian PSD)? {hess_bundle.proof_result.status.value}")

# Step 2: Strong convexity → unique portfolio
eigvals = sorted(H.eigenvals().keys())
m = eigvals[0]  # minimum eigenvalue
print(f"2. Min eigenvalue: {sympy.simplify(m)} ≈ {float(m):.6f}")

strong_bundle = strongly_convex(axioms, portfolio_var, [w1, w2], m)
print(f"   {m}-strongly convex? {strong_bundle.proof_result.status.value}")

# Step 3: Unique minimizer
unique_bundle = unique_minimizer(axioms, portfolio_var, [w1, w2], m)
print(f"3. Unique optimal portfolio? {unique_bundle.proof_result.status.value}")

# Step 4: Compose into single certification
hyp = axioms.hypothesis(
    "portfolio_certified",
    expr=sympy.Gt(m, 0),
    description="Convex + strictly convex + unique optimum",
)
script = (
    ProofBuilder(axioms, hyp.name, name="cert", claim="portfolio certified")
    .import_bundle(hess_bundle)
    .import_bundle(unique_bundle)
    .lemma("m_positive", LemmaKind.QUERY, expr=sympy.Q.positive(m))
    .build()
)
bundle = seal(axioms, hyp, script)
print(f"\n4. Sealed certification: {bundle.bundle_hash[:32]}...")


# ─── FAILURE CASE: perfectly correlated assets (rho = 1) ────
#
# When rho = 1, the covariance matrix is rank-1 (singular).
# The objective is still convex (PSD) but NOT strictly convex —
# the minimum is not unique (any portfolio on the efficient
# frontier is equally optimal).
#
# The strong convexity proof should FAIL.

print("\n" + "="*60)
print("  Failure case: perfectly correlated assets (rho = 1)")
print("="*60)

portfolio_corr1 = (
    w1**2 * sympy.Rational(1, 25)
    + 2 * sympy.Rational(1, 5) * sympy.Rational(3, 10) * w1 * w2
    + w2**2 * sympy.Rational(9, 100)
)

H_corr1 = sympy.hessian(portfolio_corr1, [w1, w2])
det_corr1 = H_corr1.det()
print(f"  Hessian: {H_corr1}")
print(f"  det(Hessian) = {det_corr1}")

# Convexity still holds (PSD — det >= 0)
hess_corr1 = convex_hessian(axioms, portfolio_corr1, [w1, w2])
print(f"\n  Convex? {hess_corr1.proof_result.status.value} (still PSD)")

# But strong convexity FAILS (det = 0 means eigenvalue = 0)
eigvals_corr1 = sorted(H_corr1.eigenvals().keys())
m_corr1 = eigvals_corr1[0]
print(f"  Min eigenvalue: {m_corr1}")

try:
    strong_corr1 = strongly_convex(
        axioms, portfolio_corr1, [w1, w2], m_corr1,
    )
    print(f"  Strictly convex? {strong_corr1.proof_result.status.value}")
except ValueError:
    print("  Strictly convex? REJECTED (correctly)")
    print("  → Multiple optimal portfolios exist")
    print("  → Need additional constraints to select among them")

print()
print("Scaling note: symbolic Hessian works for 2-4 assets.")
print("For 500-asset portfolios, use: np.linalg.eigvalsh(Sigma).min() > 0")
print("symproof certifies the FORMULATION; numerical checks certify the DATA.")
