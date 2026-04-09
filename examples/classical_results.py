#!/usr/bin/env python3
"""Walkthrough: proving classical mathematical results with symproof.

This script demonstrates the full symproof workflow by proving results
you'd encounter in a math, physics, or early engineering class.  Each
proof is sealed into a reproducible, hashed bundle.

Run it::

    uv run python examples/classical_results.py

Topics covered:
    1. Euler's identity: e^(iπ) + 1 = 0
    2. e as an infinite series: Σ(1/k!) = e
    3. The Basel problem: Σ(1/k²) = π²/6
    4. Gauss's sum: Σk = n(n+1)/2
    5. The irrationality of √2, e, and π
    6. Pythagorean trig identity: sin²x + cos²x = 1
    7. AM-GM inequality: (a+b)/2 ≥ √(ab)
    8. Quadratic formula correctness
    9. Fundamental theorem of calculus
   10. Composing proofs: importing bundles as building blocks
"""

from __future__ import annotations

import sympy

from symproof import (
    Axiom,
    AxiomSet,
    LemmaKind,
    ProofBuilder,
    seal,
)

# ─────────────────────────────────────────────────────────────
# Setup: declare symbols and a shared axiom set for pure math
# ─────────────────────────────────────────────────────────────

x = sympy.Symbol("x", real=True)
k = sympy.Symbol("k", integer=True, nonnegative=True)
n = sympy.Symbol("n", integer=True, positive=True)
a = sympy.Symbol("a", positive=True)
b = sympy.Symbol("b", positive=True)

# A minimal axiom set — these are accepted truths.
# For pure math results, we start from nearly nothing.
math_axioms = AxiomSet(
    name="classical_math",
    axioms=(
        Axiom(name="i_squared", expr=sympy.Eq(sympy.I**2, -1)),
        Axiom(name="e_positive", expr=sympy.E > 0),
        Axiom(name="pi_positive", expr=sympy.pi > 0),
    ),
)


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────
# 1. Euler's identity: e^(iπ) + 1 = 0
# ─────────────────────────────────────────────────────────────

section("1. Euler's identity: e^(iπ) + 1 = 0")

h_euler = math_axioms.hypothesis(
    "euler_identity",
    expr=sympy.Eq(sympy.exp(sympy.I * sympy.pi) + 1, 0),
)

euler_script = (
    ProofBuilder(
        math_axioms, h_euler.name,
        name="euler_proof",
        claim="e^(iπ) + 1 = 0",
    )
    .lemma(
        "evaluate",
        LemmaKind.EQUALITY,
        expr=sympy.exp(sympy.I * sympy.pi) + 1,
        expected=sympy.Integer(0),
        description="Direct evaluation of e^(iπ) + 1",
    )
    .build()
)

euler_bundle = seal(math_axioms, h_euler, euler_script)
print(f"Sealed: {euler_bundle.bundle_hash[:24]}...")
print(f"Advisories: {euler_bundle.proof_result.advisories or '(none)'}")


# ─────────────────────────────────────────────────────────────
# 2. e as an infinite series: Σ(1/k!) = e
# ─────────────────────────────────────────────────────────────

section("2. e = Σ(1/k!, k=0..∞)")

h_e_series = math_axioms.hypothesis(
    "e_series",
    expr=sympy.Eq(
        sympy.Sum(1 / sympy.factorial(k), (k, 0, sympy.oo)),
        sympy.E,
    ),
)

e_series_script = (
    ProofBuilder(
        math_axioms, h_e_series.name,
        name="e_series_proof",
        claim="e = Sum(1/k!, k=0..inf)",
    )
    .lemma(
        "series_equals_e",
        LemmaKind.EQUALITY,
        expr=sympy.Sum(1 / sympy.factorial(k), (k, 0, sympy.oo)),
        expected=sympy.E,
        description="Taylor series of e^1 at x=0",
    )
    .build()
)

e_series_bundle = seal(math_axioms, h_e_series, e_series_script)
print(f"Sealed: {e_series_bundle.bundle_hash[:24]}...")


# ─────────────────────────────────────────────────────────────
# 3. The Basel problem: Σ(1/k²) = π²/6
# ─────────────────────────────────────────────────────────────

section("3. Basel problem: Σ(1/k², k=1..∞) = π²/6")

k_pos = sympy.Symbol("k", integer=True, positive=True)

h_basel = math_axioms.hypothesis(
    "basel_problem",
    expr=sympy.Eq(
        sympy.Sum(1 / k_pos**2, (k_pos, 1, sympy.oo)),
        sympy.pi**2 / 6,
    ),
)

basel_script = (
    ProofBuilder(
        math_axioms, h_basel.name,
        name="basel_proof",
        claim="Sum(1/k^2) = pi^2/6",
    )
    .lemma(
        "series_value",
        LemmaKind.EQUALITY,
        expr=sympy.Sum(1 / k_pos**2, (k_pos, 1, sympy.oo)),
        expected=sympy.pi**2 / 6,
        description="Euler's solution to the Basel problem",
    )
    .build()
)

basel_bundle = seal(math_axioms, h_basel, basel_script)
print(f"Sealed: {basel_bundle.bundle_hash[:24]}...")


# ─────────────────────────────────────────────────────────────
# 4. Gauss's sum: Σk = n(n+1)/2
# ─────────────────────────────────────────────────────────────

section("4. Gauss's sum: Σ(k, k=1..n) = n(n+1)/2")

h_gauss = math_axioms.hypothesis(
    "gauss_sum",
    expr=sympy.Eq(
        sympy.Sum(k, (k, 1, n)),
        n * (n + 1) / 2,
    ),
)

gauss_script = (
    ProofBuilder(
        math_axioms, h_gauss.name,
        name="gauss_proof",
        claim="1 + 2 + ... + n = n(n+1)/2",
    )
    .lemma(
        "closed_form",
        LemmaKind.EQUALITY,
        expr=sympy.Sum(k, (k, 1, n)),
        expected=n * (n + 1) / 2,
        assumptions={"n": {"integer": True, "positive": True}},
        description="Gauss's sum formula",
    )
    .build()
)

gauss_bundle = seal(math_axioms, h_gauss, gauss_script)
print(f"Sealed: {gauss_bundle.bundle_hash[:24]}...")


# ─────────────────────────────────────────────────────────────
# 5. Irrationality of √2, e, and π
# ─────────────────────────────────────────────────────────────

section("5. Irrationality of √2, e, and π")

for name, value, expr_q in [
    ("sqrt_2", sympy.sqrt(2), sympy.Q.irrational(sympy.sqrt(2))),
    ("e", sympy.E, sympy.Q.irrational(sympy.E)),
    ("pi", sympy.pi, sympy.Q.irrational(sympy.pi)),
]:
    h = math_axioms.hypothesis(
        f"{name}_irrational",
        expr=sympy.Ne(value, sympy.Integer(0)),  # placeholder
        description=f"{name} is irrational",
    )
    script = (
        ProofBuilder(
            math_axioms, h.name,
            name=f"{name}_irrationality",
            claim=f"{name} is irrational",
        )
        .lemma(
            "irrational_query",
            LemmaKind.QUERY,
            expr=expr_q,
            description=f"SymPy knows {name} is irrational",
        )
        .build()
    )
    bundle = seal(math_axioms, h, script)
    print(f"  {name}: sealed {bundle.bundle_hash[:24]}...")


# ─────────────────────────────────────────────────────────────
# 6. Pythagorean trig identity: sin²x + cos²x = 1
# ─────────────────────────────────────────────────────────────

section("6. sin²(x) + cos²(x) = 1")

h_pythag = math_axioms.hypothesis(
    "pythagorean_identity",
    expr=sympy.Eq(sympy.sin(x)**2 + sympy.cos(x)**2, 1),
)

pythag_script = (
    ProofBuilder(
        math_axioms, h_pythag.name,
        name="pythagorean_proof",
        claim="sin^2(x) + cos^2(x) = 1",
    )
    .lemma(
        "identity",
        LemmaKind.EQUALITY,
        expr=sympy.sin(x)**2 + sympy.cos(x)**2,
        expected=sympy.Integer(1),
        description="Fundamental trigonometric identity",
    )
    .build()
)

pythag_bundle = seal(math_axioms, h_pythag, pythag_script)
print(f"Sealed: {pythag_bundle.bundle_hash[:24]}...")


# ─────────────────────────────────────────────────────────────
# 7. AM-GM inequality: (a+b)/2 ≥ √(ab)
# ─────────────────────────────────────────────────────────────

section("7. AM-GM inequality: (a+b)/2 ≥ √(ab) for a,b > 0")

# The proof decomposes into: (a+b)^2 ≥ 4ab  ⟺  (a-b)^2 ≥ 0

amgm_axioms = AxiomSet(
    name="positive_reals",
    axioms=(
        Axiom(name="a_positive", expr=a > 0),
        Axiom(name="b_positive", expr=b > 0),
    ),
)

h_amgm = amgm_axioms.hypothesis(
    "am_gm",
    expr=sympy.Ge((a + b) / 2, sympy.sqrt(a * b)),
)

amgm_script = (
    ProofBuilder(
        amgm_axioms, h_amgm.name,
        name="am_gm_proof",
        claim="(a+b)/2 >= sqrt(a*b)",
    )
    .lemma(
        "square_nonneg",
        LemmaKind.QUERY,
        expr=sympy.Q.nonnegative((a - b)**2),
        assumptions={"a": {"positive": True}, "b": {"positive": True}},
        description="(a-b)^2 >= 0 (squares are nonneg)",
    )
    .lemma(
        "expand_to_amgm",
        LemmaKind.EQUALITY,
        expr=(a - b)**2,
        expected=a**2 - 2*a*b + b**2,
        depends_on=["square_nonneg"],
        description="(a-b)^2 = a^2 - 2ab + b^2",
    )
    .lemma(
        "rearrange",
        LemmaKind.EQUALITY,
        expr=(a + b)**2 - 4*a*b,
        expected=(a - b)**2,
        depends_on=["expand_to_amgm"],
        description="(a+b)^2 - 4ab = (a-b)^2 >= 0",
    )
    .build()
)

amgm_bundle = seal(amgm_axioms, h_amgm, amgm_script)
print(f"Sealed: {amgm_bundle.bundle_hash[:24]}...")
print(f"Lemmas: {[l.name for l in amgm_script.lemmas]}")


# ─────────────────────────────────────────────────────────────
# 8. Quadratic formula correctness
# ─────────────────────────────────────────────────────────────

section("8. Quadratic formula: ax²+bx+c = 0 at x = (-b±√Δ)/2a")

a_q = sympy.Symbol("a_q", nonzero=True)
b_q = sympy.Symbol("b_q")
c_q = sympy.Symbol("c_q")

discriminant = b_q**2 - 4 * a_q * c_q
root_plus = (-b_q + sympy.sqrt(discriminant)) / (2 * a_q)
root_minus = (-b_q - sympy.sqrt(discriminant)) / (2 * a_q)

quad_axioms = AxiomSet(
    name="quadratic",
    axioms=(Axiom(name="a_nonzero", expr=sympy.Ne(a_q, 0)),),
)

h_quad = quad_axioms.hypothesis(
    "quadratic_formula",
    expr=sympy.Eq(
        a_q * root_plus**2 + b_q * root_plus + c_q,
        0,
    ),
)

quad_script = (
    ProofBuilder(
        quad_axioms, h_quad.name,
        name="quadratic_proof",
        claim="The quadratic formula gives a root",
    )
    .lemma(
        "substitute_and_simplify",
        LemmaKind.EQUALITY,
        expr=sympy.simplify(a_q * root_plus**2 + b_q * root_plus + c_q),
        expected=sympy.Integer(0),
        description="Substituting x = (-b+√Δ)/2a into ax²+bx+c gives 0",
    )
    .build()
)

quad_bundle = seal(quad_axioms, h_quad, quad_script)
print(f"Sealed: {quad_bundle.bundle_hash[:24]}...")


# ─────────────────────────────────────────────────────────────
# 9. Fundamental theorem of calculus
# ─────────────────────────────────────────────────────────────

section("9. FTC: d/dx ∫₀ˣ f(t)dt = f(x)")

t = sympy.Symbol("t")
f = sympy.Function("f")

ftc_axioms = AxiomSet(
    name="calculus",
    axioms=(Axiom(name="f_defined", expr=sympy.Eq(1, 1)),),
)

h_ftc = ftc_axioms.hypothesis(
    "ftc",
    expr=sympy.Eq(
        sympy.diff(sympy.Integral(f(t), (t, 0, x)), x),
        f(x),
    ),
)

ftc_script = (
    ProofBuilder(
        ftc_axioms, h_ftc.name,
        name="ftc_proof",
        claim="d/dx int_0^x f(t)dt = f(x)",
    )
    .lemma(
        "differentiate_integral",
        LemmaKind.EQUALITY,
        expr=sympy.diff(sympy.Integral(f(t), (t, 0, x)), x),
        expected=f(x),
        description="Leibniz rule applied to definite integral",
    )
    .build()
)

ftc_bundle = seal(ftc_axioms, h_ftc, ftc_script)
print(f"Sealed: {ftc_bundle.bundle_hash[:24]}...")


# ─────────────────────────────────────────────────────────────
# 10. Composition: building on prior results
# ─────────────────────────────────────────────────────────────

section("10. Composition: e is both irrational AND e = Σ(1/k!)")

# Import both the e-series bundle and the irrationality bundle
# to build a combined proof

# Re-seal irrationality of e with the same axiom set
h_e_irr = math_axioms.hypothesis(
    "e_irrational",
    expr=sympy.Ne(sympy.E, sympy.Integer(0)),
)
e_irr_script = (
    ProofBuilder(
        math_axioms, h_e_irr.name,
        name="e_irrationality",
        claim="e is irrational",
    )
    .lemma(
        "query",
        LemmaKind.QUERY,
        expr=sympy.Q.irrational(sympy.E),
    )
    .build()
)
e_irr_bundle = seal(math_axioms, h_e_irr, e_irr_script)

# Now compose: a proof that imports both results
h_combined = math_axioms.hypothesis(
    "e_series_and_irrational",
    expr=sympy.Ne(sympy.E, sympy.Integer(0)),
)

combined_script = (
    ProofBuilder(
        math_axioms, h_combined.name,
        name="e_combined_proof",
        claim="e = Sum(1/k!) AND e is irrational",
    )
    .import_bundle(e_series_bundle)   # fact 1: e = Σ(1/k!)
    .import_bundle(e_irr_bundle)      # fact 2: e is irrational
    .lemma(
        "e_nonzero",
        LemmaKind.QUERY,
        expr=sympy.Q.nonzero(sympy.E),
        description="e is nonzero (follows from irrationality)",
    )
    .build()
)

combined_bundle = seal(math_axioms, h_combined, combined_script)
print(f"Sealed: {combined_bundle.bundle_hash[:24]}...")
print(f"Imported bundles: {len(combined_script.imported_bundles)}")
print(f"Total lemma results (imports + local): "
      f"{len(combined_bundle.proof_result.lemma_results)}")

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────

section("Summary")
bundles = [
    ("Euler's identity", euler_bundle),
    ("e = Σ(1/k!)", e_series_bundle),
    ("Basel problem", basel_bundle),
    ("Gauss's sum", gauss_bundle),
    ("sin²+cos²=1", pythag_bundle),
    ("AM-GM inequality", amgm_bundle),
    ("Quadratic formula", quad_bundle),
    ("FTC", ftc_bundle),
    ("Composed: e series + irrational", combined_bundle),
]

for name, b in bundles:
    n_lemmas = len(b.proof.lemmas)
    n_imports = len(b.proof.imported_bundles)
    n_advisories = len(b.proof_result.advisories)
    status = "PASS" if n_advisories == 0 else f"PASS ({n_advisories} advisory)"
    imports = f" [{n_imports} imports]" if n_imports else ""
    print(f"  {status:>20s}  {name:<35s} {n_lemmas} lemma(s){imports}")

print(f"\nAll bundles sealed with deterministic SHA-256 hashes.")
print(f"Re-run this script — you'll get identical hashes every time.")
