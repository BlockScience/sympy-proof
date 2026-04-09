#!/usr/bin/env python3
"""LaTeX export: human-readable views of computational proofs.

The computational proof (srepr + SHA-256) is canonical and machine-
readable.  The LaTeX view is what a reviewer, collaborator, or design
review board reads.  Both are derived from the same sealed bundle —
the hash in the LaTeX footer matches the hash in the Python object.

This example takes proofs from earlier examples and renders them as
LaTeX fragments you can paste into a paper, audit report, or V&V
traceability matrix.

Run: uv run python examples/06_latex_export.py

Run with file output:
    uv run python examples/06_latex_export.py > proof.tex
    pdflatex proof.tex   # if you have LaTeX installed
"""

from __future__ import annotations

import sys

import sympy

from symproof import Axiom, AxiomSet, LemmaKind, ProofBuilder, seal
from symproof.export import latex_bundle

# ─── Build a few proofs (same patterns as earlier examples) ──────

# 1. Euler's identity
euler_axioms = AxiomSet(
    name="complex_analysis",
    axioms=(Axiom(name="i_squared", expr=sympy.Eq(sympy.I**2, -1)),),
)
h_euler = euler_axioms.hypothesis(
    "euler_identity",
    expr=sympy.Eq(sympy.exp(sympy.I * sympy.pi) + 1, 0),
    description="Euler's identity",
)
euler_script = (
    ProofBuilder(
        euler_axioms, h_euler.name,
        name="euler_proof",
        claim="Euler's identity: e^(i*pi) + 1 = 0",
    )
    .lemma(
        "evaluate",
        LemmaKind.EQUALITY,
        expr=sympy.exp(sympy.I * sympy.pi) + 1,
        expected=sympy.Integer(0),
        description="Direct evaluation via SymPy",
    )
    .build()
)
euler_bundle = seal(euler_axioms, h_euler, euler_script)

# 2. AM-GM inequality (multi-lemma)
a = sympy.Symbol("a", positive=True)
b = sympy.Symbol("b", positive=True)
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
    description="AM-GM inequality for two positive reals",
)
amgm_script = (
    ProofBuilder(
        amgm_axioms, h_amgm.name,
        name="am_gm_proof",
        claim="(a+b)/2 >= sqrt(a*b) for a, b > 0",
    )
    .lemma(
        "square_nonneg",
        LemmaKind.QUERY,
        expr=sympy.Q.nonnegative((a - b) ** 2),
        assumptions={"a": {"positive": True}, "b": {"positive": True}},
        description="Any square is nonnegative",
    )
    .lemma(
        "expand_square",
        LemmaKind.EQUALITY,
        expr=(a - b) ** 2,
        expected=a**2 - 2 * a * b + b**2,
        depends_on=["square_nonneg"],
        description="Expand (a-b)^2",
    )
    .lemma(
        "rearrange",
        LemmaKind.EQUALITY,
        expr=(a + b) ** 2 - 4 * a * b,
        expected=(a - b) ** 2,
        depends_on=["expand_square"],
        description="(a+b)^2 - 4ab = (a-b)^2 >= 0",
    )
    .build()
)
amgm_bundle = seal(amgm_axioms, h_amgm, amgm_script)

# 3. Gauss's sum (uses .doit() fallback — triggers advisory)
k = sympy.Symbol("k", integer=True, nonnegative=True)
n = sympy.Symbol("n", integer=True, positive=True)
gauss_axioms = AxiomSet(
    name="arithmetic",
    axioms=(Axiom(name="n_positive", expr=n > 0),),
)
h_gauss = gauss_axioms.hypothesis(
    "gauss_sum",
    expr=sympy.Eq(sympy.Sum(k, (k, 1, n)), n * (n + 1) / 2),
    description="Gauss's sum formula",
)
gauss_script = (
    ProofBuilder(
        gauss_axioms, h_gauss.name,
        name="gauss_proof",
        claim="1 + 2 + ... + n = n(n+1)/2",
    )
    .lemma(
        "closed_form",
        LemmaKind.EQUALITY,
        expr=sympy.Sum(k, (k, 1, n)),
        expected=n * (n + 1) / 2,
        assumptions={"n": {"integer": True, "positive": True}},
        description="Evaluate the sum to its closed form",
    )
    .build()
)
gauss_bundle = seal(gauss_axioms, h_gauss, gauss_script)


# ─── Render to LaTeX ─────────────────────────────────────────────

# If output is a terminal, show fragments with section breaks.
# If piped to a file, produce a standalone document.

if sys.stdout.isatty():
    print("=" * 64)
    print("  LaTeX views of sealed proof bundles")
    print("=" * 64)

    for name, bundle in [
        ("Euler's Identity", euler_bundle),
        ("AM-GM Inequality", amgm_bundle),
        ("Gauss's Sum", gauss_bundle),
    ]:
        print(f"\n{'─' * 64}")
        print(f"  {name}")
        print(f"  Hash: {bundle.bundle_hash}")
        print(f"{'─' * 64}\n")
        print(latex_bundle(bundle))
        print()

    print("=" * 64)
    print("  To get a compilable .tex file, pipe to a file:")
    print("    uv run python examples/06_latex_export.py > proof.tex")
    print("=" * 64)

else:
    # Piped to file — produce a standalone document with all three
    # proofs.  We use latex_document for the first and latex_bundle
    # for the rest, assembling manually.
    body_parts = [latex_bundle(b) for b in [
        euler_bundle, amgm_bundle, gauss_bundle,
    ]]
    body = "\n\n\\newpage\n\n".join(body_parts)
    print(rf"""\documentclass[11pt]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage{{amsmath,amssymb,amsthm}}
\usepackage{{geometry}}
\geometry{{margin=1in}}

\title{{Proof Bundle Collection}}
\author{{Generated by symproof}}
\date{{\today}}

\begin{{document}}
\maketitle

{body}

\end{{document}}
""")
