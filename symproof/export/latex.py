"""LaTeX rendering for symproof proof artifacts.

All functions return LaTeX source as ``str``.  No file I/O — the
caller decides where to write it.

Expression rendering uses ``sympy.latex()`` with default settings.
Assumptions dicts are rendered as mathematical statements.
Hashes are truncated to 16 characters in display text.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy

if TYPE_CHECKING:
    from symproof.models import (
        Lemma,
        LemmaResult,
        ProofBundle,
        ProofResult,
        ProofScript,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _escape(text: str) -> str:
    """Escape LaTeX special characters in plain text.

    Uses ``\\texorpdfstring`` workarounds where the standard escape
    produces ugly output (e.g., ``^`` → ``\\^{}`` for a clean caret).
    """
    # Order matters: backslash first, then characters that could
    # appear in the replacement strings.
    text = text.replace("\\", r"\textbackslash{}")
    text = text.replace("&", r"\&")
    text = text.replace("%", r"\%")
    text = text.replace("$", r"\$")
    text = text.replace("#", r"\#")
    text = text.replace("_", r"\_")
    text = text.replace("{", r"\{")
    text = text.replace("}", r"\}")
    text = text.replace("~", r"\~{}")
    text = text.replace("^", r"\^{}")
    text = text.replace("<", r"\textless{}")
    text = text.replace(">", r"\textgreater{}")
    return text


def _hash_short(h: str) -> str:
    """Truncate a hash for display."""
    return f"{h[:16]}\\ldots"


def _strip_assumptions(expr: sympy.Basic) -> sympy.Basic:
    """Replace assumption-bearing symbols with bare symbols.

    SymPy eagerly evaluates expressions like ``x > 0`` to ``True``
    when ``x`` has ``positive=True``.  For display purposes we want
    the unevaluated form, so we substitute bare symbols first.
    """
    subs = {}
    for sym in expr.free_symbols:
        if sym.assumptions0:  # has non-default assumptions
            bare = sympy.Symbol(sym.name)
            if bare != sym:
                subs[sym] = bare
    return expr.subs(subs) if subs else expr


def _expr_tex(expr: sympy.Basic) -> str:
    """Render a SymPy expression to LaTeX math.

    Strips symbol assumptions before rendering so that expressions
    like ``x > 0`` display as ``x > 0`` instead of ``\\text{True}``.
    """
    stripped = _strip_assumptions(expr)
    return sympy.latex(stripped)


def _expr_or_fallback(expr: sympy.Basic, fallback: str) -> str:
    """Render expression, falling back to text if it evaluated to True/False.

    SymPy eagerly evaluates concrete expressions: ``Eq(I**2, -1)``
    becomes ``True``, ``x > 0`` with ``positive=True`` becomes ``True``.
    When this happens the mathematical content is lost, so we show the
    fallback (typically the axiom name or description) instead.
    """
    if expr is sympy.true:
        return rf"\text{{{_escape(fallback)}}}" if fallback else r"\top"
    if expr is sympy.false:
        return rf"\text{{{_escape(fallback)}}}" if fallback else r"\bot"
    stripped = _strip_assumptions(expr)
    if stripped is sympy.true:
        return rf"\text{{{_escape(fallback)}}}" if fallback else r"\top"
    if stripped is sympy.false:
        return rf"\text{{{_escape(fallback)}}}" if fallback else r"\bot"
    return sympy.latex(stripped)


_ASSUMPTION_LABELS: dict[str, str] = {
    "positive": "> 0",
    "nonnegative": r"\geq 0",
    "negative": "< 0",
    "nonpositive": r"\leq 0",
    "nonzero": r"\neq 0",
    "integer": r"\in \mathbb{Z}",
    "real": r"\in \mathbb{R}",
    "rational": r"\in \mathbb{Q}",
    "even": r"\text{ even}",
    "odd": r"\text{ odd}",
    "prime": r"\text{ prime}",
    "finite": r"\text{ finite}",
}


def _render_assumptions(assumptions: dict[str, dict]) -> str:
    """Render assumptions dict as comma-separated math statements."""
    parts: list[str] = []
    for sym_name, asm in sorted(assumptions.items()):
        sym_tex = sympy.latex(sympy.Symbol(sym_name))
        for prop, val in sorted(asm.items()):
            if val:
                label = _ASSUMPTION_LABELS.get(prop, rf"\text{{{prop}}}")
                parts.append(f"{sym_tex} {label}")
    return ", ".join(parts) if parts else ""


_KIND_LABELS: dict[str, str] = {
    "equality": "equality",
    "boolean": "boolean",
    "query": "query",
    "property": "property",
    "inference": "inference",
    "coordinate_transform": "coordinate transform",
}


# ---------------------------------------------------------------------------
# Single lemma
# ---------------------------------------------------------------------------


def latex_lemma(
    lemma: Lemma,
    result: LemmaResult | None = None,
    *,
    index: int | None = None,
) -> str:
    """Render a single lemma as a LaTeX description item.

    Parameters
    ----------
    lemma:
        The lemma to render.
    result:
        Optional verification result for this lemma.
    index:
        Optional 1-based index for numbering.
    """
    lines: list[str] = []

    # Header
    kind_label = _KIND_LABELS.get(lemma.kind.value, lemma.kind.value)
    prefix = f"Lemma {index}" if index is not None else _escape(lemma.name)
    lines.append(
        rf"\item[{prefix}] \textbf{{{_escape(lemma.name)}}}"
        rf" \emph{{({kind_label})}}"
    )

    # Main expression
    if lemma.rule:
        # INFERENCE: show the rule being applied
        lines.append(
            rf"  \emph{{Rule:}} {_escape(lemma.rule)}"
        )
    elif lemma.property_name:
        # PROPERTY: show "subject.property_name" for readability
        lines.append(
            rf"  $\displaystyle {_expr_tex(lemma.expr)}$"
            rf" \texttt{{.{_escape(lemma.property_name)}}}"
        )
    else:
        lines.append(rf"  $\displaystyle {_expr_tex(lemma.expr)}$")

    # Expected (for EQUALITY)
    if lemma.expected is not None:
        lines.append(rf"  $= {_expr_tex(lemma.expected)}$")

    # Transform maps (for COORDINATE_TRANSFORM)
    if lemma.transform is not None:
        maps = ", \\quad ".join(
            f"{sympy.latex(sympy.Symbol(k))} \\mapsto {_expr_tex(v)}"
            for k, v in lemma.transform.items()
        )
        lines.append(rf"  \emph{{Forward:}} ${maps}$")
    if lemma.inverse_transform is not None:
        maps = ", \\quad ".join(
            f"{sympy.latex(sympy.Symbol(k))} \\mapsto {_expr_tex(v)}"
            for k, v in lemma.inverse_transform.items()
        )
        lines.append(rf"  \emph{{Inverse:}} ${maps}$")

    # Assumptions
    asm_tex = _render_assumptions(lemma.assumptions)
    if asm_tex:
        lines.append(rf"  \emph{{Assumptions:}} ${asm_tex}$")

    # Dependencies
    if lemma.depends_on:
        deps = ", ".join(_escape(d) for d in lemma.depends_on)
        lines.append(rf"  \emph{{Depends on:}} {deps}")

    # Description
    if lemma.description:
        lines.append(rf"  \par {_escape(lemma.description)}")

    # Result
    if result is not None:
        status = r"\textsc{passed}" if result.passed else r"\textsc{failed}"
        lines.append(rf"  \par {status}")
        if result.error:
            lines.append(rf"  --- {_escape(result.error)}")
        for adv in result.advisories:
            lines.append(rf"  \par \emph{{Advisory:}} {_escape(adv)}")

    return " \\\\\n".join(lines)


# ---------------------------------------------------------------------------
# Proof script
# ---------------------------------------------------------------------------


def _latex_imported_bundle(
    bundle: ProofBundle,
    depth: int = 0,
) -> str:
    """Render an imported bundle showing its internal proof structure.

    Each imported bundle is a complete proof: hypothesis, its own
    lemma chain, and possibly its own imports (recursively).
    The ``depth`` parameter controls indentation for nested imports.
    """
    indent = "  " * depth
    lines: list[str] = []
    h = bundle.hypothesis
    h_fb = h.description or h.name
    h_tex = _expr_or_fallback(h.expr, h_fb)

    lines.append(
        rf"{indent}\item \textbf{{{_escape(h.name)}}}: "
        rf"${h_tex}$ "
        rf"\quad \texttt{{{_hash_short(bundle.bundle_hash)}}}"
    )

    # Show the lemma chain inside this imported bundle
    if bundle.proof.lemmas:
        lemma_names = ", ".join(
            _escape(lem.name) for lem in bundle.proof.lemmas
        )
        lines.append(
            rf"{indent}  \par \emph{{Proof chain:}} {lemma_names}"
        )

    # Recurse into sub-imports
    if bundle.proof.imported_bundles:
        lines.append(
            rf"{indent}  \par \emph{{Built on:}}"
        )
        lines.append(rf"{indent}  \begin{{itemize}}")
        for sub in bundle.proof.imported_bundles:
            lines.append(_latex_imported_bundle(sub, depth=depth + 2))
        lines.append(rf"{indent}  \end{{itemize}}")

    return "\n".join(lines)


def latex_proof(
    script: ProofScript,
    result: ProofResult | None = None,
) -> str:
    """Render a proof script as a LaTeX fragment.

    Returns a ``\\subsection*`` level fragment with the lemma chain.
    Imported bundles are rendered recursively, showing their internal
    proof structure — because each import is itself a complete proof.

    Does not include the top-level axioms or hypothesis — use
    ``latex_bundle`` for the complete rendering.
    """
    lines: list[str] = []

    # Imported bundles (rendered recursively)
    if script.imported_bundles:
        lines.append(r"\subsubsection*{Imported Proofs}")
        lines.append(
            r"\emph{Each imported proof is a sealed bundle with its "
            r"own hypothesis and lemma chain.}"
        )
        lines.append(r"\begin{itemize}")
        for imp in script.imported_bundles:
            lines.append(_latex_imported_bundle(imp, depth=1))
        lines.append(r"\end{itemize}")
        lines.append("")

    # Lemma chain
    lines.append(r"\subsubsection*{Lemma Chain}")
    lines.append(r"\begin{description}")

    # Match lemma results to lemmas by name
    result_map: dict[str, LemmaResult] = {}
    if result is not None:
        for lr in result.lemma_results:
            result_map[lr.lemma_name] = lr

    for i, lemma in enumerate(script.lemmas, 1):
        lr = result_map.get(lemma.name)
        lines.append(latex_lemma(lemma, result=lr, index=i))
        lines.append("")

    lines.append(r"\end{description}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Full bundle
# ---------------------------------------------------------------------------


def latex_bundle(bundle: ProofBundle) -> str:
    r"""Render a sealed proof bundle as a LaTeX fragment.

    Returns a ``\section*`` level fragment.  Wrap in a document
    with ``latex_document()`` for a standalone ``.tex`` file.
    """
    lines: list[str] = []
    proof = bundle.proof
    result = bundle.proof_result

    # Title
    lines.append(rf"\section*{{Proof: {_escape(proof.name)}}}")
    lines.append(
        rf"\textbf{{Claim:}} {_escape(proof.claim)} \\"
    )
    lines.append(
        rf"\textbf{{Status:}} \textsc{{{result.status.value.lower()}}}"
        rf" \quad \texttt{{{_hash_short(bundle.bundle_hash)}}}"
    )
    lines.append("")

    # Axioms — grouped by provenance
    ax = bundle.axiom_set
    posited = [a for a in ax.axioms if not a.inherited]
    inherited = [a for a in ax.axioms if a.inherited]

    lines.append(rf"\subsection*{{Axioms: {_escape(ax.name)}}}")

    if posited:
        lines.append(r"\paragraph{Posited (design choices)}")
        lines.append(r"\begin{enumerate}")
        for axiom in posited:
            name = _escape(axiom.name)
            fallback = axiom.description or axiom.name
            expr_tex = _expr_or_fallback(axiom.expr, fallback)
            lines.append(rf"  \item \textbf{{{name}}}: ${expr_tex}$")
            if axiom.description and axiom.expr is not sympy.true:
                lines.append(rf"    \par {_escape(axiom.description)}")
        lines.append(r"\end{enumerate}")

    if inherited:
        lines.append(r"\paragraph{Inherited (from foundation proofs)}")
        lines.append(
            r"\emph{These conditions were not chosen by the proof "
            r"author --- they are required by external results the "
            r"proof depends on.}"
        )
        lines.append(r"\begin{enumerate}")
        for axiom in inherited:
            name = _escape(axiom.name)
            fallback = axiom.description or axiom.name
            expr_tex = _expr_or_fallback(axiom.expr, fallback)
            lines.append(rf"  \item \textbf{{{name}}}: ${expr_tex}$")
            if axiom.citation:
                lines.append(
                    rf"    \par \emph{{Source:}} "
                    rf"{_escape(axiom.citation.source)}"
                )
            if axiom.description and axiom.expr is not sympy.true:
                lines.append(rf"    \par {_escape(axiom.description)}")
        lines.append(r"\end{enumerate}")
    lines.append("")

    # Hypothesis
    lines.append(r"\subsection*{Hypothesis}")
    h = bundle.hypothesis
    fallback = h.description or h.name
    lines.append(rf"$\displaystyle {_expr_or_fallback(h.expr, fallback)}$")
    if h.description and h.expr is not sympy.true:
        lines.append(rf"\par {_escape(h.description)}")
    lines.append("")

    # Proof chain
    lines.append(r"\subsection*{Proof}")
    lines.append(latex_proof(proof, result=result))
    lines.append("")

    # Advisories
    if result.advisories:
        lines.append(r"\subsection*{Advisories}")
        lines.append(r"\begin{itemize}")
        for adv in result.advisories:
            lines.append(rf"  \item {_escape(adv)}")
        lines.append(r"\end{itemize}")
        lines.append("")

    # Hash footer
    lines.append(r"\vspace{1em}")
    lines.append(r"\noindent\rule{\textwidth}{0.4pt}")
    lines.append(
        rf"\texttt{{bundle\_hash: {bundle.bundle_hash}}}"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Standalone document
# ---------------------------------------------------------------------------


def latex_document(bundle: ProofBundle) -> str:
    r"""Render a sealed proof bundle as a standalone LaTeX document.

    Wraps ``latex_bundle()`` in a minimal ``article`` preamble.
    The output can be compiled directly with ``pdflatex``.
    """
    body = latex_bundle(bundle)
    return rf"""\documentclass[11pt]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage{{amsmath,amssymb,amsthm}}
\usepackage{{geometry}}
\geometry{{margin=1in}}

\title{{Proof Bundle: {_escape(bundle.proof.name)}}}
\date{{\today}}

\begin{{document}}
\maketitle

{body}

\end{{document}}
"""
