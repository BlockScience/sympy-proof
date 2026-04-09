"""LaTeX export for symproof proof bundles.

Renders the computational proof models as human-readable LaTeX.
The computational models (``sympy.srepr``, SHA-256 hashes) remain
canonical; the LaTeX is a *view*, not a source of truth.

Public API::

    from symproof.export import latex_bundle, latex_proof, latex_lemma

    tex = latex_bundle(bundle)          # full bundle as LaTeX fragment
    tex = latex_document(bundle)        # standalone .tex file
    tex = latex_proof(script, result)   # just the proof chain
    tex = latex_lemma(lemma, result)    # single lemma
"""

from symproof.export.latex import (
    latex_bundle,
    latex_document,
    latex_lemma,
    latex_proof,
)

__all__ = [
    "latex_bundle",
    "latex_document",
    "latex_lemma",
    "latex_proof",
]
