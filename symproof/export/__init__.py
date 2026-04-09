"""Export views for symproof proof bundles.

The computational models (``sympy.srepr``, SHA-256 hashes) are the
source of truth.  These modules produce *views* — human-readable
renderings for different audiences and tools.

LaTeX::

    from symproof.export import latex_bundle, latex_document
    tex = latex_bundle(bundle)
    tex = latex_document(bundle)

Graph (DAG)::

    from symproof.export import proof_dag, proof_dag_dot, proof_dag_mermaid
    dag = proof_dag(bundle)            # plain dict: nodes + edges
    dot = proof_dag_dot(bundle)        # graphviz DOT source
    mmd = proof_dag_mermaid(bundle)    # mermaid source for GitHub/Notion
"""

from symproof.export.graph import (
    proof_dag,
    proof_dag_dot,
    proof_dag_json,
    proof_dag_mermaid,
)
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
    "proof_dag",
    "proof_dag_dot",
    "proof_dag_json",
    "proof_dag_mermaid",
]
