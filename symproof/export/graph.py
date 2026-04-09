"""DAG visualization for symproof proof bundles.

Renders proof bundles as directed acyclic graphs where:
- Nodes are (axiom_set, hypothesis, proof) triples or individual lemmas
- Edges are ``depends_on`` (within a proof) or ``imports`` (across proofs)

All functions return plain dicts/lists — no external dependencies.
Render with graphviz, networkx, mermaid, or any graph library.

Public API::

    from symproof.export.graph import (
        proof_dag, proof_dag_json, proof_dag_dot, proof_dag_mermaid,
    )

    dag = proof_dag(bundle)            # dict with nodes + edges
    jsn = proof_dag_json(bundle)       # JSON node-link (d3/networkx/neo4j)
    dot = proof_dag_dot(bundle)        # graphviz DOT source
    mmd = proof_dag_mermaid(bundle)    # mermaid source
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sympy

if TYPE_CHECKING:
    from symproof.models import LemmaResult, ProofBundle


# ---------------------------------------------------------------------------
# DAG extraction
# ---------------------------------------------------------------------------


def _bundle_id(bundle: ProofBundle) -> str:
    """Short stable identifier for a bundle node."""
    return bundle.bundle_hash[:16]


def _lemma_id(bundle_hash: str, lemma_name: str) -> str:
    """Unique lemma node identifier within a bundle."""
    return f"{bundle_hash[:16]}::{lemma_name}"


def _extract_bundle(
    bundle: ProofBundle,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    visited: set[str],
    *,
    expand_lemmas: bool = True,
) -> None:
    """Recursively extract nodes and edges from a bundle and its imports."""
    bid = _bundle_id(bundle)
    if bid in visited:
        return
    visited.add(bid)

    result = bundle.proof_result
    result_map: dict[str, LemmaResult] = {}
    for lr in result.lemma_results:
        result_map[lr.lemma_name] = lr

    # Bundle node
    n_advisories = len(result.advisories)
    nodes.append({
        "id": bid,
        "type": "bundle",
        "name": bundle.proof.name,
        "hypothesis": bundle.hypothesis.name,
        "claim": bundle.proof.claim,
        "status": result.status.value,
        "hash": bundle.bundle_hash,
        "lemma_count": len(bundle.proof.lemmas),
        "import_count": len(bundle.proof.imported_bundles),
        "advisory_count": n_advisories,
    })

    # Lemma nodes + intra-proof edges
    if expand_lemmas:
        for lemma in bundle.proof.lemmas:
            lid = _lemma_id(bundle.bundle_hash, lemma.name)
            lr = result_map.get(lemma.name)

            nodes.append({
                "id": lid,
                "type": "lemma",
                "name": lemma.name,
                "kind": lemma.kind.value,
                "parent_bundle": bid,
                "passed": lr.passed if lr else None,
                "has_advisories": bool(lr and lr.advisories),
                "description": lemma.description,
                "expr_preview": sympy.latex(lemma.expr)[:80],
            })

            # Edge: bundle contains this lemma
            edges.append({
                "source": bid,
                "target": lid,
                "type": "contains",
            })

            # Edges: lemma depends_on
            for dep in lemma.depends_on:
                dep_id = _lemma_id(bundle.bundle_hash, dep)
                edges.append({
                    "source": lid,
                    "target": dep_id,
                    "type": "depends_on",
                })

    # Recurse into imported bundles
    for imp in bundle.proof.imported_bundles:
        imp_id = _bundle_id(imp)
        edges.append({
            "source": bid,
            "target": imp_id,
            "type": "imports",
        })
        _extract_bundle(
            imp, nodes, edges, visited, expand_lemmas=expand_lemmas,
        )


def proof_dag(
    *bundles: ProofBundle,
    expand_lemmas: bool = True,
) -> dict[str, Any]:
    """Extract a DAG from one or more proof bundles.

    Returns a dict with ``nodes`` and ``edges`` lists suitable for
    any graph rendering library.

    Parameters
    ----------
    *bundles:
        One or more sealed ProofBundles.  Imported bundles are
        included recursively.
    expand_lemmas:
        If True (default), each lemma is a separate node with
        ``depends_on`` edges.  If False, only bundle-level nodes
        and import edges are shown.

    Returns
    -------
    dict
        ``{"nodes": [...], "edges": [...]}`` where each node and
        edge is a plain dict with string keys.
    """
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    visited: set[str] = set()

    for bundle in bundles:
        _extract_bundle(
            bundle, nodes, edges, visited,
            expand_lemmas=expand_lemmas,
        )

    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# JSON node-link format (d3.js / networkx / neo4j)
# ---------------------------------------------------------------------------


def proof_dag_json(
    *bundles: ProofBundle,
    expand_lemmas: bool = True,
    indent: int = 2,
) -> str:
    """Render a proof DAG as JSON in the node-link format.

    The output is directly consumable by:

    - **d3.js** force-directed graph (``d3.forceSimulation``)
    - **networkx** (``nx.node_link_graph(json.loads(s))``)
    - **neo4j** (``UNWIND $nodes AS n CREATE (n:Node) SET n = n``)
    - **vis.js**, **cytoscape.js**, or any tool expecting node-link JSON

    Schema::

        {
          "directed": true,
          "multigraph": false,
          "graph": {                      # metadata
            "name": "Proof DAG",
            "schema_version": 1,
            "tool_hints": {
              "d3": "nodes[].id as key, links[].source/target as id strings",
              "networkx": "nx.node_link_graph(data)",
              "neo4j": "UNWIND $nodes AS n CREATE (n:Node) SET n = n"
            }
          },
          "nodes": [
            {"id": "...", "type": "bundle"|"lemma", ...},
          ],
          "links": [
            {"source": "...", "target": "...",
             "type": "imports"|"depends_on"|"contains"},
          ]
        }

    The ``directed``, ``multigraph``, and ``graph`` keys follow the
    networkx node-link convention so that ``nx.node_link_graph()``
    works without any transformation.

    Parameters
    ----------
    *bundles:
        Proof bundles to serialize.
    expand_lemmas:
        Include individual lemma nodes (True) or bundle-level only.
    indent:
        JSON indentation.  Set to ``None`` for compact output.

    Returns
    -------
    str
        JSON string.
    """
    import json

    dag = proof_dag(*bundles, expand_lemmas=expand_lemmas)

    output = {
        "directed": True,
        "multigraph": False,
        "graph": {
            "name": "Proof DAG",
            "schema_version": 1,
            "tool_hints": {
                "d3": (
                    "Use nodes[].id as node key. "
                    "links[].source and links[].target are id strings. "
                    "Node type is 'bundle' or 'lemma'."
                ),
                "networkx": (
                    "import networkx as nx; "
                    "G = nx.node_link_graph(json.loads(s))"
                ),
                "neo4j": (
                    "UNWIND $data.nodes AS n "
                    "CREATE (p:ProofNode) SET p = n; "
                    "UNWIND $data.links AS e "
                    "MATCH (a:ProofNode {id: e.source}), "
                    "(b:ProofNode {id: e.target}) "
                    "CREATE (a)-[:LINK {type: e.type}]->(b)"
                ),
            },
        },
        "nodes": dag["nodes"],
        "links": [
            {
                "source": e["source"],
                "target": e["target"],
                "type": e["type"],
            }
            for e in dag["edges"]
        ],
    }

    return json.dumps(output, indent=indent)


# ---------------------------------------------------------------------------
# Graphviz DOT output
# ---------------------------------------------------------------------------

_STATUS_COLORS = {
    "VERIFIED": "#2d7d46",
    "FAILED": "#c0392b",
    "UNCHECKED": "#95a5a6",
}

_KIND_SHAPES = {
    "equality": "box",
    "boolean": "diamond",
    "query": "ellipse",
    "coordinate_transform": "hexagon",
}


def proof_dag_dot(
    *bundles: ProofBundle,
    expand_lemmas: bool = True,
    title: str = "Proof DAG",
) -> str:
    """Render a proof DAG as Graphviz DOT source.

    Parameters
    ----------
    *bundles:
        Proof bundles to visualize.
    expand_lemmas:
        Show individual lemma nodes (True) or bundle-level only (False).
    title:
        Graph title.

    Returns
    -------
    str
        DOT source string.  Render with ``dot -Tpng`` or ``graphviz``.
    """
    dag = proof_dag(*bundles, expand_lemmas=expand_lemmas)
    lines = [
        f'digraph "{title}" {{',
        '  rankdir=BT;',
        '  node [fontname="Helvetica", fontsize=10];',
        '  edge [fontsize=8];',
        '',
    ]

    for node in dag["nodes"]:
        nid = node["id"].replace("::", "__").replace("-", "_")
        if node["type"] == "bundle":
            color = _STATUS_COLORS.get(node["status"], "#95a5a6")
            label = (
                f'{node["name"]}\\n'
                f'{node["hypothesis"]}\\n'
                f'{node["hash"][:12]}...'
            )
            adv = (
                f'\\n({node["advisory_count"]} advisory)'
                if node["advisory_count"] else ""
            )
            lines.append(
                f'  "{nid}" ['
                f'shape=box3d, '
                f'style=filled, '
                f'fillcolor="{color}20", '
                f'color="{color}", '
                f'penwidth=2, '
                f'label="{label}{adv}"'
                f'];'
            )
        else:
            passed = node.get("passed")
            color = (
                "#2d7d46" if passed
                else "#c0392b" if passed is False
                else "#95a5a6"
            )
            shape = _KIND_SHAPES.get(node["kind"], "box")
            label = f'{node["name"]}\\n({node["kind"]})'
            if node.get("has_advisories"):
                label += "\\n⚠"
            lines.append(
                f'  "{nid}" ['
                f'shape={shape}, '
                f'style=filled, '
                f'fillcolor="{color}20", '
                f'color="{color}", '
                f'label="{label}"'
                f'];'
            )

    lines.append('')

    for edge in dag["edges"]:
        src = edge["source"].replace("::", "__").replace("-", "_")
        tgt = edge["target"].replace("::", "__").replace("-", "_")
        etype = edge["type"]
        if etype == "imports":
            lines.append(
                f'  "{src}" -> "{tgt}" ['
                f'style=bold, color="#2c3e50", '
                f'label="imports"'
                f'];'
            )
        elif etype == "depends_on":
            lines.append(
                f'  "{src}" -> "{tgt}" ['
                f'style=dashed, color="#7f8c8d", '
                f'label="depends_on"'
                f'];'
            )
        elif etype == "contains":
            lines.append(
                f'  "{src}" -> "{tgt}" ['
                f'style=dotted, color="#bdc3c7", arrowhead=none'
                f'];'
            )

    lines.append('}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Mermaid output
# ---------------------------------------------------------------------------


def proof_dag_mermaid(
    *bundles: ProofBundle,
    expand_lemmas: bool = True,
) -> str:
    """Render a proof DAG as Mermaid source.

    Mermaid renders in GitHub markdown, Notion, and many documentation
    tools without needing Graphviz installed.

    Returns
    -------
    str
        Mermaid source string.
    """
    dag = proof_dag(*bundles, expand_lemmas=expand_lemmas)
    lines = ["graph BT"]

    for node in dag["nodes"]:
        nid = (
            node["id"]
            .replace("::", "__")
            .replace("-", "_")
            .replace(" ", "_")
        )
        if node["type"] == "bundle":
            label = f'{node["name"]}<br/>{node["hypothesis"]}'
            status = node["status"]
            lines.append(f'    {nid}["{label}"]')
            if status == "VERIFIED":
                lines.append(f'    style {nid} fill:#d4edda,stroke:#2d7d46')
            elif status == "FAILED":
                lines.append(f'    style {nid} fill:#f8d7da,stroke:#c0392b')
        else:
            label = f'{node["name"]}<br/>({node["kind"]})'
            kind = node["kind"]
            if kind == "equality":
                lines.append(f'    {nid}["{label}"]')
            elif kind == "boolean":
                lines.append(f"    {nid}{{{{{label}}}}}")
            elif kind == "query":
                lines.append(f'    {nid}("{label}")')
            else:
                lines.append(f'    {nid}["{label}"]')

            if node.get("passed"):
                lines.append(
                    f'    style {nid} fill:#d4edda,stroke:#2d7d46'
                )
            elif node.get("passed") is False:
                lines.append(
                    f'    style {nid} fill:#f8d7da,stroke:#c0392b'
                )

    for edge in dag["edges"]:
        src = (
            edge["source"]
            .replace("::", "__")
            .replace("-", "_")
            .replace(" ", "_")
        )
        tgt = (
            edge["target"]
            .replace("::", "__")
            .replace("-", "_")
            .replace(" ", "_")
        )
        etype = edge["type"]
        if etype == "imports":
            lines.append(f'    {src} ==>|imports| {tgt}')
        elif etype == "depends_on":
            lines.append(f'    {src} -.->|depends_on| {tgt}')
        elif etype == "contains":
            lines.append(f'    {src} --- {tgt}')

    return '\n'.join(lines)
