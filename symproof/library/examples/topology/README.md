# Point-Set Topology (Undergraduate)

Topological properties of sets and functions on the real line.
This library stages toward advanced results (Brouwer fixed point
theorem) by establishing the prerequisites.

## Examples

| File | What it proves |
|------|---------------|
| `01_open_closed.py` | Open/closed/compact sets, boundary computation |
| `02_continuity.py` | Continuity via limits, intermediate value theorem |
| `03_extreme_value.py` | EVT: continuous f on [a,b] attains max and min |

## Run

```bash
uv run python -m symproof.library.examples.topology.01_open_closed
uv run python -m symproof.library.examples.topology.02_continuity
```

## Scope

Proves topological properties in **R** (real line):

- Openness, closedness via SymPy's set API
- Compactness via Heine-Borel (closed + bounded)
- Continuity at a point via epsilon-delta (limit = value)
- IVT: sign change implies root exists
- EVT: continuous on closed bounded interval has max and min

Does **not** cover:

- Abstract topological spaces (only subsets of R)
- R^n topology (no multi-dimensional sets)
- Metric space completeness, Cauchy sequences
- Brouwer fixed point theorem (future library — this builds prerequisites)

## Staging toward Brouwer

The Brouwer fixed point theorem requires: continuous function,
compact convex domain, mapping to itself. This library provides:

1. `verify_compact` — prove the domain is compact
2. `continuous_at_point` — prove the function is continuous
3. `intermediate_value` — the 1D special case of Brouwer

Combined with `symproof.library.convex` for convexity of the domain,
a future advanced library can axiomatize Brouwer and verify its
conditions are met for specific instances.
