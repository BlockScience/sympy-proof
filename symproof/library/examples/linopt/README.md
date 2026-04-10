# Linear Optimization (LP/ILP/MILP) Validation

Validate solutions from LP/ILP solvers using symproof.  The library
does not solve — it proves that a given solution has the claimed
properties: feasibility, optimality, duality, integrality.

## Examples

| File | What it proves |
|------|---------------|
| `01_feasibility.py` | A candidate point satisfies Ax = b and x >= 0 |
| `02_optimality.py` | Full LP optimality via primal + dual feasibility + strong duality |
| `03_integer.py` | ILP feasibility (integer + constraints) and LP relaxation bound |

## Run

```bash
uv run python -m symproof.library.examples.linopt.01_feasibility
uv run python -m symproof.library.examples.linopt.02_optimality
```

## Scope

These proofs validate **solution certificates**, not solver correctness:

- Is x* feasible? (substitution check)
- Is x* optimal? (KKT / duality check)
- Is the LP relaxation a valid bound? (algebraic comparison)

They do **not** cover:

- Solver correctness (is Gurobi's simplex implementation bug-free?)
- Numerical stability (floating-point pivoting, degeneracy)
- Algorithm complexity (simplex iterations, branch-and-bound nodes)
