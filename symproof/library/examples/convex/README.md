# Convex Optimization Examples

Mathematical validation of convexity properties — certifying that
objective functions, constraints, and compositions satisfy the
conditions required by optimization algorithms.

These examples prove properties of **mathematical programs**. They
establish that the problem formulation is actually convex before
a solver is called or code is deployed.

## Examples

| File | What it proves |
|------|---------------|
| `01_loss_function.py` | ML loss function convexity (squared error, cross-entropy, log-barrier) |
| `02_regularization.py` | Ridge regression: regularization strengthens convexity |
| `03_portfolio.py` | Markowitz portfolio: strict convexity and unique minimizer |

## Run

```bash
uv run python -m symproof.library.examples.convex.01_loss_function
uv run python -m symproof.library.examples.convex.02_regularization
uv run python -m symproof.library.examples.convex.03_portfolio
```

## Scope

These proofs cover the **validation** layer:

- Is the objective function convex (or strongly convex)?
- Does composition preserve convexity (DCP rules)?
- Does the regularizer guarantee a unique minimizer?
- Is the Hessian positive (semi)definite over the domain?

They do **not** cover:

- Numerical conditioning of the solver → simulation / benchmarking
- Floating-point convergence behavior → numerical analysis
- Implementation correctness of the solver → code verification
- Constraint feasibility under real-world data → integration testing
