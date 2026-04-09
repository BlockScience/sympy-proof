# Control Systems Examples

Mathematical validation of control system designs — stability,
controllability, observability, and Lyapunov analysis.

These examples prove properties of **linear continuous-time models**.
They establish the analytical prerequisites before simulation and
hardware-in-the-loop testing.

## Examples

| File | What it proves |
|------|---------------|
| `01_stability.py` | Plant + controller closed-loop Hurwitz stability |
| `02_lyapunov.py` | Lyapunov function construction and positive definiteness |
| `03_controllability.py` | Controllability and observability rank conditions |
| `04_composition.py` | Compose stability + controllability + Lyapunov into one traceable artifact |

## Run

```bash
uv run python -m symproof.library.examples.control.01_stability
uv run python -m symproof.library.examples.control.04_composition
```

## Scope

These proofs cover the **validation** layer:

- Is the closed-loop polynomial Hurwitz stable?
- Does a Lyapunov function exist for the linearized model?
- Is the system controllable with the chosen actuator?
- Is the system observable with the chosen sensors?

They do **not** cover:

- Robustness to parameter uncertainty → mu-analysis / Monte Carlo
- Discrete-time controller stability → discretize and re-verify
- Actuator saturation or sensor noise → nonlinear simulation
- Fatigue, thermal, or aging effects → environmental testing
