# High School Physics with Calculus

Mathematical validation of standard AP Physics C results using
symproof's deterministic proof framework.  Each proof uses calculus
(derivatives, integrals) to establish a physics result.

## Examples

| File | What it proves |
|------|---------------|
| `01_kinematics.py` | Constant acceleration: v = dx/dt, a = dv/dt (linear and rotational) |
| `02_energy.py` | Work-energy theorem (W = delta KE) and impulse-momentum (J = delta p) |
| `03_shm.py` | SHM solution satisfies ODE; total energy is constant |
| `04_gravitation.py` | Gravitational potential U = -GMm/r from F = -GMm/r^2 |

## Run

```bash
uv run python -m symproof.library.examples.physics.01_kinematics
uv run python -m symproof.library.examples.physics.03_shm
```

## Scope

These proofs cover the **symbolic/calculus** content:

- Differentiation of position to get velocity and acceleration
- Verification that proposed solutions satisfy differential equations
- Conservation laws via time derivatives
- Integration of force to get potential energy

They do **not** cover:

- Numerical simulation (trajectories, collisions)
- Vector calculus (curl, divergence, Gauss's law)
- Relativistic corrections
- Quantum mechanics
