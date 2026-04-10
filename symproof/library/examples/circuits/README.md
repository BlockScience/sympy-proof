# Boolean Circuits with ZK Applications

Verify properties of boolean circuits and their use in zero-knowledge
proof systems.  The library validates circuit correctness, equivalence,
and witness satisfaction — it does not construct or simulate circuits.

## Examples

| File | What it proves |
|------|---------------|
| `01_gates.py` | XOR truth table, AND gate output for specific inputs |
| `02_equivalence.py` | XOR == (a AND NOT b) OR (NOT a AND b) |
| `03_zk_witness.py` | R1CS witness satisfies constraints; output entropy quantified |

## Run

```bash
uv run python -m symproof.library.examples.circuits.01_gates
uv run python -m symproof.library.examples.circuits.03_zk_witness
```

## Scope

Proves **structural properties** of boolean circuits:

- Gate correctness against truth tables
- Circuit functional equivalence
- Witness satisfaction for R1CS (ZK-SNARK constraint format)
- Output entropy (information leakage measure)

Does **not** cover:

- Circuit synthesis or optimization
- ZK proof generation (Groth16, PLONK, etc.)
- Cryptographic security proofs
- Large-scale circuit simulation (limited by SymPy's symbolic engine)

## ZK context

In a ZK-SNARK, a prover demonstrates knowledge of a witness w such
that A*w . B*w = C*w (R1CS format) without revealing w.  The
`r1cs_witness_check` function verifies this relationship — the sealed
proof bundle certifies "a valid witness exists" with a deterministic hash.

## Staging toward Shannon theory

`boolean_entropy` computes the Shannon entropy of a circuit's output
distribution.  XOR has maximum entropy (1 bit) — it perfectly hides
input information.  AND has entropy 0.811 bits — the output distribution
is biased toward False, leaking information about inputs.  A future
Shannon theory library will build on this for channel capacity and
mutual information analysis.
