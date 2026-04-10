# Shannon Information Theory

Exact symbolic proofs of information-theoretic quantities: entropy,
mutual information, KL divergence, and channel capacity.  All values
are computed in exact symbolic form (no floating-point approximation).

## Examples

| File | What it proves |
|------|---------------|
| `01_entropy.py` | Shannon entropy of distributions; binary entropy H(p) |
| `02_mutual_info.py` | Mutual information: I=0 for independent, I=1 for perfectly correlated |
| `03_channels.py` | BSC capacity; KL divergence with Gibbs' inequality |

## Run

```bash
uv run python -m symproof.library.examples.information.01_entropy
uv run python -m symproof.library.examples.information.03_channels
```

## Connection to boolean circuits

The `boolean_entropy` function in `symproof.library.circuits` computes
the output entropy of a boolean function — it uses the same Shannon
entropy formula but over the truth table distribution.  This library
provides the general-purpose information-theoretic tools; the circuits
library applies them to specific boolean functions.
