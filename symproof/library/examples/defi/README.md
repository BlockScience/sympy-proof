# DeFi Examples

Mathematical validation of DeFi mechanisms — the symbolic-math leg of a
three-part verification architecture (symbolic → simulation → code audit).

These examples prove properties of **formulas**, not code. They establish
that the economist's math has the behavioral properties the protocol claims,
before any Solidity is written or audited.

## Examples

| File | What it proves |
|------|---------------|
| `01_amm_swap_audit.py` | Full AMM audit: rounding direction, phantom overflow, directional error chains, decimal normalization, composed safety bundle |

## Run

```bash
uv run python -m symproof.library.examples.defi.01_amm_swap_audit
```

## Scope

These proofs cover the **validation** layer:

- Is the swap formula correct?
- Does rounding error favor the right party?
- Does the net rounding across a pipeline protect the protocol?
- Do intermediate products overflow machine word sizes?
- Is cross-decimal pricing normalized?

They do **not** cover:

- Does the Solidity match the formula? → code audit / formal verification
- Is the system robust to adversarial inputs? → simulation / fuzzing
- Are state transitions atomic? → static analysis
