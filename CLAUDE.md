# CLAUDE.md — symproof

## Package Identity

`symproof` — deterministic proof writing with SymPy. Declare axioms, bind hypotheses, build lemma chains, and seal reproducible hashed proof bundles.

- **Import**: `import symproof`
- **Dependencies**: `sympy>=1.12`, `pydantic>=2.0`
- **Python**: `>=3.12`
- **Layout**: flat (no src/)

## Commands

```bash
uv sync
uv run python -m pytest tests/ -v
uv run ruff check symproof/
uv run ruff format --check symproof/
```

## Architecture

Ten modules in a flat package:

| Module | Purpose |
|--------|---------|
| `types.py` | `SympyExpr`, `SympyBoolean` type aliases |
| `models.py` | All Pydantic models: Axiom, AxiomSet, Citation, Hypothesis, Lemma, ProofScript, ProofBundle, Disproof |
| `serialization.py` | `canonical_srepr`, `restore_expr`, `make_canonical_dict` |
| `hashing.py` | `hash_axiom_set`, `hash_proof`, `hash_bundle` — SHA-256 identity |
| `evaluation.py` | `unevaluated`, `evaluation` — explicit SymPy evaluation control |
| `verification.py` | `verify_lemma`, `verify_proof` — three-strategy dispatch (gated by `evaluation()`) |
| `tactics.py` | `try_simplify`, `try_implication`, `try_query`, `auto_lemma` — automatic helpers |
| `builder.py` | `ProofBuilder` — chainable construction API |
| `bundle.py` | `seal`, `disprove`, `check_consistency` — bundle operations + foundation/consistency/load-bearing checks |

### Evidence Chain

```
AxiomSet.canonical_dict() → hash_axiom_set → axiom_set_hash
    → Hypothesis carries axiom_set_hash
    → ProofScript carries axiom_set_hash
        → hash_proof(script) → proof_hash
            → verify_proof(script) → ProofResult
                → seal(axiom_set, hypothesis, script) → ProofBundle(bundle_hash)
                    → disprove(H, negation_bundle) → Disproof(disproof_hash)
```

### Key Design Principles

1. **No hypothesis without axioms** — Hypothesis always carries `axiom_set_hash`. Preferred construction: `axiom_set.hypothesis(...)`.
2. **No hidden axioms** — When a proof depends on an external theorem via `seal(foundations=...)`, every axiom in the foundation must appear in the downstream axiom set. Missing axioms are a hard error. Inherited axioms are marked `Axiom(inherited=True)` with `citation=Citation(source="...")`.
3. **Explicit evaluation** — Expressions are constructed under `unevaluated()` to preserve structure. Evaluation happens only at explicit `evaluation()` gates around `simplify()`/`ask()`/`refine()`. No hidden eager evaluation.
4. **Deterministic identity** — `sympy.srepr()` for expressions, `json.dumps(sort_keys=True)` for dicts, SHA-256 for hashes.
5. **Separation of concerns** — `verify_proof` checks mathematical validity only. Context binding, foundation coverage, consistency, and load-bearing checks enforced in `seal()`.
6. **Compositional disproof** — Proving ~H under A, then composing into a Disproof of H.
7. **All models frozen** — Pydantic frozen=True for all value objects.

### Foundation Enforcement

When a proof axiomatises an external theorem (`expr=sympy.S.true`), the foundation proof that backs it may require additional assumptions. `seal()` enforces coverage:

```python
seal(axioms, hypothesis, script,
     foundations=[(foundation_bundle, "theorem_name")])
# Raises ValueError if foundation has axioms not in `axioms`
```

The `Axiom.inherited` field distinguishes posited axioms (design choices) from inherited axioms (conditions forced by the proof chain). Inherited axioms require `citation=Citation(source="...")` for traceability. Both affect the axiom set hash.

### Additional Guards

- **No false axioms** — `AxiomSet` rejects axioms whose `simplify(expr)` is `False` at construction time.
- **Pairwise consistency** — `seal()` rejects axiom pairs where `simplify(And(a, b))` is `False` (opt out with `check_consistency=False`).
- **Load-bearing accounting** — `seal()` rejects proofs where `Symbol("x", positive=True)` affects verification but `x > 0` is not declared as an axiom.
- **Assumption reporting** — Every sealed bundle's advisories enumerate all posited, inherited, and external assumptions with citations.

### Key Constraints

- All SymPy serialization uses `sympy.srepr()`, never `str()`
- Absolute imports only
- `__version__` in `__init__.py` is single source of truth
- `make_canonical_dict()` is the single normalization entry point
- `seal()` is the only path to produce a `ProofBundle`
- `disprove()` is the only path to produce a `Disproof`
