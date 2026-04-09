# symproof

Deterministic proof writing with SymPy.

Declare axioms, bind hypotheses, build lemma chains, and seal reproducible hashed proof bundles.

## Install

```bash
uv add symproof
```

## Quick Start

```python
import sympy
from symproof import Axiom, AxiomSet, ProofBuilder, LemmaKind, seal

x, y = sympy.symbols("x y")

# Declare axioms
axioms = AxiomSet(name="positive_reals", axioms=(
    Axiom(name="x_positive", expr=x > 0),
    Axiom(name="y_positive", expr=y > 0),
))

# Bind a hypothesis to the axiom set
h = axioms.hypothesis("product_positive", expr=x * y > 0)

# Build a proof
script = (
    ProofBuilder(axioms, h.name, name="pos_product", claim="x*y > 0")
    .lemma("xy_positive", LemmaKind.QUERY,
           expr=sympy.Q.positive(x * y),
           assumptions={"x": {"positive": True}, "y": {"positive": True}})
    .build()
)

# Seal into a reproducible bundle
bundle = seal(axioms, h, script)
print(bundle.bundle_hash)  # deterministic 64-char SHA-256
```
