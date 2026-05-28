# token-budgets-python

Python port of the [token-budgets](https://github.com/sajjadanwar0/token-budgets)
Rust library. Part of the *Token Budgets* artifact (preprint, 2026).

This port exists for two reasons:

1. **Reproducibility of LANG-001 and related incidents.** Many of the
   catalogue's documented failures originated in Python frameworks (LangChain,
   LangGraph, CrewAI, AutoGen, Aider). Reproducing them in a Python harness is
   methodologically cleaner than rewriting them in Rust.
2. **Practical accessibility.** Most LLM agent developers work in Python.

## Important caveats

- **The Python port does not provide the affine type guarantees of the Rust
  crate.** Python has no equivalent of Rust's move semantics or affine types.
  The port enforces caps via *runtime* checks (raising `BudgetExceeded` on
  overspend), not compile-time guarantees.
- It is therefore operationally equivalent to runtime cost-monitoring approaches
  (Agent Contracts; AgentGuard; LiteLLM proxy budgets): it catches violations
  dynamically but does not prevent budget cloning, double-spending, or
  post-delegation use at the type level. The paper's contribution is the Rust
  crate's compile-time prevention; this port is supplementary.
- A best-effort `mypy` plugin (v0.2) catches single-function double-spend and
  use-after-split patterns at type-check time, but it is **not sound**; treat the
  runtime `_consumed` flag as the primary enforcement layer.

## Quick start

```bash
pip install token-budgets        # or: uv pip install token-budgets
```

```python
from token_budgets import Budget, BudgetExceeded

b = Budget.new(1000)
lhs, rhs = b.split(400)
lhs.spend(350)
try:
    lhs.spend(100)               # runtime exception, not a compile-time error
except BudgetExceeded as e:
    print(f"Caught at runtime: {e}")
```

`copy.copy`, `copy.deepcopy`, and `pickle` are blocked via `AffineViolation`; a
residual `__new__` escape remains as a Python language limitation.

## LANG-001 and other reproductions

```bash
cd lang001_reproduction
python3 reproduce_lang001.py --max-budget 5000
```

| Incident | Script                    | Framework |
|----------|---------------------------|-----------|
| LANG-001 | `lang001_reproduction/`   | LangChain |
| CRAI-001 | `crai001_reproduction/`   | CrewAI    |
| AIDR-003 | `aidr003_reproduction/`   | Aider     |
| ATGN-018 | `atgn018_reproduction/`   | AutoGen   |

The runtime catch occurs *after* some cost has been incurred — unlike the Rust
version, which refuses the cap-violating call pre-flight.

## Companion components

- [token-budgets](https://github.com/sajjadanwar0/token-budgets) — main Rust library (canonical)
- token-budgets-formals — mechanised cross-checks + IRR (κ = 0.837, N = 113)
- token-budgets-experiments — multi-runtime evaluation

## License

Dual MIT/Apache-2.0.