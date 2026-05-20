# token-budgets-python

Python port of the [token-budgets](https://github.com/sajjadanwar0/token-budgets) Rust library.

This port exists for two reasons:

1. **Reproducibility of LANG-001 and related incidents.** Many of the catalog's documented failures originated in Python frameworks (LangChain, LangGraph, CrewAI, AutoGen, Aider). Reproducing them in a Python harness is methodologically cleaner than rewriting them in Rust.
2. **Practical accessibility.** Most LLM agent developers work in Python. A Python port lets the ideas in the paper be evaluated by practitioners directly, without requiring a Rust build chain.

## Important caveats

- **The Python port does not provide the affine type guarantees of the Rust crate.** Python has no equivalent of Rust's move semantics or affine type system. The Python port enforces budget caps via *runtime* checks (raising `BudgetExceeded` when a budget is overspent), not compile-time guarantees.
- This means the Python port is, in effect, equivalent to the Agent Contracts (Ye & Tan, COINE 2026) approach: runtime cost monitoring. It catches violations dynamically but does not prevent budget cloning, double-spending, or post-delegation use at the type level.
- The Rust crate in [token-budgets](https://github.com/sajjadanwar0/token-budgets) is the canonical implementation; this Python port is supplementary.

## Quick start

```bash
pip install token-budgets  # or: uv pip install token-budgets
```

```python
from token_budgets import Budget, BudgetExceeded

b = Budget.new(1000)
lhs, rhs = b.split(400)
lhs.spend(350)
try:
    lhs.spend(100)
except BudgetExceeded as e:
    print(f"Caught at runtime: {e}")
```

Unlike the Rust version, `lhs.spend(100)` here raises a runtime exception rather than being prevented at compile time. The catch is documented; the paper's contribution is the Rust crate's compile-time prevention.

## LANG-001 reproduction

```bash
cd lang001_reproduction
python3 reproduce_lang001.py --max-budget 5000
```

This reproduces the LangChain retry-loop incident documented in the main catalog as LANG-001. The reproduction confirms that:

1. Without a budget cap, the agent loops indefinitely on a tool error, consuming all available API credits.
2. With the Python `Budget`, the agent halts gracefully when the runtime budget is exhausted.
3. The runtime catch occurs *after* some cost has been incurred (unlike the Rust version which prevents the cost from being incurred at all).

## Other reproductions

| Incident   | Script                              | Framework   |
|------------|-------------------------------------|-------------|
| LANG-001   | `lang001_reproduction/`             | LangChain   |
| CRAI-001   | `crai001_reproduction/`             | CrewAI      |
| AIDR-003   | `aidr003_reproduction/`             | Aider       |
| ATGN-018   | `atgn018_reproduction/`             | AutoGen     |

Each reproduction has a README explaining the failure mode, the reproduction methodology, and the per-framework setup required.

## Why doesn't this Python port use type annotations or runtime contracts?

Both were prototyped:

- **`typing.NewType` and `Final`** can't prevent `b.spend(100); b.spend(100)` because Python's type checker doesn't track move semantics.
- **`functools.singledispatch` with explicit ownership transfer** adds significant boilerplate and is easily bypassed.
- **`__slots__` with `__del__` ownership tracking** is fragile against exception paths.

The cleanest path to compile-time guarantees in Python would be an external static analyzer (e.g., a `mypy` plugin) — identified as future work.

## Companion repositories

- [token-budgets](https://github.com/sajjadanwar0/token-budgets) — main Rust library (canonical implementation)
- [token-budgets-formals](https://github.com/sajjadanwar0/token-budgets-formals) — 4-tier formal verification
- [token-budgets-experiments](https://github.com/sajjadanwar0/token-budgets-experiments) — multi-runtime evaluation

## License

[Add license. Apache-2.0 OR MIT recommended.]