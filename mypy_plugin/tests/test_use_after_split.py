"""Use of parent budget after split — must be flagged."""
from token_budgets import Budget

def use_after_split_violation() -> None:
    b = Budget(initial_uc=1000, max_uc=10_000)
    lhs, rhs = b.split(400)
    b.spend(100)  # b was consumed by split — should fail mypy
