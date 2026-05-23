"""Module-global Budget consumed in one function, used in another."""
from token_budgets import Budget

GLOBAL_BUDGET = Budget(initial_uc=1000, max_uc=10_000)

def use_global_first() -> None:
    GLOBAL_BUDGET.spend(100)

def use_global_second() -> None:
    GLOBAL_BUDGET.spend(50)  # should fail mypy (same file)
