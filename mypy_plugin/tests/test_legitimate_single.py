"""A single spend on a fresh Budget is legitimate."""
from token_budgets import Budget

def single_spend() -> None:
    b = Budget(initial_uc=1000, max_uc=10_000)
    b.spend(400)
