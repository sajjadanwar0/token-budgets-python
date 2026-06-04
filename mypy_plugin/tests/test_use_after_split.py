from token_budgets import Budget

def use_after_split_violation() -> None:
    b = Budget(initial_uc=1000, max_uc=10_000)
    b.spend(100)
