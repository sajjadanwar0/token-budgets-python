from token_budgets import Budget

def iter_legitimate() -> None:
    b1 = Budget(initial_uc=100, max_uc=10_000)
    b2 = Budget(initial_uc=200, max_uc=10_000)
    b1.spend(50)
    b2.spend(75)
