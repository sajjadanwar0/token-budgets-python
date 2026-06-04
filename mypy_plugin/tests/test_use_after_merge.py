from token_budgets import Budget

def use_after_merge_violation() -> None:
    a = Budget(initial_uc=400, max_uc=10_000)
    c = Budget(initial_uc=600, max_uc=10_000)
    d = a.merge_with(c)
    c.spend(100)
