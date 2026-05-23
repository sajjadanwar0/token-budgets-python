"""Direct double-spend without rebinding — must be flagged."""
from token_budgets import Budget

def double_spend_violation() -> None:
    b = Budget(initial_uc=1000, max_uc=10_000)
    b.spend(400)
    b.spend(300)  # double-use — should fail mypy
