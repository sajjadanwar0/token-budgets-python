"""For-loop iteration where each b is a different Budget instance —
the plugin sees each as the same name 'b' but, because tests run
in separate mypy invocations, no state leaks between iterations."""
from token_budgets import Budget

def iter_legitimate() -> None:
    b1 = Budget(initial_uc=100, max_uc=10_000)
    b2 = Budget(initial_uc=200, max_uc=10_000)
    b1.spend(50)
    b2.spend(75)
