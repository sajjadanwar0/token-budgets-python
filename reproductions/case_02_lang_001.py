import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from token_budgets import Budget, BudgetExhausted

def cost_for_iteration(iteration_index: int) -> int:
    return 20 + 15 * iteration_index

def agent_recursion_limit_only(recursion_limit, dollar_cap_uc):
    iterations = 0
    spent = 0

    while iterations < recursion_limit:
        cost = cost_for_iteration(iterations)
        spent += cost
        iterations += 1

    return iterations, spent

def agent_dollar_bounded(recursion_limit, dollar_cap_uc):
    budget = Budget(initial_uc=dollar_cap_uc, max_uc=dollar_cap_uc * 10)
    iterations = 0

    while iterations < recursion_limit:
        cost = cost_for_iteration(iterations)
        budget = budget.spend(cost)  # raises BudgetExhausted on overshoot
        iterations += 1
    return iterations, dollar_cap_uc - budget.micro_cents()

def test_recursion_only_overshoots_dollar_cap():
    iterations, spent = agent_recursion_limit_only(recursion_limit=25, dollar_cap_uc=200)
    expected = sum(cost_for_iteration(i) for i in range(25))
    assert spent == expected
    assert spent > 200, f"spent {spent} did not overshoot 200 — increase iterations"

    print(f"  Recursion-only: 25 iters, {spent} uc spent (intended cap 200 uc) "
          f"-> OVERSHOOT by {spent - 200} uc [LANG-001 reproduced]")


def test_dollar_bounded_stops_at_cap():
    try:
        agent_dollar_bounded(recursion_limit=25, dollar_cap_uc=200)
        assert False, "expected BudgetExhausted"
    except BudgetExhausted:
        print(f"  Dollar-bounded: cap fired before recursion exhausted "
              f"[cap-respecting under same workload]")


def test_dollar_bounded_loose_cap_completes():
    iters, spent = agent_dollar_bounded(recursion_limit=5, dollar_cap_uc=500)
    expected = sum(cost_for_iteration(i) for i in range(5))
    assert spent == expected
    assert iters == 5
    print(f"  Dollar-bounded with generous cap: {iters} iters, {spent}/500 uc "
          f"[no false rejection]")


if __name__ == "__main__":
    print("=" * 60)
    print("LANG-001: Recursion limit fails to bound dollar cost")
    print("=" * 60)
    test_recursion_only_overshoots_dollar_cap()
    test_dollar_bounded_stops_at_cap()
    test_dollar_bounded_loose_cap_completes()
    print("All LANG-001 reproduction tests passed.")