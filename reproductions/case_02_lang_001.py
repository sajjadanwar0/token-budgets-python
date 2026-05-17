"""
case_02_lang_001.py — Reproduction of LANG-001 with Token Budgets fix.

Catalog entry: LANG-001 (langgraph, #6731, 2026-01)
Title: "Agent infinite looping until recursion limit"
Cluster: recursion-limit-only-protection
Failure pattern: LangGraph's recursion_limit bounds the number of agent
loop iterations but says nothing about per-iteration cost. An agent that
makes very expensive calls each iteration can blow through a dollar cap
even at low recursion depth.

The catalog reproduction notes: "text-to-SQL retry; reproduced in
Section 5.3". The retry loop in the agent makes 1-N model calls per
attempt, and each retry can have different per-call cost. recursion_limit
constrains LOOP COUNT but not DOLLAR COST.

Vulnerable: recursion_limit alone
Protected: Budget enforces dollar cap regardless of loop count
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from token_budgets import Budget, BudgetExhausted


def cost_for_iteration(iteration_index: int) -> int:
    """Cost grows with iteration (sql retry adds context each time)."""
    return 20 + 15 * iteration_index  # 20, 35, 50, 65, ... uc


# ----------------------------------------------------------------------
# Vulnerable: LangGraph-style recursion_limit only
# ----------------------------------------------------------------------
def agent_recursion_limit_only(recursion_limit, dollar_cap_uc):
    """Mimics LANG-001: bounds count but not dollars."""
    iterations = 0
    spent = 0
    while iterations < recursion_limit:
        cost = cost_for_iteration(iterations)
        spent += cost
        iterations += 1
        # No dollar check — just count
    return iterations, spent


# ----------------------------------------------------------------------
# Protected: Token Budgets dollar cap
# ----------------------------------------------------------------------
def agent_dollar_bounded(recursion_limit, dollar_cap_uc):
    """Token Budgets fix: dollar cap enforced at every iteration."""
    budget = Budget(initial_uc=dollar_cap_uc, max_uc=dollar_cap_uc * 10)
    iterations = 0
    while iterations < recursion_limit:
        cost = cost_for_iteration(iterations)
        budget = budget.spend(cost)  # raises BudgetExhausted on overshoot
        iterations += 1
    return iterations, dollar_cap_uc - budget.micro_cents()


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_recursion_only_overshoots_dollar_cap():
    """With recursion_limit=25 (LangGraph default) and growing per-iter cost,
    spend can blow past a $0.01 cap even though loop count is bounded."""
    iterations, spent = agent_recursion_limit_only(recursion_limit=25, dollar_cap_uc=200)
    # Sum of 20, 35, 50, ... for 25 iterations:
    expected = sum(cost_for_iteration(i) for i in range(25))
    assert spent == expected
    assert spent > 200, f"spent {spent} did not overshoot 200 — increase iterations"
    print(f"  Recursion-only: 25 iters, {spent} uc spent (intended cap 200 uc) "
          f"-> OVERSHOOT by {spent - 200} uc [LANG-001 reproduced]")


def test_dollar_bounded_stops_at_cap():
    """Token Budgets enforces the dollar cap regardless of recursion budget."""
    try:
        agent_dollar_bounded(recursion_limit=25, dollar_cap_uc=200)
        assert False, "expected BudgetExhausted"
    except BudgetExhausted:
        print(f"  Dollar-bounded: cap fired before recursion exhausted "
              f"[cap-respecting under same workload]")


def test_dollar_bounded_loose_cap_completes():
    """When the dollar cap is generous, the protected loop completes normally."""
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