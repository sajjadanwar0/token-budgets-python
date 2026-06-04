import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from token_budgets import Budget, BudgetExhausted


def simulate_step(actual_cost_uc: int, allocated_budget_uc: int) -> int:
    spent = 0
    chunk = 10

    while spent + chunk <= actual_cost_uc and spent + chunk <= allocated_budget_uc:
        spent += chunk

    return spent

def agent_no_step_cap(total_cap_uc, step_actual_costs):
    spent = 0

    for step_cost in step_actual_costs:
        actual = simulate_step(step_cost, allocated_budget_uc=10**9)
        spent += actual

    return spent

def agent_per_step_capped(total_cap_uc, step_actual_costs, per_step_cap_uc):
    budget = Budget(initial_uc=total_cap_uc, max_uc=total_cap_uc * 10)
    total_spent = 0

    for step_cost in step_actual_costs:
        if budget.micro_cents() < per_step_cap_uc:
            break

        child, budget = budget.split(per_step_cap_uc)

        actual = simulate_step(step_cost, allocated_budget_uc=child.micro_cents())

        total_spent += actual

        if actual < child.micro_cents():
            refund_uc = child.micro_cents() - actual
            child_remainder = Budget(initial_uc=refund_uc,
                                     max_uc=child.max_uc)
            budget = budget.merge_with(child_remainder)

    return total_spent


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_no_step_cap_lets_runaway_consume_all():
    step_costs = [50, 50, 5000, 50]
    cap = 200
    spent = agent_no_step_cap(cap, step_costs)
    assert spent > cap, f"expected runaway to overshoot but spent={spent}"
    print(f"  No per-step cap: 4 steps spent {spent} uc (cap was {cap}) "
          f"-> runaway absorbed {spent} uc [SMAG-004 reproduced]")

def test_split_merge_isolates_runaway_cost():
    step_costs = [50, 50, 5000, 50]
    cap = 200
    per_step_cap = 60
    spent = agent_per_step_capped(cap, step_costs, per_step_cap)

    assert spent <= cap, f"spent {spent} should be within cap {cap}"
    print(f"  Per-step capped: 4 steps spent {spent}/{cap} uc, "
          f"runaway bounded to {per_step_cap} uc [step-isolation works]")

def test_split_merge_normal_workload_unaffected():
    step_costs = [40, 40, 40, 40]  # all normal
    cap = 250
    per_step_cap = 60
    spent = agent_per_step_capped(cap, step_costs, per_step_cap)
    assert spent == sum(step_costs)

    print(f"  Normal workload with per-step cap: spent {spent}/{cap} uc "
          f"[no false rejection on well-behaved steps]")

if __name__ == "__main__":
    print("=" * 60)
    print("SMAG-004: Missing per-step timeout / cost bound")
    print("=" * 60)
    test_no_step_cap_lets_runaway_consume_all()
    test_split_merge_isolates_runaway_cost()
    test_split_merge_normal_workload_unaffected()
    print("All SMAG-004 reproduction tests passed.")