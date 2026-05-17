"""
case_04_smag_004.py — Reproduction of SMAG-004 with Token Budgets fix.

Catalog entry: SMAG-004 (smolagents, #1129, 2025-04)
Title: "Is there a step timeout feature?"
Cluster: missing-per-step-bound
Failure pattern: smolagents had no per-step cost or time bound. A single
agent step could spawn a runaway sub-task (tool call returning massive
output, model generation hitting max_tokens). Token Budgets fix:
budget.split(per_step_cap) creates a child budget; the step runs against
the child; merge restores residual to parent.

Vulnerable: no per-step budget
Protected: split/merge isolates step cost
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from token_budgets import Budget, BudgetExhausted


def simulate_step(actual_cost_uc: int, allocated_budget_uc: int) -> int:
    """Simulates one agent step. Returns ACTUAL cost charged.
    If actual_cost > allocated, the budget rejects further charges
    (representing the per-step cap kicking in)."""
    spent = 0
    chunk = 10  # 10-uc chunks (model streams)
    while spent + chunk <= actual_cost_uc and spent + chunk <= allocated_budget_uc:
        spent += chunk
    return spent


# ----------------------------------------------------------------------
# Vulnerable: no per-step bound (SMAG-004 pattern)
# ----------------------------------------------------------------------
def agent_no_step_cap(total_cap_uc, step_actual_costs):
    """Steps consume aggregate budget but no per-step cap."""
    spent = 0
    for step_cost in step_actual_costs:
        # No per-step cap; runaway step absorbs all remaining budget
        actual = simulate_step(step_cost, allocated_budget_uc=10**9)
        spent += actual
    return spent


# ----------------------------------------------------------------------
# Protected: split/merge per-step
# ----------------------------------------------------------------------
def agent_per_step_capped(total_cap_uc, step_actual_costs, per_step_cap_uc):
    """Each step gets its own child budget via split; merge returns residual."""
    budget = Budget(initial_uc=total_cap_uc, max_uc=total_cap_uc * 10)
    total_spent = 0
    for step_cost in step_actual_costs:
        if budget.micro_cents() < per_step_cap_uc:
            # Not enough left for a full per-step allocation
            break
        # Split off a child budget for this step
        child, budget = budget.split(per_step_cap_uc)
        # Step runs against child; can only spend up to per_step_cap_uc
        actual = simulate_step(step_cost, allocated_budget_uc=child.micro_cents())
        total_spent += actual
        # Refund unused child capacity by merging back
        if actual < child.micro_cents():
            refund_uc = child.micro_cents() - actual
            child_remainder = Budget(initial_uc=refund_uc,
                                     max_uc=child.max_uc)
            budget = budget.merge_with(child_remainder)
        # child (or its remnant) is now consumed
    return total_spent


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_no_step_cap_lets_runaway_consume_all():
    """SMAG-004 pattern: one runaway step burns through the entire budget."""
    # Normal steps cost 50 uc; one runaway costs 5000 uc
    step_costs = [50, 50, 5000, 50]  # step 3 is a runaway
    cap = 200
    spent = agent_no_step_cap(cap, step_costs)
    # The runaway step alone exceeds cap
    assert spent > cap, f"expected runaway to overshoot but spent={spent}"
    print(f"  No per-step cap: 4 steps spent {spent} uc (cap was {cap}) "
          f"-> runaway absorbed {spent} uc [SMAG-004 reproduced]")


def test_split_merge_isolates_runaway_cost():
    """Token Budgets fix: per-step split bounds the runaway."""
    step_costs = [50, 50, 5000, 50]
    cap = 200
    per_step_cap = 60  # generous for normal steps, bounds runaway
    spent = agent_per_step_capped(cap, step_costs, per_step_cap)
    # The runaway step is bounded to per_step_cap=60 instead of 5000
    # Each normal step uses ~50, runaway uses 60, then more normal steps
    assert spent <= cap, f"spent {spent} should be within cap {cap}"
    print(f"  Per-step capped: 4 steps spent {spent}/{cap} uc, "
          f"runaway bounded to {per_step_cap} uc [step-isolation works]")


def test_split_merge_normal_workload_unaffected():
    """When steps are normal-sized, per-step split doesn't hurt utilization."""
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