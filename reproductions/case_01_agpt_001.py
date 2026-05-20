import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from token_budgets import Budget, BudgetExhausted, AffineViolation


def estimated_cost_uc(step: str) -> int:
    """Mock cost estimator: 1 micro-cent per byte (conservative byte-length)."""
    return len(step)

def run_agent_unprotected(plan_steps, dollar_cap_uc):
    """Per-step cost tracking but NO aggregate enforcement."""
    spent = 0
    for step in plan_steps:
        cost = estimated_cost_uc(step)
        spent += cost
        # No cap check — runs to completion of plan_steps regardless of total
    return spent


# ----------------------------------------------------------------------
# Protected: Token Budgets discipline
# ----------------------------------------------------------------------
def run_agent_protected(plan_steps, cap_uc):
    """Budget is checked at every step; raises BudgetExhausted on overshoot."""
    budget = Budget(initial_uc=cap_uc, max_uc=cap_uc * 10)
    for step in plan_steps:
        cost = estimated_cost_uc(step)
        budget = budget.spend(cost)  # consumes self; returns smaller Budget
    return cap_uc - budget.micro_cents()


# ----------------------------------------------------------------------
# Test cases demonstrating AGPT-001 prevention
# ----------------------------------------------------------------------
def test_unprotected_blows_past_cap():
    """The original AGPT-001 vulnerability: total spend exceeds intended cap."""
    plan_steps = [
        "Generate analysis report part 1...",  # 38 uc
        "Generate analysis report part 2...",  # 38 uc
        "Generate analysis report part 3...",  # 38 uc
        "Generate analysis report part 4...",  # 38 uc
        "Generate analysis report part 5...",  # 38 uc
        "Generate analysis report part 6...",  # 38 uc
    ]
    dollar_cap_uc = 100  # intended cap
    total = run_agent_unprotected(plan_steps, dollar_cap_uc)
    assert total > dollar_cap_uc, "expected overshoot — that's the bug"
    print(f"  Unprotected: spent {total} uc (cap was {dollar_cap_uc} uc) "
          f"-> OVERSHOOT by {total - dollar_cap_uc} uc [AGPT-001 reproduced]")


def test_protected_fails_closed_at_cap():
    """The Token Budgets fix: total spend bounded by cap, BudgetExhausted raised."""
    plan_steps = [
        "Generate analysis report part 1...",
        "Generate analysis report part 2...",
        "Generate analysis report part 3...",
        "Generate analysis report part 4...",
        "Generate analysis report part 5...",
        "Generate analysis report part 6...",
    ]
    cap_uc = 100
    try:
        run_agent_protected(plan_steps, cap_uc)
        assert False, "expected BudgetExhausted but none raised"
    except BudgetExhausted as e:
        print(f"  Protected: discipline fired -> {e} [cap-respecting]")


def test_protected_completes_under_cap():
    """When cap is generous, the protected version completes normally."""
    plan_steps = ["short step"] * 3  # ~33 uc total
    cap_uc = 1000
    spent = run_agent_protected(plan_steps, cap_uc)
    assert spent < cap_uc
    print(f"  Protected with generous cap: spent {spent}/{cap_uc} uc [no false rejection]")


if __name__ == "__main__":
    print("=" * 60)
    print("AGPT-001: Auto-GPT lacks aggregate budget awareness")
    print("=" * 60)
    test_unprotected_blows_past_cap()
    test_protected_fails_closed_at_cap()
    test_protected_completes_under_cap()
    print("All AGPT-001 reproduction tests passed.")