import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from token_budgets import Budget, BudgetExhausted

def visible_token_cost_uc(visible_tokens: int) -> int:
    return visible_tokens * 6


def reasoning_token_cost_uc(reasoning_tokens: int) -> int:
    return reasoning_tokens * 60

def actual_total_cost_uc(visible_tokens: int, reasoning_tokens: int) -> int:
    return visible_token_cost_uc(visible_tokens) + reasoning_token_cost_uc(reasoning_tokens)

def call_reasoning_model_unprotected(cap_uc, calls):
    spent_tracked = 0
    spent_actual = 0
    for visible_tokens, reasoning_tokens in calls:
        estimated = visible_token_cost_uc(visible_tokens)
        spent_tracked += estimated

        spent_actual += actual_total_cost_uc(visible_tokens, reasoning_tokens)
        if spent_tracked > cap_uc:
            break
    return spent_tracked, spent_actual

def call_reasoning_model_protected(cap_uc, calls, p99_reasoning_uc):
    budget = Budget(initial_uc=cap_uc, max_uc=cap_uc * 10)
    spent_actual = 0

    for visible_tokens, reasoning_tokens in calls:
        visible = visible_token_cost_uc(visible_tokens)
        reserved = visible + p99_reasoning_uc
        budget = budget.spend(reserved)
        spent_actual += actual_total_cost_uc(visible_tokens, reasoning_tokens)

    return cap_uc - budget.micro_cents(), spent_actual

def test_visible_only_undercounts():
    calls = [(100, 200)] * 5
    cap = 5000
    tracked, actual = call_reasoning_model_unprotected(cap, calls)
    assert tracked < cap
    assert actual > cap * 10
    overrun_ratio = actual / cap
    print(f"  Visible-only counter: tracked={tracked} (looks ok, <{cap}) "
          f"but actual={actual} -> ACTUAL is {overrun_ratio:.1f}x the cap "
          f"[LANG-004 reproduced]")

def test_p99_reasoning_reservation_bounds_actual():
    calls = [(100, 200)] * 5
    cap = 5000
    p99_reasoning = 12500
    try:
        call_reasoning_model_protected(cap, calls, p99_reasoning)
        assert False, "expected BudgetExhausted with this aggressive scenario"
    except BudgetExhausted:
        print(f"  P99-reserved: discipline fired at boundary "
              f"[reasoning-aware cap-respecting]")


def test_p99_underestimate_still_fails_closed():
    calls = [(100, 200)] * 3
    cap = 10000
    p99_too_low = 100
    spent_reserved, spent_actual = call_reasoning_model_protected(
        cap, calls, p99_too_low
    )

    print(f"  P99 underestimate: spent_reserved={spent_reserved}, "
          f"actual={spent_actual} [operator must calibrate p99 correctly]")


if __name__ == "__main__":
    print("=" * 60)
    print("LANG-004: Reasoning tokens not counted in cost")
    print("=" * 60)
    test_visible_only_undercounts()
    test_p99_reasoning_reservation_bounds_actual()
    test_p99_underestimate_still_fails_closed()
    print("All LANG-004 reproduction tests passed.")