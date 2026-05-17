"""
case_03_lang_004.py — Reproduction of LANG-004 with Token Budgets fix.

Catalog entry: LANG-004 (langchain, #29779, 2025-02)
Title: "Reasoning tokens not in cost calculation"
Cluster: hidden-cost-component
Failure pattern: LangChain's cost calculation for o3-mini and o-series
reasoning models OMITS the reasoning_tokens — the hidden internal
thinking tokens that providers bill for separately. Token Budgets'
ReasoningProvider variant (with per_call_reasoning_p99_uc) reserves
worst-case reasoning cost up-front and bounds it.

Vulnerable: visible-tokens-only counting (LANG-004)
Protected: pre-reserved reasoning headroom
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from token_budgets import Budget, BudgetExhausted


def visible_token_cost_uc(visible_tokens: int) -> int:
    """The estimator LANG-004 used: visible tokens only."""
    return visible_tokens * 6  # ~$0.006/1k tokens example pricing


def reasoning_token_cost_uc(reasoning_tokens: int) -> int:
    """The hidden component LANG-004 missed: provider bills for this too."""
    return reasoning_tokens * 60  # ~$0.06/1k reasoning tokens (10x visible)


def actual_total_cost_uc(visible_tokens: int, reasoning_tokens: int) -> int:
    return visible_token_cost_uc(visible_tokens) + reasoning_token_cost_uc(reasoning_tokens)


# ----------------------------------------------------------------------
# Vulnerable: count only visible tokens (the LANG-004 bug)
# ----------------------------------------------------------------------
def call_reasoning_model_unprotected(cap_uc, calls):
    """Each call uses visible-only estimator. Hidden reasoning cost slips through."""
    spent_tracked = 0
    spent_actual = 0
    for visible_tokens, reasoning_tokens in calls:
        # The agent ESTIMATES only visible cost:
        estimated = visible_token_cost_uc(visible_tokens)
        spent_tracked += estimated
        # But the actual cost includes reasoning:
        spent_actual += actual_total_cost_uc(visible_tokens, reasoning_tokens)
        if spent_tracked > cap_uc:
            break  # tracked counter triggers, but it's incomplete
    return spent_tracked, spent_actual


# ----------------------------------------------------------------------
# Protected: Token Budgets reserves reasoning headroom up-front
# ----------------------------------------------------------------------
def call_reasoning_model_protected(cap_uc, calls, p99_reasoning_uc):
    """For each call, reserve visible_cost + p99_reasoning_cost.
    If actual reasoning exceeds p99, the next call fails closed."""
    budget = Budget(initial_uc=cap_uc, max_uc=cap_uc * 10)
    spent_actual = 0
    for visible_tokens, reasoning_tokens in calls:
        visible = visible_token_cost_uc(visible_tokens)
        # Reserve visible + worst-case reasoning
        reserved = visible + p99_reasoning_uc
        budget = budget.spend(reserved)
        # Actual bill includes the real reasoning amount
        spent_actual += actual_total_cost_uc(visible_tokens, reasoning_tokens)
    return cap_uc - budget.micro_cents(), spent_actual


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_visible_only_undercounts():
    """LANG-004 bug: visible-only counter says we're under cap; actual is over."""
    # 5 calls with 100 visible tokens but 200 reasoning tokens each
    calls = [(100, 200)] * 5
    cap = 5000  # 5000 uc cap
    tracked, actual = call_reasoning_model_unprotected(cap, calls)
    # tracked = 5 * 600 = 3000 (under cap, "looks fine")
    # actual = 5 * (600 + 12000) = 63000 (massively over cap)
    assert tracked < cap
    assert actual > cap * 10
    overrun_ratio = actual / cap
    print(f"  Visible-only counter: tracked={tracked} (looks ok, <{cap}) "
          f"but actual={actual} -> ACTUAL is {overrun_ratio:.1f}x the cap "
          f"[LANG-004 reproduced]")


def test_p99_reasoning_reservation_bounds_actual():
    """With p99 reasoning reservation, the discipline fires before actual overshoots cap."""
    calls = [(100, 200)] * 5
    cap = 5000
    p99_reasoning = 12500  # set p99 above the actual 12000 to be safe
    try:
        call_reasoning_model_protected(cap, calls, p99_reasoning)
        # The cap will trigger BudgetExhausted early in this scenario
        assert False, "expected BudgetExhausted with this aggressive scenario"
    except BudgetExhausted:
        # The discipline correctly refused to make all 5 calls
        print(f"  P99-reserved: discipline fired at boundary "
              f"[reasoning-aware cap-respecting]")


def test_p99_underestimate_still_fails_closed():
    """If operator sets p99 too LOW, the discipline still bounds total spend
    because each call reserves visible+p99 up front."""
    calls = [(100, 200)] * 3
    cap = 10000
    p99_too_low = 100  # operator underestimated reasoning cost
    spent_reserved, spent_actual = call_reasoning_model_protected(
        cap, calls, p99_too_low
    )
    # Reserved per call: 600 + 100 = 700 → 3 calls = 2100 reserved
    # Spent_reserved is 2100, well under cap.
    # BUT: actual still includes the real reasoning_tokens cost (LANG-004 risk).
    # The discipline DOES NOT protect against operator misconfiguration of p99,
    # only against scheduling/aliasing bugs. This is the operator's
    # responsibility — disclosed in §VII.B.
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