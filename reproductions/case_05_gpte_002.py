"""
case_05_gpte_002.py — Reproduction of GPTE-002 with Token Budgets fix.

Catalog entry: GPTE-002 (gpt-engineer, #114, 2023-06)
Title: "Specifications repeating, consuming many tokens"
Cluster: context-growth-unbounded
Failure pattern: gpt-engineer was repeatedly including the full
specification in each follow-up prompt, causing context size (and per-call
cost) to grow without bound. The framework had no per-call cap to limit
prompt size or signal when the cumulative context cost exceeded a budget.

Vulnerable: prompt grows each turn, no per-turn cap
Protected: pre-flight pool reservation; the pool refuses oversized turns
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reproductions.token_budgets import (
    Budget, BudgetPool, BudgetExhausted, AffineViolation,
)
# WithReservation pattern requires the typestate API in pool form; we mimic
# the closure pattern with BudgetPool.with_reservation (already in port).


def prompt_cost_uc(turn_index: int, base_spec_uc: int, growth_per_turn_uc: int) -> int:
    """Per-turn cost: spec size + accumulated context."""
    return base_spec_uc + growth_per_turn_uc * turn_index


# ----------------------------------------------------------------------
# Vulnerable: GPTE-002 pattern — context grows, no per-turn cap
# ----------------------------------------------------------------------
def gpte_unprotected(total_cap_uc, num_turns, base_spec, growth):
    """Each turn includes growing spec; no per-turn check."""
    spent = 0
    for turn in range(num_turns):
        cost = prompt_cost_uc(turn, base_spec, growth)
        spent += cost
    return spent


# ----------------------------------------------------------------------
# Protected: BudgetPool with closure-based per-turn reservation
# ----------------------------------------------------------------------
def gpte_protected(total_cap_uc, num_turns, base_spec, growth,
                   max_per_turn_uc):
    """Each turn reserves against a pool; if a turn exceeds the per-turn
    cap, with_reservation returns BudgetExhausted (reservation refused)."""
    pool = BudgetPool(available_uc=total_cap_uc)
    spent = 0
    refused_turns = 0
    for turn in range(num_turns):
        cost = prompt_cost_uc(turn, base_spec, growth)
        if cost > max_per_turn_uc:
            # Pre-flight refusal: prompt is too large
            refused_turns += 1
            continue
        try:
            def run_turn(receipt):
                # Simulate actually executing the turn at `cost`
                return receipt.commit(cost, cost)
            cost_committed = pool.with_reservation(cost, run_turn)
            spent += cost_committed
        except BudgetExhausted:
            refused_turns += 1
            break  # pool exhausted
    return spent, refused_turns


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_unprotected_grows_unbounded():
    """GPTE-002 pattern: 10 turns with growing prompt = unbounded total spend."""
    # Base spec is 100 uc; growth is 50 uc per turn (spec is re-included)
    # Turn 0: 100, Turn 1: 150, ..., Turn 9: 550
    # Total: 10 * 100 + 50 * (0+1+...+9) = 1000 + 2250 = 3250
    spent = gpte_unprotected(total_cap_uc=500, num_turns=10,
                             base_spec=100, growth=50)
    assert spent == 3250
    assert spent > 500, "expected overshoot of intended 500-uc cap"
    print(f"  Unprotected: 10 turns with growing context spent {spent} uc "
          f"(intended cap 500) -> OVERSHOOT by {spent - 500} uc "
          f"[GPTE-002 reproduced]")


def test_protected_refuses_oversized_turns():
    """Per-turn cap refuses turns whose prompt exceeds max_per_turn_uc."""
    spent, refused = gpte_protected(
        total_cap_uc=500, num_turns=10,
        base_spec=100, growth=50, max_per_turn_uc=200,
    )
    # Turns 0-1 fit (cost 100, 150). Turn 2 cost 200 fits exactly.
    # Turn 3 cost 250 > 200 → refused. Continue refusing thereafter.
    # Expected: turns 0,1,2 committed (100+150+200=450), turns 3-9 refused (7 turns)
    assert refused >= 7, f"expected >=7 refusals, got {refused}"
    assert spent <= 500
    print(f"  Per-turn cap: spent {spent}/500 uc, {refused} turns refused "
          f"as oversized [per-turn cap-respecting]")


def test_protected_with_generous_per_turn_completes_until_pool_exhausted():
    """When per-turn cap is generous, the pool itself bounds total."""
    spent, refused = gpte_protected(
        total_cap_uc=500, num_turns=10,
        base_spec=100, growth=50, max_per_turn_uc=10_000,
    )
    # Per-turn doesn't restrict; pool runs out at turn 3:
    # Turn 0: 100, Turn 1: 150, Turn 2: 200, Turn 3: 250 → cumulative 700 > 500
    # So turn 3's reservation is REFUSED (BudgetExhausted from pool)
    # Spent: 100 + 150 + 200 = 450
    assert spent <= 500
    assert refused >= 1
    print(f"  Pool-bounded only: spent {spent}/500 uc, {refused} turn(s) refused "
          f"by pool exhaustion [aggregate cap-respecting]")


if __name__ == "__main__":
    print("=" * 60)
    print("GPTE-002: Specifications repeating, context growth unbounded")
    print("=" * 60)
    test_unprotected_grows_unbounded()
    test_protected_refuses_oversized_turns()
    test_protected_with_generous_per_turn_completes_until_pool_exhausted()
    print("All GPTE-002 reproduction tests passed.")