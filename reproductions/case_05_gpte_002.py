import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reproductions.token_budgets import (BudgetPool, BudgetExhausted)

def prompt_cost_uc(turn_index: int, base_spec_uc: int, growth_per_turn_uc: int) -> int:
    return base_spec_uc + growth_per_turn_uc * turn_index

def gpte_unprotected(total_cap_uc, num_turns, base_spec, growth):
    spent = 0

    for turn in range(num_turns):
        cost = prompt_cost_uc(turn, base_spec, growth)
        spent += cost

    return spent

def gpte_protected(total_cap_uc, num_turns, base_spec, growth,
                   max_per_turn_uc):
    pool = BudgetPool(available_uc=total_cap_uc)
    spent = 0
    refused_turns = 0

    for turn in range(num_turns):
        cost = prompt_cost_uc(turn, base_spec, growth)
        if cost > max_per_turn_uc:
            refused_turns += 1
            continue
        try:
            def run_turn(receipt):
                return receipt.commit(cost, cost)

            cost_committed = pool.with_reservation(cost, run_turn)

            spent += cost_committed

        except BudgetExhausted:
            refused_turns += 1
            break

    return spent, refused_turns

def test_unprotected_grows_unbounded():
    spent = gpte_unprotected(total_cap_uc=500, num_turns=10,
                             base_spec=100, growth=50)
    assert spent == 3250
    assert spent > 500, "expected overshoot of intended 500-uc cap"
    print(f"  Unprotected: 10 turns with growing context spent {spent} uc "
          f"(intended cap 500) -> OVERSHOOT by {spent - 500} uc "
          f"[GPTE-002 reproduced]")

def test_protected_refuses_oversized_turns():
    spent, refused = gpte_protected(
        total_cap_uc=500, num_turns=10,
        base_spec=100, growth=50, max_per_turn_uc=200,
    )
    assert refused >= 7, f"expected >=7 refusals, got {refused}"
    assert spent <= 500
    print(f"  Per-turn cap: spent {spent}/500 uc, {refused} turns refused "
          f"as oversized [per-turn cap-respecting]")

def test_protected_with_generous_per_turn_completes_until_pool_exhausted():
    spent, refused = gpte_protected(
        total_cap_uc=500, num_turns=10,
        base_spec=100, growth=50, max_per_turn_uc=10_000,
    )

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