"""
langchain_real_validation_v2.py — Real-SDK validation of Token Budgets.

v2 calibration changes from v1 (smoke test showed S3/S4 didn't fire):
  S3: replaced "small + runaway + small" with "oversized-first-call".
      cap_factor 1.5 -> 0.4; runaway expanded to ~3300 chars.
      Expected: pre_flight_refused at call 0.
  S4: replaced trivial "Turn N: reply ok" with explicit growing-context
      prompts (each turn adds ~5 to ~60 'background-context' tokens).
      cap_factor 1.2 -> 0.4. Expected: mid_loop_fired on turn 3-5.

All other components unchanged from v1:
  - Real langchain_anthropic.ChatAnthropic and langchain_openai.ChatOpenAI
  - Affine pre-flight pattern (Budget.spend before invoke)
  - Passive TokenUsageRecorder as BaseCallbackHandler
"""

import sys, os, json, csv, argparse, time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from token_budgets import Budget, BudgetExhausted

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI


class TokenUsageRecorder(BaseCallbackHandler):
    """Records real-API token usage from LangChain's LLMResult."""
    def __init__(self):
        self.last_input = 0
        self.last_output = 0

    def on_llm_end(self, response, **kwargs):
        in_t, out_t = 0, 0
        try:
            msg = response.generations[0][0].message
            md = getattr(msg, "usage_metadata", None) or {}
            if md:
                in_t = md.get("input_tokens", 0)
                out_t = md.get("output_tokens", 0)
        except (AttributeError, IndexError):
            pass
        if not (in_t or out_t):
            try:
                usage = (response.llm_output or {}).get("token_usage", {}) or {}
                in_t = usage.get("prompt_tokens", 0) or in_t
                out_t = usage.get("completion_tokens", 0) or out_t
            except (AttributeError, TypeError):
                pass
        self.last_input = in_t
        self.last_output = out_t


@dataclass
class ProviderConfig:
    name: str
    model_id: str
    cost_in_uc: int
    cost_out_uc: int
    chat_cls: Any

PROVIDERS = {
    "anthropic": ProviderConfig(
        name="anthropic",
        model_id="claude-haiku-4-5",
        cost_in_uc=80,
        cost_out_uc=400,
        chat_cls=ChatAnthropic,
    ),
    "openai": ProviderConfig(
        name="openai",
        model_id="gpt-4o-mini-2024-07-18",
        cost_in_uc=15,
        cost_out_uc=60,
        chat_cls=ChatOpenAI,
    ),
}


def estimate_call_uc(prompt: str, cost_in_uc: int, max_tokens: int,
                     cost_out_uc: int, margin: float = 1.5) -> int:
    n_bytes = len(prompt.encode("utf-8"))
    input_tokens_est = max(1, int(n_bytes / 4 * margin))
    return input_tokens_est * cost_in_uc + max_tokens * cost_out_uc


@dataclass
class StepRecord:
    estimate_uc: int = 0
    actual_uc: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


def run_agent_step(llm, prompt: str, history: List, budget: Budget,
                   provider: ProviderConfig, max_tokens: int
                   ) -> Tuple[Budget, Optional[Any], StepRecord]:
    estimate_uc = estimate_call_uc(
        prompt, provider.cost_in_uc, max_tokens, provider.cost_out_uc
    )
    new_budget = budget.spend(estimate_uc)  # affine pre-flight

    recorder = TokenUsageRecorder()
    messages = history + [HumanMessage(content=prompt)]
    response = llm.invoke(messages, config={"callbacks": [recorder]})

    actual_uc = (recorder.last_input * provider.cost_in_uc +
                 recorder.last_output * provider.cost_out_uc)

    return new_budget, response, StepRecord(
        estimate_uc=estimate_uc,
        actual_uc=actual_uc,
        input_tokens=recorder.last_input,
        output_tokens=recorder.last_output,
    )


# ============================================================
# CALIBRATED SCENARIOS v2
# ============================================================

SCENARIOS = {
    "S1_steady_within_cap": {
        "description": "4 short calls; generous cap (all complete)",
        "cap_factor": 3.0,
        "max_tokens": 50,
    },
    "S2_aggregate_overshoot": {
        "description": "6 short calls; tight cap; aggregate exceeds (mid_loop_fires)",
        "cap_factor": 0.6,
        "max_tokens": 50,
    },
    "S3_oversized_first_call": {
        "description": "1st call has oversized prompt; cap < single-call estimate (pre_flight_refused at call 0)",
        "cap_factor": 0.4,
        "max_tokens": 50,
    },
    "S4_context_growth": {
        "description": "5 turns with explicit context growth; mid_loop_fires on later turn",
        "cap_factor": 0.4,
        "max_tokens": 50,
    },
}


def build_prompts(scenario_id: str) -> List[str]:
    if scenario_id == "S1_steady_within_cap":
        return [f"Say one short sentence about the number {i}." for i in range(4)]

    elif scenario_id == "S2_aggregate_overshoot":
        return [f"Briefly describe the color {c} in one sentence."
                for c in ["red", "blue", "green", "yellow", "purple", "orange"]]

    elif scenario_id == "S3_oversized_first_call":
        # Single oversized first call (~3500 chars => ~1300 input tokens
        # estimate => with Anthropic rates ~125K uc estimate, which will
        # exceed any cap calibrated to 0.4 x sum-of-estimates).
        # Calls 1 and 2 are tiny - they will not run because pre-flight
        # refuses call 0.
        return [
            ("Write a detailed multi-page analysis of distributed system "
             "architectures and trade-offs, addressing each of these topics: "
             + ", ".join([f"distributed-topic-{i}" for i in range(200)])
             + ". For each topic include examples, common failure modes, "
               "trade-offs, and references to relevant literature."),
            "Hello.",
            "Goodbye.",
        ]

    elif scenario_id == "S4_context_growth":
        # Each turn appends EXPLICIT growing context.
        # Turn 1: ~80 chars; Turn 2: ~280 chars; Turn 3: ~580 chars;
        # Turn 4: ~980 chars; Turn 5: ~1480 chars.
        # Plus accumulated history from prior turns.
        contexts = [5, 15, 25, 40, 60]  # repetitions of "background-context "
        return [
            f"Turn {i+1}: Continue our design discussion. "
            + ("background-context " * contexts[i])
            for i in range(5)
        ]

    return []


def estimate_scenario_total_uc(scenario_id: str, provider: ProviderConfig,
                               max_tokens: int) -> int:
    prompts = build_prompts(scenario_id)
    history_text = ""
    total = 0
    for p in prompts:
        if scenario_id == "S4_context_growth":
            full = history_text + p
            total += estimate_call_uc(full, provider.cost_in_uc, max_tokens,
                                      provider.cost_out_uc)
            history_text += p + " ok\n"  # short simulated reply
        else:
            total += estimate_call_uc(p, provider.cost_in_uc, max_tokens,
                                      provider.cost_out_uc)
    return total


@dataclass
class RunResult:
    provider: str
    model: str
    scenario: str
    rep: int
    cap_uc: int
    outcome: str
    n_calls_attempted: int
    n_calls_completed: int
    total_estimate_uc: int
    total_actual_uc: int
    cap_violations: int
    error_msg: str = ""


def run_scenario(provider: ProviderConfig, scenario_id: str, rep: int,
                 cap_uc: int, verbose: bool = False) -> RunResult:
    sdef = SCENARIOS[scenario_id]
    max_tokens = sdef["max_tokens"]

    llm = provider.chat_cls(
        model=provider.model_id,
        max_tokens=max_tokens,
        temperature=0,
    )

    budget = Budget(initial_uc=cap_uc, max_uc=cap_uc * 100)
    prompts = build_prompts(scenario_id)
    history = []

    n_attempted = 0
    n_completed = 0
    outcome = "completed_within_cap"
    err_msg = ""
    step_records: List[StepRecord] = []

    for i, prompt in enumerate(prompts):
        n_attempted += 1
        try:
            budget, response, step = run_agent_step(
                llm, prompt, history, budget, provider, max_tokens
            )
            n_completed += 1
            step_records.append(step)
            if scenario_id == "S4_context_growth":
                history.append(HumanMessage(content=prompt))
                history.append(AIMessage(content=response.content))
        except BudgetExhausted as e:
            outcome = "pre_flight_refused" if n_completed == 0 else "mid_loop_fired"
            err_msg = str(e)[:200]
            break
        except Exception as e:
            outcome = "error"
            err_msg = f"{type(e).__name__}: {str(e)[:180]}"
            break

    total_actual_uc = sum(s.actual_uc for s in step_records)
    total_estimate_uc = sum(s.estimate_uc for s in step_records)
    cap_violation = 1 if total_actual_uc > cap_uc else 0

    if verbose:
        actual_usd = total_actual_uc / 1e8
        print(f"    [rep {rep}] {scenario_id}: {outcome} "
              f"({n_completed}/{n_attempted} calls, "
              f"actual ${actual_usd:.5f})")

    return RunResult(
        provider=provider.name,
        model=provider.model_id,
        scenario=scenario_id,
        rep=rep,
        cap_uc=cap_uc,
        outcome=outcome,
        n_calls_attempted=n_attempted,
        n_calls_completed=n_completed,
        total_estimate_uc=total_estimate_uc,
        total_actual_uc=total_actual_uc,
        cap_violations=cap_violation,
        error_msg=err_msg,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--providers", nargs="+",
                        default=["anthropic", "openai"],
                        choices=list(PROVIDERS.keys()))
    parser.add_argument("--scenarios", nargs="+",
                        default=list(SCENARIOS.keys()))
    parser.add_argument("--out-csv", default="langchain_real_results_v2.csv")
    parser.add_argument("--out-json", default="langchain_real_summary_v2.json")
    parser.add_argument("--smoke", action="store_true",
                        help="1 rep per cell (~$0.005 quick smoke)")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    if args.smoke:
        args.reps = 1

    for p in args.providers:
        key = "ANTHROPIC_API_KEY" if p == "anthropic" else "OPENAI_API_KEY"
        if not os.environ.get(key):
            print(f"ERROR: {key} environment variable not set", file=sys.stderr)
            sys.exit(1)

    total_runs = len(args.providers) * len(args.scenarios) * args.reps
    print(f"\nLangChain Real-SDK Validation (v2 calibrated)")
    print(f"=" * 60)
    print(f"Providers:  {args.providers}")
    print(f"Scenarios:  {args.scenarios}")
    print(f"Replicas:   {args.reps}")
    print(f"Total runs: {total_runs}")
    print()

    all_results: List[RunResult] = []
    t0 = time.time()

    for p_name in args.providers:
        provider = PROVIDERS[p_name]
        print(f"\n=== Provider: {provider.name} ({provider.model_id}) ===")
        for s in args.scenarios:
            workload = estimate_scenario_total_uc(
                s, provider, SCENARIOS[s]["max_tokens"]
            )
            cap = max(int(workload * SCENARIOS[s]["cap_factor"]), 5000)
            print(f"  {s}: workload estimate {workload} uc; cap {cap} uc")
            for r in range(args.reps):
                result = run_scenario(provider, s, r, cap, verbose=args.verbose)
                all_results.append(result)

    wall = time.time() - t0

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(all_results[0]).keys()))
        writer.writeheader()
        for r in all_results:
            writer.writerow(asdict(r))

    summary: Dict[str, Any] = {}
    for r in all_results:
        cell = f"{r.provider}/{r.scenario}"
        if cell not in summary:
            summary[cell] = {
                "n_total": 0,
                "outcomes": Counter(),
                "total_actual_uc": 0,
                "total_estimate_uc": 0,
                "max_actual_uc": 0,
                "cap_violations": 0,
                "cap_uc": r.cap_uc,
            }
        s = summary[cell]
        s["n_total"] += 1
        s["outcomes"][r.outcome] += 1
        s["total_actual_uc"] += r.total_actual_uc
        s["total_estimate_uc"] += r.total_estimate_uc
        s["max_actual_uc"] = max(s["max_actual_uc"], r.total_actual_uc)
        s["cap_violations"] += r.cap_violations

    for k, v in summary.items():
        v["outcomes"] = dict(v["outcomes"])
        v["mean_actual_uc"] = v["total_actual_uc"] / v["n_total"]
        v["mean_over_reservation_x"] = (
            v["total_estimate_uc"] / v["total_actual_uc"]
            if v["total_actual_uc"] > 0 else None
        )

    total_violations = sum(r.cap_violations for r in all_results)
    total_spent_uc = sum(r.total_actual_uc for r in all_results)
    total_spent_usd = total_spent_uc / 100_000_000.0

    summary["__totals__"] = {
        "n_total": len(all_results),
        "wall_time_sec": wall,
        "total_cap_violations": total_violations,
        "total_spent_uc": total_spent_uc,
        "estimated_total_cost_usd": total_spent_usd,
    }

    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n\n========== SUMMARY ==========")
    print(f"Total runs:           {len(all_results)}")
    print(f"Wall time:            {wall:.1f}s")
    print(f"Total API spend:      ${total_spent_usd:.5f} ({total_spent_uc} uc)")
    print(f"Cap-soundness violations: {total_violations} / {len(all_results)}")
    print()

    for k, v in summary.items():
        if k == "__totals__":
            continue
        out_str = ", ".join(f"{kk}={vv}" for kk, vv in v["outcomes"].items())
        or_str = (f"{v['mean_over_reservation_x']:.2f}x"
                  if v["mean_over_reservation_x"] else "n/a")
        print(f"  {k:<42} cap={v['cap_uc']:6} n={v['n_total']:2} "
              f"mean_actual={v['mean_actual_uc']:7.0f} "
              f"over_res={or_str:7} | {out_str}")

    if total_violations > 0:
        print(f"\n!!! WARNING: {total_violations} cap-soundness violation(s) !!!")
        sys.exit(1)

    print(f"\nResults written to:")
    print(f"  CSV:  {args.out_csv}")
    print(f"  JSON: {args.out_json}")


if __name__ == "__main__":
    main()