import argparse
import csv
import json
import os
import sys
import time
import statistics
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional


HERE = os.path.dirname(os.path.abspath(__file__))
PORT_DIR = os.path.join(HERE, "..", "reproductions")
if os.path.isdir(PORT_DIR):
    sys.path.insert(0, PORT_DIR)

try:
    from token_budgets import Budget, BudgetExhausted
    from token_budgets import LangChainBudgetCallback as _RawBudgetCallback
except ImportError as e:
    print(f"ERROR: could not import token_budgets — {e}", file=sys.stderr)
    print(f"Expected at: {PORT_DIR}/token_budgets.py", file=sys.stderr)
    sys.exit(1)

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError as e:
    print(f"ERROR: langchain_core not installed — {e}", file=sys.stderr)
    sys.exit(1)

class LangChainBudgetCallback(BaseCallbackHandler):

    def __init__(self, budget, rate_per_input_token_uc=0.15,
                 rate_per_output_token_uc=0.60):
        super().__init__()
        self._budget = budget
        self._cap = budget.micro_cents() if hasattr(budget, 'micro_cents') \
            else budget
        self._cap = self._cap() if callable(self._cap) else self._cap
        self.rate_in = float(rate_per_input_token_uc)
        self.rate_out = float(rate_per_output_token_uc)
        self._spent_so_far = 0.0

    def on_llm_start(self, serialized, prompts, **kwargs):
        # Pre-flight: estimate is conservative byte-length input + max output.
        est_input = sum(len(p) for p in prompts) * self.rate_in
        # Assume up to 500 output tokens per call as conservative ceiling.
        est_output = 500 * self.rate_out
        est = est_input + est_output
        if self._spent_so_far + est > self._cap:
            raise BudgetExhausted(
                f"pre-flight estimate {est:.0f} uc + running spend "
                f"{self._spent_so_far:.0f} uc would exceed cap {self._cap} uc"
            )

    def on_llm_end(self, response, **kwargs):
        try:
            for gen_list in response.generations:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    if msg is not None:
                        usage = getattr(msg, "usage_metadata", None)
                        if usage:
                            cost = (
                                    usage.get("input_tokens", 0) * self.rate_in
                                    + usage.get("output_tokens", 0) * self.rate_out
                            )
                            self._spent_so_far += cost
                            if self._spent_so_far > self._cap:
                                raise BudgetExhausted(
                                    f"running spend {self._spent_so_far:.0f} uc "
                                    f"exceeded cap {self._cap} uc"
                                )
                            return
        except BudgetExhausted:
            raise
        except Exception:
            pass
        usage = getattr(response, "llm_output", {}).get("token_usage", {})
        if usage:
            cost = (
                    usage.get("prompt_tokens", 0) * self.rate_in
                    + usage.get("completion_tokens", 0) * self.rate_out
            )
            self._spent_so_far += cost
            if self._spent_so_far > self._cap:
                raise BudgetExhausted(
                    f"running spend {self._spent_so_far:.0f} uc "
                    f"exceeded cap {self._cap} uc"
                )

TASK_PROMPT = (
    "I need to find the internal user ID for the email "
    "'sajjad@example.com' in our system. Use the lookup_user tool. "
    "Try variations of the email if the first attempt doesn't return "
    "an exact match. Keep trying until you find an exact match."
)

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a user-lookup tool. "
    "Use the tool to satisfy the user's request. If the tool's answer "
    "is ambiguous, try again with a slightly different query."
)

RATE_IN_UC  = 0.15
RATE_OUT_UC = 0.60

PROTECTED_CAP_UC = 2000


@dataclass
class RunResult:
    condition: str
    run_index: int
    outcome: str
    n_tool_calls: int
    n_llm_calls: int
    total_input_tokens: int
    total_output_tokens: int
    actual_spent_uc: float
    wall_time_sec: float
    error_msg: Optional[str] = None

def build_lookup_tool():
    try:
        from langchain_core.tools import tool
    except ImportError:
        print("ERROR: langchain_core not installed.", file=sys.stderr)
        sys.exit(1)

    @tool
    def lookup_user(query: str) -> str:
        return (
            f"Looked up '{query}'. Found 3 partial matches but no exact "
            f"match. Try a more specific query, e.g. the full email "
            f"including any aliases (e.g. user+tag@domain)."
        )

    return lookup_user


def build_agent(model: str, callbacks: Optional[List] = None):
    try:
        from langchain_openai import ChatOpenAI
        from langgraph.prebuilt import create_react_agent
    except ImportError as e:
        print(f"ERROR: required LangChain module not installed — {e}",
              file=sys.stderr)
        print(
            "Install: pip install 'langchain>=0.3,<0.4' "
            "'langchain-openai>=0.2,<0.3' 'langgraph>=0.2,<0.4'",
            file=sys.stderr,
        )
        sys.exit(1)

    llm = ChatOpenAI(
        model=model,
        temperature=0.7,           # non-zero so we get variation
        callbacks=callbacks or [],
    )
    tool = build_lookup_tool()
    agent = create_react_agent(llm, tools=[tool])
    return agent

class TokenCounter(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.input_tokens = 0
        self.output_tokens = 0
        self.llm_call_count = 0
        self.tool_call_count = 0

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.llm_call_count += 1

    def on_llm_end(self, response, **kwargs):
        try:
            for gen_list in response.generations:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    if msg is not None:
                        usage = getattr(msg, "usage_metadata", None)
                        if usage:
                            self.input_tokens += usage.get("input_tokens", 0)
                            self.output_tokens += usage.get("output_tokens", 0)
                            return
            usage = getattr(response, "llm_output", {}).get("token_usage", {})
            if usage:
                self.input_tokens += usage.get("prompt_tokens", 0)
                self.output_tokens += usage.get("completion_tokens", 0)
        except Exception:
            pass

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.tool_call_count += 1

    def total_spent_uc(self) -> float:
        return self.input_tokens * RATE_IN_UC + self.output_tokens * RATE_OUT_UC


def run_vulnerable(run_index: int, model: str, recursion_limit: int) -> RunResult:
    counter = TokenCounter()
    agent = build_agent(model, callbacks=[counter])
    start = time.time()
    outcome = "completed"
    error_msg = None
    try:
        result = agent.invoke(
            {"messages": [
                ("system", SYSTEM_PROMPT),
                ("user", TASK_PROMPT),
            ]},
            config={"recursion_limit": recursion_limit, "callbacks": [counter]},
        )

        if counter.tool_call_count >= recursion_limit // 2:
            outcome = "recursion_limit_hit"
    except Exception as e:
        msg = str(e).lower()
        if "recursion" in msg or "iteration" in msg or "limit" in msg:
            outcome = "recursion_limit_hit"
        else:
            outcome = "error"
            error_msg = str(e)[:200]
    elapsed = time.time() - start

    return RunResult(
        condition="vulnerable",
        run_index=run_index,
        outcome=outcome,
        n_tool_calls=counter.tool_call_count,
        n_llm_calls=counter.llm_call_count,
        total_input_tokens=counter.input_tokens,
        total_output_tokens=counter.output_tokens,
        actual_spent_uc=counter.total_spent_uc(),
        wall_time_sec=elapsed,
        error_msg=error_msg,
    )


def run_protected(run_index: int, model: str, recursion_limit: int,
                  cap_uc: int) -> RunResult:
    counter = TokenCounter()
    budget = Budget(initial_uc=cap_uc, max_uc=cap_uc * 10)
    budget_cb = LangChainBudgetCallback(
        budget,
        rate_per_input_token_uc=RATE_IN_UC,
        rate_per_output_token_uc=RATE_OUT_UC,
    )
    agent = build_agent(model, callbacks=[counter, budget_cb])
    start = time.time()
    outcome = "completed"
    error_msg = None
    try:
        agent.invoke(
            {"messages": [
                ("system", SYSTEM_PROMPT),
                ("user", TASK_PROMPT),
            ]},
            config={"recursion_limit": recursion_limit,
                    "callbacks": [counter, budget_cb]},
        )
        if counter.tool_call_count >= recursion_limit // 2:
            outcome = "recursion_limit_hit"
    except BudgetExhausted as e:
        outcome = "budget_exhausted"
        error_msg = str(e)[:200]
    except Exception as e:
        msg = str(e).lower()
        if "budgetexhausted" in msg or "budget" in msg:
            outcome = "budget_exhausted"
            error_msg = str(e)[:200]
        elif "recursion" in msg or "iteration" in msg or "limit" in msg:
            outcome = "recursion_limit_hit"
        else:
            outcome = "error"
            error_msg = str(e)[:200]
    elapsed = time.time() - start

    return RunResult(
        condition="protected",
        run_index=run_index,
        outcome=outcome,
        n_tool_calls=counter.tool_call_count,
        n_llm_calls=counter.llm_call_count,
        total_input_tokens=counter.input_tokens,
        total_output_tokens=counter.output_tokens,
        actual_spent_uc=counter.total_spent_uc(),
        wall_time_sec=elapsed,
        error_msg=error_msg,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5,
                        help="number of replicas per condition")
    parser.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--recursion-limit", type=int, default=25,
                        help="LangGraph recursion limit (default 25)")
    parser.add_argument("--cap-uc", type=int, default=PROTECTED_CAP_UC,
                        help="Budget cap in micro-cents for protected condition")
    parser.add_argument("--smoke", action="store_true",
                        help="N=1 per condition, ~$0.02")
    parser.add_argument("--out-csv", default="lang_001_real_results.csv")
    parser.add_argument("--out-json", default="lang_001_real_summary.json")
    args = parser.parse_args()

    if args.smoke:
        args.runs = 1

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    all_runs: List[RunResult] = []
    start = time.time()

    print(f"\n=== LANG-001 reproduction ===")
    print(f"Model: {args.model}")
    print(f"Recursion limit: {args.recursion_limit}")
    print(f"Cap (protected): {args.cap_uc} uc")
    print(f"Replicas per condition: {args.runs}")
    running_spend = 0.0

    print(f"\n--- Vulnerable (no budget) ---")
    for i in range(args.runs):
        r = run_vulnerable(i, args.model, args.recursion_limit)
        all_runs.append(r)
        running_spend += r.actual_spent_uc / 1_000_000
        print(
            f"  run {i+1}/{args.runs}: outcome={r.outcome:<22s} "
            f"tool_calls={r.n_tool_calls} llm_calls={r.n_llm_calls} "
            f"in={r.total_input_tokens} out={r.total_output_tokens} "
            f"spent_uc={r.actual_spent_uc:.0f} "
            f"wall={r.wall_time_sec:.1f}s "
            f"(running $: {running_spend:.4f})"
        )
        if r.error_msg:
            print(f"    error: {r.error_msg}")

    print(f"\n--- Protected (Budget cap={args.cap_uc} uc) ---")
    for i in range(args.runs):
        r = run_protected(i, args.model, args.recursion_limit, args.cap_uc)
        all_runs.append(r)
        running_spend += r.actual_spent_uc / 1_000_000
        print(
            f"  run {i+1}/{args.runs}: outcome={r.outcome:<22s} "
            f"tool_calls={r.n_tool_calls} llm_calls={r.n_llm_calls} "
            f"in={r.total_input_tokens} out={r.total_output_tokens} "
            f"spent_uc={r.actual_spent_uc:.0f} "
            f"wall={r.wall_time_sec:.1f}s "
            f"(running $: {running_spend:.4f})"
        )
        if r.error_msg:
            print(f"    error: {r.error_msg}")

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(asdict(all_runs[0]).keys())
        )
        writer.writeheader()
        for r in all_runs:
            writer.writerow(asdict(r))

    summary = {}
    for cond in ("vulnerable", "protected"):
        runs = [r for r in all_runs if r.condition == cond]
        if not runs:
            continue
        spent = [r.actual_spent_uc for r in runs]
        tool_calls = [r.n_tool_calls for r in runs]
        outcomes: Dict[str, int] = {}
        for r in runs:
            outcomes[r.outcome] = outcomes.get(r.outcome, 0) + 1
        summary[cond] = {
            "n_total": len(runs),
            "outcomes": outcomes,
            "mean_spent_uc": statistics.mean(spent) if spent else 0,
            "stdev_spent_uc": statistics.stdev(spent) if len(spent) > 1 else 0,
            "max_spent_uc": max(spent) if spent else 0,
            "mean_tool_calls": statistics.mean(tool_calls) if tool_calls else 0,
            "max_tool_calls": max(tool_calls) if tool_calls else 0,
        }

    if "vulnerable" in summary and "protected" in summary:
        v = summary["vulnerable"]["mean_spent_uc"]
        p = summary["protected"]["mean_spent_uc"]
        if v > 0:
            summary["bug_prevented"] = {
                "vulnerable_mean_uc": v,
                "protected_mean_uc": p,
                "reduction_factor": v / p if p > 0 else float("inf"),
                "cap_effective": p <= args.cap_uc,
                "protected_outcomes": summary["protected"]["outcomes"],
            }

    summary["__totals__"] = {
        "n_total": len(all_runs),
        "wall_time_sec": time.time() - start,
        "estimated_cost_usd": sum(r.actual_spent_uc for r in all_runs) / 1_000_000,
    }

    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary===")
    print(f"Total runs:              {len(all_runs)}")
    print(f"Estimated total spend:   ${summary['__totals__']['estimated_cost_usd']:.4f}")
    print(f"Wall time:               {summary['__totals__']['wall_time_sec']:.1f}s")
    print(f"CSV:                     {args.out_csv}")
    print(f"Summary:                 {args.out_json}")
    if "bug_prevented" in summary:
        bp = summary["bug_prevented"]
        print(f"\n--- Bug prevention ---")
        print(f"Vulnerable mean spend:   {bp['vulnerable_mean_uc']:.0f} uc")
        print(f"Protected mean spend:    {bp['protected_mean_uc']:.0f} uc")
        print(f"Reduction factor:        {bp['reduction_factor']:.1f}x")
        print(f"Cap respected:           {bp['cap_effective']}")


if __name__ == "__main__":
    main()