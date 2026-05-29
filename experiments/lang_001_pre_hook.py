import argparse
import csv
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Optional


HERE = os.path.dirname(os.path.abspath(__file__))
PORT_DIR = os.path.join(HERE, "..", "reproductions")
if os.path.isdir(PORT_DIR):
    sys.path.insert(0, PORT_DIR)

try:
    from token_budgets import Budget, BudgetExhausted
except ImportError as e:
    print(f"ERROR: cannot import token_budgets — {e}", file=sys.stderr)
    print(f"Expected at: {PORT_DIR}/token_budgets.py", file=sys.stderr)
    print("Run from token-budgets-python/experiments/ directory.", file=sys.stderr)
    sys.exit(1)


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

EST_OUTPUT_TOKENS = 500


@dataclass
class RunResult:
    condition: str
    run_index: int
    outcome: str
    n_tool_calls: int
    n_llm_calls: int
    spent_uc: float
    pre_hook_fired_at_call: Optional[int]
    wall_sec: float
    error_msg: Optional[str] = None


def _build_lookup_tool():
    from langchain_core.tools import tool

    @tool
    def lookup_user(query: str) -> str:
        return (
            f"Looked up '{query}'. Found 3 partial matches but no exact "
            f"match. Try a more specific query, e.g. the full email "
            f"including any aliases (e.g. user+tag@domain)."
        )
    return lookup_user


def _build_token_counter_callback(state: dict):
    from langchain_core.callbacks import BaseCallbackHandler

    class _CB(BaseCallbackHandler):
        def on_llm_start(self, serialized, prompts, **kwargs):
            state["n_llm_calls"] += 1

        def on_chat_model_start(self, serialized, messages, **kwargs):
            state["n_llm_calls"] += 1

        def on_llm_end(self, response, **kwargs):
            # LangChain 0.2+ usage_metadata path
            try:
                for gl in response.generations:
                    for g in gl:
                        m = getattr(g, "message", None)
                        if m is not None:
                            u = getattr(m, "usage_metadata", None)
                            if u:
                                in_t = u.get("input_tokens", 0) or 0
                                out_t = u.get("output_tokens", 0) or 0
                                state["spent_uc"] += (
                                        in_t * RATE_IN_UC + out_t * RATE_OUT_UC
                                )
                                return
            except Exception:
                pass
            # Legacy fallback
            try:
                u = (getattr(response, "llm_output", None) or {}).get(
                    "token_usage", {}
                )
                if u:
                    state["spent_uc"] += (
                            u.get("prompt_tokens", 0) * RATE_IN_UC
                            + u.get("completion_tokens", 0) * RATE_OUT_UC
                    )
            except Exception:
                pass

        def on_tool_start(self, serialized, input_str, **kwargs):
            state["tool_calls"] += 1
    return _CB()


def _build_pre_model_hook(cap_uc: float, state: dict):
    def hook(graph_state):
        msgs = graph_state.get("messages", [])
        prompt_text = ""
        for m in msgs:
            c = getattr(m, "content", None)
            if isinstance(c, str):
                prompt_text += c
            elif isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        prompt_text += part["text"]
        est = len(prompt_text.encode("utf-8")) * RATE_IN_UC + EST_OUTPUT_TOKENS * RATE_OUT_UC
        if state["spent_uc"] + est > cap_uc:
            state["pre_hook_fired_at_call"] = state["n_llm_calls"] + 1
            raise BudgetExhausted(
                f"pre_model_hook: running spend {state['spent_uc']:.0f} uc "
                f"+ estimated next-call cost {est:.0f} uc > cap {cap_uc} uc"
            )
        return graph_state

    return hook


def run_one(condition: str, run_index: int, model: str,
            recursion_limit: int, cap_uc: Optional[float]) -> RunResult:

    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    state = {
        "spent_uc": 0.0, "n_llm_calls": 0,
        "tool_calls": 0, "pre_hook_fired_at_call": None,
    }
    counter_cb = _build_token_counter_callback(state)
    llm = ChatOpenAI(model=model, temperature=0.7, callbacks=[counter_cb])
    lookup_tool = _build_lookup_tool()

    if condition == "protected":
        if cap_uc is None:
            raise ValueError("protected condition requires cap_uc")
        hook = _build_pre_model_hook(cap_uc, state)
        try:
            agent = create_react_agent(
                llm, tools=[lookup_tool], pre_model_hook=hook,
            )
        except TypeError as e:
            return RunResult(
                condition=condition, run_index=run_index, outcome="error",
                n_tool_calls=0, n_llm_calls=0, spent_uc=0.0,
                pre_hook_fired_at_call=None, wall_sec=0.0,
                error_msg=(
                    f"LangGraph does not support pre_model_hook in this "
                    f"version. Upgrade to >=0.3. ({e})"
                ),
            )
    else:
        agent = create_react_agent(llm, tools=[lookup_tool])

    start = time.time()
    outcome = "completed"
    err_msg = None
    try:
        agent.invoke(
            {"messages": [("system", SYSTEM_PROMPT), ("user", TASK_PROMPT)]},
            config={"recursion_limit": recursion_limit, "callbacks": [counter_cb]},
        )

        if state["tool_calls"] >= recursion_limit // 2:
            outcome = "recursion_limit_hit"
    except BudgetExhausted as e:
        outcome = "budget_exhausted"
        err_msg = str(e)[:200]
    except Exception as e:
        msg = str(e).lower()
        if "budgetexhausted" in msg or "budget exhausted" in msg:
            outcome = "budget_exhausted"; err_msg = str(e)[:200]
        elif "recursion" in msg or "iteration" in msg or "limit" in msg:
            outcome = "recursion_limit_hit"
        else:
            outcome = "error"; err_msg = str(e)[:200]
    elapsed = time.time() - start

    return RunResult(
        condition=condition, run_index=run_index, outcome=outcome,
        n_tool_calls=state["tool_calls"], n_llm_calls=state["n_llm_calls"],
        spent_uc=state["spent_uc"],
        pre_hook_fired_at_call=state["pre_hook_fired_at_call"],
        wall_sec=elapsed, error_msg=err_msg,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=5,
                   help="replicas per condition")
    p.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    p.add_argument("--recursion-limit", type=int, default=25)
    p.add_argument("--cap-uc", type=float, default=700,
                   help="micro-cents cap for protected condition")
    p.add_argument("--smoke", action="store_true",
                   help="N=1 per condition (~$0.003)")
    p.add_argument("--out-csv", default="lang_001_pre_hook_results.csv")
    p.add_argument("--out-json", default="lang_001_pre_hook_summary.json")
    args = p.parse_args()

    if args.smoke:
        args.runs = 1
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    all_runs: List[RunResult] = []
    t0 = time.time()
    running_spend_usd = 0.0

    print(f"\n LANG-001 pre_model_hook enforcement ")
    print(f"Model:           {args.model}")
    print(f"Recursion limit: {args.recursion_limit}")
    print(f"Cap (protected): {args.cap_uc} uc")
    print(f"Replicas:        {args.runs} per condition")

    print(f"\n--- Vulnerable (no pre_model_hook) ---")
    for i in range(args.runs):
        r = run_one("vulnerable", i, args.model, args.recursion_limit, None)
        all_runs.append(r)
        running_spend_usd += r.spent_uc / 1_000_000
        print(
            f"  {i+1}/{args.runs}: outcome={r.outcome:<22s} "
            f"tool={r.n_tool_calls} llm={r.n_llm_calls} "
            f"spent_uc={r.spent_uc:.0f} wall={r.wall_sec:.1f}s "
            f"(running $: {running_spend_usd:.4f})"
        )
        if r.error_msg:
            print(f"    error: {r.error_msg}")

    print(f"\n--- Protected (pre_model_hook cap={args.cap_uc} uc) ---")
    for i in range(args.runs):
        r = run_one("protected", i, args.model, args.recursion_limit, args.cap_uc)
        all_runs.append(r)
        running_spend_usd += r.spent_uc / 1_000_000
        fire_str = (f"hook_fired_at_call={r.pre_hook_fired_at_call}"
                    if r.pre_hook_fired_at_call else "hook_never_fired")
        print(
            f"  {i+1}/{args.runs}: outcome={r.outcome:<22s} "
            f"tool={r.n_tool_calls} llm={r.n_llm_calls} "
            f"spent_uc={r.spent_uc:.0f} {fire_str} "
            f"wall={r.wall_sec:.1f}s (running $: {running_spend_usd:.4f})"
        )
        if r.error_msg:
            print(f"    error: {r.error_msg}")

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(all_runs[0]).keys()))
        w.writeheader()
        for r in all_runs:
            w.writerow(asdict(r))

    summary = {}
    for cond in ("vulnerable", "protected"):
        rs = [r for r in all_runs if r.condition == cond]
        if not rs:
            continue
        outcomes = {}
        for r in rs:
            outcomes[r.outcome] = outcomes.get(r.outcome, 0) + 1
        spent = [r.spent_uc for r in rs]
        tools = [r.n_tool_calls for r in rs]
        summary[cond] = {
            "n_total": len(rs),
            "outcomes": outcomes,
            "mean_spent_uc": statistics.mean(spent),
            "stdev_spent_uc": statistics.stdev(spent) if len(spent) > 1 else 0,
            "max_spent_uc": max(spent),
            "mean_tool_calls": statistics.mean(tools),
            "max_tool_calls": max(tools),
        }

    if "vulnerable" in summary and "protected" in summary:
        v = summary["vulnerable"]["mean_spent_uc"]
        p = summary["protected"]["mean_spent_uc"]
        prot_runs = [r for r in all_runs if r.condition == "protected"]
        bx_outcomes = sum(1 for r in prot_runs if r.outcome == "budget_exhausted")
        cap_respected = sum(1 for r in prot_runs if r.spent_uc <= args.cap_uc)
        summary["bug_prevention"] = {
            "vulnerable_mean_uc": v,
            "protected_mean_uc": p,
            "reduction_factor": v / p if p > 0 else float("inf"),
            "n_budget_exhausted": bx_outcomes,
            "n_total_protected": len(prot_runs),
            "n_cap_respected": cap_respected,
        }
    summary["__totals__"] = {
        "n_total": len(all_runs),
        "wall_sec": time.time() - t0,
        "estimated_cost_usd": sum(r.spent_uc for r in all_runs) / 1_000_000,
        "cap_uc": args.cap_uc,
        "model": args.model,
    }

    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"Total runs:           {len(all_runs)}")
    print(f"Total spend:          ${summary['__totals__']['estimated_cost_usd']:.4f}")
    print(f"Wall time:            {summary['__totals__']['wall_sec']:.1f}s")
    if "bug_prevention" in summary:
        bp = summary["bug_prevention"]
        print(f"\n--- Bug prevention ---")
        print(f"Vulnerable mean spend: {bp['vulnerable_mean_uc']:.0f} uc")
        print(f"Protected mean spend:  {bp['protected_mean_uc']:.0f} uc")
        print(f"Reduction factor:      {bp['reduction_factor']:.1f}x")
        print(f"budget_exhausted:      {bp['n_budget_exhausted']}/{bp['n_total_protected']}")
        print(f"cap_respected:         {bp['n_cap_respected']}/{bp['n_total_protected']}")


if __name__ == "__main__":
    main()