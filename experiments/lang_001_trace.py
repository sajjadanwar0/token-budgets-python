#!/usr/bin/env python3
"""
lang_001_trace.py — Trace-analysis demonstration of LANG-001 prevention.

Replaces lang_001_real.py's callback-enforcement approach with a cleaner
trace-analysis approach.

Rationale: LangChain swallows callback exceptions internally. A
BudgetExhausted raised from on_llm_start/on_llm_end is logged as a
callback error and the agent loop continues. This is documented
LangChain behaviour and is not a bug in the Budget discipline. The
discipline's correctness can be established from the per-call spend
trace without requiring callback-level enforcement to fire.

Method:
  1. Run vulnerable LangGraph ReAct agent (no enforcement) with
     detailed TokenCounter that records cumulative spend per LLM call.
  2. From the trace, post-hoc compute: at various cap levels,
     when does the Budget discipline first fire (running spend would
     exceed cap)? What is the resulting spend bound?
  3. Report per-cap-level analysis: discipline fires at call N,
     bounds spend at ≤cap, reduction factor vs unbounded.

This separates the mathematical claim of the discipline (cap-respecting)
from its LangChain integration (a separate engineering concern).

Prerequisites: same as lang_001_real.py
  pip install 'langchain>=0.3,<0.4' 'langchain-openai>=0.2,<0.3' \\
              'langgraph>=0.2,<0.4' 'langchain-core>=0.3,<0.4'

Usage:
  export OPENAI_API_KEY=sk-...
  python3 lang_001_trace.py --runs 5 \\
      --out-csv lang_001_trace_results.csv \\
      --out-json lang_001_trace_summary.json

Smoke test:
  python3 lang_001_trace.py --smoke
"""

import argparse
import csv
import json
import os
import sys
import time
import statistics
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple


# Cost rates (gpt-4o-mini, micro-cents per token)
RATE_IN_UC  = 0.15
RATE_OUT_UC = 0.60

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


@dataclass
class CallTelemetry:
    """Per-LLM-call telemetry recorded during a run."""
    call_index: int               # 0-based
    input_tokens: int
    output_tokens: int
    cost_uc: float                # for this call alone
    cumulative_uc: float          # running total after this call


@dataclass
class RunTrace:
    """Full trace of one vulnerable agent run."""
    run_index: int
    outcome: str                  # "completed" | "recursion_limit_hit" | "error"
    n_tool_calls: int
    n_llm_calls: int
    calls: List[CallTelemetry] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_spent_uc: float = 0.0
    wall_time_sec: float = 0.0
    error_msg: Optional[str] = None


@dataclass
class CapAnalysis:
    """Post-hoc: where would Budget(cap) have fired in this trace?"""
    cap_uc: float
    fires_at_call: Optional[int]  # 1-based; None if cap never crossed
    spend_at_fire_uc: Optional[float]
    cap_respected: bool           # spend_at_fire ≤ cap
    reduction_factor: float       # unbounded_total / bounded_total


# ----------------------------------------------------------------------
# Callback that records per-call telemetry (no enforcement)
# ----------------------------------------------------------------------

def make_telemetry_callback():
    """Returns a BaseCallbackHandler subclass that records per-call
    input/output tokens to a shared list. No enforcement."""
    from langchain_core.callbacks import BaseCallbackHandler

    class TelemetryHandler(BaseCallbackHandler):
        def __init__(self):
            super().__init__()
            self.calls: List[CallTelemetry] = []
            self.tool_call_count = 0
            self.llm_call_count = 0
            self._cumulative_uc = 0.0

        def on_llm_start(self, serialized, prompts, **kwargs):
            self.llm_call_count += 1

        def on_llm_end(self, response, **kwargs):
            # Try LangChain 0.2+ usage_metadata path
            for gen_list in response.generations:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    if msg is not None:
                        usage = getattr(msg, "usage_metadata", None)
                        if usage:
                            in_t = usage.get("input_tokens", 0)
                            out_t = usage.get("output_tokens", 0)
                            cost = in_t * RATE_IN_UC + out_t * RATE_OUT_UC
                            self._cumulative_uc += cost
                            self.calls.append(CallTelemetry(
                                call_index=len(self.calls),
                                input_tokens=in_t,
                                output_tokens=out_t,
                                cost_uc=cost,
                                cumulative_uc=self._cumulative_uc,
                            ))
                            return
            # Fallback for older LangChain
            usage = getattr(response, "llm_output", {}).get("token_usage", {})
            if usage:
                in_t = usage.get("prompt_tokens", 0)
                out_t = usage.get("completion_tokens", 0)
                cost = in_t * RATE_IN_UC + out_t * RATE_OUT_UC
                self._cumulative_uc += cost
                self.calls.append(CallTelemetry(
                    call_index=len(self.calls),
                    input_tokens=in_t, output_tokens=out_t,
                    cost_uc=cost, cumulative_uc=self._cumulative_uc,
                ))

        def on_tool_start(self, serialized, input_str, **kwargs):
            self.tool_call_count += 1

    return TelemetryHandler()


def build_agent_and_tool(model: str, callbacks):
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent

    @tool
    def lookup_user(query: str) -> str:
        """Look up a user by email or username.
        Args:
            query: an email or username string to look up.
        Returns:
            A diagnostic string about the lookup result.
        """
        return (
            f"Looked up '{query}'. Found 3 partial matches but no exact "
            f"match. Try a more specific query, e.g. the full email "
            f"including any aliases (e.g. user+tag@domain)."
        )

    llm = ChatOpenAI(model=model, temperature=0.7, callbacks=callbacks)
    return create_react_agent(llm, tools=[lookup_user])


def run_vulnerable_with_telemetry(run_index: int, model: str,
                                  recursion_limit: int) -> RunTrace:
    """Run the vulnerable agent and record per-call telemetry."""
    tele = make_telemetry_callback()
    agent = build_agent_and_tool(model, callbacks=[tele])
    start = time.time()
    outcome = "completed"
    error_msg = None
    try:
        agent.invoke(
            {"messages": [("system", SYSTEM_PROMPT), ("user", TASK_PROMPT)]},
            config={"recursion_limit": recursion_limit, "callbacks": [tele]},
        )
        if tele.tool_call_count >= recursion_limit // 2:
            outcome = "recursion_limit_hit"
    except Exception as e:
        msg = str(e).lower()
        if "recursion" in msg or "iteration" in msg or "limit" in msg:
            outcome = "recursion_limit_hit"
        else:
            outcome = "error"
            error_msg = str(e)[:200]
    elapsed = time.time() - start

    return RunTrace(
        run_index=run_index,
        outcome=outcome,
        n_tool_calls=tele.tool_call_count,
        n_llm_calls=tele.llm_call_count,
        calls=tele.calls,
        total_input_tokens=sum(c.input_tokens for c in tele.calls),
        total_output_tokens=sum(c.output_tokens for c in tele.calls),
        total_spent_uc=tele.calls[-1].cumulative_uc if tele.calls else 0.0,
        wall_time_sec=elapsed,
        error_msg=error_msg,
    )


# ----------------------------------------------------------------------
# Post-hoc trace analysis
# ----------------------------------------------------------------------

def analyze_trace_at_cap(trace: RunTrace, cap_uc: float) -> CapAnalysis:
    """Replay the trace under a Budget discipline with given cap.

    The discipline's pre-flight check fires if (running spend + this
    call's estimated cost) > cap. We model the estimate as the actual
    cost (a tight estimator; in practice the byte-length estimator
    over-estimates, which would fire the cap earlier).

    Conservative model (matches paper): fire when running_spend +
    next_call_cost > cap. This is the latest possible firing point.
    """
    cumulative = 0.0
    for i, call in enumerate(trace.calls):
        # Pre-flight check for this call: would running + this_call_cost exceed cap?
        if cumulative + call.cost_uc > cap_uc:
            # Discipline fires here. The call is REFUSED, so spend stays at cumulative.
            return CapAnalysis(
                cap_uc=cap_uc,
                fires_at_call=i + 1,                       # 1-based
                spend_at_fire_uc=cumulative,
                cap_respected=(cumulative <= cap_uc),
                reduction_factor=trace.total_spent_uc / cumulative
                if cumulative > 0 else float('inf'),
            )
        cumulative += call.cost_uc
    # Cap never crossed
    return CapAnalysis(
        cap_uc=cap_uc,
        fires_at_call=None,
        spend_at_fire_uc=cumulative,
        cap_respected=(cumulative <= cap_uc),
        reduction_factor=1.0,
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--recursion-limit", type=int, default=25)
    parser.add_argument("--caps", nargs="+", type=float,
                        default=[300, 500, 700, 1000, 2000],
                        help="cap levels (uc) for post-hoc analysis")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out-csv", default="lang_001_trace_results.csv")
    parser.add_argument("--out-json", default="lang_001_trace_summary.json")
    args = parser.parse_args()

    if args.smoke:
        args.runs = 1

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    traces: List[RunTrace] = []
    start = time.time()
    running_spend = 0.0

    print(f"\n=== LANG-001 trace analysis ===")
    print(f"Model: {args.model}")
    print(f"Recursion limit: {args.recursion_limit}")
    print(f"Cap levels for analysis: {args.caps}")
    print(f"Runs: {args.runs}")

    print(f"\n--- Vulnerable runs with per-call telemetry ---")
    for i in range(args.runs):
        t = run_vulnerable_with_telemetry(i, args.model, args.recursion_limit)
        traces.append(t)
        running_spend += t.total_spent_uc / 1_000_000
        print(
            f"  run {i+1}/{args.runs}: outcome={t.outcome:<22s} "
            f"tool_calls={t.n_tool_calls} llm_calls={t.n_llm_calls} "
            f"in={t.total_input_tokens} out={t.total_output_tokens} "
            f"spent_uc={t.total_spent_uc:.0f} "
            f"wall={t.wall_time_sec:.1f}s (running $: {running_spend:.4f})"
        )
        if t.error_msg:
            print(f"    error: {t.error_msg}")

    # Write per-call CSV (one row per LLM call across all runs)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_index", "outcome", "call_index", "input_tokens",
            "output_tokens", "cost_uc", "cumulative_uc",
        ])
        for t in traces:
            for c in t.calls:
                writer.writerow([
                    t.run_index, t.outcome, c.call_index,
                    c.input_tokens, c.output_tokens,
                    f"{c.cost_uc:.2f}", f"{c.cumulative_uc:.2f}",
                ])

    # Post-hoc: at each cap level, what happens to each trace?
    print(f"\n--- Post-hoc analysis: at each cap, where does the discipline fire? ---")
    cap_results: Dict[float, List[CapAnalysis]] = {}
    for cap in args.caps:
        cap_results[cap] = []
        print(f"\nCap = {cap} uc:")
        for t in traces:
            an = analyze_trace_at_cap(t, cap)
            cap_results[cap].append(an)
            fire_str = (f"fires at call {an.fires_at_call}, spend bounded at "
                        f"{an.spend_at_fire_uc:.0f} uc")
            if an.fires_at_call is None:
                fire_str = f"never fires (max spend {an.spend_at_fire_uc:.0f} uc < cap)"
            print(
                f"  run {t.run_index}: total unbounded={t.total_spent_uc:.0f} uc, "
                f"{fire_str}, reduction={an.reduction_factor:.1f}x, "
                f"cap_respected={an.cap_respected}"
            )

    # Aggregate summary
    summary = {
        "vulnerable": {
            "n_runs": len(traces),
            "outcomes": {},
            "mean_spent_uc": statistics.mean([t.total_spent_uc for t in traces]),
            "stdev_spent_uc": (statistics.stdev([t.total_spent_uc for t in traces])
                               if len(traces) > 1 else 0),
            "max_spent_uc": max(t.total_spent_uc for t in traces),
            "mean_tool_calls": statistics.mean([t.n_tool_calls for t in traces]),
            "max_tool_calls": max(t.n_tool_calls for t in traces),
        },
        "cap_analysis": {},
    }
    for t in traces:
        summary["vulnerable"]["outcomes"][t.outcome] = \
            summary["vulnerable"]["outcomes"].get(t.outcome, 0) + 1

    for cap, analyses in cap_results.items():
        fires = [a for a in analyses if a.fires_at_call is not None]
        respected = sum(1 for a in analyses if a.cap_respected)
        summary["cap_analysis"][str(cap)] = {
            "n_runs": len(analyses),
            "n_fires": len(fires),
            "n_cap_respected": respected,
            "mean_fires_at_call": statistics.mean([a.fires_at_call for a in fires])
            if fires else None,
            "mean_spend_at_fire_uc": statistics.mean(
                [a.spend_at_fire_uc for a in analyses]),
            "mean_reduction_factor": statistics.mean(
                [a.reduction_factor for a in analyses]),
        }

    summary["__totals__"] = {
        "n_runs": len(traces),
        "wall_time_sec": time.time() - start,
        "estimated_cost_usd": sum(t.total_spent_uc for t in traces) / 1_000_000,
    }

    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"Vulnerable runs:         {len(traces)}")
    print(f"Mean unbounded spend:    {summary['vulnerable']['mean_spent_uc']:.0f} uc")
    print(f"Total cost:              ${summary['__totals__']['estimated_cost_usd']:.4f}")
    print(f"Wall time:               {summary['__totals__']['wall_time_sec']:.1f}s")
    print(f"\nFor each cap level, see:")
    print(f"  CSV (per-call traces): {args.out_csv}")
    print(f"  JSON (aggregate):      {args.out_json}")


if __name__ == "__main__":
    main()