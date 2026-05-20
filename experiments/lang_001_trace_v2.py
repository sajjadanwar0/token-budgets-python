import argparse
import csv
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict


HERE = os.path.dirname(os.path.abspath(__file__))
PORT_DIR = os.path.join(HERE, "..", "reproductions")
if os.path.isdir(PORT_DIR):
    sys.path.insert(0, PORT_DIR)

RATES = {
    "openai": {
        "gpt-4o-mini-2024-07-18": {"in": 0.15, "out": 0.60},
        "gpt-4o-2024-08-06":      {"in": 2.50, "out": 10.0},
    },
    "anthropic": {
        "claude-haiku-4-5-20251001": {"in": 1.0,  "out": 5.0},
        "claude-sonnet-4-5-20250929": {"in": 3.0, "out": 15.0},
    },
}

DEFAULT_MODELS = {
    "openai":    "gpt-4o-mini-2024-07-18",
    "anthropic": "claude-haiku-4-5-20251001",
}


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
    call_index: int
    input_tokens: int
    output_tokens: int
    cost_uc: float
    cumulative_uc: float


@dataclass
class RunTrace:
    provider: str
    model: str
    temperature: float
    run_index: int
    outcome: str
    n_tool_calls: int
    n_llm_calls: int
    calls: List[CallTelemetry] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_spent_uc: float = 0.0
    wall_sec: float = 0.0
    error_msg: Optional[str] = None


@dataclass
class CapAnalysis:
    cap_uc: float
    fires_at_call: Optional[int]
    spend_at_fire_uc: Optional[float]
    cap_respected: bool
    reduction_factor: float


def _rates_for(provider: str, model: str) -> Dict[str, float]:
    if provider in RATES and model in RATES[provider]:
        return RATES[provider][model]
    # Best-effort fallback
    fallback = {"in": 1.0, "out": 3.0}
    print(f"WARNING: no rate entry for {provider}/{model}; using fallback {fallback}",
          file=sys.stderr)
    return fallback


def _build_telemetry_callback(rate_in: float, rate_out: float):
    from langchain_core.callbacks import BaseCallbackHandler

    class _TH(BaseCallbackHandler):
        def __init__(self):
            super().__init__()
            self.calls: List[CallTelemetry] = []
            self.tool_call_count = 0
            self.llm_call_count = 0
            self._cumulative_uc = 0.0

        def on_llm_start(self, serialized, prompts, **kwargs):
            self.llm_call_count += 1
        def on_chat_model_start(self, serialized, messages, **kwargs):
            self.llm_call_count += 1

        def on_llm_end(self, response, **kwargs):
            in_t = out_t = 0
            try:
                for gl in response.generations:
                    for g in gl:
                        m = getattr(g, "message", None)
                        if m is not None:
                            u = getattr(m, "usage_metadata", None)
                            if u:
                                in_t = u.get("input_tokens", 0) or 0
                                out_t = u.get("output_tokens", 0) or 0
                                break
                    if in_t or out_t:
                        break
            except Exception:
                pass
            if not (in_t or out_t):
                try:
                    u = (getattr(response, "llm_output", None) or {}).get(
                        "token_usage", {}
                    )
                    in_t = u.get("prompt_tokens", 0) or 0
                    out_t = u.get("completion_tokens", 0) or 0
                except Exception:
                    pass
            cost = in_t * rate_in + out_t * rate_out
            self._cumulative_uc += cost
            self.calls.append(CallTelemetry(
                call_index=len(self.calls),
                input_tokens=in_t, output_tokens=out_t,
                cost_uc=cost, cumulative_uc=self._cumulative_uc,
            ))

        def on_tool_start(self, serialized, input_str, **kwargs):
            self.tool_call_count += 1
    return _TH()


def _build_llm(provider: str, model: str, temperature: float, callbacks):
    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            sys.exit("ERROR: langchain-openai not installed. "
                     "pip install 'langchain-openai>=0.2,<0.3'")
        return ChatOpenAI(model=model, temperature=temperature, callbacks=callbacks)
    elif provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            sys.exit("ERROR: langchain-anthropic not installed. "
                     "pip install 'langchain-anthropic>=0.2,<0.3'")
        return ChatAnthropic(
            model=model, temperature=temperature,
            max_tokens=512, callbacks=callbacks,
        )
    else:
        sys.exit(f"ERROR: unknown provider {provider}")


def _build_agent(provider: str, model: str, temperature: float, callbacks):
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent

    @tool
    def lookup_user(query: str) -> str:
        return (
            f"Looked up '{query}'. Found 3 partial matches but no exact "
            f"match. Try a more specific query, e.g. the full email "
            f"including any aliases (e.g. user+tag@domain)."
        )

    llm = _build_llm(provider, model, temperature, callbacks)
    return create_react_agent(llm, tools=[lookup_user])


def _check_api_key(provider: str):
    key_env = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
    if not os.environ.get(key_env):
        sys.exit(f"ERROR: {key_env} not set")


def run_one(provider: str, model: str, temperature: float, run_index: int,
            recursion_limit: int) -> RunTrace:
    rates = _rates_for(provider, model)
    tele = _build_telemetry_callback(rates["in"], rates["out"])
    agent = _build_agent(provider, model, temperature, [tele])
    t0 = time.time()
    outcome = "completed"
    err = None
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
            outcome = "error"; err = str(e)[:200]
    elapsed = time.time() - t0

    return RunTrace(
        provider=provider, model=model, temperature=temperature,
        run_index=run_index, outcome=outcome,
        n_tool_calls=tele.tool_call_count, n_llm_calls=tele.llm_call_count,
        calls=tele.calls,
        total_input_tokens=sum(c.input_tokens for c in tele.calls),
        total_output_tokens=sum(c.output_tokens for c in tele.calls),
        total_spent_uc=tele.calls[-1].cumulative_uc if tele.calls else 0.0,
        wall_sec=elapsed, error_msg=err,
    )


def analyze_at_cap(trace: RunTrace, cap_uc: float) -> CapAnalysis:
    cum = 0.0
    for i, c in enumerate(trace.calls):
        if cum + c.cost_uc > cap_uc:
            return CapAnalysis(
                cap_uc=cap_uc, fires_at_call=i + 1,
                spend_at_fire_uc=cum,
                cap_respected=(cum <= cap_uc),
                reduction_factor=(trace.total_spent_uc / cum) if cum > 0 else float("inf"),
            )
        cum += c.cost_uc
    return CapAnalysis(
        cap_uc=cap_uc, fires_at_call=None,
        spend_at_fire_uc=cum,
        cap_respected=(cum <= cap_uc),
        reduction_factor=1.0,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    p.add_argument("--model", default=None,
                   help="provider-specific model id (default: provider's default)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--recursion-limit", type=int, default=25)
    p.add_argument("--caps", nargs="+", type=float,
                   default=[300, 500, 700, 1000, 2000])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out-csv", default="lang_001_trace_v2_results.csv")
    p.add_argument("--out-json", default="lang_001_trace_v2_summary.json")
    args = p.parse_args()

    if args.smoke:
        args.runs = 1
    if args.model is None:
        args.model = DEFAULT_MODELS[args.provider]
    _check_api_key(args.provider)

    traces: List[RunTrace] = []
    t0 = time.time()
    running_spend = 0.0

    print(f"\n=== LANG-001 trace v2 ===")
    print(f"Provider:   {args.provider}")
    print(f"Model:      {args.model}")
    print(f"Temp:       {args.temperature}")
    print(f"Caps:       {args.caps}")
    print(f"Runs:       {args.runs}")

    print(f"\n--- Vulnerable runs ---")
    for i in range(args.runs):
        tr = run_one(args.provider, args.model, args.temperature, i,
                     args.recursion_limit)
        traces.append(tr)
        running_spend += tr.total_spent_uc / 1_000_000
        print(
            f"  {i+1}/{args.runs}: outcome={tr.outcome:<22s} "
            f"tool={tr.n_tool_calls} llm={tr.n_llm_calls} "
            f"in={tr.total_input_tokens} out={tr.total_output_tokens} "
            f"spent_uc={tr.total_spent_uc:.0f} (running $: {running_spend:.4f})"
        )
        if tr.error_msg:
            print(f"    error: {tr.error_msg}")

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "provider", "model", "temperature", "run_index", "outcome",
            "call_index", "input_tokens", "output_tokens",
            "cost_uc", "cumulative_uc",
        ])
        for tr in traces:
            for c in tr.calls:
                w.writerow([
                    tr.provider, tr.model, tr.temperature, tr.run_index,
                    tr.outcome, c.call_index, c.input_tokens, c.output_tokens,
                    f"{c.cost_uc:.4f}", f"{c.cumulative_uc:.4f}",
                ])

    print(f"\n--- Post-hoc cap analysis ---")
    cap_results: Dict[float, List[CapAnalysis]] = {}
    for cap in args.caps:
        cap_results[cap] = []
        print(f"\nCap = {cap} uc:")
        for tr in traces:
            an = analyze_at_cap(tr, cap)
            cap_results[cap].append(an)
            if an.fires_at_call is None:
                s = f"never fires (max {an.spend_at_fire_uc:.0f} < cap)"
            else:
                s = (f"fires at call {an.fires_at_call}, "
                     f"bounded {an.spend_at_fire_uc:.0f} uc")
            print(f"  run {tr.run_index}: {s}, reduction={an.reduction_factor:.1f}x")

    summary = {
        "config": {
            "provider": args.provider, "model": args.model,
            "temperature": args.temperature,
            "recursion_limit": args.recursion_limit,
            "n_runs": args.runs, "caps": args.caps,
        },
        "vulnerable": {
            "n_runs": len(traces),
            "outcomes": {},
            "mean_spent_uc": statistics.mean([t.total_spent_uc for t in traces]),
            "stdev_spent_uc": (statistics.stdev([t.total_spent_uc for t in traces])
                               if len(traces) > 1 else 0),
            "min_spent_uc": min(t.total_spent_uc for t in traces),
            "max_spent_uc": max(t.total_spent_uc for t in traces),
            "mean_tool_calls": statistics.mean([t.n_tool_calls for t in traces]),
            "max_tool_calls": max(t.n_tool_calls for t in traces),
        },
        "cap_analysis": {},
    }
    for tr in traces:
        summary["vulnerable"]["outcomes"][tr.outcome] = \
            summary["vulnerable"]["outcomes"].get(tr.outcome, 0) + 1
    for cap, ans in cap_results.items():
        fires = [a for a in ans if a.fires_at_call is not None]
        summary["cap_analysis"][str(cap)] = {
            "n_runs": len(ans),
            "n_fires": len(fires),
            "n_cap_respected": sum(1 for a in ans if a.cap_respected),
            "mean_fires_at_call": (statistics.mean([a.fires_at_call for a in fires])
                                   if fires else None),
            "mean_spend_at_fire_uc": statistics.mean(
                [a.spend_at_fire_uc for a in ans]),
            "mean_reduction_factor": statistics.mean(
                [a.reduction_factor for a in ans]),
        }
    summary["__totals__"] = {
        "n_runs": len(traces),
        "wall_sec": time.time() - t0,
        "estimated_cost_usd": sum(t.total_spent_uc for t in traces) / 1_000_000,
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"Vulnerable mean spend: {summary['vulnerable']['mean_spent_uc']:.0f} uc")
    print(f"Total cost:            ${summary['__totals__']['estimated_cost_usd']:.4f}")
    print(f"Wall time:             {summary['__totals__']['wall_sec']:.1f}s")


if __name__ == "__main__":
    main()