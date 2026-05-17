#!/usr/bin/env python3
"""
temperature_variance_v2.py — Corrected temperature-variance harness.

Bug fixes vs v1 (experiments/temperature_variance.py):

  Bug 1: actual_spent_uc was integer-divided by 1_000_000 and floored
         to zero on every per-call cost. v2 uses float arithmetic for
         cost and rounds at the end.

  Bug 2: byte_length_estimate_uc returned raw bytes (with margin), but
         was compared to cap_uc as if it were micro-cents. v2 keeps
         everything in micro-cents: byte_length is multiplied by the
         per-token input rate to produce a real input-cost estimate.

  Bug 3: message history accumulated the full assistant response on
         every turn, which made the pre-flight estimate exceed the cap
         after exactly one call. v2 truncates history to (system +
         most-recent user turn) so multiple calls actually happen.

Design changes for proper variance signal:

  - max_output_tokens raised from 200 → 500. At T=0 the model emits
    far fewer than 500 tokens naturally, so we no longer artificially
    truncate to a constant.

  - Default caps recalibrated to allow ~3-5 calls per run before the
    estimator pre-flights a refusal. This gives variance in n_calls.

  - Workload is a refinement loop: "fix this; elaborate; elaborate;
    elaborate" mimicking agent retry behaviour. History is trimmed
    each turn so the prompt doesn't grow without bound.

Cost estimate at default config:
  ~3-5 calls per run × 80 runs = ~240-400 calls total
  At ~$0.003/claude call + ~$0.0001/openai call = ~$1-2 total

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  export OPENAI_API_KEY=sk-...
  python3 temperature_variance_v2.py \\
      --n 10 \\
      --temperatures 0.0 0.3 0.7 1.0 \\
      --out-csv temperature_variance_v2_results.csv \\
      --out-json temperature_variance_v2_summary.json

Smoke test first (8 runs, ~$0.02):
  python3 temperature_variance_v2.py --smoke
"""

import argparse
import csv
import json
import os
import sys
import time
import statistics
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional


# ----------------------------------------------------------------------
# Pricing — all values in micro-cents per token.
# 1 dollar = 100 cents = 1_000_000 micro-cents.
# Therefore $X per million tokens = X micro-cents per token.
# (Example: $3 / 1M tokens = 3 uc / token.)
# ----------------------------------------------------------------------

PRICING_UC_PER_TOKEN = {
    "claude-sonnet-4-5-20250929": {"input": 3.0,  "output": 15.0},   # $3 / $15 per MT
    "claude-sonnet-4-20250514":   {"input": 3.0,  "output": 15.0},
    "gpt-4o-mini-2024-07-18":     {"input": 0.15, "output": 0.60},   # $0.15 / $0.60 per MT
    "gpt-4o-2024-08-06":          {"input": 2.50, "output": 10.0},   # $2.50 / $10 per MT
}


def rates_for(model: str) -> Dict[str, float]:
    if model in PRICING_UC_PER_TOKEN:
        return PRICING_UC_PER_TOKEN[model]
    # Best-effort fallback by substring
    for k, v in PRICING_UC_PER_TOKEN.items():
        if k.split("-")[0] in model.lower():
            return v
    return {"input": 1.0, "output": 3.0}


# ----------------------------------------------------------------------
# Caps — calibrated to allow ~3-5 calls before the cap fires.
# These are real micro-cents (consistent units throughout).
# ----------------------------------------------------------------------

DEFAULT_CAPS_UC = {
    "claude-sonnet-4-5-20250929": 30_000,   # ~3-5 sonnet calls @ ~3-5k uc each
    "claude-sonnet-4-20250514":   30_000,
    "gpt-4o-mini-2024-07-18":     1_500,    # ~3-5 mini calls @ ~300 uc each
    "gpt-4o-2024-08-06":          25_000,   # ~3-5 gpt-4o calls @ ~5-7k uc each
}


# ----------------------------------------------------------------------
# Workload — refinement loop with bounded history
# ----------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a debugging assistant. The user describes a Python "
    "ImportError. Propose a concise fix. If the user replies "
    "'elaborate', expand your previous answer with one additional "
    "concrete suggestion. Be concise: at most 4 sentences."
)

INITIAL_USER_MSG = (
    "fix this: ImportError: cannot import name 'foo' from 'mypkg'"
)

ELABORATE_USER_MSG = "elaborate"

MAX_RETRIES = 8                   # generous upper bound on calls per run
MAX_OUTPUT_TOKENS = 500           # loose; temperature can vary natural length
SAFETY_MARGIN_ANTHROPIC = 2.0     # matches AnthropicEstimator default


@dataclass
class Run:
    model: str
    temperature: float
    cap_uc: int
    run_index: int
    outcome: str                  # "completed_within_cap" | "mid_loop_fired" | "pre_flight_refused" | "max_retries_exhausted" | "error"
    n_calls_made: int
    total_input_tokens: int
    total_output_tokens: int
    actual_spent_uc: float        # FLOAT to avoid integer-divide-to-zero bug
    pre_flight_estimate_uc: float # what the estimator predicted just before cap fired
    error_msg: Optional[str] = None


# ----------------------------------------------------------------------
# Estimator (consistent micro-cent units throughout)
# ----------------------------------------------------------------------

def estimate_call_uc(prompt_text: str, model: str, max_output_tokens: int) -> float:
    """
    Pre-flight cost estimate for one LLM call, in micro-cents.
    Uses byte-length as a conservative input-token estimate (bytes ≥
    BPE tokens for any byte-level tokenizer), then multiplies by per-
    token rate. Includes a 2.0× margin for Anthropic models, matching
    AnthropicEstimator's default.
    """
    rates = rates_for(model)
    margin = SAFETY_MARGIN_ANTHROPIC if "claude" in model.lower() else 1.0
    input_token_est = len(prompt_text.encode("utf-8")) * margin
    input_cost_uc   = input_token_est * rates["input"]
    output_cost_uc  = max_output_tokens * rates["output"]
    return input_cost_uc + output_cost_uc


def actual_call_uc(input_tokens: int, output_tokens: int, model: str) -> float:
    rates = rates_for(model)
    return input_tokens * rates["input"] + output_tokens * rates["output"]


# ----------------------------------------------------------------------
# Anthropic runner
# ----------------------------------------------------------------------

def run_anthropic(model: str, temperature: float, cap_uc: int,
                  run_index: int) -> Run:
    try:
        import anthropic
    except ImportError:
        return Run(model=model, temperature=temperature, cap_uc=cap_uc,
                   run_index=run_index, outcome="error",
                   n_calls_made=0, total_input_tokens=0,
                   total_output_tokens=0, actual_spent_uc=0.0,
                   pre_flight_estimate_uc=0.0,
                   error_msg="anthropic library not installed")

    client = anthropic.Anthropic()
    spent_uc = 0.0
    n_calls = 0
    total_input = 0
    total_output = 0
    last_assistant_text: Optional[str] = None
    pre_flight_estimate_at_fire = 0.0

    for call_index in range(MAX_RETRIES):
        # Build messages for this turn — bounded history (last assistant + new user)
        if call_index == 0:
            user_msg = INITIAL_USER_MSG
            messages = [{"role": "user", "content": user_msg}]
        else:
            user_msg = ELABORATE_USER_MSG
            messages = [
                {"role": "user", "content": INITIAL_USER_MSG},
                {"role": "assistant", "content": last_assistant_text or ""},
                {"role": "user", "content": user_msg},
            ]

        # Pre-flight check — what would this call cost if it ran?
        full_prompt = SYSTEM_PROMPT + json.dumps(messages)
        est = estimate_call_uc(full_prompt, model, MAX_OUTPUT_TOKENS)
        if spent_uc + est > cap_uc:
            outcome = "pre_flight_refused" if n_calls == 0 else "mid_loop_fired"
            return Run(model=model, temperature=temperature, cap_uc=cap_uc,
                       run_index=run_index, outcome=outcome,
                       n_calls_made=n_calls,
                       total_input_tokens=total_input,
                       total_output_tokens=total_output,
                       actual_spent_uc=spent_uc,
                       pre_flight_estimate_uc=est)

        # Make the call
        try:
            response = client.messages.create(
                model=model,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=temperature,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
            n_calls += 1
            in_t  = response.usage.input_tokens
            out_t = response.usage.output_tokens
            total_input  += in_t
            total_output += out_t
            spent_uc += actual_call_uc(in_t, out_t, model)
            last_assistant_text = (
                response.content[0].text if response.content else ""
            )

            if response.stop_reason == "end_turn" and call_index >= 2:
                # Model converged on a final answer after at least 3 turns;
                # legitimate self-termination.
                return Run(model=model, temperature=temperature, cap_uc=cap_uc,
                           run_index=run_index, outcome="completed_within_cap",
                           n_calls_made=n_calls,
                           total_input_tokens=total_input,
                           total_output_tokens=total_output,
                           actual_spent_uc=spent_uc,
                           pre_flight_estimate_uc=est)

        except Exception as e:
            return Run(model=model, temperature=temperature, cap_uc=cap_uc,
                       run_index=run_index, outcome="error",
                       n_calls_made=n_calls,
                       total_input_tokens=total_input,
                       total_output_tokens=total_output,
                       actual_spent_uc=spent_uc,
                       pre_flight_estimate_uc=0.0,
                       error_msg=str(e))

    # Exhausted MAX_RETRIES without cap firing — should be rare with these caps
    return Run(model=model, temperature=temperature, cap_uc=cap_uc,
               run_index=run_index, outcome="max_retries_exhausted",
               n_calls_made=n_calls,
               total_input_tokens=total_input,
               total_output_tokens=total_output,
               actual_spent_uc=spent_uc,
               pre_flight_estimate_uc=0.0)


# ----------------------------------------------------------------------
# OpenAI runner — same shape as Anthropic, GPT-4o family
# ----------------------------------------------------------------------

def run_openai(model: str, temperature: float, cap_uc: int,
               run_index: int) -> Run:
    try:
        from openai import OpenAI
    except ImportError:
        return Run(model=model, temperature=temperature, cap_uc=cap_uc,
                   run_index=run_index, outcome="error",
                   n_calls_made=0, total_input_tokens=0,
                   total_output_tokens=0, actual_spent_uc=0.0,
                   pre_flight_estimate_uc=0.0,
                   error_msg="openai library not installed")

    client = OpenAI()
    spent_uc = 0.0
    n_calls = 0
    total_input = 0
    total_output = 0
    last_assistant_text: Optional[str] = None

    for call_index in range(MAX_RETRIES):
        if call_index == 0:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": INITIAL_USER_MSG},
            ]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": INITIAL_USER_MSG},
                {"role": "assistant", "content": last_assistant_text or ""},
                {"role": "user", "content": ELABORATE_USER_MSG},
            ]

        full_prompt = json.dumps(messages)
        est = estimate_call_uc(full_prompt, model, MAX_OUTPUT_TOKENS)
        if spent_uc + est > cap_uc:
            outcome = "pre_flight_refused" if n_calls == 0 else "mid_loop_fired"
            return Run(model=model, temperature=temperature, cap_uc=cap_uc,
                       run_index=run_index, outcome=outcome,
                       n_calls_made=n_calls,
                       total_input_tokens=total_input,
                       total_output_tokens=total_output,
                       actual_spent_uc=spent_uc,
                       pre_flight_estimate_uc=est)

        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=temperature,
                messages=messages,
            )
            n_calls += 1
            in_t  = response.usage.prompt_tokens
            out_t = response.usage.completion_tokens
            total_input  += in_t
            total_output += out_t
            spent_uc += actual_call_uc(in_t, out_t, model)
            last_assistant_text = response.choices[0].message.content

            if (response.choices[0].finish_reason == "stop"
                    and call_index >= 2):
                return Run(model=model, temperature=temperature, cap_uc=cap_uc,
                           run_index=run_index, outcome="completed_within_cap",
                           n_calls_made=n_calls,
                           total_input_tokens=total_input,
                           total_output_tokens=total_output,
                           actual_spent_uc=spent_uc,
                           pre_flight_estimate_uc=est)

        except Exception as e:
            return Run(model=model, temperature=temperature, cap_uc=cap_uc,
                       run_index=run_index, outcome="error",
                       n_calls_made=n_calls,
                       total_input_tokens=total_input,
                       total_output_tokens=total_output,
                       actual_spent_uc=spent_uc,
                       pre_flight_estimate_uc=0.0,
                       error_msg=str(e))

    return Run(model=model, temperature=temperature, cap_uc=cap_uc,
               run_index=run_index, outcome="max_retries_exhausted",
               n_calls_made=n_calls,
               total_input_tokens=total_input,
               total_output_tokens=total_output,
               actual_spent_uc=spent_uc,
               pre_flight_estimate_uc=0.0)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10,
                        help="runs per cell")
    parser.add_argument("--temperatures", nargs="+", type=float,
                        default=[0.0, 0.3, 0.7, 1.0])
    parser.add_argument("--models", nargs="+",
                        default=list(DEFAULT_CAPS_UC.keys())[:2])  # default to first 2 (claude-sonnet, claude-sonnet-4)
    parser.add_argument("--cap-override", type=int, default=None,
                        help="override per-model cap (uc); applies to all models")
    parser.add_argument("--smoke", action="store_true",
                        help="N=1 quick test (~$0.02)")
    parser.add_argument("--out-csv", default="temperature_variance_v2_results.csv")
    parser.add_argument("--out-json", default="temperature_variance_v2_summary.json")
    args = parser.parse_args()

    if args.smoke:
        args.n = 1
        # smoke test runs a single claude + single openai model
        args.models = ["claude-sonnet-4-5-20250929", "gpt-4o-mini-2024-07-18"]

    # Pick default models from the user's --models (claude-sonnet + gpt-4o-mini)
    if not args.models:
        args.models = ["claude-sonnet-4-5-20250929", "gpt-4o-mini-2024-07-18"]

    all_runs: List[Run] = []
    start = time.time()
    estimated_spend_running = 0.0

    for model in args.models:
        cap = args.cap_override or DEFAULT_CAPS_UC.get(model, 5000)
        runner = run_anthropic if "claude" in model.lower() else run_openai
        print(f"\n=== {model} (cap_uc={cap}) ===")
        for temp in args.temperatures:
            for i in range(args.n):
                run = runner(model, temp, cap, i)
                all_runs.append(run)
                estimated_spend_running += run.actual_spent_uc / 1_000_000  # uc → dollars
                print(
                    f"  T={temp:.1f} run {i+1}/{args.n}: "
                    f"outcome={run.outcome:<22s} "
                    f"calls={run.n_calls_made} "
                    f"in_tok={run.total_input_tokens} "
                    f"out_tok={run.total_output_tokens} "
                    f"spent_uc={run.actual_spent_uc:.1f} "
                    f"(running $: {estimated_spend_running:.4f})"
                )

    # Write CSV
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(all_runs[0]).keys()))
        writer.writeheader()
        for r in all_runs:
            writer.writerow(asdict(r))

    # Summary JSON with proper statistics
    summary: Dict = {}
    cell_groups: Dict[tuple, List[Run]] = {}
    for r in all_runs:
        cell_groups.setdefault((r.model, r.temperature, r.cap_uc), []).append(r)

    for (model, temp, cap), runs in cell_groups.items():
        key = f"{model}@T={temp}@cap={cap}"
        outcomes: Dict[str, int] = {}
        for r in runs:
            outcomes[r.outcome] = outcomes.get(r.outcome, 0) + 1
        spent = [r.actual_spent_uc for r in runs]
        n_calls = [r.n_calls_made for r in runs]
        out_tokens = [r.total_output_tokens for r in runs]
        cap_violations = sum(1 for r in runs if r.actual_spent_uc > r.cap_uc)
        summary[key] = {
            "n_total": len(runs),
            "outcomes": outcomes,
            "mean_spent_uc": statistics.mean(spent) if spent else 0.0,
            "stdev_spent_uc": statistics.stdev(spent) if len(spent) > 1 else 0.0,
            "max_spent_uc": max(spent) if spent else 0.0,
            "mean_n_calls": statistics.mean(n_calls) if n_calls else 0.0,
            "stdev_n_calls": statistics.stdev(n_calls) if len(n_calls) > 1 else 0.0,
            "mean_output_tokens": statistics.mean(out_tokens) if out_tokens else 0.0,
            "stdev_output_tokens": statistics.stdev(out_tokens) if len(out_tokens) > 1 else 0.0,
            "cap_violations": cap_violations,
        }

    summary["__totals__"] = {
        "n_total": len(all_runs),
        "wall_time_sec": time.time() - start,
        "total_cap_violations": sum(1 for r in all_runs if r.actual_spent_uc > r.cap_uc),
        "estimated_cost_usd": sum(r.actual_spent_uc for r in all_runs) / 1_000_000,
    }

    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    print(f"Total runs:              {len(all_runs)}")
    print(f"Total cap violations:    {summary['__totals__']['total_cap_violations']}")
    print(f"Estimated total spend:   ${summary['__totals__']['estimated_cost_usd']:.4f}")
    print(f"Wall time:               {summary['__totals__']['wall_time_sec']:.1f}s")
    print(f"CSV:                     {args.out_csv}")
    print(f"Summary:                 {args.out_json}")


if __name__ == "__main__":
    main()