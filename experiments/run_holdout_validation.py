#!/usr/bin/env python3
"""
run_holdout_validation.py — Validate AnthropicEstimator A1 on the
pre-registered holdout corpus (experiment #6).

For each prompt in holdout_corpus.json:
  - byte_len = len(prompt.encode("utf-8"))
  - estimate = byte_len * 2.0    (AnthropicEstimator's margin)
  - actual   = Anthropic count_tokens API result
  - ratio    = estimate / actual
  - a1_holds = ratio >= 1.0

Pre-registered pass criterion: >=95/100 prompts satisfy A1.

Output: holdout_results.csv

Cost: ~$0.05 (100 count_tokens calls — these are free or near-free
on Anthropic's API as of 2026).

Usage:
  export ANTHROPIC_API_KEY=sk-...
  python3 run_holdout_validation.py
"""

import csv
import json
import os
import statistics
import sys


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: ANTHROPIC_API_KEY not set")
    try:
        import anthropic
    except ImportError:
        sys.exit("ERROR: anthropic package not installed. pip install anthropic")
    if not os.path.exists("holdout_corpus.json"):
        sys.exit("ERROR: holdout_corpus.json not found. "
                 "Run generate_holdout_corpus.py first.")

    client = anthropic.Anthropic()
    with open("holdout_corpus.json") as f:
        corpus = json.load(f)

    print(f"Running A1 validation on {len(corpus)} prompts...")
    results = []
    for item in corpus:
        prompt = item["prompt"]
        byte_len = len(prompt.encode("utf-8"))
        estimate = byte_len * 2.0
        try:
            resp = client.messages.count_tokens(
                model="claude-haiku-4-5-20251001",
                messages=[{"role": "user", "content": prompt}],
            )
            actual = resp.input_tokens
        except Exception as e:
            print(f"  prompt {item['idx']}: ERROR {e}", file=sys.stderr)
            actual = -1
        ratio = (estimate / actual) if actual > 0 else None
        results.append({
            "idx": item["idx"],
            "byte_len": byte_len,
            "estimate": estimate,
            "actual": actual,
            "ratio": ratio if ratio is not None else "ERROR",
            "a1_holds": (ratio is not None and ratio >= 1.0),
        })
        status = "OK" if results[-1]["a1_holds"] else "FAIL"
        print(f"  {item['idx']+1:3d}/{len(corpus)}: "
              f"byte_len={byte_len:5d} est={estimate:7.0f} "
              f"actual={actual:5d} ratio={ratio:.3f if ratio else 'ERR'} "
              f"[{status}]" if ratio else
              f"  {item['idx']+1:3d}/{len(corpus)}: byte_len={byte_len} "
              f"actual=ERROR")

    # Write CSV
    with open("holdout_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            w.writerow(r)

    # Summary
    n_holds = sum(1 for r in results if r["a1_holds"])
    n_total = len(results)
    valid_ratios = [r["ratio"] for r in results
                    if isinstance(r["ratio"], (int, float))]
    fails = [r for r in results if not r["a1_holds"]
             and isinstance(r["ratio"], (int, float))]

    print(f"\n=== A1 Hold-out Validation Result ===")
    print(f"A1 holds:           {n_holds}/{n_total}")
    print(f"Pass criterion:     >=95/{n_total}")
    print(f"Outcome:            {'PASS' if n_holds >= 95 else 'FAIL'}")
    if valid_ratios:
        print(f"Mean ratio:         {statistics.mean(valid_ratios):.3f}")
        print(f"Median ratio:       {statistics.median(valid_ratios):.3f}")
        print(f"Min ratio:          {min(valid_ratios):.3f}")
        print(f"Max ratio:          {max(valid_ratios):.3f}")
    if fails:
        print(f"\nFailing prompts (ratio < 1.0):")
        for r in fails:
            print(f"  idx={r['idx']:3d}  byte_len={r['byte_len']:5d}  "
                  f"actual={r['actual']:5d}  ratio={r['ratio']:.3f}")
    print(f"\nFull per-prompt results: holdout_results.csv")


if __name__ == "__main__":
    main()