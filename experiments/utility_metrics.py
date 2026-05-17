"""
utility_metrics.py — Analyze existing experiment CSVs for utility metrics.

Computes for each (model, workload, cap) cell:

  1. Capital efficiency: (mean_actual_spent / cap) — how much of the cap
     is actually used. Low values indicate over-reservation.

  2. False rejection rate: # runs where the discipline refused a call
     that, in retrospect, would have fit within cap.

  3. Task completion rate: # runs where the agent completed its work
     within cap.

  4. Over-reservation ratio: (mean_reserved / mean_actual_spent) — how
     much the conservative estimator inflates the reserved amount above
     the actual billed cost.

Usage:
    python3 utility_metrics.py \
        --boundary-csv data/boundary_runs.csv \
        --production-csv data/production_tier_validation_results.csv \
        --cap-sweep-csv data/production_tier_cap_sweep_results.csv

The script expects CSVs with columns: model, workload, cap_uc, outcome,
actual_spent_uc (optional), reserved_uc (optional). Adjust column names
below if your CSVs differ.
"""

import argparse
import csv
import os
import statistics
from collections import defaultdict
from typing import Dict, List, Optional


def read_csv(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_cell_metrics(rows: List[Dict[str, str]],
                         cell_key=("model", "workload", "cap_uc"),
                         outcome_col="outcome",
                         spent_col="actual_spent_uc",
                         reserved_col="reserved_uc") -> Dict:
    """Group rows by cell_key and compute utility metrics per cell."""
    by_cell = defaultdict(list)
    for r in rows:
        key = tuple(r.get(k, "") for k in cell_key)
        by_cell[key].append(r)

    metrics = {}
    for key, cell_rows in by_cell.items():
        n_total = len(cell_rows)
        cap_uc = float(key[2]) if key[2] else 0.0

        # Outcome buckets — outcomes vary by harness; tolerate different naming
        n_completed = 0
        n_refused_preflight = 0
        n_mid_loop_fired = 0
        n_error = 0
        spent_amounts = []
        reserved_amounts = []

        for r in cell_rows:
            outcome = r.get(outcome_col, "").lower().strip()
            if outcome in {"completed", "completed_within_cap", "ok"}:
                n_completed += 1
            elif outcome in {"pre_flight_refused", "preflight_refused", "refused_preflight"}:
                n_refused_preflight += 1
            elif outcome in {"mid_loop_fired", "mid_loop", "midloop"}:
                n_mid_loop_fired += 1
            elif outcome in {"error", "http_error", "429"}:
                n_error += 1

            spent_str = r.get(spent_col, "0")
            try:
                spent_amounts.append(float(spent_str) if spent_str else 0.0)
            except ValueError:
                pass

            reserved_str = r.get(reserved_col, "")
            try:
                if reserved_str:
                    reserved_amounts.append(float(reserved_str))
            except ValueError:
                pass

        mean_spent = statistics.mean(spent_amounts) if spent_amounts else 0.0
        mean_reserved = statistics.mean(reserved_amounts) if reserved_amounts else None

        # Capital efficiency: how much of the cap is actually used (committed)?
        capital_eff = mean_spent / cap_uc if cap_uc > 0 else 0.0

        # Over-reservation: how much larger is reserved vs actual?
        over_reservation = (mean_reserved / mean_spent
                            if mean_reserved and mean_spent > 0 else None)

        # Task completion rate
        completion_rate = (n_completed / n_total) if n_total > 0 else 0.0

        # False-rejection rate proxy:
        # If pre-flight refused but cap > mean_spent, it would have fit.
        # We can't know per-run whether a refused call would have fit
        # without simulating; report the bound as: refusals where the
        # cell's MEAN spent < cap.
        if mean_spent > 0 and mean_spent < cap_uc:
            potential_false_rejections = n_refused_preflight
        else:
            potential_false_rejections = 0
        false_rej_rate = (potential_false_rejections / n_total) if n_total > 0 else 0.0

        metrics[key] = {
            "model": key[0],
            "workload": key[1],
            "cap_uc": cap_uc,
            "n_total": n_total,
            "n_completed": n_completed,
            "n_refused_preflight": n_refused_preflight,
            "n_mid_loop_fired": n_mid_loop_fired,
            "n_error": n_error,
            "mean_spent_uc": mean_spent,
            "mean_reserved_uc": mean_reserved,
            "capital_efficiency": capital_eff,
            "over_reservation_x": over_reservation,
            "completion_rate": completion_rate,
            "potential_false_rejection_rate": false_rej_rate,
        }
    return metrics


def print_summary(metrics: Dict):
    print(f"\n{'Cell (model/workload/cap)':<55} {'N':<4} {'Compl':<6} {'PreRef':<7} "
          f"{'MidLp':<6} {'CapEff':<8} {'OverRes':<8} {'CompRate':<9} {'FalseRefRate':<13}")
    print("-" * 130)
    for key, m in sorted(metrics.items()):
        cell_desc = f"{m['model']} / {m['workload']} / {m['cap_uc']:.0f}"[:54]
        over_res_str = f"{m['over_reservation_x']:.2f}x" if m['over_reservation_x'] else "n/a"
        print(f"{cell_desc:<55} {m['n_total']:<4} {m['n_completed']:<6} "
              f"{m['n_refused_preflight']:<7} {m['n_mid_loop_fired']:<6} "
              f"{m['capital_efficiency']:<8.3f} {over_res_str:<8} "
              f"{m['completion_rate']:<9.3f} {m['potential_false_rejection_rate']:<13.3f}")


def print_aggregate(all_metrics: List[Dict]):
    """Aggregate across all cells: useful for paper headline statistics."""
    all_capital_eff = [m['capital_efficiency']
                       for cell_metrics in all_metrics
                       for m in cell_metrics.values()
                       if m['capital_efficiency'] > 0]
    all_over_res = [m['over_reservation_x']
                    for cell_metrics in all_metrics
                    for m in cell_metrics.values()
                    if m['over_reservation_x']]
    all_completion = [m['completion_rate']
                      for cell_metrics in all_metrics
                      for m in cell_metrics.values()]
    all_false_rej = [m['potential_false_rejection_rate']
                     for cell_metrics in all_metrics
                     for m in cell_metrics.values()]

    print("\n========== AGGREGATE ACROSS ALL CELLS ==========")
    if all_capital_eff:
        print(f"Capital efficiency (mean): {statistics.mean(all_capital_eff):.3f}")
        print(f"Capital efficiency (median): {statistics.median(all_capital_eff):.3f}")
    if all_over_res:
        print(f"Over-reservation (mean): {statistics.mean(all_over_res):.2f}x")
        print(f"Over-reservation (median): {statistics.median(all_over_res):.2f}x")
        print(f"Over-reservation (max): {max(all_over_res):.2f}x")
    if all_completion:
        print(f"Task completion rate (mean): {statistics.mean(all_completion):.3f}")
    if all_false_rej:
        print(f"Potential false-rejection rate (mean): {statistics.mean(all_false_rej):.3f}")
        print(f"Potential false-rejection rate (max): {max(all_false_rej):.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", action="append", required=True,
                        help="One or more CSV paths; pass --csv FILE multiple times")
    parser.add_argument("--cell-key", nargs="+",
                        default=["model", "workload", "cap_uc"],
                        help="Column names to group by")
    parser.add_argument("--outcome-col", default="outcome")
    parser.add_argument("--spent-col", default="actual_spent_uc")
    parser.add_argument("--reserved-col", default="reserved_uc")
    args = parser.parse_args()

    all_metrics = []
    for path in args.csv:
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            continue
        rows = read_csv(path)
        metrics = compute_cell_metrics(
            rows,
            cell_key=tuple(args.cell_key),
            outcome_col=args.outcome_col,
            spent_col=args.spent_col,
            reserved_col=args.reserved_col,
        )
        print(f"\n========== {path} ==========")
        print_summary(metrics)
        all_metrics.append(metrics)

    print_aggregate(all_metrics)


if __name__ == "__main__":
    main()