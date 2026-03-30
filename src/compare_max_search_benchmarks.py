# -*- coding: utf-8 -*-
"""
Statistical comparison helper for mission-1 benchmark outputs.

This script compares two benchmark folders produced by
`run_max_search_benchmark.py` using paired Monte Carlo runs
(same seeds, same N, same difficulty).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two max-search benchmark result folders."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "max_search_benchmark",
        help="Base benchmark output directory.",
    )
    parser.add_argument(
        "--algo-a",
        default="control_algo_potential_ldc",
        help="First algorithm folder name.",
    )
    parser.add_argument(
        "--algo-b",
        default="control_algo_potential",
        help="Second algorithm folder name.",
    )
    return parser.parse_args()


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(row, key):
    value = row[key]
    if value == "" or value.lower() == "nan":
        return math.nan
    return float(value)


def finite(values):
    return [value for value in values if math.isfinite(value)]


def median(values):
    values = sorted(finite(values))
    if not values:
        return math.nan
    n = len(values)
    if n % 2 == 1:
        return float(values[n // 2])
    return float(0.5 * (values[n // 2 - 1] + values[n // 2]))


def quantile(values, q):
    values = sorted(finite(values))
    if not values:
        return math.nan
    if len(values) == 1:
        return float(values[0])
    pos = q * (len(values) - 1)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return float(values[low])
    weight = pos - low
    return float((1.0 - weight) * values[low] + weight * values[high])


def sign_test_pvalue(wins_a, wins_b):
    n = wins_a + wins_b
    if n == 0:
        return math.nan
    k = min(wins_a, wins_b)
    cumulative = 0.0
    for i in range(k + 1):
        cumulative += math.comb(n, i)
    p_value = 2.0 * cumulative / (2.0 ** n)
    return float(min(1.0, p_value))


def build_index(rows):
    indexed = {}
    for row in rows:
        key = (
            row["suite"],
            int(row["nb_robots"]),
            int(row["difficulty"]),
            int(row["seed"]),
        )
        indexed[key] = row
    return indexed


def compare_metric(rows_a, rows_b, metric, lower_is_better=True):
    wins_a = 0
    wins_b = 0
    ties = 0
    deltas = []

    for row_a, row_b in zip(rows_a, rows_b):
        value_a = to_float(row_a, metric)
        value_b = to_float(row_b, metric)

        if not (math.isfinite(value_a) and math.isfinite(value_b)):
            continue

        delta = value_b - value_a
        deltas.append(delta)

        if abs(delta) <= 1e-12:
            ties += 1
            continue

        if lower_is_better:
            if value_a < value_b:
                wins_a += 1
            else:
                wins_b += 1
        else:
            if value_a > value_b:
                wins_a += 1
            else:
                wins_b += 1

    return {
        "metric": metric,
        "n_pairs": wins_a + wins_b + ties,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "median_delta_b_minus_a": median(deltas),
        "q10_delta_b_minus_a": quantile(deltas, 0.1),
        "q90_delta_b_minus_a": quantile(deltas, 0.9),
        "sign_test_pvalue": sign_test_pvalue(wins_a, wins_b),
    }


def main():
    args = parse_args()
    runs_a = read_csv(args.output_dir / args.algo_a / "max_search_runs.csv")
    runs_b = read_csv(args.output_dir / args.algo_b / "max_search_runs.csv")

    index_a = build_index(runs_a)
    index_b = build_index(runs_b)

    common_keys = sorted(set(index_a.keys()) & set(index_b.keys()))
    paired_rows_a = [index_a[key] for key in common_keys]
    paired_rows_b = [index_b[key] for key in common_keys]

    monte_keys = [key for key in common_keys if key[0] == "monte_carlo"]
    monte_rows_a = [index_a[key] for key in monte_keys]
    monte_rows_b = [index_b[key] for key in monte_keys]

    metrics = [
        ("distance_to_first_hit_strict", True),
        ("total_distance", True),
        ("best_source_distance", True),
        ("final_mean_distance_to_source", True),
        ("min_inter_robot_distance", False),
        ("distance_workload_gini", True),
    ]

    overall = [
        compare_metric(monte_rows_a, monte_rows_b, metric, lower_is_better=lower_is_better)
        for metric, lower_is_better in metrics
    ]

    per_configuration = []
    for nb_robots in (3, 5):
        for difficulty in (1, 2, 3):
            cfg_rows_a = [
                row for key, row in zip(monte_keys, monte_rows_a)
                if key[1] == nb_robots and key[2] == difficulty
            ]
            cfg_rows_b = [
                row for key, row in zip(monte_keys, monte_rows_b)
                if key[1] == nb_robots and key[2] == difficulty
            ]
            per_configuration.append(
                {
                    "nb_robots": nb_robots,
                    "difficulty": difficulty,
                    "metrics": [
                        compare_metric(
                            cfg_rows_a,
                            cfg_rows_b,
                            metric,
                            lower_is_better=lower_is_better,
                        )
                        for metric, lower_is_better in metrics
                    ],
                }
            )

    payload = {
        "algo_a": args.algo_a,
        "algo_b": args.algo_b,
        "n_common_runs": len(common_keys),
        "n_common_monte_carlo_runs": len(monte_keys),
        "overall_monte_carlo": overall,
        "per_configuration_monte_carlo": per_configuration,
    }

    output_path = args.output_dir / f"compare_{args.algo_a}_vs_{args.algo_b}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Comparison written to: {output_path}")


if __name__ == "__main__":
    main()
