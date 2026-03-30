# -*- coding: utf-8 -*-
"""
Benchmark runner dedicated to mission 1: locating the global maximum.

This script is intentionally separate from the isopotential benchmark:
- it focuses only on source localisation
- it reuses the original eval_metrics baseline
- it adds spatial metrics that are specific to "find the source"
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import io
import json
import math
import sys
import time
import types
from itertools import permutations
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


def _install_scipy_fallback():
    """
    Provide the tiny scipy surface needed by this project when scipy is
    unavailable in the local Python environment.
    """
    try:
        import scipy.optimize  # noqa: F401
        import scipy.stats  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    class _FallbackMultivariateNormal:
        def __init__(self, mean, cov):
            self.mean = np.asarray(mean, dtype=float)
            self.cov = np.asarray(cov, dtype=float)
            self.inv_cov = np.linalg.inv(self.cov)
            det_cov = float(np.linalg.det(self.cov))
            dim = self.mean.shape[0]
            self.normalization = 1.0 / np.sqrt(((2.0 * np.pi) ** dim) * det_cov)

        def pdf(self, pos):
            positions = np.asarray(pos, dtype=float)
            delta = positions - self.mean
            exponent = np.einsum("...i,ij,...j->...", delta, self.inv_cov, delta)
            return self.normalization * np.exp(-0.5 * exponent)

    stats_module = types.ModuleType("scipy.stats")
    stats_module.multivariate_normal = (
        lambda mean, cov: _FallbackMultivariateNormal(mean, cov)
    )

    def _linear_sum_assignment(cost_matrix):
        cost = np.asarray(cost_matrix, dtype=float)
        if cost.ndim != 2:
            raise ValueError("cost_matrix must be 2-dimensional")

        n_rows, n_cols = cost.shape
        if n_rows > n_cols:
            raise ValueError("fallback linear_sum_assignment expects rows <= cols")

        best_perm = None
        best_cost = None

        for perm in permutations(range(n_cols), n_rows):
            total_cost = float(sum(cost[row_idx, col_idx] for row_idx, col_idx in enumerate(perm)))
            if (best_cost is None) or (total_cost < best_cost):
                best_cost = total_cost
                best_perm = perm

        return (
            np.arange(n_rows, dtype=int),
            np.asarray(best_perm, dtype=int),
        )

    optimize_module = types.ModuleType("scipy.optimize")
    optimize_module.linear_sum_assignment = _linear_sum_assignment

    scipy_module = types.ModuleType("scipy")
    scipy_module.stats = stats_module
    scipy_module.optimize = optimize_module

    sys.modules.setdefault("scipy", scipy_module)
    sys.modules["scipy.stats"] = stats_module
    sys.modules["scipy.optimize"] = optimize_module


_install_scipy_fallback()

from eval_metrics_max_search import eval_max_search_metrics
from lib.robot import Fleet
from lib.simulation import FleetSimulation, generate_init_positions


@dataclass(frozen=True)
class BenchmarkConfig:
    difficulties: tuple[int, ...] = (1, 2, 3)
    nb_robots_values: tuple[int, ...] = (3, 5)
    tsim: float = 60.0
    ts: float = 0.05
    dynamics: str = "singleIntegrator2D"
    init_mode: str = "grid"
    init_center_x: float = -20.0
    init_center_y: float = -20.0
    init_spacing: float = 1.0
    strict_radius: float = 1.0
    relaxed_radius: float = 2.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the maximum-search benchmark on a control module."
    )
    parser.add_argument(
        "--control-module",
        default="control_algo_potential_ldc",
        help="Python module exposing potential_seeking_ctrl(...).",
    )
    parser.add_argument(
        "--suite",
        choices=("deterministic", "monte_carlo", "both"),
        default="both",
        help="Benchmark suite to run.",
    )
    parser.add_argument(
        "--monte-carlo-count",
        type=int,
        default=12,
        help="Number of random-field seeds per (N, difficulty) in the Monte Carlo suite.",
    )
    parser.add_argument(
        "--deterministic-seed",
        type=int,
        default=0,
        help="Seed used for deterministic-field runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "max_search_benchmark",
        help="Base output directory for CSV and JSON exports.",
    )
    parser.add_argument(
        "--tsim",
        type=float,
        default=BenchmarkConfig.tsim,
        help="Simulation duration in seconds.",
    )
    parser.add_argument(
        "--quiet-controller",
        action="store_true",
        help="Silence controller prints during the benchmark.",
    )
    return parser.parse_args()


def ensure_output_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def build_init_positions(config: BenchmarkConfig, nb_robots: int):
    return generate_init_positions(
        nb_robots,
        mode=config.init_mode,
        center=(config.init_center_x, config.init_center_y),
        spacing=config.init_spacing,
    )


def load_control_module(module_name: str):
    module = importlib.import_module(module_name)
    return importlib.reload(module)


def quiet_controller_if_possible(control_algo, nb_robots: int):
    """
    Best-effort silencing for controllers that expose ad hoc verbosity globals.
    """
    if hasattr(control_algo, "verbose"):
        try:
            control_algo.verbose = [False for _ in range(max(nb_robots, 5))]
        except Exception:
            pass


def simulate_once(
    control_module_name: str,
    seed: int,
    difficulty: int,
    nb_robots: int,
    random_field: bool,
    config: BenchmarkConfig,
    quiet_controller: bool,
):
    np.random.seed(int(seed))
    control_algo = load_control_module(control_module_name)
    quiet_controller_if_possible(control_algo, nb_robots)

    if hasattr(control_algo, "reset_controller"):
        control_algo.reset_controller()

    init_positions = build_init_positions(config, nb_robots)
    fleet = Fleet(
        nb_robots,
        dynamics=config.dynamics,
        initStates=np.asarray(init_positions, dtype=float),
    )
    simulation = FleetSimulation(fleet, t0=0.0, tf=config.tsim, dt=config.ts)
    potential_measurements = np.zeros((simulation.t.shape[0], nb_robots), dtype=float)

    start_perf = time.perf_counter()
    pot = None

    def run_loop():
        nonlocal pot

        for step_idx, t_value in enumerate(simulation.t):
            robots_poses = fleet.getPosesArray()

            for robot_idx in range(fleet.nbOfRobots):
                vx, vy, pot = control_algo.potential_seeking_ctrl(
                    t_value,
                    robot_idx,
                    robots_poses,
                    difficulty=difficulty,
                    random=random_field,
                )
                fleet.robot[robot_idx].ctrl = np.array([vx, vy], dtype=float)

            current_measurements = np.asarray(pot.value(robots_poses[:, 0:2]), dtype=float)
            potential_measurements[step_idx, :] = current_measurements

            simulation.addDataFromFleet(fleet)
            fleet.integrateMotion(config.ts)

    if quiet_controller:
        with contextlib.redirect_stdout(io.StringIO()):
            run_loop()
    else:
        run_loop()

    elapsed = time.perf_counter() - start_perf

    row = {
        "control_module": control_module_name,
        "seed": int(seed),
        "difficulty": int(difficulty),
        "nb_robots": int(nb_robots),
        "random_field": int(random_field),
        "runtime_s": float(elapsed),
    }
    row.update(
        eval_max_search_metrics(
            simulation,
            potential_measurements,
            pot,
            strict_radius=config.strict_radius,
            relaxed_radius=config.relaxed_radius,
        )
    )
    return row


def run_suite(
    suite_name: str,
    config: BenchmarkConfig,
    control_module_name: str,
    seed_count: int,
    deterministic_seed: int,
    quiet_controller: bool,
):
    rows = []

    for nb_robots in config.nb_robots_values:
        for difficulty in config.difficulties:
            if suite_name == "deterministic":
                seed_values = [int(deterministic_seed)]
                random_field = False
            else:
                seed_values = list(range(int(seed_count)))
                random_field = True

            print(
                f"[{suite_name}] N={nb_robots}, difficulty={difficulty}: "
                f"{len(seed_values)} run(s)"
            )

            for seed in seed_values:
                row = simulate_once(
                    control_module_name=control_module_name,
                    seed=seed,
                    difficulty=difficulty,
                    nb_robots=nb_robots,
                    random_field=random_field,
                    config=config,
                    quiet_controller=quiet_controller,
                )
                row["suite"] = suite_name
                rows.append(row)

    return rows


def finite_values(rows, key):
    values = [float(row[key]) for row in rows]
    return np.asarray([value for value in values if math.isfinite(value)], dtype=float)


def median_or_nan(values):
    if values.size == 0:
        return math.nan
    return float(np.median(values))


def quantile_or_nan(values, quantile):
    if values.size == 0:
        return math.nan
    return float(np.quantile(values, quantile))


def summarise_rows(rows):
    summary_rows = []

    suites = sorted({row["suite"] for row in rows})
    for suite_name in suites:
        suite_rows = [row for row in rows if row["suite"] == suite_name]
        for nb_robots in sorted({int(row["nb_robots"]) for row in suite_rows}):
            robot_rows = [row for row in suite_rows if int(row["nb_robots"]) == nb_robots]
            for difficulty in sorted({int(row["difficulty"]) for row in robot_rows}):
                diff_rows = [
                    row
                    for row in robot_rows
                    if int(row["difficulty"]) == difficulty
                ]

                summary_rows.append(
                    {
                        "suite": suite_name,
                        "nb_robots": int(nb_robots),
                        "difficulty": int(difficulty),
                        "n_runs": len(diff_rows),
                        "hit_strict_rate": float(
                            np.mean([row["global_hit_strict"] for row in diff_rows])
                        ),
                        "hit_relaxed_rate": float(
                            np.mean([row["global_hit_relaxed"] for row in diff_rows])
                        ),
                        "distance_to_first_hit_strict_median": median_or_nan(
                            finite_values(diff_rows, "distance_to_first_hit_strict")
                        ),
                        "distance_to_first_hit_strict_p90": quantile_or_nan(
                            finite_values(diff_rows, "distance_to_first_hit_strict"),
                            0.9,
                        ),
                        "distance_to_first_hit_relaxed_median": median_or_nan(
                            finite_values(diff_rows, "distance_to_first_hit_relaxed")
                        ),
                        "best_source_distance_median": median_or_nan(
                            finite_values(diff_rows, "best_source_distance")
                        ),
                        "best_source_distance_p90": quantile_or_nan(
                            finite_values(diff_rows, "best_source_distance"),
                            0.9,
                        ),
                        "final_mean_distance_to_source_median": median_or_nan(
                            finite_values(diff_rows, "final_mean_distance_to_source")
                        ),
                        "final_group_dispersion_median": median_or_nan(
                            finite_values(diff_rows, "final_group_dispersion")
                        ),
                        "robots_ever_near_source_relaxed_median": median_or_nan(
                            finite_values(diff_rows, "robots_ever_near_source_relaxed")
                        ),
                        "total_distance_median": median_or_nan(
                            finite_values(diff_rows, "total_distance")
                        ),
                        "relative_pot_found_error_median": median_or_nan(
                            finite_values(diff_rows, "relative_pot_found_error")
                        ),
                        "min_inter_robot_distance_min": (
                            math.nan
                            if len(diff_rows) == 0
                            else float(
                                np.min(
                                    [float(row["min_inter_robot_distance"]) for row in diff_rows]
                                )
                            )
                        ),
                        "distance_workload_gini_median": median_or_nan(
                            finite_values(diff_rows, "distance_workload_gini")
                        ),
                        "runtime_median_s": median_or_nan(
                            finite_values(diff_rows, "runtime_s")
                        ),
                    }
                )

    return summary_rows


def write_csv(path: Path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def format_float(value, digits=3):
    if value is None or not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def print_summary(summary_rows):
    print("")
    print("=== MAX-SEARCH SUMMARY ===")
    print(
        "suite | N | diff | hit@1m | hit@2m | D_hit_med | d_best_med | "
        "d_final_med | dist_med"
    )
    for row in summary_rows:
        print(
            f"{row['suite']:>12} | "
            f"{row['nb_robots']:>1d} | "
            f"{row['difficulty']:>4d} | "
            f"{format_float(row['hit_strict_rate'], 3):>6} | "
            f"{format_float(row['hit_relaxed_rate'], 3):>6} | "
            f"{format_float(row['distance_to_first_hit_strict_median'], 2):>9} | "
            f"{format_float(row['best_source_distance_median'], 2):>10} | "
            f"{format_float(row['final_mean_distance_to_source_median'], 2):>11} | "
            f"{format_float(row['total_distance_median'], 2):>8}"
        )


def main():
    args = parse_args()
    config = BenchmarkConfig(tsim=float(args.tsim))
    control_module_slug = args.control_module.split(".")[-1]
    output_dir = args.output_dir / control_module_slug
    ensure_output_dir(output_dir)

    suite_rows = []
    metadata = {
        "config": asdict(config),
        "control_module": args.control_module,
        "suite": args.suite,
        "monte_carlo_count": int(args.monte_carlo_count),
        "deterministic_seed": int(args.deterministic_seed),
        "quiet_controller": bool(args.quiet_controller),
    }

    if args.suite in ("deterministic", "both"):
        suite_rows.extend(
            run_suite(
                suite_name="deterministic",
                config=config,
                control_module_name=args.control_module,
                seed_count=args.monte_carlo_count,
                deterministic_seed=args.deterministic_seed,
                quiet_controller=args.quiet_controller,
            )
        )

    if args.suite in ("monte_carlo", "both"):
        suite_rows.extend(
            run_suite(
                suite_name="monte_carlo",
                config=config,
                control_module_name=args.control_module,
                seed_count=args.monte_carlo_count,
                deterministic_seed=args.deterministic_seed,
                quiet_controller=args.quiet_controller,
            )
        )

    summary_rows = summarise_rows(suite_rows)

    write_csv(output_dir / "max_search_runs.csv", suite_rows)
    write_csv(output_dir / "max_search_summary.csv", summary_rows)
    write_json(
        output_dir / "max_search_summary.json",
        {
            "metadata": metadata,
            "summary": summary_rows,
        },
    )

    print_summary(summary_rows)
    print("")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
