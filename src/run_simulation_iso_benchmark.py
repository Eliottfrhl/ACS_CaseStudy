# -*- coding: utf-8 -*-
"""
Headless benchmark runner for the isopotential controller.

The benchmark is tailored for the report:
- five robots
- random fields
- all difficulties
- fixed iso-level target
- CSV/JSON exports
- representative trajectory figures
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

import control_algo_isopotential as control_algo
from lib.robot import Fleet
from lib.simulation import FleetSimulation, generate_init_positions


@dataclass(frozen=True)
class BenchmarkConfig:
    nb_robots: int = 5
    tsim: float = 30.0
    ts: float = 0.05
    iso_level: float = 260.0
    difficulties: tuple[int, ...] = (1, 2, 3)
    init_mode: str = "grid"
    init_center_x: float = -20.0
    init_center_y: float = -20.0
    init_spacing: float = 1.0
    dynamics: str = "singleIntegrator2D"
    random: bool = True
    near_band: float = 5.0
    success_min_robots: int = 3
    success_fraction_threshold: float = 0.5
    lock_hold_seconds: float = 1.0
    coverage_distance: float = 1.5
    shape_tsim: float = 60.0
    shape_distance_budgets: tuple[float, ...] = (100.0, 150.0)
    shape_target_coverages: tuple[float, ...] = (0.5, 0.8)
    contour_grid_size: int = 401
    figure_grid_size: int = 201


COLOR_LIST = ["tab:red", "tab:green", "tab:blue", "tab:orange", "tab:purple"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark the isopotential controller on all random difficulties."
    )
    parser.add_argument(
        "--suite",
        choices=("preflight", "main", "shape", "both", "all"),
        default="both",
        help="Benchmark suite to run.",
    )
    parser.add_argument(
        "--preflight-count",
        type=int,
        default=10,
        help="Number of seeds for the preflight suite.",
    )
    parser.add_argument(
        "--main-count",
        type=int,
        default=100,
        help="Number of seeds for the main suite.",
    )
    parser.add_argument(
        "--shape-count",
        type=int,
        default=100,
        help="Number of seeds for the shape suite.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "isopotential_benchmark",
        help="Directory used for CSV, JSON and figure exports.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip representative trajectory figures.",
    )
    return parser.parse_args()


def ensure_output_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def build_init_positions(config: BenchmarkConfig):
    return generate_init_positions(
        config.nb_robots,
        mode=config.init_mode,
        center=(config.init_center_x, config.init_center_y),
        spacing=config.init_spacing,
    )


def simulate_once(seed: int, difficulty: int, config: BenchmarkConfig):
    np.random.seed(seed)

    init_positions = build_init_positions(config)
    fleet = Fleet(
        config.nb_robots,
        dynamics=config.dynamics,
        initStates=np.asarray(init_positions, dtype=float),
    )
    simulation = FleetSimulation(fleet, t0=0.0, tf=config.tsim, dt=config.ts)
    potential_measurements = np.zeros((simulation.t.shape[0], config.nb_robots), dtype=float)

    control_algo.reset_controller({"iso_level": config.iso_level})

    start_perf = time.perf_counter()
    pot = None
    for step_idx, t_value in enumerate(simulation.t):
        robots_poses = fleet.getPosesArray()
        for robot_idx in range(fleet.nbOfRobots):
            vx, vy, pot = control_algo.potential_seeking_ctrl(
                t_value,
                robot_idx,
                robots_poses,
                difficulty=difficulty,
                random=config.random,
            )
            fleet.robot[robot_idx].ctrl = np.array([vx, vy], dtype=float)

        current_measurements = np.asarray(pot.value(robots_poses[:, 0:2]), dtype=float)
        potential_measurements[step_idx, :] = current_measurements
        simulation.addDataFromFleet(fleet)
        fleet.integrateMotion(config.ts)

    elapsed = time.perf_counter() - start_perf
    diagnostics = control_algo.get_controller_diagnostics()

    return {
        "simulation": simulation,
        "potential_measurements": potential_measurements,
        "pot": pot,
        "diagnostics": diagnostics,
        "init_positions": np.asarray(init_positions, dtype=float),
        "runtime_s": float(elapsed),
    }


def extract_trajectory(simulation):
    return np.stack(
        [simulation.robotSimulation[i].state[:, 0:2] for i in range(simulation.nbOfRobots)],
        axis=1,
    )


def compute_total_distance(trajectory):
    per_robot_distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=2), axis=0)
    return float(np.sum(per_robot_distance)), per_robot_distance


def compute_cumulative_total_distance(trajectory):
    step_total_distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=2), axis=1)
    return np.concatenate(([0.0], np.cumsum(step_total_distance)))


def compute_min_inter_robot_distance(trajectory):
    min_distance = math.inf
    nb_robots = trajectory.shape[1]
    for step_positions in trajectory:
        for robot_i in range(nb_robots):
            for robot_j in range(robot_i + 1, nb_robots):
                distance = float(np.linalg.norm(step_positions[robot_i] - step_positions[robot_j]))
                min_distance = min(min_distance, distance)
    return float(min_distance)


def find_lock_index(measurements, iso_level, near_band, min_robots, hold_steps):
    counts = np.sum(np.abs(measurements - iso_level) <= near_band, axis=1)
    enough_robots = counts >= min_robots
    streak = 0
    for idx, is_good in enumerate(enough_robots):
        streak = streak + 1 if is_good else 0
        if streak >= hold_steps:
            return idx - hold_steps + 1
    return None


def extract_dominant_contour(pot, iso_level, grid_size):
    x_coords = np.linspace(float(pot.xmin), float(pot.xmax), int(grid_size))
    y_coords = np.linspace(float(pot.ymin), float(pot.ymax), int(grid_size))
    xx, yy = np.meshgrid(x_coords, y_coords)
    field = pot.value(np.dstack((xx, yy)))

    fig = Figure()
    ax = fig.add_subplot(111)
    contour_set = ax.contour(x_coords, y_coords, field, levels=[iso_level])
    segments = [
        np.asarray(segment, dtype=float)
        for segment in contour_set.allsegs[0]
        if len(segment) >= 2
    ]
    if not segments:
        return None

    def contour_length(points):
        return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))

    segments.sort(key=contour_length, reverse=True)
    return segments[0]


def compute_coverage(contour_points, trajectory_points, coverage_distance):
    if contour_points is None or trajectory_points.size == 0:
        return math.nan
    distances = np.linalg.norm(
        contour_points[:, None, :] - trajectory_points[None, :, :],
        axis=2,
    )
    return float(np.mean(np.min(distances, axis=1) <= coverage_distance))


def compute_progressive_shape_metrics(
    contour_points,
    trajectory,
    lock_idx,
    coverage_distance,
    distance_budgets,
    target_coverages,
):
    metrics = {}
    for budget in distance_budgets:
        metrics[f"gamma_{int(round(float(budget)))}m"] = math.nan
    for target in target_coverages:
        metrics[f"d{int(round(100.0 * float(target)))}_distance"] = math.nan

    if contour_points is None or lock_idx is None:
        return metrics

    post_lock_trajectory = trajectory[int(lock_idx) :, :, :]
    if post_lock_trajectory.shape[0] == 0:
        return metrics

    step_total_distance = np.sum(
        np.linalg.norm(np.diff(post_lock_trajectory, axis=0), axis=2),
        axis=1,
    )
    post_lock_cumulative = np.concatenate(([0.0], np.cumsum(step_total_distance)))

    running_min_distance = np.full(contour_points.shape[0], np.inf, dtype=float)
    coverage_history = np.zeros(post_lock_trajectory.shape[0], dtype=float)

    for step_idx in range(post_lock_trajectory.shape[0]):
        robot_positions = post_lock_trajectory[step_idx, :, :]
        distances = np.linalg.norm(
            contour_points[:, None, :] - robot_positions[None, :, :],
            axis=2,
        )
        running_min_distance = np.minimum(running_min_distance, np.min(distances, axis=1))
        coverage_history[step_idx] = float(np.mean(running_min_distance <= coverage_distance))

    for budget in distance_budgets:
        budget_idx = np.searchsorted(post_lock_cumulative, float(budget), side="right") - 1
        budget_idx = max(min(int(budget_idx), coverage_history.shape[0] - 1), 0)
        metrics[f"gamma_{int(round(float(budget)))}m"] = float(coverage_history[budget_idx])

    for target in target_coverages:
        target_idx = np.flatnonzero(coverage_history >= float(target))
        if target_idx.size > 0:
            metrics[f"d{int(round(100.0 * float(target)))}_distance"] = float(
                post_lock_cumulative[int(target_idx[0])]
            )

    return metrics


def compute_best_k_iso_error(measurements, iso_level, k):
    if measurements.size == 0:
        return math.nan
    abs_error = np.abs(measurements - iso_level)
    best_k_error = np.sort(abs_error, axis=1)[:, : min(k, abs_error.shape[1])]
    return float(np.mean(best_k_error))


def compute_run_metrics(suite_name, seed, difficulty, config: BenchmarkConfig, result):
    simulation = result["simulation"]
    measurements = result["potential_measurements"]
    pot = result["pot"]
    diagnostics = result["diagnostics"]
    trajectory = extract_trajectory(simulation)

    total_distance, per_robot_distance = compute_total_distance(trajectory)
    cumulative_total_distance = compute_cumulative_total_distance(trajectory)
    min_inter_robot_distance = compute_min_inter_robot_distance(trajectory)
    max_val_target = float(np.max(pot.value(pot.mu)))
    max_val_found = float(np.max(measurements))
    relative_max_error = abs((max_val_target - max_val_found) / max(max_val_target, 1e-12))

    last_third_start = measurements.shape[0] * 2 // 3
    last_third_measurements = measurements[last_third_start:, :]
    near_last_third = np.abs(last_third_measurements - config.iso_level) <= config.near_band
    robots_in_band_last_third = np.sum(near_last_third, axis=1)
    group_tracking_ratio = float(
        np.mean(robots_in_band_last_third >= config.success_min_robots)
    )
    success_group = int(group_tracking_ratio >= config.success_fraction_threshold)
    mean_robots_in_band_last_third = float(np.mean(robots_in_band_last_third))
    iso_error_all_last_third = float(np.mean(np.abs(last_third_measurements - config.iso_level)))
    iso_error_best3_last_third = compute_best_k_iso_error(
        last_third_measurements, config.iso_level, 3
    )

    hold_steps = max(int(round(config.lock_hold_seconds / config.ts)), 1)
    lock_idx = find_lock_index(
        measurements,
        config.iso_level,
        config.near_band,
        config.success_min_robots,
        hold_steps,
    )
    lock_time = math.nan if lock_idx is None else float(simulation.t[lock_idx])
    lock_distance = (
        math.nan if lock_idx is None else float(cumulative_total_distance[int(lock_idx)])
    )

    contour_points = extract_dominant_contour(pot, config.iso_level, config.contour_grid_size)
    contour_length = (
        math.nan
        if contour_points is None
        else float(np.sum(np.linalg.norm(np.diff(contour_points, axis=0), axis=1)))
    )
    last_third_positions = trajectory[last_third_start:, :, :].reshape(-1, 2)
    coverage_last_third = compute_coverage(
        contour_points, last_third_positions, config.coverage_distance
    )
    shape_metrics = compute_progressive_shape_metrics(
        contour_points,
        trajectory,
        lock_idx,
        config.coverage_distance,
        config.shape_distance_budgets,
        config.shape_target_coverages,
    )

    row = {
        "suite": suite_name,
        "difficulty": int(difficulty),
        "seed": int(seed),
        "success_group": int(success_group),
        "group_tracking_ratio_last_third": group_tracking_ratio,
        "lock_time_s": float(lock_time),
        "lock_distance": float(lock_distance),
        "iso_error_best3_last_third": iso_error_best3_last_third,
        "iso_error_all_last_third": iso_error_all_last_third,
        "coverage_last_third": float(coverage_last_third),
        "relative_max_error": float(relative_max_error),
        "max_val_target": max_val_target,
        "max_val_found": max_val_found,
        "total_distance": total_distance,
        "min_inter_robot_distance": min_inter_robot_distance,
        "mean_robots_in_band_last_third": mean_robots_in_band_last_third,
        "tracking_start_time_diag": (
            math.nan
            if diagnostics.get("tracking_start_time") is None
            else float(diagnostics["tracking_start_time"])
        ),
        "runtime_s": float(result["runtime_s"]),
        "contour_length": float(contour_length),
        "per_robot_distance_mean": float(np.mean(per_robot_distance)),
        "per_robot_distance_std": float(np.std(per_robot_distance)),
    }
    row.update(shape_metrics)
    return row


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


def summarise_rows(rows, suite_name):
    summary_rows = []
    for difficulty in sorted({int(row["difficulty"]) for row in rows}):
        diff_rows = [row for row in rows if int(row["difficulty"]) == difficulty]
        success_rate = float(np.mean([row["success_group"] for row in diff_rows]))
        group_ratio_values = finite_values(diff_rows, "group_tracking_ratio_last_third")
        lock_distance_values = finite_values(diff_rows, "lock_distance")
        lock_time_values = finite_values(diff_rows, "lock_time_s")
        iso_best3_values = finite_values(diff_rows, "iso_error_best3_last_third")
        iso_all_values = finite_values(diff_rows, "iso_error_all_last_third")
        distance_values = finite_values(diff_rows, "total_distance")
        max_error_values = finite_values(diff_rows, "relative_max_error")
        safety_values = finite_values(diff_rows, "min_inter_robot_distance")
        runtime_values = finite_values(diff_rows, "runtime_s")
        mean_robots_values = finite_values(diff_rows, "mean_robots_in_band_last_third")
        coverage_values = finite_values(diff_rows, "coverage_last_third")
        gamma_100_values = finite_values(diff_rows, "gamma_100m")
        gamma_150_values = finite_values(diff_rows, "gamma_150m")
        d50_values = finite_values(diff_rows, "d50_distance")
        d80_values = finite_values(diff_rows, "d80_distance")

        representative_seed = select_representative_seed(diff_rows)

        summary_rows.append(
            {
                "suite": suite_name,
                "difficulty": difficulty,
                "n_runs": len(diff_rows),
                "success_rate": success_rate,
                "r3_median": median_or_nan(group_ratio_values),
                "r3_p10": quantile_or_nan(group_ratio_values, 0.1),
                "r3_p90": quantile_or_nan(group_ratio_values, 0.9),
                "lock_rate": float(lock_distance_values.size / max(len(diff_rows), 1)),
                "lock_distance_median": median_or_nan(lock_distance_values),
                "lock_distance_p90": quantile_or_nan(lock_distance_values, 0.9),
                "lock_time_median_s": median_or_nan(lock_time_values),
                "iso_error_best3_median": median_or_nan(iso_best3_values),
                "iso_error_best3_p10": quantile_or_nan(iso_best3_values, 0.1),
                "iso_error_best3_p90": quantile_or_nan(iso_best3_values, 0.9),
                "iso_error_all_median": median_or_nan(iso_all_values),
                "total_distance_median": median_or_nan(distance_values),
                "relative_max_error_median": median_or_nan(max_error_values),
                "mean_robots_in_band_median": median_or_nan(mean_robots_values),
                "coverage_last_third_median": median_or_nan(coverage_values),
                "gamma_100m_median": median_or_nan(gamma_100_values),
                "gamma_100m_q10": quantile_or_nan(gamma_100_values, 0.1),
                "gamma_150m_median": median_or_nan(gamma_150_values),
                "gamma_150m_q10": quantile_or_nan(gamma_150_values, 0.1),
                "d50_rate": float(d50_values.size / max(len(diff_rows), 1)),
                "d50_distance_median": median_or_nan(d50_values),
                "d50_distance_p90": quantile_or_nan(d50_values, 0.9),
                "d80_rate": float(d80_values.size / max(len(diff_rows), 1)),
                "d80_distance_median": median_or_nan(d80_values),
                "min_inter_robot_distance_min": (
                    math.nan if safety_values.size == 0 else float(np.min(safety_values))
                ),
                "runtime_median_s": median_or_nan(runtime_values),
                "representative_seed": representative_seed,
            }
        )

    return summary_rows


def select_representative_seed(rows):
    iso_values = finite_values(rows, "iso_error_best3_last_third")
    if iso_values.size == 0:
        return None
    target = float(np.median(iso_values))
    best_row = min(
        rows,
        key=lambda row: (
            abs(float(row["iso_error_best3_last_third"]) - target),
            int(row["seed"]),
        ),
    )
    return int(best_row["seed"])


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


def print_summary(summary_rows, suite_name):
    print("")
    print(f"=== {suite_name.upper()} SUMMARY ===")
    if suite_name == "shape":
        print(
            "difficulty | G100_med | G150_med | D50_med | D80_rate | dist_med | rep_seed"
        )
    else:
        print(
            "difficulty | success_rate | R3_med | D_lock_med | E_iso3_med | dist_med | rep_seed"
        )
    for row in summary_rows:
        if suite_name == "shape":
            print(
                f"{row['difficulty']:>10d} | "
                f"{format_float(row['gamma_100m_median'], 3):>8} | "
                f"{format_float(row['gamma_150m_median'], 3):>8} | "
                f"{format_float(row['d50_distance_median'], 2):>7} | "
                f"{format_float(row['d80_rate'], 3):>8} | "
                f"{format_float(row['total_distance_median'], 2):>8} | "
                f"{str(row['representative_seed']):>8}"
            )
        else:
            print(
                f"{row['difficulty']:>10d} | "
                f"{format_float(row['success_rate'], 3):>12} | "
                f"{format_float(row['r3_median'], 3):>6} | "
                f"{format_float(row['lock_distance_median'], 2):>10} | "
                f"{format_float(row['iso_error_best3_median'], 3):>10} | "
                f"{format_float(row['total_distance_median'], 2):>8} | "
                f"{str(row['representative_seed']):>8}"
            )


def plot_representative_run(seed, difficulty, config: BenchmarkConfig, output_path: Path):
    result = simulate_once(seed, difficulty, config)
    trajectory = extract_trajectory(result["simulation"])
    pot = result["pot"]
    contour_points = extract_dominant_contour(pot, config.iso_level, config.contour_grid_size)

    x_coords = np.linspace(float(pot.xmin), float(pot.xmax), int(config.figure_grid_size))
    y_coords = np.linspace(float(pot.ymin), float(pot.ymax), int(config.figure_grid_size))
    xx, yy = np.meshgrid(x_coords, y_coords)
    field = pot.value(np.dstack((xx, yy)))

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    filled = ax.contourf(x_coords, y_coords, field, levels=20, cmap="BrBG")
    fig.colorbar(filled, ax=ax, shrink=0.9)

    if contour_points is not None:
        ax.plot(
            contour_points[:, 0],
            contour_points[:, 1],
            "k--",
            linewidth=2.0,
            label=f"Iso-level {config.iso_level:.0f}",
        )

    for robot_idx in range(config.nb_robots):
        color = COLOR_LIST[robot_idx % len(COLOR_LIST)]
        robot_traj = trajectory[:, robot_idx, :]
        ax.plot(robot_traj[:, 0], robot_traj[:, 1], color=color, linewidth=1.8)
        ax.scatter(robot_traj[0, 0], robot_traj[0, 1], color=color, marker="o", s=30)
        ax.scatter(robot_traj[-1, 0], robot_traj[-1, 1], color=color, marker="x", s=40)

    ax.set_aspect("equal")
    ax.set_xlim(float(pot.xmin), float(pot.xmax))
    ax.set_ylim(float(pot.ymin), float(pot.ymax))
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Representative run - difficulty {difficulty}, seed {seed}")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_suite(suite_name, seed_count, config: BenchmarkConfig):
    rows = []
    for difficulty in config.difficulties:
        print(f"[{suite_name}] difficulty {difficulty}: running {seed_count} seeds")
        for seed in range(seed_count):
            result = simulate_once(seed, difficulty, config)
            rows.append(compute_run_metrics(suite_name, seed, difficulty, config, result))
    return rows


def export_suite(output_dir: Path, suite_name, rows, summary_rows):
    write_csv(output_dir / f"{suite_name}_runs.csv", rows)
    write_csv(output_dir / f"{suite_name}_summary.csv", summary_rows)
    write_json(
        output_dir / f"{suite_name}_summary.json",
        {"suite": suite_name, "rows": summary_rows},
    )


def main():
    args = parse_args()
    base_config = BenchmarkConfig()
    ensure_output_dir(args.output_dir)

    suite_order = []
    if args.suite in ("preflight", "both", "all"):
        suite_order.append(("preflight", int(args.preflight_count), base_config))
    if args.suite in ("main", "both", "all"):
        suite_order.append(("main", int(args.main_count), base_config))
    shape_config = replace(base_config, tsim=float(base_config.shape_tsim))
    if args.suite in ("shape", "all"):
        suite_order.append(("shape", int(args.shape_count), shape_config))

    metadata = {
        "config": asdict(base_config),
        "suite": args.suite,
        "preflight_count": int(args.preflight_count),
        "main_count": int(args.main_count),
        "shape_count": int(args.shape_count),
        "output_dir": str(args.output_dir),
        "suites": {},
    }

    representative_source = None

    for suite_name, seed_count, suite_config in suite_order:
        suite_start = time.perf_counter()
        rows = run_suite(suite_name, seed_count, suite_config)
        summary_rows = summarise_rows(rows, suite_name)
        export_suite(args.output_dir, suite_name, rows, summary_rows)
        print_summary(summary_rows, suite_name)

        metadata["suites"][suite_name] = {
            "seed_count": seed_count,
            "config": asdict(suite_config),
            "runtime_s": float(time.perf_counter() - suite_start),
            "summary": summary_rows,
        }

        if suite_name == "main" or representative_source is None:
            representative_source = (suite_name, rows, summary_rows, suite_config)

    if representative_source is not None and not args.skip_figures:
        suite_name, rows, summary_rows, figure_config = representative_source
        for summary in summary_rows:
            representative_seed = summary["representative_seed"]
            if representative_seed is None:
                continue
            figure_name = (
                f"{suite_name}_difficulty_{summary['difficulty']}_"
                f"seed_{representative_seed}_trajectory.png"
            )
            plot_representative_run(
                representative_seed,
                int(summary["difficulty"]),
                figure_config,
                args.output_dir / figure_name,
            )

    write_json(args.output_dir / "benchmark_metadata.json", metadata)

    print("")
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
