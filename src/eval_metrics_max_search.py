# -*- coding: utf-8 -*-
"""
Evaluation metrics dedicated to the "find the global maximum" mission.

This module intentionally complements `eval_metrics.py` instead of replacing it.
It keeps the original metrics and adds spatial metrics that are more
discriminative for source localisation than the potential value alone.
"""

from __future__ import annotations

import math

import numpy as np

import eval_metrics


def extract_trajectory(simulation):
    """Return the fleet trajectory with shape (T, N, 2)."""
    return np.stack(
        [simulation.robotSimulation[i].state[:, 0:2] for i in range(simulation.nbOfRobots)],
        axis=1,
    )


def compute_per_robot_distance(trajectory):
    """Distance travelled by each robot over the whole trajectory."""
    if trajectory.shape[0] <= 1:
        return np.zeros(trajectory.shape[1], dtype=float)

    step_distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=2)
    return np.sum(step_distances, axis=0)


def compute_cumulative_total_distance(trajectory):
    """Fleet-level cumulative travelled distance as a function of time index."""
    if trajectory.shape[0] <= 1:
        return np.zeros(trajectory.shape[0], dtype=float)

    step_total_distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=2), axis=1)
    return np.concatenate(([0.0], np.cumsum(step_total_distance)))


def compute_min_inter_robot_distance(trajectory):
    """Minimum pairwise distance encountered during the run."""
    min_distance = math.inf
    nb_robots = trajectory.shape[1]

    for positions in trajectory:
        for robot_i in range(nb_robots):
            for robot_j in range(robot_i + 1, nb_robots):
                distance = float(np.linalg.norm(positions[robot_i] - positions[robot_j]))
                min_distance = min(min_distance, distance)

    return float(min_distance)


def compute_gini(values):
    """Gini coefficient for non-negative workloads."""
    x = np.sort(np.asarray(values, dtype=float))
    if x.size == 0:
        return math.nan

    total = float(np.sum(x))
    if total <= 1e-12:
        return 0.0

    indices = np.arange(1, x.size + 1, dtype=float)
    return float((2.0 * np.sum(indices * x) / (x.size * total)) - (x.size + 1.0) / x.size)


def get_global_source(pot):
    """
    Return the position and value of the true global source.

    The field library already exposes the Gaussian centers through `pot.mu`.
    We still re-evaluate the potential on those points to identify the true
    global maximum instead of assuming it is always `mu1`.
    """
    candidate_positions = np.asarray(pot.mu, dtype=float)
    candidate_values = np.asarray(pot.value(candidate_positions), dtype=float)
    best_idx = int(np.argmax(candidate_values))

    return (
        candidate_positions[best_idx].copy(),
        float(candidate_values[best_idx]),
        best_idx,
    )


def find_first_hit_index(source_distances, radius):
    """First time index where at least one robot enters the hit radius."""
    hit_steps = np.flatnonzero(np.any(source_distances <= float(radius), axis=1))
    if hit_steps.size == 0:
        return None
    return int(hit_steps[0])


def eval_max_search_metrics(
    simulation,
    potential_measurements,
    pot,
    strict_radius=1.0,
    relaxed_radius=2.0,
):
    """
    Compute mission-1 metrics for maximum search.

    Parameters
    ----------
    simulation : FleetSimulation
    potential_measurements : np.ndarray
        Potential values with shape (T, N).
    pot : Potential
    strict_radius : float
        Radius used for the strict hit criterion.
    relaxed_radius : float
        Radius used for the relaxed hit criterion.

    Returns
    -------
    dict
        Flat metrics dictionary suitable for CSV/JSON export.
    """
    relative_pot_found_error, total_distance = eval_metrics.eval_metrics(
        simulation,
        potential_measurements,
        pot,
    )

    trajectory = extract_trajectory(simulation)
    per_robot_distance = compute_per_robot_distance(trajectory)
    cumulative_total_distance = compute_cumulative_total_distance(trajectory)
    source_pos, max_val_target, source_index = get_global_source(pot)

    source_distances = np.linalg.norm(trajectory - source_pos[None, None, :], axis=2)
    best_source_distance = float(np.min(source_distances))

    first_hit_strict_idx = find_first_hit_index(source_distances, strict_radius)
    first_hit_relaxed_idx = find_first_hit_index(source_distances, relaxed_radius)

    final_positions = trajectory[-1]
    final_source_distances = np.linalg.norm(final_positions - source_pos[None, :], axis=1)
    final_centroid = np.mean(final_positions, axis=0)
    final_group_dispersion = float(
        np.mean(np.linalg.norm(final_positions - final_centroid[None, :], axis=1))
    )

    distance_mean = float(np.mean(per_robot_distance))
    distance_std = float(np.std(per_robot_distance))
    distance_cv = 0.0 if distance_mean <= 1e-12 else float(distance_std / distance_mean)

    metrics = {
        "source_x": float(source_pos[0]),
        "source_y": float(source_pos[1]),
        "source_index": int(source_index),
        "max_val_target": float(max_val_target),
        "max_val_found": float(np.max(potential_measurements)),
        "relative_pot_found_error": float(relative_pot_found_error),
        "total_distance": float(total_distance),
        "global_hit_strict": int(first_hit_strict_idx is not None),
        "global_hit_relaxed": int(first_hit_relaxed_idx is not None),
        "time_to_first_hit_strict_s": (
            math.nan if first_hit_strict_idx is None else float(simulation.t[first_hit_strict_idx])
        ),
        "time_to_first_hit_relaxed_s": (
            math.nan if first_hit_relaxed_idx is None else float(simulation.t[first_hit_relaxed_idx])
        ),
        "distance_to_first_hit_strict": (
            math.nan
            if first_hit_strict_idx is None
            else float(cumulative_total_distance[first_hit_strict_idx])
        ),
        "distance_to_first_hit_relaxed": (
            math.nan
            if first_hit_relaxed_idx is None
            else float(cumulative_total_distance[first_hit_relaxed_idx])
        ),
        "best_source_distance": best_source_distance,
        "robots_ever_near_source_relaxed": int(
            np.sum(np.any(source_distances <= float(relaxed_radius), axis=0))
        ),
        "final_best_robot_distance_to_source": float(np.min(final_source_distances)),
        "final_mean_distance_to_source": float(np.mean(final_source_distances)),
        "final_group_dispersion": final_group_dispersion,
        "min_inter_robot_distance": compute_min_inter_robot_distance(trajectory),
        "per_robot_distance_mean": distance_mean,
        "per_robot_distance_std": distance_std,
        "distance_workload_cv": distance_cv,
        "distance_workload_gini": compute_gini(per_robot_distance),
    }

    return metrics
