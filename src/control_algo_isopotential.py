# -*- coding: utf-8 -*-
"""
Simple shared-memory controller for cooperative iso-level tracking.

This version is intentionally sensor-only:
- it reads the potential only at the current robot positions
- it stores only past robot poses and the measured values at those poses
"""

from copy import deepcopy
import math

import numpy as np

from lib.potential import Potential


DEFAULT_CONTROLLER_CONFIG = {
    "iso_level": 260.0,
    "max_history": 900,
    "gradient_radius": 4.0,
    "min_gradient_samples": 8,
    "search_speed": 3.2,
    "search_spiral_gain": 0.55,
    "search_omega": 0.55,
    "search_gradient_gain": 1.2,
    "track_tangential_speed": 1.35,
    "k_iso": 0.12,
    "k_phase": 0.55,
    "repulsion_gain": 0.9,
    "min_robot_distance": 1.6,
    "u_max": 3.6,
    "center_top_k": 25,
    "center_ema_alpha": 0.25,
    "track_entry_margin": 2.0,
    "track_min_samples": 20,
    "near_iso_band": 3.0,
    "search_anchors": None,
}


# User-facing configuration block for the legacy runner.
CONTROLLER_CONFIG = deepcopy(DEFAULT_CONTROLLER_CONFIG)


firstCall = True
pot = None
controller_state = {}


def _merge_config(base_config, override_config):
    merged = deepcopy(base_config)
    if not override_config:
        return merged

    for key, value in override_config.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalise_config(config):
    normalised = deepcopy(config)
    if "v_t" in normalised:
        normalised["track_tangential_speed"] = float(normalised["v_t"])
    if "search_anchors" in normalised and normalised["search_anchors"] is not None:
        normalised["search_anchors"] = np.asarray(
            normalised["search_anchors"], dtype=float
        )
    return normalised


def _wrap_angle(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _rotate_left(vector):
    return np.array([-vector[1], vector[0]], dtype=float)


def _normalise_vector(vector):
    vec = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return np.zeros(2, dtype=float), 0.0
    return vec / norm, norm


def _saturate(vector, max_norm):
    vec = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm <= max_norm or norm <= 1e-12:
        return vec
    return vec * (max_norm / norm)


def _copy_config_for_diagnostics(config):
    return {
        "iso_level": float(config["iso_level"]),
        "max_history": int(config["max_history"]),
        "gradient_radius": float(config["gradient_radius"]),
        "min_gradient_samples": int(config["min_gradient_samples"]),
        "search_speed": float(config["search_speed"]),
        "track_tangential_speed": float(config["track_tangential_speed"]),
        "k_iso": float(config["k_iso"]),
        "k_phase": float(config["k_phase"]),
        "repulsion_gain": float(config["repulsion_gain"]),
        "min_robot_distance": float(config["min_robot_distance"]),
        "u_max": float(config["u_max"]),
    }


def reset_controller(controller_config=None):
    global firstCall
    global pot
    global controller_state

    firstCall = True
    pot = None

    config = _merge_config(DEFAULT_CONTROLLER_CONFIG, CONTROLLER_CONFIG)
    config = _merge_config(config, controller_config)
    config = _normalise_config(config)

    controller_state = {
        "config": config,
        "step": -1,
        "last_t": None,
        "commands": None,
        "samples": [],
        "sample_count": 0,
        "last_positions": None,
        "last_values": None,
        "observed_max_value": -math.inf,
        "observed_max_pos": None,
        "center_estimate": None,
        "phase": "search",
        "tracking_start_time": None,
        "search_anchors": None
        if config.get("search_anchors") is None
        else np.asarray(config["search_anchors"], dtype=float).copy(),
        "phase_history": [],
    }


def get_controller_diagnostics():
    config = controller_state.get("config", {})
    center_estimate = controller_state.get("center_estimate")
    observed_max_value = controller_state.get("observed_max_value", -math.inf)
    return {
        "config": _copy_config_for_diagnostics(config) if config else {},
        "phase": controller_state.get("phase"),
        "tracking_start_time": controller_state.get("tracking_start_time"),
        "sample_count": int(controller_state.get("sample_count", 0)),
        "observed_max_value": (
            None if not math.isfinite(observed_max_value) else float(observed_max_value)
        ),
        "center_estimate": (
            None if center_estimate is None else np.asarray(center_estimate, dtype=float).copy()
        ),
        "phase_history": list(controller_state.get("phase_history", [])),
    }


def _ensure_initialized(difficulty, random, _pot, nb_robots):
    global firstCall
    global pot

    if firstCall or pot is None:
        pot = _pot if _pot is not None else Potential(difficulty=difficulty, random=random)
        firstCall = False

    if controller_state.get("commands") is None or controller_state["commands"].shape[0] != nb_robots:
        controller_state["commands"] = np.zeros((nb_robots, 2), dtype=float)


def _ensure_search_anchors(positions):
    anchors = controller_state.get("search_anchors")
    if anchors is None or anchors.shape != positions.shape:
        controller_state["search_anchors"] = np.asarray(positions, dtype=float).copy()


def _append_measurements(t, positions, values):
    state = controller_state
    samples = state["samples"]
    max_history = int(state["config"]["max_history"])

    for robot_idx in range(positions.shape[0]):
        position = np.asarray(positions[robot_idx], dtype=float).copy()
        value = float(values[robot_idx])
        samples.append((float(t), int(robot_idx), position, value))
        if value > state["observed_max_value"]:
            state["observed_max_value"] = value
            state["observed_max_pos"] = position.copy()

    if len(samples) > max_history:
        del samples[: len(samples) - max_history]

    state["sample_count"] += int(positions.shape[0])
    state["last_positions"] = np.asarray(positions, dtype=float).copy()
    state["last_values"] = np.asarray(values, dtype=float).copy()


def _update_center_estimate():
    state = controller_state
    samples = state["samples"]
    if not samples:
        return

    top_k = max(int(state["config"]["center_top_k"]), 1)
    best_samples = sorted(samples, key=lambda item: item[3], reverse=True)[:top_k]
    best_positions = np.vstack([item[2] for item in best_samples])
    best_values = np.asarray([max(item[3], 1.0) for item in best_samples], dtype=float)
    candidate_center = np.average(best_positions, axis=0, weights=best_values)

    previous_center = state.get("center_estimate")
    if previous_center is None:
        state["center_estimate"] = np.asarray(candidate_center, dtype=float)
        return

    alpha = float(state["config"]["center_ema_alpha"])
    state["center_estimate"] = (
        (1.0 - alpha) * np.asarray(previous_center, dtype=float) + alpha * candidate_center
    )


def _estimate_local_gradient(position):
    config = controller_state["config"]
    samples = controller_state["samples"]
    min_samples = int(config["min_gradient_samples"])
    if len(samples) < min_samples:
        return None

    sample_positions = np.vstack([item[2] for item in samples])
    sample_values = np.asarray([item[3] for item in samples], dtype=float)
    delta = sample_positions - np.asarray(position, dtype=float)
    distances = np.linalg.norm(delta, axis=1)
    radius = float(config["gradient_radius"])
    mask = distances <= radius
    if int(np.sum(mask)) < min_samples:
        return None

    local_delta = delta[mask]
    local_values = sample_values[mask]
    local_distances = distances[mask]
    weights = np.clip(1.0 - local_distances / max(radius, 1e-9), 0.1, None)

    design = np.column_stack(
        (local_delta[:, 0], local_delta[:, 1], np.ones(local_delta.shape[0]))
    )
    design_weighted = design * np.sqrt(weights)[:, None]
    values_weighted = local_values * np.sqrt(weights)
    coefficients, _, _, _ = np.linalg.lstsq(design_weighted, values_weighted, rcond=None)
    gradient = np.asarray(coefficients[:2], dtype=float)
    if not np.all(np.isfinite(gradient)):
        return None
    if float(np.linalg.norm(gradient)) <= 1e-3:
        return None
    return gradient


def _repulsion_term(robot_idx, positions):
    config = controller_state["config"]
    position = np.asarray(positions[robot_idx], dtype=float)
    repulsion = np.zeros(2, dtype=float)
    min_distance = float(config["min_robot_distance"])

    for other_idx in range(positions.shape[0]):
        if other_idx == robot_idx:
            continue

        direction = position - np.asarray(positions[other_idx], dtype=float)
        unit_direction, distance = _normalise_vector(direction)
        if distance >= min_distance or distance <= 1e-12:
            continue

        strength = float(config["repulsion_gain"]) * (min_distance / distance - 1.0)
        repulsion += strength * unit_direction

    return repulsion


def _spiral_reference(t, robot_idx, position, reference_center, nb_robots):
    config = controller_state["config"]
    radius = 1.0 + float(config["search_spiral_gain"]) * max(float(t), 0.0)
    angle = float(config["search_omega"]) * float(t)
    angle += 2.0 * math.pi * float(robot_idx) / max(int(nb_robots), 1)
    target = np.asarray(reference_center, dtype=float) + radius * np.array(
        [math.cos(angle), math.sin(angle)],
        dtype=float,
    )
    return target - np.asarray(position, dtype=float)


def _search_commands(t, positions, values):
    state = controller_state
    config = state["config"]
    nb_robots = positions.shape[0]
    commands = np.zeros((nb_robots, 2), dtype=float)
    center_estimate = state.get("center_estimate")
    observed_max_pos = state.get("observed_max_pos")

    for robot_idx in range(nb_robots):
        anchor = state["search_anchors"][robot_idx]
        reference_center = center_estimate if center_estimate is not None else anchor
        spiral_direction, _ = _normalise_vector(
            _spiral_reference(t, robot_idx, positions[robot_idx], reference_center, nb_robots)
        )

        raw_command = 0.8 * spiral_direction

        gradient = _estimate_local_gradient(positions[robot_idx])
        if gradient is not None:
            gradient_direction, _ = _normalise_vector(gradient)
            raw_command += float(config["search_gradient_gain"]) * gradient_direction

        if observed_max_pos is not None:
            best_direction, _ = _normalise_vector(observed_max_pos - positions[robot_idx])
            if state["observed_max_value"] > float(values[robot_idx]) + float(
                config["track_entry_margin"]
            ):
                raw_command += 0.9 * best_direction

        raw_command += _repulsion_term(robot_idx, positions)
        commands[robot_idx] = _saturate(raw_command, float(config["search_speed"]))

    return commands


def _track_commands(positions, values):
    state = controller_state
    config = state["config"]
    nb_robots = positions.shape[0]
    center_estimate = state.get("center_estimate")
    if center_estimate is None:
        return _search_commands(state["last_t"], positions, values)

    center_estimate = np.asarray(center_estimate, dtype=float)
    offsets = positions - center_estimate[None, :]
    radii = np.linalg.norm(offsets, axis=1)
    angles = np.arctan2(offsets[:, 1], offsets[:, 0])
    order = np.argsort(angles)
    ranks = np.empty(nb_robots, dtype=int)
    ranks[order] = np.arange(nb_robots, dtype=int)
    target_gap = 2.0 * math.pi / max(nb_robots, 1)
    commands = np.zeros((nb_robots, 2), dtype=float)

    for robot_idx in range(nb_robots):
        radial_direction, radius = _normalise_vector(offsets[robot_idx])
        if radius <= 1e-9:
            fallback = positions[robot_idx] - state.get("observed_max_pos", center_estimate)
            radial_direction, radius = _normalise_vector(fallback)
        if radius <= 1e-9:
            angle = 2.0 * math.pi * float(robot_idx) / max(nb_robots, 1)
            radial_direction = np.array([math.cos(angle), math.sin(angle)], dtype=float)

        tangential_direction = _rotate_left(radial_direction)

        rank = ranks[robot_idx]
        prev_idx = order[(rank - 1) % nb_robots]
        next_idx = order[(rank + 1) % nb_robots]
        ahead_gap = _wrap_angle(angles[next_idx] - angles[robot_idx])
        behind_gap = _wrap_angle(angles[robot_idx] - angles[prev_idx])
        if ahead_gap <= 0.0:
            ahead_gap += 2.0 * math.pi
        if behind_gap <= 0.0:
            behind_gap += 2.0 * math.pi
        spacing_error = (ahead_gap - behind_gap) / max(target_gap, 1e-9)

        radial_term = float(config["k_iso"]) * (
            float(values[robot_idx]) - float(config["iso_level"])
        ) * radial_direction
        tangential_speed = max(
            0.2,
            float(config["track_tangential_speed"]) + float(config["k_phase"]) * spacing_error,
        )
        tangential_term = tangential_speed * tangential_direction

        raw_command = radial_term + tangential_term
        raw_command += _repulsion_term(robot_idx, positions)
        commands[robot_idx] = _saturate(raw_command, float(config["u_max"]))

    return commands


def _should_track():
    state = controller_state
    config = state["config"]
    if state.get("center_estimate") is None:
        return False
    if state.get("sample_count", 0) < int(config["track_min_samples"]):
        return False
    return state.get("observed_max_value", -math.inf) >= float(config["iso_level"]) + float(
        config["track_entry_margin"]
    )


def _update_controller_step(t, positions):
    state = controller_state
    values = np.asarray(pot.value(positions), dtype=float)
    state["step"] += 1
    _append_measurements(t, positions, values)
    _update_center_estimate()

    if _should_track():
        state["phase"] = "track"
        if state["tracking_start_time"] is None:
            state["tracking_start_time"] = float(t)
        state["commands"] = _track_commands(positions, values)
    else:
        state["phase"] = "search"
        state["commands"] = _search_commands(t, positions, values)

    state["last_t"] = float(t)
    state["phase_history"].append((float(t), state["phase"]))


def potential_seeking_ctrl(
    t,
    robotNo,
    robots_poses,
    _eval=False,
    _pot=None,
    difficulty=1,
    random=False,
):
    del _eval
    global pot

    positions = np.asarray(robots_poses[:, 0:2], dtype=float)
    if controller_state.get("last_t") is not None and t < controller_state["last_t"] - 1e-12:
        reset_controller(controller_state.get("config"))

    _ensure_initialized(difficulty, random, _pot, positions.shape[0])
    _ensure_search_anchors(positions)

    if controller_state["last_t"] is None or abs(controller_state["last_t"] - t) > 1e-12:
        _update_controller_step(t, positions)

    command = np.asarray(controller_state["commands"][robotNo], dtype=float)
    return float(command[0]), float(command[1]), pot


reset_controller()
