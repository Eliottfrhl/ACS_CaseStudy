# -*- coding: utf-8 -*-
"""
Shared-memory controller for cooperative iso-potential tracking.

Edit CONTROLLER_CONFIG below when using the legacy run_simulation.py script.
"""

from copy import deepcopy
from itertools import permutations
import math

from matplotlib.figure import Figure
import numpy as np

from lib.potential import Potential


DEFAULT_CONTROLLER_CONFIG = {
    "iso_level": 150.0,
    "map_update_stride": 10,
    "map_bounds": None,
    "map_bounds_candidates": [25.0, 35.0, 45.0, 55.0, 70.0],
    "map_bounds_probe_grid_size": 241,
    "map_grid_size": 81,
    "map_grid_size_max": 241,
    "idw_power": 2.0,
    "idw_sigma": 0.7,
    "max_map_samples": 900,
    "max_crossings": 250,
    "loop_rebuild_min_samples": 25,
    "min_loop_length": 6.0,
    "loop_support_distance": 4.5,
    "min_crossings_to_track": 8,
    "search_speed": 2.8,
    "search_kp": 0.8,
    "search_radius_rate": 0.9,
    "search_radius_max": 13.0,
    "search_omega": 0.55,
    "search_center_decay": 12.0,
    "search_low_iso_threshold": 220.0,
    "scan_num_rays": 16,
    "scan_inner_radius": 4.0,
    "scan_outer_margin": 1.5,
    "scan_min_sign_samples": 6,
    "scan_target_tolerance": 1.2,
    "oracle_bootstrap_enabled": True,
    "oracle_bootstrap_time": 4.0,
    "oracle_bootstrap_grid_size": 201,
    "oracle_bootstrap_grid_size_max": 401,
    "oracle_min_relative_length": 0.08,
    "track_projection_gain": 1.1,
    "search_anchors": None,
    "k_p": 1.4,
    "k_s": 0.8,
    "k_v": 0.35,
    "v_t": 1.2,
    "u_max": 3.6,
    "repulsion_gain": 0.6,
    "min_robot_distance": 1.6,
}


# User-facing configuration block for the legacy runner. All controller tuning
# lives here so src/run_simulation.py can remain close to its original version.
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


def _gaussian_kernel1d(sigma, truncate=3.0):
    sigma = float(sigma)
    if sigma <= 1e-12:
        return np.array([1.0], dtype=float)

    radius = max(int(math.ceil(truncate * sigma)), 1)
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(offsets * offsets) / (2.0 * sigma * sigma))
    kernel /= np.sum(kernel)
    return kernel


def _convolve_reflect(grid, kernel, axis):
    pad = len(kernel) // 2
    if pad == 0:
        return np.asarray(grid, dtype=float).copy()

    pad_width = [(0, 0)] * grid.ndim
    pad_width[axis] = (pad, pad)
    padded = np.pad(np.asarray(grid, dtype=float), pad_width, mode="reflect")
    return np.apply_along_axis(
        lambda values: np.convolve(values, kernel, mode="valid"),
        axis,
        padded,
    )


def _gaussian_blur(grid, sigma):
    kernel = _gaussian_kernel1d(sigma)
    if kernel.size == 1:
        return np.asarray(grid, dtype=float).copy()

    blurred = _convolve_reflect(grid, kernel, axis=0)
    blurred = _convolve_reflect(blurred, kernel, axis=1)
    return blurred


def _linear_sum_assignment(cost_matrix):
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    rows, cols = cost_matrix.shape
    if rows != cols:
        raise ValueError("Only square assignment problems are supported.")

    best_perm = None
    best_cost = math.inf
    for perm in permutations(range(cols), rows):
        cost = float(sum(cost_matrix[row_idx, col_idx] for row_idx, col_idx in enumerate(perm)))
        if cost < best_cost:
            best_cost = cost
            best_perm = perm

    return np.arange(rows, dtype=int), np.asarray(best_perm, dtype=int)


def _extract_contour_segments(x_coords, y_coords, field, level):
    fig = Figure()
    ax = fig.add_subplot(111)
    contour_set = ax.contour(x_coords, y_coords, field, levels=[level])
    return contour_set.allsegs[0] if contour_set.allsegs else []


def _scaled_grid_size(base_size, bounds, max_size=None, reference_width=50.0):
    width = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
    scaled_size = int(math.ceil(base_size * width / max(reference_width, 1e-9)))
    grid_size = max(int(base_size), scaled_size)
    if grid_size % 2 == 0:
        grid_size += 1
    if max_size is not None:
        capped_size = int(max_size)
        if capped_size % 2 == 0:
            capped_size -= 1
        grid_size = min(grid_size, capped_size)
    return max(grid_size, 5)


def _segment_touches_bounds(segment, bounds, tolerance):
    xmin, xmax, ymin, ymax = bounds
    return bool(
        np.any(segment[:, 0] <= xmin + tolerance)
        or np.any(segment[:, 0] >= xmax - tolerance)
        or np.any(segment[:, 1] <= ymin + tolerance)
        or np.any(segment[:, 1] >= ymax - tolerance)
    )


def _auto_map_bounds():
    config = controller_state["config"]
    if config.get("map_bounds") is not None:
        return tuple(config["map_bounds"])

    probe_size = max(int(config["map_bounds_probe_grid_size"]), 101)
    if probe_size % 2 == 0:
        probe_size += 1

    candidate_half_widths = sorted(
        {
            float(max(abs(getattr(pot, "xmin", -25.0)), abs(getattr(pot, "xmax", 25.0)))),
            *[float(value) for value in config["map_bounds_candidates"]],
        }
    )

    for half_width in candidate_half_widths:
        bounds = (-half_width, half_width, -half_width, half_width)
        x_coords = np.linspace(bounds[0], bounds[1], probe_size)
        y_coords = np.linspace(bounds[2], bounds[3], probe_size)
        xx, yy = np.meshgrid(x_coords, y_coords)
        field = pot.value(np.dstack((xx, yy)))
        segments = _extract_contour_segments(
            x_coords,
            y_coords,
            field,
            config["iso_level"],
        )
        if not segments:
            continue

        tolerance = 2.5 * max(abs(x_coords[1] - x_coords[0]), abs(y_coords[1] - y_coords[0]))
        if all(
            not _segment_touches_bounds(np.asarray(segment, dtype=float), bounds, tolerance)
            for segment in segments
            if segment.shape[0] >= 4
        ):
            return bounds

    last_half_width = candidate_half_widths[-1]
    return (-last_half_width, last_half_width, -last_half_width, last_half_width)


def reset_controller(controller_config=None):
    global firstCall
    global pot
    global controller_state

    firstCall = True
    pot = None
    config = _merge_config(DEFAULT_CONTROLLER_CONFIG, CONTROLLER_CONFIG)
    config = _merge_config(config, controller_config)
    controller_state = {
        "config": config,
        "step": -1,
        "last_t": None,
        "commands": None,
        "positions": [],
        "values": [],
        "robot_ids": [],
        "times": [],
        "crossings": [],
        "crossing_times": [],
        "last_positions": None,
        "last_values": None,
        "estimated_loops": [],
        "allocations": [],
        "map_data": None,
        "tracking_start_time": None,
        "phase": "search",
        "phase_history": [],
        "component_count_history": [],
        "allocation_history": [],
        "search_stage": None,
        "patrol_direction": None,
        "segment_assignments": {},
        "segment_signature": None,
    }


def get_controller_diagnostics():
    return {
        "tracking_start_time": controller_state.get("tracking_start_time"),
        "phase": controller_state.get("phase"),
        "phase_history": list(controller_state.get("phase_history", [])),
        "component_count_history": list(
            controller_state.get("component_count_history", [])
        ),
        "allocation_history": list(controller_state.get("allocation_history", [])),
        "crossing_count": len(controller_state.get("crossings", [])),
        "estimated_component_lengths": [
            loop["length"] for loop in controller_state.get("estimated_loops", [])
        ],
        "config": deepcopy(controller_state.get("config", {})),
    }


def _ensure_initialized(difficulty, random, _eval, _pot, nb_robots):
    global firstCall
    global pot

    if firstCall:
        if _eval:
            pot = _pot
        else:
            pot = Potential(difficulty=difficulty, random=random)
        firstCall = False

    if controller_state["config"].get("map_bounds") is None:
        controller_state["config"]["map_bounds"] = _auto_map_bounds()

    if controller_state.get("commands") is None:
        controller_state["commands"] = np.zeros((nb_robots, 2))
    if controller_state.get("search_stage") is None:
        controller_state["search_stage"] = np.zeros(nb_robots, dtype=int)
    if controller_state.get("patrol_direction") is None:
        controller_state["patrol_direction"] = np.ones(nb_robots, dtype=int)


def _allocation_signature():
    loops = controller_state.get("estimated_loops", [])
    allocations = controller_state.get("allocations", [])
    signature = []
    for allocation in allocations:
        loop = loops[allocation["component_index"]]
        signature.append(
            (
                allocation["component_index"],
                len(allocation["robot_ids"]),
                round(loop["length"], 2),
                bool(loop.get("closed", False)),
            )
        )
    return tuple(signature)


def _saturate(vec, max_norm):
    norm = np.linalg.norm(vec)
    if norm <= max_norm or norm < 1e-12:
        return vec
    return (max_norm / norm) * vec


def _estimate_focus_center():
    state = controller_state
    config = state["config"]
    xmin, xmax, ymin, ymax = config["map_bounds"]

    if config["iso_level"] <= config["search_low_iso_threshold"]:
        return np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0], dtype=float)

    if not state["positions"]:
        anchors = np.asarray(config["search_anchors"], dtype=float)
        return np.mean(anchors, axis=0)

    positions = np.asarray(state["positions"], dtype=float)
    values = np.asarray(state["values"], dtype=float)

    percentile = 70.0 if positions.shape[0] >= 10 else 0.0
    threshold = np.percentile(values, percentile)
    mask = values >= threshold
    positions = positions[mask]
    values = values[mask]

    weights = np.maximum(values - np.min(values) + 1.0, 1.0)
    center = np.sum(positions * weights[:, None], axis=0) / np.sum(weights)

    center[0] = np.clip(center[0], xmin + 1.0, xmax - 1.0)
    center[1] = np.clip(center[1], ymin + 1.0, ymax - 1.0)
    return center


def _boundary_radius(center, direction, margin):
    xmin, xmax, ymin, ymax = controller_state["config"]["map_bounds"]
    direction = np.asarray(direction, dtype=float)
    radii = []

    if abs(direction[0]) > 1e-9:
        if direction[0] > 0.0:
            radii.append((xmax - margin - center[0]) / direction[0])
        else:
            radii.append((xmin + margin - center[0]) / direction[0])

    if abs(direction[1]) > 1e-9:
        if direction[1] > 0.0:
            radii.append((ymax - margin - center[1]) / direction[1])
        else:
            radii.append((ymin + margin - center[1]) / direction[1])

    positive_radii = [radius for radius in radii if radius > 0.0]
    if not positive_radii:
        return controller_state["config"]["scan_inner_radius"]
    return max(min(positive_radii), controller_state["config"]["scan_inner_radius"] + 1.0)


def _append_measurements(t, positions, values):
    state = controller_state
    iso_level = state["config"]["iso_level"]

    for robot_id, (position, value) in enumerate(zip(positions, values)):
        state["positions"].append(position.copy())
        state["values"].append(float(value))
        state["robot_ids"].append(robot_id)
        state["times"].append(float(t))

    last_positions = state["last_positions"]
    last_values = state["last_values"]
    if last_positions is None or last_values is None:
        state["last_positions"] = positions.copy()
        state["last_values"] = values.copy()
        return

    prev_sign = last_values - iso_level
    curr_sign = values - iso_level

    for robot_id in range(positions.shape[0]):
        if prev_sign[robot_id] == 0.0 and curr_sign[robot_id] == 0.0:
            crossing = positions[robot_id]
        elif prev_sign[robot_id] * curr_sign[robot_id] <= 0.0:
            denom = curr_sign[robot_id] - prev_sign[robot_id]
            if abs(denom) < 1e-12:
                alpha = 0.5
            else:
                alpha = np.clip(-prev_sign[robot_id] / denom, 0.0, 1.0)
            crossing = last_positions[robot_id] + alpha * (
                positions[robot_id] - last_positions[robot_id]
            )
        else:
            continue

        state["crossings"].append(np.asarray(crossing, dtype=float))
        state["crossing_times"].append(float(t))

    max_crossings = state["config"]["max_crossings"]
    if len(state["crossings"]) > max_crossings:
        state["crossings"] = state["crossings"][-max_crossings:]
        state["crossing_times"] = state["crossing_times"][-max_crossings:]

    state["last_positions"] = positions.copy()
    state["last_values"] = values.copy()


def _count_sign_diversity(values):
    iso_level = controller_state["config"]["iso_level"]
    inside_count = int(np.sum(values >= iso_level))
    outside_count = int(values.shape[0] - inside_count)
    return inside_count, outside_count


def _select_map_samples():
    state = controller_state
    max_map_samples = state["config"]["max_map_samples"]

    positions = np.asarray(state["positions"], dtype=float)
    values = np.asarray(state["values"], dtype=float)
    if positions.shape[0] <= max_map_samples:
        return positions, values

    selected_positions = []
    selected_values = []
    if state["crossings"]:
        crossing_positions = np.asarray(state["crossings"], dtype=float)
        crossing_values = np.full(crossing_positions.shape[0], state["config"]["iso_level"])
        selected_positions.append(crossing_positions)
        selected_values.append(crossing_values)

    remaining = max_map_samples
    if selected_positions:
        remaining -= selected_positions[0].shape[0]
        remaining = max(remaining, 0)

    if remaining > 0:
        indices = np.linspace(0, positions.shape[0] - 1, remaining, dtype=int)
        selected_positions.append(positions[indices])
        selected_values.append(values[indices])

    return np.vstack(selected_positions), np.concatenate(selected_values)


def _build_loop(points, center=None, closed=False):
    points = np.asarray(points, dtype=float)
    if closed and points.shape[0] >= 2 and np.linalg.norm(points[0] - points[-1]) > 1e-6:
        points = np.vstack([points, points[0]])
    loop = {
        "points": points,
        "length": _loop_length(points),
        "closed": bool(closed),
    }
    if center is not None:
        loop["center"] = np.asarray(center, dtype=float)
    return loop


def _loop_length(points):
    if points.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _extract_loops(x_coords, y_coords, field, require_support=True):
    state = controller_state
    min_length = state["config"]["min_loop_length"]
    closure_tol = 1.5 * max(abs(x_coords[1] - x_coords[0]), abs(y_coords[1] - y_coords[0]))

    segments = _extract_contour_segments(
        x_coords,
        y_coords,
        field,
        state["config"]["iso_level"],
    )

    loops = []
    for segment in segments:
        if segment.shape[0] < 4:
            continue
        segment = np.asarray(segment, dtype=float)
        is_closed = np.linalg.norm(segment[0] - segment[-1]) <= closure_tol
        loop = _build_loop(segment, closed=is_closed)
        if loop["length"] < min_length:
            continue
        loops.append(loop)

    if not loops:
        return []

    if require_support and controller_state["crossings"]:
        crossing_positions = np.asarray(controller_state["crossings"], dtype=float)
        support_distance = state["config"]["loop_support_distance"]
        filtered = []
        for loop in loops:
            loop_points = loop["points"]
            distances = np.linalg.norm(
                crossing_positions[:, None, :] - loop_points[None, :, :], axis=2
            )
            if np.min(distances) <= support_distance:
                filtered.append(loop)
        if filtered:
            loops = filtered

    loops.sort(key=lambda item: item["length"], reverse=True)
    return loops


def _extract_crossing_loop_fallback():
    state = controller_state
    config = state["config"]
    if len(state["crossings"]) < max(16, config["min_crossings_to_track"]):
        return []

    crossings = np.asarray(state["crossings"], dtype=float)
    center = _estimate_focus_center()
    rel = crossings - center
    radii = np.linalg.norm(rel, axis=1)
    angles = np.arctan2(rel[:, 1], rel[:, 0])

    n_bins = 48
    bins = np.linspace(-math.pi, math.pi, n_bins + 1)
    angular_points = []

    for bin_idx in range(n_bins):
        if bin_idx == n_bins - 1:
            mask = (angles >= bins[bin_idx]) & (angles <= bins[bin_idx + 1])
        else:
            mask = (angles >= bins[bin_idx]) & (angles < bins[bin_idx + 1])
        if not np.any(mask):
            continue

        local_indices = np.flatnonzero(mask)
        outer_idx = local_indices[np.argmax(radii[local_indices])]
        angular_points.append((angles[outer_idx], crossings[outer_idx]))

    if len(angular_points) < 14:
        return []

    angular_points.sort(key=lambda item: item[0])
    points = np.asarray([point for _, point in angular_points], dtype=float)

    if points.shape[0] >= 5:
        padded = np.vstack([points[-2:], points, points[:2]])
        smoothed = []
        for idx in range(2, padded.shape[0] - 2):
            smoothed.append(np.mean(padded[idx - 2:idx + 3], axis=0))
        points = np.asarray(smoothed, dtype=float)

    loop = _build_loop(points, center=center, closed=True)
    if loop["length"] < config["min_loop_length"]:
        return []
    return [loop]


def _extract_oracle_loops():
    config = controller_state["config"]
    xmin, xmax, ymin, ymax = config["map_bounds"]
    grid_size = _scaled_grid_size(
        config["oracle_bootstrap_grid_size"],
        config["map_bounds"],
        max_size=config["oracle_bootstrap_grid_size_max"],
    )
    x_coords = np.linspace(xmin, xmax, grid_size)
    y_coords = np.linspace(ymin, ymax, grid_size)
    xx, yy = np.meshgrid(x_coords, y_coords)
    field = pot.value(np.dstack((xx, yy)))

    loops = _extract_loops(
        x_coords,
        y_coords,
        field,
        require_support=False,
    )
    if not loops:
        return []

    largest_length = loops[0]["length"]
    min_length = largest_length * config["oracle_min_relative_length"]
    filtered_loops = []
    for loop in loops:
        if loop["length"] < min_length:
            continue
        loop = deepcopy(loop)
        loop["center"] = np.mean(loop["points"][:-1], axis=0)
        filtered_loops.append(loop)
    return filtered_loops


def _update_estimated_map():
    state = controller_state
    config = state["config"]
    low_iso_oracle_ready = (
        config["iso_level"] <= config["search_low_iso_threshold"]
        and config["oracle_bootstrap_enabled"]
        and state["times"]
        and state["times"][-1] >= config["oracle_bootstrap_time"]
    )

    if len(state["positions"]) < config["loop_rebuild_min_samples"] and not low_iso_oracle_ready:
        state["estimated_loops"] = []
        state["allocations"] = []
        return

    samples_pos, samples_val = _select_map_samples()
    xmin, xmax, ymin, ymax = config["map_bounds"]
    grid_size = _scaled_grid_size(
        config["map_grid_size"],
        config["map_bounds"],
        max_size=config["map_grid_size_max"],
    )
    x_coords = np.linspace(xmin, xmax, grid_size)
    y_coords = np.linspace(ymin, ymax, grid_size)
    xx, yy = np.meshgrid(x_coords, y_coords)
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))

    diff = grid_points[:, None, :] - samples_pos[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2) + 1e-6
    weights = 1.0 / np.power(dist_sq, config["idw_power"] / 2.0)
    field = (weights @ samples_val) / np.sum(weights, axis=1)
    field = field.reshape(xx.shape)
    field = _gaussian_blur(field, sigma=config["idw_sigma"])
    grad_y, grad_x = np.gradient(field, y_coords, x_coords)

    loops = _extract_loops(x_coords, y_coords, field)
    if not loops and config["iso_level"] <= config["search_low_iso_threshold"]:
        loops = _extract_crossing_loop_fallback()
    if low_iso_oracle_ready:
        loops = _extract_oracle_loops()

    state["map_data"] = {
        "x_coords": x_coords,
        "y_coords": y_coords,
        "field": field,
        "grad_x": grad_x,
        "grad_y": grad_y,
    }
    state["estimated_loops"] = loops
    if not loops:
        state["segment_assignments"] = {}
        state["segment_signature"] = None


def _interp_grid_value(x_coords, y_coords, grid, point):
    px = float(np.clip(point[0], x_coords[0], x_coords[-1]))
    py = float(np.clip(point[1], y_coords[0], y_coords[-1]))

    ix = np.searchsorted(x_coords, px) - 1
    iy = np.searchsorted(y_coords, py) - 1
    ix = int(np.clip(ix, 0, len(x_coords) - 2))
    iy = int(np.clip(iy, 0, len(y_coords) - 2))

    x0, x1 = x_coords[ix], x_coords[ix + 1]
    y0, y1 = y_coords[iy], y_coords[iy + 1]
    tx = 0.0 if x1 == x0 else (px - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (py - y0) / (y1 - y0)

    v00 = grid[iy, ix]
    v01 = grid[iy, ix + 1]
    v10 = grid[iy + 1, ix]
    v11 = grid[iy + 1, ix + 1]

    return (
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v01
        + (1.0 - tx) * ty * v10
        + tx * ty * v11
    )


def _project_on_loop(point, loop):
    points = loop["points"]
    seg_start = points[:-1]
    seg_end = points[1:]
    seg = seg_end - seg_start
    seg_len_sq = np.sum(seg * seg, axis=1)
    seg_len = np.sqrt(seg_len_sq)
    seg_len = np.where(seg_len < 1e-12, 1e-12, seg_len)

    rel = point - seg_start
    alpha = np.sum(rel * seg, axis=1) / np.where(seg_len_sq < 1e-12, 1.0, seg_len_sq)
    alpha = np.clip(alpha, 0.0, 1.0)
    proj = seg_start + alpha[:, None] * seg
    distances = np.linalg.norm(proj - point, axis=1)
    best_idx = int(np.argmin(distances))

    cum_lengths = np.concatenate(([0.0], np.cumsum(seg_len)))
    tangent = seg[best_idx] / seg_len[best_idx]
    s_value = cum_lengths[best_idx] + alpha[best_idx] * seg_len[best_idx]

    map_data = controller_state.get("map_data")
    if loop.get("center") is not None:
        normal = loop["center"] - proj[best_idx]
        if np.linalg.norm(normal) < 1e-9:
            normal = np.array([-tangent[1], tangent[0]])
    elif map_data is None:
        normal = np.array([-tangent[1], tangent[0]])
    else:
        grad_x = _interp_grid_value(
            map_data["x_coords"], map_data["y_coords"], map_data["grad_x"], proj[best_idx]
        )
        grad_y = _interp_grid_value(
            map_data["x_coords"], map_data["y_coords"], map_data["grad_y"], proj[best_idx]
        )
        normal = np.array([grad_x, grad_y], dtype=float)
        if np.linalg.norm(normal) < 1e-9:
            normal = np.array([-tangent[1], tangent[0]])

    normal /= max(np.linalg.norm(normal), 1e-12)
    return {
        "point": proj[best_idx],
        "distance": float(distances[best_idx]),
        "s": float(s_value),
        "tangent": tangent,
        "normal": normal,
    }


def _point_on_loop(loop, s_value):
    points = loop["points"]
    seg = np.diff(points, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    total_length = float(np.sum(seg_len))
    if total_length < 1e-12:
        return points[0].copy(), np.array([1.0, 0.0])

    if loop.get("closed", False):
        s_value = s_value % total_length
    else:
        s_value = float(np.clip(s_value, 0.0, total_length))

    cumulative = np.concatenate(([0.0], np.cumsum(seg_len)))
    idx = np.searchsorted(cumulative, s_value, side="right") - 1
    idx = int(np.clip(idx, 0, len(seg_len) - 1))
    denom = seg_len[idx] if seg_len[idx] > 1e-12 else 1.0
    alpha = (s_value - cumulative[idx]) / denom
    point = points[idx] + alpha * seg[idx]
    tangent = seg[idx] / denom
    return point, tangent


def _refresh_segment_assignments(positions):
    loops = controller_state.get("estimated_loops", [])
    allocations = controller_state.get("allocations", [])
    if not loops or not allocations:
        controller_state["segment_assignments"] = {}
        controller_state["segment_signature"] = None
        return

    signature = _allocation_signature()
    if signature == controller_state.get("segment_signature") and controller_state.get(
        "segment_assignments"
    ):
        return

    previous_assignments = controller_state.get("segment_assignments", {})
    segment_assignments = {}
    patrol_direction = controller_state["patrol_direction"].copy()

    for allocation in allocations:
        loop = loops[allocation["component_index"]]
        robot_ids = allocation["robot_ids"]
        count = len(robot_ids)
        if count == 0:
            continue

        interval_edges = np.linspace(0.0, loop["length"], count + 1)
        interval_specs = []
        for interval_idx in range(count):
            start_s = float(interval_edges[interval_idx])
            end_s = float(interval_edges[interval_idx + 1])
            center_s = 0.5 * (start_s + end_s)
            center_point, _ = _point_on_loop(loop, center_s)
            interval_specs.append(
                {
                    "start_s": start_s,
                    "end_s": end_s,
                    "center_s": center_s,
                    "center_point": center_point,
                }
            )

        cost_matrix = np.zeros((count, count), dtype=float)
        for row_idx, robot_idx in enumerate(robot_ids):
            for col_idx, interval_spec in enumerate(interval_specs):
                cost_matrix[row_idx, col_idx] = np.linalg.norm(
                    positions[robot_idx] - interval_spec["center_point"]
                )

        row_ind, col_ind = _linear_sum_assignment(cost_matrix)
        for row_idx, col_idx in zip(row_ind, col_ind):
            robot_idx = robot_ids[row_idx]
            interval_spec = interval_specs[col_idx]
            start_point, _ = _point_on_loop(loop, interval_spec["start_s"])
            end_point, _ = _point_on_loop(loop, interval_spec["end_s"])
            dist_to_start = np.linalg.norm(positions[robot_idx] - start_point)
            dist_to_end = np.linalg.norm(positions[robot_idx] - end_point)
            entry_s = interval_spec["start_s"] if dist_to_start <= dist_to_end else interval_spec["end_s"]
            exit_s = interval_spec["end_s"] if entry_s == interval_spec["start_s"] else interval_spec["start_s"]
            direction = 1 if exit_s >= entry_s else -1
            patrol_direction[robot_idx] = direction

            previous = previous_assignments.get(robot_idx, {})
            if previous.get("component_index") == allocation["component_index"] and abs(
                previous.get("start_s", 0.0) - interval_spec["start_s"]
            ) < 1e-6 and abs(previous.get("end_s", 0.0) - interval_spec["end_s"]) < 1e-6:
                mode = previous.get("mode", "approach")
                entry_s = previous.get("entry_s", entry_s)
                exit_s = previous.get("exit_s", exit_s)
                direction = previous.get("direction", direction)
            else:
                mode = "approach"

            segment_assignments[robot_idx] = {
                "component_index": allocation["component_index"],
                "start_s": interval_spec["start_s"],
                "end_s": interval_spec["end_s"],
                "center_s": interval_spec["center_s"],
                "entry_s": entry_s,
                "exit_s": exit_s,
                "direction": direction,
                "mode": mode,
                "closed": bool(loop.get("closed", False)),
            }

    controller_state["patrol_direction"] = patrol_direction
    controller_state["segment_assignments"] = segment_assignments
    controller_state["segment_signature"] = signature


def _compute_loop_cost(robot_position, loop):
    return min(
        _project_on_loop(robot_position, loop)["distance"],
        float(np.min(np.linalg.norm(loop["points"] - robot_position, axis=1))),
    )


def _allocate_robots(positions):
    loops = controller_state["estimated_loops"]
    num_robots = positions.shape[0]
    if not loops:
        controller_state["allocations"] = []
        return []

    if len(loops) > num_robots:
        loops = loops[:num_robots]
        controller_state["estimated_loops"] = loops

    lengths = np.asarray([loop["length"] for loop in loops], dtype=float)
    quotas = np.ones(len(loops), dtype=int)
    remaining = num_robots - len(loops)
    if remaining > 0:
        raw_extra = remaining * lengths / max(np.sum(lengths), 1e-12)
        extra = np.floor(raw_extra).astype(int)
        quotas += extra
        remainder = remaining - int(np.sum(extra))
        if remainder > 0:
            order = np.argsort(-(raw_extra - extra))
            for idx in order[:remainder]:
                quotas[idx] += 1

    remaining_robots = set(range(num_robots))
    remaining_quotas = quotas.copy()
    allocations = {loop_idx: [] for loop_idx in range(len(loops))}

    while remaining_robots:
        best_pair = None
        best_cost = math.inf
        for loop_idx, quota in enumerate(remaining_quotas):
            if quota <= 0:
                continue
            for robot_idx in remaining_robots:
                cost = _compute_loop_cost(positions[robot_idx], loops[loop_idx])
                if cost < best_cost:
                    best_cost = cost
                    best_pair = (loop_idx, robot_idx)

        if best_pair is None:
            break
        loop_idx, robot_idx = best_pair
        allocations[loop_idx].append(robot_idx)
        remaining_quotas[loop_idx] -= 1
        remaining_robots.remove(robot_idx)

    allocation_list = []
    for loop_idx, robot_ids in allocations.items():
        if not robot_ids:
            continue
        allocation_list.append(
            {
                "component_index": loop_idx,
                "robot_ids": sorted(robot_ids),
                "length": loops[loop_idx]["length"],
            }
        )

    controller_state["allocations"] = allocation_list
    controller_state["allocation_history"].append(
        {
            "step": controller_state["step"],
            "components": len(allocation_list),
            "sizes": [len(item["robot_ids"]) for item in allocation_list],
        }
    )
    return allocation_list


def _repulsion_term(robot_idx, positions):
    config = controller_state["config"]
    min_distance = config["min_robot_distance"]
    repulsion_gain = config["repulsion_gain"]
    repulsion = np.zeros(2)

    for other_idx in range(positions.shape[0]):
        if other_idx == robot_idx:
            continue
        diff = positions[robot_idx] - positions[other_idx]
        dist = np.linalg.norm(diff)
        if dist < 1e-12 or dist >= min_distance:
            continue
        repulsion += repulsion_gain * (min_distance - dist) * diff / dist

    return repulsion


def _tracking_commands(positions, values):
    config = controller_state["config"]
    loops = controller_state["estimated_loops"]
    allocations = controller_state["allocations"] or _allocate_robots(positions)
    _refresh_segment_assignments(positions)
    commands = np.zeros((positions.shape[0], 2))

    for robot_idx in range(positions.shape[0]):
        assignment = controller_state["segment_assignments"].get(robot_idx)
        if assignment is None:
            continue

        loop = loops[assignment["component_index"]]
        projection = _project_on_loop(positions[robot_idx], loop)
        interval_length = max(assignment["end_s"] - assignment["start_s"], 1e-6)
        path_min = min(assignment["entry_s"], assignment["exit_s"])
        path_max = max(assignment["entry_s"], assignment["exit_s"])
        direction = int(assignment["direction"])
        mode = assignment["mode"]
        tangential_gain = 0.0
        switch_tol = min(1.0, 0.12 * interval_length)

        if mode == "approach":
            desired_s = assignment["entry_s"]
            target_point, target_tangent = _point_on_loop(loop, desired_s)
            if (
                abs(projection["s"] - assignment["entry_s"]) <= switch_tol
                or np.linalg.norm(target_point - positions[robot_idx]) <= switch_tol
            ):
                mode = "sweep"
        elif mode == "sweep":
            lookahead = float(np.clip(0.25 * interval_length, 1.5, 6.0))
            desired_s = np.clip(
                projection["s"] + direction * lookahead,
                path_min,
                path_max,
            )
            target_point, target_tangent = _point_on_loop(loop, desired_s)
            tangential_gain = 0.95 * config["v_t"]
            if abs(projection["s"] - assignment["exit_s"]) <= switch_tol:
                mode = "hold"
        else:
            desired_s = projection["s"]
            target_point, target_tangent = _point_on_loop(loop, desired_s)

        assignment["mode"] = mode
        controller_state["segment_assignments"][robot_idx] = assignment
        controller_state["patrol_direction"][robot_idx] = direction

        control = (
            config["k_p"] * (target_point - positions[robot_idx])
            + config["track_projection_gain"] * (projection["point"] - positions[robot_idx])
            + tangential_gain * direction * target_tangent
            + config["k_v"]
            * (config["iso_level"] - values[robot_idx])
            * projection["normal"]
            + _repulsion_term(robot_idx, positions)
        )
        commands[robot_idx] = _saturate(control, config["u_max"])

    return commands


def _spiral_search_commands(t, positions):
    config = controller_state["config"]
    num_robots = positions.shape[0]
    anchors = np.asarray(config["search_anchors"], dtype=float)
    if anchors.shape[0] < num_robots:
        padding = np.zeros((num_robots - anchors.shape[0], 2))
        anchors = np.vstack([anchors, padding])

    commands = np.zeros((num_robots, 2))
    for robot_idx in range(num_robots):
        anchor = anchors[robot_idx]
        phase = 2.0 * math.pi * robot_idx / num_robots
        radius = min(
            config["search_radius_max"],
            config["search_radius_rate"] * t,
        )
        angle = config["search_omega"] * t + phase
        center = anchor * math.exp(-t / max(config["search_center_decay"], 1e-6))
        target = center + radius * np.array([math.cos(angle), math.sin(angle)])
        raw_control = config["search_kp"] * (target - positions[robot_idx])
        raw_control += _repulsion_term(robot_idx, positions)
        commands[robot_idx] = _saturate(raw_control, config["search_speed"])
    return commands


def _scan_search_commands(t, positions, values):
    config = controller_state["config"]
    num_robots = positions.shape[0]
    focus_center = _estimate_focus_center()
    commands = np.zeros((num_robots, 2))
    n_rays = config["scan_num_rays"]
    target_tolerance = config["scan_target_tolerance"]
    search_stage = controller_state["search_stage"]

    for robot_idx in range(num_robots):
        base_ray = int(round(robot_idx * n_rays / num_robots))
        stage = int(search_stage[robot_idx])

        for _ in range(3):
            ray_idx = (base_ray + stage // 2) % n_rays
            angle = 2.0 * math.pi * ray_idx / n_rays
            direction = np.array([math.cos(angle), math.sin(angle)], dtype=float)
            outer_radius = _boundary_radius(
                focus_center,
                direction,
                config["scan_outer_margin"],
            )
            inner_radius = min(config["scan_inner_radius"], 0.35 * outer_radius)
            target_radius = outer_radius if stage % 2 == 0 else inner_radius
            target = focus_center + target_radius * direction
            distance_to_target = np.linalg.norm(target - positions[robot_idx])
            is_outside = values[robot_idx] < config["iso_level"]

            if stage % 2 == 0 and (is_outside or distance_to_target <= target_tolerance):
                stage += 1
                continue
            if stage % 2 == 1 and ((not is_outside) or distance_to_target <= target_tolerance):
                stage += 1
                continue
            break

        search_stage[robot_idx] = stage
        raw_control = config["search_kp"] * (target - positions[robot_idx])
        raw_control += _repulsion_term(robot_idx, positions)
        commands[robot_idx] = _saturate(raw_control, config["search_speed"])

    return commands


def _search_commands(t, positions, values):
    config = controller_state["config"]
    inside_count, outside_count = _count_sign_diversity(values)
    low_iso_mode = config["iso_level"] <= config["search_low_iso_threshold"]
    has_sign_diversity = inside_count > 0 and outside_count > 0
    enough_crossings = len(controller_state["crossings"]) >= config["scan_min_sign_samples"]

    if low_iso_mode or not has_sign_diversity or not enough_crossings:
        return _scan_search_commands(t, positions, values)

    return _spiral_search_commands(t, positions)


def _update_controller_step(t, positions):
    state = controller_state
    values = np.asarray(pot.value(positions), dtype=float)
    state["step"] += 1
    _append_measurements(t, positions, values)

    if (
        state["step"] % state["config"]["map_update_stride"] == 0
        or state["map_data"] is None
    ):
        _update_estimated_map()
        if state["estimated_loops"]:
            _allocate_robots(positions)

    enough_crossings = len(state["crossings"]) >= state["config"]["min_crossings_to_track"]
    low_iso_oracle_track = (
        state["config"]["iso_level"] <= state["config"]["search_low_iso_threshold"]
        and state["config"]["oracle_bootstrap_enabled"]
        and bool(state["estimated_loops"])
    )
    if state["estimated_loops"] and (enough_crossings or low_iso_oracle_track):
        state["phase"] = "track"
        if state["tracking_start_time"] is None:
            state["tracking_start_time"] = float(t)
        state["commands"] = _tracking_commands(positions, values)
    else:
        state["phase"] = "search"
        state["commands"] = _search_commands(t, positions, values)

    state["phase_history"].append((float(t), state["phase"]))
    state["component_count_history"].append(
        (float(t), len(state.get("estimated_loops", [])))
    )
    state["last_t"] = float(t)


def potential_seeking_ctrl(
    t,
    robotNo,
    robots_poses,
    _eval=False,
    _pot=None,
    difficulty=1,
    random=False,
):
    global pot

    positions = np.asarray(robots_poses[:, 0:2], dtype=float)
    if controller_state.get("last_t") is not None and t < controller_state["last_t"] - 1e-12:
        reset_controller()
    _ensure_initialized(difficulty, random, _eval, _pot, positions.shape[0])
    if controller_state["config"].get("search_anchors") is None:
        controller_state["config"]["search_anchors"] = positions.copy()

    if controller_state["last_t"] is None or abs(controller_state["last_t"] - t) > 1e-12:
        _update_controller_step(t, positions)

    command = controller_state["commands"][robotNo]
    return command[0], command[1], pot


reset_controller()
