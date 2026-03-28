# -*- coding: utf-8 -*-
"""
author: Sylvain Bertrand, 2023

   All variables are in SI units


   Variables used by the functions of this script
    - t: time instant (s)
    - robotNo: no of the current robot for which control is coputed (0 .. nbRobots-1)
    - poses:  size (3 x nbRobots)
        eg. of use: the pose of robot 'robotNo' can be obtained by: poses[:,robotNo]
            poses[robotNo,0]: x-coordinate of robot position (in m)
            poses[robotNo,1]: y-coordinate of robot position (in m)
            poses[robotNo,2]: orientation angle of robot (in rad)   (in case of unicycle dynamics only)
"""

import numpy as np
import math
from lib.potential import Potential
from lib.simulation import generate_init_positions


# =============================================================================
# CONSTANTS  —  edit all tuning parameters here
# =============================================================================

# --- Memory ---
MAX_HIST            = 500     # maximum number of (x, v) samples kept per robot

# --- Gradient estimation ---
GRADIENT_RADIUS     = 2.0     # neighbourhood radius for local least-squares fit
GRADIENT_MIN_POINTS = 4       # minimum samples required for a valid gradient estimate
G_HAT_EMA_ALPHA     = 0.3     # EMA smoothing factor for the gradient (0 = frozen, 1 = raw)

# --- Mode switching ---
LOCK_RADIUS         = 2.0     # distance below which a robot is considered "at" a known max
SEARCH_RADIUS       = 25.0    # radius of the search boundary circle
BORDER_MARGIN       = 1.0     # robot enters border mode when r >= SEARCH_RADIUS - BORDER_MARGIN
MODE_TIMER_STEPS    = 5       # number of steps a mode is held after a transition
ANGLE_TOWARD_MAX    = 20.0    # (deg) gradient cone that counts as "pointing toward known max"
ANGLE_AWAY_FROM_MAX = 45.0    # (deg) gradient cone that counts as "pointing away" (wider hysteresis)

# --- Exploration control ---
EXPLORATION_GAIN_AWAY = 1.0   # repulsion gain from the known maximum
EXPLORATION_GAIN_PERP = 0.6   # lateral (perpendicular) gradient steering gain

# --- Border control ---
BORDER_GAIN_TANGENT = 2.0     # tangential speed along the boundary circle
BORDER_GAIN_RADIAL  = 1.0     # radial correction to stay on the circle

# --- Final formation ---
FORMATION_RADIUS      = 3.0   # radius of the circle formed around the global max
FORMATION_GAIN        = 1.0   # inter-robot spacing gain (keep_formation)
FORMATION_ANCHOR_GAIN = 0.8   # centroid-to-global-max attraction gain

# --- Dither ---
DITHER_AMPLITUDE  = 1         # amplitude of the rotating dither signal
DITHER_FREQUENCY  = 0.5       # angular frequency of the dither (rad/s)
DITHER_HISTORY    = 100       # number of recent v samples used to compute the cushion

# --- Safety ---
REPULSION_D_MIN = 0.5         # minimum inter-robot distance before repulsion kicks in
REPULSION_GAIN  = 0.8         # repulsion gain
U_MAX           = 10          # saturation norm for the final control output

# --- Maximum detection ---
MAX_DETECTION_WINDOW    = 20    # sliding window length (steps)
MAX_DETECTION_EPSILON_V = 0.01  # max allowed value spread inside the window
MAX_DETECTION_EPSILON_X = 0.2   # max allowed distance from current pos to best pos in window
MAX_CHECK_MIN_DIST      = 1.0   # minimum distance between two distinct maxima
MIN_VALID_POTENTIAL     = 300   # a candidate maximum must exceed this value


# ==============   "GLOBAL" VARIABLES KNOWN BY ALL THE FUNCTIONS ==============
# all variables declared here will be known by functions below
# use keyword "global" inside a function if the variable needs to be modified by the function

global firstCall   # can be used to check the first call ever of a function
firstCall = True
verbose = [True for i in range(5)]

global pot # DO NOT MODIFY - allows initialisation of potential function from this script

_robot_mem = {}

# =============================================================================
# ROBOT MEMORY
# =============================================================================
def init_robot_memory(i, x_i, v_i):
    """Create memory entries for all robots if needed."""
    global _robot_mem

    if i not in _robot_mem:
        _robot_mem[i] = {
            "x_hist": [x_i],
            "v_hist": [v_i],
            "phase": 2.0 * np.pi * np.random.rand(),
            "found_max": None,          # value of found maximum
            "found_max_pos": None,      # position where it was found
            "mode": "gradient_ascent",
            "g_hat_smooth": np.zeros(2, dtype=float),  # B: EMA-smoothed gradient
            "mode_timer": 0,                           # C: steps left in mode lock
        }


def update_robot_memories(i, x_i, v_i, max_hist=MAX_HIST):
    """
    Append current sample for every robot.
    Keeps trajectories separated by robot.
    """
    global _robot_mem

    if _robot_mem[i]["found_max"] is not None and v_i > _robot_mem[i]["found_max"]:
        _robot_mem[i]["found_max"] = v_i
        _robot_mem[i]["found_max_pos"] = x_i.copy()

    _robot_mem[i]["x_hist"].append(x_i)
    _robot_mem[i]["v_hist"].append(float(v_i))

    if max_hist is not None:
        _robot_mem[i]["x_hist"] = _robot_mem[i]["x_hist"][-max_hist:]
        _robot_mem[i]["v_hist"] = _robot_mem[i]["v_hist"][-max_hist:]


# =============================================================================
# GRADIENT ESTIMATION
# =============================================================================
def get_local_samples(x_i, x_current, v_current, radius):
    global _robot_mem

    X_list = []
    V_list = []

    r2 = radius * radius

    # 1) current positions of all robots
    for k in range(len(x_current)):
        dx = x_current[k] - x_i
        if np.dot(dx, dx) <= r2:
            X_list.append(np.array(x_current[k], dtype=float))
            V_list.append(float(v_current[k]))

    # 2) past positions of all robots
    for mem in _robot_mem.values():
        x_hist = mem["x_hist"]
        v_hist = mem["v_hist"]

        for xp, vp in zip(x_hist, v_hist):
            xp = np.array(xp, dtype=float)
            dx = xp - x_i
            if np.dot(dx, dx) <= r2:
                X_list.append(xp)
                V_list.append(float(vp))

    if len(X_list) == 0:
        return np.zeros((0, 2), dtype=float), np.zeros(0, dtype=float)

    return np.array(X_list, dtype=float), np.array(V_list, dtype=float)


def estimate_gradient(mem, x_i, v_i):
    """Simple one-point gradient using the latest sample in memory."""
    dx = x_i - mem["x_hist"][-1]
    dv = float(v_i - mem["v_hist"][-1])

    g_hat = np.zeros(2, dtype=float)
    dx2 = float(np.dot(dx, dx))

    if dx2 > 1e-8:
        g_hat = (dv / dx2) * dx

    return g_hat, dx2


def estimate_gradient_local(x_i, x_current, v_current, radius=GRADIENT_RADIUS, min_points=GRADIENT_MIN_POINTS):
    """Least-squares affine fit over all samples within radius of x_i."""
    X_loc, V_loc = get_local_samples(x_i, x_current, v_current, radius)
    n_used = len(V_loc)

    if n_used < min_points:
        return np.zeros(2, dtype=float), n_used

    # Center coordinates around x_i
    DX = X_loc - x_i   # shape (M,2)

    # Design matrix for local affine fit: V ≈ a + gx*dx + gy*dy
    A = np.column_stack([
        np.ones(n_used),
        DX[:, 0],
        DX[:, 1]
    ])

    # Least-squares fit
    coeffs, _, _, _ = np.linalg.lstsq(A, V_loc, rcond=None)

    # coeffs = [a, gx, gy]
    g_hat = np.array([coeffs[1], coeffs[2]], dtype=float)

    return g_hat, n_used


# =============================================================================
# MODE SWITCHING
# =============================================================================
def points_toward_direction(vec, direction, angle_deg=10.0, eps=1e-8):
    nv = np.linalg.norm(vec)
    nd = np.linalg.norm(direction)

    if nv < eps or nd < eps:
        return False

    cosang = float(np.dot(vec, direction) / (nv * nd))
    cosang = np.clip(cosang, -1.0, 1.0)

    return cosang >= np.cos(np.deg2rad(angle_deg))


def update_robot_mode(i, x_i, g_hat_smooth, lock_radius=LOCK_RADIUS, search_radius=SEARCH_RADIUS, border_margin=BORDER_MARGIN):
    global _robot_mem
    mem = _robot_mem[i]
    maxima = [
        mem_j["found_max_pos"]
        for mem_j in _robot_mem.values()
        if mem_j.get("found_max_pos") is not None
    ]

    # Three maxima known → final_formation (except robot sitting on the global max)
    if len(maxima) >= 3:
        best_val = -np.inf
        closest_id = None
        for j, mem_j in _robot_mem.items():
            if mem_j["found_max"] is not None and mem_j["found_max"] > best_val:
                best_val = mem_j["found_max"]
                closest_id = j
        if i == closest_id:
            mem["mode"] = "gradient_ascent"
            return mem["mode"], None
        mem["mode"] = "final_formation"
        return mem["mode"], None

    # No maxima yet → pure gradient ascent
    if len(maxima) == 0:
        mem["mode"] = "gradient_ascent"
        return mem["mode"], None

    # This robot already found a max → keep ascending
    if mem["found_max"] is not None:
        mem["mode"] = "gradient_ascent"
        return mem["mode"], None

    closest_max = min(maxima, key=lambda m: np.linalg.norm(np.array(m) - x_i))
    away       = x_i - closest_max
    dist_to_max = np.linalg.norm(away)

    # Hold mode for at least 5 steps
    if mem["mode_timer"] > 0:
        mem["mode_timer"] -= 1

    else:
        #   exploration → gradient_ascent : gradient must point clearly AWAY (>45°)
        #   gradient_ascent → exploration : gradient must point clearly TOWARD (<20°)
        angle_threshold = ANGLE_AWAY_FROM_MAX if mem["mode"] in ("exploration", "border") else ANGLE_TOWARD_MAX
        points_to_found_max = points_toward_direction(g_hat_smooth, -away, angle_deg=angle_threshold)

        if dist_to_max <= lock_radius or points_to_found_max:
            new_mode_candidate = "exploration"

            # Border check (only from exploration)
            r = np.linalg.norm(x_i)
            if r >= search_radius - border_margin:
                new_mode_candidate = "border"
        else:
            new_mode_candidate = "gradient_ascent"

        # Start the persistence lock whenever the mode actually changes
        if new_mode_candidate != mem["mode"]:
            mem["mode_timer"] = MODE_TIMER_STEPS
        mem["mode"] = new_mode_candidate

    return mem["mode"], closest_max if mem["mode"] == "exploration" else None


# =============================================================================
# SEEK CONTROLS
# =============================================================================
def exploration_control(i, x_i, g_hat, max_pos, gain_away=EXPLORATION_GAIN_AWAY, gain_perp=EXPLORATION_GAIN_PERP, eps=1e-8):
    """
    Exploration mode: move away from the known maximum while using the
    gradient's perpendicular component to steer toward new high-value regions.

    - gain_away  : strength of the repulsion from the known max
    - gain_perp  : strength of the lateral gradient steering
                   (parallel component toward the known max is stripped out
                   so the robot cannot be pulled back into the known basin)
    """
    away = np.array(x_i, dtype=float) - np.array(max_pos, dtype=float)
    away_norm = np.linalg.norm(away)

    if away_norm < eps:
        return np.zeros(2, dtype=float)

    away_dir = away / away_norm

    # Remove the gradient component pointing back toward the known max
    toward_max = -away_dir
    g_parallel = np.dot(g_hat, toward_max) * toward_max
    g_perp     = g_hat - g_parallel

    return gain_away * away_dir + gain_perp * g_perp


def border_control(i, x_i, r_border=SEARCH_RADIUS, gain_tangent=BORDER_GAIN_TANGENT, gain_radial=BORDER_GAIN_RADIAL):
    """
    Follow the search boundary circle.
    Even robots: clockwise. Odd robots: counter-clockwise.
    Exit is handled by update_robot_mode when gradient no longer points to known max.
    """
    x_i = np.asarray(x_i, float)
    r = np.linalg.norm(x_i)

    if r < 1e-8:
        return np.array([gain_tangent, 0.0])

    e_r = x_i / r                          # outward radial unit vector
    e_t = np.array([-e_r[1], e_r[0]])      # CCW tangent

    if i % 2 == 0:                         # even → clockwise
        e_t = -e_t

    u_radial = -gain_radial * (r - r_border) * e_r   # stay on the circle
    return gain_tangent * e_t + u_radial


def keep_formation(N, x, i, radius=2.0, gain=1.0):
    """Consensus-style circular formation control."""
    A = np.ones((N, N)) - np.eye(N)

    angles = 2.0 * np.pi * np.arange(N) / N
    rel = np.column_stack((np.cos(angles), np.sin(angles))) * radius

    u = np.zeros(2, dtype=float)
    for j in range(N):
        u -= gain * A[i, j] * ((x[i] - x[j]) - (rel[i] - rel[j]))
    return u


def final_formation_control(N, x, i, radius=FORMATION_RADIUS, gain_formation=FORMATION_GAIN, gain_anchor=FORMATION_ANCHOR_GAIN):
    """
    Form a circle of robots around the global maximum.
    keep_formation handles inter-robot spacing; gain_anchor pulls the centroid
    toward the global max so the group drifts there as a unit.
    """
    global _robot_mem

    best_val, global_max_pos = -np.inf, None
    for mem_j in _robot_mem.values():
        if mem_j["found_max"] is not None and mem_j["found_max"] > best_val:
            best_val = mem_j["found_max"]
            global_max_pos = mem_j["found_max_pos"]

    if global_max_pos is None:
        return np.zeros(2)

    global_max_pos = np.array(global_max_pos, dtype=float)

    u_form = keep_formation(N, x, i, radius=radius, gain=gain_formation)

    centroid = np.mean(x, axis=0)
    u_anchor = gain_anchor * (global_max_pos - centroid)

    return u_form + u_anchor


def compute_seek_control(i, N, x, new_mode, g_hat, max_pos):
    """Dispatch to the appropriate control law for the current mode."""
    x_i = x[i]

    if new_mode == "gradient_ascent":
        return g_hat

    if new_mode == "exploration":
        return exploration_control(i, x_i, g_hat, max_pos)

    if new_mode == "border":
        return border_control(i, x_i)

    if new_mode == "final_formation":
        return final_formation_control(N, x, i)

    return np.zeros(2, dtype=float)


# =============================================================================
# DITHER
# =============================================================================
def make_dither(t, phase, amplitude=DITHER_AMPLITUDE, frequency=DITHER_FREQUENCY):
    """Small rotating exploration signal."""
    return amplitude * np.array([
        np.cos(frequency * t + phase),
        np.sin(frequency * t + phase)
    ], dtype=float)


def compute_dither_term(t, mem, v_i, new_mode, explore, amplitude=DITHER_AMPLITUDE, frequency=DITHER_FREQUENCY):
    """
    Dither to add on top of u_seek.
    - Zero for border and final_formation (prescribed paths).
    - Full amplitude when gradient info is poor (explore=True).
    - Cushioned by how far v_i is below the recent peak otherwise.
      Exploration mode gets 2x the cushion to help escape known maxima basins.
    """
    if new_mode in ("border", "final_formation"):
        return np.zeros(2, dtype=float)

    recent = mem["v_hist"][-DITHER_HISTORY:]
    v_max_seen = max(recent) if recent else max(v_i, 1e-3)
    cushion = max((v_max_seen - v_i) / max(v_max_seen, 1e-3), 0)

    dither = make_dither(t, mem["phase"], amplitude=amplitude, frequency=frequency)

    if explore:
        return dither
    if new_mode == "exploration":
        return 2 * cushion * dither
    return cushion * dither   # gradient_ascent


# =============================================================================
# SAFETY
# =============================================================================
def repulsion_control(N, x, i, d_min=REPULSION_D_MIN, gain=REPULSION_GAIN, eps=1e-6):
    """Collision avoidance by short-range repulsion."""
    u = np.zeros(2, dtype=float)

    for j in range(N):
        if j == i:
            continue

        rij = x[i] - x[j]
        dist = float(np.linalg.norm(rij))

        if eps < dist < d_min:
            penetration = d_min - dist
            direction = rij / dist
            magnitude = gain * penetration
            u += magnitude * direction

    return u


def sat_norm(u, umax):
    n = np.linalg.norm(u)
    if n <= umax or n < 1e-12:
        return u
    return (umax / n) * u


# =============================================================================
# DETECT MAXIMUM
# =============================================================================
def check_maximum(i, x_candidate, min_dist=MAX_CHECK_MIN_DIST):
    """Check whether x_candidate is a new maximum (not already found by another robot)."""
    global _robot_mem

    for j, mem_j in _robot_mem.items():
        if j == i:
            continue

        x_found = mem_j.get("found_max_pos")
        if x_found is None:
            continue

        if np.linalg.norm(np.array(x_candidate) - np.array(x_found)) < min_dist:
            return False

    return True


def detect_local_maximum(i, epsilon_v=MAX_DETECTION_EPSILON_V, epsilon_x=MAX_DETECTION_EPSILON_X, window=MAX_DETECTION_WINDOW):
    """Detect if robot i has likely reached a local maximum."""
    global _robot_mem

    mem = _robot_mem.get(i)
    if mem is None:
        return False

    if mem["found_max"] is not None:
        return True

    v_hist, x_hist = mem["v_hist"], mem["x_hist"]
    if len(v_hist) < window + 1:
        return False

    v_window = v_hist[-window:]
    v_best = max(v_window)
    stagnant = (v_best - min(v_window)) < epsilon_v

    idx_best = len(v_window) - 1 - v_window[::-1].index(v_best)
    x_best = np.array(x_hist[-window + idx_best])
    near_best = np.linalg.norm(np.array(x_hist[-1]) - x_best) < epsilon_x

    found = stagnant and near_best and (v_best > MIN_VALID_POTENTIAL)

    if found and check_maximum(i, x_best):
        mem["found_max"] = v_best
        mem["found_max_pos"] = x_best.copy()
        return True

    return False


# =============================================================================
# MAIN CONTROLLER
# =============================================================================
def potential_seeking_ctrl(
    t,
    robotNo,
    robots_poses,
    _eval=False,
    _pot=None,
    difficulty=1,
    random=False
):
    global pot

    # -------------------------------------------------------------------------
    # 1. Initialization
    # -------------------------------------------------------------------------
    global firstCall
    global pot
    global _robot_mem

    # --- part to be run only once ---
    if firstCall:
        if not _eval:
            pot = Potential(difficulty=difficulty, random=random)
        else:
            pot = _pot
        firstCall = False

    # -------------------------------------------------------------------------
    # 2. Basic quantities
    # -------------------------------------------------------------------------
    N = robots_poses.shape[0]
    i = robotNo

    x = robots_poses[:, 0:2]
    x_i = x[i]

    pot_measurement = np.array([pot.value(xi) for xi in x], dtype=float)
    v_i = pot_measurement[i]

    # Memory
    init_robot_memory(i, x_i, v_i)
    mem = _robot_mem[i]

    # -------------------------------------------------------------------------
    # 3. Local gradient estimate
    # -------------------------------------------------------------------------
    one_point_gradient = False
    if mem["found_max"] is not None:
        one_point_gradient = True

    if one_point_gradient:
        g_hat, dx2 = estimate_gradient(mem, x_i, v_i)
        explore = dx2 <= 0
    else:
        g_hat, n_used = estimate_gradient_local(x_i, x, pot_measurement)
        explore = n_used < 4

    # -------------------------------------------------------------------------
    # 4. Mode update  (uses EMA-smoothed gradient to reduce noise sensitivity)
    # -------------------------------------------------------------------------
    mem["g_hat_smooth"] = G_HAT_EMA_ALPHA * g_hat + (1.0 - G_HAT_EMA_ALPHA) * mem["g_hat_smooth"]

    old_mode = mem["mode"]
    new_mode, max_pos = update_robot_mode(i, x_i, mem["g_hat_smooth"])

    if new_mode != old_mode:
        print(f"[MODE] Robot {i} → switching to {new_mode}")

    # -------------------------------------------------------------------------
    # 5. Main seek control + dither
    # -------------------------------------------------------------------------
    u_seek = compute_seek_control(i, N, x, new_mode, g_hat, max_pos)
    u_dither = compute_dither_term(t, mem, v_i, new_mode, explore)

    # -------------------------------------------------------------------------
    # 6. Safety + final control term
    # -------------------------------------------------------------------------
    if new_mode != "final_formation":
        u_safety = repulsion_control(N, x, i)
    else:
        u_safety = np.zeros(2, dtype=float)

    u_des = u_safety + u_seek + u_dither

    u = sat_norm(u_des, U_MAX)

    # -------------------------------------------------------------------------
    # 7. Memory update
    # -------------------------------------------------------------------------
    update_robot_memories(i, x_i, v_i)

    # -------------------------------------------------------------------------
    # 8. Check if the robot has found a maximum
    # -------------------------------------------------------------------------
    detect_local_maximum(i)

    if mem["found_max"] is not None and verbose[i]:
        print(
            f"Robot {i} found a maximum at t = {t:.1f},"
            f"(V ≈ {mem['found_max']:.3f}, "
            f"X={mem['x_hist'][-1][0]:.1f}, "
            f"Y={mem['x_hist'][-1][1]:.1f}, "
        )
        verbose[i] = False

    return float(u[0]), float(u[1]), pot
