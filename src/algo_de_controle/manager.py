# -*- coding: utf-8 -*-
"""
Minimal ControlManager (single `default` mode).

Responsibilities:
- Hold basic simulation bounds and speed settings.
- Wrap a `GradientSeeker` instance and provide a `control()` method
  that returns a 2D velocity command for a robot.

Behavior:
- For each robot, call `seeker.update()` to get (gvx, gvy).
- If the base command has a positive y component, apply it as-is.
  Otherwise apply the opposite vector (flip both components) so the
  robot moves upward.
- Enforce horizontal boundary preference and clip to `vmax`.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


class ControlManager:
    def __init__(self, nb_robots, vmax=5.0, cruise_scale=0.5,
                 xmin=-25., xmax=25., ymin=-25., ymax=25., tol_pos=1.0):
        self.nb_robots = int(nb_robots)
        self.vmax = float(vmax)
        self.cruise_speed = float(cruise_scale) * self.vmax
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.ymin = float(ymin)
        self.ymax = float(ymax)
        self.tol_pos = float(tol_pos)

        # minimal commanded speed (fraction of vmax)
        self.min_speed = 0.5 * self.vmax
        self.min_yspeed = 0.8 * self.min_speed

        # internal seeker used for gradient-based suggestions
        from gradient_seeker import GradientSeeker
        # per-robot rectangular areas (cover full y range)
        total_width = (self.xmax - self.xmin)
        self.area_width = total_width / float(self.nb_robots) if self.nb_robots > 0 else total_width
        self.areas = []
        for i in range(self.nb_robots):
            axmin = self.xmin + i * self.area_width
            axmax = axmin + self.area_width
            self.areas.append((axmin, axmax))
        # precompute area centers (at ymin) for initial convergence
        self.area_centers = [((a[0] + a[1]) / 2.0) for a in self.areas]
        self.seeker = GradientSeeker(self.nb_robots, history_len=8, gain=0.8, max_speed=self.vmax)
        # exploration bookkeeping for gathering phase
        self.at_top = [False] * self.nb_robots
        self.gathering = False
        self.best_pos = None
        self.best_val = -float('inf')
        self.kp_gather = 1.0
        # initialization phase: converge to center_x at y = ymin
        self.init_phase = True
        self.init_reached = [False] * self.nb_robots
        self.kp_init = 1.0
        
        # assignment for initial targets (computed once at start of init phase)
        self.init_assigned = False
        self.init_assignment = None
        # per-robot assigned area index (defaults to identity mapping)
        self.assigned_areas = list(range(self.nb_robots))
        self.bounds_tol = 0.5

    def _clip_speed(self, vx, vy):
        s = np.hypot(vx, vy)
        if s > self.vmax and s > 0:
            scale = self.vmax / s
            vx *= scale
            vy *= scale
        return float(vx), float(vy)

    def _ensure_min_speed(self, vx, vy):
        """Ensure the vector (vx,vy) has norm >= self.min_speed.

        If norm==0, return an upward vector of magnitude min_speed.
        If 0<norm<min_speed, scale up preserving direction.
        """
        vx = float(vx)
        vy = float(vy)
        s = np.hypot(vx, vy)
        if s >= self.min_speed:
            return vx, vy
        if s > 0.0:
            scale = (self.min_speed / s)
            vx, vy = vx * scale, vy * scale
            if vy <= self.min_yspeed:
                vy = float(self.min_yspeed)
            return vx, vy
        # zero vector -> provide small upward motion
        return 0.0, float(self.min_speed)

    def control(self, t, robot_no, robots_poses, pot=None):
        """Return (vx, vy) for robot `robot_no`.

        Args:
            t: current time (unused)
            robot_no: index of robot
            robots_poses: array-like (N x 2 or N x 3) with robot states
            pot: optional Potential (passed to seeker for measurement)

        Behavior: single default mode described in module docstring.
        """
        r = int(robot_no)
        pos = np.asarray(robots_poses[r, :2], dtype=float)

        # Compute an optimal assignment from current robot positions to
        # the set of starting targets (area centers at y = ymin) once.
        if self.init_phase and (not self.init_assigned):
            poses = np.asarray(robots_poses)[:, :2]
            targets = np.array([[x, self.ymin] for x in self.area_centers], dtype=float)
            # cost: Euclidean distance
            cost = np.linalg.norm(poses[:, None, :] - targets[None, :, :], axis=2)
            row_ind, col_ind = linear_sum_assignment(cost)
            assignment = np.empty(self.nb_robots, dtype=int)
            assignment[row_ind] = col_ind
            self.init_assignment = assignment.tolist()
            # set per-robot assigned area according to the optimization result
            self.assigned_areas = self.init_assignment.copy()
            self.init_assigned = True

        # Initialization phase: converge to the assigned target (area center at y = ymin)
        if self.init_phase:
            # default to own index center if assignment not computed for some reason
            assigned_idx = (self.init_assignment[r] if (self.init_assignment is not None) else r)
            tx = float(self.area_centers[int(assigned_idx)])
            ty = float(self.ymin)
            dx = tx - pos[0]
            dy = ty - pos[1]
            # if close enough mark reached and stop
            if np.hypot(dx, dy) <= self.tol_pos:
                self.init_reached[r] = True
                if all(self.init_reached):
                    self.init_phase = False
                return 0.0, 0.0
            vx = self.kp_init * dx
            vy = self.kp_init * dy
            return float(vx), float(vy)

        # get current potential reading if available and update best-known
        current_pot = float(pot.value(pos)) if pot is not None else 0.0
        if pot is not None:
            if current_pot > self.best_val:
                self.best_val = current_pot
                self.best_pos = pos.copy()

        # ask seeker for a suggested velocity
        gvx, gvy = self.seeker.update(r, pos, current_pot) if self.seeker is not None else (0.0, self.cruise_speed)

        # If currently gathering (all robots reached top), move proportionally to best_pos
        if self.gathering and (self.best_pos is not None):
            dx = float(self.best_pos[0]) - pos[0]
            dy = float(self.best_pos[1]) - pos[1]
            # if close enough, stop
            if np.hypot(dx, dy) <= 1e-3:
                return 0.0, 0.0
            vx = self.kp_gather * dx
            vy = self.kp_gather * dy
            return float(vx), float(vy)

        # If base command has positive y, use it; otherwise invert vector
        # so we always have an upward motion component.
        if gvy >= 0.0:
            vx_use = float(gvx)
            vy_use = float(gvy)
        else:
            vx_use = -float(gvx)
            vy_use = -float(gvy)

        # prefer inward motion when near the edges of the robot's assigned area
        x, y = pos
        # use the area assigned by the initialization assignment
        assigned_area_idx = (self.assigned_areas[r] if (r >= 0 and r < len(self.assigned_areas)) else r)
        axmin, axmax = self.areas[int(assigned_area_idx)]
        if x <= axmin + self.bounds_tol:
            vx_use = max(vx_use, 0.0)
        if x >= axmax - self.bounds_tol:
            vx_use = min(vx_use, 0.0)

        # If robot reached the top, stop (unless gathering has started)
        if pos[1] >= (self.ymax - self.tol_pos):
            self.at_top[r] = True
            # if all robots at top, start gathering
            if (not self.gathering) and all(self.at_top):
                self.gathering = True
            if not self.gathering:
                return 0.0, 0.0

        vx_use, vy_use = self._ensure_min_speed(vx_use, vy_use)
        vx_use, vy_use = self._clip_speed(vx_use, vy_use)
        return vx_use, vy_use
