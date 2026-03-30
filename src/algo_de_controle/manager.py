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
        # per-robot initial vertical target choice: 'ymin' or 'ymax'
        self.init_targets = [None] * self.nb_robots
        self.bounds_tol = 0.5

    def _clip_speed(self, vx, vy):
        s = np.hypot(vx, vy)
        if s > self.vmax and s > 0:
            scale = self.vmax / s
            vx *= scale
            vy *= scale
        return float(vx), float(vy)

    def _ensure_min_speed(self, vx, vy, robot_no=None):
        """Ensure the vector (vx,vy) has norm >= self.min_speed and
        that the vertical component matches the robot's intended initial
        movement direction (up for `ymin`, down for `ymax`).

        If `robot_no` is provided and `self.init_targets[robot_no]` is
        set to `'ymax'`, a downward minimum y-speed is enforced.
        """
        vx = float(vx)
        vy = float(vy)
        s = np.hypot(vx, vy)

        # preferred vertical direction: +1 for up (ymin), -1 for down (ymax)
        pref = 1
        if (robot_no is not None) and (0 <= robot_no < len(self.init_targets)):
            if self.init_targets[robot_no] == 'ymax':
                pref = -1

        if s >= self.min_speed:
            # ensure vertical component has at least min_yspeed in preferred dir
            if pref * vy < self.min_yspeed:
                vy = float(pref * self.min_yspeed)
            return vx, vy

        if s > 0.0:
            scale = (self.min_speed / s)
            vx, vy = vx * scale, vy * scale
            if pref * vy < self.min_yspeed:
                vy = float(pref * self.min_yspeed)
            return vx, vy

        # zero vector -> provide small motion in preferred vertical direction
        return 0.0, float(pref * self.min_speed)

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
            centers = np.array(self.area_centers, dtype=float)
            # compute cost matrix: for each robot i and each area j,
            # cost = min(distance to (center_j, ymin), distance to (center_j, ymax))
            dx = poses[:, None, 0] - centers[None, :]
            dy_ymin = poses[:, None, 1] - float(self.ymin)
            dy_ymax = poses[:, None, 1] - float(self.ymax)
            dist_ymin = np.hypot(dx, dy_ymin[:, :, None].squeeze()) if False else np.sqrt(dx**2 + dy_ymin**2)
            dist_ymax = np.sqrt(dx**2 + dy_ymax**2)
            cost = np.minimum(dist_ymin, dist_ymax)
            # solve assignment robot->area
            row_ind, col_ind = linear_sum_assignment(cost)
            assignment = np.empty(self.nb_robots, dtype=int)
            assignment[row_ind] = col_ind
            # record assigned area per robot
            self.init_assignment = assignment.tolist()
            self.assigned_areas = self.init_assignment.copy()
            # set init_targets (ymin or ymax) for each assigned robot
            # row_ind maps robot indices from assignment; iterate pairs
            for ri, ai in zip(row_ind.tolist(), col_ind.tolist()):
                # choose ymin if it's closer or equal
                cx = float(self.area_centers[ai])
                d_ymin = np.hypot(poses[ri,0]-cx, poses[ri,1]-float(self.ymin))
                d_ymax = np.hypot(poses[ri,0]-cx, poses[ri,1]-float(self.ymax))
                self.init_targets[ri] = 'ymin' if d_ymin <= d_ymax else 'ymax'
            self.init_assigned = True

        # Initialization phase: converge to the assigned target (area center at y = ymin)
        if self.init_phase:
            # default to own index center if assignment not computed for some reason
            assigned_idx = (self.init_assignment[r] if (self.init_assignment is not None) else r)
            tx = float(self.area_centers[int(assigned_idx)])
            # target y depends on assignment (ymin or ymax)
            ty = float(self.ymin) if (self.init_targets[r] == 'ymin') else float(self.ymax)
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
        # so we always have an upward motion component for robots assigned to ymin.
        # For robots assigned to ymax we prefer downward motion.
        target_y = (self.init_targets[r] if (r >= 0 and r < len(self.init_targets)) else 'ymin')
        if target_y == 'ymax':
            # prefer negative y (downwards)
            if gvy <= 0.0:
                vx_use = float(gvx)
                vy_use = float(gvy)
            else:
                vx_use = -float(gvx)
                vy_use = -float(gvy)
        else:
            # default: prefer upward motion
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

        # Stop condition depends on where the robot started in init phase:
        # - if it started at ymin (init_targets == 'ymin'), stop when reaching ymax
        # - if it started at ymax (init_targets == 'ymax'), stop when reaching ymin
        start_side = (self.init_targets[r] if (r >= 0 and r < len(self.init_targets)) else 'ymin')
        reached_target = False
        if start_side == 'ymax':
            # robot started at ymax and moves down: stop at ymin
            if pos[1] <= (self.ymin + self.tol_pos):
                reached_target = True
        else:
            # default: robot started at ymin and moves up: stop at ymax
            if pos[1] >= (self.ymax - self.tol_pos):
                reached_target = True

        if reached_target:
            self.at_top[r] = True
            # if all robots reached their respective targets, start gathering
            if (not self.gathering) and all(self.at_top):
                self.gathering = True
            if not self.gathering:
                return 0.0, 0.0

        vx_use, vy_use = self._ensure_min_speed(vx_use, vy_use, r)
        vx_use, vy_use = self._clip_speed(vx_use, vy_use)
        return vx_use, vy_use
