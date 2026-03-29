# -*- coding: utf-8 -*-
"""
Simplified ControlManager implementing the requested behaviors:

- Maintains a shared list of detected maxima (`max_list`). Each entry:
  {'pos': np.array([x,y]), 'x_visited': [bool]*N, 'rearranged_x': np.linspace(...)}
- Default phase: robots ascend (cruise upward) while allowing lateral
  movement at borders; vy is kept non-negative (always points upward).
- When robot reaches the y level of a detected max, it enters
  'reposition' to pick the closest unused x from `rearranged_x` and move
  horizontally there, then resume ascending.
- If robot gets within `MARKED_AREA_RADIUS` of a detected max while
  ascending, it immediately switches to reposition to the chosen x at
  that max's y.
- When all robots reach the top (`y >= ymax - tol_pos`), they all go to
  the highest-value max recorded (final gathering phase).
"""

import numpy as np

# radius to consider two detections the same marked maximum (meters)
MARKED_AREA_RADIUS = 1.0


class ControlManager:
    def __init__(self, nb_robots, vmax=5.0, cruise_scale=0.1,
                 xmin=-25., xmax=25., ymin=-25., ymax=25., tol_pos=1.0):
        self.nb_robots = int(nb_robots)
        self.vmax = float(vmax)
        self.cruise_speed = float(cruise_scale) * self.vmax
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.ymin = float(ymin)
        self.ymax = float(ymax)
        self.tol_pos = float(tol_pos)

        # lightweight seeker (kept as attribute to be set externally if desired)
        # expected to be set by caller as manager.seeker = GradientSeeker(...)
        from gradient_seeker import GradientSeeker
        self.seeker = GradientSeeker(self.nb_robots, history_len=8, gain=0.8, max_speed=self.vmax)

        # per-robot state
        self.mode = ['default'] * self.nb_robots     # 'default' | 'reposition' | 'at_top' | 'goto_max'
        self.reposition_target = [None] * self.nb_robots  # (x_target, y_target, max_idx)

        # simple potential-history based detection
        # require `inc_required` increases then `dec_required` decreases in a sliding window
        self.inc_required = 2
        self.dec_required = 2
        self.window_len = self.inc_required + self.dec_required + 1
        self.pot_hist = [list() for _ in range(self.nb_robots)]
        self.found_max = [False] * self.nb_robots

        # shared detected maxima list
        # each entry: {'pos': np.array([x,y]), 'x_visited': [False]*N, 'rearranged_x': np.linspace(xmin,xmax,N)}
        self.max_list = []

        # final gathering target (index into max_list) once all at top
        self.final_target_idx = None

    def _clip_speed(self, vx, vy):
        s = np.hypot(vx, vy)
        if s > self.vmax and s > 0:
            scale = self.vmax / s
            vx *= scale
            vy *= scale
        return float(vx), float(vy)

    def _choose_x_for_max(self, max_entry, pos_x):
        # choose closest unused x from rearranged_x
        refs = np.asarray(max_entry['rearranged_x'])
        used = np.asarray(max_entry['x_visited'], dtype=bool)
        candidates = np.where(~used)[0]
        if candidates.size == 0:
            return None, None
        sub = candidates
        idx = sub[int(np.argmin(np.abs(refs[sub] - pos_x)))]
        return float(refs[idx]), int(idx)

    def _add_marked_position(self, pos):
        # if near existing, do not add; otherwise create new entry
        pos = np.asarray(pos, dtype=float)
        for idx, m in enumerate(self.max_list):
            if np.linalg.norm(m['pos'] - pos) <= MARKED_AREA_RADIUS:
                return idx
        entry = {
            'pos': pos.copy(),
            'x_visited': [False] * self.nb_robots,
            # generate nb_robots + 2 points and drop the extremes so no robot is assigned edges
            'rearranged_x': np.linspace(self.xmin, self.xmax, self.nb_robots + 2)[1:-1]
        }
        self.max_list.append(entry)
        return len(self.max_list) - 1

    def control(self, t, robot_no, robots_poses, pot=None):
        r = int(robot_no)
        pos = np.asarray(robots_poses[r, :2], dtype=float)

        # final check: are all robots at top?
        all_at_top = all(np.asarray(robots_poses[:, 1]) >= (self.ymax - self.tol_pos))
        if all_at_top and len(self.max_list) > 0:
            # compute best max by potential value
            if self.final_target_idx is None and pot is not None:
                vals = [float(pot.value(m['pos'])) for m in self.max_list]
                self.final_target_idx = int(np.argmax(vals))
                # switch robots to goto_max
                for i in range(self.nb_robots):
                    self.mode[i] = 'goto_max'

        # if robot is already at top
        if pos[1] >= (self.ymax - self.tol_pos):
            # Only mark 'at_top' and idle if final_target not yet decided.
            if self.final_target_idx is None:
                self.mode[r] = 'at_top'
                return 0.0, 0.0

        # GOTO_FINAL_MAX behavior: simple proportional control (no seeker, no clipping)
        if self.mode[r] == 'goto_max' and self.final_target_idx is not None:
            target = self.max_list[self.final_target_idx]['pos']
            dx = float(target[0]) - pos[0]
            dy = float(target[1]) - pos[1]
            if np.hypot(dx, dy) <= MARKED_AREA_RADIUS:
                return 0.0, 0.0
            kp = 1.0
            vx = kp * dx
            vy = kp * dy
            return float(vx), float(vy)

        # If currently repositioning: force horizontal move toward target x
        if self.mode[r] == 'reposition':
            tx, ty, midx = self.reposition_target[r]
            if tx is None:
                # fallback to default
                self.mode[r] = 'default'
            else:
                dx = tx - pos[0]
                if abs(dx) <= self.tol_pos:
                    # reached lateral reference -> resume ascending
                    self.mode[r] = 'default'
                    self.reposition_target[r] = None
                    return 0.0, self.cruise_speed
                # horizontal motion only
                vx = np.clip(dx * 2.0, -self.vmax, self.vmax)
                vy = 0.0
                vx, vy = self._clip_speed(vx, vy)
                return vx, vy

        # DEFAULT (ascending / exploration) behavior
        # update potential history (for simple peak detection)
        current_pot = float(pot.value(pos)) if pot is not None else 0.0
        ph = self.pot_hist[r]
        ph.append(current_pot)
        if len(ph) > self.window_len:
            ph.pop(0)

        # detection of local maximum using sliding window: inc_required increases
        # followed by dec_required decreases
        if (not self.found_max[r]) and len(ph) >= self.window_len:
            w = ph[-self.window_len:]
            inc = self.inc_required
            dec = self.dec_required
            inc_ok = all(w[i] < w[i+1] for i in range(0, inc))
            dec_ok = all(w[inc + j] > w[inc + j + 1] for j in range(0, dec))
            if inc_ok and dec_ok:
                self.found_max[r] = True
                midx = self._add_marked_position(pos)

        # run seeker to update its internal history and get a suggested direction
        gvx, gvy = self.seeker.update(r, pos, current_pot) if pot is not None else self.seeker.update(r, pos, 0.0)

        # If while default ascending we get close to an existing marked max, switch to reposition for its level
        for idx, m in enumerate(self.max_list):
            if np.linalg.norm(m['pos'] - pos) <= MARKED_AREA_RADIUS:
                # pick x and go there at y = m['pos'][1]
                chosen_x, xi = self._choose_x_for_max(m, pos[0])
                if chosen_x is not None:
                    m['x_visited'][xi] = True
                    self.reposition_target[r] = (chosen_x, float(m['pos'][1]), idx)
                    self.mode[r] = 'reposition'
                    return self.control(t, r, robots_poses, pot)

        # If robot crosses the y-level of any detected max (within tol), enter reposition
        for idx, m in enumerate(self.max_list):
            if abs(pos[1] - float(m['pos'][1])) <= self.tol_pos:
                chosen_x, xi = self._choose_x_for_max(m, pos[0])
                if chosen_x is not None:
                    m['x_visited'][xi] = True
                    self.reposition_target[r] = (chosen_x, float(m['pos'][1]), idx)
                    self.mode[r] = 'reposition'
                    return self.control(t, r, robots_poses, pot)

        # Normal ascending: combine seeker lateral suggestion with enforced upward component
        # keep vy at least cruise_speed (always upward)
        vy_use = max(gvy, self.cruise_speed)
        vx_use = gvx

        # if at vertical boundaries keep vy positive and prefer move inward
        x, y = pos
        if x <= self.xmin + 1e-6:
            vx_use = max(vx_use, 0.0)
        if x >= self.xmax - 1e-6:
            vx_use = min(vx_use, 0.0)

        vx_use, vy_use = self._clip_speed(vx_use, vy_use)
        return vx_use, vy_use
