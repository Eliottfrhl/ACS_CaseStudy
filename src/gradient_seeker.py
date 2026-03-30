# -*- coding: utf-8 -*-
import numpy as np


class GradientSeeker:
    def __init__(self, nb_robots, history_len=10, gain=1.0, max_speed=1.5):
        self.nb = int(nb_robots)
        self.history_len = int(history_len)
        self.gain = float(gain)
        self.max_speed = float(max_speed)

        self.pos_hist = [list() for _ in range(self.nb)]
        self.v_hist = [list() for _ in range(self.nb)]

    def _is_psd_2x2(self, H, tol=1e-8):
        if not np.allclose(H, H.T):
            return False
        eigvals = np.linalg.eigvalsh(H)
        return np.all(eigvals >= -tol)

    def update(self, robot_no, position, measurement):

        r = int(robot_no)
        x = np.asarray(position, dtype=float).reshape(2)
        v = float(measurement)

        self.pos_hist[r].append(x)
        self.v_hist[r].append(v)

        if len(self.pos_hist[r]) > self.history_len:
            self.pos_hist[r].pop(0)
            self.v_hist[r].pop(0)

        n_samples = len(self.pos_hist[r])
        small_step = self.max_speed / 10.0

        # -------------------------------------------------
        # 1) Exploration forcée sur les deux premiers pas
        # -------------------------------------------------
        if n_samples == 1:
            # petit pas vers la gauche
            return -small_step, 0.0

        if n_samples == 2:
            # petit pas vers la droite
            return small_step, 0.0

        # -------------------------------------------------
        # 2) 3 à 5 points → modèle linéaire
        # -------------------------------------------------
        if n_samples < 6:

            X = np.vstack(self.pos_hist[r])
            V = np.asarray(self.v_hist[r])

            A_lin = np.hstack((X, np.ones((X.shape[0], 1))))

            try:
                sol, *_ = np.linalg.lstsq(A_lin, V, rcond=None)
            except Exception:
                return 0.0, 0.0

            grad = sol[:2]

            vx, vy = self.gain * grad
            speed = np.hypot(vx, vy)
            if speed > self.max_speed and speed > 0:
                scale = self.max_speed / speed
                vx *= scale
                vy *= scale

            return float(vx), float(vy)

        # -------------------------------------------------
        # 3) ≥ 6 points → modèle quadratique
        # -------------------------------------------------
        X = np.vstack(self.pos_hist[r])
        V = np.asarray(self.v_hist[r])

        Phi = []
        for xi in X:
            xx, yy = xi
            Phi.append([
                xx**2,
                yy**2,
                2*xx*yy,
                xx,
                yy,
                1.0
            ])
        Phi = np.array(Phi)

        try:
            theta, *_ = np.linalg.lstsq(Phi, V, rcond=None)
        except Exception:
            return 0.0, 0.0

        a11, a22, a12, b1, b2, _ = theta

        A_quad = np.array([[a11, a12],
                           [a12, a22]])

        b_vec = np.array([b1, b2])

        grad = 2 * A_quad @ x + b_vec
        H = 2 * A_quad

        # -------------------------------------------------
        # 4) Test convexité locale
        # -------------------------------------------------
        if self._is_psd_2x2(H):
            norm = np.linalg.norm(grad)
            if norm > 0:
                direction = grad / norm
                vx, vy = self.max_speed * direction
            else:
                vx, vy = 0.0, 0.0
        else:
            vx, vy = self.gain * grad
            speed = np.hypot(vx, vy)
            if speed > self.max_speed and speed > 0:
                scale = self.max_speed / speed
                vx *= scale
                vy *= scale

        return float(vx), float(vy)