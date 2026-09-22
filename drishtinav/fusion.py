"""GNSS + INS fusion: a vehicle-constrained EKF with AI-adapted pseudo-measurements.

State  x = [E, N, psi, v, b_g, b_a]
    E, N   position in the local tangent plane (m)
    psi    vehicle heading (rad, CCW from East)
    v      forward speed (m/s)
    b_g    gyro yaw-rate bias (rad/s)
    b_a    longitudinal accelerometer bias incl. residual road grade (m/s^2)

Mechanisation uses the vehicle-frame yaw rate and longitudinal specific force.
Velocity is modelled *along the heading only*, i.e. the non-holonomic
constraint (no side-slip, no vertical motion) is built into the process model
rather than added as a soft update.

Measurement models (all optional, all gated by a chi-square innovation test):
    GNSS position / speed / course     - when a fix is available
    ZUPT + ZARU (v = 0, b_g = w)       - vehicle stationary (rule + AI detector)
    AI pseudo-odometer (v = v_nn)      - SpeedNet speed with its *own* predicted
                                         variance as R (AI-adaptive noise)
    Map cross-track + road heading     - HMM map-matched road, when confident
"""
from __future__ import annotations

import numpy as np

from .geo import wrap_angle

CHI2_1 = 10.83   # 99.9 % gate, 1 dof
CHI2_2 = 13.82   # 99.9 % gate, 2 dof


class VehicleEKF:
    N = 6

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.x = np.zeros(self.N)
        self.P = np.diag([1e4, 1e4, np.pi ** 2, 25.0, 0.02 ** 2, 0.3 ** 2])
        self.pos_init = False
        self.head_init = False
        self.stats = {"rejected": 0, "updates": 0}

    # ----------------------------------------------------------------- predict
    def predict(self, yaw_rate: float, a_fwd: float, dt: float, accel_noise_scale: float = 1.0):
        c = self.cfg
        E, N, psi, v, bg, ba = self.x
        w = yaw_rate - bg
        psi_m = psi + 0.5 * w * dt
        cm, sm = np.cos(psi_m), np.sin(psi_m)
        self.x[0] = E + v * cm * dt
        self.x[1] = N + v * sm * dt
        self.x[2] = wrap_angle(psi + w * dt)
        self.x[3] = v + (a_fwd - ba) * dt
        F = np.eye(self.N)
        F[0, 2] = -v * sm * dt
        F[0, 3] = cm * dt
        F[0, 4] = 0.5 * v * sm * dt * dt
        F[1, 2] = v * cm * dt
        F[1, 3] = sm * dt
        F[1, 4] = -0.5 * v * cm * dt * dt
        F[2, 4] = -dt
        F[3, 5] = -dt
        q = np.array([
            c["q_pos"] * dt, c["q_pos"] * dt,
            (c["gyro_noise"] ** 2) * dt,
            (c["accel_noise"] * accel_noise_scale) ** 2 * dt,
            (c["gyro_bias_rw"] ** 2) * dt,
            (c["accel_bias_rw"] ** 2) * dt,
        ])
        self.P = F @ self.P @ F.T + np.diag(q)

    # ------------------------------------------------------------------ update
    def _update(self, z, h, H, R, gate, angle_idx=None, update_states=None):
        y = np.atleast_1d(z - h)
        if angle_idx is not None:
            y[angle_idx] = wrap_angle(y[angle_idx])
        S = H @ self.P @ H.T + R
        Sinv = np.linalg.inv(S)
        nis = float(y @ Sinv @ y)
        if gate is not None and nis > gate:
            self.stats["rejected"] += 1
            return False, nis
        K = self.P @ H.T @ Sinv
        if update_states is not None:
            # Schmidt "consider" update: the measurement may only correct the
            # listed states; the others keep their estimate (covariance stays
            # consistent through the Joseph form below).
            mask = np.zeros(self.N, bool)
            mask[list(update_states)] = True
            K[~mask] = 0.0
        self.x = self.x + K @ y
        self.x[2] = wrap_angle(self.x[2])
        I_KH = np.eye(self.N) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.stats["updates"] += 1
        return True, nis

    def init_position(self, en, sigma):
        self.x[0:2] = en
        self.P[0:2, :] = 0
        self.P[:, 0:2] = 0
        self.P[0, 0] = self.P[1, 1] = sigma ** 2
        self.pos_init = True

    def init_heading(self, psi, speed, sigma=np.radians(10)):
        self.x[2] = psi
        self.x[3] = speed
        self.P[2, :] = 0
        self.P[:, 2] = 0
        self.P[2, 2] = sigma ** 2
        self.head_init = True

    def update_gnss_pos(self, en, sigma, gate=True):
        H = np.zeros((2, self.N)); H[0, 0] = H[1, 1] = 1
        only = None if self.cfg.get("gnss_pos_updates_biases", True) else (0, 1, 2, 3)
        return self._update(np.asarray(en), self.x[:2], H, np.eye(2) * sigma ** 2, CHI2_2 if gate else None,
                            update_states=only)

    def update_gnss_speed(self, speed, sigma, gate=True):
        H = np.zeros((1, self.N)); H[0, 3] = 1
        return self._update(np.array([speed]), self.x[3:4], H, np.array([[sigma ** 2]]), CHI2_1 if gate else None)

    def update_heading(self, psi, sigma, gate=True):
        H = np.zeros((1, self.N)); H[0, 2] = 1
        return self._update(np.array([psi]), self.x[2:3], H, np.array([[sigma ** 2]]), CHI2_1 if gate else None,
                            angle_idx=0)

    def update_zupt(self, yaw_rate_raw):
        H = np.zeros((2, self.N)); H[0, 3] = 1; H[1, 4] = 1
        R = np.diag([self.cfg["zupt_sigma"] ** 2, self.cfg["zaru_sigma"] ** 2])
        return self._update(np.array([0.0, yaw_rate_raw]), np.array([self.x[3], self.x[4]]), H, R, None)

    def update_ai_speed(self, speed, sigma):
        H = np.zeros((1, self.N)); H[0, 3] = 1
        s = max(self.cfg["ai_sigma_floor"], sigma * self.cfg["ai_sigma_scale"])
        return self._update(np.array([speed]), self.x[3:4], H, np.array([[s * s]]), CHI2_1 * 4)

    def update_map(self, road_xy, road_heading, cross_sigma, heading_sigma=None):
        """Cross-track distance to the matched road = 0 (+ optional road heading).

        Map constraints are geometric: they may move the position and rotate the
        heading, but must not leak into speed or sensor biases through the
        filter's cross-covariances (a wrong match would otherwise corrupt the
        speed estimate and cascade into wrong turns).
        """
        n = np.array([-np.sin(road_heading), np.cos(road_heading)])
        H = np.zeros((1, self.N)); H[0, 0:2] = n
        ok, _ = self._update(np.array([0.0]), np.array([n @ (self.x[:2] - road_xy)]), H,
                             np.array([[cross_sigma ** 2]]), CHI2_1, update_states=(0, 1, 2))
        if heading_sigma is not None:
            Hh = np.zeros((1, self.N)); Hh[0, 2] = 1
            self._update(np.array([road_heading]), self.x[2:3], Hh, np.array([[heading_sigma ** 2]]), CHI2_1,
                         angle_idx=0, update_states=(0, 1, 2))
        return ok

    # ---------------------------------------------------------------- outputs
    @property
    def pos_sigma(self):
        return float(np.sqrt(max(self.P[0, 0] + self.P[1, 1], 0.0) / 2))

    @property
    def heading_sigma(self):
        return float(np.sqrt(max(self.P[2, 2], 0.0)))
