"""In-vehicle alignment: phone (sensor) frame -> vehicle frame (x fwd, y left, z up).

A phone in a dashboard holder sits at an arbitrary pitch/roll/yaw relative to
the car, and the mount changes every trip. We estimate the rotation online:

1. **Pitch & roll (levelling)** - the gravity vector seen by the accelerometer
   when the car is not accelerating. Updated fast while stationary, slowly
   while cruising at constant speed.
2. **Yaw misalignment (heading of the phone vs. the car's nose)** - during
   straight-line acceleration / braking the horizontal specific force points
   along the car's longitudinal axis. With GNSS we regress the levelled
   horizontal acceleration on the GNSS-derived longitudinal acceleration
   dv/dt, which also resolves the forward/backward sign. (Wang et al.,
   PLANS 2023; Chen, Zhang & Niu, IEEE T-ITS 2020.)
3. **Gyro yaw axis** - normally the accelerometer's up axis. But loggers may
   permute gyro columns (AndroSensor in IO-VNBD does) and a phone wobbling in
   its holder shows large pitch/roll rates, so in ``auto`` mode we regress the
   GNSS course rate on the 3-axis gyro (ridge least squares over 1 s GNSS
   intervals). The solution vector is the yaw axis *with its sign*, and its
   norm is the gyro scale factor. Until enough turning has been observed the
   accelerometer up axis is used.
4. **Mount disturbance** - a phone slipping or being re-seated in its holder is a
   step change of the gravity direction in the sensor frame. A fast gravity
   estimate is compared with a slow reference whenever the car is not
   accelerating or turning; a sustained difference > ``mount_step_deg`` triggers
   re-levelling and re-learning of the forward and yaw axes (and the engine
   restarts the gyro-bias estimate).
"""
from __future__ import annotations

import numpy as np

G = 9.80665


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


class MountAlignment:
    def __init__(self, fs: float, gyro_frame: str = "auto", mount_step_deg: float = 12.0):
        self.fs = fs
        self.gyro_frame = gyro_frame
        self.g_vec = None
        self.g_fast = None
        self.g_ref = None
        self.mount_step_deg = mount_step_deg
        self.mount_events = 0
        self._step_count = 0
        self.ref_axis = None
        self.m = 0.0                 # yaw misalignment angle in the horizontal basis
        self.m_obs = 0.0             # accumulated |dv| evidence
        self.S = np.zeros(2)         # sum(a_h * dv/dt)
        self.S_aa = 0.0
        self.S_vv = 0.0
        self.fwd_known = False
        # gyro yaw-axis regression (GNSS course rate ~ c . gyro)
        self.S_gg = np.zeros((3, 3))
        self.S_gc = np.zeros(3)
        self.S_cc = 0.0
        self.n_gc = 0
        self.turn_ev = 0.0
        self.gyro_up = None           # unit yaw axis in the gyro frame (sign included)
        self.gyro_scale = 1.0
        self.gyro_separate = False
        self.gyro_version = 0         # bumps when the yaw axis estimate moves materially
        # interval accumulators between GNSS fixes
        self._acc_h = np.zeros(2)
        self._gyr_vec = np.zeros(3)
        self._n_int = 0
        self._yaw_abs = 0.0

    # ------------------------------------------------------------------ frames
    @property
    def up(self):
        return _unit(self.g_vec) if self.g_vec is not None else np.array([0, 0, 1.0])

    def _basis(self):
        u = self.up
        if self.ref_axis is None:
            self.ref_axis = np.eye(3)[int(np.argmin(np.abs(u)))]
        h1 = _unit(self.ref_axis - np.dot(self.ref_axis, u) * u)
        h2 = np.cross(u, h1)
        return u, h1, h2

    def rotation(self):
        """Rows = vehicle axes (fwd, left, up) expressed in the accelerometer frame."""
        u, h1, h2 = self._basis()
        f = np.cos(self.m) * h1 + np.sin(self.m) * h2
        l = np.cross(u, f)
        return np.vstack([f, l, u])

    def euler_deg(self):
        """Mount angles for display: (roll, pitch, yaw misalignment) in degrees."""
        R = self.rotation()
        up = R[2]
        roll = np.degrees(np.arctan2(up[1], up[2]))
        pitch = np.degrees(np.arctan2(-up[0], np.hypot(up[1], up[2])))
        yaw = np.degrees(np.arctan2(R[0, 1], R[0, 0]))
        return float(roll), float(pitch), float(yaw)

    # ------------------------------------------------------------------ update
    def update(self, accel, gyro, static: bool, dvdt_hint: float = float("nan")):
        accel = np.asarray(accel, float)
        gyro = np.asarray(gyro, float)
        if self.g_vec is None:
            self.g_vec = accel.copy()
        # Levelling: trust the accelerometer as a gravity sensor when the car is
        # stationary (fast) or its specific force magnitude is ~g (slow).
        dev = abs(np.linalg.norm(accel) - G)
        if static and dev < 0.5:          # reject shocks (door slam, pothole) even at rest
            a = 1.0 / (0.5 * self.fs)
        elif dev < 0.3 and np.linalg.norm(gyro) < 0.05 and abs(dvdt_hint) < 0.15:
            # steady cruise: no turn (centripetal) and no longitudinal accel
            # (dv/dt from the navigation filter; NaN -> unknown -> no update)
            a = 1.0 / (20.0 * self.fs)
        else:
            a = 0.0
        self.g_vec += a * (accel - self.g_vec)
        self._check_mount(accel, gyro, static, dvdt_hint, dev)

        u, h1, h2 = self._basis()
        self._acc_h += np.array([np.dot(accel, h1), np.dot(accel, h2)])
        self._gyr_vec += gyro
        self._yaw_abs += abs(self.yaw_rate(gyro))
        self._n_int += 1

    def _check_mount(self, accel, gyro, static, dvdt_hint, dev):
        """Detect a sudden re-orientation of the phone in its holder.

        Evidence is collected only in calm moments (stopped, or cruising straight
        at constant speed) and accumulates across them: a fast gravity estimate
        (1 s) is compared with a slow reference (60 s). Road grade changes stay
        below the threshold; a phone slipping in its holder does not.
        """
        if self.g_fast is None:
            self.g_fast = accel.copy()
            self.g_ref = accel.copy()
        calm = static or (np.isfinite(dvdt_hint) and abs(dvdt_hint) < 0.3 and np.linalg.norm(gyro) < 0.1)
        if not calm or dev > 0.6:
            return
        self.g_fast += (1.0 / (1.0 * self.fs)) * (accel - self.g_fast)
        self.g_ref += (1.0 / (60.0 * self.fs)) * (accel - self.g_ref)
        ang = np.degrees(np.arccos(np.clip(_unit(self.g_fast) @ _unit(self.g_ref), -1, 1)))
        if ang > self.mount_step_deg:
            self._step_count += 1
        elif ang < 0.5 * self.mount_step_deg:
            self._step_count = 0
        if self._step_count > 3 * self.fs:
            self._remount()

    def _remount(self):
        self.mount_events += 1
        self.g_vec = self.g_fast.copy()
        self.g_ref = self.g_fast.copy()
        self.ref_axis = None
        self.m, self.m_obs, self.S[:], self.S_aa, self.S_vv, self.fwd_known = 0.0, 0.0, 0.0, 0.0, 0.0, False
        self.S_gg[:], self.S_gc[:], self.S_cc, self.n_gc, self.turn_ev = 0.0, 0.0, 0.0, 0, 0.0
        self.gyro_up, self.gyro_scale, self.gyro_separate = None, 1.0, False
        self.gyro_version += 1
        self._step_count = 0

    def yaw_rate(self, gyro):
        """Vehicle yaw rate (rad/s, CCW positive) from a raw gyro sample."""
        if self.gyro_up is not None:
            return float(np.dot(gyro, self.gyro_up) * self.gyro_scale)
        return float(np.dot(gyro, self.up))

    def on_gnss(self, dvdt: float, course_rate: float, speed: float):
        """Consume the accumulated interval since the last fix with GNSS kinematics."""
        if self._n_int == 0:
            return
        a_h = self._acc_h / self._n_int
        g_mean = self._gyr_vec / self._n_int
        yaw_abs = self._yaw_abs / self._n_int
        self._acc_h[:] = 0.0
        self._gyr_vec[:] = 0.0
        self._yaw_abs = 0.0
        self._n_int = 0
        if speed < 2.0 or not np.isfinite(dvdt):
            return
        # Yaw misalignment: straight-line longitudinal manoeuvres only.
        if yaw_abs < 0.05 and abs(dvdt) > 0.3:
            self.S += a_h * dvdt
            self.S_aa += float(a_h @ a_h)
            self.S_vv += dvdt * dvdt
            self.m_obs += abs(dvdt)
            self.m = float(np.arctan2(self.S[1], self.S[0]))
            if self.m_obs > 8.0:
                corr = np.linalg.norm(self.S) / np.sqrt(self.S_aa * self.S_vv + 1e-9)
                self.fwd_known = corr > 0.3
        # Gyro yaw axis + sign + scale from GNSS course rate.
        if self.gyro_frame == "auto" and np.isfinite(course_rate) and speed > 4.0 and abs(course_rate) < 1.0:
            self.S_gg += np.outer(g_mean, g_mean)
            self.S_gc += g_mean * course_rate
            self.S_cc += course_rate * course_rate
            self.n_gc += 1
            self.turn_ev += abs(course_rate)
            if self.turn_ev > 3.0:     # ~170 deg of accumulated turning
                # ridge scaled to the data: axes the gyro never excites (a clean,
                # rigidly mounted IMU) must not pick up weight from noise
                lam = 1e-2 * np.trace(self.S_gg) + 1e-9
                c = np.linalg.solve(self.S_gg + lam * np.eye(3), self.S_gc)
                nrm = np.linalg.norm(c)
                if nrm > 0.5:
                    new_up = c / nrm
                    ref = self.gyro_up if self.gyro_up is not None else self.up
                    # a material change of the yaw axis (e.g. discovering permuted gyro
                    # columns) invalidates the learnt bias; small refinements do not
                    if np.degrees(np.arccos(np.clip(new_up @ ref, -1, 1))) > 20.0:
                        self.gyro_version += 1
                    self.gyro_up = new_up
                    # scale: 1-D least squares along the axis (the ridge would bias it low),
                    # applied only when it differs from 1 by > 3 standard errors: 1 Hz GNSS
                    # course noise makes the estimate weak, and gyros are factory-trimmed.
                    den = float(new_up @ self.S_gg @ new_up)
                    if den > 0 and self.n_gc > 10:
                        sc = float((new_up @ self.S_gc) / den)
                        rss = max(self.S_cc - 2 * sc * float(new_up @ self.S_gc) + sc * sc * den, 0.0)
                        se = np.sqrt(rss / (self.n_gc - 1) / den)
                        self.gyro_scale = float(np.clip(sc, 0.9, 1.1)) if abs(sc - 1.0) > 3 * se else 1.0
                    self.gyro_separate = abs(np.dot(self.gyro_up, self.up)) < 0.9

    def to_vehicle(self, accel, gyro):
        """Return (a_fwd, a_lat, a_up - g, yaw_rate) in the vehicle frame."""
        R = self.rotation()
        a = R @ np.asarray(accel, float)
        return float(a[0]), float(a[1]), float(a[2] - G), self.yaw_rate(np.asarray(gyro, float))

    @property
    def state(self):
        roll, pitch, yaw = self.euler_deg()
        return {"roll_deg": roll, "pitch_deg": pitch, "yaw_mis_deg": yaw, "forward_converged": bool(self.fwd_known),
                "gyro_frame": "separate" if self.gyro_separate else "shared",
                "gyro_axis_calibrated": self.gyro_up is not None, "gyro_scale": self.gyro_scale,
                "mount_events": self.mount_events}


def align_batch(accel, gyro, speed_ref, t, yaw_rate_ref=None):
    """Offline alignment for building training sets (uses reference speed/yaw).

    Returns vehicle-frame arrays (a_fwd, a_lat, a_up_dev, yaw_rate) and the
    rotation used. Mirrors ``MountAlignment`` but estimates once per segment.
    """
    accel = np.asarray(accel, float)
    gyro = np.asarray(gyro, float)
    # Up: mean specific force (vehicle accelerations average out over minutes).
    u = _unit(np.median(accel, axis=0))
    ref = np.eye(3)[int(np.argmin(np.abs(u)))]
    h1 = _unit(ref - np.dot(ref, u) * u)
    h2 = np.cross(u, h1)
    dvdt = np.gradient(speed_ref, t)
    k = max(1, int(round(1.0 / np.median(np.diff(t)))))
    ker = np.ones(k) / k
    a1 = np.convolve(accel @ h1, ker, "same")
    a2 = np.convolve(accel @ h2, ker, "same")
    dv = np.convolve(dvdt, ker, "same")
    sel = (np.abs(dv) > 0.3) & (speed_ref > 2)
    m = np.arctan2(np.sum(a2[sel] * dv[sel]), np.sum(a1[sel] * dv[sel]))
    f = np.cos(m) * h1 + np.sin(m) * h2
    l = np.cross(u, f)
    R = np.vstack([f, l, u])
    av = accel @ R.T
    # Gyro yaw axis: regress the reference yaw rate on the 3-axis gyro (same
    # estimator as MountAlignment.on_gnss, but with the reference signal).
    if yaw_rate_ref is not None:
        mov = speed_ref > 2
        G_ = gyro[mov]
        S_gg, S_gc = G_.T @ G_, G_.T @ yaw_rate_ref[mov]
        c = np.linalg.solve(S_gg + 1e-2 * np.trace(S_gg) * np.eye(3), S_gc)
        axis = c / np.linalg.norm(c)
        yaw = gyro @ axis * np.clip((axis @ S_gc) / (axis @ S_gg @ axis), 0.9, 1.1)
    else:
        yaw = gyro @ u
    return av[:, 0], av[:, 1], av[:, 2] - G, yaw, R
