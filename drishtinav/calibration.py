"""Streaming IMU conditioning: vibration filtering, stationarity, bumps, gyro bias.

All classes are causal, O(1) per sample and free of SciPy so the identical
logic runs in the phone app (see web/mobile/engine.js) and on an edge CPU.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class Biquad:
    """2nd-order Butterworth low-pass (bilinear transform), one filter per channel.

    Engine harmonics (25-100 Hz at idle/cruise) and road texture alias into a
    10 Hz phone stream as broadband noise; a 2-3 Hz low-pass keeps the vehicle
    dynamics (braking, turning take >0.5 s) and removes most of it.
    """

    def __init__(self, fs: float, fc: float, channels: int = 1):
        fc = min(fc, 0.45 * fs)
        k = np.tan(np.pi * fc / fs)
        q = 1 / np.sqrt(2)
        norm = 1 / (1 + k / q + k * k)
        self.b = np.array([k * k * norm, 2 * k * k * norm, k * k * norm])
        self.a = np.array([1.0, 2 * (k * k - 1) * norm, (1 - k / q + k * k) * norm])
        self.z = np.zeros((2, channels))
        self.primed = False

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        if not self.primed:  # start at steady state to avoid a start-up transient
            self.z[0] = (1 - self.b[0]) * x
            self.z[1] = (self.b[2] - self.a[2]) * x
            self.primed = True
        y = self.b[0] * x + self.z[0]
        self.z[0] = self.b[1] * x - self.a[1] * y + self.z[1]
        self.z[1] = self.b[2] * x - self.a[2] * y
        return y


class RunningStats:
    """Mean / std of a scalar over a sliding window."""

    def __init__(self, n: int):
        self.buf = deque(maxlen=n)
        self.s = 0.0
        self.s2 = 0.0

    def push(self, x: float):
        if len(self.buf) == self.buf.maxlen:
            old = self.buf[0]
            self.s -= old
            self.s2 -= old * old
        self.buf.append(x)
        self.s += x
        self.s2 += x * x

    @property
    def full(self):
        return len(self.buf) == self.buf.maxlen

    @property
    def std(self):
        n = len(self.buf)
        if n < 2:
            return np.inf
        return float(np.sqrt(max(self.s2 / n - (self.s / n) ** 2, 0.0)))


class StaticDetector:
    """Zero-velocity detector for a vehicle (engine idling, traffic lights).

    An idling engine still shakes the phone, so accelerometer variance alone is
    unreliable; the gyroscope is far less sensitive to engine vibration than to
    real vehicle motion, so both must be quiet for ``min_duration``. The AI
    speed network provides a second, learned opinion (p_stationary) which the
    engine combines with this rule-based one.
    """

    def __init__(self, fs: float, window_s: float = 1.0, acc_std_thr: float = 0.25,
                 gyro_std_thr: float = 0.02, gyro_mean_thr: float = 0.05, min_duration: float = 1.0,
                 horiz_thr: float = 0.25):
        n = max(3, int(round(window_s * fs)))
        self.acc = RunningStats(n)
        self.gyr = RunningStats(n)
        self.gyr_mean = deque(maxlen=n)
        self.horiz = deque(maxlen=n)
        self.horiz_thr = horiz_thr
        self.acc_std_thr = acc_std_thr
        self.gyro_std_thr = gyro_std_thr
        self.gyro_mean_thr = gyro_mean_thr
        self.min_samples = int(min_duration * fs)
        self.count = 0

    def update(self, accel, gyro, horiz_accel: float = 0.0) -> bool:
        """``horiz_accel``: magnitude of the levelled horizontal specific force.

        A smooth car accelerating on a straight road with a clean IMU has low
        variance too; only a stationary car has ~zero horizontal specific force.
        """
        self.acc.push(float(np.linalg.norm(accel)))
        g = float(np.linalg.norm(gyro))
        self.gyr.push(g)
        self.gyr_mean.append(g)
        self.horiz.append(horiz_accel)
        quiet = (self.acc.full and self.acc.std < self.acc_std_thr and self.gyr.std < self.gyro_std_thr
                 and np.mean(self.gyr_mean) < self.gyro_mean_thr and abs(np.mean(self.horiz)) < self.horiz_thr)
        self.count = self.count + 1 if quiet else 0
        return self.count >= self.min_samples


class BumpDetector:
    """Pothole / speed-breaker detector on the vertical specific force.

    A bump is a short (<0.5 s) vertical shock several m/s^2 above the gravity
    baseline. While it rings, horizontal accelerometer samples are corrupted by
    pitch/roll transients, so the engine inflates accelerometer noise and the
    AI model's windows are flagged.
    """

    def __init__(self, fs: float, thr: float = 2.5, hold_s: float = 0.6):
        self.thr = thr
        self.hold = int(hold_s * fs)
        self.remaining = 0
        self.events = 0

    def update(self, a_up_dev: float) -> bool:
        if abs(a_up_dev) > self.thr:
            if self.remaining == 0:
                self.events += 1
            self.remaining = self.hold
        elif self.remaining > 0:
            self.remaining -= 1
        return self.remaining > 0


def butter_lowpass_batch(x, fs, fc):
    """Causal biquad applied to a whole (n, c) array (training-time helper)."""
    x = np.asarray(x, dtype=float)
    two_d = x.ndim == 2
    x2 = x if two_d else x[:, None]
    f = Biquad(fs, fc, x2.shape[1])
    out = np.empty_like(x2)
    for i in range(len(x2)):
        out[i] = f(x2[i])
    return out if two_d else out[:, 0]
