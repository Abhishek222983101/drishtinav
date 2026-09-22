"""Input channels of the AI speed / motion network (shared by training and runtime).

Per IMU sample the network sees vehicle-frame, low-passed kinematics plus
rotation-invariant vibration energy:

    0 a_fwd      longitudinal specific force (m/s^2), 2 Hz low-pass
    1 a_lat      lateral specific force (m/s^2), 2 Hz low-pass
    2 a_up       vertical specific force minus g (m/s^2), 2 Hz low-pass
    3 yaw_rate   (rad/s), 2 Hz low-pass
    4 vib_acc    |a - lowpass(a)|  (m/s^2): road/engine vibration energy
    5 vib_gyr    |w - lowpass(w)|  (rad/s)
    6 a_lat/w    centripetal speed cue: in a turn v = a_lat / yaw_rate, clipped

The vibration channels are what let a model infer speed without an odometer:
tyre/road excitation grows with speed, and an idling engine has a distinct,
speed-independent signature (so the model also learns stationarity).
"""
from __future__ import annotations

import numpy as np

from .calibration import Biquad

CHANNELS = ["a_fwd", "a_lat", "a_up", "yaw_rate", "vib_acc", "vib_gyr", "turn_speed"]
N_CH = len(CHANNELS)
LP_HZ = 2.0


class FeatureStream:
    """Per-sample causal feature extractor (runtime twin of ``vehicle_channels``)."""

    def __init__(self, fs: float):
        self.lp_kin = Biquad(fs, LP_HZ, 4)
        self.lp_acc = Biquad(fs, LP_HZ, 3)
        self.lp_gyr = Biquad(fs, LP_HZ, 3)
        self.lp_vib = Biquad(fs, 0.5, 2)

    def __call__(self, a_fwd, a_lat, a_up, yaw, accel, gyro):
        k = self.lp_kin(np.array([a_fwd, a_lat, a_up, yaw]))
        va = np.linalg.norm(np.asarray(accel) - self.lp_acc(accel))
        vg = np.linalg.norm(np.asarray(gyro) - self.lp_gyr(gyro))
        vib = self.lp_vib(np.array([va, vg]))
        return np.array([k[0], k[1], k[2], k[3], vib[0], vib[1], turn_speed(k[1], k[3])], dtype=np.float32)


def turn_speed(a_lat, yaw):
    """v = a_lat / yaw_rate when turning (|yaw| > 0.05 rad/s), else 0."""
    y = np.asarray(yaw)
    ok = np.abs(y) > 0.05
    out = np.where(ok, np.asarray(a_lat) / np.where(ok, y, 1.0), 0.0)
    return np.clip(out, 0.0, 40.0)


def vehicle_channels(a_fwd, a_lat, a_up, yaw, accel, gyro, fs):
    """Vectorised equivalent of ``FeatureStream`` over a whole drive."""
    fsx = FeatureStream(fs)
    out = np.empty((len(a_fwd), N_CH), dtype=np.float32)
    for i in range(len(a_fwd)):
        out[i] = fsx(a_fwd[i], a_lat[i], a_up[i], yaw[i], accel[i], gyro[i])
    return out
