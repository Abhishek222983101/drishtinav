"""Sensor-agnostic container for one recorded (or simulated) drive.

Every data source (IO-VNBD smartphone logs, generic external IMU CSVs, the
synthetic simulator, the mobile app recorder) is converted into a ``Drive`` so the
navigation engine never has to know where the data came from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..geo import LocalFrame


@dataclass
class Drive:
    name: str
    t: np.ndarray                      # (n,) seconds, monotonic
    accel: np.ndarray                  # (n,3) specific force in sensor frame, m/s^2 (includes gravity)
    gyro: np.ndarray                   # (n,3) angular rate in sensor frame, rad/s
    mag: Optional[np.ndarray] = None   # (n,3) magnetic field, uT (optional)

    # GNSS as the receiver reports it. gnss_new[i] is True only on epochs where a
    # fresh fix arrived (phones deliver 1 Hz fixes that loggers repeat at 10 Hz).
    gnss_lat: Optional[np.ndarray] = None
    gnss_lon: Optional[np.ndarray] = None
    gnss_speed: Optional[np.ndarray] = None      # m/s
    gnss_course: Optional[np.ndarray] = None     # deg, clockwise from North
    gnss_accuracy: Optional[np.ndarray] = None   # m (1-sigma-ish, receiver reported)
    gnss_new: Optional[np.ndarray] = None        # bool

    # Reference trajectory used only for evaluation / training labels.
    truth_lat: Optional[np.ndarray] = None
    truth_lon: Optional[np.ndarray] = None
    truth_speed: Optional[np.ndarray] = None     # m/s
    truth_course: Optional[np.ndarray] = None    # deg, clockwise from North

    # Samples where the scenario itself has no GNSS (e.g. a simulated tunnel).
    gnss_denied: Optional[np.ndarray] = None     # bool
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.t)

    @property
    def rate_hz(self) -> float:
        dt = np.diff(self.t)
        dt = dt[dt > 0]
        return float(1.0 / np.median(dt)) if len(dt) else 0.0

    def frame(self) -> LocalFrame:
        if self.truth_lat is not None:
            ok = np.isfinite(self.truth_lat) & np.isfinite(self.truth_lon)
            if ok.any():
                i = int(np.argmax(ok))
                return LocalFrame(self.truth_lat[i], self.truth_lon[i])
        ok = np.isfinite(self.gnss_lat) & np.isfinite(self.gnss_lon)
        i = int(np.argmax(ok))
        return LocalFrame(self.gnss_lat[i], self.gnss_lon[i])

    def slice(self, i0: int, i1: int, name: Optional[str] = None) -> "Drive":
        kw = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray) and len(v) == len(self.t):
                kw[k] = v[i0:i1]
            else:
                kw[k] = v
        kw["name"] = name or f"{self.name}[{i0}:{i1}]"
        kw["meta"] = dict(self.meta)
        return Drive(**kw)

    def truth_en(self, frame: Optional[LocalFrame] = None):
        frame = frame or self.frame()
        return np.column_stack(frame.to_en(self.truth_lat, self.truth_lon))

    def distance_travelled(self) -> np.ndarray:
        """Cumulative along-track distance of the reference trajectory (m)."""
        en = self.truth_en()
        d = np.r_[0.0, np.hypot(np.diff(en[:, 0]), np.diff(en[:, 1]))]
        d[~np.isfinite(d)] = 0.0
        return np.cumsum(d)
