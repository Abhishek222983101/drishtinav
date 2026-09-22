"""Generic CSV adapter so the engine runs on *any* IMU, not only phones.

Expected columns (header names, case-insensitive; extra columns ignored)::

    t                     seconds (or t_ms / timestamp_ms / time_ns)
    ax, ay, az            m/s^2  (specific force, gravity included)   [or accel_x...; g units with --accel-g]
    gx, gy, gz            rad/s                                       [or gyro_x...;  deg/s with --gyro-deg]
    mx, my, mz            optional magnetometer
    lat, lon              optional GNSS fix (blank when no fix)
    speed                 optional GNSS speed m/s
    course                optional GNSS course, deg clockwise from North
    hacc                  optional horizontal accuracy (m)
    truth_lat, truth_lon, truth_speed, truth_course   optional reference for scoring

The mobile app's recorder writes exactly this format, and IO-VNBD files can be
converted to it with ``python -m drishtinav convert``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .drive import Drive

ALIASES = {
    "t": ["t", "time", "time_s", "timestamp", "t_s"],
    "t_ms": ["t_ms", "timestamp_ms", "time_ms"],
    "t_ns": ["time_ns", "timestamp_ns", "t_ns"],
    "ax": ["ax", "accel_x", "acc_x", "a_x"], "ay": ["ay", "accel_y", "acc_y", "a_y"], "az": ["az", "accel_z", "acc_z", "a_z"],
    "gx": ["gx", "gyro_x", "gyr_x", "w_x"], "gy": ["gy", "gyro_y", "gyr_y", "w_y"], "gz": ["gz", "gyro_z", "gyr_z", "w_z"],
    "mx": ["mx", "mag_x"], "my": ["my", "mag_y"], "mz": ["mz", "mag_z"],
    "lat": ["lat", "latitude", "gnss_lat", "gps_lat"], "lon": ["lon", "lng", "longitude", "gnss_lon", "gps_lon"],
    "speed": ["speed", "gnss_speed", "gps_speed"], "course": ["course", "bearing", "gnss_course", "heading"],
    "hacc": ["hacc", "accuracy", "gnss_accuracy", "h_acc"],
    "truth_lat": ["truth_lat", "ref_lat"], "truth_lon": ["truth_lon", "ref_lon"],
    "truth_speed": ["truth_speed", "ref_speed"], "truth_course": ["truth_course", "ref_course"],
}


def _pick(df, key):
    cols = {c.lower().strip(): c for c in df.columns}
    for a in ALIASES[key]:
        if a in cols:
            return pd.to_numeric(df[cols[a]], errors="coerce").values.astype(float)
    return None


def load_generic_csv(path, accel_in_g=False, gyro_in_deg=False, name=None) -> Drive:
    df = pd.read_csv(path)
    t = _pick(df, "t")
    if t is None and _pick(df, "t_ms") is not None:
        t = _pick(df, "t_ms") / 1e3
    if t is None and _pick(df, "t_ns") is not None:
        t = _pick(df, "t_ns") / 1e9
    if t is None:
        raise ValueError("CSV needs a time column (t / t_ms / time_ns)")
    t = t - t[0]
    acc = np.column_stack([_pick(df, k) for k in ("ax", "ay", "az")])
    gyr = np.column_stack([_pick(df, k) for k in ("gx", "gy", "gz")])
    if accel_in_g:
        acc = acc * 9.80665
    if gyro_in_deg:
        gyr = np.radians(gyr)
    mag = None
    if _pick(df, "mx") is not None:
        mag = np.column_stack([_pick(df, k) for k in ("mx", "my", "mz")])
    lat, lon = _pick(df, "lat"), _pick(df, "lon")
    new = None
    if lat is not None:
        new = np.isfinite(lat) & np.isfinite(lon)
        # loggers that repeat the last fix: keep only changes
        rep = np.r_[False, (np.diff(np.nan_to_num(lat)) == 0) & (np.diff(np.nan_to_num(lon)) == 0)]
        new &= ~rep
    d = Drive(name=name or Path(path).stem, t=t, accel=acc, gyro=gyr, mag=mag,
              gnss_lat=lat, gnss_lon=lon, gnss_speed=_pick(df, "speed"), gnss_course=_pick(df, "course"),
              gnss_accuracy=_pick(df, "hacc"), gnss_new=new,
              truth_lat=_pick(df, "truth_lat"), truth_lon=_pick(df, "truth_lon"),
              truth_speed=_pick(df, "truth_speed"), truth_course=_pick(df, "truth_course"),
              meta={"source": "generic-csv", "path": str(path)})
    return d


def save_generic_csv(drive: Drive, path):
    cols = {"t": drive.t, "ax": drive.accel[:, 0], "ay": drive.accel[:, 1], "az": drive.accel[:, 2],
            "gx": drive.gyro[:, 0], "gy": drive.gyro[:, 1], "gz": drive.gyro[:, 2]}
    if drive.gnss_lat is not None:
        m = drive.gnss_new if drive.gnss_new is not None else np.isfinite(drive.gnss_lat)
        for k, v in (("lat", drive.gnss_lat), ("lon", drive.gnss_lon), ("speed", drive.gnss_speed),
                     ("course", drive.gnss_course), ("hacc", drive.gnss_accuracy)):
            if v is not None:
                cols[k] = np.where(m, v, np.nan)
    for k in ("truth_lat", "truth_lon", "truth_speed", "truth_course"):
        v = getattr(drive, k)
        if v is not None:
            cols[k] = v
    pd.DataFrame(cols).to_csv(path, index=False, float_format="%.10g")
