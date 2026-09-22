"""Loader for the IO-VNBD benchmark (Onyekpe et al., Data in Brief 2021).

Dataset layout (https://github.com/onyekpeu/IO-VNBD):
  * ``S-<name>.csv`` - Android smartphone in a dashboard holder, AndroSensor app,
    10 Hz: GPS (1 Hz fixes repeated), accelerometer, gravity, gyroscope,
    magnetometer, orientation.  24 columns.
  * ``V-<name>.csv`` - Ford Fiesta CAN bus + Racelogic VBOX 10 Hz GPS: wheel
    speeds, indicated speed, yaw rate, steering... 29 columns.
  * The "Synchronised V and S datasets" folder row-aligns the two files.

We use the smartphone file as the *input* (that is what the PS says the engine
must run on) and the VBOX/CAN file only as *ground truth* for training labels
and for scoring.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .drive import Drive

PHONE_COLS = [
    "gps_lat", "gps_lon", "gps_alt", "gps_speed_kmh", "gps_accuracy", "gps_course",
    "gps_sats", "time_ms", "date",
    "acc_x", "acc_y", "acc_z", "grav_x", "grav_y", "grav_z",
    "gyr_x", "gyr_y", "gyr_z", "mag_x", "mag_y", "mag_z",
    "ori_yaw", "ori_pitch", "ori_roll",
]

VEHICLE_COLS = [
    "sats", "tod_s", "lat", "lon", "speed_kmh", "heading_deg", "height_km", "vz_kmh",
    "sample_period", "steer_deg", "ws_fl", "ws_fr", "ws_rl", "ws_rr", "yaw_rate_dps",
    "ind_speed_kmh", "ind_long_g", "ind_lat_g", "handbrake", "gear_req", "gear",
    "rpm", "coolant", "clutch", "brake_psi", "brake", "battery_v", "air_temp", "accel_pedal",
]


def read_phone_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1", header=0, on_bad_lines="skip", low_memory=False)
    df = df.iloc[:, : len(PHONE_COLS)]
    df.columns = PHONE_COLS[: df.shape[1]]
    for c in df.columns:
        if c not in ("date", "gps_sats"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def read_vehicle_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1", header=0, on_bad_lines="skip", low_memory=False)
    df = df.iloc[:, : len(VEHICLE_COLS)]
    df.columns = VEHICLE_COLS[: df.shape[1]]
    return df.apply(pd.to_numeric, errors="coerce")


def _fresh_fix_mask(lat, lon, spd):
    """AndroSensor repeats the last 1 Hz fix on every 10 Hz row; flag real updates."""
    changed = np.r_[True, (np.diff(lat) != 0) | (np.diff(lon) != 0) | (np.diff(spd) != 0)]
    return changed & np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) > 1e-6)


def load_drive(phone_csv, vehicle_csv: Optional[str] = None, name: Optional[str] = None) -> Drive:
    """Load one IO-VNBD drive (smartphone + optional synchronised vehicle reference)."""
    s = read_phone_csv(phone_csv)
    v = read_vehicle_csv(vehicle_csv) if vehicle_csv else None
    if v is not None:
        n = min(len(s), len(v))
        s, v = s.iloc[:n].reset_index(drop=True), v.iloc[:n].reset_index(drop=True)

    ok = s[["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z", "time_ms"]].notna().all(axis=1).to_numpy().copy()
    if v is not None:
        ok &= v[["lat", "lon", "ind_speed_kmh"]].notna().all(axis=1).values
    s = s[ok].reset_index(drop=True)
    if v is not None:
        v = v[ok].reset_index(drop=True)

    # AndroSensor's "time since start" resets when a recording session restarts
    # inside one file; rebuild a continuous clock from the per-row increments and
    # treat resets / gaps as one nominal 100 ms step.
    dt = np.diff(s["time_ms"].values.astype(float)) / 1000.0
    dt[(dt <= 0.0) | (dt > 1.0)] = 0.1
    t = np.r_[0.0, np.cumsum(dt)]

    lat, lon = s["gps_lat"].values, s["gps_lon"].values
    spd = s["gps_speed_kmh"].values / 3.6
    drive = Drive(
        name=name or Path(phone_csv).stem,
        t=t,
        accel=s[["acc_x", "acc_y", "acc_z"]].values.astype(float),
        gyro=s[["gyr_x", "gyr_y", "gyr_z"]].values.astype(float),
        mag=s[["mag_x", "mag_y", "mag_z"]].values.astype(float),
        gnss_lat=lat, gnss_lon=lon, gnss_speed=spd,
        gnss_course=s["gps_course"].values,
        gnss_accuracy=s["gps_accuracy"].values,
        gnss_new=_fresh_fix_mask(lat, lon, spd),
        meta={"source": "IO-VNBD", "phone_csv": str(phone_csv), "vehicle_csv": str(vehicle_csv or "")},
    )
    if v is not None:
        drive.truth_lat = v["lat"].values
        # VBOX logs West longitudes as positive minutes in some files; the phone
        # fix tells us the hemisphere.
        vlon = v["lon"].values
        plon = np.nanmedian(lon[np.abs(lon) > 1e-6]) if np.any(np.abs(lon) > 1e-6) else np.nan
        if np.isfinite(plon) and np.sign(np.nanmedian(vlon)) != np.sign(plon):
            vlon = -vlon
        drive.truth_lon = vlon
        drive.truth_course = v["heading_deg"].values
        can_speed = v["ind_speed_kmh"].values / 3.6
        gps_speed = v["speed_kmh"].values / 3.6
        can_ok = np.nanmax(np.abs(can_speed)) > 1.0
        drive.truth_speed = can_speed if can_ok else gps_speed
        drive.meta["truth_speed_source"] = "CAN" if can_ok else "VBOX-GPS"
        drive.meta["vehicle_yaw_rate"] = np.radians(v["yaw_rate_dps"].values) if can_ok else None
        # VBOX course-rate yaw reference (used when CAN yaw is missing/unreliable).
        crs = np.unwrap(np.radians(v["heading_deg"].values))
        yr_gps = -np.gradient(crs) / 0.1       # VBOX logs at 10 Hz; course is clockwise, yaw CCW
        yr_gps[gps_speed < 3.0] = 0.0
        drive.meta["vbox_yaw_rate"] = np.clip(np.nan_to_num(yr_gps), -1.5, 1.5)
    return drive


def _best_lag(a, b, max_lag):
    """Lag L maximising corr(a[i], b[i+L]) via FFT cross-correlation."""
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    n = len(a)
    size = 1 << int(np.ceil(np.log2(2 * n)))
    xc = np.fft.irfft(np.conj(np.fft.rfft(a, size)) * np.fft.rfft(b, size), size)
    lags = np.r_[np.arange(0, max_lag + 1), np.arange(-max_lag, 0)]
    vals = np.r_[xc[: max_lag + 1], xc[-max_lag:]] / n
    i = int(np.argmax(vals))
    return int(lags[i]), float(vals[i])


def _smooth(x, k=5):
    return np.convolve(x, np.ones(k) / k, "same")


def resync_segments(drive: Drive, chunk: int = 1500, max_lag: int = 400,
                    min_corr: float = 0.8, min_len: int = 3000, relaxed_corr: float = 0.35) -> list[Drive]:
    """Repair the phone<->vehicle alignment of an IO-VNBD "synchronised" drive.

    The published S-/V- pairs are only coarsely aligned: the offset is a few
    samples to several seconds and jumps where either logger dropped rows. We
    cross-correlate the phone gyro's yaw axis with the CAN yaw-rate in chunks,
    shift the reference by the local lag and keep only chunks whose correlation
    proves they are truly aligned. Contiguous accepted chunks with the same lag
    become clean segments usable for training and scoring.
    """
    n = len(drive)
    # Which yaw reference (CAN yaw-rate or VBOX course-rate) and which phone gyro
    # axis/sign carry vehicle yaw? Decide on the whole drive.
    refs = [(label, r) for label, r in (("CAN", drive.meta.get("vehicle_yaw_rate")),
                                        ("VBOX-course", drive.meta.get("vbox_yaw_rate"))) if r is not None]
    if not refs:
        return [drive]
    axis_scores = []
    for ri, (_, ref) in enumerate(refs):
        for k in range(3):
            for sgn in (1, -1):
                axis_scores.append((_best_lag(ref, sgn * drive.gyro[:, k], max_lag)[1], ri, k, sgn))
    _, ri, k, sgn = max(axis_scores)
    ref_label, yr = refs[ri]
    g = sgn * drive.gyro[:, k]

    thr = min_corr if ref_label == "CAN" else min_corr - 0.2   # course-rate reference is noisier
    raw = []
    for c0 in range(0, n - chunk + 1, chunk):
        a, b = yr[c0:c0 + chunk], g[c0:c0 + chunk]
        if np.std(a) < 0.01:          # no turning -> alignment unobservable
            raw.append((c0, None, 0.0))
            continue
        raw.append((c0, *_best_lag(_smooth(a), _smooth(b), max_lag)))
    # A lag that recurs across many chunks is real even when a noisy phone gyro
    # keeps each chunk's correlation low (relaxed QC, still >= relaxed_corr).
    lags_seen = [l for _, l, c in raw if l is not None and c >= relaxed_corr]
    modal = max(set(lags_seen), key=lags_seen.count) if lags_seen else None
    modal_ok = modal is not None and sum(abs(l - modal) <= 2 for l in lags_seen) >= max(3, 0.5 * len(lags_seen))
    accepted = []
    for c0, lag, corr in raw:
        if lag is None:
            accepted.append((c0, None))
        elif corr >= thr:
            accepted.append((c0, lag))
        elif modal_ok and corr >= relaxed_corr and abs(lag - modal) <= 2:
            accepted.append((c0, modal))
        else:
            accepted.append((c0, "bad"))

    # Fill quiet chunks with neighbouring lags, then group runs of equal lag.
    lags = [l for _, l in accepted]
    for i, l in enumerate(lags):
        if l is None:
            prev = next((lags[j] for j in range(i - 1, -1, -1) if isinstance(lags[j], int)), None)
            nxt = next((lags[j] for j in range(i + 1, len(lags)) if isinstance(lags[j], int)), None)
            lags[i] = prev if prev is not None and prev == nxt else (prev if nxt is None else nxt if prev is None else "bad")
    segments, start, cur = [], None, None
    for i, l in enumerate(lags + ["end"]):
        if isinstance(l, int) and l == cur:
            continue
        if cur is not None and isinstance(cur, int):
            segments.append((start * chunk, i * chunk, cur))
        start, cur = i, l if isinstance(l, int) else None

    truth_keys = ("truth_lat", "truth_lon", "truth_speed", "truth_course")
    out = []
    for s0, s1, lag in segments:
        # phone sample j <-> vehicle sample j - lag
        lo, hi = max(s0, lag), min(s1, n + lag)
        # drop long parked stretches at either end (nothing to navigate)
        spd = drive.truth_speed[lo - lag:hi - lag]
        moving = np.where(spd > 0.5)[0]
        if len(moving) == 0:
            continue
        lo, hi = lo + max(0, moving[0] - 300), lo + min(len(spd), moving[-1] + 300)
        if hi - lo < min_len:
            continue
        seg = drive.slice(lo, hi, name=f"{drive.name}#{len(out)}")
        for key in truth_keys:
            setattr(seg, key, getattr(drive, key)[lo - lag:hi - lag].copy())
        for mk in ("vehicle_yaw_rate", "vbox_yaw_rate"):
            if drive.meta.get(mk) is not None:
                seg.meta[mk] = drive.meta[mk][lo - lag:hi - lag].copy()
        seg.meta.update(resync_lag=lag, resync_range=(int(lo), int(hi)), yaw_gyro_axis=int(k),
                        yaw_gyro_sign=int(sgn), resync_reference=ref_label)
        out.append(seg)
    return out


def find_drives(root) -> list[tuple[str, str, Optional[str]]]:
    """Discover (name, phone_csv, vehicle_csv) triples under a folder."""
    out = []
    for dirpath, _, files in os.walk(root):
        s_files = [f for f in files if f.lower().startswith("s-") and f.lower().endswith(".csv")]
        v_files = [f for f in files if f.lower().startswith("v-") and f.lower().endswith(".csv")]
        for sf in s_files:
            key = sf[2:-4].lower()
            vf = next((f for f in v_files if f[2:-4].lower() == key), v_files[0] if len(v_files) == 1 else None)
            out.append((sf[2:-4], os.path.join(dirpath, sf), os.path.join(dirpath, vf) if vf else None))
    return sorted(out)
