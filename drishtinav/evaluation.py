"""Benchmark harness: simulated GNSS outages on real drives, scored against truth.

Protocol (follows the IO-VNBD literature, e.g. Onyekpe et al. WhONet 2021):
  * GNSS aiding comes from the reference receiver, down-sampled to 1 Hz and
    degraded to smartphone quality (coloured noise, sigma ~2.5 m). The phone's
    own GNSS in IO-VNBD updates only every ~9 s and lags by seconds, so it is
    unusable as either aiding or truth (see docs/DATASET.md).
  * Outages of L = 30 / 60 / 120 / 180 s are injected repeatedly, separated by
    ``gap_s`` of GNSS so the filter reconverges between them.
  * Per outage we report the distance driven (truth), horizontal error at the
    end of the outage and its maximum, and drift = end error / distance.
    The PS benchmark is drift < 10 % of distance travelled.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Optional

import numpy as np

from .config import make_config
from .engine import GnssFix, NavEngine
from .io.drive import Drive


def reference_gnss(drive: Drive, rate_hz=1.0, sigma=2.5, tau=30.0, seed=0):
    """1 Hz smartphone-grade GNSS synthesised from the drive's reference trajectory."""
    rng = np.random.default_rng(seed)
    frame = drive.frame()
    en = drive.truth_en(frame)
    step = max(1, int(round(drive.rate_hz / rate_hz)))
    idx = np.arange(0, len(drive), step)
    # first-order Gauss-Markov (multipath/atmosphere) + white noise
    a = np.exp(-step / drive.rate_hz / tau)
    gm = np.zeros((len(idx), 2))
    for k in range(1, len(idx)):
        gm[k] = a * gm[k - 1] + np.sqrt(1 - a * a) * rng.normal(0, sigma * 0.8, 2)
    noise = gm + rng.normal(0, sigma * 0.6, (len(idx), 2))
    p = en[idx] + noise
    lat, lon = frame.to_ll(p[:, 0], p[:, 1])
    spd = np.maximum(drive.truth_speed[idx] + rng.normal(0, 0.15, len(idx)), 0.0)
    crs = (drive.truth_course[idx] + rng.normal(0, 1.5, len(idx))) % 360 if drive.truth_course is not None else None
    fixes = {}
    for k, i in enumerate(idx):
        if np.isfinite(lat[k]):
            fixes[int(i)] = GnssFix(float(lat[k]), float(lon[k]), float(spd[k]),
                                    float(crs[k]) if crs is not None else None, sigma)
    return fixes


def device_gnss(drive: Drive):
    """GNSS exactly as the device logged it (fresh fixes only)."""
    fixes = {}
    for i in np.where(drive.gnss_new)[0]:
        acc = drive.gnss_accuracy[i] if drive.gnss_accuracy is not None else 5.0
        fixes[int(i)] = GnssFix(float(drive.gnss_lat[i]), float(drive.gnss_lon[i]),
                                float(drive.gnss_speed[i]) if drive.gnss_speed is not None else None,
                                float(drive.gnss_course[i]) if drive.gnss_course is not None else None,
                                float(acc) if np.isfinite(acc) else 5.0)
    return fixes


def outage_schedule(drive: Drive, length_s: float, warmup_s: float = 120.0, gap_s: float = 90.0,
                    min_dist: float = 0.0):
    """Half-open [i0, i1) index ranges of GNSS outages (deny with ``mask[i0:i1] = True``)."""
    t = drive.t
    out, t0 = [], t[0] + warmup_s
    dist = drive.distance_travelled()
    while t0 + length_s <= t[-1]:
        i0 = int(np.searchsorted(t, t0))
        i1 = int(np.searchsorted(t, t0 + length_s))
        if dist[min(i1, len(t) - 1)] - dist[i0] >= min_dist:
            out.append((i0, min(i1, len(t) - 1)))
        t0 += length_s + gap_s
    return out


def run(drive: Drive, cfg=None, speed_model=None, roads=None, fixes=None, denied=None,
        record_every: int = 1):
    """Run the engine over a drive. ``denied`` masks GNSS (outages / tunnels)."""
    cfg = cfg or make_config()
    frame = drive.frame()
    eng = NavEngine(cfg, speed_model=speed_model, roads=roads, frame=frame)
    fixes = fixes if fixes is not None else reference_gnss(drive)
    n = len(drive)
    keys = ("e", "n", "heading_deg", "speed", "pos_sigma", "ai_speed", "ai_sigma", "p_stationary",
            "matched_e", "matched_n", "match_conf", "latency_us", "yaw_rate", "gyro_bias")
    rec = {k: np.full(n, np.nan) for k in keys}
    mode = np.empty(n, dtype=object)
    flags = {k: np.zeros(n, bool) for k in ("stationary", "bump", "in_tunnel")}
    t_start = time.perf_counter()
    for i in range(n):
        fix = fixes.get(i)
        if fix is not None and denied is not None and denied[i]:
            fix = None
        st = eng.step(drive.t[i], drive.accel[i], drive.gyro[i], fix)
        for k in keys:
            rec[k][i] = getattr(st, k)
        mode[i] = st.mode
        for k in flags:
            flags[k][i] = getattr(st, k)
    wall = time.perf_counter() - t_start
    rec.update(flags)
    rec["mode"] = mode
    rec["frame"] = frame
    rec["wall_s"] = wall
    rec["diagnostics"] = eng.diagnostics
    rec["denied"] = denied if denied is not None else np.zeros(n, bool)
    return rec


def score(drive: Drive, rec, outages):
    """Per-outage metrics. Outages are half-open index ranges [i0, i1): GNSS is denied
    on i0 .. i1-1 and a fix may arrive at i1, so the dead-reckoning error is taken at
    the last denied sample i1-1 - before the first recovery fix can pull it in."""
    en = drive.truth_en(rec["frame"])
    est = np.column_stack([rec["e"], rec["n"]])
    mm = np.column_stack([rec["matched_e"], rec["matched_n"]])
    mm = np.where(np.isfinite(mm), mm, est)
    err = np.hypot(*(est - en).T)
    err_mm = np.hypot(*(mm - en).T)
    dist = drive.distance_travelled()
    rows = []
    for i0, i1 in outages:
        e = max(i0, i1 - 1)                     # last GNSS-denied sample
        d = float(dist[e] - dist[i0])
        rows.append({
            "start_s": float(drive.t[i0]), "length_s": float(drive.t[e] - drive.t[i0]),
            "distance_m": d, "mean_speed_kmh": d / max(drive.t[e] - drive.t[i0], 1e-6) * 3.6,
            "start_err_m": float(err[i0]), "end_err_m": float(err[e]), "max_err_m": float(np.nanmax(err[i0:e + 1])),
            "end_err_matched_m": float(err_mm[e]),
            "drift_pct": float(100 * err[e] / d) if d > 1 else float("nan"),
            "drift_pct_matched": float(100 * err_mm[e] / d) if d > 1 else float("nan"),
        })
    return rows, err


def summarise(rows, min_dist=100.0):
    """Aggregate per-outage rows. Drift % is only meaningful when the car moved."""
    moving = [r for r in rows if r["distance_m"] >= min_dist]
    if not rows:
        return {}
    arr = lambda k, rs: np.array([r[k] for r in rs], float)
    out = {
        "n_outages": len(rows), "n_scored": len(moving),
        "mean_distance_m": float(arr("distance_m", rows).mean()),
        "end_err_mean_m": float(arr("end_err_m", rows).mean()),
        "end_err_median_m": float(np.median(arr("end_err_m", rows))),
        "max_err_mean_m": float(arr("max_err_m", rows).mean()),
        "end_err_rms_m": float(np.sqrt(np.mean(arr("end_err_m", rows) ** 2))),
    }
    if moving:
        dp = arr("drift_pct", moving)
        dpm = arr("drift_pct_matched", moving)
        tot = float(arr("end_err_m", moving).sum() / arr("distance_m", moving).sum() * 100)
        out.update({
            "drift_pct_median": float(np.median(dp)), "drift_pct_mean": float(dp.mean()),
            "drift_pct_p90": float(np.percentile(dp, 90)),
            "drift_pct_aggregate": tot,
            "pass_rate_10pct": float(np.mean(dp < 10.0)),
            "drift_pct_matched_median": float(np.median(dpm)),
            "pass_rate_10pct_matched": float(np.mean(dpm < 10.0)),
        })
    return out
