"""Physics-based scenario simulator on real OpenStreetMap roads.

Builds a drive along a routed OSM path (e.g. through Delhi's 1.3 km Pragati
Maidan tunnel), with:

* a realistic speed profile - launch from rest, curvature-limited cornering
  (|a_lat| <= 2.5 m/s^2), traffic-light stops, constant-ish cruise in tunnels;
* an IMU synthesised from the true kinematics and rotated into an arbitrary
  phone mount, with the error sources the PS lists: turn-on biases, white noise,
  engine harmonics (aliased at low sample rates), speed-dependent road
  vibration, phone-holder wobble, and pothole / speed-breaker shocks;
* 1 Hz GNSS with coloured noise, denied inside OSM ``tunnel=yes`` ways and
  degraded (multipath) near the portals.

Two sensor profiles: ``smartphone`` (consumer MEMS, 10 Hz) and ``fog``
(navigation-grade fibre-optic gyro IMU, 200 Hz, rigid mount).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..geo import LocalFrame, math_to_compass
from ..roadnet import RoadNetwork
from .drive import Drive

G = 9.80665


@dataclass
class SensorModel:
    rate_hz: float
    gyro_bias: float          # rad/s, turn-on bias (1-sigma per axis)
    gyro_noise: float         # rad/s per sample (white)
    accel_bias: float         # m/s^2 (1-sigma per axis)
    accel_noise: float        # m/s^2 per sample
    vib_road: float           # m/s^2 per (m/s) of speed
    vib_engine: float         # m/s^2 amplitude of engine harmonic
    wobble_deg: float         # phone-holder pitch/roll wobble amplitude (deg)
    mount_rpy_deg: tuple      # phone mount roll, pitch, yaw relative to the car


PROFILES = {
    "smartphone": SensorModel(10.0, 0.006, 0.01, 0.08, 0.15, 0.035, 0.25, 1.5, (4.0, -62.0, 35.0)),
    "fog": SensorModel(200.0, 5e-7, 2e-4, 0.01, 0.02, 0.004, 0.02, 0.0, (0.3, -0.5, 1.0)),
}


def _rot(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def route_polyline(net: RoadNetwork, start_ll, end_ll):
    fr = net.frame
    s = net.nearest_node(fr.to_en(*start_ll))
    t = net.nearest_node(fr.to_en(*end_ll))
    nodes, segs, L = net.shortest_path(s, t)
    if nodes is None:
        raise ValueError("no route between the given points")
    return net.node_xy[np.asarray(nodes)], np.asarray(segs)


def _resample(poly, step):
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(poly, axis=0).T))]
    s = np.arange(0, d[-1], step)
    return np.column_stack([np.interp(s, d, poly[:, 0]), np.interp(s, d, poly[:, 1])]), s


def simulate_route(net: RoadNetwork, start_ll, end_ll, profile="smartphone", seed=7, name=None,
                   cruise_kmh=55.0, stops=((0.22, 25.0),), portal_margin_m=40.0,
                   potholes_per_km=3.0) -> Drive:
    rng = np.random.default_rng(seed)
    sm = PROFILES[profile]
    poly, segs = route_polyline(net, start_ll, end_ll)

    # --- geometry: smooth the polyline, arc-length parameterisation ---------
    pts, s_grid = _resample(poly, 1.0)
    k = 9
    ker = np.ones(k) / k
    sm_pts = np.column_stack([np.convolve(np.pad(pts[:, i], k // 2, mode="edge"), ker, "valid") for i in (0, 1)])
    d = np.gradient(sm_pts, axis=0)
    heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    curv = np.gradient(heading) / 1.0                       # rad per metre
    curv = np.convolve(np.pad(curv, 7, mode="edge"), np.ones(15) / 15, "valid")
    total = s_grid[-1]

    # tunnel mask along the route (from OSM tags of the routed segments)
    seg_start = np.r_[0, np.cumsum(net.seg_len[segs])]
    tun = np.zeros(len(s_grid), bool)
    for j, sg in enumerate(segs):
        if net.seg_tunnel[sg]:
            tun[(s_grid >= seg_start[j]) & (s_grid < seg_start[j + 1])] = True

    # --- speed profile (forward/backward pass with accel limits) ------------
    v_cruise = cruise_kmh / 3.6
    v_max = np.minimum(v_cruise, np.sqrt(2.5 / np.maximum(np.abs(curv), 1e-4)))
    v_max[tun] = np.minimum(v_max[tun], 60 / 3.6)
    stop_idx = [int(f * len(s_grid)) for f, _ in stops]
    for i in stop_idx:
        v_max[i] = 0.0
    v_max[0] = 0.0
    v_max[-1] = 0.0
    v = v_max.copy()
    for i in range(1, len(v)):                        # accel limit 1.6 m/s^2
        v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * 1.6 * 1.0))
    for i in range(len(v) - 2, -1, -1):               # braking limit 2.2 m/s^2
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * 2.2 * 1.0))

    # --- time parameterisation with dwell at stops --------------------------
    fs = sm.rate_hz
    dt = 1.0 / fs
    t_list, s_list = [0.0], [0.0]
    s_pos, t_now = 0.0, 0.0
    dwell = {i: dur for i, (_, dur) in zip(stop_idx, stops)}
    dwell[0] = 8.0
    done_dwell = set()
    while s_pos < total - 0.5:
        i = int(s_pos)
        if i in dwell and i not in done_dwell and np.interp(s_pos, s_grid, v) < 0.3:
            for _ in range(int(dwell[i] * fs)):
                t_now += dt
                t_list.append(t_now); s_list.append(s_pos)
            done_dwell.add(i)
        vi = max(np.interp(s_pos, s_grid, v), 0.4)
        s_pos = min(total, s_pos + vi * dt)
        t_now += dt
        t_list.append(t_now); s_list.append(s_pos)
    t = np.array(t_list)
    s = np.array(s_list)
    n = len(t)
    x = np.interp(s, s_grid, sm_pts[:, 0])
    y = np.interp(s, s_grid, sm_pts[:, 1])
    psi = np.interp(s, s_grid, heading)
    speed = np.r_[0, np.diff(s) / dt]
    speed = np.convolve(np.pad(speed, 2, mode="edge"), np.ones(5) / 5, "valid")
    yaw_rate = np.gradient(psi, t)
    a_long = np.gradient(speed, t)
    a_lat = speed * yaw_rate
    in_tunnel = np.interp(s, s_grid, tun.astype(float)) > 0.5

    # --- IMU synthesis -------------------------------------------------------
    R_mount = _rot(*np.radians(sm.mount_rpy_deg))     # phone axes expressed in vehicle frame
    f_veh = np.column_stack([a_long, a_lat, np.full(n, G)])
    w_veh = np.column_stack([np.zeros(n), np.zeros(n), yaw_rate])
    # holder wobble: small pitch/roll oscillation excited by acceleration + road
    wob = np.zeros((n, 2))
    if sm.wobble_deg > 0:
        for ax in range(2):
            ph = rng.uniform(0, 2 * np.pi)
            wob[:, ax] = np.radians(sm.wobble_deg) * (0.4 + 0.6 * np.clip(speed / 15, 0, 1)) * \
                np.sin(2 * np.pi * rng.uniform(0.2, 0.6) * t + ph) + np.radians(0.8) * np.convolve(
                    rng.normal(0, 1, n), np.ones(5) / 5, "same")
        wob[:, 0] += 0.02 * a_long                    # holder pitches under braking
    accel = np.empty((n, 3))
    gyro = np.empty((n, 3))
    wob_rate = np.gradient(wob, t, axis=0)
    for i in range(n):
        Rw = _rot(wob[i, 1], wob[i, 0], 0.0)
        R = (R_mount @ Rw).T                           # vehicle -> phone
        accel[i] = R @ f_veh[i]
        gyro[i] = R @ (w_veh[i] + np.array([wob_rate[i, 1], wob_rate[i, 0], 0.0]))
    accel += rng.normal(0, sm.accel_bias, 3)
    gyro += rng.normal(0, sm.gyro_bias, 3)
    accel += rng.normal(0, sm.accel_noise, (n, 3))
    gyro += rng.normal(0, sm.gyro_noise, (n, 3))
    # engine (2nd order, ~20-60 Hz: aliases broadband at 10 Hz) + road texture ~ speed
    rpm_hz = 13 + 2.2 * speed
    eng = sm.vib_engine * np.sin(2 * np.pi * np.cumsum(2 * rpm_hz * dt))
    accel += eng[:, None] * np.array([0.3, 0.2, 1.0]) + \
        (sm.vib_road * speed)[:, None] * rng.normal(0, 1, (n, 3)) * np.array([0.6, 0.6, 1.0])
    gyro += (0.002 * speed)[:, None] * rng.normal(0, 1, (n, 3)) * (1 if profile == "smartphone" else 0.05)
    # potholes / speed breakers: vertical shock + pitch kick
    n_holes = rng.poisson(potholes_per_km * s[-1] / 1000)
    hole_s = rng.uniform(0.05, 0.95, n_holes) * s[-1]
    for hs in hole_s:
        i0 = int(np.searchsorted(s, hs))
        if i0 >= n - 5 or speed[i0] < 2:
            continue
        L = max(3, int(0.4 * fs))
        shock = 6.0 * np.exp(-np.arange(L) / (0.08 * fs)) * np.cos(2 * np.pi * 8 * np.arange(L) / fs)
        R = R_mount.T
        accel[i0:i0 + L] += np.outer(shock, R @ np.array([0.2, 0.0, 1.0]))[: n - i0]
        gyro[i0:i0 + L] += np.outer(0.05 * shock, R @ np.array([0.0, 1.0, 0.0]))[: n - i0]

    # --- geodetic truth + GNSS ----------------------------------------------
    lat, lon = net.frame.to_ll(x, y)
    course = math_to_compass(psi)
    denied = np.zeros(n, bool)
    s_t = np.interp(s, s_grid, s_grid)
    tun_s = s_grid[tun]
    if len(tun_s):
        # a tunnel may be several OSM ways; deny from first portal to last portal
        denied = (s_t >= tun_s.min() - portal_margin_m * 0.25) & (s_t <= tun_s.max() + portal_margin_m * 0.25)
    near_portal = np.zeros(n, bool)
    if len(tun_s):
        near_portal = ((np.abs(s_t - tun_s.min()) < portal_margin_m) | (np.abs(s_t - tun_s.max()) < portal_margin_m)) & ~denied
    step = int(round(fs))
    gm = np.zeros(2)
    glat = np.full(n, np.nan); glon = np.full(n, np.nan)
    gspd = np.full(n, np.nan); gcrs = np.full(n, np.nan); gacc = np.full(n, np.nan)
    gnew = np.zeros(n, bool)
    for i in range(0, n, step):
        gm = 0.967 * gm + 0.255 * rng.normal(0, 2.0, 2)
        if denied[i]:
            continue
        sig = 2.5
        noise = gm + rng.normal(0, 1.5, 2)
        if near_portal[i]:
            sig = 12.0
            noise = noise * 4 + rng.normal(0, 8, 2)       # multipath at portals
        la, lo = net.frame.to_ll(x[i] + noise[0], y[i] + noise[1])
        glat[i], glon[i], gacc[i] = la, lo, sig
        gspd[i] = max(0.0, speed[i] + rng.normal(0, 0.15))
        gcrs[i] = (course[i] + rng.normal(0, 1.5)) % 360 if speed[i] > 1 else np.nan
        gnew[i] = True

    drive = Drive(
        name=name or f"sim-{profile}", t=t, accel=accel, gyro=gyro, mag=None,
        gnss_lat=glat, gnss_lon=glon, gnss_speed=gspd, gnss_course=gcrs, gnss_accuracy=gacc, gnss_new=gnew,
        truth_lat=lat, truth_lon=lon, truth_speed=speed, truth_course=course, gnss_denied=denied,
        meta={"source": "synthetic", "profile": profile, "route_m": float(s[-1]),
              "tunnel_m": float(tun_s.max() - tun_s.min()) if len(tun_s) else 0.0,
              "mount_rpy_deg": sm.mount_rpy_deg, "potholes": int(n_holes), "seed": seed,
              "vehicle_yaw_rate": yaw_rate, "in_tunnel": in_tunnel},
    )
    return drive


SCENARIOS = {
    "delhi_pragati_tunnel": {
        "title": "Delhi - Pragati Maidan tunnel (1.3 km, GNSS-denied)",
        "roads": "delhi_central.json",
        "start": (28.628, 77.249), "end": (28.6129, 77.2297),
        "stops": ((0.12, 20.0), (0.83, 15.0)),
    },
}


def build_scenario(key: str, roads_dir, profile="smartphone", seed=7) -> tuple[Drive, RoadNetwork]:
    sc = SCENARIOS[key]
    net = RoadNetwork.load(f"{roads_dir}/{sc['roads']}")
    drv = simulate_route(net, sc["start"], sc["end"], profile=profile, seed=seed, name=f"{key}-{profile}",
                         stops=sc.get("stops", ()))
    drv.meta["title"] = sc["title"]
    return drv, net
