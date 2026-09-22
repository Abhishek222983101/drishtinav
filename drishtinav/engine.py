"""NavEngine: the edge-deployable Intelligent Dead Reckoning engine.

Streaming API (one call per IMU sample, any rate 10-400 Hz)::

    eng = NavEngine(make_config("smartphone"), speed_model=SpeedNetRuntime(), roads=RoadNetwork.load(...))
    for sample in stream:
        st = eng.step(t, accel_xyz, gyro_xyz, gnss=GnssFix(...) or None)
        st.lat, st.lon, st.heading_deg, st.speed, st.mode ...

Per-sample pipeline
    1. Static detector (rule)       -> ZUPT / ZARU candidates
    2. Mount alignment              -> vehicle-frame a_fwd, a_lat, a_up, yaw rate
    3. Vibration low-pass + bump    -> clean kinematics, pothole flags
    4. SpeedNet @10 Hz              -> AI speed, its sigma, p(stationary)
    5. EKF predict                  -> continuous INS solution (never pauses)
    6. GNSS deficit handler         -> which aiding sources are trusted now
    7. EKF updates                  -> GNSS | ZUPT | AI speed | map constraints
    8. HMM map matching (1 Hz)      -> matched road, lane-level snapped output
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .alignment import MountAlignment
from .calibration import Biquad, BumpDetector, StaticDetector
from .config import make_config
from .features import FeatureStream
from .fusion import VehicleEKF
from .geo import LocalFrame, compass_to_math, math_to_compass, wrap_angle
from .gnss_handler import DR, GnssDeficitHandler, RECOVERY
from .mapmatch import HMMMapMatcher
from .models.speednet import SpeedNetStream


@dataclass
class GnssFix:
    lat: float
    lon: float
    speed: Optional[float] = None       # m/s
    course: Optional[float] = None      # deg clockwise from North
    accuracy: float = 5.0               # m


@dataclass
class NavState:
    t: float
    e: float
    n: float
    lat: float
    lon: float
    heading_deg: float                  # compass (clockwise from North)
    speed: float
    mode: str
    pos_sigma: float
    ai_speed: float = float("nan")
    ai_sigma: float = float("nan")
    p_stationary: float = float("nan")
    stationary: bool = False
    bump: bool = False
    matched_e: float = float("nan")
    matched_n: float = float("nan")
    match_conf: float = 0.0
    in_tunnel: bool = False
    yaw_rate: float = 0.0               # vehicle-frame yaw rate used by the INS (rad/s)
    gyro_bias: float = 0.0
    latency_us: float = 0.0
    extras: dict = field(default_factory=dict)


class NavEngine:
    def __init__(self, cfg: Optional[dict] = None, speed_model=None, roads=None,
                 frame: Optional[LocalFrame] = None):
        self.cfg = cfg or make_config()
        c = self.cfg
        fs = c["imu_rate_hz"]
        self.fs = fs
        self.frame = frame
        self.roads = roads
        if roads is not None and frame is not None:
            self.roads = roads.reframe(frame)
        self.ekf = VehicleEKF(c)
        self.align = MountAlignment(fs, c["gyro_frame"])
        self.static_det = StaticDetector(fs, acc_std_thr=c["static_acc_std"], gyro_std_thr=c["static_gyro_std"],
                                         gyro_mean_thr=c["static_gyro_mean"], horiz_thr=c["static_horiz_accel"])
        self.bump_det = BumpDetector(fs, thr=c["bump_thr"])
        self.lp = Biquad(fs, 2.0 if fs <= 20 else 5.0, 4)
        self.handler = GnssDeficitHandler(c)
        self.decim = max(1, int(round(fs / c["model_rate_hz"])))
        self._acc_sum = np.zeros(3)
        self._gyr_sum = np.zeros(3)
        self._veh_sum = np.zeros(4)
        self._n_sum = 0
        self.features = FeatureStream(c["model_rate_hz"])
        self.nn = SpeedNetStream(speed_model if c["use_ai_speed"] else None)
        self.matcher = HMMMapMatcher(self.roads) if (self.roads is not None and c["use_map"]) else None
        self._last_t = None
        self._last_map_t = -1e9
        self._travel = 0.0
        self._last_fix = None
        self._ai = (float("nan"), float("nan"), float("nan"))
        self._match = None
        self._gyro_version = 0
        self._dvdt = 0.0
        self._v_prev = 0.0
        self._calib_nn = 0.0          # online SpeedNet scale calibration (sum of AI speeds)
        self._calib_gnss = 0.0        # ... and of matching GNSS speeds

    @property
    def ai_scale(self):
        if not self.cfg["ai_online_calibration"] or self._calib_nn < 200.0:
            return 1.0
        return float(np.clip(self._calib_gnss / self._calib_nn, 0.75, 1.35))

    # ----------------------------------------------------------------- helpers
    def _ensure_frame(self, fix: GnssFix):
        if self.frame is None:
            self.frame = LocalFrame(fix.lat, fix.lon)
            if self.roads is not None:
                self.roads = self.roads.reframe(self.frame)
                if self.matcher is not None:
                    self.matcher = HMMMapMatcher(self.roads)

    def initialise(self, lat, lon, heading_deg, speed=0.0, pos_sigma=3.0):
        """Start without GNSS (pure dead reckoning from a known pose)."""
        self._ensure_frame(GnssFix(lat, lon))
        self.ekf.init_position(np.array(self.frame.to_en(lat, lon)), pos_sigma)
        self.ekf.init_heading(float(compass_to_math(heading_deg)), speed)
        self.handler.mode = DR
        self.handler.last_fix_t = -1e9

    # -------------------------------------------------------------------- step
    def step(self, t: float, accel, gyro, gnss: Optional[GnssFix] = None, gnss_lost: bool = False) -> NavState:
        """One IMU epoch. ``gnss`` = a fresh fix or None; ``gnss_lost`` = the receiver
        reports no signal (switches to dead reckoning in this very epoch)."""
        t0 = time.perf_counter()
        c = self.cfg
        accel = np.asarray(accel, float)
        gyro = np.asarray(gyro, float)
        dt = 1.0 / self.fs if self._last_t is None else float(np.clip(t - self._last_t, 1e-4, 0.5))
        self._last_t = t
        ekf = self.ekf

        # 1-3. conditioning + alignment
        u = self.align.up
        h = accel - np.dot(accel, u) * u
        static_rule = self.static_det.update(accel, gyro, float(np.linalg.norm(h)) if self.align.g_vec is not None else 0.0)
        if self._last_fix is not None and self._last_fix[1] is not None and t - self._last_fix[0] < 2.0 \
                and self._last_fix[1] > 1.0:
            static_rule = False            # GNSS says we are moving
        # filtered dv/dt of the navigation solution (for motion-aware levelling)
        if ekf.head_init:
            self._dvdt += (1.0 - np.exp(-dt / 1.0)) * ((ekf.x[3] - self._v_prev) / dt - self._dvdt)
            self._v_prev = ekf.x[3]
        self.align.update(accel, gyro, static_rule,
                          dvdt_hint=self._dvdt if (ekf.head_init and self.handler.gnss_trusted) else float("nan"))
        a_fwd, a_lat, a_up, yaw = self.align.to_vehicle(accel, gyro)
        kin = self.lp(np.array([a_fwd, a_lat, a_up, yaw]))
        bump = self.bump_det.update(a_up)

        # 4. SpeedNet at model rate (decimate fast IMUs by block averaging)
        self._acc_sum += accel; self._gyr_sum += gyro; self._veh_sum += (a_fwd, a_lat, a_up, yaw); self._n_sum += 1
        ai_new = None
        if self._n_sum >= self.decim:
            k = self._n_sum
            f = self.features(*(self._veh_sum / k), self._acc_sum / k, self._gyr_sum / k)
            self._acc_sum[:] = 0; self._gyr_sum[:] = 0; self._veh_sum[:] = 0; self._n_sum = 0
            ai_new = self.nn.push(f)
            if ai_new is not None:
                self._ai = ai_new
        ai_speed, ai_sigma, p_stat = self._ai

        # The gyro yaw axis was (re)calibrated: the bias learnt so far belongs to
        # the old axis, so restart its estimate instead of letting it linger.
        if self.align.gyro_version != self._gyro_version:
            self._gyro_version = self.align.gyro_version
            ekf.x[4] = 0.0
            ekf.P[4, :] = 0.0
            ekf.P[:, 4] = 0.0
            ekf.P[4, 4] = 0.01 ** 2
            ekf.P[2, 2] += np.radians(5) ** 2

        # 5. INS mechanisation (always running -> seamless output)
        initialised = ekf.pos_init and ekf.head_init
        if initialised:
            # Until the yaw mount angle is known the accelerometer's forward axis
            # is unknown: propagate speed as a random walk instead.
            use_acc = self.align.fwd_known and (c["dr_use_accel"] or self.handler.mode != DR)
            a_in = kin[0] if use_acc else 0.0
            if use_acc:
                scale = 3.0 if bump else 1.0
            elif self.handler.mode == DR:
                scale = c["speed_rw_dr"] / c["accel_noise"]      # speed persists, vehicle-like random walk
            else:                                                 # forward axis still unknown: speed must be
                scale = c["speed_rw_unaligned"] / c["accel_noise"]  # free to follow real vehicle dynamics
            ekf.predict(kin[3], a_in, dt, accel_noise_scale=scale)

        # 6. GNSS deficit handler
        fix_ok = gnss is not None and np.isfinite(gnss.lat) and np.isfinite(gnss.lon)
        infl = self.handler.tick(t, fix_ok, gnss.accuracy if fix_ok else None, initialised, lost=gnss_lost)

        # stationarity: rule-based AND/OR confident AI detector
        # Rule and network must not contradict each other: the rule alone can be
        # fooled by very smooth slow rolling, the network alone by idling.
        have_ai = np.isfinite(p_stat)
        ai_static = have_ai and p_stat > c["ai_stationary_p"] and ai_speed < 0.5
        ai_moving = have_ai and p_stat < 0.2 and ai_speed > 2.0
        stationary = (static_rule and not ai_moving) or (ai_static and self.handler.mode == DR and
                                                          self.static_det.gyr.std < 2 * c["static_gyro_std"])
        if initialised and stationary:
            ekf.update_zupt(kin[3])

        # 7. GNSS updates
        if fix_ok:
            self._ensure_frame(gnss)
            en = np.array(self.frame.to_en(gnss.lat, gnss.lon))
            sig = max(c["gnss_pos_sigma_min"], gnss.accuracy or 0.0)
            if not ekf.pos_init:
                ekf.init_position(en, sig)
            spd = gnss.speed
            if not ekf.head_init and spd is not None and spd > c["gnss_course_min_speed"] and gnss.course is not None:
                ekf.init_heading(float(compass_to_math(gnss.course)), spd)
            if ekf.pos_init and ekf.head_init and infl is not None:
                gate = self.handler.mode != RECOVERY and self.handler.consecutive_rejects < 3
                ok, _ = ekf.update_gnss_pos(en, sig * infl, gate=gate)
                self.handler.report(ok)
                if spd is not None and np.isfinite(spd):
                    ekf.update_gnss_speed(spd, c["gnss_speed_sigma"] * infl, gate=gate)
                    if gnss.course is not None and spd > c["gnss_course_min_speed"] and not stationary \
                            and abs(kin[3] - ekf.x[4]) < c["gnss_course_max_yaw_rate"]:
                        ekf.update_heading(float(compass_to_math(gnss.course)),
                                           np.radians(c["gnss_course_sigma_deg"]) * infl, gate=gate)
            # online self-calibration of the AI pseudo-odometer against GNSS speed
            if spd is not None and np.isfinite(ai_speed) and spd > 3.0 and self.handler.mode == "GNSS":
                fdecay = np.exp(-1.0 / c["ai_calib_tau_s"])
                self._calib_nn = self._calib_nn * fdecay + ai_speed
                self._calib_gnss = self._calib_gnss * fdecay + spd
            # alignment learns the yaw mount angle from GNSS kinematics
            if self._last_fix is not None and spd is not None:
                t_prev, spd_prev, crs_prev = self._last_fix
                dtf = t - t_prev
                if 0.2 < dtf < 3.0:
                    crate = float("nan")
                    if gnss.course is not None and crs_prev is not None and spd > 3:
                        crate = float(wrap_angle(compass_to_math(gnss.course) - compass_to_math(crs_prev))) / dtf
                    self.align.on_gnss((spd - spd_prev) / dtf, crate, spd)
            self._last_fix = (t, spd, gnss.course)

        dr = self.handler.mode == DR
        # AI pseudo-odometer
        if initialised and ai_new is not None and c["use_ai_speed"] and not stationary \
                and (dr or c["ai_speed_in_gnss_mode"]) and np.isfinite(ai_speed):
            k = self.ai_scale
            ekf.update_ai_speed(ai_speed * k, ai_sigma * k)

        if ekf.x[3] < 0.0:          # forward-only vehicle model (no reversing in navigation)
            ekf.x[3] = 0.0

        # 8. map matching + map-aided constraints
        self._travel += abs(ekf.x[3]) * dt
        if self.matcher is not None and initialised and t - self._last_map_t >= c["map_interval_s"]:
            self._last_map_t = t
            m = self.matcher.update(t, ekf.x[:2].copy(), ekf.x[2], abs(ekf.x[3]),
                                    pos_sigma=ekf.pos_sigma, travelled=self._travel)
            self._travel = 0.0
            self._match = m
            if m is not None and (dr or c["map_in_gnss_mode"]) and m.confidence >= c["map_min_conf"] \
                    and abs(ekf.x[3]) > 1.0 and ekf.pos_sigma <= c["map_max_pos_sigma"]:
                near_junction = min(m.t_on_seg * m.seg_len, (1 - m.t_on_seg) * m.seg_len) < 8.0
                hs = None if (near_junction or m.seg_len < 25 or not c["map_use_heading"]) \
                    else np.radians(c["map_heading_sigma_deg"])
                ekf.update_map(m.xy, m.road_heading, max(1.0, m.half_width * c["map_cross_sigma_scale"]), hs)

        # outputs
        E, N = ekf.x[0], ekf.x[1]
        lat = lon = float("nan")
        if self.frame is not None and ekf.pos_init:
            lat, lon = self.frame.to_ll(E, N)
        st = NavState(
            t=float(t), e=float(E), n=float(N), lat=float(lat), lon=float(lon),
            heading_deg=float(math_to_compass(ekf.x[2])), speed=float(ekf.x[3]),
            mode=self.handler.mode, pos_sigma=ekf.pos_sigma,
            ai_speed=float(ai_speed), ai_sigma=float(ai_sigma), p_stationary=float(p_stat),
            stationary=bool(stationary), bump=bool(bump),
            yaw_rate=float(kin[3]), gyro_bias=float(ekf.x[4]),
        )
        m = self._match
        if m is not None:
            # display position: snapped onto the road when the match is confident
            if m.confidence >= c["map_min_conf"] and m.dist < 3 * max(m.half_width, st.pos_sigma):
                n_vec = np.array([-np.sin(m.road_heading), np.cos(m.road_heading)])
                off = n_vec @ (ekf.x[:2] - m.xy)
                snapped = ekf.x[:2] - n_vec * off
                st.matched_e, st.matched_n = float(snapped[0]), float(snapped[1])
            st.match_conf = m.confidence
            st.in_tunnel = m.tunnel
        st.latency_us = (time.perf_counter() - t0) * 1e6
        return st

    @property
    def diagnostics(self):
        return {"alignment": self.align.state, "transitions": self.handler.transitions,
                "ekf_updates": self.ekf.stats, "bumps": self.bump_det.events,
                "gyro_bias": float(self.ekf.x[4]), "accel_bias": float(self.ekf.x[5])}
