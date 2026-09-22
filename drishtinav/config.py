"""Engine configuration profiles.

``smartphone``  consumer MEMS IMU at 10-100 Hz, 1 Hz phone GNSS (PS on-device target)
``mems_edge``   external automotive-grade MEMS IMU at 100 Hz
``fog``         fibre-optic-gyro IMU at 200 Hz (PS edge-engine target)

Any external IMU can be used by picking the closest profile and overriding the
noise figures from its datasheet (see docs/EDGE_ENGINE.md).
"""
from __future__ import annotations

import copy

BASE = {
    "imu_rate_hz": 10.0,
    "model_rate_hz": 10.0,          # SpeedNet is trained at 10 Hz; faster IMUs are decimated
    "gyro_frame": "auto",           # "auto" | "shared"
    # process noise
    "q_pos": 0.02,                  # m^2/s  (unmodelled side-slip / lever arm)
    "gyro_noise": 0.01,             # rad/s  white
    "accel_noise": 1.0,             # m/s^2  after 2 Hz low-pass: phone-in-holder wobble leaks gravity
                                    #        (IO-VNBD phones: r(a_fwd, dv/dt) ~ 0.45), so speed leans on SpeedNet
    "gyro_bias_rw": 2e-5,           # rad/s/sqrt(s)  (MEMS bias is stable over minutes)
    "accel_bias_rw": 5e-3,          # m/s^2/sqrt(s)
    # measurement noise
    "gnss_pos_sigma_min": 2.5,      # m
    "gnss_pos_updates_biases": False,  # position fixes (multipath-prone) may not steer sensor biases
    "gnss_speed_sigma": 0.3,        # m/s
    "gnss_course_sigma_deg": 3.0,
    "gnss_course_min_speed": 4.0,   # m/s
    "gnss_course_max_yaw_rate": 0.08,  # rad/s: GNSS course lags in turns -> use on straights only
    "zupt_sigma": 0.03,
    "zaru_sigma": 0.002,
    "ai_sigma_scale": 2.0,           # SpeedNet errors are correlated over ~10-20 s: inflate R (tuned on val drives)
    "ai_sigma_floor": 0.35,
    # AI speed use
    "use_ai_speed": True,
    "dr_use_accel": False,          # phone a_fwd is too wobbly to integrate while GNSS-denied (see docs)
    "speed_rw_dr": 1.0,             # m/s^2/sqrt(s): speed random walk when a_fwd is not integrated
    "speed_rw_unaligned": 1.5,      # m/s^2/sqrt(s): vehicle dynamics before the forward axis is learnt
    "ai_online_calibration": True,  # learn a per-trip SpeedNet scale from GNSS while it is available
    "ai_calib_tau_s": 300.0,
    "ai_speed_in_gnss_mode": False,
    "ai_stationary_p": 0.85,
    # map matching
    "use_map": True,
    "map_min_conf": 0.9,             # tuned on validation drives (never on test)
    "map_cross_sigma_scale": 0.6,   # x road half-width
    "map_heading_sigma_deg": 6.0,
    "map_interval_s": 1.0,
    "map_in_gnss_mode": False,
    "map_max_pos_sigma": 25.0,      # m: only constrain while the INS is still tight (ambiguity guard)
    "map_use_heading": True,
    # GNSS deficit handler
    "gnss_timeout_s": 1.5,          # no fix for this long -> dead-reckoning mode
    "gnss_max_accuracy_m": 25.0,    # worse than this -> degraded
    "recovery_fixes": 2,            # consistent fixes before GNSS is fully trusted again
    "recovery_sigma_inflate": 3.0,
    # stationarity (rule-based detector)
    # thresholds fitted on IO-VNBD training drives: 78 % recall, 6 % false alarms
    "static_acc_std": 0.15,
    "static_gyro_std": 0.01,
    "static_gyro_mean": 0.03,
    "static_horiz_accel": 0.3,      # m/s^2: levelled horizontal specific force at rest
    "bump_thr": 2.5,
}

PROFILES = {
    "smartphone": {},
    "mems_edge": {"imu_rate_hz": 100.0, "dr_use_accel": True, "gyro_noise": 0.004, "accel_noise": 0.2, "gyro_bias_rw": 5e-5,
                  "accel_bias_rw": 2e-3, "gnss_timeout_s": 0.5},
    "fog": {"imu_rate_hz": 200.0, "dr_use_accel": True, "gyro_noise": 3e-4, "accel_noise": 0.08, "gyro_bias_rw": 1e-6,
            "accel_bias_rw": 5e-4, "gnss_timeout_s": 0.3, "static_gyro_std": 0.005, "static_gyro_mean": 0.01,
            "static_acc_std": 0.05, "static_horiz_accel": 0.08,
            # SpeedNet is trained on smartphone vibration signatures; a FOG unit's
            # accelerometer is good enough to integrate, so the physics INS carries
            # speed (retrain SpeedNet on the target IMU with training/ to enable it).
            "use_ai_speed": False},
}


def make_config(profile: str = "smartphone", **overrides) -> dict:
    cfg = copy.deepcopy(BASE)
    cfg.update(PROFILES[profile])
    cfg["profile"] = profile
    for k, v in overrides.items():
        if k not in cfg:
            raise KeyError(f"unknown config key {k!r}")
        cfg[k] = v
    return cfg
