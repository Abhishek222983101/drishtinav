"""Unit tests for the navigation building blocks."""
import numpy as np
import pytest

from drishtinav.alignment import MountAlignment
from drishtinav.calibration import Biquad, BumpDetector, StaticDetector
from drishtinav.config import make_config
from drishtinav.fusion import VehicleEKF
from drishtinav.geo import LocalFrame, compass_to_math, math_to_compass, wrap_angle
from drishtinav.gnss_handler import DR, GNSS, RECOVERY, GnssDeficitHandler

G = 9.80665


def test_local_frame_roundtrip():
    fr = LocalFrame(28.6139, 77.2090)
    e, n = fr.to_en(28.6239, 77.2190)
    lat, lon = fr.to_ll(e, n)
    assert abs(lat - 28.6239) < 1e-9 and abs(lon - 77.2190) < 1e-9
    assert 1100 < n < 1112 and 970 < e < 985          # ~0.01 deg in Delhi


def test_heading_conventions():
    assert abs(compass_to_math(0) - np.pi / 2) < 1e-12        # North
    assert abs(compass_to_math(90)) < 1e-12                   # East
    assert abs(math_to_compass(np.pi) - 270) < 1e-9           # West
    assert abs(wrap_angle(3 * np.pi) + np.pi) < 1e-12


def test_biquad_dc_gain_and_attenuation():
    f = Biquad(10.0, 2.0, 1)
    y = [f(np.array([1.0]))[0] for _ in range(50)]
    assert abs(y[-1] - 1.0) < 1e-6                            # unity DC gain, no start-up transient
    f = Biquad(10.0, 1.0, 1)
    t = np.arange(400) / 10.0
    out = np.array([f(np.array([np.sin(2 * np.pi * 4.0 * ti)]))[0] for ti in t])
    assert np.std(out[100:]) < 0.1                            # 4 Hz vibration strongly attenuated


def test_static_detector_and_bumps():
    rng = np.random.default_rng(0)
    det = StaticDetector(10.0, acc_std_thr=0.15, gyro_std_thr=0.01, gyro_mean_thr=0.03)
    for _ in range(30):
        s = det.update(np.array([0, 0, G]) + rng.normal(0, 0.03, 3), rng.normal(0, 0.002, 3))
    assert s
    for _ in range(20):
        s = det.update(np.array([0, 0, G]) + rng.normal(0, 0.8, 3), rng.normal(0, 0.1, 3))
    assert not s
    b = BumpDetector(10.0, thr=2.5)
    flags = [b.update(v) for v in [0, 0.1, 5.0, 1.0, 0.2, 0, 0, 0, 0, 0, 0, 0]]
    assert b.events == 1 and flags[2] and not flags[-1]


def _rot(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))


def _drive_alignment(R_mount, permute_gyro=None, n=6000, seed=1, R_after=None, switch_at=None):
    """Stop-and-go driving with turns, sensed by a phone mounted with R_mount."""
    rng = np.random.default_rng(seed)
    fs = 10.0
    t = np.arange(n) / fs
    v = (8 + 6 * np.sin(2 * np.pi * t / 40) ** 2) * np.clip((t - 10) / 8, 0, 1)   # rest, then launch
    a_long = np.gradient(v, t)
    yaw = 0.25 * np.sin(2 * np.pi * t / 23) * (v > 1)
    al = MountAlignment(fs)
    course = np.cumsum(yaw) / fs
    last = None
    out = []
    for i in range(n):
        if switch_at is not None and i == switch_at:
            R_mount = R_after                    # the phone slips in its holder
        f_veh = np.array([a_long[i], v[i] * yaw[i], G])
        acc = R_mount.T @ f_veh + rng.normal(0, 0.05, 3)
        gyr = R_mount.T @ np.array([0, 0, yaw[i]]) + rng.normal(0, 0.002, 3)
        if permute_gyro is not None:
            gyr = gyr[permute_gyro]
        al.update(acc, gyr, v[i] < 0.1, 0.0 if abs(a_long[i]) < 0.1 else np.nan)
        if i % 10 == 0 and i > 0:
            if last is not None:
                cr = (course[i] - course[last]) / ((i - last) / fs)
                al.on_gnss((v[i] - v[last]) / ((i - last) / fs), cr, v[i])
            last = i
        out.append((al.to_vehicle(acc, gyr), a_long[i], yaw[i]))
    return al, out


def test_alignment_recovers_mount_and_forward_axis():
    R = _rot(np.radians(5), np.radians(-60), np.radians(35))
    al, out = _drive_alignment(R)
    assert al.fwd_known
    fwd = np.array([o[0][0] for o in out[3000:]])
    ref = np.array([o[1] for o in out[3000:]])
    assert np.corrcoef(fwd, ref)[0, 1] > 0.95
    yaw = np.array([o[0][3] for o in out[3000:]])
    assert np.corrcoef(yaw, [o[2] for o in out[3000:]])[0, 1] > 0.99


def test_alignment_handles_permuted_gyro_columns():
    """IO-VNBD's AndroSensor logs put the yaw rate in the 2nd gyro column."""
    al, out = _drive_alignment(np.eye(3), permute_gyro=[0, 2, 1])
    assert al.gyro_up is not None and al.gyro_separate
    yaw = np.array([o[0][3] for o in out[3000:]])
    assert np.corrcoef(yaw, [o[2] for o in out[3000:]])[0, 1] > 0.99
    assert abs(al.gyro_scale - 1.0) < 0.05


def test_ekf_converges_and_zupt():
    cfg = make_config()
    ekf = VehicleEKF(cfg)
    ekf.init_position(np.array([0.0, 0.0]), 5.0)
    ekf.init_heading(0.3, 10.0)
    for k in range(600):                       # true: 10 m/s due East
        ekf.predict(0.0, 0.0, 0.1)
        if k % 10 == 0:
            ekf.update_gnss_pos(np.array([(k + 1) * 1.0, 0.0]), 2.5)
            ekf.update_gnss_speed(10.0, 0.3)
            ekf.update_heading(0.0, np.radians(3))
    assert abs(ekf.x[2]) < np.radians(2) and abs(ekf.x[3] - 10) < 0.3
    for _ in range(30):
        ekf.predict(0.0, 0.0, 0.1)
        ekf.update_zupt(0.0)
    assert abs(ekf.x[3]) < 0.2


def test_map_update_never_touches_speed_or_biases():
    ekf = VehicleEKF(make_config())
    ekf.init_position(np.array([0.0, 0.0]), 3.0)
    ekf.init_heading(0.1, 12.0)
    for _ in range(50):
        ekf.predict(0.0, 0.0, 0.1)
    before = ekf.x[3:].copy()
    ekf.update_map(np.array([60.0, -5.0]), 0.0, 2.0, np.radians(5))
    assert np.allclose(ekf.x[3:], before)       # Schmidt-consider update: geometry only


def test_gnss_handler_modes_and_instant_loss():
    cfg = make_config()
    h = GnssDeficitHandler(cfg)
    t = 0.0
    for _ in range(50):                          # 1 Hz fixes at 10 Hz IMU
        for k in range(10):
            t += 0.1
            h.tick(t, k == 0, 3.0)
    assert h.mode == GNSS
    t += 0.1
    h.tick(t, False, None, lost=True)            # receiver reports loss -> same epoch
    assert h.mode == DR
    t += 0.1
    h.tick(t, True, 3.0)
    assert h.mode == RECOVERY
    assert abs(h.timeout - 1.6) < 0.2            # adapts to the 1 Hz fix rate


def test_gnss_handler_timeout_adapts_to_fix_rate():
    cfg = make_config("fog")                     # base timeout 0.3 s but GNSS still 1 Hz
    h = GnssDeficitHandler(cfg)
    t, modes = 0.0, set()
    for _ in range(60):
        for k in range(200):
            t += 0.005
            h.tick(t, k == 0, 3.0)
            modes.add(h.mode)
    assert DR not in modes                       # no false outages between 1 Hz fixes


def test_mount_disturbance_is_detected_and_relearned():
    R1 = _rot(np.radians(3), np.radians(-55), np.radians(20))
    R2 = _rot(np.radians(-8), np.radians(-35), np.radians(60))      # phone knocked in the holder
    al, out = _drive_alignment(R1, n=9000, R_after=R2, switch_at=3000)
    assert al.mount_events >= 1
    fwd = np.array([o[0][0] for o in out[6500:]])
    assert np.corrcoef(fwd, [o[1] for o in out[6500:]])[0, 1] > 0.9      # forward axis re-learned
    yaw = np.array([o[0][3] for o in out[6500:]])
    assert np.corrcoef(yaw, [o[2] for o in out[6500:]])[0, 1] > 0.99


def test_no_false_mount_events_in_normal_driving():
    al, _ = _drive_alignment(_rot(np.radians(5), np.radians(-60), np.radians(35)), n=6000)
    assert al.mount_events == 0
