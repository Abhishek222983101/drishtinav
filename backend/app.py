"""Flask Backend for DrishtiNav - serves API + static frontend."""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from flask import Flask, send_from_directory, jsonify, send_file

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import generate_synthetic_data, gps_to_enu
from imu_engine import (
    IMUCalibrator, AlignmentEngine, SpeedPredictor,
    DeadReckoner, EKFusion, MapMatcher
)

BASE = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE / "output"
FRONTEND_DIR = BASE / "frontend"
OUTPUT_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=str(FRONTEND_DIR))

# Cache for last run
_last_run = {}


def run_pipeline(duration_sec=120, dt=0.1):
    """Execute full navigation pipeline."""
    global _last_run

    # 1. Generate/load data
    df = generate_synthetic_data(duration_sec, dt)
    df = gps_to_enu(df)
    n = len(df)
    actual_dt = dt

    # 2. Extract sensor data
    accel = df[['accel_x', 'accel_y', 'accel_z']].values
    gyro = df[['gyro_yaw', 'gyro_pitch', 'gyro_roll']].values
    mag = df[['mag_x', 'mag_y', 'mag_z']].values
    gps_speed_kmh = df['gps_speed_kmh'].values
    gps_speed_ms = np.where(np.isnan(gps_speed_kmh), 0, gps_speed_kmh / 3.6)

    # GPS positions (ENU)
    gps_positions = np.column_stack([df['enu_e'].values, df['enu_n'].values])
    gps_available = ~np.isnan(df['gps_lat'].values)

    # Ground truth
    gt_x = df.attrs.get('gt_x', np.zeros(n))
    gt_y = df.attrs.get('gt_y', np.zeros(n))
    gt_speed = df.attrs.get('gt_speed', np.zeros(n))
    gt_heading = df.attrs.get('gt_heading', np.zeros(n))
    tunnel_start = df.attrs.get('tunnel_start', 400)
    tunnel_end = df.attrs.get('tunnel_end', 700)

    # 3. IMU Calibration
    calibrator = IMUCalibrator(sample_rate=1.0/dt)
    accel_cal, gyro_cal = calibrator.process(accel, gyro)

    # 4. Alignment (yaw from tilt-compensated magnetometer)
    aligner = AlignmentEngine()
    yaw0, pitch0, roll0 = aligner.estimate_alignment(accel_cal, gyro_cal, mag=mag)

    # Magnetometer heading: tilt-compensated with estimated pitch/roll.
    # The geomagnetic field persists inside tunnels, making this the key
    # drift-free heading source when GPS is denied.
    cp, sp = np.cos(pitch0), np.sin(pitch0)
    cr, sr = np.cos(roll0), np.sin(roll0)
    mx, my, mz = mag[:, 0], mag[:, 1], mag[:, 2]
    mx_h = mx * cp + my * sr * sp + mz * cr * sp
    my_h = my * cr - mz * sr
    mag_heading = np.arctan2(my_h, mx_h)

    # 5. Rotate to nav frame
    accel_nav = aligner.rotate_body_to_nav(accel_cal)

    # 6. Train speed predictor on GPS-available segments
    # window_size=15 (1.5s): short enough to track speed ramps with low lag,
    # long enough for stable statistics of the speed-dependent vibration
    speed_predictor = SpeedPredictor(window_size=15)
    imu_features = np.column_stack([accel_cal, gyro_cal])

    # Pothole/bump isolation: bump spikes corrupt IMU windows used for
    # training and inference (5 m/s^2 spikes vs 0.5 vibration signal)
    bump_mask = calibrator.detect_potholes(accel_cal[:, 2], threshold=3.0, min_gap=15)
    # dilate mask by +/- window_size so no training window touches a bump
    bump_pad = np.copy(bump_mask)
    pad = speed_predictor.window_size
    for i in np.where(bump_mask)[0]:
        bump_pad[max(0, i - pad):i + pad] = True

    windows = []
    speed_targets = []
    for i in range(speed_predictor.window_size, n):
        if gps_available[i] and gps_speed_ms[i] > 0.5 and not bump_pad[i]:
            windows.append(imu_features[i-speed_predictor.window_size:i])
            speed_targets.append(gps_speed_ms[i])

    if len(windows) > 50:
        speed_predictor.train(windows, speed_targets)

    # Predict speed for all (AI model - works with or without GPS)
    speed_pred = speed_predictor.predict_sequence(imu_features, dt)

    # Inference outlier rejection: bump spikes corrupt feature windows, so
    # replace those predictions with interpolated values from clean neighbors
    bad_pred = bump_pad & (speed_pred > 0)
    if bad_pred.any():
        good_idx = np.where(~bad_pred)[0]
        speed_pred[bad_pred] = np.interp(
            np.where(bad_pred)[0], good_idx, speed_pred[good_idx]
        )

    # 7. Dead Reckoning (zero GPS): AI-predicted speed + mag-aided gyro heading.
    #    This is the "what IMU+AI alone can do" baseline - no satellite info at all.
    dr = DeadReckoner(dt=actual_dt)
    dr_pos, dr_vel, dr_heading = dr.integrate(accel_nav, gyro_cal, speed_pred, mag_heading)

    # 8. EKF Fusion: GPS position (when visible) + magnetometer heading + AI speed
    ekf = EKFusion(dt=actual_dt)
    ekf_pos, ekf_vel = ekf.process_sequence(
        accel_nav, gyro_cal, gps_available, gps_positions, speed_pred, mag_heading
    )

    # 9. Map Matching on EKF output
    matcher = MapMatcher()
    matcher.set_road_network(gps_positions)
    ekf_matched, ekf_matched_h = matcher.match_sequence(ekf_pos, dr_heading)

    # 9b. Map Matching on pure DR output (shows how much road-constraints
    #     can rescue even a GPS-free solution - good for the demo story)
    matcher2 = MapMatcher()
    matcher2.set_road_network(gps_positions)
    dr_matched, _ = matcher2.match_sequence(dr_pos, dr_heading)

    # 10. Compute metrics
    def compute_drift(est_pos, gt_x, gt_y):
        error = np.sqrt((est_pos[:, 0] - gt_x)**2 + (est_pos[:, 1] - gt_y)**2)
        # Path length (arc length), not displacement - honest drift denominator
        total_dist = float(np.sum(np.sqrt(np.diff(gt_x)**2 + np.diff(gt_y)**2)))
        final_drift = error[-1]
        drift_pct = (final_drift / total_dist * 100) if total_dist > 0 else 999
        return {
            'rmse': float(np.sqrt(np.mean(error**2))),
            'max_error': float(np.max(error)),
            'final_error': float(final_drift),
            'drift_pct': float(min(drift_pct, 999)),
            'error_over_time': error.tolist()
        }

    dr_metrics = compute_drift(dr_pos, gt_x, gt_y)
    ekf_metrics = compute_drift(ekf_pos, gt_x, gt_y)
    map_metrics = compute_drift(ekf_matched, gt_x, gt_y)
    dr_matched_metrics = compute_drift(dr_matched, gt_x, gt_y)

    # Speed prediction metrics (only where GPS speed is a valid reference)
    valid_mask = gps_available & (gps_speed_ms > 0.5)
    if valid_mask.sum() > 10:
        speed_rmse = float(np.sqrt(np.mean((speed_pred[valid_mask] - gps_speed_ms[valid_mask])**2)))
    else:
        speed_rmse = float(np.sqrt(np.mean((speed_pred[100:300] - gps_speed_ms[100:300])**2)))

    # 11. Generate plots
    plot_drift_analysis(gt_x, gt_y, dr_pos, ekf_pos, ekf_matched, tunnel_start, tunnel_end)
    plot_speed_comparison(gt_speed, speed_pred, gps_speed_ms, tunnel_start, tunnel_end)
    plot_trajectory(gt_x, gt_y, dr_pos, ekf_pos, ekf_matched, tunnel_start, tunnel_end)

    # 12. Prepare JSON response
    # Subsample for frontend (every 5th point)
    step = max(1, n // 200)
    indices = list(range(0, n, step))

    _last_run = {
        'ground_truth': {
            'x': gt_x[indices].tolist(),
            'y': gt_y[indices].tolist(),
        },
        'gps_available': gps_available[indices].tolist(),
        'gt_heading_deg': np.degrees(gt_heading)[indices].tolist(),
        'gt_speed_ms': gt_speed[indices].tolist(),
        'dead_reckoning': {
            'x': dr_pos[indices, 0].tolist(),
            'y': dr_pos[indices, 1].tolist(),
            'metrics': dr_metrics,
        },
        'dr_map_matched': {
            'x': dr_matched[indices, 0].tolist(),
            'y': dr_matched[indices, 1].tolist(),
            'metrics': dr_matched_metrics,
        },
        'ekf_fusion': {
            'x': ekf_pos[indices, 0].tolist(),
            'y': ekf_pos[indices, 1].tolist(),
            'metrics': ekf_metrics,
        },
        'map_matched': {
            'x': ekf_matched[indices, 0].tolist(),
            'y': ekf_matched[indices, 1].tolist(),
            'metrics': map_metrics,
        },
        'speed': {
            'ai_predicted': speed_pred[indices].tolist(),
            'gps_ground_truth': gt_speed[indices].tolist(),
            'timestamps': (np.arange(n)[indices] * dt).tolist(),
        },
        'tunnel': {
            'start_idx': int(tunnel_start),
            'end_idx': int(tunnel_end),
            'start_sec': float(tunnel_start * dt),
            'end_sec': float(tunnel_end * dt),
        },
        'metrics_summary': {
            'dr_rmse': dr_metrics['rmse'],
            'dr_drift_pct': dr_metrics['drift_pct'],
            'dr_matched_rmse': dr_matched_metrics['rmse'],
            'dr_matched_drift_pct': dr_matched_metrics['drift_pct'],
            'ekf_rmse': ekf_metrics['rmse'],
            'ekf_drift_pct': ekf_metrics['drift_pct'],
            'map_rmse': map_metrics['rmse'],
            'map_drift_pct': map_metrics['drift_pct'],
            'speed_rmse': speed_rmse,
            'total_distance': float(np.sum(np.sqrt(np.diff(gt_x)**2 + np.diff(gt_y)**2))),
            'total_time': float(duration_sec),
            'dr_pass': dr_metrics['drift_pct'] < 10,
            'ekf_pass': ekf_metrics['drift_pct'] < 10,
            'map_pass': map_metrics['drift_pct'] < 10,
        }
    }

    return _last_run


def plot_trajectory(gt_x, gt_y, dr, ekf, map_matched, ts, te):
    """Plot trajectory comparison."""
    plt.figure(figsize=(10, 8))
    plt.plot(gt_x, gt_y, 'k-', linewidth=2, label='Ground Truth', zorder=5)
    plt.plot(dr[:, 0], dr[:, 1], 'r--', linewidth=1.5, alpha=0.8, label='Dead Reckoning (GPS-denied)')
    plt.plot(ekf[:, 0], ekf[:, 1], 'b-', linewidth=1.5, alpha=0.8, label='EKF Fusion')
    plt.plot(map_matched[:, 0], map_matched[:, 1], 'g-', linewidth=1.5, alpha=0.8, label='Map-Matched Fusion')
    # Shade tunnel region
    if ts < len(gt_x) and te < len(gt_x):
        plt.axvspan(gt_x[ts], gt_x[te], alpha=0.15, color='red', label='GPS Outage Zone')
    plt.xlabel('East (m)', fontsize=12)
    plt.ylabel('North (m)', fontsize=12)
    plt.title('DrishtiNav - Trajectory Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / 'trajectory.png'), dpi=100)
    plt.close()


def plot_drift_analysis(gt_x, gt_y, dr, ekf, map_matched, ts, te):
    """Plot drift over time."""
    n = len(gt_x)
    t = np.arange(n) * 0.1

    dr_err = np.sqrt((dr[:, 0] - gt_x)**2 + (dr[:, 1] - gt_y)**2)
    ekf_err = np.sqrt((ekf[:, 0] - gt_x)**2 + (ekf[:, 1] - gt_y)**2)
    map_err = np.sqrt((map_matched[:, 0] - gt_x)**2 + (map_matched[:, 1] - gt_y)**2)

    plt.figure(figsize=(10, 6))
    plt.plot(t, dr_err, 'r-', linewidth=2, label='Dead Reckoning')
    plt.plot(t, ekf_err, 'b-', linewidth=2, label='EKF Fusion')
    plt.plot(t, map_err, 'g-', linewidth=2, label='Map-Matched Fusion')
    plt.axvspan(ts * 0.1, te * 0.1, alpha=0.2, color='red', label='GPS Outage')
    plt.axhline(y=10, color='orange', linestyle=':', alpha=0.5, label='10% Threshold')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Position Error (m)', fontsize=12)
    plt.title('DrishtiNav - Position Drift Over Time', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / 'drift.png'), dpi=100)
    plt.close()


def plot_speed_comparison(gt_speed, ai_speed, gps_speed, ts, te):
    """Plot AI speed prediction vs ground truth."""
    n = len(gt_speed)
    t = np.arange(n) * 0.1

    plt.figure(figsize=(10, 5))
    plt.plot(t, gt_speed * 3.6, 'k-', linewidth=2, label='True Speed')
    plt.plot(t, ai_speed * 3.6, 'orange', linewidth=1.5, alpha=0.8, label='AI Predicted Speed')
    valid_gps = ~np.isnan(gps_speed) & (gps_speed > 0)
    if valid_gps.sum() > 0:
        plt.scatter(t[valid_gps], gps_speed[valid_gps] * 3.6, s=10, alpha=0.5, color='green', label='GPS Speed (noisy)')
    plt.axvspan(ts * 0.1, te * 0.1, alpha=0.2, color='red', label='GPS Outage')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Speed (km/h)', fontsize=12)
    plt.title('DrishtiNav - AI Speed Estimation', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / 'speed.png'), dpi=100)
    plt.close()


# --- Routes ---
@app.route('/')
def index():
    return send_from_directory(str(FRONTEND_DIR), 'index.html')

@app.route('/frontend/<path:filename>')
def serve_frontend(filename):
    return send_from_directory(str(FRONTEND_DIR), filename)

@app.route('/output/<path:filename>')
def serve_output(filename):
    return send_from_directory(str(OUTPUT_DIR), filename)

@app.route('/api/run')
def api_run():
    from flask import request
    try:
        duration = request.args.get('duration', 120, type=int)
        duration = max(30, min(300, duration))
        result = run_pipeline(duration_sec=duration)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/plots/<name>')
def api_plot(name):
    filepath = OUTPUT_DIR / name
    if filepath.exists():
        return send_file(str(filepath), mimetype='image/png')
    return jsonify({'error': 'not found'}), 404


if __name__ == '__main__':
    print("=" * 50)
    print("  DRISHTINAV - AI Dead Reckoning System")
    print("  Starting server at http://localhost:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False)
