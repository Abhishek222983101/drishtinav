"""Data Loader for DrishtiNav - IO-VNBD CSV or synthetic data."""
import numpy as np
import pandas as pd


def load_ionvbd_csv(filepath):
    """Load an IO-VNBD smartphone CSV file."""
    df = pd.read_csv(filepath, header=None, on_bad_lines='skip')
    n_cols = df.shape[1]
    if n_cols >= 24:
        cols = ['gps_lat','gps_lon','gps_alt','gps_speed_kmh','gps_accuracy',
                'gps_orientation','gps_satellites','time_ms','date',
                'accel_x','accel_y','accel_z',
                'gravity_x','gravity_y','gravity_z',
                'gyro_yaw','gyro_pitch','gyro_roll',
                'mag_x','mag_y','mag_z',
                'orient_yaw','orient_pitch','orient_roll']
        df.columns = cols[:n_cols]
    elif n_cols >= 12:
        cols = ['gps_lat','gps_lon','gps_alt','gps_speed_kmh','gps_accuracy',
                'time_ms','date','accel_x','accel_y','accel_z','gyro_yaw','gyro_pitch']
        df.columns = cols[:n_cols]
    else:
        raise ValueError(f"CSV has {n_cols} columns, need >= 12")
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=['accel_x','accel_y','accel_z'])
    return df


def gps_to_enu(df, origin_lat=None, origin_lon=None):
    """Convert GPS lat/lon to local East-North-Up meters."""
    df = df.copy()
    if 'gps_lat' not in df.columns:
        df['enu_e'] = 0.0
        df['enu_n'] = 0.0
        return df
    mask = df['gps_lat'].notna() & df['gps_lon'].notna()
    if mask.sum() == 0:
        df['enu_e'] = 0.0
        df['enu_n'] = 0.0
        return df
    if origin_lat is None:
        origin_lat = df.loc[mask, 'gps_lat'].iloc[0]
        origin_lon = df.loc[mask, 'gps_lon'].iloc[0]
    R = 6378137.0
    lat_r = np.radians(df['gps_lat'].values.astype(float))
    lon_r = np.radians(df['gps_lon'].values.astype(float))
    olat = np.radians(origin_lat)
    olon = np.radians(origin_lon)
    df['enu_e'] = (lon_r - olon) * np.cos((lat_r + olat) / 2) * R
    df['enu_n'] = (lat_r - olat) * R
    df['origin_lat'] = origin_lat
    df['origin_lon'] = origin_lon
    return df


def generate_synthetic_data(duration_sec=120, dt=0.1):
    """Generate realistic synthetic IMU+GPS with tunnel outage."""
    np.random.seed(42)
    n = int(duration_sec / dt)
    t = np.arange(n) * dt

    # Key time indices (scaled to duration)
    turn1_start = max(10, int(n * 0.25))
    turn1_end = min(n, turn1_start + max(10, int(n * 0.08)))
    turn1_done = min(n, turn1_end + max(30, int(n * 0.25)))
    tunnel_start = max(0, int(n * 0.33))
    tunnel_end = min(n, int(n * 0.58))
    turn2_start = max(tunnel_end + 1, int(n * 0.62))
    turn2_end = min(n, turn2_start + max(10, int(n * 0.08)))

    # Ground truth: speed profile
    speed = np.ones(n) * 15.0
    if turn1_end <= n and turn1_start < turn1_end:
        seg_len = turn1_end - turn1_start
        speed[turn1_start:turn1_end] = np.linspace(15, 8, seg_len)
    if turn1_done <= n and turn1_end < turn1_done:
        seg_len = turn1_done - turn1_end
        speed[turn1_end:turn1_done] = np.linspace(8, 15, seg_len)
    if turn2_end <= n and turn2_start < turn2_end:
        seg_len = turn2_end - turn2_start
        speed[turn2_start:turn2_end] = np.linspace(15, 8, seg_len)

    # Heading: straight, turn right, straight, turn left, straight
    heading = np.zeros(n)
    turn1_mid = min(n, turn1_start + (turn1_end - turn1_start))
    for i in range(turn1_start, min(turn1_done, n)):
        frac = (i - turn1_start) / max(1, turn1_done - turn1_start)
        heading[i] = heading[max(0, turn1_start-1)] + (np.pi / 2) * frac
    for i in range(min(turn1_done, n), min(turn2_start, n)):
        heading[i] = heading[min(turn1_done, n) - 1] if turn1_done > 0 else 0
    turn2_mid = min(n, turn2_start + (turn2_end - turn2_start))
    for i in range(turn2_start, min(n, turn2_end)):
        frac = (i - turn2_start) / max(1, turn2_end - turn2_start)
        heading[i] = heading[max(0, turn2_start-1)] - (np.pi / 2) * frac

    # Gentle traffic modulation on cruise segments (not during turns):
    # gives the AI speed predictor training examples across the full speed
    # range while GPS is available, and makes the speed profile realistic.
    cruise = (
        (np.arange(n) < turn1_start)
        | ((np.arange(n) >= turn1_done) & (np.arange(n) < turn2_start))
        | (np.arange(n) >= turn2_end)
    )
    mod = 1.0 + 0.16 * np.sin(2 * np.pi * t / 17.0) + 0.11 * np.sin(2 * np.pi * t / 9.0 + 1.3)
    speed[cruise] = np.clip(speed[cruise] * mod[cruise], 6.5, 16.0)

    # Smooth the whole profile: removes 1-sample discontinuities at segment
    # boundaries (which would otherwise create huge accel spikes in
    # np.gradient and corrupt IMU features / AI training windows).
    pad = 12
    speed_padded = np.concatenate([np.full(pad, speed[0]), speed, np.full(pad, speed[-1])])
    speed = np.convolve(speed_padded, np.ones(25) / 25.0, mode='valid')[:n]

    # Integrate GT position
    gt_x = np.zeros(n)
    gt_y = np.zeros(n)
    for i in range(1, n):
        gt_x[i] = gt_x[i-1] + speed[i] * np.cos(heading[i]) * dt
        gt_y[i] = gt_y[i-1] + speed[i] * np.sin(heading[i]) * dt

    # Simulate GPS: noisy but available, except tunnel
    gps_speed = speed.copy()
    gps_speed[tunnel_start:tunnel_end] = np.nan
    gps_x = gt_x + np.random.randn(n) * 2.0
    gps_y = gt_y + np.random.randn(n) * 2.0
    gps_x[tunnel_start:tunnel_end] = np.nan
    gps_y[tunnel_start:tunnel_end] = np.nan

    # Convert to lat/lon (simple ENU->GPS around a reference point)
    ref_lat, ref_lon = 28.6139, 77.2090  # Delhi
    lat = ref_lat + gt_y / 111320.0
    lon = ref_lon + gt_x / (111320.0 * np.cos(np.radians(ref_lat)))
    gps_lat = lat.copy()
    gps_lon = lon.copy()
    gps_lat[tunnel_start:tunnel_end] = np.nan
    gps_lon[tunnel_start:tunnel_end] = np.nan

    # Generate IMU: accel + gyro
    # Accelerometer: forward accel + gravity + noise
    accel_forward = np.gradient(speed, dt)
    accel_x = accel_forward * np.cos(heading) + np.random.randn(n) * 0.08
    accel_y = accel_forward * np.sin(heading) + np.random.randn(n) * 0.08
    accel_z = 9.81 + np.random.randn(n) * 0.1  # gravity + noise

    # Add vibration/engine noise - amplitude scales with speed (physically real:
    # engine RPM and road-induced vibration increase with vehicle speed, giving
    # the AI speed predictor a learnable IMU -> speed mapping).
    # NOTE: frequencies stay below Nyquist (fs=10Hz) so the speed-dependent
    # amplitude survives the anti-vibration low-pass filter downstream.
    engine_vib = (0.05 + 0.09 * speed) * np.sin(2 * np.pi * 3.5 * t)
    accel_x += engine_vib
    accel_y += (0.03 + 0.055 * speed) * np.sin(2 * np.pi * 2.2 * t)
    accel_x += 0.022 * speed * np.random.randn(n)
    accel_y += 0.016 * speed * np.random.randn(n)

    # Add pothole bumps (relative to duration)
    bump_indices = [int(n*0.12), int(n*0.28), int(n*0.42), int(n*0.55), int(n*0.7)]
    for idx in bump_indices:
        if 0 < idx < n:
            bump = np.exp(-np.arange(min(10, n-idx)) * 0.5) * 5.0
            accel_z[idx:idx+len(bump)] += bump

    # Gyroscope: heading rate + noise
    gyro_yaw = np.gradient(heading, dt) + np.random.randn(n) * 0.02
    gyro_pitch = np.random.randn(n) * 0.01
    gyro_roll = np.random.randn(n) * 0.01

    # GPS speed with noise
    gps_speed_kmh = speed.copy() * 3.6
    gps_speed_kmh[tunnel_start:tunnel_end] = np.nan
    gps_speed_kmh += np.random.randn(n) * 1.0

    # Build DataFrame
    df = pd.DataFrame({
        'gps_lat': gps_lat, 'gps_lon': gps_lon, 'gps_alt': 215.0,
        'gps_speed_kmh': gps_speed_kmh, 'gps_accuracy': 5.0,
        'gps_orientation': np.degrees(heading),
        'gps_satellites': np.where(np.isnan(gps_lat), 0, 8),
        'time_ms': t * 1000, 'date': '2026-09-01 00:00:00',
        'accel_x': accel_x, 'accel_y': accel_y, 'accel_z': accel_z,
        'gravity_x': np.zeros(n), 'gravity_y': np.zeros(n), 'gravity_z': 9.81 * np.ones(n),
        'gyro_yaw': gyro_yaw, 'gyro_pitch': gyro_pitch, 'gyro_roll': gyro_roll,
        'mag_x': 25.0 * np.cos(heading) + np.random.randn(n) * 2,
        'mag_y': 25.0 * np.sin(heading) + np.random.randn(n) * 2,
        'mag_z': 40.0 + np.random.randn(n) * 2,
        'orient_yaw': np.degrees(heading), 'orient_pitch': np.zeros(n), 'orient_roll': np.zeros(n),
    })

    # Ground truth as attributes for evaluation
    df.attrs['gt_x'] = gt_x
    df.attrs['gt_y'] = gt_y
    df.attrs['gt_speed'] = speed
    df.attrs['gt_heading'] = heading
    df.attrs['tunnel_start'] = tunnel_start
    df.attrs['tunnel_end'] = tunnel_end

    return df


def create_tunnel_mask(df, start_sec=40.0, end_sec=70.0):
    """Create boolean mask for GPS-denied periods."""
    if 'time_ms' in df.columns:
        t_sec = df['time_ms'].values / 1000.0
    else:
        t_sec = np.arange(len(df)) * 0.1
    return (t_sec >= start_sec) & (t_sec <= end_sec)


if __name__ == "__main__":
    df = generate_synthetic_data(120, 0.1)
    df = gps_to_enu(df)
    print(f"Generated {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    print(f"GPS valid: {df['gps_lat'].notna().sum()}/{len(df)}")
    gt_x = df.attrs.get('gt_x')
    if gt_x is not None:
        print(f"Total distance: {np.sqrt(gt_x[-1]**2 + df.attrs['gt_y'][-1]**2):.1f} m")
