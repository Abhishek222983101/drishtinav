"""Core IMU Navigation Engine for DrishtiNav."""
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.linear_model import Ridge


class IMUCalibrator:
    """Bias estimation, gravity removal, vibration filtering."""

    def __init__(self, sample_rate=10.0):
        self.fs = sample_rate
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)

    def estimate_bias(self, accel, gyro, static_threshold=0.5):
        """Estimate sensor bias from stationary segments."""
        speed = np.sqrt(np.sum(accel**2, axis=1))
        static_mask = speed < static_threshold
        if static_mask.sum() > 10:
            self.accel_bias = np.mean(accel[static_mask], axis=0) - np.array([0, 0, 9.81])
            self.gyro_bias = np.mean(gyro[static_mask], axis=0)
        accel_corrected = accel - self.accel_bias
        accel_corrected[:, 2] -= 9.81
        gyro_corrected = gyro - self.gyro_bias
        return accel_corrected, gyro_corrected

    def filter_vibration(self, signal_data, cutoff=20.0, order=4):
        """Butterworth low-pass filter to remove engine/road vibration."""
        nyq = self.fs / 2.0
        if cutoff >= nyq:
            cutoff = nyq * 0.9
        b, a = butter(order, cutoff / nyq, btype='low')
        return filtfilt(b, a, signal_data, axis=0)

    def detect_potholes(self, accel_z, threshold=3.0, min_gap=5):
        """Detect and flag pothole/bump events."""
        events = np.abs(accel_z) > threshold
        filtered = np.copy(events)
        indices = np.where(events)[0]
        if len(indices) == 0:
            return np.zeros(len(accel_z), dtype=bool)
        for i in range(1, len(indices)):
            if indices[i] - indices[i-1] < min_gap:
                filtered[indices[i]] = False
        return filtered

    def process(self, accel, gyro):
        """Full calibration pipeline."""
        accel_corr, gyro_corr = self.estimate_bias(accel, gyro)
        accel_filtered = np.zeros_like(accel_corr)
        gyro_filtered = np.zeros_like(gyro_corr)
        for i in range(3):
            accel_filtered[:, i] = self.filter_vibration(accel_corr[:, i], cutoff=20.0)
            gyro_filtered[:, i] = self.filter_vibration(gyro_corr[:, i], cutoff=15.0)
        return accel_filtered, gyro_filtered


class AlignmentEngine:
    """Determine phone orientation relative to vehicle."""

    def __init__(self):
        self.rotation_matrix = np.eye(3)

    def estimate_alignment(self, accel, gyro, mag=None, heading_est=None):
        """Estimate pitch, roll from gravity, yaw from magnetometer."""
        n = min(100, len(accel))
        gravity = np.mean(accel[:n], axis=0) + np.array([0, 0, 9.81])
        norm = np.linalg.norm(gravity)
        if norm > 0:
            gravity = gravity / norm

        # Pitch and roll from gravity
        pitch = np.arctan2(-gravity[0], np.sqrt(gravity[1]**2 + gravity[2]**2))
        roll = np.arctan2(gravity[1], gravity[2])

        # Yaw from magnetometer (tilt-compensated compass - reliable in tunnels
        # where satellite positioning fails but the geomagnetic field persists)
        if heading_est is not None:
            yaw = heading_est
        elif mag is not None and len(mag) >= 10:
            mx = np.mean(mag[:n, 0])
            my = np.mean(mag[:n, 1])
            yaw = np.arctan2(my, mx)
        else:
            yaw = 0.0

        # Build rotation matrix from euler angles
        self.rotation_matrix = self._euler_to_rot(yaw, pitch, roll)
        return yaw, pitch, roll

    def _euler_to_rot(self, yaw, pitch, roll):
        """Euler angles to rotation matrix (ZYX convention)."""
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        return np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])

    def rotate_body_to_nav(self, accel_body):
        """Rotate accelerometer from body frame to navigation frame."""
        return accel_body @ self.rotation_matrix.T


class SpeedPredictor:
    """AI speed estimation from IMU window using Ridge regression."""

    def __init__(self, window_size=30):
        self.window_size = window_size
        self.model = Ridge(alpha=1.0)
        self.is_trained = False

    def _extract_features(self, window):
        """Extract features from IMU window."""
        feats = []
        for i in range(window.shape[1]):
            feats.append(np.mean(window[:, i]))
            feats.append(np.std(window[:, i]))
            feats.append(np.max(window[:, i]))
            feats.append(np.min(window[:, i]))
        # Cross-axis features
        for i in range(0, window.shape[1]-1, 2):
            feats.append(np.mean(np.abs(window[:, i] * window[:, i+1])))
        # RMS
        for i in range(window.shape[1]):
            feats.append(np.sqrt(np.mean(window[:, i]**2)))
        return np.array(feats)

    def train(self, imu_windows, speed_targets):
        """Train speed predictor on labeled data."""
        X = []
        y = []
        for w, s in zip(imu_windows, speed_targets):
            if not np.isnan(s) and w.shape[0] == self.window_size:
                X.append(self._extract_features(w))
                y.append(s)
        if len(X) > 10:
            self.model.fit(np.array(X), np.array(y))
            self.is_trained = True
        return self.is_trained

    def predict(self, imu_window):
        """Predict speed from a single IMU window."""
        if not self.is_trained or imu_window.shape[0] != self.window_size:
            return 0.0
        feat = self._extract_features(imu_window).reshape(1, -1)
        return max(0.0, self.model.predict(feat)[0])

    def predict_sequence(self, accel_gyro, dt=0.1):
        """Vectorized speed prediction for full sequence."""
        n = len(accel_gyro)
        ws = self.window_size
        speeds = np.zeros(n)
        if not self.is_trained or n < ws:
            return speeds
        # Batch feature extraction for all windows
        num_w = n - ws
        X = np.zeros((num_w, 6 * 4 + 3 + 6))  # 33 features
        for j in range(num_w):
            w = accel_gyro[j:j+ws]
            idx = 0
            for ch in range(w.shape[1]):
                X[j, idx] = np.mean(w[:, ch]); idx += 1
                X[j, idx] = np.std(w[:, ch]); idx += 1
                X[j, idx] = np.max(w[:, ch]); idx += 1
                X[j, idx] = np.min(w[:, ch]); idx += 1
            for ch in range(0, w.shape[1]-1, 2):
                X[j, idx] = np.mean(np.abs(w[:, ch] * w[:, ch+1])); idx += 1
            for ch in range(w.shape[1]):
                X[j, idx] = np.sqrt(np.mean(w[:, ch]**2)); idx += 1
        preds = self.model.predict(X)
        speeds[ws:] = np.maximum(preds, 0)
        kernel = np.ones(5) / 5
        speeds = np.convolve(speeds, kernel, mode='same')
        return speeds


class DeadReckoner:
    """Inertial dead reckoning: gyro heading bounded by magnetometer,
    velocity magnitude from the AI speed predictor.

    First-principles design for a car:
      1. Heading = integral of gyro yaw rate (short-term accurate).
         Gyro bias/noise causes unbounded heading drift, so the estimate
         is slowly pulled toward a tilt-compensated magnetometer heading
         (complementary filter). The geomagnetic field exists inside
         tunnels, so this aiding survives GNSS-denied zones.
      2. Speed = AI-predicted speed (Ridge regression on IMU windows).
         Consumer-grade accelerometers double-integrate to garbage within
         seconds, so forward speed is taken from the learned model -
         this is exactly what PS-168 demands: AI-guided dead reckoning.
      3. Non-holonomic constraint: a car does not slide sideways, so
         velocity is strictly along the heading vector.
    """

    def __init__(self, dt=0.1, mag_gain=0.02, speed_gain=0.08):
        self.dt = dt
        self.mag_gain = mag_gain      # complementary filter gain for mag heading
        self.speed_gain = speed_gain  # complementary filter gain for AI speed

    @staticmethod
    def _wrap_angle(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def integrate(self, accel_nav, gyro_nav, speed_pred=None, mag_heading=None):
        """Integrate IMU to get position.

        mag_heading: per-sample magnetometer heading (rad) or None.
        speed_pred:  per-sample AI-predicted speed (m/s) or None.
        """
        n = len(accel_nav)
        pos = np.zeros((n, 2))
        vel = np.zeros((n, 2))
        heading = np.zeros(n)
        heading[0] = gyro_nav[0, 2] * self.dt if n > 0 else 0.0

        # Complementary speed estimate: accelerometer integral tracks fast
        # transients (zero lag); the AI model reference removes integration
        # drift. Classic sensor-fusion architecture, applied to speed.
        v_fused = 0.0

        for i in range(1, n):
            # --- Heading: gyro integration ---
            h = heading[i-1] + gyro_nav[i, 2] * self.dt

            # --- Complementary filter: pull toward magnetometer ---
            if mag_heading is not None and not np.isnan(mag_heading[i]):
                err = self._wrap_angle(mag_heading[i] - h)
                h += self.mag_gain * err

            heading[i] = h
            ch, sh = np.cos(h), np.sin(h)

            # --- Forward accel projected onto heading ---
            ax_fwd = accel_nav[i, 0] * ch + accel_nav[i, 1] * sh
            v_fused += ax_fwd * self.dt
            v_fused *= 0.999  # leaky integrator for numerical safety

            # --- Blend in AI speed reference ---
            if speed_pred is not None and speed_pred[i] > 0:
                v_fused += self.speed_gain * (speed_pred[i] - v_fused)

            v = max(0.0, v_fused)

            # NHC: velocity strictly along heading (car cannot slide sideways)
            vel[i, 0] = v * ch
            vel[i, 1] = v * sh
            pos[i] = pos[i-1] + vel[i] * self.dt

        return pos, vel, heading


class EKFusion:
    """5-state EKF with unicycle kinematic model.

    State: [x, y, v, yaw, gyro_bias]
      - v is a scalar along heading, so the non-holonomic constraint
        (no sideways sliding) is inherent in the model, not bolted on.
    Aiding sources (each independently usable in GNSS-denied zones):
      1. GPS position updates (when satellites visible)
      2. Magnetometer heading updates (works in tunnels)
      3. AI-predicted speed updates (Ridge model on IMU)
    Result: yaw and speed stay bounded during outage, so position
    error grows slowly and snaps back instantly at GPS re-acquisition.
    """

    def __init__(self, dt=0.1):
        self.dt = dt
        self.n = 5
        # Process noise
        self.Q = np.diag([0.02, 0.02, 0.25, 0.001, 0.00002])
        # Measurement noise
        self.R_gps = np.eye(2) * 4.0        # GPS sigma ~2m
        self.R_mag = np.array([[0.008]])    # mag heading sigma ~5.1 deg
        self.R_speed = np.array([[1.2]])    # speed sigma ~1.1 m/s
        self.ekf = None  # replaced by manual implementation below
        self.x = np.zeros(5)
        self.P = np.eye(5)

    @staticmethod
    def _wrap_angle(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _init_filter(self):
        self.x = np.zeros(5)
        self.P = np.diag([25.0, 25.0, 9.0, 0.25, 0.01])

    def predict_step(self, accel_body, gyro_body):
        """Time update: propagate state with gyro + current speed."""
        dt = self.dt
        v, yaw, b_g = self.x[2], self.x[3], self.x[4]
        gyro_corr = gyro_body[2] - b_g

        self.x[3] = yaw + gyro_corr * dt
        c, s = np.cos(self.x[3]), np.sin(self.x[3])
        self.x[0] += v * c * dt
        self.x[1] += v * s * dt

        F = np.eye(5)
        F[0, 2] = c * dt
        F[0, 3] = -v * s * dt
        F[1, 2] = s * dt
        F[1, 3] = v * c * dt
        F[3, 4] = -dt
        self.P = F @ self.P @ F.T + self.Q

    def _update(self, z, R, H):
        """Generic scalar/vector update with Joseph form covariance."""
        innov = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innov
        IKH = np.eye(self.n) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

    def update_step(self, gps_pos):
        """GPS position measurement (2D)."""
        H = np.zeros((2, 5))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        self._update(np.asarray(gps_pos[:2]), self.R_gps, H)

    def update_mag(self, mag_heading):
        """Magnetometer heading measurement (1D, angle-aware)."""
        H = np.zeros((1, 5))
        H[0, 3] = 1.0
        z = np.array([self.x[3] + self._wrap_angle(mag_heading - self.x[3])])
        self._update(z, self.R_mag, H)

    def update_speed(self, speed):
        """Speed measurement from AI predictor / GPS (1D)."""
        H = np.zeros((1, 5))
        H[0, 2] = 1.0
        self._update(np.array([speed]), self.R_speed, H)

    def process_sequence(self, accel, gyro, gps_available, gps_positions,
                         speed_pred=None, mag_heading=None):
        """Run EKF through entire sequence with all aiding sources."""
        n = len(accel)
        positions = np.zeros((n, 2))
        velocities = np.zeros((n, 2))

        self._init_filter()

        for i in range(n):
            self.predict_step(accel[i], gyro[i])

            if gps_available[i] and not np.any(np.isnan(gps_positions[i])):
                self.update_step(gps_positions[i])

            if mag_heading is not None and not np.isnan(mag_heading[i]):
                self.update_mag(mag_heading[i])

            if speed_pred is not None and speed_pred[i] > 0:
                self.update_speed(speed_pred[i])

            # Re-apply kinematic consistency after updates
            c, s = np.cos(self.x[3]), np.sin(self.x[3])
            self.x[2] = max(0.0, self.x[2])
            velocities[i, 0] = self.x[2] * c
            velocities[i, 1] = self.x[2] * s
            positions[i] = self.x[:2]

        return positions, velocities


class MapMatcher:
    """Vectorized snap-to-nearest-road-segment."""

    def __init__(self):
        self.seg_starts = None
        self.seg_vecs = None
        self.seg_lens = None

    def set_road_network(self, gps_positions):
        """Build vectorized road segments from GPS trajectory."""
        valid = ~np.any(np.isnan(gps_positions), axis=1)
        pts = gps_positions[valid]
        if len(pts) < 2:
            self.seg_starts = np.empty((0, 2))
            self.seg_vecs = np.empty((0, 2))
            self.seg_lens = np.empty(0)
            return
        self.seg_starts = pts[:-1]
        vecs = pts[1:] - pts[:-1]
        self.seg_vecs = vecs
        self.seg_lens = np.sqrt(np.sum(vecs**2, axis=1))

    def match_sequence(self, positions, headings):
        """Vectorized snap full trajectory to road network."""
        n = len(positions)
        matched = np.copy(positions)
        matched_h = np.copy(headings) if headings is not None else None

        if self.seg_starts is None or len(self.seg_starts) == 0:
            return matched, matched_h

        for i in range(n):
            pos = positions[i]
            if np.any(np.isnan(pos)):
                continue
            # Vectorized distance to all segments
            ap = pos - self.seg_starts
            t_num = np.sum(ap * self.seg_vecs, axis=1)
            t = np.clip(t_num / (self.seg_lens**2 + 1e-10), 0, 1)
            projections = self.seg_starts + t[:, None] * self.seg_vecs
            dists = np.sqrt(np.sum((pos - projections)**2, axis=1))
            best = np.argmin(dists)
            if dists[best] < 20.0:
                matched[i] = projections[best]
                if matched_h is not None:
                    matched_h[i] = np.arctan2(self.seg_vecs[best, 1], self.seg_vecs[best, 0])

        return matched, matched_h
