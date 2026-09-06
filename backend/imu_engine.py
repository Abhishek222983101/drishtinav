"""Core IMU Navigation Engine for DrishtiNav."""
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.linear_model import Ridge
from filterpy.kalman import ExtendedKalmanFilter


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

    def estimate_alignment(self, accel, gyro, heading_est=None):
        """Estimate pitch, roll from gravity, yaw from driving direction."""
        n = min(100, len(accel))
        gravity = np.mean(accel[:n], axis=0) + np.array([0, 0, 9.81])
        norm = np.linalg.norm(gravity)
        if norm > 0:
            gravity = gravity / norm

        # Pitch and roll from gravity
        pitch = np.arctan2(-gravity[0], np.sqrt(gravity[1]**2 + gravity[2]**2))
        roll = np.arctan2(gravity[1], gravity[2])

        # Yaw from driving direction (acceleration PCA)
        if heading_est is not None:
            yaw = heading_est
        else:
            if n < 10:
                yaw = 0.0
            else:
                acc_horiz = accel[:n, :2]
                if np.std(acc_horiz) > 0.01:
                    pca_direction = np.mean(acc_horiz, axis=0)
                    yaw = np.arctan2(pca_direction[1], pca_direction[0])
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
    """Inertial dead reckoning with NHC constraints."""

    def __init__(self, dt=0.1):
        self.dt = dt

    def integrate(self, accel_nav, gyro_nav, speed_pred=None):
        """Integrate IMU to get position."""
        n = len(accel_nav)
        pos = np.zeros((n, 2))
        vel = np.zeros((n, 2))
        heading = np.zeros(n)
        heading[0] = 0.0

        for i in range(1, n):
            # Heading from gyro integration
            heading[i] = heading[i-1] + gyro_nav[i, 2] * self.dt
            ch, sh = np.cos(heading[i]), np.sin(heading[i])

            # Velocity integration with NHC
            vx = vel[i-1, 0] + accel_nav[i, 0] * self.dt
            vy = vel[i-1, 1] + accel_nav[i, 1] * self.dt

            # Non-Holonomic Constraints: zero lateral velocity
            v_forward = vx * ch + vy * sh
            v_lateral = -vx * sh + vy * ch
            v_lateral = 0.0  # NHC
            vx = v_forward * ch - v_lateral * sh
            vy = v_forward * sh + v_lateral * ch

            # Speed magnitude constraint from AI model
            if speed_pred is not None and speed_pred[i] > 0:
                current_speed = np.sqrt(vx**2 + vy**2)
                if current_speed > 0.1:
                    scale = speed_pred[i] / current_speed
                    scale = np.clip(scale, 0.3, 3.0)
                    vx *= scale
                    vy *= scale

            vel[i] = [vx, vy]
            pos[i] = pos[i-1] + vel[i] * self.dt

        return pos, vel, heading


class EKFusion:
    """Extended Kalman Filter for GNSS+INS fusion."""

    def __init__(self, dt=0.1):
        self.dt = dt
        self.ekf = ExtendedKalmanFilter(dim_x=9, dim_z=2)
        self._init_filter()

    def _init_filter(self):
        """Initialize EKF state and matrices."""
        self.ekf.x = np.zeros(9)  # [x,y,vx,vy,yaw,b_gz,b_ax,b_ay,alt]
        self.ekf.P *= 10.0

        # Process noise
        self.ekf.Q[0:2, 0:2] = np.eye(2) * 0.1
        self.ekf.Q[2:4, 2:4] = np.eye(2) * 1.0
        self.ekf.Q[4, 4] = 0.01
        self.ekf.Q[5, 5] = 0.001
        self.ekf.Q[6:8, 6:8] = np.eye(2) * 0.01

        # Measurement noise (GPS)
        self.ekf.R = np.eye(2) * 4.0

    def FJacobian(self, x):
        """State transition Jacobian."""
        F = np.eye(9)
        dt = self.dt
        yaw = x[4]
        F[0, 2] = dt * np.cos(yaw)
        F[0, 3] = -dt * np.sin(yaw)
        F[1, 2] = dt * np.sin(yaw)
        F[1, 3] = dt * np.cos(yaw)
        return F

    def HJacobian(self, x):
        """Measurement Jacobian (GPS observes position)."""
        H = np.zeros((2, 9))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        return H

    def predict_step(self, accel_body, gyro_body):
        """Prediction using IMU."""
        x = self.ekf.x
        dt = self.dt
        yaw = x[4]
        ch, sh = np.cos(yaw), np.sin(yaw)

        # Acceleration in nav frame
        ax = accel_body[0] * ch - accel_body[1] * sh - x[6]
        ay = accel_body[0] * sh + accel_body[1] * ch - x[7]

        # NHC: zero lateral velocity
        v_forward = x[2] * ch + x[3] * sh
        ax_body = ax * ch + ay * sh
        ay_body = -ax * sh + ay * ch
        ay_body = 0.0  # NHC

        self.ekf.F = self.FJacobian(x)
        self.ekf.predict()

        # State correction for NHC
        self.ekf.x[0] += x[2] * dt
        self.ekf.x[1] += x[3] * dt
        self.ekf.x[2] += (ax_body * ch - ay_body * sh) * dt
        self.ekf.x[3] += (ax_body * sh + ay_body * ch) * dt
        self.ekf.x[4] += (gyro_body[2] - x[5]) * dt

    def update_step(self, gps_pos):
        """Update with GPS position measurement."""
        self.ekf.update(gps_pos[:2], HJacobian=self.HJacobian, Hx=lambda x: x[:2])

    def process_sequence(self, accel, gyro, gps_available, gps_positions, speed_pred=None):
        """Run EKF through entire sequence."""
        n = len(accel)
        positions = np.zeros((n, 2))
        velocities = np.zeros((n, 2))

        self._init_filter()

        for i in range(n):
            self.predict_step(accel[i], gyro[i])

            if gps_available[i] and not np.any(np.isnan(gps_positions[i])):
                self.update_step(gps_positions[i])

            if speed_pred is not None and speed_pred[i] > 0:
                ch = np.cos(self.ekf.x[4])
                sh = np.sin(self.ekf.x[4])
                v_fwd = speed_pred[i]
                self.ekf.x[2] = v_fwd * ch
                self.ekf.x[3] = v_fwd * sh

            positions[i] = self.ekf.x[:2]
            velocities[i] = self.ekf.x[2:4]

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
