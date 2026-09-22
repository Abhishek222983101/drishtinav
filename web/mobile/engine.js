/* DrishtiNav on-device engine (JavaScript port of drishtinav/*.py).
 *
 * Runs entirely in the phone's browser: calibration, mount alignment, SpeedNet
 * inference (weights from models/speednet.json), vehicle EKF with NHC / ZUPT /
 * AI pseudo-odometer, GNSS deficit handler and map matching on an offline OSM
 * extract. No network is needed once the app shell, model and roads are cached.
 * Kept deliberately 1:1 with the Python reference so results can be compared
 * (tests/js/parity.mjs).
 */
(function (root) {
  'use strict';
  const G = 9.80665, D2R = Math.PI / 180;
  const wrap = a => { a = (a + Math.PI) % (2 * Math.PI); if (a < 0) a += 2 * Math.PI; return a - Math.PI; };
  const clip = (x, a, b) => Math.min(b, Math.max(a, x));
  const norm3 = v => Math.hypot(v[0], v[1], v[2]);
  const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const unit3 = v => { const n = norm3(v); return n > 1e-9 ? [v[0] / n, v[1] / n, v[2] / n] : v.slice(); };

  const CONFIG = {
    smartphone: {
      imu_rate_hz: 10, model_rate_hz: 10, q_pos: 0.02, gnss_pos_updates_biases: false, gyro_noise: 0.01, accel_noise: 1.0, gyro_bias_rw: 2e-5,
      accel_bias_rw: 5e-3, gnss_pos_sigma_min: 2.5, gnss_speed_sigma: 0.3, gnss_course_sigma_deg: 3,
      gnss_course_min_speed: 4, gnss_course_max_yaw_rate: 0.08, zupt_sigma: 0.03, zaru_sigma: 0.002,
      ai_sigma_scale: 2.0, ai_sigma_floor: 0.35, use_ai_speed: true, dr_use_accel: false, speed_rw_dr: 1.0, speed_rw_unaligned: 1.5,
      ai_online_calibration: true, ai_calib_tau_s: 300, ai_stationary_p: 0.85, use_map: true, map_min_conf: 0.9,
      map_max_pos_sigma: 25, map_cross_sigma_scale: 0.6, map_heading_sigma_deg: 6, map_interval_s: 1,
      map_use_heading: true, gnss_timeout_s: 1.5, gnss_max_accuracy_m: 25, recovery_fixes: 2,
      recovery_sigma_inflate: 3, static_acc_std: 0.15, static_gyro_std: 0.01, static_gyro_mean: 0.03,
      static_horiz_accel: 0.3, bump_thr: 2.5,
    },
  };

  /* ---------------------------------------------------------------- geo */
  class LocalFrame {
    constructor(lat0, lon0) {
      this.lat0 = lat0; this.lon0 = lon0;
      const phi = lat0 * D2R, e2 = 6.69437999014e-3, a = 6378137.0, s2 = Math.sin(phi) ** 2;
      this.rn = a * (1 - e2) / Math.pow(1 - e2 * s2, 1.5);
      this.re = a / Math.sqrt(1 - e2 * s2);
      this.c = Math.cos(phi);
    }
    toEN(lat, lon) { return [(lon - this.lon0) * D2R * this.re * this.c, (lat - this.lat0) * D2R * this.rn]; }
    toLL(e, n) { return [this.lat0 + n / this.rn / D2R, this.lon0 + e / (this.re * this.c) / D2R]; }
  }
  const compassToMath = deg => wrap((90 - deg) * D2R);
  const mathToCompass = rad => ((90 - rad / D2R) % 360 + 360) % 360;

  /* ---------------------------------------------------------- filtering */
  class Biquad {
    constructor(fs, fc, ch) {
      fc = Math.min(fc, 0.45 * fs);
      const k = Math.tan(Math.PI * fc / fs), q = Math.SQRT1_2, n = 1 / (1 + k / q + k * k);
      this.b = [k * k * n, 2 * k * k * n, k * k * n];
      this.a = [1, 2 * (k * k - 1) * n, (1 - k / q + k * k) * n];
      this.z0 = new Float64Array(ch); this.z1 = new Float64Array(ch); this.primed = false;
    }
    step(x) {
      const { a, b } = this, y = new Array(x.length);
      if (!this.primed) { for (let i = 0; i < x.length; i++) { this.z0[i] = (1 - b[0]) * x[i]; this.z1[i] = (b[2] - a[2]) * x[i]; } this.primed = true; }
      for (let i = 0; i < x.length; i++) {
        y[i] = b[0] * x[i] + this.z0[i];
        this.z0[i] = b[1] * x[i] - a[1] * y[i] + this.z1[i];
        this.z1[i] = b[2] * x[i] - a[2] * y[i];
      }
      return y;
    }
  }

  class Window {
    constructor(n) { this.n = n; this.buf = []; this.s = 0; this.s2 = 0; }
    push(x) {
      if (this.buf.length === this.n) { const o = this.buf.shift(); this.s -= o; this.s2 -= o * o; }
      this.buf.push(x); this.s += x; this.s2 += x * x;
    }
    get full() { return this.buf.length === this.n; }
    get mean() { return this.buf.length ? this.s / this.buf.length : 0; }
    get std() { const n = this.buf.length; return n < 2 ? Infinity : Math.sqrt(Math.max(this.s2 / n - (this.s / n) ** 2, 0)); }
  }

  class StaticDetector {
    constructor(fs, c) {
      const n = Math.max(3, Math.round(fs));
      this.acc = new Window(n); this.gyr = new Window(n); this.hor = new Window(n);
      this.c = c; this.minSamples = Math.round(fs); this.count = 0;
    }
    update(accel, gyro, horiz) {
      this.acc.push(norm3(accel)); this.gyr.push(norm3(gyro)); this.hor.push(horiz);
      const c = this.c;
      const quiet = this.acc.full && this.acc.std < c.static_acc_std && this.gyr.std < c.static_gyro_std &&
        this.gyr.mean < c.static_gyro_mean && Math.abs(this.hor.mean) < c.static_horiz_accel;
      this.count = quiet ? this.count + 1 : 0;
      return this.count >= this.minSamples;
    }
  }

  class BumpDetector {
    constructor(fs, thr) { this.thr = thr; this.hold = Math.round(0.6 * fs); this.left = 0; this.events = 0; }
    update(dev) {
      if (Math.abs(dev) > this.thr) { if (this.left === 0) this.events++; this.left = this.hold; }
      else if (this.left > 0) this.left--;
      return this.left > 0;
    }
  }

  /* ---------------------------------------------------------- alignment */
  class MountAlignment {
    constructor(fs) {
      this.fs = fs; this.g = null; this.ref = null; this.m = 0; this.mObs = 0; this.S = [0, 0]; this.Saa = 0; this.Svv = 0;
      this.fwdKnown = false; this.Sgg = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]; this.Sgc = [0, 0, 0]; this.Scc = 0; this.nGc = 0; this.turnEv = 0;
      this.gyroUp = null; this.gyroScale = 1; this.version = 0; this.accH = [0, 0]; this.gyrVec = [0, 0, 0]; this.nInt = 0; this.yawAbs = 0;
      this.gFast = null; this.gRef = null; this.stepCount = 0; this.mountEvents = 0;
    }
    checkMount(accel, gyro, isStatic, dvdt, dev) {    // phone slipped / re-seated in the holder?
      if (!this.gFast) { this.gFast = accel.slice(); this.gRef = accel.slice(); }
      const calm = isStatic || (isFinite(dvdt) && Math.abs(dvdt) < 0.3 && norm3(gyro) < 0.1);
      if (!calm || dev > 0.6) return;
      for (let i = 0; i < 3; i++) { this.gFast[i] += (accel[i] - this.gFast[i]) / this.fs; this.gRef[i] += (accel[i] - this.gRef[i]) / (60 * this.fs); }
      const ang = Math.acos(clip(dot3(unit3(this.gFast), unit3(this.gRef)), -1, 1)) / D2R;
      if (ang > 12) this.stepCount++; else if (ang < 6) this.stepCount = 0;
      if (this.stepCount > 3 * this.fs) {
        this.mountEvents++; this.g = this.gFast.slice(); this.gRef = this.gFast.slice(); this.ref = null;
        this.m = 0; this.mObs = 0; this.S = [0, 0]; this.Saa = 0; this.Svv = 0; this.fwdKnown = false;
        this.Sgg = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]; this.Sgc = [0, 0, 0]; this.Scc = 0; this.nGc = 0; this.turnEv = 0;
        this.gyroUp = null; this.gyroScale = 1; this.version++; this.stepCount = 0;
      }
    }
    get up() { return this.g ? unit3(this.g) : [0, 0, 1]; }
    basis() {
      const u = this.up;
      if (!this.ref) { const a = u.map(Math.abs); const k = a.indexOf(Math.min(...a)); this.ref = [0, 0, 0]; this.ref[k] = 1; }
      const d = dot3(this.ref, u);
      const h1 = unit3([this.ref[0] - d * u[0], this.ref[1] - d * u[1], this.ref[2] - d * u[2]]);
      return [u, h1, cross3(u, h1)];
    }
    rotation() {
      const [u, h1, h2] = this.basis(), c = Math.cos(this.m), s = Math.sin(this.m);
      const f = [c * h1[0] + s * h2[0], c * h1[1] + s * h2[1], c * h1[2] + s * h2[2]];
      return [f, cross3(u, f), u];
    }
    yawRate(gyro) { return this.gyroUp ? dot3(gyro, this.gyroUp) * this.gyroScale : dot3(gyro, this.up); }
    update(accel, gyro, isStatic, dvdt) {
      if (!this.g) this.g = accel.slice();
      const dev = Math.abs(norm3(accel) - G);
      let a = 0;
      if (isStatic && dev < 0.5) a = 1 / (0.5 * this.fs);          // reject shocks even at rest
      else if (dev < 0.3 && norm3(gyro) < 0.05 && Math.abs(dvdt) < 0.15) a = 1 / (20 * this.fs);
      for (let i = 0; i < 3; i++) this.g[i] += a * (accel[i] - this.g[i]);
      this.checkMount(accel, gyro, isStatic, dvdt, dev);
      const [, h1, h2] = this.basis();
      this.accH[0] += dot3(accel, h1); this.accH[1] += dot3(accel, h2);
      for (let i = 0; i < 3; i++) this.gyrVec[i] += gyro[i];
      this.yawAbs += Math.abs(this.yawRate(gyro)); this.nInt++;
    }
    onGnss(dvdt, courseRate, speed) {
      if (!this.nInt) return;
      const n = this.nInt, ah = [this.accH[0] / n, this.accH[1] / n], gm = this.gyrVec.map(v => v / n), yawAbs = this.yawAbs / n;
      this.accH = [0, 0]; this.gyrVec = [0, 0, 0]; this.yawAbs = 0; this.nInt = 0;
      if (speed < 2 || !isFinite(dvdt)) return;
      if (yawAbs < 0.05 && Math.abs(dvdt) > 0.3) {
        this.S[0] += ah[0] * dvdt; this.S[1] += ah[1] * dvdt;
        this.Saa += ah[0] * ah[0] + ah[1] * ah[1]; this.Svv += dvdt * dvdt; this.mObs += Math.abs(dvdt);
        this.m = Math.atan2(this.S[1], this.S[0]);
        if (this.mObs > 8) this.fwdKnown = Math.hypot(this.S[0], this.S[1]) / Math.sqrt(this.Saa * this.Svv + 1e-9) > 0.3;
      }
      if (isFinite(courseRate) && speed > 4 && Math.abs(courseRate) < 1) {
        for (let i = 0; i < 3; i++) { for (let j = 0; j < 3; j++) this.Sgg[i][j] += gm[i] * gm[j]; this.Sgc[i] += gm[i] * courseRate; }
        this.Scc += courseRate * courseRate; this.nGc++;
        this.turnEv += Math.abs(courseRate);
        if (this.turnEv > 3) {
          const lam = 1e-2 * (this.Sgg[0][0] + this.Sgg[1][1] + this.Sgg[2][2]) + 1e-9;   // data-scaled ridge
          const A = this.Sgg.map((r, i) => r.map((v, j) => v + (i === j ? lam : 0)));
          const c = solve3(A, this.Sgc), nrm = norm3(c);
          if (nrm > 0.5) {
            const up = c.map(v => v / nrm), ref = this.gyroUp || this.up;
            if (Math.acos(clip(dot3(up, ref), -1, 1)) / D2R > 20) this.version++;   // only a material axis change resets the bias
            this.gyroUp = up;
            const Su = [0, 1, 2].map(i => dot3(this.Sgg[i], up)), den = dot3(up, Su);   // 1-D LS scale along the axis
            if (den > 0 && this.nGc > 10) {                                              // ... only if significant (> 3 SE)
              const sc = dot3(up, this.Sgc) / den, rss = Math.max(this.Scc - 2 * sc * dot3(up, this.Sgc) + sc * sc * den, 0);
              const se = Math.sqrt(rss / (this.nGc - 1) / den);
              this.gyroScale = Math.abs(sc - 1) > 3 * se ? clip(sc, 0.9, 1.1) : 1;
            }
          }
        }
      }
    }
    toVehicle(accel, gyro) {
      const R = this.rotation();
      return [dot3(R[0], accel), dot3(R[1], accel), dot3(R[2], accel) - G, this.yawRate(gyro)];
    }
    get state() {
      const R = this.rotation(), up = R[2];
      return { roll_deg: Math.atan2(up[1], up[2]) / D2R, pitch_deg: Math.atan2(-up[0], Math.hypot(up[1], up[2])) / D2R,
        yaw_mis_deg: Math.atan2(R[0][1], R[0][0]) / D2R, forward_converged: this.fwdKnown, gyro_axis_calibrated: !!this.gyroUp,
        mount_events: this.mountEvents };
    }
  }

  function solve3(A, b) {  // Cramer's rule for the 3x3 normal equations
    const det = m => m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    const d = det(A);
    return [0, 1, 2].map(k => det(A.map((r, i) => r.map((v, j) => j === k ? b[i] : v))) / d);
  }

  /* ---------------------------------------------------- AI: SpeedNet */
  function turnSpeed(aLat, yaw) { return Math.abs(yaw) > 0.05 ? clip(aLat / yaw, 0, 40) : 0; }

  class FeatureStream {
    constructor(fs) { this.kin = new Biquad(fs, 2, 4); this.acc = new Biquad(fs, 2, 3); this.gyr = new Biquad(fs, 2, 3); this.vib = new Biquad(fs, 0.5, 2); }
    step(aF, aL, aU, yaw, accel, gyro) {
      const k = this.kin.step([aF, aL, aU, yaw]);
      const la = this.acc.step(accel), lg = this.gyr.step(gyro);
      const va = Math.hypot(accel[0] - la[0], accel[1] - la[1], accel[2] - la[2]);
      const vg = Math.hypot(gyro[0] - lg[0], gyro[1] - lg[1], gyro[2] - lg[2]);
      const v = this.vib.step([va, vg]);
      return [k[0], k[1], k[2], k[3], v[0], v[1], turnSpeed(k[1], k[3])];
    }
  }

  class SpeedNet {
    constructor(j) {
      this.win = j.win; this.k = j.kernel; this.dil = j.dilations; this.mu = j.mu; this.sd = j.sd;
      const L = j.layers, F = name => Float32Array.from(L[name].data);
      this.convs = this.dil.map((d, i) => ({ w: F(`convs.${i}.weight`), b: F(`convs.${i}.bias`), shape: L[`convs.${i}.weight`].shape, d }));
      this.fc1w = F('fc1.weight'); this.fc1b = F('fc1.bias'); this.fc1s = L['fc1.weight'].shape;
      this.fc2w = F('fc2.weight'); this.fc2b = F('fc2.bias'); this.fc2s = L['fc2.weight'].shape;
    }
    forward(win) {           // win: array of T feature vectors
      const T = win.length, C0 = win[0].length;
      let x = [];
      for (let c = 0; c < C0; c++) { const row = new Float32Array(T); for (let t = 0; t < T; t++) row[t] = (win[t][c] - this.mu[c]) / this.sd[c]; x.push(row); }
      for (const cv of this.convs) {
        const [O, C, K] = cv.shape, d = cv.d, pad = (K - 1) * d, out = [];
        for (let o = 0; o < O; o++) {
          const r = new Float32Array(T);
          for (let t = 0; t < T; t++) {
            let s = cv.b[o];
            for (let c = 0; c < C; c++) {
              const xc = x[c], wo = (o * C + c) * K;
              for (let j = 0; j < K; j++) { const ti = t - pad + j * d; if (ti >= 0) s += cv.w[wo + j] * xc[ti]; }
            }
            r[t] = s > 0 ? s : 0;
          }
          out.push(r);
        }
        x = out;
      }
      const H = x.length, h = new Float32Array(2 * H);
      for (let c = 0; c < H; c++) { let s = 0; for (let t = 0; t < T; t++) s += x[c][t]; h[c] = s / T; h[H + c] = x[c][T - 1]; }
      const [O1, I1] = this.fc1s, h1 = new Float32Array(O1);
      for (let o = 0; o < O1; o++) { let s = this.fc1b[o]; for (let i = 0; i < I1; i++) s += this.fc1w[o * I1 + i] * h[i]; h1[o] = s > 0 ? s : 0; }
      const [O2, I2] = this.fc2s, o = new Float32Array(O2);
      for (let k = 0; k < O2; k++) { let s = this.fc2b[k]; for (let i = 0; i < I2; i++) s += this.fc2w[k * I2 + i] * h1[i]; o[k] = s; }
      const speed = Math.log1p(Math.exp(-Math.abs(o[0]))) + Math.max(o[0], 0);
      return [speed, Math.exp(0.5 * clip(o[1], -6, 6)), 1 / (1 + Math.exp(-o[2]))];
    }
  }

  /* ------------------------------------------------------------- EKF */
  const N = 6;
  const zeros = (r, c) => Array.from({ length: r }, () => new Float64Array(c));
  function matMul(A, B) { const r = A.length, k = B.length, c = B[0].length, C = zeros(r, c); for (let i = 0; i < r; i++) for (let l = 0; l < k; l++) { const a = A[i][l]; if (a) for (let j = 0; j < c; j++) C[i][j] += a * B[l][j]; } return C; }
  const T_ = A => { const C = zeros(A[0].length, A.length); for (let i = 0; i < A.length; i++) for (let j = 0; j < A[0].length; j++) C[j][i] = A[i][j]; return C; };
  function inv(A) {
    const n = A.length;
    if (n === 1) return [[1 / A[0][0]]];
    if (n === 2) { const d = A[0][0] * A[1][1] - A[0][1] * A[1][0]; return [[A[1][1] / d, -A[0][1] / d], [-A[1][0] / d, A[0][0] / d]]; }
    throw new Error('inv: n>2');
  }

  class VehicleEKF {
    constructor(c) {
      this.c = c; this.x = new Float64Array(N);
      this.P = zeros(N, N); [1e4, 1e4, Math.PI ** 2, 25, 4e-4, 0.09].forEach((v, i) => this.P[i][i] = v);
      this.posInit = false; this.headInit = false; this.stats = { rejected: 0, updates: 0 };
    }
    predict(yaw, aF, dt, scale) {
      const c = this.c, x = this.x, [E, Nn, psi, v, bg, ba] = x, w = yaw - bg, pm = psi + 0.5 * w * dt;
      const cm = Math.cos(pm), sm = Math.sin(pm);
      x[0] = E + v * cm * dt; x[1] = Nn + v * sm * dt; x[2] = wrap(psi + w * dt); x[3] = v + (aF - ba) * dt;
      const F = zeros(N, N); for (let i = 0; i < N; i++) F[i][i] = 1;
      F[0][2] = -v * sm * dt; F[0][3] = cm * dt; F[0][4] = 0.5 * v * sm * dt * dt;
      F[1][2] = v * cm * dt; F[1][3] = sm * dt; F[1][4] = -0.5 * v * cm * dt * dt; F[2][4] = -dt; F[3][5] = -dt;
      const P = matMul(matMul(F, this.P), T_(F));
      const q = [c.q_pos * dt, c.q_pos * dt, c.gyro_noise ** 2 * dt, (c.accel_noise * scale) ** 2 * dt, c.gyro_bias_rw ** 2 * dt, c.accel_bias_rw ** 2 * dt];
      for (let i = 0; i < N; i++) P[i][i] += q[i];
      this.P = P;
    }
    update(z, h, H, R, gate, angleIdx = -1, only = null) {
      const m = z.length, y = z.map((v, i) => v - h[i]);
      if (angleIdx >= 0) y[angleIdx] = wrap(y[angleIdx]);
      const PHt = matMul(this.P, T_(H)), S = matMul(H, PHt);
      for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) S[i][j] += R[i][j];
      const Si = inv(S);
      let nis = 0; for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) nis += y[i] * Si[i][j] * y[j];
      if (gate != null && nis > gate) { this.stats.rejected++; return false; }
      const K = matMul(PHt, Si);
      if (only) for (let i = 0; i < N; i++) if (!only.includes(i)) for (let j = 0; j < m; j++) K[i][j] = 0;
      for (let i = 0; i < N; i++) { let s = 0; for (let j = 0; j < m; j++) s += K[i][j] * y[j]; this.x[i] += s; }
      this.x[2] = wrap(this.x[2]);
      const IKH = zeros(N, N); const KH = matMul(K, H);
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) IKH[i][j] = (i === j ? 1 : 0) - KH[i][j];
      const KRK = matMul(matMul(K, R), T_(K)), P = matMul(matMul(IKH, this.P), T_(IKH));
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) P[i][j] += KRK[i][j];
      this.P = P; this.stats.updates++;
      return true;
    }
    row(idx) { const r = new Float64Array(N); r[idx] = 1; return r; }
    initPosition(en, s) { this.x[0] = en[0]; this.x[1] = en[1]; for (let i = 0; i < N; i++) { this.P[0][i] = this.P[i][0] = this.P[1][i] = this.P[i][1] = 0; } this.P[0][0] = this.P[1][1] = s * s; this.posInit = true; }
    initHeading(psi, v) { this.x[2] = psi; this.x[3] = v; for (let i = 0; i < N; i++) this.P[2][i] = this.P[i][2] = 0; this.P[2][2] = (10 * D2R) ** 2; this.headInit = true; }
    gnssPos(en, s, gate) {   // position fixes (multipath-prone) may not steer the sensor biases
      return this.update([en[0], en[1]], [this.x[0], this.x[1]], [this.row(0), this.row(1)], [[s * s, 0], [0, s * s]], gate ? 13.82 : null, -1,
        this.c.gnss_pos_updates_biases ? null : [0, 1, 2, 3]);
    }
    gnssSpeed(v, s, gate) { return this.update([v], [this.x[3]], [this.row(3)], [[s * s]], gate ? 10.83 : null); }
    heading(psi, s, gate, only = null) { return this.update([psi], [this.x[2]], [this.row(2)], [[s * s]], gate ? 10.83 : null, 0, only); }
    zupt(yaw) { return this.update([0, yaw], [this.x[3], this.x[4]], [this.row(3), this.row(4)], [[this.c.zupt_sigma ** 2, 0], [0, this.c.zaru_sigma ** 2]], null); }
    aiSpeed(v, sigma) { const s = Math.max(this.c.ai_sigma_floor, sigma * this.c.ai_sigma_scale); return this.update([v], [this.x[3]], [this.row(3)], [[s * s]], 10.83 * 4); }
    map(xy, head, cs, hs) {
      const n = [-Math.sin(head), Math.cos(head)], H = new Float64Array(N); H[0] = n[0]; H[1] = n[1];
      const ok = this.update([0], [n[0] * (this.x[0] - xy[0]) + n[1] * (this.x[1] - xy[1])], [H], [[cs * cs]], 10.83, -1, [0, 1, 2]);
      if (hs != null) this.heading(head, hs, true, [0, 1, 2]);
      return ok;
    }
    get posSigma() { return Math.sqrt(Math.max(this.P[0][0] + this.P[1][1], 0) / 2); }
  }

  /* ------------------------------------------------- GNSS deficit handler */
  class GnssHandler {
    constructor(c) { this.c = c; this.mode = 'INIT'; this.lastFix = -1e9; this.recLeft = 0; this.rejects = 0; this.transitions = []; this.period = null; }
    set(t, m) { if (m !== this.mode) { this.transitions.push([t, this.mode, m]); this.mode = m; } }
    get timeout() { return Math.max(this.c.gnss_timeout_s, 1.6 * (this.period == null ? 1 : this.period)); }   // 1 Hz until measured
    tick(t, hasFix, acc, init, lost = false) {
      const c = this.c;
      if (hasFix) { const d = t - this.lastFix; if (d > 0.05 && d < 5) this.period = this.period == null ? d : 0.9 * this.period + 0.1 * d; this.lastFix = t; }
      if (!init) return null;
      if (lost && !hasFix) this.lastFix = Math.min(this.lastFix, t - this.timeout - 1e-6);   // OS says no signal: switch now
      if (!hasFix && t - this.lastFix > this.timeout) { this.set(t, 'DR'); return null; }
      if (!hasFix) return null;
      if (this.mode === 'DR' || this.mode === 'INIT') { this.recLeft = c.recovery_fixes; this.set(t, this.mode === 'DR' ? 'RECOVERY' : 'GNSS'); }
      const poor = acc != null && acc > c.gnss_max_accuracy_m;
      if (this.mode === 'RECOVERY') { this.recLeft--; if (this.recLeft <= 0) this.set(t, poor ? 'DEGRADED' : 'GNSS'); return c.recovery_sigma_inflate; }
      if (poor || this.rejects >= 3) { this.set(t, 'DEGRADED'); return 2; }
      this.set(t, 'GNSS'); return 1;
    }
    get trusted() { return ['GNSS', 'DEGRADED', 'RECOVERY'].includes(this.mode); }
  }

  /* ------------------------------------------------------- map matching */
  class RoadNet {
    constructor(data, frame) {
      this.frame = frame; this.name = data.name;
      const xy = data.nodes.map(([la, lo]) => frame.toEN(la, lo));
      const sa = [], sb = [], one = [], tun = [], hw = [];
      for (const w of data.ways) for (let i = 0; i + 1 < w.nodes.length; i++) {
        let a = w.nodes[i], b = w.nodes[i + 1];
        if (w.oneway === -1) [a, b] = [b, a];
        sa.push(a); sb.push(b); one.push(w.oneway !== 0); tun.push(!!w.tunnel); hw.push(w.hw);
      }
      this.n = sa.length; this.xy = xy; this.sa = sa; this.sb = sb; this.one = one; this.tun = tun;
      this.half = hw.map(h => ({ motorway: 7, trunk: 7, primary: 6, secondary: 5, tertiary: 4.5 })[h.replace('_link', '')] || 3.5);
      this.len = sa.map((a, i) => Math.hypot(xy[sb[i]][0] - xy[a][0], xy[sb[i]][1] - xy[a][1]));
      this.head = sa.map((a, i) => Math.atan2(xy[sb[i]][1] - xy[a][1], xy[sb[i]][0] - xy[a][0]));
      this.cell = 50; this.grid = new Map(); this.byNode = new Map();
      for (let s = 0; s < this.n; s++) {
        const p = xy[sa[s]], q = xy[sb[s]];
        for (let gx = Math.floor(Math.min(p[0], q[0]) / 50); gx <= Math.floor(Math.max(p[0], q[0]) / 50); gx++)
          for (let gy = Math.floor(Math.min(p[1], q[1]) / 50); gy <= Math.floor(Math.max(p[1], q[1]) / 50); gy++) {
            const k = gx + ',' + gy; if (!this.grid.has(k)) this.grid.set(k, []); this.grid.get(k).push(s);
          }
        for (const nd of [sa[s], sb[s]]) { if (!this.byNode.has(nd)) this.byNode.set(nd, []); this.byNode.get(nd).push(s); }
      }
    }
    candidates(p, r) {
      const ids = new Set(), c = this.cell;
      for (let gx = Math.floor((p[0] - r) / c); gx <= Math.floor((p[0] + r) / c); gx++)
        for (let gy = Math.floor((p[1] - r) / c); gy <= Math.floor((p[1] + r) / c); gy++)
          for (const s of this.grid.get(gx + ',' + gy) || []) ids.add(s);
      const out = [];
      for (const s of ids) {
        const a = this.xy[this.sa[s]], b = this.xy[this.sb[s]], dx = b[0] - a[0], dy = b[1] - a[1], L2 = dx * dx + dy * dy || 1e-9;
        const t = clip(((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2, 0, 1), q = [a[0] + t * dx, a[1] + t * dy];
        const d = Math.hypot(q[0] - p[0], q[1] - p[1]);
        if (d <= r) out.push({ s, q, d, t });
      }
      return out.sort((u, v) => u.d - v.d).slice(0, 10);
    }
    neighbours(s) {   // segments sharing a node, 2 hops (cheap connectivity for the HMM transition)
      const out = new Set([s]);
      for (const nd of [this.sa[s], this.sb[s]]) for (const s2 of this.byNode.get(nd) || []) {
        out.add(s2);
        for (const nd2 of [this.sa[s2], this.sb[s2]]) for (const s3 of this.byNode.get(nd2) || []) out.add(s3);
      }
      return out;
    }
  }

  class MapMatcher {       // HMM forward filter; transition = road connectivity (phone-light variant)
    constructor(net) { this.net = net; this.prev = null; this.last = null; this.lastT = -1e9; }
    update(t, p, heading, speed, sigma) {
      if (t - this.lastT < 1) return this.last;
      this.lastT = t;
      const net = this.net, sg = Math.max(4, sigma), moving = speed > 2;
      const cands = net.candidates(p, Math.max(45, 3 * sg));
      if (!cands.length) { this.prev = null; this.last = null; return null; }
      for (const c of cands) {
        let h = net.head[c.s];
        if (moving && !net.one[c.s] && Math.abs(wrap(heading - h)) > Math.PI / 2) h = wrap(h + Math.PI);
        c.h = h;
        let lp = -0.5 * (c.d / sg) ** 2;
        if (moving) lp += -0.5 * (Math.abs(wrap(heading - h)) / (25 * D2R)) ** 2;
        let tr = 0;
        if (this.prev) {
          tr = -Infinity;
          for (const pc of this.prev) {
            const conn = pc.s === c.s ? 0 : (pc.nb.has(c.s) ? -0.5 : -8);
            tr = Math.max(tr, pc.lp + conn);
          }
        }
        c.lp = lp + tr;
      }
      const mx = Math.max(...cands.map(c => c.lp));
      let z = 0; for (const c of cands) { c.post = Math.exp(c.lp - mx); z += c.post; }
      for (const c of cands) { c.post /= z; c.lp = Math.log(Math.max(c.post, 1e-300)); c.nb = net.neighbours(c.s); }
      this.prev = cands;
      const b = cands.reduce((a, c) => c.post > a.post ? c : a);
      this.last = { seg: b.s, xy: b.q, roadHeading: b.h, dist: b.d, confidence: b.post, t: b.t, len: net.len[b.s], tunnel: net.tun[b.s], half: net.half[b.s] };
      return this.last;
    }
  }

  /* ------------------------------------------------------------ engine */
  class NavEngine {
    constructor(opts = {}) {
      this.c = Object.assign({}, CONFIG.smartphone, opts.config || {});
      const c = this.c, fs = c.imu_rate_hz;
      this.fs = fs; this.frame = opts.frame || null; this.roadsData = opts.roads || null; this.roads = null;
      this.ekf = new VehicleEKF(c); this.align = new MountAlignment(fs); this.stat = new StaticDetector(fs, c);
      this.bump = new BumpDetector(fs, c.bump_thr); this.lp = new Biquad(fs, 2, 4); this.handler = new GnssHandler(c);
      this.feat = new FeatureStream(c.model_rate_hz); this.net = opts.speednet ? new SpeedNet(opts.speednet) : null;
      this.win = []; this.count = 0; this.ai = [NaN, NaN, NaN]; this.lastT = null; this.lastMap = -1e9;
      this.lastFix = null; this.gyroVersion = 0; this.dvdt = 0; this.vPrev = 0; this.calNN = 0; this.calG = 0; this.matcher = null; this.match = null;
      if (this.frame && this.roadsData) this.buildRoads();
    }
    buildRoads() { if (this.roadsData && this.c.use_map) { this.roads = new RoadNet(this.roadsData, this.frame); this.matcher = new MapMatcher(this.roads); } }
    get aiScale() { return (!this.c.ai_online_calibration || this.calNN < 200) ? 1 : clip(this.calG / this.calNN, 0.75, 1.35); }
    step(t, accel, gyro, fix, gnssLost = false) {
      const t0 = (typeof performance !== 'undefined' ? performance.now() : Date.now());
      const c = this.c, ekf = this.ekf;
      const dt = this.lastT == null ? 1 / this.fs : clip(t - this.lastT, 1e-4, 0.5);
      this.lastT = t;
      const u = this.align.up, d = dot3(accel, u);
      const horiz = this.align.g ? Math.hypot(accel[0] - d * u[0], accel[1] - d * u[1], accel[2] - d * u[2]) : 0;
      let staticRule = this.stat.update(accel, gyro, horiz);
      if (this.lastFix && this.lastFix[1] != null && t - this.lastFix[0] < 2 && this.lastFix[1] > 1) staticRule = false;
      if (ekf.headInit) { this.dvdt += (1 - Math.exp(-dt)) * ((ekf.x[3] - this.vPrev) / dt - this.dvdt); this.vPrev = ekf.x[3]; }
      this.align.update(accel, gyro, staticRule, ekf.headInit && this.handler.trusted ? this.dvdt : NaN);
      const [aF, aL, aU, yaw] = this.align.toVehicle(accel, gyro);
      const kin = this.lp.step([aF, aL, aU, yaw]);
      const bump = this.bump.update(aU);
      let aiNew = null;
      const f = this.feat.step(aF, aL, aU, yaw, accel, gyro);
      if (this.net) {
        this.win.push(f); if (this.win.length > this.net.win) this.win.shift();
        if (this.win.length === this.net.win && (++this.count) % 5 === 0) { aiNew = this.net.forward(this.win); this.ai = aiNew; }
      }
      const [aiSpeed, aiSigma, pStat] = this.ai;
      if (this.align.version !== this.gyroVersion) {
        this.gyroVersion = this.align.version; ekf.x[4] = 0;
        for (let i = 0; i < N; i++) ekf.P[4][i] = ekf.P[i][4] = 0; ekf.P[4][4] = 1e-4; ekf.P[2][2] += (5 * D2R) ** 2;
      }
      const init = ekf.posInit && ekf.headInit;
      if (init) {
        const useAcc = this.align.fwdKnown && (c.dr_use_accel || this.handler.mode !== 'DR');
        const scale = useAcc ? (bump ? 3 : 1) : (this.handler.mode === 'DR' ? c.speed_rw_dr / c.accel_noise : c.speed_rw_unaligned / c.accel_noise);
        ekf.predict(kin[3], useAcc ? kin[0] : 0, dt, scale);
      }
      const fixOk = !!fix && isFinite(fix.lat) && isFinite(fix.lon);
      const infl = this.handler.tick(t, fixOk, fixOk ? fix.accuracy : null, init, gnssLost);
      const haveAi = isFinite(pStat);
      const aiStatic = haveAi && pStat > c.ai_stationary_p && aiSpeed < 0.5;
      const aiMoving = haveAi && pStat < 0.2 && aiSpeed > 2;
      const stationary = (staticRule && !aiMoving) || (aiStatic && this.handler.mode === 'DR' && this.stat.gyr.std < 2 * c.static_gyro_std);
      if (init && stationary) ekf.zupt(kin[3]);
      if (fixOk) {
        if (!this.frame) { this.frame = new LocalFrame(fix.lat, fix.lon); this.buildRoads(); }
        const en = this.frame.toEN(fix.lat, fix.lon), sig = Math.max(c.gnss_pos_sigma_min, fix.accuracy || 0), spd = fix.speed;
        if (!ekf.posInit) ekf.initPosition(en, sig);
        if (!ekf.headInit && spd != null && spd > c.gnss_course_min_speed && fix.course != null) ekf.initHeading(compassToMath(fix.course), spd);
        if (ekf.posInit && ekf.headInit && infl != null) {
          const gate = this.handler.mode !== 'RECOVERY' && this.handler.rejects < 3;
          const ok = ekf.gnssPos(en, sig * infl, gate);
          this.handler.rejects = ok ? 0 : this.handler.rejects + 1;
          if (spd != null && isFinite(spd)) {
            ekf.gnssSpeed(spd, c.gnss_speed_sigma * infl, gate);
            if (fix.course != null && spd > c.gnss_course_min_speed && !stationary && Math.abs(kin[3] - ekf.x[4]) < c.gnss_course_max_yaw_rate)
              ekf.heading(compassToMath(fix.course), c.gnss_course_sigma_deg * D2R * infl, gate);
          }
        }
        if (spd != null && isFinite(aiSpeed) && spd > 3 && this.handler.mode === 'GNSS') {
          const k = Math.exp(-1 / c.ai_calib_tau_s); this.calNN = this.calNN * k + aiSpeed; this.calG = this.calG * k + spd;
        }
        if (this.lastFix && spd != null) {
          const [tp, sp, cp] = this.lastFix, dtf = t - tp;
          if (dtf > 0.2 && dtf < 3) {
            const cr = (fix.course != null && cp != null && spd > 3) ? wrap(compassToMath(fix.course) - compassToMath(cp)) / dtf : NaN;
            this.align.onGnss((spd - sp) / dtf, cr, spd);
          }
        }
        this.lastFix = [t, spd, fix.course];
      }
      const dr = this.handler.mode === 'DR';
      if (init && aiNew && c.use_ai_speed && !stationary && dr && isFinite(aiSpeed)) { const k = this.aiScale; ekf.aiSpeed(aiSpeed * k, aiSigma * k); }
      if (ekf.x[3] < 0) ekf.x[3] = 0;
      if (this.matcher && init && t - this.lastMap >= c.map_interval_s) {
        this.lastMap = t;
        const m = this.matcher.update(t, [ekf.x[0], ekf.x[1]], ekf.x[2], Math.abs(ekf.x[3]), ekf.posSigma);
        this.match = m;
        if (m && dr && m.confidence >= c.map_min_conf && Math.abs(ekf.x[3]) > 1 && ekf.posSigma <= c.map_max_pos_sigma) {
          const nearJ = Math.min(m.t * m.len, (1 - m.t) * m.len) < 8;
          ekf.map(m.xy, m.roadHeading, Math.max(1, m.half * c.map_cross_sigma_scale), (nearJ || m.len < 25 || !c.map_use_heading) ? null : c.map_heading_sigma_deg * D2R);
        }
      }
      let lat = NaN, lon = NaN, mlat = NaN, mlon = NaN;
      if (this.frame && ekf.posInit) {
        [lat, lon] = this.frame.toLL(ekf.x[0], ekf.x[1]);
        [mlat, mlon] = [lat, lon];
        const m = this.match;
        if (m && m.confidence >= 0.7 && m.dist < 3 * Math.max(m.half, ekf.posSigma)) {
          const n = [-Math.sin(m.roadHeading), Math.cos(m.roadHeading)], off = n[0] * (ekf.x[0] - m.xy[0]) + n[1] * (ekf.x[1] - m.xy[1]);
          [mlat, mlon] = this.frame.toLL(ekf.x[0] - n[0] * off, ekf.x[1] - n[1] * off);
        }
      }
      const t1 = (typeof performance !== 'undefined' ? performance.now() : Date.now());
      return { t, lat, lon, matchedLat: mlat, matchedLon: mlon, e: ekf.x[0], n: ekf.x[1], headingDeg: mathToCompass(ekf.x[2]),
        speed: ekf.x[3], mode: this.handler.mode, posSigma: ekf.posSigma, aiSpeed, aiSigma, pStat, stationary, bump,
        matchConf: this.match ? this.match.confidence : 0, inTunnel: this.match ? this.match.tunnel : false, latencyMs: t1 - t0 };
    }
  }

  const api = { NavEngine, LocalFrame, SpeedNet, CONFIG, compassToMath, mathToCompass };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.DrishtiEngine = api;
})(typeof self !== 'undefined' ? self : this);
