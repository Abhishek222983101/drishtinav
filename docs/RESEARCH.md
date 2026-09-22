# Research basis

Each design decision in DrishtiNav traces back to published work or to a measurement
on IO-VNBD. Sources were found and read with web search (Exa) during development.

## Dataset & benchmark protocol

* **IO-VNBD** — U. Onyekpe, V. Palade, S. Kanarachos, A. Szkolnik, *IO-VNBD: Inertial and
  odometry benchmark dataset for ground vehicle positioning*, Data in Brief 35 (2021).
  <https://github.com/onyekpeu/IO-VNBD>. 98 h / 5 700 km; phone (AndroSensor, 10 Hz) and
  Ford Fiesta CAN + VBOX. Our findings about its alignment are in `docs/DATASET.md`.
* **WhONet** — U. Onyekpe et al., *WhONet: Wheel Odometry Neural Network for vehicular
  localisation in GNSS-deprived environments*, 2021. <https://arxiv.org/abs/2104.02581>.
  Evaluates on IO-VNBD with simulated **30 / 60 / 120 / 180 s GNSS outages** — the same
  protocol we use. Note WhONet uses *wheel-speed* (CAN) inputs, which the PS rules out;
  our input is the phone IMU only, so its numbers are not directly comparable.
* **R-WhONet** — transfer learning to recalibrate the model to a new vehicle
  (Neural Computing & Applications, 2024). Motivates our per-trip **online SpeedNet
  scale calibration** against GNSS.

## AI for dead reckoning

* **AI-IMU Dead-Reckoning** — M. Brossard, A. Barrau, S. Bonnabel, IEEE T-IV 2020,
  <https://arxiv.org/abs/1904.06064>. A CNN adapts the Kalman filter's pseudo-measurement
  noise (NHC) from raw IMU; 1.10 % translational error on KITTI. → Our EKF takes
  SpeedNet's **predicted σ as measurement noise** ("AI-based fusion"), and NHC is built
  into the process model.
* **OdoNet** — H. Tang, X. Niu et al., *Untethered speed aiding for vehicle navigation
  without hardware wheeled odometer*, <https://arxiv.org/abs/2109.03091>. A 1-D CNN
  pseudo-odometer from a single IMU plus a ZUPT detector cut 60 s-outage error by ~68 %
  vs NHC-only (RMS 12.3 m → 4.0 m). They warn that IMU biases and **mounting angles**
  "may notably ruin" the model → our alignment engine, vehicle-frame features, and
  augmentation with random mount rotations.
* **LSTM / RNN pseudo-measurements during outages** — Fang et al., Remote Sensing 2020;
  Dai et al., Defence Technology 2019 (cited by IO-VNBD) — background for learned
  outage bridging.

## Alignment & calibration

* Wahlström, Skog, Händel et al., *IMU Alignment for Smartphone-based Automotive
  Navigation*, FUSION 2015 — levelling from gravity, yaw from acceleration / braking,
  NHC with the mount as a state.
* Chen, Zhang, Niu, *Estimate the pitch and heading mounting angles of the IMU for land
  vehicular GNSS/INS*, IEEE T-ITS 2020; Wang et al., *Deep learning-driven automatic
  estimation of smartphone installation angles*, IEEE/ION PLANS 2023 (< 1° error).
* *Deep Learning for Inertial Sensor Alignment* (2022), <https://arxiv.org/abs/2212.11120>
  — yaw mount angle from IMU only: 8° within 5 s, 4° within 27 s. A possible
  GNSS-free initialiser for our regression approach.

## Map matching

* P. Newson, J. Krumm, *Hidden Markov map matching through noise and sparseness*,
  ACM SIGSPATIAL 2009. Gaussian emission on distance to road, exponential transition on
  |route − great-circle distance|, Viterbi. → `drishtinav/mapmatch.py` (causal forward
  variant with posterior confidence).
* The PS suggests "UKF + HMM map matching"; we use an EKF (the model is only mildly
  non-linear at 10 Hz) and an HMM, and apply map information as geometry-only
  (Schmidt-consider) pseudo-measurements after finding that full updates let wrong
  matches corrupt the speed state (validation: individual outages failed by 100–990 m;
  with geometry-only updates plus confidence ≥ 0.95 and σ ≤ 12 m gating, the map
  configuration no longer underperforms map-free dead reckoning).

## Scenario & platform facts

* **Pragati Maidan Integrated Transit Corridor** tunnel, New Delhi: 1.3 km, six lanes,
  Ring Road ↔ India Gate via Purana Qila Road, opened June 2022 (PIB release; Indian
  Express). Present in OpenStreetMap as `tunnel=yes` ways named "Pragati Maidan Tunnel",
  which the simulator uses to deny GNSS.
* Browser sensors: `DeviceMotionEvent` gives `accelerationIncludingGravity` (m/s²) and
  `rotationRate` (deg/s) and requires a **secure context (HTTPS)**; iOS additionally needs
  `DeviceMotionEvent.requestPermission()` from a user gesture (MDN).
