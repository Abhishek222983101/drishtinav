# Gap analysis: PS 26168 vs. DrishtiNav

*PS 26168 — "AI-ML based Intelligent Dead Reckoning system for seamless navigation"
(ISRO, Smart Vehicles). v0 = the repository before this rework (`main`, commit
`c55f521`); v1 = this branch.*

Legend: ✅ done and measured · 🟡 done, with a stated limitation · ❌ not done

## Where v0 stood

v0 was a single-page demo on **synthetic data only**. On inspection:

* the "GPS" fed to the filter was the ground truth itself (the ±2 m noise was computed
  but never used), so the EKF numbers were optimistic;
* drift % used only the *last sample* of the run (after GNSS had returned), so every
  PASS badge was green by construction (EKF "0.06 % drift" while its in-tunnel error
  peaked near 45 m);
* "map matching" snapped to the GNSS track itself, which is a straight chord through
  the tunnel, so it could not help (and did not);
* no IO-VNBD data was loaded, no model was trained on real data, and there was no
  mobile app or edge engine;
* the plots were mislabelled (a 10 m line labelled "10 % threshold"; tunnel shading on
  the wrong axis); `__pycache__` and outputs were committed; there was no
  requirements file; the launcher was Windows-only with a hard-coded Python path.

## Requirement-by-requirement

| # | PS requirement | v0 | v1 (this branch) — evidence | Status |
|---|---|---|---|---|
| 1 | **Use IO-VNBD** to train & test; include preliminary models and **position plots** on an IO-VNBD subset | synthetic only | loader + resync for the phone/vehicle pairs, 330 km usable, drive-level train/val/test split (`drishtinav/io/iovnbd.py`, `docs/DATASET.md`); position plots on held-out drives (`results/plots/iovnbd_*`) | ✅ |
| 2 | **AI/ML predicts speed** from phone accelerometer/gyro only (no speedometer / OBD) | Ridge regression on synthetic windows | **SpeedNet**: 18.8k-param temporal CNN, 20 s window, speed + calibrated σ + p(stationary); test RMSE 2.54 m/s; ONNX/JSON/numpy exports (`training/`, `models/`) | ✅ |
| 3 | Detect & filter **engine idling vibration, potholes, bumps** | fixed 3 m/s² threshold on raw z | 2 Hz Butterworth vibration filter; stationary detector fitted on IO-VNBD (6 % false alarms) + SpeedNet stationarity head; bump detector inflates accelerometer noise (`calibration.py`) | ✅ |
| 4 | Detect **accidental phone misalignment on the mount** | — | online mount-slip detector: re-levels and re-learns forward & yaw axes; unit-tested; 1 event in 166 min of real driving (`alignment.py`) | ✅ |
| 5 | **In-vehicle alignment & calibration**: phone pitch, roll, yaw relative to the car, dashboard or holder | pitch/roll from first 100 samples, yaw from magnetometer | levelling (motion-aware), GNSS-regressed yaw mount with sign, gyro yaw-axis/sign/scale regression (handles IO-VNBD's permuted gyro columns), ZUPT/ZARU gyro bias | ✅ |
| 6 | **Map matching** on an offline map (e.g. OSM) using the road layout as a constraint (e.g. UKF + HMM) | nearest point on the GNSS track | offline OSM database builder (`scripts/fetch_osm.py`); Newson–Krumm **HMM** forward filter with Dijkstra route distances (`mapmatch.py`); geometry-only EKF constraints; road-snapped display output | 🟡 improves the tunnel (smartphone mean 9.5 → 8.0 %) but only marginally on IO-VNBD; gated conservatively after validation showed wrong matches can hurt |
| 7 | **NHC** (car does not slide sideways or fly) | velocity along heading | NHC built into the EKF process model; vertical motion excluded | ✅ |
| 8 | **AI-based GNSS+INS fusion** to mitigate drift | EKF with fixed noises | EKF whose pseudo-odometer noise is SpeedNet's own predicted σ (AI-adaptive, cf. AI-IMU DR) + online GNSS-based self-calibration of the network per trip | ✅ |
| 9 | **Seamless GNSS deficit handler**: switch within milliseconds, and back | none (GNSS mask only) | 5-state handler; switch in the same epoch when the receiver reports loss (5 ms @200 Hz), adaptive timeout otherwise; gated recovery with smooth pull-in; no position jump (unit test) | ✅ |
| 10 | **Mobile application** with real-time UI and a **smooth, uninterrupted vehicle icon** | desktop web page | installable PWA (`web/mobile/`): live phone sensors, on-device engine + SpeedNet, outage simulation, replay, recorder, offline cache; icon interpolated at display rate | 🟡 a PWA, not yet a store APK (wrappable with TWA/Capacitor, see `docs/MOBILE_APP.md`) |
| 11 | **On-device inference** (training on desktop, inference on the phone) | — | JS port of the full engine + SpeedNet runs in the phone browser; parity with Python checked (`tests/js/parity.mjs`) | ✅ |
| 12 | **Edge-deployable engine** for **external IMUs** | — | `drishtinav` package (numpy only), streaming API, CLI, generic CSV format, sensor profiles (`smartphone`, `mems_edge`, `fog`) (`docs/EDGE_ENGINE.md`) | ✅ |
| 13 | **10 Hz** on phones, **~200 Hz** with FOG IMU data | — | 3 540 steps/s (Python) / ~700 steps/s (JS) for the phone pipeline; 7 300 steps/s for the 200 Hz FOG profile (`python -m drishtinav throughput`) | ✅ |
| 14 | Inputs incl. **magnetometer/compass** | mag heading used directly | ingested and logged, **deliberately not fused**: in-car magnetometer heading is 6–15° off even after calibration (measured on IO-VNBD), worse than the gyro over an outage | 🟡 documented decision |
| 15 | **Benchmark: DR drift < 10 % of distance** during blackout (e.g. < 100 m over 1 km at 60 km/h) | claimed PASS via a flawed metric | IO-VNBD test drives, phone only: **median outage 5.0–9.4 %** at 30–180 s; aggregate 12.3–14.9 %; 53–67 % of outages pass. Simulated 1.35 km tunnel: smartphone **8.0 % mean** (80 % of seeds pass), FOG edge engine **0.63 % mean** | 🟡 median meets it; aggregate does not yet (see `docs/RESULTS.md` §6) |
| 16 | Lane-level accuracy | — | GNSS-aided ~2.5 m (median); in outages tens of metres (see 15) | ❌ not lane-level during long outages |
| 17 | Two-wheelers, trucks, older cars | — | vehicle-agnostic pipeline, but SpeedNet trained on cars only; NHC assumes a car | 🟡 needs two-wheeler data |
| 18 | Bring trained models + offline maps to the finale | — | `models/speednet.*`, `data/osm/*.json` in the repo; retraining is one command | ✅ |

## Remaining work, in priority order

1. **Data from Indian roads.** Record with the app's recorder (phone rigidly mounted +
   GNSS reference) on the demo route; fine-tune SpeedNet. This attacks the dominant
   error (speed) and the IO-VNBD phone-holder wobble.
2. **Outage-aware speed model.** A recurrent model conditioned on the speed at outage
   start (IO-VNBD literature, e.g. RNN pseudo-measurements) to shrink the p90 tail.
3. **Native Android wrapper** around the PWA (TWA) or a Kotlin app using
   `speednet.onnx`, for background operation and access to raw GNSS status.
4. **Two-wheeler mode**: lean-aware alignment and a separate SpeedNet.
5. **Map matching tuned on Indian road data** (dense junctions, service roads) and a
   lane-level map if lane-level accuracy is required.
