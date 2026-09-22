# Results

All numbers are reproducible with the scripts named in each section. Figures are in
`results/plots/` (regenerate with `python scripts/make_plots.py`).

## 1. IO-VNBD, held-out test drives (smartphone IMU only)

`python scripts/benchmark.py` → `results/benchmark.{json,md}`

* Test drives **S1 and S4** (35.2 km in segments ≥ 10 min), never used for training,
  hyper-parameter tuning or model selection.
* Input: phone accelerometer + gyroscope (10 Hz). GNSS aiding: VBOX reference degraded
  to 1 Hz smartphone quality (σ ≈ 2.5 m coloured noise); outages of length L injected
  every L + 90 s after a 120 s warm-up. Truth: VBOX position.
* Drift = horizontal error at the end of the outage ÷ distance driven during it.
  *Aggregate* = Σ end errors ÷ Σ distances (robust to a few stationary outages);
  *median* = typical outage; *pass* = share of outages under the PS target of 10 %.

| Outage (mean distance) | Configuration | Drift aggregate | Drift median | Drift p90 | End error median | Pass < 10 % |
|---|---|---|---|---|---|---|
| 30 s (266 m) | INS + NHC + ZUPT (no AI) | 25.7 % | 12.2 % | 52.6 % | 19.0 m | 50 % |
| | + AI speed (SpeedNet) | 12.4 % | 9.4 % | 26.8 % | 17.3 m | 53 % |
| | **+ HMM map matching (full)** | **12.3 %** | **9.4 %** | 26.0 % | 19.7 m | 53 % |
| 60 s (500 m) | INS + NHC + ZUPT (no AI) | 44.2 % | 16.0 % | 88.5 % | 45.7 m | 44 % |
| | + AI speed (SpeedNet) | 14.5 % | 5.7 % | 42.7 % | 30.3 m | 63 % |
| | **+ HMM map matching (full)** | **14.9 %** | **5.6 %** | 46.9 % | 26.8 m | **67 %** |
| 120 s (947 m) | INS + NHC + ZUPT (no AI) | 50.3 % | 15.7 % | 176.5 % | 142.7 m | 50 % |
| | + AI speed (SpeedNet) | 12.1 % | 9.7 % | 29.1 % | 61.3 m | 56 % |
| | **+ HMM map matching (full)** | **12.3 %** | **9.1 %** | 29.9 % | 61.3 m | 56 % |
| 180 s (1 484 m) | INS + NHC + ZUPT (no AI) | 103.1 % | 21.7 % | 81.7 % | 309.2 m | 47 % |
| | + AI speed (SpeedNet) | 12.8 % | 4.8 % | 34.7 % | 92.3 m | 53 % |
| | **+ HMM map matching (full)** | **13.6 %** | **5.0 %** | 37.1 % | 97.9 m | 53 % |

**Reading it.** The AI pseudo-odometer is what makes smartphone dead reckoning work:
it cuts aggregate drift **2× (30 s) to 8× (180 s)** versus the classical INS, and
keeps the median outage **below the 10 % target at every length**. The aggregate is
12–15 %, i.e. still above 10 %: a minority of outages (p90 26–47 %) dominates it.
Map matching adds little on these drives (60 s pass rate 63 → 67 %) because it is
deliberately conservative (see §4).

![benchmark](../results/plots/benchmark_drift.png)
![S1 trajectory](../results/plots/iovnbd_s1_trajectory.png)
![S4 error](../results/plots/iovnbd_s4_error.png)

## 2. Pragati Maidan tunnel, Delhi (simulated on real OSM roads)

`python scripts/benchmark_synthetic.py --seeds 10` → `results/synthetic.json`

4.7 km route from Ring Road through the 1.35 km Pragati Maidan tunnel to India Gate,
~55 km/h in the tunnel; GNSS lost for the whole tunnel and degraded by multipath at
the portals. Each seed is a new realisation of sensor noise, turn-on biases, holder
wobble, potholes and GNSS errors.

| IMU | Configuration | Drift mean | Drift median | p90 | Worst | Seeds < 10 % |
|---|---|---|---|---|---|---|
| Smartphone @10 Hz | INS + NHC + ZUPT | 31.1 % | 26.5 % | 46.8 % | 48.8 % | 0 / 10 |
| | + AI speed | 9.5 % | 9.7 % | 14.0 % | 17.7 % | 5 / 10 |
| | **full** | **8.0 %** | **7.2 %** | 12.1 % | 17.6 % | **8 / 10** |
| FOG @200 Hz (edge) | INS + NHC + ZUPT | 0.84 % | 0.57 % | 1.9 % | 1.9 % | 10 / 10 |
| | **INS + map** | **0.63 %** | **0.42 %** | 1.5 % | 1.7 % | **10 / 10** |

PS example "less than 100 m of drift over 1 km … at 60 km/h in tunnels": the median
smartphone run ends the 1.35 km tunnel ~97 m off (7.2 %), the mean ~108 m; the FOG
edge engine ~8.5 m (0.63 %).

![Delhi trajectory](../results/plots/delhi_tunnel_trajectory.png)
![Delhi error](../results/plots/delhi_tunnel_error.png)

## 3. SpeedNet (AI speed from phone IMU only)

`python training/train_speednet.py --win 200 --epochs 15` → `models/speednet_metrics.json`

| Split | RMSE | MAE | Bias | σ-calibration (std of z, ideal 1) | within 2σ | Stationary accuracy |
|---|---|---|---|---|---|---|
| train (~250 km) | 2.48 m/s | 1.76 m/s | −0.13 | 1.00 | 94.6 % | 98.0 % |
| validation | 3.37 m/s | 2.53 m/s | +0.40 | 1.50 | 84.2 % | 97.5 % |
| **test** | **2.54 m/s** (9.1 km/h) | 1.88 m/s | +0.37 | 1.17 | 92.3 % | 94.2 % |

18 787 parameters, 20 s receptive window, ~7 min training on a laptop CPU. The
uncertainty head is close to calibrated on unseen drives, which is what lets the EKF
use it as measurement noise.

![SpeedNet](../results/plots/speednet_speed.png)

## 4. What was learned building it (engineering log)

| Finding | Effect | Fix |
|---|---|---|
| IO-VNBD phone/vehicle files are offset by up to 26 s and the offset jumps | labels and scoring would be garbage | chunked cross-correlation resync (`io/iovnbd.py`) |
| Phone gyro columns permuted vs accelerometer | heading from the wrong axis | GNSS-course regression finds the yaw axis & sign |
| Phone holder wobble: r(a_fwd, dv/dt) ≈ 0.45 | accel-integrated speed diverges (INS row above) | SpeedNet pseudo-odometer; accel off in DR for phones |
| Static detector fired at 24 % false-positive rate | ZUPT forced speed to 0 while moving, corrupted gyro bias | thresholds fitted on training data (6 % FP) + AI / GNSS vetoes |
| Gyro-bias estimate absorbed turn-lag of GNSS course | heading error 27° median at the end of 60 s outages | course updates on straights only, slower bias RW → 3.5° |
| Gyro-axis re-fits reset the learnt bias | 35° heading drift in the tunnel on some seeds | reset only on material (> 20°) axis changes |
| SpeedNet errors are correlated over 10–20 s | EKF over-trusted 2 Hz updates | R inflated ×4.5 (tuned on validation) |
| Map updates leaked into the speed state | wrong junction choices, 100–990 m failures | geometry-only (Schmidt-consider) map updates + confidence/σ gating |
| GNSS timeout shorter than the fix interval (FOG profile) | mode oscillated every second | timeout adapts to the observed fix rate; explicit `gnss_lost` input |
| Speed random walk scaled by accel noise before alignment (FOG: 0.08 m/s²) | FOG filter could not follow the launch, 37 m errors in the first minute | vehicle-dynamics random walk (1.5 m/s²) until the forward axis is learnt → FOG tunnel drift 2.3 % → 0.63 % |

## 5. Timing

| | per step | capacity | requirement |
|---|---|---|---|
| Smartphone profile, full pipeline, Python | 0.18 ms p50 (0.87 ms p99) | 3 540 steps/s | 10 Hz |
| FOG profile, Python | 0.13 ms p50 (0.28 ms p99) | 7 300 steps/s | 200 Hz |
| Phone engine (JavaScript), full pipeline | ~1.4 ms | ~700 steps/s | 10 Hz |

GNSS → DR transition: same epoch when the receiver reports loss (5 ms at 200 Hz,
100 ms at 10 Hz); otherwise after 1.6 × the observed fix interval. The position
output is continuous across the switch (unit-tested).

## 6. Limitations and what we will do next

* **Aggregate drift is 12–15 %, above the 10 % target**, driven by a minority of
  outages with large speed errors. The phones in IO-VNBD rock in their holders, which
  starves the accelerometer of signal. Next: (a) collect drives on Indian roads with
  the app's recorder and a rigid mount, (b) fine-tune SpeedNet per phone/vehicle with
  the online GNSS calibration data, (c) an outage-conditioned recurrent model that
  also sees the speed at outage start.
* **Two-wheelers** (explicitly mentioned in the PS) have very different dynamics
  (lean, no NHC in the same form); SpeedNet and the NHC assume a car. Needs its own data.
* **Magnetometer not fused** (see `docs/DATASET.md`: 6–15° median error even after
  calibration inside a car).
* The simulated tunnel uses our physics model of phone errors, not a real recording;
  its SpeedNet inputs are therefore out-of-distribution, which is part of the
  seed-to-seed spread.
* GNSS aiding in the IO-VNBD benchmark is simulated from the reference receiver (the
  phone's own GNSS in the dataset is too stale to use).
