# IO-VNBD: how we use it and what we found

The PS asks us to train and test on **IO-VNBD** (Onyekpe et al., *Data in Brief* 35, 2021,
<https://github.com/onyekpeu/IO-VNBD>). This page documents exactly how the data is
processed, because several properties of the published files have to be handled
before any model can be trained on them.

## What is in the dataset

| Prefix | Source | Rate | Content |
|---|---|---|---|
| `S-*.csv` | Android phone in a dashboard holder (AndroSensor app) | 10 Hz | GPS (lat, lon, speed, course, accuracy), accelerometer, gravity, gyroscope, magnetometer, orientation — 24 columns |
| `V-*.csv` | Ford Fiesta CAN bus + Racelogic VBOX GPS | 10 Hz | VBOX lat/lon/speed/heading, wheel speeds, *indicated speed*, yaw rate, steering, pedals… — 29 columns |

The "Synchronised V and S datasets" folder pairs phone and vehicle logs recorded on
the same drive. We use the **phone file as the only input** (that is the PS setting)
and the vehicle file **only as ground truth** (CAN wheel speed for SpeedNet labels,
VBOX position/heading for scoring).

Download the drives we use (~280 MB, Git LFS):

```bash
python scripts/download_iovnbd.py            # -> ~/iovnbd/raw
```

## Five things we had to fix before the data was usable

1. **The "synchronised" pairs are not synchronised.** Cross-correlating the phone's
   gyro yaw axis with the CAN yaw-rate shows offsets from −87 to +265 samples
   (up to 26 s), and the offset *jumps* inside a drive wherever a logger dropped rows
   (e.g. drive M: −9 → −19 → −32 samples; the second half of S4 cannot be aligned at
   all). → `drishtinav/io/iovnbd.py: resync_segments()` estimates the lag per
   150 s chunk (FFT cross-correlation), keeps only chunks whose correlation proves
   alignment, and splits drives into clean segments. Where CAN yaw is missing, the
   VBOX course rate is used as the reference.
2. **The phone clock resets mid-file.** AndroSensor's "time since start" restarts
   when a recording session restarts (drives M, S2, S4…). → rebuilt a continuous
   clock from per-row increments.
3. **Gyro columns are permuted relative to the accelerometer.** The accelerometer's
   up axis is *z*, but vehicle yaw appears on gyro column 1 ("Pitch"), r = 0.93–0.97
   with CAN yaw. An engine that assumes a shared frame would integrate the wrong
   axis. → the alignment engine regresses GNSS course-rate on the 3-axis gyro to
   find the yaw axis *and its sign* from data (works for any phone/logger).
4. **The phone's GNSS is stale.** Fresh fixes arrive only every ~9 s and lag the
   VBOX by seconds: median phone-vs-VBOX position error is 25–40 m. It can serve
   neither as aiding nor as truth. → for aiding we down-sample the VBOX fix to
   **1 Hz and degrade it to smartphone quality** (first-order Gauss-Markov + white
   noise, σ ≈ 2.5 m; speed noise 0.15 m/s; course noise 1.5°). This is the standard
   IO-VNBD protocol (GNSS from the reference receiver with simulated outages).
5. **The phones wobble in their holders.** The phone's longitudinal specific force
   correlates only r ≈ 0.45 with the true dv/dt even after perfect alignment and
   low-pass filtering, and gyro axis 0 swings ±0.1 rad/s with no relation to
   vehicle motion. The holder is rocking and leaking gravity into the
   accelerometer. → accelerometer integration is treated as a weak input for this
   phone class (`dr_use_accel=False`, `accel_noise=1.0`), and forward speed during
   outages comes from the learned SpeedNet (OdoNet-style pseudo-odometer).

Also examined and **rejected**: the phone magnetometer. Raw tilt-compensated heading
is 20–55° off (median); even after a hard/soft-iron fit against GNSS course on the
first half of a drive, the second half is 6–15° off (median) with p90 up to 79°.
That is worse than the calibrated gyro over a 3-minute outage, so the magnetometer
is logged but not fused.

## Quality control and splits

| Drive(s) | Driver | Usable after resync | Used for |
|---|---|---|---|
| M, S2, S3c, Vfa02, Vta1a, Vtb5, Vtb1 | B, A, A, E, E, E, E | ~250 km | **train** |
| S3a, S3b, Vta2 | A, A, E | ~30 km | **validation** (model + hyper-parameter selection) |
| S1, S4 | A | 54 km (35 km in ≥10 min segments) | **test** — never trained on, never used for tuning or model selection |
| Vfa01, Vta10, Vw1/5/10, Y1 | E, E, … | 0 km | excluded: failed resync QC (no consistent lag) |

The split is **by drive**: no segment of a test drive is ever seen during training,
and all hyper-parameters (noise inflation, map gating, speed random walk) and the
SpeedNet variant were chosen on the validation drives (`scripts/tune_val.py`). In the
interest of full disclosure: early in development the test drives were also used to
*diagnose engine faults* (e.g. the gyro-bias bug in docs/RESULTS.md), not to tune. Driver E's drives use a different, noisier phone (gyro–yaw correlation ≈ 0.5),
accepted through a relaxed QC rule (a lag that recurs across many chunks), which adds
phone/driver diversity to training.

## Reproduce

```bash
python scripts/download_iovnbd.py
python training/train_speednet.py --win 200 --epochs 15      # ~7 min on a laptop CPU
python scripts/benchmark.py                                  # test-drive outage benchmark
python scripts/make_demo_data.py                             # refresh data/demo/ (bundled test segments)
```
