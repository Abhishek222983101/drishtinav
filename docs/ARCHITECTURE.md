# DrishtiNav architecture

One navigation engine, three front-ends:

```
                 ┌──────────────────────────── drishtinav/ (Python, numpy only) ────────────────────────────┐
 phone / FOG IMU │  calibration.py   alignment.py    features.py +       fusion.py        mapmatch.py         │
 accel, gyro ───►│  vibration LPF    levelling,      models/speednet.py  VehicleEKF       HMM forward filter  │──► pose @ IMU rate
 (10–400 Hz)     │  ZUPT detector    yaw-mount,      SpeedNet TCN:       [E,N,ψ,v,bg,ba]  on offline OSM      │    (lat, lon, ψ, v,
 GNSS fix ──────►│  bump detector    gyro axis,      v, σ, p(stationary) NHC built in     roadnet.py          │     mode, σ, road)
 (1 Hz, optional)│                   mount-slip      @10 Hz              gnss_handler.py  (Dijkstra, grid)    │
                 └──────────────────────────────────────────────────────────────────────────────────────────────┘
        ▲                 same algorithms, 1:1 port                              ▲
        │        web/mobile/engine.js  (runs in the phone's browser)             │ server/app.py (Flask)
   training/ (PyTorch) ──► models/speednet.{npz,json,onnx}               web/dashboard (evaluation UI)
```

* **Edge engine** — `drishtinav/` Python package, streaming `NavEngine.step()`; numpy is
  the only runtime dependency (SciPy/PyTorch are not needed on the device).
* **Phone app** — `web/mobile/`, an installable PWA; `engine.js` is a line-for-line port
  of the Python engine and runs SpeedNet from `models/speednet.json` on the phone.
  `tests/js/parity.mjs` checks the two implementations agree (tunnel exit error
  44.8 m vs 45.0 m on the same drive).
* **Dashboard** — `web/dashboard/` + `server/app.py`: runs the Python engine on any
  scenario with three ablations and visualises everything.

## Per-sample pipeline (`drishtinav/engine.py`)

| # | Stage | What happens | Why |
|---|---|---|---|
| 1 | Static detector | 1 s windows of ‖a‖ std, ‖ω‖ std/mean and levelled horizontal specific force; vetoed if GNSS says we move or SpeedNet is confident we move | ZUPT/ZARU at traffic lights; an idling engine shakes the phone, so no single cue is enough. Thresholds fitted on training drives (78 % recall, 6 % false alarms) |
| 2 | Mount alignment | gravity → pitch/roll; GNSS-dv/dt regression → yaw mount + forward sign; GNSS course-rate regression → gyro yaw axis, sign, scale; mount-slip detector | phone sits at any angle; IO-VNBD even permutes gyro columns |
| 3 | Conditioning | 2nd-order Butterworth LPF (2 Hz phone, 5 Hz edge); bump detector on vertical specific force (> 2.5 m/s² for < 0.6 s) inflates accel noise | engine harmonics / road texture alias into a 10 Hz stream |
| 4 | SpeedNet @10 Hz | 7 vehicle-frame channels × 20 s → speed, σ, p(stationary); faster IMUs are block-averaged to 10 Hz | forward speed without an odometer |
| 5 | EKF predict | every IMU sample, never skipped | seamless output |
| 6 | GNSS deficit handler | INIT / GNSS / DEGRADED / DR / RECOVERY | which aiding sources to trust |
| 7 | EKF updates | GNSS pos/speed/course, ZUPT+ZARU, AI speed, map | fusion |
| 8 | HMM map matching @1 Hz | candidates within max(45 m, 3σ), Viterbi-style forward filter | on-road output + constraints |

## Alignment (`drishtinav/alignment.py`)

* **Levelling.** `g` is updated fast when stationary (τ = 0.5 s) and slowly while
  cruising straight at constant speed (τ = 20 s, only when the filter's dv/dt < 0.15 m/s²,
  ‖ω‖ < 0.05 rad/s and |‖a‖−g| < 0.3). Shocks (|‖a‖−g| > 0.5) are rejected even at rest.
* **Yaw mount.** Over 1 s GNSS intervals on straight road, regress levelled horizontal
  acceleration on the GNSS longitudinal acceleration: `m = atan2(Σ a₂·dv, Σ a₁·dv)`.
  This also resolves the forward/backward ambiguity. Declared converged after 8 m/s of
  accumulated |dv| with correlation > 0.3.
* **Gyro yaw axis.** Ridge regression `c = (Σ g gᵀ + λI)⁻¹ Σ g·ψ̇_GNSS` with
  λ = 0.01·trace (so an IMU that never excites an axis cannot pick up weight from noise);
  axis = c/‖c‖. The scale is a 1-D least-squares fit along the axis and is only applied if
  it differs from 1 by more than 3 standard errors. (1 Hz course noise makes the scale
  weak; applying a 3 % noise-driven scale error costs ~6° per U-turn.)
* **Mount slip.** A 1 s gravity estimate vs a 60 s reference, compared only in calm
  moments; > 12° for 3 s of calm evidence → re-level, re-learn forward and yaw axes,
  restart the gyro bias. 0 false events in synthetic normal driving, 1 event in 166 min
  of real IO-VNBD driving.

## SpeedNet (`training/train_speednet.py`, `drishtinav/models/speednet.py`)

* Inputs per 10 Hz sample (`drishtinav/features.py`): low-passed a_fwd, a_lat, a_up−g,
  yaw rate; vibration energy |a − LPF(a)| and |ω − LPF(ω)|; the centripetal speed cue
  a_lat/ω in turns.
* 4 dilated causal Conv1d layers (k = 5, dilations 1/2/4/8, 32 channels) → [mean-pool ‖
  last step] → 32 → 3. **18 787 parameters**, 20 s window.
* Heads: speed (softplus), log-variance (Gaussian NLL → *calibrated* uncertainty),
  stationary logit (BCE).
* Augmentation: ±4° yaw re-rotation, accel/gyro bias jitter, vibration gain jitter
  (different phones and holders).
* Exports: `.npz` (numpy runtime), `.json` (phone), `.onnx` (Android / ONNX Runtime).

## Fusion (`drishtinav/fusion.py`)

State `x = [E, N, ψ, v, b_g, b_a]`. Mechanisation with the vehicle-frame yaw rate ω and
longitudinal specific force a:

```
ψ⁺ = ψ + (ω − b_g)·dt          v⁺ = v + (a − b_a)·dt
E⁺ = E + v·cos(ψ + ½(ω−b_g)dt)·dt    N⁺ = N + v·sin(…)·dt
```

Velocity exists only along the heading, so the **non-holonomic constraint** (no side
slip, no vertical motion) is part of the model. Measurements, each χ²-gated:

| Update | Model | R | Notes |
|---|---|---|---|
| GNSS position | E, N | max(2.5 m, reported accuracy)² × inflation | may **not** update b_g, b_a (Schmidt "consider"): multipath must not steer sensor biases |
| GNSS speed | v | (0.3 m/s)² | |
| GNSS course | ψ | (3°)² | only when |ω| < 0.08 rad/s (course lags in turns) |
| ZUPT + ZARU | v = 0, b_g = ω | (0.03)², (0.002)² | stationary detector |
| **AI speed** | v = k·v_nn | max(0.35, 4.5·σ_nn)² | σ_nn is SpeedNet's own uncertainty (AI-adaptive noise, cf. AI-IMU DR); ×4.5 accounts for its errors being correlated over 10–20 s; k = online GNSS/SpeedNet scale learned before the outage (per phone, holder, vehicle) |
| **Map** | n·(p − p_road) = 0, ψ = ψ_road | (0.6·half-width)², (6°)² | only if HMM posterior ≥ 0.95 and filter σ ≤ 12 m; updates position and heading **only** |

During DR the phone accelerometer is not integrated (speed is a 0.5 m/s²/√s random
walk corrected by SpeedNet); edge profiles with good accelerometers integrate it.

## GNSS deficit handler (`drishtinav/gnss_handler.py`)

`INIT → GNSS ⇄ DEGRADED`, `→ DR` after 1.6 × the observed fix period without a fix (or
**in the same epoch** when the receiver reports loss: `step(..., gnss_lost=True)`),
`DR → RECOVERY` on the first fix (two fixes with 3× inflated noise and no gating, so a
large DR drift is pulled in smoothly), then `→ GNSS`. Because the INS runs every
sample, the output never pauses; the unit test `test_engine_streaming_is_seamless`
asserts no position jump at the GNSS→DR switch.

## Map matching (`drishtinav/mapmatch.py`, `drishtinav/roadnet.py`)

Newson & Krumm (2009) HMM run causally as a forward filter:

* emission `N(dist; 0, σ)·N(Δheading; 0, 25°)` with one-way roads honoured;
* transition `exp(−|route distance − travelled distance| / β)`, route distances by
  bounded Dijkstra on the OSM graph (`reachable_segments`);
* posterior of the best road = match confidence; back-pointers allow offline Viterbi.

The offline road database is a compact JSON built once from OpenStreetMap
(`scripts/fetch_osm.py`; Delhi central 0.7 MB, Coventry test area 1.2 MB). The phone app
uses the same format and can fetch a 5 × 5 km extract around the user once and cache it.

## Configuration profiles (`drishtinav/config.py`)

| | smartphone | mems_edge | fog |
|---|---|---|---|
| IMU rate | 10 Hz (any) | 100 Hz | 200 Hz |
| gyro white noise | 0.01 rad/s | 0.004 | 3e-4 |
| accel noise (after LPF) | 1.0 m/s² | 0.2 | 0.08 |
| integrate accel in DR | no | yes | yes |
| SpeedNet | yes | yes | off (phone-trained; retrain per sensor) |

## Throughput

Python (this laptop, one core): 180 µs median / 870 µs p99 per step for the full
smartphone pipeline incl. SpeedNet and map matching (3 540 steps/s: 354× the 10 Hz
requirement), and 126 µs median per step for the FOG profile (7 300 steps/s: 36× the
200 Hz requirement).
JavaScript on the same machine: ~1.4 ms per step. `python -m drishtinav throughput`
measures it on your hardware.
