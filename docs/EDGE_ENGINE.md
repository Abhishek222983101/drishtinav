# Edge engine: running DrishtiNav on any IMU

The PS requires the algorithms to work "with any other external IMU sensors data
(edge deployable software engine)", at ~10 Hz on phones and ~200 Hz with FOG IMUs.
The engine is the `drishtinav` Python package; its only runtime dependency is numpy.

## Install

```bash
pip install -e .            # or: pip install -r requirements.txt
python -m drishtinav --help
```

## Streaming API

```python
from drishtinav import NavEngine, GnssFix, make_config
from drishtinav.models.speednet import SpeedNetRuntime
from drishtinav.roadnet import RoadNetwork

cfg = make_config("fog", imu_rate_hz=200)                       # or "smartphone", "mems_edge"
eng = NavEngine(cfg, speed_model=SpeedNetRuntime() if cfg["use_ai_speed"] else None,
                roads=RoadNetwork.load("data/osm/delhi_central.json"))

for sample in imu_stream():                                     # any rate 10-400 Hz
    fix = GnssFix(lat, lon, speed_mps, course_deg, accuracy_m) if new_fix else None
    st = eng.step(sample.t, sample.accel_xyz, sample.gyro_xyz, fix,
                  gnss_lost=receiver_reports_no_signal)         # optional: instant DR switch
    publish(st.lat, st.lon, st.heading_deg, st.speed, st.mode, st.pos_sigma)
```

Conventions: `accel` is specific force in m/s² **including gravity**, `gyro` in rad/s,
both in the sensor's own frame; the mount rotation is learned online, so the sensor can
be installed at any orientation. GNSS course is degrees clockwise from North.

`NavState` also carries `ai_speed`, `ai_sigma`, `p_stationary`, `stationary`, `bump`,
`matched_e/n` (road-snapped position), `match_conf`, `in_tunnel`, and `latency_us`.

## Command line

```bash
# navigate a CSV recorded from any IMU (+ optional GNSS / reference columns)
python -m drishtinav run --input drive.csv --profile mems_edge --roads data/osm/delhi_central.json

# IO-VNBD phone + vehicle pair, 60 s outages injected, scored against the VBOX
python -m drishtinav run --format iovnbd --input S-S1.csv --vehicle V-S1.csv --gnss reference --outage 60 \
       --roads data/osm/coventry_test.json

# generate a 200 Hz FOG drive through the Pragati Maidan tunnel, then navigate it
python -m drishtinav simulate --profile fog --out fog_tunnel.csv
python -m drishtinav run --input fog_tunnel.csv --profile fog --roads data/osm/delhi_central.json

# per-step latency on this machine
python -m drishtinav throughput --profile fog --seconds 120
```

### Generic CSV format

| column | unit | required |
|---|---|---|
| `t` (or `t_ms`, `time_ns`) | s | yes |
| `ax, ay, az` | m/s² incl. gravity (`--accel-g` for g) | yes |
| `gx, gy, gz` | rad/s (`--gyro-deg` for deg/s) | yes |
| `lat, lon` | deg, blank when no fix | no |
| `speed, course, hacc` | m/s, deg, m | no |
| `truth_lat, truth_lon, truth_speed, truth_course` | reference for scoring | no |

The phone app's recorder writes this format, so drives you record with your own
phone can be replayed, scored and used for re-training.

## Adding a new IMU

1. Pick the closest profile in `drishtinav/config.py` and override the datasheet noise
   figures (`gyro_noise`, `accel_noise`, `gyro_bias_rw`, `accel_bias_rw`) and
   `imu_rate_hz`.
2. If the IMU's accelerometer is good enough to integrate (automotive MEMS, FOG),
   keep `dr_use_accel=True` (default in `mems_edge` / `fog`).
3. SpeedNet is trained on smartphone vibration signatures. For a new sensor, record a
   few hours with a reference speed (OBD-II or a GNSS receiver) and retrain:
   `python training/train_speednet.py` (the pipeline is sensor-agnostic; see
   `training/dataset.py`). Until then the `fog` profile runs INS + NHC + ZUPT + map.

## Deployment targets

| Target | How |
|---|---|
| Linux SBC / vehicle computer | the Python package as-is (numpy only) |
| Android native | `models/speednet.onnx` with ONNX Runtime Mobile; the EKF/HMM are ~1 000 lines to port (the JS port in `web/mobile/engine.js` is a template) |
| Browser / PWA | `web/mobile/engine.js` + `models/speednet.json` (what the phone app uses) |

## Performance (measured)

| Profile | IMU rate | per-step latency (Python, 1 core) | capacity |
|---|---|---|---|
| smartphone, full pipeline (SpeedNet + HMM) | 10 Hz | 0.18 ms p50 · 0.87 ms p99 | 3 540 Hz (354× real time) |
| FOG, INS + map | 200 Hz | 0.13 ms p50 · 0.28 ms p99 | 7 300 Hz (36× real time) |
| JS engine (phone app), full pipeline | 10 Hz | ~1.4 ms (desktop Chrome/Node) | ~700 Hz |
