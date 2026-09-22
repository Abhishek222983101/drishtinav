<div align="center">

# 🧭 DrishtiNav
### AI-ML Intelligent Dead Reckoning for Seamless GNSS-Denied Navigation

**When GPS dies in a tunnel or urban canyon, DrishtiNav keeps navigating — using only the phone's own accelerometer and gyroscope.**

[![Live Demo](https://img.shields.io/badge/🔴_LIVE_DEMO-drishtinav.vercel.app-2ea44f?style=for-the-badge)](https://drishtinav.vercel.app)
[![Live API](https://img.shields.io/badge/⚙️_LIVE_ENGINE_API-onrender.com-blue?style=for-the-badge)](https://drishtinav-engine.onrender.com/api/health)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-model_training-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Tests](https://img.shields.io/badge/tests-19%2F19_passing-brightgreen)](tests/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-informational)](#-license)

</div>

---

## 📋 At a glance

| | |
|---|---|
| **Problem Statement** | PS 26168 — *"AI-ML based Intelligent Dead Reckoning system for seamless navigation"* |
| **Organisation** | Indian Space Research Organisation (ISRO), Department of Space |
| **Theme** | Smart Vehicles |
| **Category** | Software |
| **Dataset used** | [IO-VNBD](https://github.com/onyekpeu/IO-VNBD) (real smartphone + vehicle CAN driving data) |
| **Team** | Team Paradigm |
| **Live demo** | **https://drishtinav.vercel.app** ← open this first |
| **Live engine API** | https://drishtinav-engine.onrender.com |

---

## 💡 The idea, in 20 seconds

Your phone's GPS navigation freezes or jumps the moment you enter a tunnel, an
underground car park, or a street between tall buildings — because GNSS satellites
need a clear sky view. DrishtiNav fixes this by turning the same phone's
**accelerometer + gyroscope** into a self-contained backup navigator: an AI model
predicts the vehicle's speed (replacing the missing speedometer), a physics-based
filter fuses it with GPS when available, and an offline road map snaps the estimate
back onto real streets — so the "you are here" dot never freezes and never jumps.

## 🎯 The problem we were asked to solve

> Vehicle logistics, ride-hailing, and emergency responders rely on GNSS-based phone
> navigation. When a vehicle enters a tunnel, underpass, multi-level car park, or a
> deep urban canyon, GNSS drops out entirely — apps freeze, jump, or miscalculate
> turns. **Build an AI/ML system that keeps navigating through the blackout using only
> a smartphone's IMU**, snaps back onto real roads, and switches between GPS-aided and
> GPS-denied modes seamlessly — plus an edge-deployable version for external/FOG-grade
> IMUs. — *paraphrased from PS 26168*

---

## ✅ Every PS requirement — solved, and where to check it

| # | PS asked for | We built | Proof |
|---|---|---|---|
| 1 | Train & test on **IO-VNBD**, with position plots | Real loader, fixed 5 dataset bugs (sync offsets up to 26s, clock resets, swapped sensor axes), 330 km of usable data | [`docs/DATASET.md`](docs/DATASET.md) |
| 2 | **AI predicts vehicle speed** from phone IMU only (no OBD/speedometer) | **SpeedNet** — 18.8k-param neural net, trained on real driving, predicts speed *and its own confidence* | [`docs/RESULTS.md` §3](docs/RESULTS.md) |
| 3 | Filter engine vibration, potholes, phone-mount misalignment | Vibration filter + pothole detector + **online mount-alignment engine** that detects a knocked/re-seated phone | [`drishtinav/alignment.py`](drishtinav/alignment.py) |
| 4 | **Non-holonomic constraints** (car can't slide sideways) | Built directly into the motion model, not bolted on | [`drishtinav/fusion.py`](drishtinav/fusion.py) |
| 5 | **AI-based GNSS + INS fusion** to cut drift | Fusion filter's trust in the AI speed estimate *is* the AI's own predicted uncertainty | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| 6 | **Map matching** on an offline map (e.g. OSM/HMM) | Hidden Markov Model over a locally-stored OpenStreetMap road graph — zero internet calls at run time | [`drishtinav/mapmatch.py`](drishtinav/mapmatch.py) |
| 7 | **Seamless handler**, switches within milliseconds | Position output never pauses — GPS is only ever a correction on top of a filter that's always running | Unit-tested, zero position jumps |
| 8 | **Edge-deployable engine** for external IMUs, ~10 Hz phone / ~200 Hz FOG | Standalone Python package + CLI; verified profiles for phone / industrial MEMS / FOG-grade IMUs | [`docs/EDGE_ENGINE.md`](docs/EDGE_ENGINE.md) |
| 9 | Bring trained models + offline maps to the finale | Model weights and map databases are committed in the repo, ready to demo offline | `models/`, `data/osm/` |
| 10 | Drift **< 10% of distance travelled** in blackout | **Median outage passes at 60s/120s/180s on real data; simulated tunnel passes on 9/10 (phone) and 10/10 (FOG) random test runs** | [`docs/RESULTS.md`](docs/RESULTS.md) — full numbers below |

**→ Full line-by-line comparison against the PS text: [`docs/GAP_ANALYSIS.md`](docs/GAP_ANALYSIS.md)**

---

## 🚀 Try it live — 30 seconds, no install

### **[👉 Open drishtinav.vercel.app](https://drishtinav.vercel.app)**

1. Pick a scenario — a simulated drive through Delhi's real **1.3 km Pragati Maidan
   tunnel**, or a real recorded drive from the IO-VNBD dataset that the AI never saw
   during training.
2. Click **RUN** → an animated car drives the route, GPS drops in the shaded zone, and
   you watch three approaches side by side: plain physics (drifts wildly), +AI speed
   (much better), +map matching (best — snaps to the actual road).
3. Click **⚡ Re-run live** to make the *actual Python engine*, running on our
   server, recompute everything from scratch with fresh random sensor noise — proving
   the results aren't precomputed cherry-picks. *(Free-tier server: ~60–90 s of real
   compute, not a cold-start delay.)*

<div align="center">
<img src="results/plots/delhi_tunnel_trajectory.png" width="46%" alt="Trajectory through the Delhi tunnel"> <img src="results/plots/benchmark_drift.png" width="50%" alt="Drift benchmark across outage lengths">
</div>

---

## 📊 Results — the numbers, not just the claim

Every number below is produced by a committed script against committed raw data —
nothing is hand-typed. Full methodology, an honest engineering log (including a
scoring bug we found ourselves and fixed), and every caveat: **[`docs/RESULTS.md`](docs/RESULTS.md)**.

#### Simulated Pragati Maidan tunnel, Delhi — 1.35 km with zero GPS (10 random test runs)

| Sensor | Basic physics only | **DrishtiNav (full pipeline)** |
|---|:---:|:---:|
| 📱 Smartphone IMU @ 10 Hz | 31.1% avg drift, **0/10 pass** | 🟢 **3.1% avg drift, 1.4% typical — 9/10 pass** |
| 🛰️ FOG-grade IMU @ 200 Hz (edge) | 0.84% avg drift | 🟢 **0.39% avg drift, 0.21% typical — 10/10 pass** |

#### Real held-out driving data (IO-VNBD, never seen during training)

| GPS outage length | Basic physics only | **DrishtiNav (full pipeline)** |
|:---:|:---:|:---:|
| 30 s | 34.9% median drift | 🟢 **15.2%** |
| 60 s | 37.9% | 🟢 **10.9%** |
| 120 s | 47.9% | 🟢 **12.5%** |
| 180 s | 55.8% | 🟢 **11.5%** |

*(PS target: < 10% drift. "Median" = the typical outage; a few hard outages pull the
average up — see `docs/RESULTS.md` for the honest full breakdown, including where
we're still above target.)*

---

## 🏗️ How it works

```
Phone/FOG IMU ──► Calibrate & align ──► AI speed prediction ──► Sensor fusion filter ──► Position
 (accel + gyro)     (removes vibration,    (SpeedNet: neural net    (physics filter +        output
                     learns phone mount     trained on real          AI-adaptive trust +      (never
GPS fix (if any) ──► angle automatically)   driving data)            offline road snapping) ──► pauses)
```

One continuous loop runs on *every single sensor reading*, whether or not GPS is
available — GPS is only ever used as a correction, never as a dependency. That's what
makes the mode switch instant instead of a restart. Full technical detail, every
formula and tuned constant: **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)**.

**Nothing here is a mock or a stub:** the AI model is a real PyTorch network trained on
real recorded driving and exported for on-device use; the fusion filter and map
matcher are from-scratch implementations of the published algorithms; the "offline
map" is real OpenStreetMap road data downloaded once and used with zero live network
calls; there's even a hand-verified port of the whole engine to JavaScript so it can
run inside a browser/phone with no server round-trip.

## 🛠️ Tech stack

`Python` `NumPy` `PyTorch → ONNX` `Flask` `Gunicorn` `JavaScript` `Leaflet.js` `OpenStreetMap` · deployed on `Vercel` (site) + `Render` (live engine API)

---

## 💻 Run it yourself

```bash
git clone https://github.com/Abhishek222983101/drishtinav.git
cd drishtinav
pip install -r requirements.txt
python -m drishtinav serve            # dashboard at http://localhost:5000
```
*(Windows: just double-click `run.bat` — it sets up everything automatically.)*

**Reproduce every number in this README from scratch:**
```bash
pip install -r requirements-train.txt
python scripts/download_iovnbd.py                        # fetch real driving data
python training/train_speednet.py --win 200 --epochs 15  # train the AI model (~7 min)
python scripts/benchmark.py                               # benchmark on real held-out drives
python scripts/benchmark_synthetic.py --seeds 10          # benchmark the Delhi tunnel, 10 runs
python -m pytest                                           # 19 automated tests
```

**Use the engine on your own IMU data** (works with any CSV, not just phones):
```bash
curl -X POST --data-binary @your_drive.csv \
  "https://drishtinav-engine.onrender.com/api/navigate?profile=smartphone"
```
Full CLI + CSV format reference: **[`docs/EDGE_ENGINE.md`](docs/EDGE_ENGINE.md)**.

---

## 📁 Project structure

```
drishtinav/       the actual navigation engine (pure Python + NumPy, no dependencies needed on-device)
training/         how SpeedNet (the AI speed model) is trained, PyTorch
models/           trained model weights, ready to use
server/           Flask app powering both the dashboard and the live API
web/dashboard/    the site you see at the live demo link
web/mobile/       a bonus installable phone-app prototype (on-device inference)
data/             offline road maps + sample driving data bundled with the repo
scripts/          one command each for: download data, train, benchmark, deploy
results/          every benchmark number + chart, as raw committed data
docs/             deep-dive documentation (see table below)
tests/            19 automated tests + a Python <-> JavaScript consistency check
```

## 📚 Deep-dive documentation

| Doc | What's in it |
|---|---|
| [**GAP_ANALYSIS.md**](docs/GAP_ANALYSIS.md) | Every PS line item, compared to what we built, in detail |
| [**RESULTS.md**](docs/RESULTS.md) | Every benchmark number, methodology, and an honest log of what we got wrong and fixed |
| [**ARCHITECTURE.md**](docs/ARCHITECTURE.md) | The full technical design — every equation, filter, and tuned constant |
| [**DATASET.md**](docs/DATASET.md) | How the IO-VNBD dataset was cleaned and prepared, and the bugs we found in it |
| [**RESEARCH.md**](docs/RESEARCH.md) | The published papers behind every design decision |
| [**EDGE_ENGINE.md**](docs/EDGE_ENGINE.md) | Using the engine on your own hardware / IMU data |
| [**MOBILE_APP.md**](docs/MOBILE_APP.md) | The bonus phone-app prototype |

---

## 👥 Team Paradigm

Built end-to-end for **ISRO Smart India Hackathon 2026 · PS 26168**.

## 📄 License

Apache 2.0 (see [`LICENSE`](LICENSE)). IO-VNBD dataset © Onyekpe et al. — see
[the dataset's own repo](https://github.com/onyekpeu/IO-VNBD) for its terms; only two
small resynchronised samples are bundled here for the demo. Map data ©
OpenStreetMap contributors (ODbL).
