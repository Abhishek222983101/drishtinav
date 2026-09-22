# Phone app (PWA)

`web/mobile/` is an installable Progressive Web App. **All navigation runs on the
phone**: calibration, alignment, SpeedNet inference, the EKF and map matching are in
`engine.js` (a 1:1 port of the Python engine) and the model is `models/speednet.json`.
The server only hosts files and demo drives.

## Run it on a phone

Phones only expose motion sensors to pages served over **HTTPS**.

```bash
pip install cryptography                      # once, for the self-signed certificate
python -m drishtinav serve --https            # or ./run.sh --https
# on the phone (same Wi-Fi): https://<laptop-ip>:5000/app/  -> accept the certificate once
# "Add to Home screen" installs it as an app
```

Alternatively host `web/` + `models/` + `data/osm/` on any HTTPS static host (GitHub
Pages, Vercel): the live and recording modes need no server at all.

## Modes

| Mode | What it does |
|---|---|
| **Live** | DeviceMotion (accelerometer incl. gravity, gyroscope) averaged into 10 Hz epochs + Geolocation fixes → on-device engine. Screen wake-lock keeps it running. |
| **Simulate GNSS outage** | a toggle that tells the engine "receiver lost signal" (instant switch to dead reckoning), so tunnel behaviour can be demonstrated on any road |
| **Replay demo** | streams a recorded drive (IO-VNBD test drive or the Pragati Maidan tunnel simulation) through the *on-device* engine at 1–10× speed, with the reference path drawn for comparison |
| **Record** | saves the raw 10 Hz sensors + fixes as CSV in the engine's generic format (training data from your own vehicle, as the PS suggests) |

## UI

* Full-screen map (OpenStreetMap tiles, cached as they are viewed) with a vehicle
  icon **interpolated at display rate** between 10 Hz engine outputs, rotated to the
  estimated heading and snapped to the matched road: the icon moves smoothly and
  never freezes when GNSS drops.
* Mode pill (GNSS / DR / RECOVERY / DEGRADED) and an outage banner showing time and
  distance travelled without GNSS.
* HUD: speed, heading, 1σ position uncertainty, GNSS accuracy.
* Diagnostics: SpeedNet speed ± σ, ZUPT state, map-match confidence and tunnel flag,
  learned mount angles and re-seat events, step time.

## Offline

The service worker caches the app shell, the model and the Delhi road database on
first load; map tiles are cached as viewed. In live mode the app loads a bundled road
extract if you are inside one, otherwise downloads roads within ~2.5 km from Overpass
once and keeps them in local storage.

## Sensor notes

* `rotationRate` arrives in deg/s as (alpha, beta, gamma) about (z, x, y); the app
  converts to rad/s in (x, y, z) order. Axis conventions differ slightly between
  Android and iOS — irrelevant here, because the engine learns the mount rotation and
  the gyro yaw axis (with sign) from GNSS online.
* Browser motion events typically arrive at 50–100 Hz; they are block-averaged to the
  10 Hz rate SpeedNet was trained on.

## Towards a native APK

The PWA can be wrapped as an Android app with Trusted Web Activity (Bubblewrap) or
Capacitor without code changes. A native Kotlin app would load `speednet.onnx` via
ONNX Runtime Mobile and port the engine (see `docs/EDGE_ENGINE.md`).
