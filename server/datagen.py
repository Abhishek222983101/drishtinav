"""Data products shared by the live server and the static site build.

Every JSON the front-ends read is produced here, so the Flask server (local /
Render) and ``scripts/build_static.py`` (Vercel) serve byte-identical content:

    scenarios.json            list of drives + the keys of their precomputed files
    runs/<run_key>.json       engine output for one scenario (all ablations)
    drives/<drive_key>.json   raw sensor stream for on-device replay in the phone app
    benchmark.json            IO-VNBD + tunnel benchmark summaries + SpeedNet metrics
"""
from __future__ import annotations

import json
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

from drishtinav import evaluation as ev  # noqa: E402
from drishtinav.config import make_config  # noqa: E402
from drishtinav.io.generic import load_generic_csv  # noqa: E402
from drishtinav.io.synthetic import SCENARIOS, build_scenario  # noqa: E402
from drishtinav.models.speednet import SpeedNetRuntime  # noqa: E402
from drishtinav.roadnet import RoadNetwork  # noqa: E402

OSM = ROOT / "data" / "osm"
DEMO = ROOT / "data" / "demo"
OUTAGES = (30, 60, 120, 180)
REPLAY_OUTAGE = 60
ABLATIONS = {
    "ins": ("INS + NHC + ZUPT", dict(use_ai_speed=False, use_map=False, dr_use_accel=True)),
    "ai": ("+ AI speed (SpeedNet)", dict(use_ai_speed=True, use_map=False)),
    "full": ("+ HMM map matching", dict(use_ai_speed=True, use_map=True)),
}
_engine_lock = threading.Lock()


def run_key(sid: str, outage) -> str:
    base = sid.replace(":", "__").replace(".csv", "")
    return f"{base}__{'tunnel' if sid.startswith('sim:') else int(outage)}"


def drive_key(sid: str) -> str:
    return sid.replace(":", "__").replace(".csv", "")


def scenarios() -> list[dict]:
    out = []
    for key, sc in SCENARIOS.items():
        for profile, label in (("smartphone", "smartphone IMU @10 Hz"), ("fog", "FOG IMU @200 Hz (edge)")):
            sid = f"sim:{key}:{profile}"
            out.append({"id": sid, "title": f"{sc['title']} - {label}", "kind": "synthetic", "outage": "tunnel",
                        "profile": profile, "runs": {"tunnel": run_key(sid, 0)},
                        "drive": drive_key(sid) if profile == "smartphone" else None})
    meta_path = DEMO / "demo_drives.json"
    if meta_path.exists():
        for d in json.loads(meta_path.read_text()):
            sid = f"iovnbd:{d['file']}"
            out.append({"id": sid, "title": d["title"], "kind": "iovnbd", "outage": "injected", "profile": "smartphone",
                        "runs": {str(L): run_key(sid, L) for L in OUTAGES}, "drive": drive_key(sid)})
    return out


def scenario_ids() -> set:
    return {s["id"] for s in scenarios()}


@lru_cache(maxsize=16)
def load_scenario(sid: str, seed: int = 7):
    kind, rest = sid.split(":", 1)
    if kind == "sim":
        key, profile = rest.split(":")
        drive, net = build_scenario(key, str(OSM), profile, seed=seed)
        return drive, net, profile
    if kind == "iovnbd":
        meta = {d["file"]: d for d in json.loads((DEMO / "demo_drives.json").read_text())}[rest]
        drive = load_generic_csv(DEMO / rest, name=meta["title"])
        net = RoadNetwork.load(OSM / meta["roads"]) if meta.get("roads") else None
        return drive, net, "smartphone"
    raise KeyError(sid)


@lru_cache(maxsize=1)
def speednet():
    return SpeedNetRuntime()


def clean(obj):
    """JSON-safe copy: numpy scalars -> python, NaN/inf -> None."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _r(x, nd=3):
    x = np.asarray(x, float)
    return [None if not np.isfinite(v) else round(float(v), nd) for v in x]


def _outages_for(drive, sid, outage):
    """Half-open [i0, i1) GNSS outages: the scenario's own (tunnel) or injected ones."""
    denied = drive.gnss_denied.copy() if drive.gnss_denied is not None else np.zeros(len(drive), bool)
    if denied.any():
        idx = np.where(denied)[0]
        br = np.where(np.diff(idx) > 1)[0]
        outs = list(zip(np.r_[idx[0], idx[br + 1]].tolist(), (np.r_[idx[br], idx[-1]] + 1).tolist()))
        return denied, outs
    outs = ev.outage_schedule(drive, outage, warmup_s=120, gap_s=90) if outage else []
    for i0, i1 in outs:
        denied[i0:i1] = True
    return denied, outs


def run_result(sid: str, outage: int = 60, seed: int = 7, configs=("ins", "ai", "full")) -> dict:
    """Run the Python engine on a scenario for the requested ablations."""
    drive, net, profile = load_scenario(sid, seed if sid.startswith("sim:") else 7)
    denied, outs = _outages_for(drive, sid, outage)
    fixes = ev.device_gnss(drive) if sid.startswith("sim:") else ev.reference_gnss(drive, seed=seed)
    frame = drive.frame()
    step = max(1, len(drive) // 1500)
    idx = np.arange(0, len(drive), step)
    res = {
        "scenario": sid, "title": drive.meta.get("title", drive.name), "profile": profile, "seed": seed,
        "outage_s": None if sid.startswith("sim:") else int(outage),
        "rate_hz": round(drive.rate_hz, 1), "duration_s": float(drive.t[-1] - drive.t[0]),
        "distance_m": float(drive.distance_travelled()[-1]),
        "t": _r(drive.t[idx] - drive.t[0], 2),
        "truth": {"lat": _r(drive.truth_lat[idx], 7), "lon": _r(drive.truth_lon[idx], 7),
                  "speed": _r(drive.truth_speed[idx], 2)},
        "denied": denied[idx].astype(int).tolist(),
        "gnss": [[round(f.lat, 7), round(f.lon, 7)] for i, f in sorted(fixes.items()) if not denied[i]][::max(1, step // 10)],
        "outages": [{"start_s": float(drive.t[a] - drive.t[0]), "end_s": float(drive.t[b - 1] - drive.t[0])} for a, b in outs],
        "configs": {},
    }
    for key in configs:
        label, over = ABLATIONS[key]
        if profile == "fog":
            if key == "ai":
                continue           # SpeedNet is a smartphone model: the FOG profile runs INS + map
            over = dict(over, use_ai_speed=False)
        cfg = make_config(profile, **over)
        with _engine_lock:
            rec = ev.run(drive, cfg, speed_model=speednet() if cfg["use_ai_speed"] else None, roads=net,
                         fixes=fixes, denied=denied)
        rows, err = ev.score(drive, rec, outs)
        lat, lon = frame.to_ll(rec["e"], rec["n"])
        ok = np.isfinite(rec["matched_e"])
        mlat, mlon = frame.to_ll(np.where(ok, rec["matched_e"], rec["e"]), np.where(ok, rec["matched_n"], rec["n"]))
        res["configs"][key] = {
            "label": label, "lat": _r(lat[idx], 7), "lon": _r(lon[idx], 7),
            "matched_lat": _r(mlat[idx], 7), "matched_lon": _r(mlon[idx], 7),
            "err": _r(err[idx], 2), "speed": _r(rec["speed"][idx], 2), "ai_speed": _r(rec["ai_speed"][idx], 2),
            "heading": _r(rec["heading_deg"][idx], 1), "sigma": _r(rec["pos_sigma"][idx], 2),
            "mode": [str(m) for m in rec["mode"][idx]], "stationary": rec["stationary"][idx].astype(int).tolist(),
            "match_conf": _r(rec["match_conf"][idx], 2),
            "outages": rows, "summary": ev.summarise(rows, min_dist=50.0),
            "us_per_step": float(np.nanmean(rec["latency_us"])),
            "diagnostics": clean(rec["diagnostics"]),
        }
    return clean(res)


def drive_stream(sid: str) -> dict:
    """Raw 10 Hz sensors + GNSS for replay through the phone's on-device engine."""
    drive, net, profile = load_scenario(sid)
    if profile != "smartphone":
        raise KeyError("replay is only offered for smartphone-rate drives")
    denied, _ = _outages_for(drive, sid, REPLAY_OUTAGE)
    fixes = ev.device_gnss(drive) if sid.startswith("sim:") else ev.reference_gnss(drive)
    gnss = {int(i): [round(f.lat, 7), round(f.lon, 7), round(f.speed or 0.0, 2),
                     None if f.course is None or not np.isfinite(f.course) else round(f.course, 1),
                     round(f.accuracy, 1)] for i, f in fixes.items() if not denied[i]}
    return {"title": drive.meta.get("title", drive.name), "rate_hz": drive.rate_hz,
            "outage_s": None if sid.startswith("sim:") else REPLAY_OUTAGE,
            "t": np.round(drive.t - drive.t[0], 3).tolist(), "accel": np.round(drive.accel, 4).tolist(),
            "gyro": np.round(drive.gyro, 5).tolist(), "gnss": gnss, "denied": denied.astype(int).tolist(),
            "truth": [[round(float(a), 7), round(float(b), 7)] for a, b in zip(drive.truth_lat, drive.truth_lon)],
            "roads": (net.name + ".json") if net is not None else None}


def benchmark() -> dict:
    out = {}
    p = ROOT / "results" / "benchmark.json"
    if p.exists():
        data = json.loads(p.read_text())
        data.pop("outages", None)
        out.update(data)
    m = ROOT / "models" / "speednet_metrics.json"
    if m.exists():
        out["speednet"] = json.loads(m.read_text())
    syn = ROOT / "results" / "synthetic.json"
    if syn.exists():
        out["synthetic"] = {p: {k: {kk: vv for kk, vv in v.items() if kk != "runs"} for k, v in cfgs.items()}
                            for p, cfgs in json.loads(syn.read_text()).items()}
    return out


def navigate_csv(text: str, profile: str = "smartphone", max_samples: int = 30000) -> dict:
    """Run the engine on an uploaded generic-format CSV (see docs/EDGE_ENGINE.md)."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
        fh.write(text)
        path = fh.name
    drive = load_generic_csv(path, name="upload")
    Path(path).unlink(missing_ok=True)
    if len(drive) > max_samples:
        raise ValueError(f"too many samples ({len(drive)} > {max_samples}); send a shorter drive")
    has_fix = drive.gnss_lat is not None and np.isfinite(drive.gnss_lat).any()
    if not has_fix and drive.truth_lat is None:
        raise ValueError("the CSV needs at least one GNSS fix (lat, lon) to anchor the solution")
    cfg = make_config(profile, imu_rate_hz=round(drive.rate_hz))
    fixes = ev.device_gnss(drive) if drive.gnss_lat is not None else {}
    if not fixes and drive.truth_lat is not None:
        fixes = ev.reference_gnss(drive)
    with _engine_lock:
        rec = ev.run(drive, cfg, speed_model=speednet() if cfg["use_ai_speed"] else None, fixes=fixes)
    lat, lon = rec["frame"].to_ll(rec["e"], rec["n"])
    step = max(1, len(drive) // 2000)
    out = {"samples": len(drive), "rate_hz": round(drive.rate_hz, 2), "profile": profile,
           "us_per_step": float(np.nanmean(rec["latency_us"])),
           "alignment": clean(rec["diagnostics"]["alignment"]),
           "mode_transitions": len(rec["diagnostics"]["transitions"]),
           "track": {"t": _r(drive.t[::step], 2), "lat": _r(lat[::step], 7), "lon": _r(lon[::step], 7),
                     "speed": _r(rec["speed"][::step], 2), "mode": [str(m) for m in rec["mode"][::step]]}}
    if drive.truth_lat is not None:
        en = drive.truth_en(rec["frame"])
        out["position_error_median_m"] = float(np.nanmedian(np.hypot(rec["e"] - en[:, 0], rec["n"] - en[:, 1])))
    return clean(out)
