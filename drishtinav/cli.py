"""Command line interface for the edge engine.

    python -m drishtinav run --input drive.csv --profile smartphone --roads data/osm/delhi_central.json
    python -m drishtinav run --input S-S1.csv --vehicle V-S1.csv --format iovnbd --outage 60
    python -m drishtinav simulate --scenario delhi_pragati_tunnel --profile fog --out fog_tunnel.csv
    python -m drishtinav convert --phone S-S1.csv --vehicle V-S1.csv --out s1.csv
    python -m drishtinav throughput --profile fog --seconds 120
    python -m drishtinav serve --port 5000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _load(a):
    if a.format == "iovnbd":
        from .io.iovnbd import load_drive, resync_segments
        d = load_drive(a.input, a.vehicle)
        if a.vehicle and a.resync:
            segs = resync_segments(d)
            d = max(segs, key=len) if segs else d
        return d
    from .io.generic import load_generic_csv
    return load_generic_csv(a.input, accel_in_g=a.accel_g, gyro_in_deg=a.gyro_deg)


def cmd_run(a):
    from . import evaluation as ev
    from .config import make_config
    from .models.speednet import SpeedNetRuntime
    from .roadnet import RoadNetwork

    d = _load(a)
    overrides = {"imu_rate_hz": round(d.rate_hz)} if a.rate is None else {"imu_rate_hz": a.rate}
    if a.no_ai:
        overrides["use_ai_speed"] = False
    if a.no_map:
        overrides["use_map"] = False
    cfg = make_config(a.profile, **overrides)
    model = SpeedNetRuntime() if cfg["use_ai_speed"] else None
    roads = RoadNetwork.load(a.roads) if a.roads else None
    has_truth = d.truth_lat is not None
    fixes = ev.reference_gnss(d) if (a.gnss == "reference" and has_truth) else ev.device_gnss(d)
    denied = d.gnss_denied.copy() if d.gnss_denied is not None else np.zeros(len(d), bool)
    outs = []
    if a.outage:
        outs = ev.outage_schedule(d, a.outage)
        for i0, i1 in outs:
            denied[i0:i1] = True
    elif fixes:
        # no explicit outage mask: score every gap of > 3 s in the recorded fix stream
        idx = np.array(sorted(fixes))
        gaps = np.where(np.diff(d.t[idx]) > 3.0)[0]
        outs = [(int(idx[g]), int(idx[g + 1])) for g in gaps]
    rec = ev.run(d, cfg, speed_model=model, roads=roads, fixes=fixes, denied=denied)
    lat, lon = rec["frame"].to_ll(rec["e"], rec["n"])
    out = Path(a.out or f"{d.name}_nav.csv")
    import pandas as pd
    pd.DataFrame({"t": d.t, "lat": lat, "lon": lon, "heading_deg": rec["heading_deg"], "speed": rec["speed"],
                  "mode": rec["mode"], "pos_sigma": rec["pos_sigma"], "ai_speed": rec["ai_speed"],
                  "stationary": rec["stationary"], "match_conf": rec["match_conf"]}).to_csv(out, index=False, float_format="%.10g")
    summary = {"samples": len(d), "rate_hz": round(d.rate_hz, 2), "profile": a.profile,
               "us_per_step": float(np.nanmean(rec["latency_us"])), "alignment": rec["diagnostics"]["alignment"],
               "mode_transitions": len(rec["diagnostics"]["transitions"]), "output": str(out)}
    if has_truth:
        rows, err = ev.score(d, rec, outs)
        summary["position_error_median_m"] = float(np.nanmedian(err))
        if rows:
            summary["outages"] = ev.summarise(rows)
    print(json.dumps(summary, indent=2, default=str))


def cmd_simulate(a):
    from .io.generic import save_generic_csv
    from .io.synthetic import build_scenario
    d, _ = build_scenario(a.scenario, str(ROOT / "data" / "osm"), a.profile, a.seed)
    save_generic_csv(d, a.out)
    print(f"wrote {a.out}: {len(d)} samples @ {d.rate_hz:.0f} Hz, route {d.meta['route_m']:.0f} m, "
          f"tunnel {d.meta['tunnel_m']:.0f} m")


def cmd_convert(a):
    from .io.generic import save_generic_csv
    from .io.iovnbd import load_drive, resync_segments
    d = load_drive(a.phone, a.vehicle)
    if a.vehicle:
        segs = resync_segments(d)
        for i, s in enumerate(segs):
            p = Path(a.out).with_name(f"{Path(a.out).stem}_{i}.csv")
            save_generic_csv(s, p)
            print(f"wrote {p} ({len(s)} samples, lag {s.meta['resync_lag']})")
    else:
        save_generic_csv(d, a.out)


def cmd_throughput(a):
    """Per-step latency of the full pipeline (proves 10 Hz on phone / 200 Hz edge headroom)."""
    from .config import make_config
    from .engine import NavEngine
    from .io.synthetic import build_scenario
    from .models.speednet import SpeedNetRuntime
    d, net = build_scenario("delhi_pragati_tunnel", str(ROOT / "data" / "osm"), a.profile)
    cfg = make_config(a.profile)
    eng = NavEngine(cfg, speed_model=SpeedNetRuntime() if cfg["use_ai_speed"] else None, roads=net, frame=d.frame())
    n = min(len(d), int(a.seconds * d.rate_hz))
    lat = np.empty(n)
    t0 = time.perf_counter()
    for i in range(n):
        from .engine import GnssFix
        fix = GnssFix(d.gnss_lat[i], d.gnss_lon[i], d.gnss_speed[i], d.gnss_course[i], d.gnss_accuracy[i]) \
            if d.gnss_new[i] else None
        lat[i] = eng.step(d.t[i], d.accel[i], d.gyro[i], fix).latency_us
    wall = time.perf_counter() - t0
    print(json.dumps({"profile": a.profile, "imu_rate_hz": d.rate_hz, "steps": n,
                      "latency_us_p50": float(np.percentile(lat, 50)), "latency_us_p99": float(np.percentile(lat, 99)),
                      "max_sustainable_hz": float(n / wall), "realtime_factor": float(n / wall / d.rate_hz)}, indent=2))


def cmd_serve(a):
    sys.path.insert(0, str(ROOT))
    from server.app import create_app
    ssl = None
    if a.https:
        ssl = "adhoc"   # self-signed; phones need HTTPS for motion sensors (needs `pip install cryptography`)
        print(f"HTTPS on https://<this-laptop-ip>:{a.port}/app/  (accept the certificate warning once)")
    create_app().run(host=a.host, port=a.port, debug=False, threaded=True, ssl_context=ssl)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="drishtinav", description="DrishtiNav intelligent dead-reckoning engine")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="navigate a recorded drive")
    r.add_argument("--input", required=True)
    r.add_argument("--vehicle", help="IO-VNBD V- file (reference)")
    r.add_argument("--format", choices=["generic", "iovnbd"], default="generic")
    r.add_argument("--profile", choices=["smartphone", "mems_edge", "fog"], default="smartphone")
    r.add_argument("--rate", type=float, help="override IMU rate (Hz)")
    r.add_argument("--roads", help="offline OSM road JSON (data/osm/*.json)")
    r.add_argument("--gnss", choices=["device", "reference"], default="device")
    r.add_argument("--outage", type=float, help="inject repeated GNSS outages of this length (s)")
    r.add_argument("--no-ai", action="store_true")
    r.add_argument("--no-map", action="store_true")
    r.add_argument("--no-resync", dest="resync", action="store_false")
    r.add_argument("--accel-g", action="store_true")
    r.add_argument("--gyro-deg", action="store_true")
    r.add_argument("--out")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("simulate", help="generate a synthetic drive on real OSM roads")
    s.add_argument("--scenario", default="delhi_pragati_tunnel")
    s.add_argument("--profile", choices=["smartphone", "fog"], default="smartphone")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--out", default="scenario.csv")
    s.set_defaults(func=cmd_simulate)

    c = sub.add_parser("convert", help="IO-VNBD S-/V- pair -> generic CSV (resynchronised)")
    c.add_argument("--phone", required=True)
    c.add_argument("--vehicle")
    c.add_argument("--out", default="drive.csv")
    c.set_defaults(func=cmd_convert)

    b = sub.add_parser("throughput", help="measure per-step latency")
    b.add_argument("--profile", choices=["smartphone", "fog"], default="fog")
    b.add_argument("--seconds", type=float, default=120)
    b.set_defaults(func=cmd_throughput)

    v = sub.add_parser("serve", help="start the dashboard + mobile app server")
    v.add_argument("--host", default="0.0.0.0")
    v.add_argument("--port", type=int, default=5000)
    v.add_argument("--https", action="store_true", help="self-signed HTTPS so phones allow sensor access")
    v.set_defaults(func=cmd_serve)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
