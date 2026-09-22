"""Reproduce the GNSS-outage benchmark on held-out IO-VNBD drives.

    python scripts/benchmark.py --data ~/iovnbd/raw            # full table
    python scripts/benchmark.py --lengths 60 --configs full    # quick check

Ablations (each adds one component to the previous):
    ins_nhc   : EKF INS mechanisation (accelerometer-integrated speed) with built-in NHC + ZUPT, no AI, no map
    ai_speed  : + SpeedNet pseudo-odometer with AI-adapted noise
    full      : + HMM map matching constraints (offline OSM)
Writes results/benchmark.json and results/benchmark.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drishtinav import evaluation as ev  # noqa: E402
from drishtinav.config import make_config  # noqa: E402
from drishtinav.io.iovnbd import find_drives, load_drive, resync_segments  # noqa: E402
from drishtinav.models.speednet import SpeedNetRuntime  # noqa: E402
from drishtinav.roadnet import RoadNetwork  # noqa: E402

TEST_DRIVES = ("S1", "S4")
CONFIGS = {
    "ins_nhc": dict(use_ai_speed=False, use_map=False, dr_use_accel=True),
    "ai_speed": dict(use_ai_speed=True, use_map=False),
    "full": dict(use_ai_speed=True, use_map=True),
}
LABELS = {"ins_nhc": "INS + NHC + ZUPT (no AI)", "ai_speed": "+ AI speed (SpeedNet)", "full": "+ HMM map matching (full)"}


def test_segments(root, min_len_s=600):
    segs = []
    for name, s, v in find_drives(root):
        if name in TEST_DRIVES and v:
            segs += [x for x in resync_segments(load_drive(s, v, name)) if len(x) >= min_len_s * 10]
    return segs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / "iovnbd" / "raw"))
    ap.add_argument("--lengths", nargs="+", type=int, default=[30, 60, 120, 180])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--roads", default=str(ROOT / "data" / "osm" / "coventry_test.json"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args(argv)

    segs = test_segments(a.data)
    model = SpeedNetRuntime()
    roads = RoadNetwork.load(a.roads)
    print(f"{len(segs)} test segments, {sum(x.distance_travelled()[-1] for x in segs) / 1000:.1f} km", flush=True)

    results, per_outage = {}, {}
    for L in a.lengths:
        for cname in a.configs:
            rows, wall, steps = [], 0.0, 0
            for k, seg in enumerate(segs):
                outs = ev.outage_schedule(seg, L, warmup_s=120, gap_s=90)
                den = np.zeros(len(seg), bool)
                for i0, i1 in outs:
                    den[i0:i1] = True
                fixes = ev.reference_gnss(seg, seed=k)
                rec = ev.run(seg, make_config(**CONFIGS[cname]), speed_model=model, roads=roads,
                             fixes=fixes, denied=den)
                r, _ = ev.score(seg, rec, outs)
                for row in r:
                    row["segment"] = seg.name
                rows += r
                wall += rec["wall_s"]
                steps += len(seg)
            summ = ev.summarise(rows)
            summ["us_per_step"] = wall / steps * 1e6
            results.setdefault(str(L), {})[cname] = summ
            per_outage.setdefault(str(L), {})[cname] = rows
            print(f"L={L:3d}s {cname:9s} n={summ['n_outages']:3d} dist={summ['mean_distance_m']:6.0f} m  "
                  f"drift median {summ.get('drift_pct_median', float('nan')):5.1f}%  agg {summ.get('drift_pct_aggregate', float('nan')):5.1f}%  "
                  f"pass<10% {100 * summ.get('pass_rate_10pct', float('nan')):5.1f}%  end err med {summ['end_err_median_m']:6.1f} m  "
                  f"{summ['us_per_step']:.0f} us/step", flush=True)

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    meta = {"generated": time.strftime("%Y-%m-%d %H:%M"), "test_drives": TEST_DRIVES,
            "segments": [s.name for s in segs],
            "km": round(sum(x.distance_travelled()[-1] for x in segs) / 1000, 1),
            "gnss": "VBOX reference degraded to 1 Hz smartphone quality (sigma 2.5 m, Gauss-Markov)",
            "protocol": "outage every (L + 90 s) after 120 s warm-up; drift = end error / distance driven in outage"}
    (out / "benchmark.json").write_text(json.dumps({"meta": meta, "summary": results, "outages": per_outage}, indent=1))

    lines = ["| Outage | Config | # outages | mean dist (m) | end err median (m) | drift median | drift aggregate | pass <10 % |",
             "|---|---|---|---|---|---|---|---|"]
    for L, cfgs in results.items():
        for c, s in cfgs.items():
            lines.append(f"| {L} s | {LABELS.get(c, c)} | {s['n_outages']} | {s['mean_distance_m']:.0f} | "
                         f"{s['end_err_median_m']:.1f} | {s.get('drift_pct_median', float('nan')):.1f} % | "
                         f"{s.get('drift_pct_aggregate', float('nan')):.1f} % | {100 * s.get('pass_rate_10pct', float('nan')):.0f} % |")
    (out / "benchmark.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
