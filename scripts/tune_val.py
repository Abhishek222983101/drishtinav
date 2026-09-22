"""Hyper-parameter checks on the *validation* drives only (never on test).

    python scripts/tune_val.py key=value [key=value ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "training"))
from dataset import segments_for  # noqa: E402
from drishtinav import evaluation as ev  # noqa: E402
from drishtinav.config import make_config  # noqa: E402
from drishtinav.models.speednet import SpeedNetRuntime  # noqa: E402
from drishtinav.roadnet import RoadNetwork  # noqa: E402

ROADS = [RoadNetwork.load(ROOT / "data" / "osm" / f) for f in ("coventry_val.json", "burton_val.json")]


def roads_for(seg):
    for r in ROADS:
        lat, lon = r.frame.to_ll(r.node_xy[:, 0], r.node_xy[:, 1])
        if lat.min() < np.nanmin(seg.truth_lat) and np.nanmax(seg.truth_lat) < lat.max():
            return r
    return None


def parse(argv):
    cfg = {}
    for a in argv:
        k, v = a.split("=")
        cfg[k] = {"True": True, "False": False}.get(v, None)
        if cfg[k] is None:
            cfg[k] = float(v)
    return cfg


def main():
    cfg = parse(sys.argv[1:])
    segs = [x for x in segments_for(["S3a", "S3b", "Vta2"], Path.home() / "iovnbd" / "raw") if len(x) >= 4500]
    model = SpeedNetRuntime()
    out = {}
    for L in (30, 60, 120):
        rows = []
        for k, seg in enumerate(segs):
            for w0 in (100, 160):
                outs = ev.outage_schedule(seg, L, warmup_s=w0, gap_s=60)
                den = np.zeros(len(seg), bool)
                for i0, i1 in outs:
                    den[i0:i1] = True
                rec = ev.run(seg, make_config(**cfg), speed_model=model, roads=roads_for(seg),
                             fixes=ev.reference_gnss(seg, seed=k), denied=den)
                rows += ev.score(seg, rec, outs)[0]
        s = ev.summarise(rows)
        out[L] = s
        print(f"L={L:3d} n={s['n_scored']:3d} median {s['drift_pct_median']:5.1f}%  agg {s['drift_pct_aggregate']:5.1f}%  "
              f"pass {100 * s['pass_rate_10pct']:4.0f}%  end-err med {s['end_err_median_m']:5.1f} m", flush=True)
    print("MEAN median %.2f  agg %.2f  pass %.1f" % tuple(np.mean([[v["drift_pct_median"], v["drift_pct_aggregate"],
                                                                     100 * v["pass_rate_10pct"]] for v in out.values()], 0)))


if __name__ == "__main__":
    main()
