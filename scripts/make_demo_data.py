"""Bundle a few held-out IO-VNBD test segments (resynchronised) with the repo.

The full dataset is ~300 MB of Git-LFS files; the dashboard and phone app only
need a couple of representative drives, stored as generic CSV in data/demo/.

    python scripts/download_iovnbd.py        # fetch the drives we use (once)
    python scripts/make_demo_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drishtinav.io.generic import save_generic_csv  # noqa: E402
from drishtinav.io.iovnbd import find_drives, load_drive, resync_segments  # noqa: E402
from drishtinav.roadnet import RoadNetwork  # noqa: E402

DATA = Path.home() / "iovnbd" / "raw"
PICK = ["S1", "S4"]            # test drives only (never used for training)


def main():
    roads = RoadNetwork.load(ROOT / "data" / "osm" / "coventry_test.json")
    s, w, n, e = json.loads((ROOT / "data" / "osm" / "coventry_test.json").read_text())["bbox"]
    out = ROOT / "data" / "demo"
    out.mkdir(parents=True, exist_ok=True)
    meta = []
    for name, sp, vp in find_drives(sys.argv[1] if len(sys.argv) > 1 else DATA):
        if name not in PICK or not vp:
            continue
        segs = [x for x in resync_segments(load_drive(sp, vp, name))
                if s < x.truth_lat.min() and x.truth_lat.max() < n and w < x.truth_lon.min() and x.truth_lon.max() < e]
        seg = max(segs, key=len)
        f = f"iovnbd_{name.lower()}.csv"
        save_generic_csv(seg, out / f)
        km = seg.distance_travelled()[-1] / 1000
        meta.append({"file": f, "title": f"IO-VNBD {name} (Coventry, held-out test drive) - {km:.1f} km, {len(seg) / 600:.0f} min",
                     "roads": "coventry_test.json", "source_segment": seg.name, "resync_lag": seg.meta["resync_lag"]})
        print("wrote", f, len(seg), "samples", f"{km:.1f} km")
    (out / "demo_drives.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
