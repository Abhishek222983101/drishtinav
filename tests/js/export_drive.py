"""Export a scenario in the /api/drive format for the JS parity harness (and print the Python result)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from drishtinav import evaluation as ev  # noqa: E402
from drishtinav.config import make_config  # noqa: E402
from drishtinav.io.synthetic import build_scenario  # noqa: E402
from drishtinav.models.speednet import SpeedNetRuntime  # noqa: E402

d, net = build_scenario("delhi_pragati_tunnel", str(ROOT / "data" / "osm"), "smartphone")
fixes = ev.device_gnss(d)
rec = ev.run(d, make_config(), speed_model=SpeedNetRuntime(), roads=net, fixes=fixes)
en = d.truth_en(rec["frame"])
err = np.hypot(rec["e"] - en[:, 0], rec["n"] - en[:, 1])
i1 = np.where(d.gnss_denied)[0][-1]
print(json.dumps({"python_median_err_m": float(np.median(err)), "python_denied_exit_err_m": float(err[i1])}), file=sys.stderr)
json.dump({"t": (d.t - d.t[0]).round(4).tolist(), "accel": d.accel.round(5).tolist(), "gyro": d.gyro.round(6).tolist(),
           "gnss": {int(i): [f.lat, f.lon, f.speed, None if f.course is None or not np.isfinite(f.course) else f.course, f.accuracy]
                    for i, f in fixes.items()},
           "denied": d.gnss_denied.astype(int).tolist(), "truth": np.column_stack([d.truth_lat, d.truth_lon]).tolist(),
           "roads": "delhi_central.json"}, sys.stdout)
