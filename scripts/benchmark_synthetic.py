"""Tunnel benchmark on the simulated Pragati Maidan drive, over many random seeds.

A single simulated run is one noise realisation (sensor noise, mount angle,
potholes, GNSS multipath); results are reported as the distribution over seeds.

    python scripts/benchmark_synthetic.py --seeds 10
Writes results/synthetic.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drishtinav import evaluation as ev  # noqa: E402
from drishtinav.config import make_config  # noqa: E402
from drishtinav.io.synthetic import build_scenario  # noqa: E402
from drishtinav.models.speednet import SpeedNetRuntime  # noqa: E402

CONFIGS = {
    "smartphone": {"ins": dict(use_ai_speed=False, use_map=False, dr_use_accel=True),
                   "ai": dict(use_ai_speed=True, use_map=False), "full": dict(use_ai_speed=True, use_map=True)},
    "fog": {"ins": dict(use_map=False), "full": dict(use_map=True)},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    model = SpeedNetRuntime()
    out = {}
    for profile, cfgs in CONFIGS.items():
        rows = {k: [] for k in cfgs}
        for seed in range(1, a.seeds + 1):
            d, net = build_scenario("delhi_pragati_tunnel", str(ROOT / "data" / "osm"), profile, seed=seed)
            den = d.gnss_denied
            i0, i1 = int(np.where(den)[0][0]), int(np.where(den)[0][-1])
            dist = d.distance_travelled()
            L = float(dist[i1] - dist[i0])
            for k, over in cfgs.items():
                rec = ev.run(d, make_config(profile, **over), speed_model=model, roads=net, fixes=ev.device_gnss(d))
                en = d.truth_en(rec["frame"])
                err = np.hypot(rec["e"] - en[:, 0], rec["n"] - en[:, 1])
                rows[k].append({"seed": seed, "tunnel_m": L, "exit_err_m": float(err[i1]),
                                "max_err_m": float(err[i0:i1 + 1].max()), "drift_pct": float(100 * err[i1] / L),
                                "us_per_step": float(np.mean(rec["latency_us"]))})
            print(profile, seed, {k: round(v[-1]["drift_pct"], 2) for k, v in rows.items()}, flush=True)
        out[profile] = {}
        for k, r in rows.items():
            dp = np.array([x["drift_pct"] for x in r])
            out[profile][k] = {"drift_pct_mean": float(dp.mean()), "drift_pct_median": float(np.median(dp)),
                               "drift_pct_p90": float(np.percentile(dp, 90)), "drift_pct_max": float(dp.max()),
                               "exit_err_mean_m": float(np.mean([x["exit_err_m"] for x in r])),
                               "pass_rate_10pct": float(np.mean(dp < 10)), "tunnel_m": r[0]["tunnel_m"],
                               "runs": r}
    (ROOT / "results" / "synthetic.json").write_text(json.dumps(out, indent=1))
    for p, cfgs in out.items():
        for k, s in cfgs.items():
            print(f"{p:10s} {k:5s} drift mean {s['drift_pct_mean']:5.2f}%  median {s['drift_pct_median']:5.2f}%  "
                  f"p90 {s['drift_pct_p90']:5.2f}%  max {s['drift_pct_max']:5.2f}%  pass {100 * s['pass_rate_10pct']:.0f}%")


if __name__ == "__main__":
    main()
