"""Generate the proposal figures in results/plots/ (runs from the bundled data only).

    python scripts/make_plots.py

* iovnbd_<drive>_trajectory.png  position plot on a held-out IO-VNBD drive (60 s outages)
* iovnbd_<drive>_error.png       horizontal error vs time, outages shaded
* speednet_speed.png             SpeedNet speed vs CAN reference on a held-out drive
* benchmark_drift.png            drift vs outage length, three ablations (results/benchmark.json)
* delhi_tunnel_trajectory.png    Pragati Maidan tunnel scenario on OSM roads
* delhi_tunnel_error.png         error through the tunnel, smartphone vs FOG profile

Colours: categorical slots validated for colour-vision deficiency with the
dataviz validator (blue / orange / aqua, all-pairs CVD dE >= 9.2); the reference
path is neutral ink; GNSS-denied stretches are neutral bands, labelled.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drishtinav import evaluation as ev  # noqa: E402
from drishtinav.config import make_config  # noqa: E402
from drishtinav.io.generic import load_generic_csv  # noqa: E402
from drishtinav.io.synthetic import build_scenario  # noqa: E402
from drishtinav.models.speednet import SpeedNetRuntime  # noqa: E402
from drishtinav.roadnet import RoadNetwork  # noqa: E402

OUT = ROOT / "results" / "plots"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
SLOT = {"ins": "#2a78d6", "ai": "#eb6834", "full": "#1baf7a"}
LABEL = {"ins": "INS + NHC + ZUPT (no AI)", "ai": "+ AI speed (SpeedNet)", "full": "+ HMM map matching (full)"}
CONFIGS = {"ins": dict(use_ai_speed=False, use_map=False, dr_use_accel=True),
           "ai": dict(use_ai_speed=True, use_map=False), "full": dict(use_ai_speed=True, use_map=True)}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "legend.frameon": False, "lines.solid_capstyle": "round",
})


def spans(mask):
    """Contiguous True runs as half-open [start, end) index ranges."""
    idx = np.where(mask)[0]
    if not len(idx):
        return []
    br = np.where(np.diff(idx) > 1)[0]
    return list(zip(np.r_[idx[0], idx[br + 1]], np.r_[idx[br], idx[-1]] + 1))


def draw_roads(ax, net, frame, bbox_en, pad=150):
    net = net.reframe(frame)
    (x0, y0), (x1, y1) = bbox_en[0] - pad, bbox_en[1] + pad
    p, q = net.seg_p, net.seg_p + net.seg_d
    keep = (np.maximum(p[:, 0], q[:, 0]) > x0) & (np.minimum(p[:, 0], q[:, 0]) < x1) & \
           (np.maximum(p[:, 1], q[:, 1]) > y0) & (np.minimum(p[:, 1], q[:, 1]) < y1)
    from matplotlib.collections import LineCollection
    segs = np.stack([p[keep], q[keep]], axis=1)
    ax.add_collection(LineCollection(segs, colors="#d9d8d3", linewidths=0.9, zorder=1))


def run_all(drive, net, outage=None, profile="smartphone", keys=("ins", "ai", "full")):
    denied = drive.gnss_denied.copy() if drive.gnss_denied is not None else np.zeros(len(drive), bool)
    outs = []
    if outage:
        outs = ev.outage_schedule(drive, outage, warmup_s=120, gap_s=90)
        for i0, i1 in outs:
            denied[i0:i1] = True
    else:
        outs = spans(denied)
    fixes = ev.device_gnss(drive) if drive.meta.get("source") == "synthetic" else ev.reference_gnss(drive)
    recs = {}
    model = SpeedNetRuntime()
    for k in keys:
        over = dict(CONFIGS[k])
        if profile == "fog":
            over["use_ai_speed"] = False
        recs[k] = ev.run(drive, make_config(profile, **over), speed_model=model, roads=net, fixes=fixes, denied=denied)
    return recs, denied, outs


def trajectory_plot(drive, net, recs, denied, outs, title, path, keys=("ins", "ai", "full"),
                    ref_label="Reference (VBOX / true path)"):
    frame = recs[keys[0]]["frame"]
    en = drive.truth_en(frame)
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    lo, hi = np.nanmin(en, 0), np.nanmax(en, 0)
    if net is not None:
        draw_roads(ax, net, frame, np.array([lo, hi]))
    for a, b in spans(denied):
        ax.plot(en[a:b, 0], en[a:b, 1], color=INK2, lw=9, alpha=0.18, solid_capstyle="butt", zorder=2)
    ax.plot(en[:, 0], en[:, 1], color=INK, lw=1.6, zorder=3, label=ref_label)
    widths = {"ins": 1.4, "ai": 1.6, "full": 2.2}
    for k in keys:
        r = recs[k]
        e, n = r["e"], r["n"]           # the filter estimate (what the error metrics score)
        ax.plot(e, n, color=SLOT[k], lw=widths[k], zorder=4 + list(keys).index(k), label=LABEL[k])
    ax.plot([], [], color=INK2, lw=9, alpha=0.18, label="GNSS denied (outage)")
    ax.set_aspect("equal")
    ax.set_xlim(lo[0] - 120, hi[0] + 120)
    ax.set_ylim(lo[1] - 120, hi[1] + 120)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(title, loc="left")
    ax.legend(loc="best", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def error_plot(drive, recs, denied, title, path, keys=("ins", "ai", "full"), labels=None):
    labels = labels or LABEL
    t = drive.t - drive.t[0]
    en = drive.truth_en(recs[keys[0]]["frame"])
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for a, b in spans(denied):
        ax.axvspan(t[a], t[b - 1], color=INK2, alpha=0.10, lw=0)
    errs = {k: np.hypot(recs[k]["e"] - en[:, 0], recs[k]["n"] - en[:, 1]) for k in keys}
    scale_keys = [k for k in keys if k != "ins"] or list(keys)
    ymax = max(np.nanmax(errs[k]) for k in scale_keys) * 1.3
    for k in keys:
        ax.plot(t, errs[k], color=SLOT[k], lw=1.5 if k != "full" else 2.0, label=labels[k])
    i = int(np.nanargmax(errs[keys[-1]]))
    ax.annotate(f"max {errs[keys[-1]][i]:.0f} m", (t[i], errs[keys[-1]][i]), textcoords="offset points",
                xytext=(4, 4), fontsize=8, color=INK2)
    if "ins" in errs and np.nanmax(errs["ins"]) > ymax:
        ax.text(0.99, 0.97, f"INS peaks reach {np.nanmax(errs['ins']):.0f} m (off scale)", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=INK2)
    ax.set_ylim(0, ymax)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("time (s)   ·   shaded = GNSS denied")
    ax.set_ylabel("horizontal error (m)")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", fontsize=8.5, ncol=len(keys), bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sensor_compare_plot(entries, denied, ref_drive, title, path):
    """Error vs time for the same route recorded by different sensor grades (each vs its own truth)."""
    fig, ax = plt.subplots(figsize=(9, 3.8))
    t_ref = ref_drive.t - ref_drive.t[0]
    for a, b in spans(denied):
        ax.axvspan(t_ref[a], t_ref[b - 1], color=INK2, alpha=0.10, lw=0)
    for j, (drv, rec, label) in enumerate(entries):
        en = drv.truth_en(rec["frame"])
        err = np.hypot(rec["e"] - en[:, 0], rec["n"] - en[:, 1])
        t = drv.t - drv.t[0]
        color = (SLOT["ins"], SLOT["ai"])[j]          # slots 1, 2 in fixed order
        ax.plot(t, err, color=color, lw=1.8, label=label)
        i = int(np.nanargmax(err))
        ax.annotate(f"max {err[i]:.0f} m", (t[i], err[i]), textcoords="offset points", xytext=(4, 4), fontsize=8, color=INK2)
    ax.set_xlim(t_ref[0], t_ref[-1])
    ax.set_ylim(0, None)
    ax.set_xlabel("time (s)   ·   shaded = inside the tunnel (GNSS denied)")
    ax.set_ylabel("horizontal error (m)")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def speed_plot(drive, rec, denied, path):
    t = drive.t - drive.t[0]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for a, b in spans(denied):
        ax.axvspan(t[a], t[b - 1], color=INK2, alpha=0.10, lw=0)
    ax.plot(t, drive.truth_speed * 3.6, color=INK, lw=1.6, label="Reference (CAN wheel speed)")
    ax.plot(t, rec["ai_speed"] * 3.6, color=SLOT["ai"], lw=1.3, label="SpeedNet (phone IMU only)")
    err = rec["ai_speed"] - drive.truth_speed
    rmse = np.sqrt(np.nanmean(err ** 2))
    ax.text(0.99, 0.95, f"RMSE {rmse:.2f} m/s ({rmse * 3.6:.1f} km/h)", transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=INK2)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, None)
    ax.set_xlabel("time (s)   ·   shaded = GNSS denied")
    ax.set_ylabel("speed (km/h)")
    ax.set_title("AI speed estimation on a held-out IO-VNBD drive (no odometer, no GNSS input)", loc="left")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def benchmark_plot(path):
    b = json.loads((ROOT / "results" / "benchmark.json").read_text())["summary"]
    Ls = sorted(b, key=int)
    keymap = {"ins": "ins_nhc", "ai": "ai_speed", "full": "full"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, metric, ttl in ((axes[0], "drift_pct_aggregate", "Aggregate drift (total end error / total distance)"),
                            (axes[1], "drift_pct_median", "Median drift per outage")):
        x = np.arange(len(Ls))
        w = 0.26
        for j, k in enumerate(("ins", "ai", "full")):
            vals = [b[L][keymap[k]][metric] for L in Ls]
            bars = ax.bar(x + (j - 1) * (w + 0.02), vals, width=w, color=SLOT[k], label=LABEL[k], zorder=3,
                          edgecolor=SURFACE, linewidth=2)
            for bx, v in zip(bars, vals):
                ax.text(bx.get_x() + bx.get_width() / 2, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
        ax.axhline(10, color=INK, lw=1.2, ls=(0, (4, 3)), zorder=4)
        ax.text(1.01, 10, "PS target\n10 %", transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=8, color=INK)
        ax.set_xticks(x, [f"{L} s\n({b[L]['full']['mean_distance_m']:.0f} m)" for L in Ls])
        ax.set_xlabel("GNSS outage length (mean distance driven)")
        ax.set_ylabel("drift (% of distance)")
        ax.set_title(ttl, loc="left", fontsize=11)
        ax.set_ylim(0, max(60, ax.get_ylim()[1]))
    axes[1].legend(loc="upper left", fontsize=8.5)
    fig.suptitle("IO-VNBD held-out test drives (S1, S4) - smartphone IMU, 1 Hz GNSS, repeated outages", x=0.01,
                 ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    roads = RoadNetwork.load(ROOT / "data" / "osm" / "coventry_test.json")
    meta = {d["file"]: d for d in json.loads((ROOT / "data" / "demo" / "demo_drives.json").read_text())}
    for f in meta:
        drive = load_generic_csv(ROOT / "data" / "demo" / f)
        tag = Path(f).stem
        recs, denied, outs = run_all(drive, roads, outage=60)
        km = drive.distance_travelled()[-1] / 1000
        trajectory_plot(drive, roads, recs, denied, outs,
                        f"IO-VNBD {tag.split('_')[-1].upper()} (held-out) - {km:.1f} km, 60 s GNSS outages every 150 s",
                        OUT / f"{tag}_trajectory.png")
        error_plot(drive, recs, denied, f"IO-VNBD {tag.split('_')[-1].upper()} - position error through repeated 60 s outages",
                   OUT / f"{tag}_error.png")
        if tag.endswith("s1"):
            speed_plot(drive, recs["ai"], denied, OUT / "speednet_speed.png")
        print("plots for", tag)
    benchmark_plot(OUT / "benchmark_drift.png")

    d, net = build_scenario("delhi_pragati_tunnel", str(ROOT / "data" / "osm"), "smartphone")
    recs, denied, outs = run_all(d, net)
    trajectory_plot(d, net, recs, denied, outs, "Pragati Maidan tunnel, Delhi: 1.3 km without GNSS\n(simulated smartphone IMU on OSM roads)",
                    OUT / "delhi_tunnel_trajectory.png", ref_label="True path")
    df, netf = build_scenario("delhi_pragati_tunnel", str(ROOT / "data" / "osm"), "fog")
    recf, denf, _ = run_all(df, netf, profile="fog", keys=("full",))
    sensor_compare_plot([(df, recf["full"], "FOG IMU @200 Hz (edge engine)"),
                         (d, recs["full"], "Smartphone IMU @10 Hz (phone app)")], denied, d,
                        "Pragati Maidan tunnel - full pipeline error by sensor grade", OUT / "delhi_tunnel_error.png")
    print("plots in", OUT)


if __name__ == "__main__":
    main()
