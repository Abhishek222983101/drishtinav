"""System tests: map matching, SpeedNet runtime, dataset resync, engine, server."""
import json
from pathlib import Path

import numpy as np
import pytest

from drishtinav import evaluation as ev
from drishtinav.config import make_config
from drishtinav.engine import GnssFix, NavEngine
from drishtinav.io.drive import Drive
from drishtinav.io.iovnbd import resync_segments
from drishtinav.io.synthetic import build_scenario
from drishtinav.mapmatch import HMMMapMatcher
from drishtinav.models.speednet import SpeedNetRuntime
from drishtinav.roadnet import RoadNetwork

ROOT = Path(__file__).resolve().parent.parent
OSM = ROOT / "data" / "osm"


def grid_network():
    """3x3 grid of two-way roads, 200 m spacing, around (28.6, 77.2)."""
    fr_nodes, ways = [], []
    lat0, lon0 = 28.6, 77.2
    dlat, dlon = 200 / 111320.0, 200 / (111320.0 * np.cos(np.radians(lat0)))
    for i in range(3):
        for j in range(3):
            fr_nodes.append([lat0 + i * dlat, lon0 + j * dlon])
    k = lambda i, j: i * 3 + j
    wid = 0
    for i in range(3):
        ways.append({"id": wid, "nodes": [k(i, 0), k(i, 1), k(i, 2)], "hw": "residential", "oneway": 0, "tunnel": 0,
                     "bridge": 0, "name": f"row{i}", "lanes": None}); wid += 1
        ways.append({"id": wid, "nodes": [k(0, i), k(1, i), k(2, i)], "hw": "residential", "oneway": 0, "tunnel": 0,
                     "bridge": 0, "name": f"col{i}", "lanes": None}); wid += 1
    return RoadNetwork({"name": "grid", "bbox": [lat0, lon0, lat0 + 2 * dlat, lon0 + 2 * dlon],
                        "nodes": fr_nodes, "ways": ways})


def test_hmm_matches_the_driven_road():
    net = grid_network()
    mm = HMMMapMatcher(net)
    rng = np.random.default_rng(3)
    p0 = net.node_xy[0]
    names = []
    for s in range(0, 380, 10):                       # drive East along row 0 with 6 m noise
        xy = p0 + np.array([s, 0.0]) + rng.normal(0, 6, 2)
        m = mm.update(float(s), xy, 0.0, 10.0, pos_sigma=6.0, travelled=10.0)
        names.append(net.ways[net.seg_way[m.seg]]["name"])
    assert names[-20:].count("row0") >= 18


def test_speednet_runtime_outputs():
    rt = SpeedNetRuntime()
    rng = np.random.default_rng(0)
    w = (rng.normal(0, 1, (rt.win, len(rt.mu))) * rt.sd + rt.mu).astype(np.float32)
    v, sigma, p = rt.forward(w)
    assert v >= 0 and sigma > 0 and 0 <= p <= 1


def test_speednet_json_matches_npz():
    js = json.loads((ROOT / "models" / "speednet.json").read_text())
    z = np.load(ROOT / "models" / "speednet.npz")
    w0 = np.array(js["layers"]["convs.0.weight"]["data"]).reshape(js["layers"]["convs.0.weight"]["shape"])
    assert np.allclose(w0, z["convs.0.weight"], atol=1e-5)
    assert js["win"] == int(z["win"])


def test_resync_recovers_a_known_lag():
    rng = np.random.default_rng(1)
    n, lag = 12000, 23
    yaw = np.convolve(rng.normal(0, 0.3, n + 400), np.ones(15) / 15, "same")
    phone_yaw = yaw[200:200 + n]
    veh_yaw = yaw[200 - lag:200 - lag + n]            # vehicle log shifted by `lag` samples
    gyro = np.column_stack([rng.normal(0, 0.05, n), phone_yaw + rng.normal(0, 0.02, n), rng.normal(0, 0.05, n)])
    d = Drive("syn", np.arange(n) / 10.0, np.tile([0, 0, 9.8], (n, 1)), gyro,
              truth_lat=np.full(n, 52.4), truth_lon=np.full(n, -1.5), truth_speed=np.full(n, 10.0),
              truth_course=np.zeros(n), meta={"vehicle_yaw_rate": veh_yaw})
    segs = resync_segments(d)
    assert segs and all(s.meta["resync_lag"] == -lag for s in segs)
    assert segs[0].meta["yaw_gyro_axis"] == 1


@pytest.fixture(scope="module")
def delhi_phone():
    return build_scenario("delhi_pragati_tunnel", str(OSM), "smartphone", seed=3)


def test_engine_crosses_the_tunnel_within_benchmark():
    """Median over noise realisations (one simulated run is one random draw)."""
    drifts = []
    for seed in (2, 3, 4):
        d, net = build_scenario("delhi_pragati_tunnel", str(OSM), "smartphone", seed=seed)
        rec = ev.run(d, make_config(), speed_model=SpeedNetRuntime(), roads=net, fixes=ev.device_gnss(d))
        en = d.truth_en(rec["frame"])
        err = np.hypot(rec["e"] - en[:, 0], rec["n"] - en[:, 1])
        den = d.gnss_denied
        i0, i1 = np.where(den)[0][0], np.where(den)[0][-1]
        tunnel = d.distance_travelled()[i1] - d.distance_travelled()[i0]
        assert tunnel > 1200
        drifts.append(err[i1] / tunnel)
        modes = [m for m in rec["mode"][i0:i1 + 60] if m is not None]
        assert "DR" in modes and "RECOVERY" in modes
        assert np.nanmean(rec["latency_us"]) < 5000  # far below the 100 ms budget at 10 Hz
    assert np.median(drifts) < 0.10                  # PS benchmark: < 10 % of distance


def test_engine_streaming_is_seamless(delhi_phone):
    """Position never jumps when GNSS disappears (the INS runs every epoch)."""
    d, net = delhi_phone
    eng = NavEngine(make_config(), SpeedNetRuntime(), net, frame=d.frame())
    fixes = ev.device_gnss(d)
    prev, jumps = None, []
    for i in range(len(d)):
        st = eng.step(d.t[i], d.accel[i], d.gyro[i], fixes.get(i))
        if prev is not None and st.mode in ("DR",) and prev.mode == "GNSS":
            jumps.append(np.hypot(st.e - prev.e, st.n - prev.n))
        prev = st
    assert jumps and max(jumps) < 3.0                # < one epoch of travel at 30 m/s


def test_server_api():
    import sys
    sys.path.insert(0, str(ROOT))
    from server.app import create_app
    c = create_app().test_client()

    # front-ends
    assert c.get("/app/").status_code == 200 and c.get("/").status_code == 200
    assert c.get("/config.js").status_code == 200

    # static data layer (what the Vercel build ships byte-identical to)
    sc = c.get("/data/scenarios.json").get_json()
    assert any(s["id"].startswith("sim:") for s in sc)
    tunnel = next(s for s in sc if s["id"] == "sim:delhi_pragati_tunnel:smartphone")
    drv = c.get(f"/data/drives/{tunnel['drive']}.json").get_json()
    assert len(drv["t"]) == len(drv["accel"]) and drv["roads"] == "delhi_central.json"
    run = c.get(f"/data/runs/{tunnel['runs']['tunnel']}.json").get_json()
    assert "full" in run["configs"] and len(run["t"]) > 0
    bench = c.get("/data/benchmark.json").get_json()
    assert "speednet" in bench

    # live engine API (CORS + JSON error paths)
    health = c.get("/api/health").get_json()
    assert health["status"] == "ok"
    r = c.get("/api/run?scenario=sim:delhi_pragati_tunnel:smartphone&configs=full")
    assert r.status_code == 200 and r.headers["Access-Control-Allow-Origin"] == "*"
    assert "full" in r.get_json()["configs"]
    bad = c.get("/api/run?scenario=not:a:scenario")
    assert bad.status_code == 404 and "error" in bad.get_json()
