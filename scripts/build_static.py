"""Build the static website (dashboard + phone app + precomputed engine results).

    python scripts/build_static.py --engine https://drishtinav-engine.onrender.com
    cd site && vercel deploy --prod

Every engine result shipped here is produced by the same code the live server
runs (server/datagen.py), so the static site and the live API agree exactly.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server import datagen  # noqa: E402

WEB = ROOT / "web"

VERCEL_JSON = {
    "trailingSlash": True,
    "headers": [
        {"source": "/app/sw.js", "headers": [{"key": "Service-Worker-Allowed", "value": "/app/"},
                                             {"key": "Cache-Control", "value": "no-cache"}]},
        {"source": "/config.js", "headers": [{"key": "Cache-Control", "value": "no-cache"}]},
        {"source": "/data/(.*)", "headers": [{"key": "Cache-Control", "value": "public, max-age=600"},
                                             {"key": "Access-Control-Allow-Origin", "value": "*"}]},
    ],
}


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":")))
    return path.stat().st_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "site"))
    ap.add_argument("--engine", default="", help="URL of the live Python engine API (Render)")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        # keep Vercel's project link (.vercel/) across rebuilds
        for p in out.iterdir():
            if p.name != ".vercel":
                shutil.rmtree(p) if p.is_dir() else p.unlink()
    out.mkdir(parents=True, exist_ok=True)

    # front-ends
    shutil.copy(WEB / "dashboard" / "index.html", out / "index.html")
    shutil.copytree(WEB / "dashboard", out / "dashboard", ignore=shutil.ignore_patterns("index.html"))
    shutil.copytree(WEB / "mobile", out / "app")
    shutil.copytree(WEB / "vendor", out / "vendor")
    shutil.copy(WEB / "mobile" / "icon.svg", out / "favicon.svg")
    (out / "models").mkdir()
    for f in ("speednet.json", "speednet.onnx", "speednet_metrics.json"):
        shutil.copy(ROOT / "models" / f, out / "models" / f)
    shutil.copytree(ROOT / "data" / "osm", out / "data" / "osm")
    (out / "config.js").write_text(f'window.DRISHTI = {{ data: "/data", engine: "{a.engine.rstrip("/")}" }};\n')
    write_json(out / "vercel.json", VERCEL_JSON)

    # data products
    sc = datagen.scenarios()
    write_json(out / "data" / "scenarios.json", sc)
    write_json(out / "data" / "benchmark.json", datagen.benchmark())
    for s in sc:
        for outage, key in s["runs"].items():
            t0 = time.time()
            n = write_json(out / "data" / "runs" / f"{key}.json",
                           datagen.run_result(s["id"], 0 if outage == "tunnel" else int(outage)))
            print(f"runs/{key}.json  {n / 1e3:.0f} kB  {time.time() - t0:.1f} s", flush=True)
        if s["drive"]:
            n = write_json(out / "data" / "drives" / f"{s['drive']}.json", datagen.drive_stream(s["id"]))
            print(f"drives/{s['drive']}.json  {n / 1e3:.0f} kB", flush=True)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True).stdout.strip()
    except OSError:
        commit = ""
    write_json(out / "data" / "build.json", {"built": time.strftime("%Y-%m-%d %H:%M"), "commit": commit,
                                             "engine": a.engine})
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and ".vercel" not in p.parts)
    print(f"site -> {out}  ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
