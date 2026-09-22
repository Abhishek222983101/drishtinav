"""Build an offline road database from OpenStreetMap (Overpass API).

The navigation engine never talks to the network at run time: map matching runs
against a compact JSON extract produced once by this script, which is what the PS
asks for ("teams can bring the downloaded map database").

    python scripts/fetch_osm.py --name delhi_pragati --bbox 28.585 77.215 28.635 77.265
    python scripts/fetch_osm.py --name coventry_test --bbox 52.365 -1.61 52.425 -1.495

Output format (data/osm/<name>.json):
    {"name", "bbox": [s, w, n, e], "nodes": [[lat, lon], ...],
     "ways": [{"id", "nodes": [node_idx...], "hw", "oneway", "tunnel", "bridge", "name", "lanes"}]}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
DRIVABLE = ("motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|"
            "motorway_link|trunk_link|primary_link|secondary_link|tertiary_link")


def fetch(bbox, timeout=180):
    s, w, n, e = bbox
    q = (f'[out:json][timeout:{timeout}];'
         f'way["highway"~"^({DRIVABLE})$"]["access"!~"^(no|private)$"]({s},{w},{n},{e});'
         f'out body;>;out skel qt;')
    data = urllib.parse.urlencode({"data": q}).encode()
    last = None
    for url in ENDPOINTS:
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, data=data, headers={"User-Agent": "DrishtiNav/1.0 (SIH research)"})
                with urllib.request.urlopen(req, timeout=timeout + 30) as r:
                    return json.loads(r.read().decode())
            except Exception as ex:  # network errors, 429, 504 ...
                last = ex
                time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Overpass fetch failed: {last}")


def compact(osm, name, bbox):
    nodes = {el["id"]: (el["lat"], el["lon"]) for el in osm["elements"] if el["type"] == "node"}
    idx, out_nodes, ways = {}, [], []
    for el in osm["elements"]:
        if el["type"] != "way":
            continue
        tags = el.get("tags", {})
        ids = [i for i in el["nodes"] if i in nodes]
        if len(ids) < 2:
            continue
        refs = []
        for i in ids:
            if i not in idx:
                idx[i] = len(out_nodes)
                lat, lon = nodes[i]
                out_nodes.append([round(lat, 7), round(lon, 7)])
            refs.append(idx[i])
        hw = tags.get("highway", "")
        oneway = tags.get("oneway", "no")
        ways.append({
            "id": el["id"],
            "nodes": refs,
            "hw": hw,
            "oneway": 1 if oneway in ("yes", "true", "1") or hw.startswith("motorway") or tags.get("junction") == "roundabout"
            else (-1 if oneway == "-1" else 0),
            "tunnel": 1 if tags.get("tunnel") in ("yes", "building_passage", "culvert") or tags.get("covered") == "yes" else 0,
            "bridge": 1 if tags.get("bridge") == "yes" else 0,
            "name": tags.get("name", tags.get("ref", "")),
            "lanes": int(tags["lanes"]) if tags.get("lanes", "").isdigit() else None,
        })
    return {"name": name, "bbox": list(bbox), "source": "OpenStreetMap contributors (ODbL)",
            "nodes": out_nodes, "ways": ways}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("S", "W", "N", "E"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data" / "osm"))
    a = ap.parse_args(argv)
    osm = fetch(a.bbox)
    road = compact(osm, a.name, a.bbox)
    out = Path(a.out) / f"{a.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(road, separators=(",", ":")))
    index = []
    for p in sorted(out.parent.glob("*.json")):
        if p.name != "index.json":
            d = json.loads(p.read_text())
            index.append({"file": p.name, "name": d["name"], "bbox": d["bbox"], "ways": len(d["ways"])})
    (out.parent / "index.json").write_text(json.dumps(index, indent=1))
    print(f"{out}: {len(road['ways'])} ways, {len(road['nodes'])} nodes, "
          f"{sum(w['tunnel'] for w in road['ways'])} tunnel ways, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    sys.exit(main())
