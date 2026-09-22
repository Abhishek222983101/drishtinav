"""Download the IO-VNBD drives used by DrishtiNav (smartphone S- + vehicle V- pairs).

    python scripts/download_iovnbd.py [--out ~/iovnbd/raw] [--all-synced]

Files live in Git LFS on github.com/onyekpeu/IO-VNBD; we fetch them through the
LFS media endpoint. The default set (~280 MB) is the one our train/val/test split
uses (training/dataset.py). Please be gentle with the dataset owner's LFS
bandwidth: don't pass --all-synced unless you need it.
"""
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

REPO = "onyekpeu/IO-VNBD"
BASE = "Synchronised V abd S datasets/Categorised IOVNB Dataset"
DEFAULT = ["S (Driver A)/S1", "S (Driver A)/S2", "S (Driver A)/S3a", "S (Driver A)/S3b", "S (Driver A)/S3c",
           "S (Driver A)/S4", "M (Driver B)", "Vf (Driver E)/V-Vfa02", "Vta (Driver E)/Vta01a",
           "Vta (Driver E)/Vta02", "Vtb (Driver E)/Vtb01", "Vtb (Driver E)/Vtb05"]


def tree():
    url = f"https://api.github.com/repos/{REPO}/git/trees/master?recursive=1"
    with urllib.request.urlopen(url, timeout=60) as r:
        return [x["path"] for x in json.loads(r.read())["tree"] if x["type"] == "blob" and x["path"].endswith(".csv")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "iovnbd" / "raw"))
    ap.add_argument("--all-synced", action="store_true")
    a = ap.parse_args()
    paths = [p for p in tree() if p.startswith(BASE)]
    if not a.all_synced:
        paths = [p for p in paths if any(p.startswith(f"{BASE}/{d}/") for d in DEFAULT)]
    for p in paths:
        folder = Path(p).parent.name.replace(" ", "").replace("(", "").replace(")", "")
        dst = Path(a.out) / folder / Path(p).name
        if dst.exists() and dst.stat().st_size > 1000:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://media.githubusercontent.com/media/{REPO}/master/{urllib.parse.quote(p)}"
        print("GET", p, flush=True)
        urllib.request.urlretrieve(url, dst)
    print(f"done -> {a.out}")


if __name__ == "__main__":
    main()
