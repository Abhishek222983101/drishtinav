"""Build SpeedNet training windows from IO-VNBD (phone IMU in, CAN speed out).

Pipeline per drive: load -> resync phone/vehicle (see drishtinav.io.iovnbd) ->
offline mount alignment -> vehicle-frame channels -> causal low-pass ->
sliding windows labelled with the CAN wheel-speed at the window's last sample.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drishtinav.alignment import align_batch  # noqa: E402
from drishtinav.features import CHANNELS, vehicle_channels  # noqa: E402
from drishtinav.io.iovnbd import find_drives, load_drive, resync_segments  # noqa: E402

# Drive-level split: nothing from a test drive is ever seen in training.
SPLITS = {
    "train": ["M", "S2", "S3c", "Vfa02", "Vta1a", "Vtb5", "Vtb1"],   # drivers A, B, E
    "val": ["S3a", "S3b", "Vta2"],
    "test": ["S1", "S4"],                                          # held out entirely
}


def segments_for(names, root):
    out = []
    for name, s, v in find_drives(root):
        if name in names and v:
            out.extend(resync_segments(load_drive(s, v, name)))
    return out


def segment_arrays(seg, rate=10.0):
    a_fwd, a_lat, a_up, yaw, _ = align_batch(seg.accel, seg.gyro, seg.truth_speed, seg.t,
                                             seg.meta.get("vehicle_yaw_rate", seg.meta.get("vbox_yaw_rate")))
    X = vehicle_channels(a_fwd, a_lat, a_up, yaw, seg.accel, seg.gyro, rate)
    y = np.nan_to_num(seg.truth_speed.astype(np.float32))
    return X.astype(np.float32), y


def windows(X, y, win, stride=1):
    idx = np.arange(win - 1, len(X), stride)
    W = np.stack([X[i - win + 1:i + 1] for i in idx])
    return W, y[idx]


def build(root, win, stride=2):
    data = {}
    for split, names in SPLITS.items():
        Ws, ys, segs = [], [], []
        for seg in segments_for(names, root):
            X, y = segment_arrays(seg)
            W, yy = windows(X, y, win, stride)
            Ws.append(W); ys.append(yy); segs.append(seg.name)
        data[split] = (np.concatenate(Ws), np.concatenate(ys), segs)
        print(f"{split}: {len(data[split][1])} windows from {len(segs)} segments", flush=True)
    return data


if __name__ == "__main__":
    print(CHANNELS)
