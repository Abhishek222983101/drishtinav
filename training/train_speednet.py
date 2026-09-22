"""Train SpeedNet: the AI pseudo-odometer + vibration/stationarity model.

    python training/train_speednet.py --data ~/iovnbd/raw

Architecture (≈17 k parameters, <0.1 ms per inference on a phone CPU):
    input  (7 channels x 100 samples = 10 s @ 10 Hz, vehicle frame, see drishtinav/features.py)
    4 x dilated causal Conv1d(k=5, dilation 1/2/4/8, 32 ch) + ReLU     (temporal CNN, cf. OdoNet)
    [mean-pool || last-step] -> Linear(64, 32) -> ReLU -> Linear(32, 3)
    outputs: speed (softplus, m/s), log-variance of that speed, stationary logit

The speed + variance heads are trained with a Gaussian negative log-likelihood so
the network reports *how much it trusts itself*; the EKF uses that variance as
the measurement noise of the pseudo-odometer update (the AI-adaptive noise idea
of Brossard et al., "AI-IMU Dead-Reckoning", IEEE T-IV 2020).

Exports: models/speednet.npz (numpy runtime), models/speednet.json (phone app),
models/speednet.onnx (Android / edge ONNX Runtime), results/speednet_metrics.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from drishtinav.features import CHANNELS  # noqa: E402

WIN = 100
HID = 32
KERNEL = 5
DILATIONS = (1, 2, 4, 8)
STATIONARY_MS = 0.3


class SpeedNet(nn.Module):
    def __init__(self, n_in=len(CHANNELS), hid=HID):
        super().__init__()
        self.convs = nn.ModuleList()
        c = n_in
        for d in DILATIONS:
            self.convs.append(nn.Conv1d(c, hid, KERNEL, dilation=d))
            c = hid
        self.fc1 = nn.Linear(2 * hid, hid)
        self.fc2 = nn.Linear(hid, 3)

    def forward(self, x):                       # x: (B, C, T) already normalised
        for conv in self.convs:
            pad = (KERNEL - 1) * conv.dilation[0]
            x = F.relu(conv(F.pad(x, (pad, 0))))  # causal padding
        h = torch.cat([x.mean(dim=2), x[:, :, -1]], dim=1)
        o = self.fc2(F.relu(self.fc1(h)))
        speed = F.softplus(o[:, 0])
        logvar = o[:, 1].clamp(-6, 6)
        return speed, logvar, o[:, 2]


def load_cache(path):
    z = np.load(path)
    return {s: (z[f"{s}_W"], z[f"{s}_y"]) for s in ("train", "val", "test")}


def augment(W, rng):
    """Mount / sensor robustness: small yaw re-rotation, gain and bias jitter."""
    W = W.copy()
    B = len(W)
    th = rng.normal(0, np.radians(4), B)[:, None]
    af, al = W[:, :, 0].copy(), W[:, :, 1].copy()
    W[:, :, 0] = np.cos(th) * af - np.sin(th) * al
    W[:, :, 1] = np.sin(th) * af + np.cos(th) * al
    W[:, :, 0:3] += rng.normal(0, 0.05, (B, 1, 3))           # accel bias
    W[:, :, 3] += rng.normal(0, 0.003, (B, 1))                # gyro bias
    W[:, :, 4:6] *= rng.lognormal(0, 0.15, (B, 1, 2))         # phone/holder vibration gain
    return W.astype(np.float32)


def evaluate(model, W, y, mu, sd, bs=4096):
    model.eval()
    out_s, out_v, out_p = [], [], []
    with torch.no_grad():
        for i in range(0, len(W), bs):
            x = torch.from_numpy(((W[i:i + bs] - mu) / sd).transpose(0, 2, 1).copy())
            s, lv, pl = model(x)
            out_s.append(s.numpy()); out_v.append(lv.numpy()); out_p.append(torch.sigmoid(pl).numpy())
    s, lv, p = map(np.concatenate, (out_s, out_v, out_p))
    err = s - y
    stat = y < STATIONARY_MS
    z = err / np.exp(0.5 * lv)
    return {
        "rmse_ms": float(np.sqrt(np.mean(err ** 2))),
        "mae_ms": float(np.mean(np.abs(err))),
        "rmse_moving_ms": float(np.sqrt(np.mean(err[~stat] ** 2))),
        "bias_ms": float(np.mean(err)),
        "calibration_std_of_z": float(np.std(z)),        # ~1 if predicted sigma is honest
        "within_2sigma": float(np.mean(np.abs(z) < 2)),
        "stationary_accuracy": float(np.mean((p > 0.5) == stat)),
        "stationary_recall": float(np.mean(p[stat] > 0.5)) if stat.any() else None,
        "n": int(len(y)),
    }, s, lv, p


def export(model, mu, sd, out_dir: Path, metrics):
    out_dir.mkdir(parents=True, exist_ok=True)
    sdict = {k: v.detach().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    np.savez(out_dir / "speednet.npz", mu=mu.astype(np.float32), sd=sd.astype(np.float32),
             dilations=np.array(DILATIONS), win=np.array(WIN), **sdict)
    js = {"channels": CHANNELS, "win": WIN, "kernel": KERNEL, "dilations": list(DILATIONS),
          "mu": mu.tolist(), "sd": sd.tolist(),
          "layers": {k: {"shape": list(v.shape), "data": np.round(v.ravel(), 6).tolist()} for k, v in sdict.items()},
          "metrics": metrics}
    (out_dir / "speednet.json").write_text(json.dumps(js, separators=(",", ":")))

    class Wrapped(nn.Module):  # bake normalisation into the ONNX graph
        def __init__(self, m):
            super().__init__()
            self.m = m
            self.register_buffer("mu", torch.from_numpy(mu.astype(np.float32)).view(1, -1, 1))
            self.register_buffer("sd", torch.from_numpy(sd.astype(np.float32)).view(1, -1, 1))

        def forward(self, x):   # x: (B, C, T) raw features
            s, lv, pl = self.m((x - self.mu) / self.sd)
            return torch.stack([s, torch.exp(0.5 * lv), torch.sigmoid(pl)], dim=1)

    try:
        torch.onnx.export(Wrapped(model).eval(), torch.zeros(1, len(CHANNELS), WIN), str(out_dir / "speednet.onnx"),
                          input_names=["imu_window"], output_names=["speed_sigma_pstationary"],
                          dynamic_axes={"imu_window": {0: "batch"}}, opset_version=17, dynamo=False)
    except Exception as ex:  # ONNX export is optional for the numpy/JS runtimes
        print("ONNX export skipped:", ex)


def main(argv=None):
    global WIN
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / "iovnbd" / "raw"))
    ap.add_argument("--win", type=int, default=WIN, help="window length in samples @10 Hz")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", default=str(ROOT / "models"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    WIN = a.win
    a.cache = a.cache or str(ROOT / "data" / "cache" / f"speed_w{WIN}.npz")
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    if not Path(a.cache).exists():
        from dataset import build
        d = build(a.data, WIN, stride=2)
        Path(a.cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.cache, **{f"{k}_{n}": v for k, (W, y, _) in d.items() for n, v in (("W", W), ("y", y))})
    data = load_cache(a.cache)
    Wtr, ytr = data["train"]
    mu = Wtr.reshape(-1, Wtr.shape[2]).mean(0)
    sd = Wtr.reshape(-1, Wtr.shape[2]).std(0) + 1e-6

    model = SpeedNet()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    best, best_state = np.inf, None
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        perm = rng.permutation(len(Wtr))
        tot = 0.0
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            W = augment(Wtr[b], rng)
            x = torch.from_numpy(((W - mu) / sd).transpose(0, 2, 1).copy())
            y = torch.from_numpy(ytr[b])
            s, lv, pl = model(x)
            nll = 0.5 * (lv + (s - y) ** 2 / torch.exp(lv))
            bce = F.binary_cross_entropy_with_logits(pl, (y < STATIONARY_MS).float())
            loss = nll.mean() + 0.3 * F.smooth_l1_loss(s, y) + 0.5 * bce
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        val, *_ = evaluate(model, *data["val"], mu, sd)
        print(f"epoch {ep + 1:2d} loss {tot / len(perm):.3f} val rmse {val['rmse_ms']:.2f} m/s "
              f"z-std {val['calibration_std_of_z']:.2f} stat-acc {val['stationary_accuracy']:.3f}", flush=True)
        if val["rmse_ms"] < best:
            best = val["rmse_ms"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    metrics = {split: evaluate(model, *data[split], mu, sd)[0] for split in ("train", "val", "test")}
    metrics["params"] = int(n_params)
    metrics["train_seconds"] = round(time.time() - t0, 1)
    metrics["window_s"] = WIN / 10.0
    from dataset import SPLITS
    metrics["dataset"] = "IO-VNBD synchronised drives (resynced), drive-level split: " + \
        " | ".join(f"{k} {','.join(v)}" for k, v in SPLITS.items())
    export(model, mu, sd, Path(a.out), metrics)
    (Path(a.out) / "speednet_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
