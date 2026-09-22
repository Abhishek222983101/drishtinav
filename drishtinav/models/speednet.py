"""Dependency-free (numpy) inference for the trained SpeedNet.

The same weights are exported to JSON for the phone app and to ONNX for native
Android / edge runtimes; this module is the reference implementation used by
the Python engine and by the parity tests.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "models" / "speednet.npz"


def _softplus(x):
    return np.log1p(np.exp(-abs(x))) + max(x, 0.0)


class SpeedNetRuntime:
    def __init__(self, path=DEFAULT_PATH):
        z = np.load(path)
        self.mu = z["mu"].astype(np.float32)
        self.sd = z["sd"].astype(np.float32)
        self.dilations = [int(d) for d in z["dilations"]]
        self.win = int(z["win"])
        self.convs = [(z[f"convs.{i}.weight"], z[f"convs.{i}.bias"]) for i in range(len(self.dilations))]
        self.fc1 = (z["fc1.weight"], z["fc1.bias"])
        self.fc2 = (z["fc2.weight"], z["fc2.bias"])
        self.kernel = self.convs[0][0].shape[2]

    def forward(self, window: np.ndarray):
        """window: (T, C) raw features -> (speed m/s, sigma m/s, p_stationary)."""
        x = ((window - self.mu) / self.sd).T.astype(np.float32)          # (C, T)
        T = x.shape[1]
        for (w, b), d in zip(self.convs, self.dilations):
            pad = (self.kernel - 1) * d
            xp = np.pad(x, ((0, 0), (pad, 0)))
            cols = np.concatenate([xp[:, j * d:j * d + T] for j in range(self.kernel)], axis=0)  # (k*C, T)
            wm = w.transpose(0, 2, 1).reshape(w.shape[0], -1)                                  # (O, k*C)
            x = np.maximum(wm @ cols + b[:, None], 0.0)
        h = np.concatenate([x.mean(axis=1), x[:, -1]])
        h = np.maximum(self.fc1[0] @ h + self.fc1[1], 0.0)
        o = self.fc2[0] @ h + self.fc2[1]
        speed = _softplus(float(o[0]))
        sigma = float(np.exp(0.5 * np.clip(o[1], -6, 6)))
        p_stat = float(1.0 / (1.0 + np.exp(-o[2])))
        return speed, sigma, p_stat


class SpeedNetStream:
    """Ring buffer of model-rate features with periodic inference."""

    def __init__(self, runtime: Optional[SpeedNetRuntime], every: int = 5):
        self.rt = runtime
        self.every = every
        self.buf = deque(maxlen=runtime.win if runtime else 1)
        self.count = 0
        self.last = None

    def push(self, feat) -> Optional[tuple]:
        if self.rt is None:
            return None
        self.buf.append(np.asarray(feat, np.float32))
        self.count += 1
        if len(self.buf) == self.buf.maxlen and self.count % self.every == 0:
            self.last = self.rt.forward(np.stack(self.buf))
            return self.last
        return None
