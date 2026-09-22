"""Online Hidden-Markov-Model map matching (Newson & Krumm, ACM GIS 2009).

States are candidate road segments near the current position estimate:

* emission   p(z | seg)  = N(dist; 0, sigma) * N(heading error; 0, sigma_h)
* transition p(seg_j | seg_i) = exp(-|route distance - travelled distance| / beta)

Newson & Krumm run Viterbi offline; for a navigation engine we run the HMM
*forward filter* causally at ~1 Hz, which gives a posterior probability per road
right now (used as the match confidence) plus back-pointers so the matched path
can be smoothed afterwards. When GNSS is denied the matched road feeds the EKF
two pseudo-measurements (cross-track position and road heading), which is the
"map constraint snaps the drifting IMU path back onto the road grid" the PS
describes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .geo import wrap_angle
from .roadnet import RoadNetwork


@dataclass
class MatchResult:
    seg: int
    xy: np.ndarray          # projected point on the road (local E, N)
    road_heading: float     # rad, in the direction of travel
    dist: float             # distance from estimate to road (m)
    confidence: float       # HMM posterior of the chosen road (0..1)
    t_on_seg: float         # 0..1 position along the segment
    seg_len: float
    tunnel: bool
    half_width: float


class HMMMapMatcher:
    def __init__(self, net: RoadNetwork, sigma_min: float = 4.0, beta: float = 8.0,
                 radius: float = 45.0, heading_sigma_deg: float = 25.0, max_candidates: int = 10,
                 min_interval: float = 1.0):
        self.net = net
        self.sigma_min = sigma_min
        self.beta = beta
        self.radius = radius
        self.heading_sigma = np.radians(heading_sigma_deg)
        self.max_candidates = max_candidates
        self.min_interval = min_interval
        self.reset()

    def reset(self):
        self._cands = None      # dict with arrays: seg, dirn, xy, t
        self._logp = None
        self._last_t = -np.inf
        self._travel = 0.0
        self.last: Optional[MatchResult] = None
        self.history = []       # (t, candidate arrays, back-pointers) for offline Viterbi

    # ------------------------------------------------------------------ helpers
    def _candidate_set(self, xy, heading, moving, radius):
        ids, proj, dist, t = self.net.candidates(xy, radius)
        if len(ids) == 0:
            return None
        dirn = np.ones(len(ids), dtype=int)
        head = self.net.seg_heading[ids].copy()
        if moving:
            two_way = ~self.net.seg_oneway[ids]
            rev = two_way & (np.abs(wrap_angle(heading - head)) > np.pi / 2)
            dirn[rev] = -1
            head[rev] = wrap_angle(head[rev] + np.pi)
        order = np.argsort(dist)[: self.max_candidates]
        return {"seg": ids[order], "dirn": dirn[order], "xy": proj[order], "dist": dist[order],
                "t": t[order], "head": head[order]}

    def _emission(self, c, heading, sigma, moving):
        lp = -0.5 * (c["dist"] / sigma) ** 2
        if moving:
            dh = np.abs(wrap_angle(heading - c["head"]))
            lp += -0.5 * (dh / self.heading_sigma) ** 2
        return lp

    def _route_dist(self, prev, i, cur, j, reach_cache):
        """Driving distance from previous candidate i to current candidate j."""
        s0, d0, t0 = prev["seg"][i], prev["dirn"][i], prev["t"][i]
        s1, d1, t1 = cur["seg"][j], cur["dirn"][j], cur["t"][j]
        L0, L1 = self.net.seg_len[s0], self.net.seg_len[s1]
        if s0 == s1 and d0 == d1:
            return (t1 - t0) * L0 * d0 if (t1 - t0) * d0 >= -0.05 else np.inf
        key = (int(s0), int(d0))
        if key not in reach_cache:
            reach_cache[key] = self.net.reachable_segments(int(s0), int(d0), self._travel + 60.0)
        reach = reach_cache[key]
        if int(s1) not in reach:
            return np.inf
        remain0 = (1 - t0) * L0 if d0 > 0 else t0 * L0
        into1 = t1 * L1 if d1 > 0 else (1 - t1) * L1
        return remain0 + reach[int(s1)] + into1

    # --------------------------------------------------------------------- API
    def update(self, t, xy, heading, speed, pos_sigma=5.0, travelled=None) -> Optional[MatchResult]:
        """Feed one position estimate. ``travelled`` = distance driven since the previous call."""
        self._travel += travelled if travelled is not None else 0.0
        if t - self._last_t < self.min_interval and self.last is not None:
            return self.last
        moving = speed > 2.0
        sigma = max(self.sigma_min, float(pos_sigma))
        cur = self._candidate_set(np.asarray(xy, float), heading, moving, max(self.radius, 3 * sigma))
        if cur is None:
            self.reset()
            return None
        em = self._emission(cur, heading, sigma, moving)
        back = np.full(len(cur["seg"]), -1)
        if self._cands is None:
            logp = em
        else:
            prev, plogp = self._cands, self._logp
            reach_cache = {}
            logp = np.full(len(cur["seg"]), -np.inf)
            for j in range(len(cur["seg"])):
                best, arg = -np.inf, -1
                for i in range(len(prev["seg"])):
                    if not np.isfinite(plogp[i]):
                        continue
                    rd = self._route_dist(prev, i, cur, j, reach_cache)
                    if not np.isfinite(rd):
                        continue
                    v = plogp[i] - abs(rd - self._travel) / self.beta
                    if v > best:
                        best, arg = v, i
                logp[j] = best + em[j]
                back[j] = arg
            if not np.any(np.isfinite(logp)):   # HMM break (off-map / big jump): restart
                logp = em
                back[:] = -1
        logp = logp - np.max(logp)
        post = np.exp(logp)
        post /= post.sum()
        self._cands, self._logp = cur, np.log(np.maximum(post, 1e-300))
        self.history.append((t, cur, back.copy(), post.copy()))
        self._last_t = t
        self._travel = 0.0

        k = int(np.argmax(post))
        s = int(cur["seg"][k])
        self.last = MatchResult(seg=s, xy=cur["xy"][k].copy(), road_heading=float(cur["head"][k]),
                                dist=float(cur["dist"][k]), confidence=float(post[k]),
                                t_on_seg=float(cur["t"][k]), seg_len=float(self.net.seg_len[s]),
                                tunnel=bool(self.net.seg_tunnel[s]), half_width=float(self.net.seg_half_width[s]))
        return self.last

    def viterbi_path(self):
        """Offline smoothing: back-track the most likely road sequence over history."""
        if not self.history:
            return []
        out = []
        k = int(np.argmax(self.history[-1][3]))
        for idx in range(len(self.history) - 1, -1, -1):
            t, c, back, _ = self.history[idx]
            out.append((t, int(c["seg"][k]), c["xy"][k].copy()))
            if idx > 0:
                k = int(back[k]) if back[k] >= 0 else int(np.argmax(self.history[idx - 1][3]))
        return out[::-1]
