"""Offline road network (OpenStreetMap extract) with a spatial index and routing.

Loaded from the compact JSON written by ``scripts/fetch_osm.py``. Everything is
in a local East/North frame so projections are plain 2-D vector math, cheap
enough to run at 10 Hz on a phone.
"""
from __future__ import annotations

import heapq
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from .geo import LocalFrame, wrap_angle

ROAD_HALF_WIDTH = {  # metres, typical carriageway half-width by OSM class
    "motorway": 7.0, "trunk": 7.0, "primary": 6.0, "secondary": 5.0, "tertiary": 4.5,
    "unclassified": 3.5, "residential": 3.5, "living_street": 3.0,
}


class RoadNetwork:
    def __init__(self, data: dict, frame: Optional[LocalFrame] = None, cell: float = 50.0):
        self.name = data.get("name", "roads")
        nodes = np.asarray(data["nodes"], dtype=float)
        if frame is None:
            s, w, n, e = data["bbox"]
            frame = LocalFrame((s + n) / 2, (w + e) / 2)
        self.frame = frame
        ex, ny = frame.to_en(nodes[:, 0], nodes[:, 1])
        self.node_xy = np.column_stack([ex, ny])

        a_idx, b_idx, way_of, oneway, tunnel, hw, names = [], [], [], [], [], [], []
        self.ways = data["ways"]
        for wi, w in enumerate(self.ways):
            ns = w["nodes"]
            for a, b in zip(ns[:-1], ns[1:]):
                if w["oneway"] == -1:
                    a, b = b, a
                a_idx.append(a); b_idx.append(b); way_of.append(wi)
                oneway.append(w["oneway"] != 0); tunnel.append(bool(w["tunnel"])); hw.append(w["hw"])
        self.seg_a = np.asarray(a_idx, dtype=np.int64)
        self.seg_b = np.asarray(b_idx, dtype=np.int64)
        self.seg_way = np.asarray(way_of, dtype=np.int64)
        self.seg_oneway = np.asarray(oneway, dtype=bool)
        self.seg_tunnel = np.asarray(tunnel, dtype=bool)
        self.seg_hw = np.asarray(hw)
        p, q = self.node_xy[self.seg_a], self.node_xy[self.seg_b]
        self.seg_p = p
        self.seg_d = q - p
        self.seg_len = np.hypot(self.seg_d[:, 0], self.seg_d[:, 1])
        self.seg_heading = np.arctan2(self.seg_d[:, 1], self.seg_d[:, 0])
        self.seg_half_width = np.array([ROAD_HALF_WIDTH.get(h.replace("_link", ""), 3.5) for h in hw])

        # Directed adjacency on nodes (for routing / HMM transitions).
        self.adj = defaultdict(list)          # node -> [(node, seg, length)]
        self.seg_next = defaultdict(list)     # seg -> segments leaving its end node (same direction)
        out_by_node = defaultdict(list)
        for s in range(len(self.seg_a)):
            a, b, L = int(self.seg_a[s]), int(self.seg_b[s]), float(self.seg_len[s])
            self.adj[a].append((b, s, L))
            out_by_node[a].append((s, +1))
            if not self.seg_oneway[s]:
                self.adj[b].append((a, s, L))
                out_by_node[b].append((s, -1))
        self._out_by_node = out_by_node

        # Uniform-grid spatial index over segment bounding boxes.
        self.cell = cell
        self.grid = defaultdict(list)
        lo = np.minimum(p, q) // cell
        hi = np.maximum(p, q) // cell
        for s in range(len(self.seg_a)):
            for gx in range(int(lo[s, 0]), int(hi[s, 0]) + 1):
                for gy in range(int(lo[s, 1]), int(hi[s, 1]) + 1):
                    self.grid[(gx, gy)].append(s)

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path, frame: Optional[LocalFrame] = None) -> "RoadNetwork":
        return cls(json.loads(Path(path).read_text()), frame)

    def reframe(self, frame: LocalFrame) -> "RoadNetwork":
        """Same network expressed in another local frame (e.g. a drive's origin)."""
        lat, lon = self.frame.to_ll(self.node_xy[:, 0], self.node_xy[:, 1])
        data = {"name": self.name, "nodes": np.column_stack([lat, lon]).tolist(), "ways": self.ways,
                "bbox": [float(lat.min()), float(lon.min()), float(lat.max()), float(lon.max())]}
        return RoadNetwork(data, frame, self.cell)

    # ---------------------------------------------------------------- geometry
    def candidates(self, xy, radius: float):
        """Segments within ``radius`` of point xy -> (seg_ids, proj_xy, dist, t)."""
        gx0, gy0 = int((xy[0] - radius) // self.cell), int((xy[1] - radius) // self.cell)
        gx1, gy1 = int((xy[0] + radius) // self.cell), int((xy[1] + radius) // self.cell)
        ids = set()
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                ids.update(self.grid.get((gx, gy), ()))
        if not ids:
            return np.empty(0, int), np.empty((0, 2)), np.empty(0), np.empty(0)
        ids = np.fromiter(ids, dtype=np.int64)
        return self.project(xy, ids, radius)

    def project(self, xy, ids, radius=np.inf):
        p, d, L = self.seg_p[ids], self.seg_d[ids], self.seg_len[ids]
        t = np.clip(((xy[0] - p[:, 0]) * d[:, 0] + (xy[1] - p[:, 1]) * d[:, 1]) / np.maximum(L * L, 1e-9), 0, 1)
        proj = p + t[:, None] * d
        dist = np.hypot(proj[:, 0] - xy[0], proj[:, 1] - xy[1])
        keep = dist <= radius
        return ids[keep], proj[keep], dist[keep], t[keep]

    def heading_error(self, seg_ids, heading):
        """Angle between travel heading and each segment, honouring one-way roads."""
        dh = np.abs(wrap_angle(heading - self.seg_heading[seg_ids]))
        two_way = ~self.seg_oneway[seg_ids]
        dh[two_way] = np.minimum(dh[two_way], np.pi - dh[two_way])
        return dh

    # ----------------------------------------------------------------- routing
    def nearest_node(self, xy):
        return int(np.argmin(np.hypot(*(self.node_xy - np.asarray(xy)).T)))

    def shortest_path(self, src: int, dst: int, max_dist: float = np.inf, weight=None):
        """Dijkstra on the directed graph. Returns (node list, seg list, length)."""
        dist, prev = {src: 0.0}, {}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == dst:
                break
            if d > dist.get(u, np.inf) or d > max_dist:
                continue
            for v, s, L in self.adj[u]:
                w = L * (weight(s) if weight else 1.0)
                nd = d + w
                if nd < dist.get(v, np.inf):
                    dist[v], prev[v] = nd, (u, s)
                    heapq.heappush(pq, (nd, v))
        if dst not in dist:
            return None, None, np.inf
        nodes, segs, u = [dst], [], dst
        while u != src:
            u, s = prev[u]
            nodes.append(u); segs.append(s)
        return nodes[::-1], segs[::-1], dist[dst]

    def reachable_segments(self, seg: int, direction: int, max_dist: float):
        """Segments reachable from the end of ``seg`` (travelling ``direction``) within max_dist.

        Returns {seg_id: route distance from the end of the start segment}.
        """
        end = int(self.seg_b[seg] if direction > 0 else self.seg_a[seg])
        out = {}
        dist = {end: 0.0}
        pq = [(0.0, end)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, np.inf) or d > max_dist:
                continue
            for s, _ in self._out_by_node[u]:
                if s not in out or d < out[s]:
                    out[s] = d
            for v, s, L in self.adj[u]:
                nd = d + L
                if nd < dist.get(v, np.inf) and nd <= max_dist:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return out

    def polyline(self, segs, nodes) -> np.ndarray:
        return self.node_xy[np.asarray(nodes)]

    def to_geojson(self, max_ways: Optional[int] = None) -> dict:
        lat, lon = self.frame.to_ll(self.node_xy[:, 0], self.node_xy[:, 1])
        feats = []
        for w in self.ways[:max_ways]:
            feats.append({"type": "Feature",
                          "properties": {"hw": w["hw"], "tunnel": w["tunnel"], "name": w["name"]},
                          "geometry": {"type": "LineString",
                                       "coordinates": [[round(float(lon[i]), 6), round(float(lat[i]), 6)] for i in w["nodes"]]}})
        return {"type": "FeatureCollection", "features": feats}
