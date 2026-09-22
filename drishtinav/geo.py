"""Geodesy helpers: WGS-84 lat/lon <-> local East-North (ENU) tangent plane."""
from __future__ import annotations

import numpy as np

WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


def wrap_angle(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


class LocalFrame:
    """Local tangent-plane (East, North) frame anchored at an origin.

    Uses the meridian / prime-vertical radii of curvature at the origin, which is
    accurate to centimetres over the tens of kilometres a single drive covers.
    """

    def __init__(self, lat0: float, lon0: float):
        self.lat0 = float(lat0)
        self.lon0 = float(lon0)
        phi = np.radians(self.lat0)
        s2 = np.sin(phi) ** 2
        self.r_n = WGS84_A * (1 - WGS84_E2) / (1 - WGS84_E2 * s2) ** 1.5  # meridian
        self.r_e = WGS84_A / np.sqrt(1 - WGS84_E2 * s2)                    # prime vertical
        self.cos_lat0 = np.cos(phi)

    def to_en(self, lat, lon):
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        e = np.radians(lon - self.lon0) * self.r_e * self.cos_lat0
        n = np.radians(lat - self.lat0) * self.r_n
        return e, n

    def to_ll(self, e, n):
        e = np.asarray(e, dtype=float)
        n = np.asarray(n, dtype=float)
        lat = self.lat0 + np.degrees(n / self.r_n)
        lon = self.lon0 + np.degrees(e / (self.r_e * self.cos_lat0))
        return lat, lon

    def to_dict(self):
        return {"lat0": self.lat0, "lon0": self.lon0}


def heading_from_en(de, dn):
    """Navigation heading (rad) measured from East, counter-clockwise (math convention)."""
    return np.arctan2(dn, de)


def compass_to_math(deg):
    """Compass course (deg, clockwise from North) -> math heading (rad, CCW from East)."""
    return wrap_angle(np.radians(90.0 - np.asarray(deg, dtype=float)))


def math_to_compass(rad):
    return (90.0 - np.degrees(rad)) % 360.0
