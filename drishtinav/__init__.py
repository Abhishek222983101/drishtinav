"""DrishtiNav - AI-ML Intelligent Dead Reckoning with GNSS fusion (ISRO, Smart India Hackathon 2026, PS 26168).

Edge-deployable engine. Typical use::

    from drishtinav import NavEngine, GnssFix, make_config
    from drishtinav.models.speednet import SpeedNetRuntime
    from drishtinav.roadnet import RoadNetwork

    eng = NavEngine(make_config("smartphone"), SpeedNetRuntime(), RoadNetwork.load("data/osm/delhi_central.json"))
    state = eng.step(t, accel_xyz, gyro_xyz, GnssFix(lat, lon, speed, course, accuracy) or None)
"""
from .config import make_config
from .engine import GnssFix, NavEngine, NavState

__all__ = ["NavEngine", "NavState", "GnssFix", "make_config"]
__version__ = "1.0.0"
