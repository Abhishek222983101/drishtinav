"""Seamless GNSS deficit handler: GNSS-aided INS <-> dead reckoning.

The navigation solution is produced by the INS mechanisation on *every* IMU
sample; GNSS is only a correction. Losing GNSS therefore never interrupts the
output - the handler just changes which aiding sources the EKF listens to.

Modes
    INIT        waiting for the first usable fix (position + heading)
    GNSS        fixes arriving, accurate, consistent with the INS (chi-square)
    DEGRADED    fixes arriving but poor (accuracy / repeated innovation rejects,
                e.g. urban-canyon multipath): used with inflated noise
    DR          no fix for ``gnss_timeout_s``: AI pseudo-odometer + NHC + ZUPT +
                map constraints only
    RECOVERY    first fixes after an outage: position pulled in with inflated
                noise for ``recovery_fixes`` epochs so the icon glides back
                instead of jumping, then -> GNSS

Transition latency: when the receiver / OS reports the loss (``gnss_lost``), the
switch happens in that same IMU epoch (5 ms at 200 Hz, 100 ms at 10 Hz). Without
such a signal a missing fix can only be noticed after the expected fix interval,
so the timeout adapts to the observed GNSS rate (1.6 x the fix period, never
below ``gnss_timeout_s``). Per-step compute time is reported by the engine
(``latency_us``).
"""
from __future__ import annotations

INIT, GNSS, DEGRADED, DR, RECOVERY = "INIT", "GNSS", "DEGRADED", "DR", "RECOVERY"


class GnssDeficitHandler:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.mode = INIT
        self.last_fix_t = -1e9
        self.recovery_left = 0
        self.consecutive_rejects = 0
        self.transitions = []          # (t, from, to)
        self.outage_start = None
        self.fix_period = None         # EMA of the observed GNSS fix interval (s)

    def _set(self, t, mode):
        if mode != self.mode:
            self.transitions.append((float(t), self.mode, mode))
            if mode == DR:
                self.outage_start = t
            self.mode = mode

    @property
    def timeout(self):
        base = self.cfg["gnss_timeout_s"]
        period = 1.0 if self.fix_period is None else self.fix_period   # assume 1 Hz until measured
        return max(base, 1.6 * period)

    def tick(self, t, has_fix: bool, accuracy: float = None, initialised: bool = True, lost: bool = False):
        """Called every IMU epoch before any GNSS update. Returns the noise inflation to use."""
        cfg = self.cfg
        if has_fix:
            dtf = t - self.last_fix_t
            if 0.05 < dtf < 5.0:
                self.fix_period = dtf if self.fix_period is None else 0.9 * self.fix_period + 0.1 * dtf
            self.last_fix_t = t
        if not initialised:
            return None
        if lost and not has_fix:              # receiver/OS says the signal is gone: switch now
            self.last_fix_t = min(self.last_fix_t, t - self.timeout - 1e-6)
        if not has_fix and t - self.last_fix_t > self.timeout:
            if self.mode != DR:
                self._set(t, DR)
            return None
        if not has_fix:
            return None
        # A fix arrived.
        if self.mode in (DR, INIT):
            self.recovery_left = cfg["recovery_fixes"]
            self._set(t, RECOVERY if self.mode == DR else GNSS)
        poor = accuracy is not None and accuracy > cfg["gnss_max_accuracy_m"]
        if self.mode == RECOVERY:
            self.recovery_left -= 1
            infl = cfg["recovery_sigma_inflate"]
            if self.recovery_left <= 0:
                self._set(t, DEGRADED if poor else GNSS)
            return infl
        if poor or self.consecutive_rejects >= 3:
            self._set(t, DEGRADED)
            return 2.0
        self._set(t, GNSS)
        return 1.0

    def report(self, accepted: bool):
        self.consecutive_rejects = 0 if accepted else self.consecutive_rejects + 1

    @property
    def gnss_trusted(self):
        return self.mode in (GNSS, DEGRADED, RECOVERY)
