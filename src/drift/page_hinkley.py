"""Page-Hinkley change detector (Page, 1954; Hinkley, 1971).

For a stream z_1, z_2, ... we track, since the last reset,

    zbar_t = (1/t) * sum_{i<=t} z_i                         (running mean)
    m_t    = sum_{i<=t} (z_i - zbar_i - delta)               (cumulative deviation)
    M_t    = min_{i<=t} m_i
    PH_t   = m_t - M_t                                       (test statistic)

and raise an alarm for an **increase** in the mean when PH_t > threshold.
For a **decrease** we use the mirrored statistic
m'_t = sum (zbar_i - z_i - delta),  PH'_t = m'_t - min m'.
``direction='both'`` monitors both.

``delta`` is the magnitude of change that is tolerated (it makes the
statistic drift downwards under no change), ``threshold`` (often called
lambda) trades detection delay against false alarms.  No alarm is raised
during the first ``min_instances`` observations after a reset.  After an
alarm the detector resets itself.

The detector also reports an *estimated change onset*: the time at which the
cumulative sum attained its minimum M_t, i.e. the last time before which the
stream still looked like the pre-change regime.  DA-ACP uses this to decide
which residuals are "pre-drift".

The detector is scale dependent; DA-ACP therefore feeds it residuals divided
by a scale estimated on the calibration split, so that ``delta`` and
``threshold`` are expressed in units of the typical calibration residual.
"""
from __future__ import annotations

from dataclasses import dataclass

import math


@dataclass
class DriftEvent:
    t: int            # step at which the alarm was raised
    onset: int        # estimated change onset (step index)
    direction: str    # 'increase' | 'decrease'
    statistic: float


class PageHinkley:
    def __init__(self, delta: float = 0.1, threshold: float = 50.0, min_instances: int = 30,
                 direction: str = "increase") -> None:
        delta, threshold = float(delta), float(threshold)  # also rejects non-numeric strings early
        if direction not in ("increase", "decrease", "both"):
            raise ValueError("direction must be 'increase', 'decrease' or 'both'")
        if threshold <= 0 or delta < 0:
            raise ValueError("threshold must be > 0 and delta >= 0")
        if not math.isfinite(threshold) or not math.isfinite(delta):
            raise ValueError("threshold and delta must be finite")
        if min_instances < 1:
            raise ValueError("min_instances must be >= 1")
        self.delta = float(delta)
        self.threshold = float(threshold)
        self.min_instances = int(min_instances)
        self.direction = direction
        self.t = -1  # global step counter (index of the last processed observation)
        self.events: list[DriftEvent] = []
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m_up = 0.0
        self.min_up = 0.0
        self.argmin_up = self.t + 1
        self.m_down = 0.0
        self.min_down = 0.0
        self.argmin_down = self.t + 1
        self.stat = 0.0

    def update(self, z: float) -> DriftEvent | None:
        """Process one observation; return a DriftEvent if an alarm is raised."""
        z = float(z)
        if not math.isfinite(z):
            raise ValueError("Page-Hinkley observations must be finite")
        self.t += 1
        self.n += 1
        self.mean += (z - self.mean) / self.n
        self.m_up += z - self.mean - self.delta
        self.m_down += self.mean - z - self.delta
        if self.m_up < self.min_up:
            self.min_up, self.argmin_up = self.m_up, self.t + 1
        if self.m_down < self.min_down:
            self.min_down, self.argmin_down = self.m_down, self.t + 1
        ph_up = self.m_up - self.min_up
        ph_down = self.m_down - self.min_down
        self.stat = {"increase": ph_up, "decrease": ph_down, "both": max(ph_up, ph_down)}[self.direction]
        if self.n < self.min_instances:
            return None
        event = None
        if self.direction in ("increase", "both") and ph_up > self.threshold:
            event = DriftEvent(self.t, self.argmin_up, "increase", ph_up)
        elif self.direction in ("decrease", "both") and ph_down > self.threshold:
            event = DriftEvent(self.t, self.argmin_down, "decrease", ph_down)
        if event is not None:
            self.events.append(event)
            self.reset()
        return event
