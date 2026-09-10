#!/usr/bin/env python3
"""Pure longitudinal speed safety helpers shared by runtime and tests."""

from __future__ import annotations

import math
from typing import Optional


def boost_clearance_for_speed(max_speed: float) -> float:
    """Sensor-local clearance needed before enabling a high straight speed."""
    speed = max(0.0, float(max_speed))
    return max(3.5, min(8.4, 2.0 + 2.5 * speed))


def clearance_speed_limit(
    front: Optional[float],
    max_speed: float,
    cruise_speed: float = 0.52,
    legacy_boost: float = 0.62,
) -> float:
    """Return the discrete straight-line tier allowed by forward clearance.

    The tiers deliberately do not interpolate.  A 2 m/s target therefore
    falls back through 1.4, 1.0 and 0.62 m/s as the visible stopping corridor
    gets shorter instead of dropping directly from 2.0 to the crawl speed.
    ``max_speed`` remains an absolute cap for lower-speed experiments.
    """
    maximum = max(0.0, float(max_speed))
    cruise = min(maximum, max(0.0, float(cruise_speed)))
    middle = min(maximum, max(cruise, float(legacy_boost)))
    if front is None or not math.isfinite(front):
        return cruise
    front = max(0.0, float(front))
    for required, tier in ((7.0, 2.0), (5.5, 1.4), (4.5, 1.0)):
        if front >= required:
            return min(maximum, max(cruise, tier))
    if front >= 3.5:
        return middle
    return cruise


def straight_boost_allowed(
    localization_healthy: bool,
    yaw_error: float,
    center_error: float,
    narrow_front: Optional[float],
    max_yaw_error: float = 0.08,
    max_center_error: float = 0.30,
    min_narrow_front: float = 1.60,
) -> bool:
    """Whether a pose/scan is safe enough to use a speed above crawl.

    ``narrow_front`` is intentionally distinct from the broad high-percentile
    clearance used for tier selection.  This prevents an otherwise open lobby
    from hiding a compact obstacle directly on the commanded line.
    """
    if not bool(localization_healthy):
        return False
    if not math.isfinite(float(yaw_error)) or abs(float(yaw_error)) >= max_yaw_error:
        return False
    if not math.isfinite(float(center_error)) or abs(float(center_error)) >= max_center_error:
        return False
    if narrow_front is None or not math.isfinite(float(narrow_front)):
        return False
    return float(narrow_front) >= float(min_narrow_front)


def longitudinal_slew(
    previous: float,
    target: float,
    dt: float,
    acceleration: float = 0.7,
    deceleration: float = 2.5,
    immediate_zero: bool = True,
) -> float:
    """Apply asymmetric acceleration/deceleration without overshooting target."""
    previous = float(previous)
    target = float(target)
    dt = max(0.0, float(dt))
    if immediate_zero and abs(target) <= 1e-6:
        return 0.0
    if dt <= 0.0:
        return previous
    same_direction = previous == 0.0 or target == 0.0 or previous * target > 0.0
    increasing = same_direction and abs(target) > abs(previous)
    rate = max(0.0, float(acceleration if increasing else deceleration))
    maximum_delta = rate * dt
    delta = target - previous
    if delta > maximum_delta:
        return previous + maximum_delta
    if delta < -maximum_delta:
        return previous - maximum_delta
    return target
