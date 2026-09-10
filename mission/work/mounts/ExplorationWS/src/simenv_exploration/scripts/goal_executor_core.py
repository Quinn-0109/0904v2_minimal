#!/usr/bin/env python3
"""Pure geometry and safety helpers for the lightweight exploration executor."""

import math


def clamp(value, lower, upper):
    return max(lower, min(upper, float(value)))


def angle_difference(target, current):
    return math.atan2(math.sin(target - current), math.cos(target - current))


def quaternion_yaw(x, y, z, w):
    """Return yaw without requiring ROS/tf in unit tests."""
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)


def command_for_goal(x, y, yaw, goal_x, goal_y, *,
                     goal_tolerance=0.30, minimum_speed=0.20,
                     maximum_speed=0.28, distance_gain=0.40,
                     heading_gain=0.35, maximum_yaw_rate=0.20,
                     lateral_speed_limit=None,
                     heading_slowdown_threshold=0.45,
                     turning_speed_limit=None):
    """Compute a bounded body-frame holonomic command for the official RL mode.

    The vector magnitude, rather than each component, is bounded.  This keeps
    diagonal motion inside the requested 0.2--0.3 m/s exploration envelope.
    """
    values = (x, y, yaw, goal_x, goal_y)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("pose_or_goal_nonfinite")
    dx = float(goal_x) - float(x)
    dy = float(goal_y) - float(y)
    distance = math.hypot(dx, dy)
    if distance < float(goal_tolerance):
        return 0.0, 0.0, 0.0, distance, 0.0

    target_heading = math.atan2(dy, dx)
    heading_error = angle_difference(target_heading, float(yaw))
    speed = clamp(float(distance_gain) * distance,
                  float(minimum_speed), float(maximum_speed))
    # A large body-frame heading error turns an otherwise fast map-frame
    # corridor command into a fast lateral command.  That was the common
    # precursor to FAST-LIO correspondence loss at the far room portals.
    # Preserve the cruise envelope on aligned straight segments, but complete
    # the turn at a conservative translational speed before crossing a door.
    if (turning_speed_limit is not None and
            abs(heading_error) > abs(float(heading_slowdown_threshold))):
        turning_limit = abs(float(turning_speed_limit))
        if math.isfinite(turning_limit):
            speed = min(speed, turning_limit)
    linear_x = speed * math.cos(heading_error)
    linear_y = speed * math.sin(heading_error)
    # The policy can accept holonomic commands, but a fast sideways command
    # while simultaneously rotating is the least constrained motion for
    # FAST-LIO.  Preserve fast forward corridor travel while bounding that
    # short door/turn transition.
    if lateral_speed_limit is not None:
        lateral_limit = abs(float(lateral_speed_limit))
        if math.isfinite(lateral_limit):
            linear_y = clamp(linear_y, -lateral_limit, lateral_limit)
    angular_z = clamp(float(heading_gain) * heading_error,
                      -abs(float(maximum_yaw_rate)),
                      abs(float(maximum_yaw_rate)))
    return linear_x, linear_y, angular_z, distance, heading_error


def slew(previous, target, dt, acceleration=0.20, deceleration=0.60):
    """Rate-limit one signed velocity component and allow safe zeroing."""
    previous, target = float(previous), float(target)
    dt = max(0.0, float(dt))
    same_direction = previous == 0.0 or target == 0.0 or previous * target > 0.0
    growing = same_direction and abs(target) > abs(previous)
    limit = abs(float(acceleration if growing else deceleration)) * dt
    return previous + clamp(target - previous, -limit, limit)


class ProgressWatchdog:
    """Detect a goal that is not getting measurably closer."""

    def __init__(self, timeout=15.0, minimum_progress=0.10):
        self.timeout = float(timeout)
        self.minimum_progress = float(minimum_progress)
        self.reference_distance = None
        self.reference_time = None

    def reset(self, now, distance):
        self.reference_time = float(now)
        self.reference_distance = float(distance)

    def update(self, now, distance):
        now, distance = float(now), float(distance)
        if self.reference_time is None:
            self.reset(now, distance)
            return False
        if self.reference_distance - distance >= self.minimum_progress:
            self.reset(now, distance)
            return False
        return now - self.reference_time >= self.timeout
