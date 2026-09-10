#!/usr/bin/env python3
"""Pure geometry and safety helpers for the lightweight exploration executor."""

import math


def clamp(value, lower, upper):
    return max(lower, min(upper, float(value)))


def bounded_dynamic_minimum_speed(requested, absolute_floor, dynamic_cap):
    """Clamp a per-goal speed floor without reinstating the cruise floor.

    Doorway crossing deliberately asks for a lower executable speed than the
    corridor cruise minimum.  Treating the nominal cruise value as the lower
    bound made the 0.18 m/s portal request a no-op and forced the A1 through
    the final jamb waypoint at 0.65 m/s.
    """
    values = (requested, absolute_floor, dynamic_cap)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("dynamic_minimum_speed_nonfinite")
    floor = max(0.01, float(absolute_floor))
    cap = max(floor, float(dynamic_cap))
    return clamp(float(requested), floor, cap)


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


def navigation_stability_envelope(
        linear_x, linear_y, angular_z, heading_error, previous_command, *,
        large_heading_error=0.55, high_yaw_rate=0.80,
        fast_turn_translation_threshold=0.10,
        braking_yaw_rate=0.55, medium_heading_error=0.25,
        turning_translation_limit=0.75, turning_lateral_limit=0.35):
    """Separate a fast navigation turn from the preceding lateral gait.

    This helper applies only to ordinary goal navigation; sensing rescans have
    their own rotation-only branch in ``goal_executor.py``. A large heading
    correction requesting a high yaw rate first commands translation to zero.
    While the *previous published command* is still translating, yaw is held
    to a stable braking rate. The requested fast yaw is restored once the
    translational command has entered the safe near-zero band.

    More modest turns retain holonomic motion but bound its vector magnitude
    and lateral component. Aligned straight travel and rotation-only sensing
    commands pass through unchanged.
    """
    if len(previous_command) != 3:
        raise ValueError("previous_navigation_command_size")
    values = (linear_x, linear_y, angular_z, heading_error) + \
        tuple(previous_command)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("navigation_command_nonfinite")

    linear_x = float(linear_x)
    linear_y = float(linear_y)
    angular_z = float(angular_z)
    heading_error = abs(float(heading_error))
    previous_speed = math.hypot(
        float(previous_command[0]), float(previous_command[1]))
    large_heading_error = abs(float(large_heading_error))
    high_yaw_rate = abs(float(high_yaw_rate))
    translation_threshold = abs(float(fast_turn_translation_threshold))
    braking_yaw_rate = abs(float(braking_yaw_rate))
    medium_heading_error = abs(float(medium_heading_error))
    translation_limit = abs(float(turning_translation_limit))
    lateral_limit = abs(float(turning_lateral_limit))
    parameters = (
        large_heading_error, high_yaw_rate, translation_threshold,
        braking_yaw_rate, medium_heading_error, translation_limit,
        lateral_limit,
    )
    if not all(math.isfinite(value) for value in parameters):
        raise ValueError("navigation_envelope_parameter_nonfinite")

    if (heading_error >= large_heading_error and
            abs(angular_z) >= high_yaw_rate):
        linear_x = 0.0
        linear_y = 0.0
        if previous_speed > translation_threshold:
            angular_z = clamp(
                angular_z, -braking_yaw_rate, braking_yaw_rate)
            mode = "brake_before_fast_turn"
        else:
            mode = "fast_turn_in_place"
        return linear_x, linear_y, angular_z, mode

    if heading_error >= medium_heading_error and abs(angular_z) > 1e-9:
        original_x, original_y = linear_x, linear_y
        speed = math.hypot(linear_x, linear_y)
        if speed > translation_limit and speed > 1e-9:
            scale = translation_limit / speed
            linear_x *= scale
            linear_y *= scale
        linear_y = clamp(linear_y, -lateral_limit, lateral_limit)
        if (abs(linear_x - original_x) > 1e-9 or
                abs(linear_y - original_y) > 1e-9):
            return (linear_x, linear_y, angular_z,
                    "bounded_combined_turn")

    return linear_x, linear_y, angular_z, "unrestricted"


def enforce_executable_translation_floor(linear_x, linear_y, minimum_speed, *,
                                         enabled):
    """Lift an authorized nonzero translation out of the gait deadband.

    The ROS-facing caller enables this only after a finite EXIT heading gate
    has aligned. Zero therefore remains authoritative for safety stops and
    unaligned doorway crossings. Direction is preserved exactly.
    """
    values = (linear_x, linear_y, minimum_speed)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("translation_floor_nonfinite")
    linear_x = float(linear_x)
    linear_y = float(linear_y)
    minimum_speed = abs(float(minimum_speed))
    speed = math.hypot(linear_x, linear_y)
    if not bool(enabled) or speed <= 1e-9 or speed >= minimum_speed:
        return linear_x, linear_y, False
    scale = minimum_speed / speed
    return linear_x * scale, linear_y * scale, True


def rotation_only_exit_alignment_yaw_rate(
        heading_error, nominal_yaw_rate, minimum_yaw_rate=0.42,
        fast_yaw_rate=0.75, fast_error_threshold=0.70,
        absolute_yaw_rate=0.75):
    """Return a signed EXIT alignment command for the zero-translation gate.

    Large doorway-heading errors can use the proven absolute yaw envelope
    while translation is explicitly stopped. Near the alignment latch the
    ordinary bounded command and physical deadband floor remain in force, so
    the faster in-place phase cannot become a high-yaw portal crossing.
    """
    values = (heading_error, nominal_yaw_rate, minimum_yaw_rate,
              fast_yaw_rate, fast_error_threshold, absolute_yaw_rate)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("exit_alignment_yaw_nonfinite")
    error = float(heading_error)
    if abs(error) <= 1e-9:
        return 0.0
    cap = max(0.0, abs(float(absolute_yaw_rate)))
    minimum = min(cap, max(0.0, abs(float(minimum_yaw_rate))))
    fast = min(cap, max(minimum, abs(float(fast_yaw_rate))))
    nominal = min(cap, max(minimum, abs(float(nominal_yaw_rate))))
    magnitude = (fast if abs(error) >= abs(float(fast_error_threshold))
                 else nominal)
    return math.copysign(magnitude, error)


def pose_from_truth_delta(odom_anchor, truth_anchor, truth_pose):
    """Express live truth displacement in an anchored odometry frame."""
    if len(odom_anchor) != 4 or len(truth_anchor) != 4 or len(truth_pose) != 4:
        raise ValueError("truth_delta_pose_size")
    values = tuple(odom_anchor) + tuple(truth_anchor) + tuple(truth_pose)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("truth_delta_pose_nonfinite")
    rotation = float(odom_anchor[3]) - float(truth_anchor[3])
    c, s = math.cos(rotation), math.sin(rotation)
    dx = float(truth_pose[0]) - float(truth_anchor[0])
    dy = float(truth_pose[1]) - float(truth_anchor[1])
    return (
        float(odom_anchor[0]) + c * dx - s * dy,
        float(odom_anchor[1]) + s * dx + c * dy,
        float(odom_anchor[2]) + float(truth_pose[2]) - float(truth_anchor[2]),
        angle_difference(float(truth_pose[3]) + rotation, 0.0),
    )


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
