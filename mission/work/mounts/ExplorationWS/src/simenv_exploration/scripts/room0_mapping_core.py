#!/usr/bin/env python3
"""Pure geometry and control helpers for the Room0 mapping diagnostic."""

import json
import math
import os


def clamp(value, low, high):
    return max(low, min(high, value))


def angle_diff(target, source):
    return math.atan2(math.sin(target - source), math.cos(target - source))


def load_room(layout_file, room_id="floor_0_room_0"):
    with open(layout_file, encoding="utf-8") as stream:
        document = json.load(stream)
    for floor in document.get("floors", []):
        for room in floor.get("rooms", []):
            if room.get("id") == room_id:
                return floor, room
    raise ValueError("room not found: {}".format(room_id))


def _obstacle_half_extents(item):
    if "size" in item and item["size"]:
        return 0.5 * float(item["size"][0]), 0.5 * float(item["size"][1])
    radius = float(item.get("radius", 0.15))
    return radius, radius


def load_room_obstacles(layout_file, room, room_id="floor_0_room_0"):
    """Furniture plus seed danger/distractors inside the room."""
    obstacles = []
    for item in room.get("furniture", []):
        cx, cy = map(float, item["pose"][:2])
        half_x, half_y = _obstacle_half_extents(item)
        obstacles.append({
            "id": item.get("id", "furniture"),
            "x": cx,
            "y": cy,
            "half_x": half_x,
            "half_y": half_y,
        })
    danger_path = os.path.join(os.path.dirname(layout_file), "danger_truth.json")
    if not os.path.isfile(danger_path):
        return obstacles
    with open(danger_path, encoding="utf-8") as stream:
        danger_doc = json.load(stream)
    for bucket in ("danger_sources", "distraction_sources"):
        for item in danger_doc.get(bucket, []):
            if item.get("room_id") != room_id:
                continue
            cx, cy = map(float, item["position"][:2])
            half_x, half_y = _obstacle_half_extents(item)
            obstacles.append({
                "id": item.get("model_name", bucket),
                "x": cx,
                "y": cy,
                "half_x": half_x,
                "half_y": half_y,
            })
    return obstacles


def _clearance_to_obstacles(x, y, obstacles):
    if not obstacles:
        return float("inf")
    return min(
        max(abs(x - item["x"]) - item["half_x"], abs(y - item["y"]) - item["half_y"])
        for item in obstacles
    )


def _leg_min_clearance(x, y_min, y_max, obstacles, footprint_clearance, door_y=None):
    worst = float("inf")
    y = y_min
    while y <= y_max + 1e-9:
        worst = min(worst, _clearance_to_obstacles(x, y, obstacles) - footprint_clearance)
        y += 0.20
    if door_y is not None:
        worst = min(
            worst,
            _clearance_to_obstacles(x, float(door_y), obstacles) - footprint_clearance,
        )
    return worst


def _choose_vertical_leg(bounds, door_y, obstacles, footprint_clearance, x_lo, x_hi):
    y_min = float(bounds["y_min"]) + 1.20
    y_max = float(bounds["y_max"]) - 1.20
    best_x = 0.5 * (x_lo + x_hi)
    best_score = -1e9
    x = min(x_lo, x_hi)
    x_end = max(x_lo, x_hi)
    while x <= x_end + 1e-9:
        min_clear = _leg_min_clearance(
            x, y_min, y_max, obstacles, footprint_clearance, door_y
        )
        if min_clear >= 0.08 and min_clear > best_score + 1e-9:
            best_score = min_clear
            best_x = x
        x += 0.05
    return best_x, best_score


def generate_room0_route(layout_file, room_id="floor_0_room_0"):
    floor, room = load_room(layout_file, room_id)
    bounds = room["bounds"]
    door_x, door_y = map(float, room["door_pose"][:2])
    corridor = floor["corridor_bounds"]
    corridor_x = 0.5 * (float(corridor["x_min"]) + float(corridor["x_max"]))
    obstacles = load_room_obstacles(layout_file, room, room_id)
    footprint = 0.45

    x_min = float(bounds["x_min"]) + footprint + 0.10
    x_max = float(bounds["x_max"]) - footprint - 0.10
    mid_x = 0.5 * (x_min + x_max)
    # Seed-77 puts distractor_red_box_04 between the east wall and the old
    # x=door_x-0.55 squeeze; walking that lane stalls classic Trotting.
    # Pick separate east/west legs that clear furniture + clutter.
    right_x, right_clear = _choose_vertical_leg(
        bounds, door_y, obstacles, footprint, mid_x, x_max
    )
    left_x, left_clear = _choose_vertical_leg(
        bounds, door_y, obstacles, footprint, x_min, mid_x
    )
    if right_clear < 0.08 or left_clear < 0.08 or abs(right_x - left_x) < 2.0:
        raise ValueError("unable to find a clear Room0 loop for {}".format(room_id))

    lower_y = float(bounds["y_min"]) + 1.20
    upper_y = float(bounds["y_max"]) - 1.20
    for _ in range(40):
        if min(
            _clearance_to_obstacles(right_x, lower_y, obstacles),
            _clearance_to_obstacles(left_x, lower_y, obstacles),
        ) - footprint >= 0.08:
            break
        lower_y += 0.05
    for _ in range(40):
        if min(
            _clearance_to_obstacles(right_x, upper_y, obstacles),
            _clearance_to_obstacles(left_x, upper_y, obstacles),
        ) - footprint >= 0.08:
            break
        upper_y -= 0.05

    points = [
        (corridor_x, 0.8, "entrance", "corridor"),
        (corridor_x, door_y - 1.2, "door_approach", "corridor"),
        (corridor_x, door_y, "door_align", "door"),
        (door_x + 0.45, door_y, "door_crossing_outside", "door"),
        (door_x - 0.75, door_y, "door_crossing_inside", "door"),
        (right_x, door_y, "loop_start", "room"),
        (right_x, lower_y, "lower_right", "room"),
        (left_x, lower_y, "lower_left", "room"),
        (left_x, upper_y, "upper_left", "room"),
        (right_x, upper_y, "upper_right", "room"),
        (right_x, door_y, "loop_closed", "room"),
        (door_x - 0.65, door_y, "exit_inside", "door"),
        (door_x + 0.50, door_y, "exit_outside", "door"),
    ]
    route = [
        {"x": float(x), "y": float(y), "name": name, "zone": zone}
        for x, y, name, zone in points
    ]
    validate_route(route, room, footprint_clearance=footprint, obstacles=obstacles)
    return route, room, floor


def generate_room0_small_loop_route(layout_file, room_id="floor_0_room_0"):
    """Seed-77 diagnostic route: enter Room0, make one central safe loop, stop."""
    floor, room = load_room(layout_file, room_id)
    door_x, door_y = map(float, room["door_pose"][:2])
    corridor = floor["corridor_bounds"]
    corridor_x = 0.5 * (float(corridor["x_min"]) + float(corridor["x_max"]))
    obstacles = load_room_obstacles(layout_file, room, room_id)
    footprint = 0.45
    points = [
        (corridor_x, 0.8, "entrance", "corridor"),
        (corridor_x, door_y - 1.2, "door_approach", "corridor"),
        (corridor_x, door_y, "door_align", "door"),
        (door_x + 0.45, door_y, "door_crossing_outside", "door"),
        (door_x - 0.75, door_y, "door_crossing_inside", "door"),
        (-4.5, door_y, "loop_start", "room"),
        (-4.5, 13.5, "small_lower_right", "room"),
        (-6.0, 13.5, "small_lower_left", "room"),
        (-6.0, 16.0, "small_upper_left", "room"),
        (-4.5, 16.0, "small_upper_right", "room"),
        (-4.5, door_y, "loop_closed", "room"),
    ]
    route = [
        {"x": float(x), "y": float(y), "name": name, "zone": zone}
        for x, y, name, zone in points
    ]
    validate_route(route, room, footprint_clearance=footprint, obstacles=obstacles)
    return route, room, floor


def generate_room0_coverage_loop_route(layout_file, room_id="floor_0_room_0"):
    """Seed-77 safe loop extended to observe Room0 west and north walls."""
    floor, room = load_room(layout_file, room_id)
    bounds = room["bounds"]
    door_x, door_y = map(float, room["door_pose"][:2])
    corridor = floor["corridor_bounds"]
    corridor_x = 0.5 * (float(corridor["x_min"]) + float(corridor["x_max"]))
    obstacles = load_room_obstacles(layout_file, room, room_id)
    footprint = 0.45
    west_x = float(bounds["x_min"]) + 0.70
    north_y = float(bounds["y_max"]) - 1.38
    points = [
        (corridor_x, 0.8, "entrance", "corridor"),
        (corridor_x, door_y - 1.2, "door_approach", "corridor"),
        (corridor_x, door_y, "door_align", "door"),
        (door_x + 0.45, door_y, "door_crossing_outside", "door"),
        (door_x - 0.75, door_y, "door_crossing_inside", "door"),
        (-4.5, door_y, "loop_start", "room"),
        (-4.5, 13.5, "small_lower_right", "room"),
        (-6.0, 13.5, "small_lower_left", "room"),
        (west_x, 13.5, "coverage_west_entry", "room"),
        (west_x, north_y, "coverage_west_north", "room"),
        (-4.5, north_y, "coverage_north_east", "room"),
        (-4.5, door_y, "loop_closed", "room"),
    ]
    route = [
        {"x": float(x), "y": float(y), "name": name, "zone": zone}
        for x, y, name, zone in points
    ]
    validate_route(route, room, footprint_clearance=footprint, obstacles=obstacles)
    return route, room, floor


def _segment_distance(point, start, end):
    px, py = point
    ax, ay = start
    bx, by = end
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return math.hypot(px - ax, py - ay)
    ratio = clamp(((px - ax) * dx + (py - ay) * dy) / denom, 0.0, 1.0)
    return math.hypot(px - (ax + ratio * dx), py - (ay + ratio * dy))


def validate_route(route, room, footprint_clearance=0.45, obstacles=None):
    """Reject a diagnostic loop that intersects a wall, furniture, or clutter."""
    if len(route) < 8:
        raise ValueError("route is too short")
    bounds = room["bounds"]
    interior_names = {
        "loop_start", "lower_right", "lower_left", "upper_left",
        "upper_right", "loop_closed", "small_lower_right",
        "small_lower_left", "small_upper_left", "small_upper_right",
        "coverage_west_entry", "coverage_west_north", "coverage_north_east",
    }
    for waypoint in route:
        if waypoint["name"] not in interior_names:
            continue
        if not (
            float(bounds["x_min"]) + footprint_clearance <= waypoint["x"]
            <= float(bounds["x_max"]) - footprint_clearance
            and float(bounds["y_min"]) + footprint_clearance <= waypoint["y"]
            <= float(bounds["y_max"]) - footprint_clearance
        ):
            raise ValueError("waypoint lacks wall clearance: {}".format(waypoint["name"]))

    if obstacles is None:
        obstacles = []
        for item in room.get("furniture", []):
            cx, cy = map(float, item["pose"][:2])
            half_x, half_y = _obstacle_half_extents(item)
            obstacles.append({
                "id": item.get("id", "furniture"),
                "x": cx,
                "y": cy,
                "half_x": half_x,
                "half_y": half_y,
            })

    loop = [w for w in route if w["name"] in interior_names]
    segments = list(zip(loop, loop[1:]))
    for item in obstacles:
        half_x = item["half_x"] + footprint_clearance
        half_y = item["half_y"] + footprint_clearance
        cx, cy = item["x"], item["y"]
        for first, second in segments:
            if abs(first["x"] - second["x"]) < 1e-6:
                x = first["x"]
                low, high = sorted((first["y"], second["y"]))
                hit = (
                    cx - half_x <= x <= cx + half_x
                    and high >= cy - half_y
                    and low <= cy + half_y
                )
            elif abs(first["y"] - second["y"]) < 1e-6:
                y = first["y"]
                low, high = sorted((first["x"], second["x"]))
                hit = (
                    cy - half_y <= y <= cy + half_y
                    and high >= cx - half_x
                    and low <= cx + half_x
                )
            else:
                hit = _segment_distance(
                    (cx, cy), (first["x"], first["y"]), (second["x"], second["y"])
                ) <= max(half_x, half_y)
            if hit:
                raise ValueError(
                    "route intersects expanded obstacle: {}".format(item["id"])
                )
    return True


def control_for_target(x, y, yaw, target, zone="room", forward_limits=None):
    """Return classic-Trotting Joy forward and yaw axes."""
    dx, dy = float(target[0]) - x, float(target[1]) - y
    distance = math.hypot(dx, dy)
    heading_error = angle_diff(math.atan2(dy, dx), yaw)
    if forward_limits is None:
        forward_limits = {"corridor": 0.50, "door": 0.32, "room": 0.42}
    max_forward = forward_limits.get(zone, 0.35)
    abs_heading = abs(heading_error)
    if abs_heading > 0.12:
        # Pure yaw above this threshold; classic trot tips if translation is
        # mixed with a large heading error near walls/obstacles.
        forward = 0.0
        yaw_axis = -clamp(0.90 * heading_error, -0.40, 0.40)
    else:
        heading_scale = max(0.20, 1.0 - abs_heading / 0.12)
        distance_scale = clamp(distance / 1.0, 0.25, 1.0)
        forward = max_forward * heading_scale * distance_scale
        yaw_axis = -clamp(0.55 * heading_error, -0.08, 0.08)
    return forward, yaw_axis, distance, heading_error


def rl_control_for_target(x, y, yaw, target, zone="room", speed_limits=None,
                          maximum_yaw_rate=0.35, turning=False,
                          arc_turning=False):
    """Return metric Twist commands for the official RL ``/cmd_vel`` mode.

    The diagnostic route deliberately turns in place before translating.  It
    keeps the learned locomotion policy inside the low-speed indoor envelope
    while leaving acceleration limiting to the sole cmd_vel safety gate.
    """
    dx, dy = float(target[0]) - x, float(target[1]) - y
    distance = math.hypot(dx, dy)
    heading_error = angle_diff(math.atan2(dy, dx), yaw)
    if speed_limits is None:
        speed_limits = {"corridor": 0.35, "door": 0.20, "room": 0.25}
    maximum = float(speed_limits.get(zone, 0.20))
    yaw_limit = abs(float(maximum_yaw_rate))
    # Use hysteresis around the route bearing.  The supplied plane policy
    # reacts much more strongly to yaw than a classical velocity controller;
    # toggling at one threshold repeatedly resets the linear acceleration
    # ramp.  Enter turning at 0.18 rad and remain there down to 0.06 rad.
    turn_threshold = 0.06 if turning else 0.18
    if abs(heading_error) > turn_threshold:
        if arc_turning:
            # The official stair policy does not support reliable in-place
            # rotation. Keep it in its learned walking band and bend the path
            # with a conservative simultaneous yaw command.
            arc_linear = min(maximum, 0.30)
            arc_yaw = clamp(0.50 * heading_error, -yaw_limit, yaw_limit)
            return arc_linear, arc_yaw, distance, heading_error
        return 0.0, clamp(0.70 * heading_error, -yaw_limit, yaw_limit), distance, heading_error
    if arc_turning:
        # The stair policy collapses to a standing fixed point below roughly
        # 0.30 m/s. The caller checks waypoint tolerance before commanding,
        # so retain the learned gait speed until the target is accepted.
        return maximum, 0.0, distance, heading_error
    distance_scale = clamp(distance / 1.0, 0.20, 1.0)
    return maximum * distance_scale, 0.0, distance, heading_error


def rl_holonomic_control_for_target(x, y, yaw, target, zone="room",
                                     speed_limits=None):
    """Return body-frame vx/vy for the official holonomic stair policy."""
    dx, dy = float(target[0]) - x, float(target[1]) - y
    distance = math.hypot(dx, dy)
    heading_error = angle_diff(math.atan2(dy, dx), yaw) if distance > 1e-9 else 0.0
    if speed_limits is None:
        speed_limits = {"corridor": 0.35, "door": 0.30, "room": 0.32}
    speed = float(speed_limits.get(zone, 0.30))
    if distance <= 1e-9:
        return 0.0, 0.0, distance, heading_error
    # World displacement rotated into the current body frame. The supplied
    # policy is trained with both command[0] (forward) and command[1]
    # (lateral), so sharp room turns do not require unsupported in-place yaw.
    forward = (math.cos(yaw) * dx + math.sin(yaw) * dy) / distance
    lateral = (-math.sin(yaw) * dx + math.cos(yaw) * dy) / distance
    return speed * forward, speed * lateral, distance, heading_error


def rl_guard_command(desired_linear, desired_yaw, *, armed, desired_age,
                     pose_age, z, roll, pitch, aborted=False,
                     stability_hold=False,
                     warning_tilt=0.22, abort_tilt=0.35,
                     minimum_height=0.27, maximum_linear=0.35,
                     maximum_yaw=0.35):
    """Pure safety decision used by the Room0 RL cmd_vel gate."""
    if aborted:
        return 0.0, 0.0, "abort_latched", True
    if not armed:
        return 0.0, 0.0, "not_armed", False
    if desired_age > 0.25:
        return 0.0, 0.0, "desired_stale", False
    if pose_age > 0.30:
        return 0.0, 0.0, "pose_stale", True
    if not all(math.isfinite(value) for value in (z, roll, pitch)):
        return 0.0, 0.0, "pose_nonfinite", True
    if z < minimum_height:
        return 0.0, 0.0, "body_too_low", True
    if max(abs(roll), abs(pitch)) > abort_tilt:
        return 0.0, 0.0, "tilt_abort", True
    if stability_hold:
        return 0.0, 0.0, "stability_hold", False
    linear = clamp(float(desired_linear), -maximum_linear, maximum_linear)
    yaw = clamp(float(desired_yaw), -maximum_yaw, maximum_yaw)
    if max(abs(roll), abs(pitch)) > warning_tilt:
        linear = clamp(linear, -0.10, 0.10)
        yaw = clamp(yaw, -0.15, 0.15)
        return linear, yaw, "tilt_limited", False
    return linear, yaw, "normal", False


def is_fallen(z, roll, pitch):
    return float(z) < 0.18 or abs(float(roll)) > 1.0 or abs(float(pitch)) > 1.0


def is_unstable(z, roll, pitch):
    return float(z) < 0.27 or abs(float(roll)) > 0.22 or abs(float(pitch)) > 0.22


def route_progress_error(x, y, route, active_index):
    if not route:
        return float("inf")
    index = max(0, min(int(active_index), len(route) - 1))
    start = (x, y) if index == 0 else (route[index - 1]["x"], route[index - 1]["y"])
    end = (route[index]["x"], route[index]["y"])
    return _segment_distance((x, y), start, end)
