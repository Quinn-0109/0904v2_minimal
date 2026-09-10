#!/usr/bin/env python3
"""ROS-free parallel-wall corridor semantics from an online occupancy grid."""

import math

import numpy as np


def _angle_difference_unoriented(a, b):
    value = abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)
    return min(value, math.pi - value)


def detect_parallel_wall_corridor(grid, robot_xy, search_radius=8.0,
                                  minimum_width=1.0, maximum_width=4.0,
                                  minimum_length=4.0, angle_step_degrees=5.0,
                                  wall_bin_size=0.20):
    """Return the best long, balanced, known-free strip near ``robot_xy``.

    The detector uses only occupied/free/unknown raster geometry.  It does not
    use a building layout, room labels, Gazebo pose, or fixed world direction.
    """
    occupied_y, occupied_x = np.nonzero(grid.data >= 50)
    if occupied_x.size < 20:
        return None
    world_x = grid.origin_x + (occupied_x.astype(float) + 0.5) * grid.resolution
    world_y = grid.origin_y + (occupied_y.astype(float) + 0.5) * grid.resolution
    relative = np.column_stack((world_x - robot_xy[0], world_y - robot_xy[1]))
    mask = np.sum(relative * relative, axis=1) <= search_radius * search_radius
    points = relative[mask]
    if len(points) < 20:
        return None
    # Cap pathological dense maps without biasing one direction.
    if len(points) > 5000:
        points = points[::int(math.ceil(len(points) / 5000.0))]

    best = None
    for angle_deg in np.arange(0.0, 180.0, angle_step_degrees):
        angle = math.radians(float(angle_deg))
        axis = np.asarray([math.cos(angle), math.sin(angle)])
        normal = np.asarray([-axis[1], axis[0]])
        longitudinal = points.dot(axis)
        lateral = points.dot(normal)
        bins = np.floor(lateral / wall_bin_size).astype(int)
        wall_lines = []
        for bin_id in np.unique(bins):
            selected = longitudinal[bins == bin_id]
            if selected.size < 5:
                continue
            low, high = np.percentile(selected, (5.0, 95.0))
            if high - low < minimum_length:
                continue
            lateral_values = lateral[bins == bin_id]
            wall_lines.append((float(np.median(lateral_values)),
                               float(low), float(high), int(selected.size)))
        for index, left in enumerate(wall_lines):
            for right in wall_lines[index + 1:]:
                width = abs(right[0] - left[0])
                if not minimum_width <= width <= maximum_width:
                    continue
                overlap_low = max(left[1], right[1])
                overlap_high = min(left[2], right[2])
                overlap = overlap_high - overlap_low
                if overlap < minimum_length:
                    continue
                center_lateral = 0.5 * (left[0] + right[0])
                # Verify that the space between the walls is actually known
                # free, rather than accepting two unrelated parallel walls.
                samples = max(8, int(overlap / max(0.25, grid.resolution)))
                free_count = 0
                sample_count = 0
                for along in np.linspace(overlap_low, overlap_high, samples):
                    world = (np.asarray(robot_xy) + center_lateral * normal +
                             along * axis)
                    cell = grid.world_to_cell(world)
                    if cell is None:
                        continue
                    sample_count += 1
                    if int(grid.data[cell[1], cell[0]]) == 0:
                        free_count += 1
                free_ratio = free_count / max(1, sample_count)
                if sample_count < 6 or free_ratio < 0.60:
                    continue
                clamped_t = min(max(0.0, overlap_low), overlap_high)
                approach_relative = center_lateral * normal + clamped_t * axis
                approach_distance = float(np.linalg.norm(approach_relative))
                balance = min(left[3], right[3]) / max(left[3], right[3])
                score = overlap * (0.65 + 0.35 * balance) * free_ratio
                score -= 0.25 * approach_distance
                inside = (overlap_low <= 0.0 <= overlap_high and
                          abs(center_lateral) <= 0.5 * width + 0.25)
                positive_room = overlap_high
                negative_room = -overlap_low
                direction = axis if positive_room >= negative_room else -axis
                candidate = {
                    "axis": axis.tolist(),
                    "direction": direction.tolist(),
                    "heading": math.atan2(direction[1], direction[0]),
                    "center_lateral": center_lateral,
                    "centerline_point": (np.asarray(robot_xy) +
                                         center_lateral * normal).tolist(),
                    "approach_point": (np.asarray(robot_xy) +
                                       approach_relative).tolist(),
                    "width": width,
                    "visible_length": overlap,
                    "free_ratio": free_ratio,
                    "wall_balance": balance,
                    "robot_inside": bool(inside),
                    "score": score,
                }
                if best is None or candidate["score"] > best["score"]:
                    best = candidate
    return best


def corridor_observations_match(first, second, maximum_angle_degrees=15.0,
                                maximum_width_difference=0.65):
    if first is None or second is None:
        return False
    first_angle = math.atan2(first["axis"][1], first["axis"][0])
    second_angle = math.atan2(second["axis"][1], second["axis"][0])
    return (_angle_difference_unoriented(first_angle, second_angle) <=
            math.radians(maximum_angle_degrees) and
            abs(first["width"] - second["width"]) <= maximum_width_difference)


def score_path_for_corridor(points, robot_xy, corridor):
    """Rank an existing path; never create or alter a waypoint."""
    if not points or corridor is None:
        return 0.0
    robot = np.asarray(robot_xy, dtype=float)
    path = np.asarray(points, dtype=float)
    if corridor.get("robot_inside"):
        desired = np.asarray(corridor["direction"], dtype=float)
        displacement = path[min(len(path) - 1, 4)] - robot
        norm = float(np.linalg.norm(displacement))
        alignment = float(np.dot(displacement / max(norm, 1e-9), desired))
        return 3.0 * alignment + 0.25 * norm
    approach = np.asarray(corridor["approach_point"], dtype=float)
    initial_distance = float(np.linalg.norm(robot - approach))
    distances = np.linalg.norm(path - approach, axis=1)
    progress = initial_distance - float(np.min(distances))
    displacement = path[min(len(path) - 1, 4)] - robot
    norm = float(np.linalg.norm(displacement))
    toward = approach - robot
    toward_norm = float(np.linalg.norm(toward))
    alignment = float(np.dot(displacement, toward) /
                      max(1e-9, norm * toward_norm))
    return 3.0 * progress + alignment + 0.15 * norm
