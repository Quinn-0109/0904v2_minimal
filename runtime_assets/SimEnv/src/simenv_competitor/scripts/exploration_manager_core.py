#!/usr/bin/env python3
"""Pure bookkeeping helpers for the Step 2.4 exploration manager."""

import math


def coverage_from_statistics(statistics):
    occupied = max(0, int(statistics.get("occupied_voxel_count", 0)))
    free = max(0, int(statistics.get("free_voxel_count", 0)))
    unknown = max(0, int(statistics.get("unknown_voxel_count", 0)))
    total = occupied + free + unknown
    return 0.0 if total <= 0 else float(occupied + free) / float(total)


def visited_area_from_statistics(statistics):
    """Return the documented free-voxel 2-D area proxy in square metres."""
    resolution = float(statistics.get("resolution", 0.15))
    free = max(0, int(statistics.get("free_voxel_count", 0)))
    return free * resolution * resolution


def termination_reason(*, frontier_count, coverage_ratio, max_information_gain,
                       stagnant_seconds, coverage_threshold=0.95,
                       information_gain_threshold=20.0,
                       stagnation_threshold=30.0):
    if frontier_count is not None and int(frontier_count) == 0:
        return "frontier_empty"
    if float(coverage_ratio) >= float(coverage_threshold):
        return "coverage_reached"
    if (max_information_gain is not None and
            float(max_information_gain) < float(information_gain_threshold)):
        return "information_gain_below_threshold"
    if float(stagnant_seconds) >= float(stagnation_threshold):
        return "map_free_voxel_stagnant"
    return None


def is_duplicate_goal(position, history, radius=0.35):
    x, y = float(position[0]), float(position[1])
    return any(math.hypot(x - float(item[0]), y - float(item[1])) < float(radius)
               for item in history)


def mark_visited_disk(cells, x, y, radius=1.0, resolution=0.25):
    """Mark only the trajectory-swept disk; remote LiDAR free cells stay unvisited."""
    radius_cells = max(1, int(math.ceil(float(radius) / float(resolution))))
    center_x = int(round(float(x) / float(resolution)))
    center_y = int(round(float(y) / float(resolution)))
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if dx * dx + dy * dy <= radius_cells * radius_cells:
                cells.add((center_x + dx, center_y + dy))
    return cells


def confirmed_frontier_exhaustion(*, empty_cycles, empty_seconds,
                                  visited_stagnant_seconds,
                                  large_connected_unknown_remains,
                                  required_cycles=3, required_seconds=30.0,
                                  stagnation_seconds=30.0):
    """A single empty plan is never a normal exploration completion."""
    return (int(empty_cycles) >= int(required_cycles) and
            float(empty_seconds) >= float(required_seconds) and
            float(visited_stagnant_seconds) >= float(stagnation_seconds) and
            not bool(large_connected_unknown_remains))
