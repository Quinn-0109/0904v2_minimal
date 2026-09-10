#!/usr/bin/env python3
"""Pure helpers for conservative endpoint-only safe-goal sampling."""

import math
from collections import Counter

import numpy as np

from structured_topology_core import interpolate_polyline


LEVELS = ((0.6, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5))
PATH_SPACING = 0.075
FOOTPRINT_LENGTH = 0.70
FOOTPRINT_WIDTH = 0.44
FOOTPRINT_MARGIN = 0.05
MIN_OBSTACLE_CLEARANCE = 0.05


def footprint_points(point, yaw, length=FOOTPRINT_LENGTH,
                     width=FOOTPRINT_WIDTH, margin=FOOTPRINT_MARGIN,
                     step=0.05):
    """Yield the same filled oriented rectangle used by footprint_safe."""
    half_l = 0.5 * float(length) + float(margin)
    half_w = 0.5 * float(width) + float(margin)
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    for forward in np.arange(-half_l, half_l + 0.5 * step, step):
        for lateral in np.arange(-half_w, half_w + 0.5 * step, step):
            yield (float(point[0]) + c * forward - s * lateral,
                   float(point[1]) + s * forward + c * lateral)


def footprint_state_counts(grid, point, yaw, margin=FOOTPRINT_MARGIN):
    step = max(0.04, min(grid.resolution * 0.5, 0.08))
    counts = Counter()
    for sample in footprint_points(point, yaw, margin=margin, step=step):
        state = grid.state(sample)
        if state < 0:
            counts["unknown"] += 1
        elif state != 0:
            counts["occupied"] += 1
        else:
            counts["free"] += 1
    counts["total"] = counts["free"] + counts["unknown"] + counts["occupied"]
    return counts


class ConsecutiveFootprintGate:
    """Count fully known-free footprint observations on distinct grids only."""

    def __init__(self, required=3):
        self.required = int(required)
        self.last_token = None
        self.consecutive = 0

    def observe(self, token, footprint_counts):
        if token == self.last_token:
            return self.consecutive
        self.last_token = token
        fully_free = (footprint_counts.get("total", 0) > 0 and
                      footprint_counts.get("unknown", 0) == 0 and
                      footprint_counts.get("occupied", 0) == 0)
        self.consecutive = self.consecutive + 1 if fully_free else 0
        return self.consecutive


def warmup_ready(*, odometry_valid, locomotion_ready, distinct_grid_sequences,
                 grid_elapsed_s, pose_age_s, map_age_s,
                 consecutive_full_footprints, minimum_sequences=10,
                 minimum_elapsed_s=2.0, maximum_age_s=2.0,
                 required_consecutive=3):
    return bool(
        odometry_valid and locomotion_ready and
        int(distinct_grid_sequences) >= int(minimum_sequences) and
        float(grid_elapsed_s) >= float(minimum_elapsed_s) and
        pose_age_s is not None and float(pose_age_s) <= float(maximum_age_s) and
        map_age_s is not None and float(map_age_s) <= float(maximum_age_s) and
        int(consecutive_full_footprints) >= int(required_consecutive))


def make_clearance_function(grid):
    occupied = np.argwhere(grid.data >= 50)

    def clearance(point):
        cell = grid.world_to_cell(point)
        if cell is None:
            return 0.0
        if not len(occupied):
            return math.inf
        dx = (occupied[:, 1] - cell[0]) * grid.resolution
        dy = (occupied[:, 0] - cell[1]) * grid.resolution
        return float(np.sqrt(dx * dx + dy * dy).min())

    return clearance


def evaluate_direct_candidate(grid, start, robot_yaw, point,
                              spacing=PATH_SPACING,
                              minimum_clearance=MIN_OBSTACLE_CLEARANCE,
                              clearance_at=None,
                              allow_initial_overlap_unknown=False):
    """Validate the straight, fixed-yaw motion actually executed by the controller."""
    distance = math.hypot(float(point[0]) - float(start[0]),
                          float(point[1]) - float(start[1]))
    item = {"x": float(point[0]), "y": float(point[1]), "distance": distance,
            "accepted": False, "reason": None, "path": [start, point],
            "path_length": distance}
    cell = grid.world_to_cell(point)
    if cell is None:
        item["reason"] = "outside_map"
        return item
    endpoint_state = grid.state(point)
    if endpoint_state < 0:
        item["reason"] = "endpoint_unknown"
        return item
    if endpoint_state != 0:
        item["reason"] = "endpoint_occupied"
        return item
    item["endpoint_known_free"] = True

    samples = interpolate_polyline([start, point], spacing)
    states = [grid.state(sample) for sample in samples]
    if any(state < 0 for state in states):
        item["reason"] = "path_unknown"
        return item
    if any(state != 0 for state in states):
        item["reason"] = "path_occupied"
        return item
    item["connected"] = True

    clearance_at = clearance_at or make_clearance_function(grid)
    worst_clearance = math.inf
    for index, sample in enumerate(samples):
        counts = footprint_state_counts(grid, sample, robot_yaw)
        prefix = "start" if index == 0 else "swept"
        if counts["unknown"]:
            unknown_outside_initial = True
            if allow_initial_overlap_unknown:
                c, s = math.cos(robot_yaw), math.sin(robot_yaw)
                half_l = 0.5 * FOOTPRINT_LENGTH + FOOTPRINT_MARGIN
                half_w = 0.5 * FOOTPRINT_WIDTH + FOOTPRINT_MARGIN
                unknown_outside_initial = False
                for footprint_sample in footprint_points(
                        sample, robot_yaw,
                        step=max(0.04, min(grid.resolution * 0.5, 0.08))):
                    if grid.state(footprint_sample) >= 0:
                        continue
                    dx = footprint_sample[0] - start[0]
                    dy = footprint_sample[1] - start[1]
                    forward = c * dx + s * dy
                    lateral = -s * dx + c * dy
                    if abs(forward) > half_l + 1e-9 or abs(lateral) > half_w + 1e-9:
                        unknown_outside_initial = True
                        break
            if unknown_outside_initial:
                item["reason"] = prefix + "_footprint_unknown"
                return item
            item["initial_overlap_unknown_allowed"] = True
        if counts["occupied"]:
            item["reason"] = prefix + "_footprint_occupied"
            return item
        for footprint_sample in footprint_points(
                sample, robot_yaw, margin=0.0,
                step=max(0.04, min(grid.resolution * 0.5, 0.08))):
            worst_clearance = min(worst_clearance, clearance_at(footprint_sample))
    item["footprint_safe"] = True
    item["minimum_clearance"] = worst_clearance
    if worst_clearance + 1e-9 < float(minimum_clearance):
        item["reason"] = "insufficient_clearance"
        return item

    target_angle = math.atan2(point[1] - start[1], point[0] - start[0])
    item["heading_error"] = abs(math.atan2(
        math.sin(target_angle - robot_yaw), math.cos(target_angle - robot_yaw)))
    item["accepted"] = True
    item["reason"] = "accepted"
    return item


def sample_safe_goal(grid, start, robot_yaw, trajectory=(), levels=LEVELS):
    rejected = Counter()
    level_records, all_candidates = [], []
    selected = None
    maximum_safe_distance = 0.0
    clearance_at = make_clearance_function(grid)
    for level_index, (low, high) in enumerate(levels):
        counts = {key: 0 for key in (
            "candidate_generated", "candidate_in_known_free", "candidate_connected",
            "candidate_after_footprint_check", "candidate_after_clearance_check")}
        accepted = []
        for angle_index in range(72):
            angle = -math.pi + angle_index * 2.0 * math.pi / 72.0
            for radius in np.arange(high, low - 1e-6, -0.10):
                point = (start[0] + float(radius) * math.cos(angle),
                         start[1] + float(radius) * math.sin(angle))
                item = evaluate_direct_candidate(
                    grid, start, robot_yaw, point, clearance_at=clearance_at)
                item["level"] = level_index
                counts["candidate_generated"] += 1
                if item.get("endpoint_known_free"):
                    counts["candidate_in_known_free"] += 1
                if item.get("connected"):
                    counts["candidate_connected"] += 1
                if item.get("footprint_safe"):
                    counts["candidate_after_footprint_check"] += 1
                if item["accepted"]:
                    counts["candidate_after_clearance_check"] += 1
                    maximum_safe_distance = max(maximum_safe_distance, item["distance"])
                    item["trajectory_overlap"] = sum(
                        math.hypot(item["x"] - old[0], item["y"] - old[1]) < 0.30
                        for old in trajectory)
                    accepted.append(item)
                else:
                    rejected[item["reason"]] += 1
                all_candidates.append(item)
        level_records.append({"range_m": [low, high], **counts,
                              "search_status": "searched"})
        if accepted:
            selected = max(accepted, key=lambda item: (
                item["minimum_clearance"], -item["heading_error"],
                item["distance"], -item["trajectory_overlap"]))
            for later_low, later_high in levels[level_index + 1:]:
                level_records.append({
                    "range_m": [later_low, later_high],
                    **{key: 0 for key in counts},
                    "search_status": "not_searched_after_selection"})
            break
    return level_records, rejected, all_candidates, selected, maximum_safe_distance
