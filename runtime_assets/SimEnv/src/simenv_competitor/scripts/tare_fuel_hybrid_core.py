#!/usr/bin/env python3
"""ROS-free policy helpers for persistent TARE/FUEL arbitration."""

from dataclasses import dataclass, field
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Point = Tuple[float, float, float]


def planar_distance(first, second):
    return math.hypot(float(first[0]) - float(second[0]),
                      float(first[1]) - float(second[1]))


def wrap_angle(value):
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def bearing(origin, target):
    return math.atan2(float(target[1]) - float(origin[1]),
                      float(target[0]) - float(origin[0]))


def angle_difference(first, second):
    return abs(wrap_angle(float(first) - float(second)))


def forward_candidate_points(pose, yaw, distances=(1.5, 1.2, 0.9),
                             offsets=(0.0, math.radians(15.0),
                                      -math.radians(15.0))):
    """Body-forward local candidates, ordered by straightness then distance."""
    points = []
    for offset in offsets:
        heading = float(yaw) + float(offset)
        for distance in distances:
            points.append((
                float(pose[0]) + float(distance) * math.cos(heading),
                float(pose[1]) + float(distance) * math.sin(heading),
                float(pose[2]) if len(pose) > 2 else 0.0))
    return points


def select_safe_forward_candidate(pose, yaw, planner, occupancy_grid,
                                  coverage_query=None,
                                  maximum_path_ratio=1.35,
                                  maximum_coverage=0.65):
    """Return the first short body-forward point with a near-direct safe path.

    The helper deliberately proposes a navigation goal rather than a velocity:
    the normal A*/FAR/SCAN-lite execution chain remains responsible for motion.
    """
    for point in forward_candidate_points(pose, yaw):
        result = planner.plan(pose, point, occupancy_grid)
        direct = planar_distance(pose, point)
        coverage = (0.0 if coverage_query is None else
                    float(coverage_query(point)))
        if (result.success and direct > 1e-6 and
                result.travel_cost <= float(maximum_path_ratio) * direct and
                coverage < float(maximum_coverage)):
            return point
    return None


def repeated_goal_region(samples, radius, required_count, minimum_span=0.0):
    required_count = max(2, int(required_count))
    if len(samples) < required_count:
        return False
    recent = list(samples[-required_count:])
    if float(recent[-1][0]) - float(recent[0][0]) < float(minimum_span):
        return False
    points = [item[1] for item in recent]
    return max(planar_distance(a, b) for a in points for b in points) <= float(radius)


def clustered_pose_stall(samples, now, window, radius, minimum_samples=4):
    cutoff = float(now) - float(window)
    recent = [item for item in samples if float(item[0]) >= cutoff]
    if len(recent) < int(minimum_samples):
        return False
    if float(recent[-1][0]) - float(recent[0][0]) < 0.85 * float(window):
        return False
    origin = recent[0][1]
    return max(planar_distance(origin, item[1]) for item in recent) <= float(radius)


def valid_fuel_goal(payload, pose, minimum_distance, maximum_distance):
    if not isinstance(payload, dict):
        return None
    position = payload.get("position")
    if not isinstance(position, (list, tuple)) or len(position) < 2:
        return None
    try:
        point = (float(position[0]), float(position[1]),
                 float(position[2]) if len(position) > 2 else float(pose[2]))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in point):
        return None
    distance = planar_distance(point, pose)
    if distance < float(minimum_distance) or distance > float(maximum_distance):
        return None
    if payload.get("reachable") is False or payload.get("free_space") is False:
        return None
    return point


@dataclass
class FrontierIntent:
    endpoint: Point
    frontier: Optional[Tuple[float, float]] = None
    gain: float = 0.0
    source: str = "unknown"
    stamp: float = 0.0
    completed: bool = False
    door_id: Optional[str] = None

    @property
    def direction(self):
        return None if self.frontier is None else bearing(self.endpoint, self.frontier)

    def to_dict(self):
        return {
            "endpoint": list(self.endpoint),
            "frontier": None if self.frontier is None else list(self.frontier),
            "gain": self.gain, "source": self.source, "stamp": self.stamp,
            "completed": self.completed, "door_id": self.door_id,
        }


@dataclass
class PersistentCoverageMemory:
    endpoint_radius: float = 1.0
    frontier_radius: float = 1.5
    direction_tolerance: float = math.radians(35.0)
    trajectory_spacing: float = 0.45
    observed_cell_size: float = 0.45
    trajectory: List[Point] = field(default_factory=list)
    intents: List[FrontierIntent] = field(default_factory=list)
    observed_free_first_seen: Dict[Tuple[int, int], int] = field(default_factory=dict)

    def add_pose(self, point: Sequence[float]):
        pose = (float(point[0]), float(point[1]),
                float(point[2]) if len(point) > 2 else 0.0)
        if not self.trajectory or planar_distance(self.trajectory[-1], pose) >= self.trajectory_spacing:
            self.trajectory.append(pose)
            return True
        return False

    def add_intent(self, intent: FrontierIntent):
        self.intents.append(intent)

    def observe_free_cells(self, points):
        """Record when an online occupancy cell first became LiDAR-observed."""
        index = len(self.trajectory)
        scale = max(float(self.observed_cell_size), 1e-3)
        for point in points:
            key = (int(round(float(point[0]) / scale)),
                   int(round(float(point[1]) / scale)))
            if key not in self.observed_free_first_seen:
                self.observed_free_first_seen[key] = index

    def observed_coverage(self, point, radius=1.35, minimum_age_points=10):
        """Fraction of a goal neighbourhood observed before the recent path."""
        if not self.observed_free_first_seen:
            return 0.0
        scale = max(float(self.observed_cell_size), 1e-3)
        cx = int(round(float(point[0]) / scale))
        cy = int(round(float(point[1]) / scale))
        cells = max(1, int(math.ceil(float(radius) / scale)))
        cutoff = len(self.trajectory) - max(1, int(minimum_age_points))
        total = old = 0
        for dx in range(-cells, cells + 1):
            for dy in range(-cells, cells + 1):
                if math.hypot(dx * scale, dy * scale) > float(radius):
                    continue
                total += 1
                first_seen = self.observed_free_first_seen.get((cx + dx, cy + dy))
                if first_seen is not None and first_seen <= cutoff:
                    old += 1
        return float(old) / float(max(total, 1))

    def trajectory_distance(self, point):
        if not self.trajectory:
            return float("inf")
        return min(planar_distance(point, item) for item in self.trajectory)

    def frontier_novelty(self, frontier):
        if frontier is None:
            return 1.0
        previous = [item.frontier for item in self.intents if item.frontier is not None]
        if not previous:
            return 1.0
        distance = min(planar_distance(frontier, item) for item in previous)
        return min(1.0, distance / max(self.frontier_radius * 2.0, 1e-3))

    def repeat_reason(self, candidate: FrontierIntent, maximum_gain: float,
                      strict_visited_endpoint: bool = False):
        near_trajectory = self.trajectory_distance(candidate.endpoint) < self.endpoint_radius
        low_gain = candidate.gain <= max(1e-6, 0.35 * float(maximum_gain))
        opposite_intent = False
        for old in reversed(self.intents):
            if planar_distance(old.endpoint, candidate.endpoint) >= self.endpoint_radius:
                continue
            if old.frontier is None or candidate.frontier is None:
                if low_gain:
                    return "covered_endpoint_low_gain"
                continue
            old_direction, new_direction = old.direction, candidate.direction
            if angle_difference(old_direction, new_direction) >= math.radians(100.0):
                opposite_intent = True
                continue
            same_frontier = planar_distance(old.frontier, candidate.frontier) < self.frontier_radius
            same_direction = angle_difference(old_direction, new_direction) < self.direction_tolerance
            if (same_frontier or same_direction) and low_gain:
                return "repeated_frontier_intent"
        # Bootstrap must leave already traversed lobby endpoints even when a
        # distant unknown frontier makes the raw gain artificially enormous.
        # An explicit opposite-side intent remains eligible.
        if strict_visited_endpoint and near_trajectory and not opposite_intent:
            return "bootstrap_visited_endpoint"
        if near_trajectory and low_gain and not opposite_intent:
            return "covered_trajectory_low_gain"
        return None

    def to_dict(self):
        return {
            "trajectory": [list(item) for item in self.trajectory],
            "intents": [item.to_dict() for item in self.intents],
            "observed_free_cell_count": len(self.observed_free_first_seen),
            "observed_cell_size": self.observed_cell_size,
        }


def candidate_intent(candidate, score=None, source="fuel", stamp=0.0):
    score = score or {}
    position = candidate.get("position") or score.get("position")
    frontier = candidate.get("frontier_center") or score.get("frontier_center")
    if not position or len(position) < 2:
        return None
    endpoint = (float(position[0]), float(position[1]),
                float(position[2]) if len(position) > 2 else 0.0)
    frontier_point = None
    if frontier and len(frontier) >= 2:
        frontier_point = (float(frontier[0]), float(frontier[1]))
    return FrontierIntent(
        endpoint=endpoint, frontier=frontier_point,
        gain=float(score.get("information_gain", candidate.get("information_gain", 0.0))),
        source=source, stamp=float(stamp), door_id=candidate.get("door_id"))


def intent_key(intent):
    """Stable spatial key used by the temporary SCAN-lite blacklist."""
    frontier = intent.frontier or (99999.0, 99999.0)
    return (round(intent.endpoint[0], 1), round(intent.endpoint[1], 1),
            round(frontier[0], 1), round(frontier[1], 1))


def pop_next_eligible(queue, blacklist, now):
    """Pop the next candidate, skipping intents rejected in this time window."""
    while queue:
        selected = queue.pop(0)
        if blacklist.get(intent_key(selected["intent"]), -1e9) <= float(now):
            return selected
    return None


def _normalize(values, reverse=False):
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low <= 1e-9:
        result = [0.5] * len(values)
    else:
        result = [(value - low) / (high - low) for value in values]
    return [1.0 - value for value in result] if reverse else result


def rank_fuel_candidates(candidates: Iterable[dict], scores: Iterable[dict],
                         pose: Sequence[float], memory: PersistentCoverageMemory,
                         minimum_distance=0.45, maximum_distance=3.2,
                         strict_visited_endpoint=False, now=0.0,
                         include_rejected=False):
    """Merge FUEL artifacts and apply persistent/topological arbitration."""
    score_by_id = {int(item.get("candidate_id", item.get("goal_id", -1))): item
                   for item in scores}
    rows = []
    for candidate in candidates:
        identifier = int(candidate.get("id", -1))
        score = score_by_id.get(identifier, {})
        intent = candidate_intent(candidate, score, stamp=now)
        if intent is None:
            continue
        distance = planar_distance(intent.endpoint, pose)
        if not (float(minimum_distance) <= distance <= float(maximum_distance)):
            continue
        if candidate.get("reachable") is False:
            continue
        rows.append({"candidate": dict(candidate), "score_payload": dict(score),
                     "intent": intent, "distance": distance,
                     "base": float(score.get("score", score.get("final_score", 0.0))),
                     "gain": intent.gain,
                     "unknown": float(score.get("local_unknown_volume", 0.0)),
                     "topology": max(0.0, min(1.0, float(candidate.get("topology_bonus", 0.0)))),
                     "coverage": memory.observed_coverage(intent.endpoint),
                     "door_id": candidate.get("door_id")})
    if not rows:
        return []
    maximum_gain = max(item["gain"] for item in rows)
    base_n = _normalize([item["base"] for item in rows])
    unknown_n = _normalize([item["unknown"] for item in rows])
    distance_n = _normalize([item["distance"] for item in rows])
    for index, row in enumerate(rows):
        row["novelty"] = memory.frontier_novelty(row["intent"].frontier)
        row["repeat_reason"] = memory.repeat_reason(
            row["intent"], maximum_gain, strict_visited_endpoint)
        if (row["repeat_reason"] is None and row["coverage"] >= 0.65 and
                row["topology"] < 0.80):
            row["repeat_reason"] = "historical_lidar_coverage"
        row["components"] = {
            "fuel": base_n[index], "frontier_novelty": row["novelty"],
            "unknown": unknown_n[index], "topology": row["topology"],
            "distance": distance_n[index], "covered": row["coverage"],
        }
        row["final_score"] = (
            0.30 * base_n[index] + 0.25 * row["novelty"]
            + 0.15 * unknown_n[index] + 0.40 * row["topology"]
            - 0.15 * distance_n[index] - 0.35 * row["coverage"])
    eligible = [item for item in rows if item["repeat_reason"] is None]
    eligible.sort(key=lambda item: item["final_score"], reverse=True)
    if not include_rejected:
        return eligible
    rejected = [item for item in rows if item["repeat_reason"] is not None]
    rejected.sort(key=lambda item: item["final_score"], reverse=True)
    return eligible + rejected


def point_in_corridor(point, corridor, margin=0.15):
    if corridor is None or not corridor.centerline:
        return False
    start, end = corridor.centerline[0], corridor.centerline[-1]
    vx, vy = end[0] - start[0], end[1] - start[1]
    length2 = vx * vx + vy * vy
    if length2 <= 1e-9:
        return False
    t = ((float(point[0]) - start[0]) * vx +
         (float(point[1]) - start[1]) * vy) / length2
    if t < -0.1 or t > 1.1:
        return False
    projection = (start[0] + t * vx, start[1] + t * vy)
    return planar_distance(point, projection) <= 0.5 * corridor.estimated_width - margin


def corridor_entry_ready(point, corridor, bootstrap_origin,
                         minimum_progress=2.0):
    """Require physical progress so an open lobby is not an entered corridor."""
    if bootstrap_origin is None:
        return False
    return (getattr(corridor, "confirmed", False) and
            planar_distance(point, bootstrap_origin) >= float(minimum_progress) and
            point_in_corridor(point, corridor))


def bilateral_corridor_evidence(corridor, minimum_wall_span=3.5,
                                maximum_side_imbalance=0.45):
    """Validate the two-sided, long and approximately symmetric corridor cue."""
    if (corridor is None or not getattr(corridor, "confirmed", False) or
            not 1.0 <= float(corridor.estimated_width) <= 4.0 or
            float(corridor.forward_extent) < 4.0 or
            float(corridor.confidence) < 0.60):
        return False
    left = list(getattr(corridor, "left_wall", []) or [])
    right = list(getattr(corridor, "right_wall", []) or [])
    centerline = list(getattr(corridor, "centerline", []) or [])
    if len(left) < 3 or len(right) < 3 or len(centerline) < 2:
        return False
    start, end = centerline[0], centerline[-1]
    vx, vy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(vx, vy)
    if length < 1e-6:
        return False
    ux, uy = vx / length, vy / length

    def projections(points):
        return [(float(p[0]) - start[0]) * ux +
                (float(p[1]) - start[1]) * uy for p in points]

    left_p, right_p = projections(left), projections(right)
    left_span = max(left_p) - min(left_p)
    right_span = max(right_p) - min(right_p)
    imbalance = abs(left_span - right_span) / max(left_span, right_span, 1e-6)
    return (left_span >= float(minimum_wall_span) and
            right_span >= float(minimum_wall_span) and
            imbalance <= float(maximum_side_imbalance))
