#!/usr/bin/env python3
"""ROS-free receding-horizon local exploration path generation."""

from dataclasses import dataclass, field
import math
from typing import Callable, Iterable, List, Optional, Sequence, Set, Tuple


Point = Tuple[float, float, float]
Cell = Tuple[int, int]


def planar_distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.hypot(float(first[0]) - float(second[0]),
                      float(first[1]) - float(second[1]))


def wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def bearing(origin: Sequence[float], target: Sequence[float]) -> float:
    return math.atan2(float(target[1]) - float(origin[1]),
                      float(target[0]) - float(origin[0]))


@dataclass
class LocalPathConfig:
    alpha: float = 1.0
    beta: float = 0.4
    gamma: float = 1.0
    delta: float = 0.5
    eta: float = 1.0
    mu: float = 1.5
    waypoint_spacing: float = 0.75
    waypoint_count: int = 4
    minimum_waypoints: int = 2
    sample_free_search_radius: float = 0.45
    free_clearance: float = 0.30
    sensor_range: float = 3.0
    risk_radius: float = 0.75
    revisit_radius: float = 0.65
    backward_penalty: float = 5.0
    minimum_waypoint_separation: float = 0.40


@dataclass
class LocalWaypoint:
    x: float
    y: float
    yaw: float
    expected_gain: float
    safety_status: str
    unknown_cell_count: int = 0

    def point(self, z: float = 0.0) -> Point:
        return (self.x, self.y, z)

    def to_dict(self) -> dict:
        return {
            "x": self.x, "y": self.y, "yaw": self.yaw,
            "expected_gain": self.expected_gain,
            "unknown_cell_count": self.unknown_cell_count,
            "safety_status": self.safety_status,
        }


@dataclass
class CandidatePath:
    path_id: str
    heading: float
    source: str
    active_region_id: Optional[str]
    waypoints: List[LocalWaypoint] = field(default_factory=list)
    total_information_gain: float = 0.0
    information_cell_count: int = 0
    information_cells: List[Tuple[float, float]] = field(default_factory=list)
    path_length: float = 0.0
    revisit_cost: float = 0.0
    turning_cost: float = 0.0
    risk_cost: float = 0.0
    forward_progress: float = 0.0
    final_score: float = float("-inf")
    rejected: bool = False
    rejection_reason: str = ""

    def to_dict(self, include_information_cells: bool = False) -> dict:
        payload = {
            "path_id": self.path_id,
            "heading": self.heading,
            "source": self.source,
            "active_region_id": self.active_region_id,
            "waypoints": [item.to_dict() for item in self.waypoints],
            "total_information_gain": self.total_information_gain,
            "information_cell_count": self.information_cell_count,
            "path_length": self.path_length,
            "revisit_cost": self.revisit_cost,
            "turning_cost": self.turning_cost,
            "risk_cost": self.risk_cost,
            "forward_progress": self.forward_progress,
            "final_score": (self.final_score
                            if math.isfinite(self.final_score) else None),
            "rejected": self.rejected,
            "rejection_reason": self.rejection_reason,
        }
        if include_information_cells:
            payload["information_cells"] = [list(item)
                                             for item in self.information_cells]
        return payload


def unique_headings(items: Iterable[Tuple[float, str, Optional[str]]],
                    tolerance_degrees: float = 10.0):
    output = []
    tolerance = math.radians(float(tolerance_degrees))
    for heading, source, region_id in items:
        heading = wrap_angle(heading)
        if any(abs(wrap_angle(heading - old[0])) < tolerance
               for old in output):
            continue
        output.append((heading, source, region_id))
    return output


def cell_is_free_with_clearance(grid, cell: Cell, clearance: float) -> bool:
    radius = max(1, int(math.ceil(float(clearance) / grid.resolution)))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if math.hypot(dx, dy) * grid.resolution > float(clearance):
                continue
            x, y = cell[0] + dx, cell[1] + dy
            if (not 0 <= x < grid.width or not 0 <= y < grid.height or
                    int(grid.data[y, x]) != 0):
                return False
    return True


def nearest_free_point(grid, requested: Sequence[float], clearance: float,
                       search_radius: float) -> Optional[Tuple[float, float]]:
    center = grid.world_to_cell(requested)
    if center is None:
        return None
    radius = max(0, int(math.ceil(float(search_radius) / grid.resolution)))
    candidates = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            cell = center[0] + dx, center[1] + dy
            if (not 0 <= cell[0] < grid.width or
                    not 0 <= cell[1] < grid.height or
                    not cell_is_free_with_clearance(grid, cell, clearance)):
                continue
            point = grid.cell_to_world(cell)
            candidates.append((
                math.hypot(point[0] - float(requested[0]),
                           point[1] - float(requested[1])), point))
    return min(candidates, default=(None, None), key=lambda item: item[0])[1]


def visible_unknown_cells(grid, point: Sequence[float], radius: float) -> Set[Cell]:
    """Approximate sensor gain by unique unknown cells in a circular footprint."""
    center = grid.world_to_cell(point)
    if center is None:
        return set()
    bound = max(1, int(math.ceil(float(radius) / grid.resolution)))
    cells = set()
    for dy in range(-bound, bound + 1):
        for dx in range(-bound, bound + 1):
            if math.hypot(dx, dy) * grid.resolution > float(radius):
                continue
            x, y = center[0] + dx, center[1] + dy
            if (0 <= x < grid.width and 0 <= y < grid.height and
                    int(grid.data[y, x]) < 0):
                cells.add((x, y))
    return cells


def boundary_risk(grid, point: Sequence[float], radius: float) -> float:
    center = grid.world_to_cell(point)
    if center is None:
        return 1.0
    bound = max(1, int(math.ceil(float(radius) / grid.resolution)))
    nearest_occupied = float("inf")
    nearest_unknown = float("inf")
    for dy in range(-bound, bound + 1):
        for dx in range(-bound, bound + 1):
            distance = math.hypot(dx, dy) * grid.resolution
            if distance > float(radius):
                continue
            x, y = center[0] + dx, center[1] + dy
            if not 0 <= x < grid.width or not 0 <= y < grid.height:
                nearest_occupied = min(nearest_occupied, distance)
                continue
            state = int(grid.data[y, x])
            if state >= 50:
                nearest_occupied = min(nearest_occupied, distance)
            elif state < 0:
                nearest_unknown = min(nearest_unknown, distance)
    occupied_risk = max(0.0, 1.0 - nearest_occupied / float(radius))
    unknown_risk = max(0.0, 1.0 - nearest_unknown / float(radius))
    return occupied_risk + 0.5 * unknown_risk


def path_score(candidate: CandidatePath, config: LocalPathConfig) -> float:
    return (config.alpha * candidate.total_information_gain -
            config.beta * candidate.path_length -
            config.gamma * candidate.revisit_cost -
            config.delta * candidate.turning_cost -
            config.eta * candidate.risk_cost +
            config.mu * candidate.forward_progress)


def path_commit_released(executed_waypoints: int, travelled_distance: float,
                         elapsed: float, remaining_gain: float,
                         min_waypoints: int = 3,
                         min_distance: float = 1.5,
                         min_time: float = 12.0,
                         minimum_remaining_gain: float = 1.0) -> bool:
    """The commitment contract uses the requested OR release semantics."""
    return bool(
        int(executed_waypoints) >= int(min_waypoints) or
        float(travelled_distance) >= float(min_distance) or
        float(elapsed) >= float(min_time) or
        float(remaining_gain) < float(minimum_remaining_gain))


def build_candidate_path(
        grid, pose: Point, robot_yaw: float, heading: float,
        planner, visited: Sequence[Point], config: LocalPathConfig,
        path_id: str, source: str = "sampled_heading",
        active_region_id: Optional[str] = None,
        scan_checker: Optional[Callable[[Point, float], Tuple[bool, str, float]]] = None,
        recovery: bool = False) -> CandidatePath:
    candidate = CandidatePath(path_id, wrap_angle(heading), source,
                              active_region_id)
    previous = pose
    information_cells: Set[Cell] = set()
    previous_heading = float(robot_yaw)
    for index in range(1, max(1, int(config.waypoint_count)) + 1):
        distance = index * float(config.waypoint_spacing)
        requested = (
            float(pose[0]) + distance * math.cos(heading),
            float(pose[1]) + distance * math.sin(heading))
        point2 = nearest_free_point(
            grid, requested, config.free_clearance,
            config.sample_free_search_radius)
        if point2 is None:
            continue
        point = (point2[0], point2[1], float(pose[2]))
        if (planar_distance(previous, point) <
                config.minimum_waypoint_separation):
            continue
        route = planner.plan(previous, point, grid)
        if not route.success:
            continue
        waypoint_yaw = bearing(previous, point)
        safety_status, scan_risk = "astar_observed_free", 0.0
        if scan_checker is not None:
            safe, safety_status, scan_risk = scan_checker(point, waypoint_yaw)
            if not safe:
                continue
        visible = visible_unknown_cells(grid, point, config.sensor_range)
        novel = visible - information_cells
        information_cells.update(novel)
        gain_area = len(novel) * grid.resolution ** 2
        candidate.waypoints.append(LocalWaypoint(
            point[0], point[1], waypoint_yaw, gain_area,
            safety_status, len(novel)))
        candidate.path_length += float(route.travel_cost)
        candidate.turning_cost += abs(wrap_angle(
            waypoint_yaw - previous_heading))
        candidate.risk_cost += boundary_risk(
            grid, point, config.risk_radius) + max(0.0, float(scan_risk))
        candidate.revisit_cost += sum(
            max(0.0, 1.0 - planar_distance(point, old) /
                config.revisit_radius)
            for old in visited
            if planar_distance(point, old) < config.revisit_radius)
        previous, previous_heading = point, waypoint_yaw

    if candidate.waypoints:
        endpoint = candidate.waypoints[-1]
        dx, dy = endpoint.x - pose[0], endpoint.y - pose[1]
        candidate.forward_progress = (
            dx * math.cos(robot_yaw) + dy * math.sin(robot_yaw))
        if candidate.forward_progress < 0.0 and not recovery:
            candidate.revisit_cost += (
                config.backward_penalty + abs(candidate.forward_progress))
    candidate.information_cell_count = len(information_cells)
    candidate.total_information_gain = (
        len(information_cells) * grid.resolution ** 2)
    candidate.information_cells = [grid.cell_to_world(cell)
                                   for cell in sorted(information_cells)]
    if len(candidate.waypoints) < int(config.minimum_waypoints):
        candidate.rejected = True
        candidate.rejection_reason = "insufficient_safe_free_waypoints"
    else:
        candidate.final_score = path_score(candidate, config)
    return candidate


def generate_candidate_paths(
        grid, pose: Point, robot_yaw: float, planner,
        visited: Sequence[Point], directional_targets: Sequence[dict],
        config: Optional[LocalPathConfig] = None,
        scan_checker=None, recovery: bool = False,
        path_prefix: str = "local"):
    config = config or LocalPathConfig()
    headings = [
        (robot_yaw, "heading_forward", None),
        (robot_yaw + math.radians(35.0), "heading_left_front", None),
        (robot_yaw - math.radians(35.0), "heading_right_front", None),
    ]
    for target in directional_targets:
        point = target.get("point")
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        headings.append((
            bearing(pose, point), str(target.get("source", "frontier")),
            target.get("active_region_id")))
    candidates = []
    for index, (heading, source, region_id) in enumerate(
            unique_headings(headings), 1):
        candidates.append(build_candidate_path(
            grid, pose, robot_yaw, heading, planner, visited, config,
            "{}_{:02d}".format(path_prefix, index), source, region_id,
            scan_checker, recovery))
    eligible = [item for item in candidates if not item.rejected]
    selected = max(eligible, default=None, key=lambda item: item.final_score)
    return selected, candidates
