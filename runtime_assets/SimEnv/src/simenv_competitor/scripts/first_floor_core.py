#!/usr/bin/env python3
"""ROS-free building blocks for the first-floor autonomous mission.

The runtime nodes intentionally keep mapping, topology and safety decisions in
this module so the competition-critical behaviour can be tested without a
running Gazebo instance.
"""

from __future__ import annotations

import math
import heapq
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def bresenham_cells(
    start: Tuple[int, int], end: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """Return every raster cell intersected by a two-dimensional ray."""
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    result: List[Tuple[int, int]] = []
    while True:
        result.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        twice = 2 * error
        if twice >= dy:
            error += dy
            x0 += sx
        if twice <= dx:
            error += dx
            y0 += sy
    return result


class RayOccupancyGrid:
    """Fixed-size log-odds grid with explicit unknown/free/occupied state."""

    def __init__(
        self,
        width: int,
        height: int,
        resolution: float,
        origin_x: float,
        origin_y: float,
        free_delta: float = -0.45,
        occupied_delta: float = 0.55,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.resolution = float(resolution)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        if self.width <= 0 or self.height <= 0 or self.resolution <= 0.0:
            raise ValueError("invalid occupancy-grid geometry")
        self.free_delta = float(free_delta)
        self.occupied_delta = float(occupied_delta)
        self.log_odds = np.zeros((self.height, self.width), dtype=np.float32)
        self.observed = np.zeros((self.height, self.width), dtype=np.bool_)

    def world_to_cell(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        ix = int(math.floor((float(x) - self.origin_x) / self.resolution))
        iy = int(math.floor((float(y) - self.origin_y) / self.resolution))
        if 0 <= ix < self.width and 0 <= iy < self.height:
            return ix, iy
        return None

    def cell_to_world(self, ix: int, iy: int) -> Tuple[float, float]:
        return (
            self.origin_x + (float(ix) + 0.5) * self.resolution,
            self.origin_y + (float(iy) + 0.5) * self.resolution,
        )

    def update_ray(
        self,
        origin: Sequence[float],
        endpoint: Sequence[float],
        hit: bool = True,
    ) -> bool:
        """Apply free evidence along a ray and occupied evidence at its end."""
        start = self.world_to_cell(float(origin[0]), float(origin[1]))
        end = self.world_to_cell(float(endpoint[0]), float(endpoint[1]))
        if start is None or end is None:
            return False
        cells = bresenham_cells(start, end)
        free_cells = cells[:-1] if hit and len(cells) > 1 else cells
        for ix, iy in free_cells:
            self.observed[iy, ix] = True
            self.log_odds[iy, ix] = np.clip(
                self.log_odds[iy, ix] + self.free_delta, -4.0, 4.0
            )
        if hit:
            ix, iy = cells[-1]
            self.observed[iy, ix] = True
            self.log_odds[iy, ix] = np.clip(
                self.log_odds[iy, ix] + self.occupied_delta, -4.0, 4.0
            )
        return True

    def occupancy(self, occupied_threshold: float = 0.70) -> np.ndarray:
        result = np.full((self.height, self.width), -1, dtype=np.int8)
        probability = 1.0 / (1.0 + np.exp(-self.log_odds))
        result[self.observed] = 0
        result[np.logical_and(self.observed, probability >= occupied_threshold)] = 100
        return result


@dataclass
class DoorObservation:
    longitudinal: float
    side: int
    width: float
    center_x: float
    center_y: float
    normal_x: float
    normal_y: float
    observations: int = 1
    corridor_half_width: float = 1.0


class DoorClusterer:
    """Merge repeated online doorway observations without layout coordinates."""

    def __init__(
        self,
        minimum_width: float = 0.8,
        maximum_width: float = 2.0,
        merge_distance: float = 1.0,
    ) -> None:
        self.minimum_width = float(minimum_width)
        self.maximum_width = float(maximum_width)
        self.merge_distance = float(merge_distance)
        self.doors: List[DoorObservation] = []

    def add(self, observation: DoorObservation) -> Optional[DoorObservation]:
        if observation.side not in (-1, 1):
            return None
        if not self.minimum_width <= observation.width <= self.maximum_width:
            return None
        nearest = None
        nearest_distance = float("inf")
        for door in self.doors:
            distance = abs(door.longitudinal - observation.longitudinal)
            if door.side == observation.side and distance < nearest_distance:
                nearest = door
                nearest_distance = distance
        if nearest is None or nearest_distance > self.merge_distance:
            self.doors.append(observation)
            return observation
        total = nearest.observations + observation.observations
        old_weight = nearest.observations / float(total)
        new_weight = observation.observations / float(total)
        for name in (
            "longitudinal",
            "width",
            "center_x",
            "center_y",
            "normal_x",
            "normal_y",
            "corridor_half_width",
        ):
            setattr(
                nearest,
                name,
                getattr(nearest, name) * old_weight
                + getattr(observation, name) * new_weight,
            )
        norm = math.hypot(nearest.normal_x, nearest.normal_y)
        if norm > 1e-6:
            nearest.normal_x /= norm
            nearest.normal_y /= norm
        nearest.observations = total
        return nearest

    def ordered(self) -> List[DoorObservation]:
        # At one longitudinal band, visit left/right consecutively before
        # progressing deeper into the corridor.
        return sorted(self.doors, key=lambda door: (door.longitudinal, -door.side))


@dataclass(frozen=True)
class DoorGroup:
    """Doorways sharing one corridor station, ordered by turn convenience."""

    longitudinal: float
    doors: Tuple[DoorObservation, ...]


def pair_door_groups(
    doors: Iterable[DoorObservation],
    station_tolerance: float = 1.5,
    minimum_observations: int = 2,
    descending: bool = True,
) -> List[DoorGroup]:
    """Group confirmed left/right doors without using building coordinates."""
    confirmed = sorted(
        (door for door in doors if door.observations >= minimum_observations),
        key=lambda door: door.longitudinal,
        reverse=descending,
    )
    buckets: List[List[DoorObservation]] = []
    for door in confirmed:
        nearest = None
        nearest_error = float("inf")
        for bucket in buckets:
            station = sum(item.longitudinal for item in bucket) / len(bucket)
            error = abs(door.longitudinal - station)
            if error <= station_tolerance and error < nearest_error:
                nearest, nearest_error = bucket, error
        if nearest is None:
            buckets.append([door])
        elif all(item.side != door.side for item in nearest):
            nearest.append(door)
        else:
            # Keep the more reliable duplicate at a station; DoorClusterer
            # normally prevents this, but this makes route synthesis stable.
            same = next(item for item in nearest if item.side == door.side)
            if door.observations > same.observations:
                nearest[nearest.index(same)] = door

    groups = []
    for bucket in buckets:
        station = sum(item.longitudinal for item in bucket) / len(bucket)
        # On the return leg prefer the side needing the smaller body turn.
        ordered = tuple(sorted(bucket, key=lambda item: item.side))
        groups.append(DoorGroup(station, ordered))
    return sorted(groups, key=lambda group: group.longitudinal, reverse=descending)


def select_room_nbvs(
    door: DoorObservation,
    is_safe: Callable[[float, float], bool],
    entry_depth: float = 2.7,
    baseline: float = 1.25,
) -> List[Tuple[float, float]]:
    """Choose two safe in-room views using only the observed door normal."""
    nx, ny = door.normal_x, door.normal_y
    norm = math.hypot(nx, ny)
    if norm < 1e-6:
        return []
    nx, ny = nx / norm, ny / norm
    tx, ty = -ny, nx
    half_baseline = max(0.6, 0.5 * float(baseline))
    # Use both depth and transverse parallax.  Fall back toward the door in
    # small rooms, but never violate the 2.5 m entry requirement.
    for deep_depth in (entry_depth + 1.2, entry_depth + 0.8, entry_depth + 0.4, entry_depth):
        first = (
            door.center_x + nx * entry_depth - tx * half_baseline,
            door.center_y + ny * entry_depth - ty * half_baseline,
        )
        second = (
            door.center_x + nx * deep_depth + tx * half_baseline,
            door.center_y + ny * deep_depth + ty * half_baseline,
        )
        if is_safe(*first) and is_safe(*second) and math.dist(first, second) >= 1.2:
            return [first, second]
    # A centred first view is useful when only one side of the room is free.
    centre = (door.center_x + nx * entry_depth, door.center_y + ny * entry_depth)
    for offset in (baseline, -baseline):
        second = (centre[0] + tx * offset, centre[1] + ty * offset)
        if is_safe(*centre) and is_safe(*second) and math.dist(centre, second) >= 1.2:
            return [centre, second]
    return []


def corridor_lookahead_s(
    current_s: float,
    free_frontier_s: float,
    lookahead: float = 6.0,
    frontier_margin: float = 0.8,
) -> Optional[float]:
    """Choose a rolling goal strictly inside currently confirmed free space."""
    current = float(current_s)
    frontier = float(free_frontier_s) - max(0.0, float(frontier_margin))
    if not math.isfinite(current) or not math.isfinite(frontier) or frontier <= current + 0.35:
        return None
    return min(current + max(0.5, float(lookahead)), frontier)


def bilateral_corridor_axis_delta(
    points_xy: Iterable[Sequence[float]],
    minimum_span: float = 1.8,
    maximum_delta: float = 0.28,
    side_agreement: float = 0.10,
) -> Optional[float]:
    """Fit the local corridor direction only when both wall lines agree.

    A single PCA over all returns is biased by lobby cross-walls and paired
    door openings.  Independent side fits make the estimate invariant to the
    corridor width and reject incomplete one-sided geometry.
    """
    points = np.asarray(list(points_xy), dtype=np.float64)
    if points.size == 0 or points.ndim != 2 or points.shape[1] < 2:
        return None
    points = points[np.all(np.isfinite(points[:, :2]), axis=1), :2]
    angles = []
    for side in (1, -1):
        selected = points[
            (side * points[:, 1] >= 0.50)
            & (side * points[:, 1] <= 1.75)
            & (points[:, 0] >= -1.0)
            & (points[:, 0] <= 7.0)
        ]
        if len(selected) < 10 or np.ptp(selected[:, 0]) < float(minimum_span):
            return None
        centred = selected - np.mean(selected, axis=0)
        values, vectors = np.linalg.eigh(centred.T @ centred)
        direction = vectors[:, int(np.argmax(values))]
        angle = wrap_angle(math.atan2(direction[1], direction[0]))
        if abs(wrap_angle(angle - math.pi)) < abs(angle):
            angle = wrap_angle(angle - math.pi)
        if abs(angle) > float(maximum_delta):
            return None
        angles.append(angle)
    if abs(wrap_angle(angles[0] - angles[1])) > float(side_agreement):
        return None
    return wrap_angle(0.5 * (angles[0] + angles[1]))


def select_projected_door_gap(
    projected_stations: Iterable[float],
    minimum_width: float = 0.8,
    maximum_width: float = 2.0,
    cluster_gap: float = 0.28,
) -> Optional[Tuple[float, float]]:
    """Select the nearest contiguous aperture projected onto a side wall."""
    values = np.asarray(list(projected_stations), dtype=np.float64)
    values = np.sort(values[np.isfinite(values)])
    values = values[np.logical_and(values >= -3.5, values <= 7.0)]
    if len(values) < 6:
        return None
    split = np.flatnonzero(np.diff(values) > float(cluster_gap)) + 1
    candidates = []
    for group in np.split(values, split):
        if len(group) < 6:
            continue
        lower, upper = np.percentile(group, (5.0, 95.0))
        width = float(upper - lower)
        if float(minimum_width) <= width <= float(maximum_width):
            centre = float(0.5 * (lower + upper))
            candidates.append((abs(centre), centre, width))
    if not candidates:
        return None
    _, centre, width = min(candidates)
    return centre, width


def extract_occupied_wall_gaps(
    stations: Sequence[float],
    wall_values: Sequence[int],
    beyond_values: Sequence[int],
    minimum_width: float = 0.8,
    maximum_width: float = 2.0,
) -> List[Tuple[float, float]]:
    """Find free, room-backed gaps bracketed by an observed occupied wall."""
    s = np.asarray(stations, dtype=np.float64)
    wall = np.asarray(wall_values, dtype=np.int16)
    beyond = np.asarray(beyond_values, dtype=np.int16)
    if len(s) < 3 or len(s) != len(wall) or len(s) != len(beyond):
        return []
    opening = np.logical_and(wall == 0, beyond == 0)
    change = np.diff(np.r_[False, opening, False].astype(np.int8))
    starts = np.flatnonzero(change == 1)
    stops = np.flatnonzero(change == -1)
    step = float(np.median(np.diff(s)))
    gaps = []
    for start, stop in zip(starts, stops):
        width = float(s[stop - 1] - s[start] + step)
        if not float(minimum_width) <= width <= float(maximum_width):
            continue
        before = wall[max(0, start - 4) : start]
        after = wall[stop : min(len(wall), stop + 4)]
        if not np.any(before == 100) or not np.any(after == 100):
            continue
        gaps.append((float(0.5 * (s[start] + s[stop - 1])), width))
    return gaps


def proportional_yaw_rate(
    yaw_error: float,
    maximum_rate: float = 0.8,
    gain: float = 1.5,
    deadband: float = 0.03,
) -> float:
    """Monotonic turn command that cannot create a minimum-rate limit cycle."""
    error = wrap_angle(float(yaw_error))
    if not math.isfinite(error) or abs(error) <= max(0.0, float(deadband)):
        return 0.0
    magnitude = min(abs(float(maximum_rate)), abs(float(gain)) * abs(error))
    return math.copysign(magnitude, error)


def supported_obstacle_points(
    points_xy: Iterable[Sequence[float]],
    support_radius: float = 0.20,
    minimum_support: int = 3,
) -> np.ndarray:
    """Reject isolated LiDAR returns while retaining spatial obstacle clusters."""
    points = np.asarray(list(points_xy), dtype=np.float64)
    if points.size == 0:
        return points.reshape((-1, 2))
    if points.ndim != 2 or points.shape[1] < 2:
        return np.empty((0, 2), dtype=np.float64)
    points = points[np.all(np.isfinite(points[:, :2]), axis=1), :2]
    if len(points) < minimum_support:
        return np.empty((0, 2), dtype=np.float64)
    cell = max(0.02, float(support_radius))
    keys = np.floor(points / cell).astype(np.int32)
    counts = {}
    for key in map(tuple, keys):
        counts[key] = counts.get(key, 0) + 1
    keep = []
    for point, key_array in zip(points, keys):
        key = tuple(key_array)
        support = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                support += counts.get((key[0] + dx, key[1] + dy), 0)
        if support >= int(minimum_support):
            keep.append(point)
    return np.asarray(keep, dtype=np.float64).reshape((-1, 2))


def vertically_supported_obstacles(
    points_xyz: Iterable[Sequence[float]],
    low_height: float = 0.10,
    support_height: float = 0.15,
    support_radius: float = 0.22,
) -> np.ndarray:
    """Keep low returns only when a taller obstacle exists at the same XY."""
    points = np.asarray(list(points_xyz), dtype=np.float64)
    if points.size == 0:
        return points.reshape((-1, 3))
    points = points[np.all(np.isfinite(points[:, :3]), axis=1), :3]
    cell = max(0.02, float(support_radius))
    high_keys = {
        tuple(key)
        for key in np.floor(points[points[:, 2] >= support_height, :2] / cell).astype(np.int32)
    }
    keep = []
    for point in points:
        if point[2] >= low_height:
            keep.append(point)
            continue
        key = np.floor(point[:2] / cell).astype(np.int32)
        supported = any(
            (int(key[0]) + dx, int(key[1]) + dy) in high_keys
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
        )
        if supported:
            keep.append(point)
    return np.asarray(keep, dtype=np.float64).reshape((-1, 3))


def clustered_front_clearance(
    points_xy: Iterable[Sequence[float]], lateral_limit: float = 0.42
) -> float:
    supported = supported_obstacle_points(points_xy)
    if len(supported) == 0:
        return 20.0
    forward = supported[
        np.logical_and(supported[:, 0] > 0.0, np.abs(supported[:, 1]) < lateral_limit)
    ]
    return float(np.min(forward[:, 0])) if len(forward) else 20.0


def occupancy_patch_traversable(
    patch: np.ndarray, centre_value: int, minimum_free_ratio: float = 0.65
) -> bool:
    """Allow a ray-map unknown fringe without ever accepting an obstacle."""
    values = np.asarray(patch, dtype=np.int16)
    if values.size == 0 or int(centre_value) != 0 or np.any(values == 100):
        return False
    return float(np.mean(values == 0)) >= float(minimum_free_ratio)


def remove_isolated_occupied_cells(
    occupancy: np.ndarray, minimum_neighbours: int = 3
) -> np.ndarray:
    """Suppress unsupported static speckles without altering unknown/free rays."""
    result = np.asarray(occupancy, dtype=np.int8).copy()
    occupied = result == 100
    if not np.any(occupied):
        return result
    padded = np.pad(occupied.astype(np.uint8), 1, mode="constant")
    support = np.zeros_like(occupied, dtype=np.uint8)
    for dy in range(3):
        for dx in range(3):
            support += padded[dy : dy + occupied.shape[0], dx : dx + occupied.shape[1]]
    result[np.logical_and(occupied, support < int(minimum_neighbours))] = 0
    return result


def wall_center_error(left_distance: float, right_distance: float) -> float:
    """Positive when the robot is closer to the left corridor wall."""
    left, right = float(left_distance), float(right_distance)
    if not math.isfinite(left) or not math.isfinite(right):
        return 0.0
    return 0.5 * (right - left)


def should_update_rolling_goal(
    remaining_distance: float,
    target_shift: float,
    yaw_shift: float,
    refresh_distance: float = 2.5,
    minimum_target_shift: float = 0.8,
) -> bool:
    """Refresh a rolling action only near its lookahead, avoiding stop pulses."""
    return float(remaining_distance) <= float(refresh_distance) and (
        float(target_shift) >= float(minimum_target_shift) or abs(float(yaw_shift)) >= 0.12
    )


def return_time_budget(
    path_length: float, nominal_return_speed: float = 0.8, fixed_margin: float = 8.0
) -> float:
    return max(0.0, float(path_length)) / max(0.05, float(nominal_return_speed)) + max(
        0.0, float(fixed_margin)
    )


def must_return(
    elapsed: float,
    mission_limit: float,
    path_length: float,
    decision_margin: float = 3.0,
) -> bool:
    remaining = float(mission_limit) - max(0.0, float(elapsed))
    return remaining <= return_time_budget(path_length) + max(0.0, decision_margin)


def astar_grid_length(
    occupancy: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    resolution: float,
) -> float:
    """Shortest free-space A* length on an OccupancyGrid raster."""
    grid = np.asarray(occupancy)
    if grid.ndim != 2 or resolution <= 0.0:
        return float("inf")
    height, width = grid.shape
    for x, y in (start, goal):
        if not (0 <= x < width and 0 <= y < height) or grid[y, x] != 0:
            return float("inf")
    if start == goal:
        return 0.0
    diagonal = math.sqrt(2.0)
    neighbours = (
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, diagonal), (-1, 1, diagonal), (1, -1, diagonal), (1, 1, diagonal),
    )
    queue = [(0.0, 0.0, start)]
    best = {start: 0.0}
    while queue:
        _, cost, current = heapq.heappop(queue)
        if current == goal:
            return cost * float(resolution)
        if cost > best.get(current, float("inf")):
            continue
        for dx, dy, step in neighbours:
            candidate = (current[0] + dx, current[1] + dy)
            x, y = candidate
            if not (0 <= x < width and 0 <= y < height) or grid[y, x] != 0:
                continue
            # Do not cut diagonally through an occupied/unknown corner.
            if dx and dy and (
                grid[current[1], current[0] + dx] != 0
                or grid[current[1] + dy, current[0]] != 0
            ):
                continue
            candidate_cost = cost + step
            if candidate_cost >= best.get(candidate, float("inf")):
                continue
            best[candidate] = candidate_cost
            heuristic = math.hypot(goal[0] - x, goal[1] - y)
            heapq.heappush(queue, (candidate_cost + heuristic, candidate_cost, candidate))
    return float("inf")


def adaptive_speed_limit(
    clearance: float,
    localization_healthy: bool,
    yaw_error: float,
    center_error: float,
    in_room: bool = False,
    near_door: bool = False,
    sharp_turn: bool = False,
    maximum: float = 2.0,
) -> float:
    """Competition speed ladder with an explicit 2.0 m/s enable gate."""
    maximum = max(0.0, float(maximum))
    if sharp_turn or not localization_healthy:
        return min(maximum, 0.3)
    if in_room or near_door:
        return min(maximum, 0.6)
    clearance = float(clearance) if math.isfinite(float(clearance)) else 0.0
    precise = abs(float(yaw_error)) < 0.08 and abs(float(center_error)) < 0.30
    if clearance >= 7.0 and precise:
        return min(maximum, 2.0)
    if clearance >= 5.5 and abs(float(yaw_error)) < 0.15:
        return min(maximum, 1.4)
    if clearance >= 4.5 and abs(float(yaw_error)) < 0.24:
        return min(maximum, 1.0)
    return min(maximum, 0.6)


def predicted_footprint_collision(
    points_xy: Iterable[Sequence[float]],
    forward_speed: float,
    horizon: float = 1.5,
    footprint_length: float = 0.70,
    footprint_width: float = 0.44,
    margin: float = 0.12,
) -> bool:
    """Check a swept full-footprint corridor in the robot frame."""
    speed = float(forward_speed)
    travel = abs(speed) * max(0.0, float(horizon))
    rear = -0.5 * float(footprint_length) - float(margin)
    front = 0.5 * float(footprint_length) + float(margin)
    if speed >= 0.0:
        front += travel
    else:
        rear -= travel
    half_width = 0.5 * float(footprint_width) + float(margin)
    for point in points_xy:
        x, y = float(point[0]), float(point[1])
        if math.isfinite(x) and math.isfinite(y) and rear <= x <= front and abs(y) <= half_width:
            return True
    return False


@dataclass(frozen=True)
class GeometryEvidence:
    label: str
    radius: float
    sphere_inlier_ratio: float
    plane_inlier_ratio: float
    sphere_residual: float
    center: Tuple[float, float, float] = (float("nan"), float("nan"), float("nan"))


def classify_sphere_or_plane(
    points: Iterable[Sequence[float]],
    minimum_radius: float = 0.10,
    maximum_radius: float = 0.20,
    sphere_tolerance: float = 0.025,
    plane_tolerance: float = 0.012,
) -> GeometryEvidence:
    """Fit a sphere and a dominant plane to an RGB-D candidate cloud."""
    cloud = np.asarray(list(points), dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[0] < 12 or cloud.shape[1] != 3:
        return GeometryEvidence("insufficient", float("nan"), 0.0, 0.0, float("inf"))
    cloud = cloud[np.all(np.isfinite(cloud), axis=1)]
    if cloud.shape[0] < 12:
        return GeometryEvidence("insufficient", float("nan"), 0.0, 0.0, float("inf"))

    # Linear least-squares sphere: 2*p*c + d = |p|^2, r^2=|c|^2+d.
    design = np.column_stack((2.0 * cloud, np.ones(cloud.shape[0])))
    rhs = np.sum(cloud * cloud, axis=1)
    solution, _, _, _ = np.linalg.lstsq(design, rhs, rcond=None)
    centre = solution[:3]
    radius_sq = float(np.dot(centre, centre) + solution[3])
    radius = math.sqrt(radius_sq) if radius_sq > 0.0 else float("nan")
    radial = np.linalg.norm(cloud - centre, axis=1)
    residuals = np.abs(radial - radius) if math.isfinite(radius) else np.full(len(cloud), np.inf)
    sphere_ratio = float(np.mean(residuals <= sphere_tolerance))
    sphere_residual = float(np.median(residuals))

    centred = cloud - np.mean(cloud, axis=0)
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    normal = vh[-1]
    plane_distance = np.abs(centred @ normal)
    plane_ratio = float(np.mean(plane_distance <= plane_tolerance))

    radius_ok = math.isfinite(radius) and minimum_radius <= radius <= maximum_radius
    is_sphere = radius_ok and sphere_ratio >= 0.65 and sphere_ratio > plane_ratio + 0.05
    return GeometryEvidence(
        "sphere" if is_sphere else "plane_or_box",
        radius,
        sphere_ratio,
        plane_ratio,
        sphere_residual,
        tuple(float(value) for value in centre),
    )


def planar_align_points(
    points: Iterable[Sequence[float]], yaw: float, tx: float, ty: float
) -> np.ndarray:
    """Apply the same SE(2) alignment to a registered cloud as to its pose."""
    cloud = np.asarray(list(points), dtype=np.float64)
    if cloud.size == 0:
        return cloud.reshape((-1, 3))
    cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
    result = cloud.copy()
    result[:, 0] = cosine * cloud[:, 0] - sine * cloud[:, 1] + float(tx)
    result[:, 1] = sine * cloud[:, 0] + cosine * cloud[:, 1] + float(ty)
    return result
