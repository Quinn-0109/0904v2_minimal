#!/usr/bin/env python3
"""Incremental frontier structure inspired by FUEL's FIS, adapted for ground robots.

FUEL maintains an Incremental Frontier Information Structure and then plans
hierarchically: cover frontiers globally, refine local viewpoints, generate
motions.  This module keeps the same information hierarchy for a quadruped:

  radar (360 deg, long range)  -> frontier clusters / coverage gaps
  camera (60 deg, short range) -> local viewpoint refinement for color ID

It does NOT replace TARE.  It scores where the narrow RGB camera should look
and how long to dwell, using distance-dependent policies.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def angle_diff(target: float, source: float) -> float:
    return (target - source + math.pi) % (2.0 * math.pi) - math.pi


def wrap_yaw(yaw: float) -> float:
    return (yaw + math.pi) % (2.0 * math.pi) - math.pi


def corridor_anchor_from_door(
    door_x: float, door_y: float, inward_yaw: float, offset: float = 0.9
) -> Tuple[float, float]:
    """Return the corridor-side target opposite a room's inward normal."""
    outward = wrap_yaw(inward_yaw + math.pi)
    return (
        door_x + max(0.1, offset) * math.cos(outward),
        door_y + max(0.1, offset) * math.sin(outward),
    )


def corridor_exit_confirmed(
    distance_to_anchor: float,
    consecutive_corridor_hits: int,
    tolerance: float = 0.35,
    required_hits: int = 3,
) -> bool:
    """Pure exit gate used by the state machine and its regression tests."""
    return (
        math.isfinite(distance_to_anchor)
        and distance_to_anchor <= tolerance
        and consecutive_corridor_hits >= required_hits
    )


def forward_escape_offset(
    front_left: Optional[float],
    front_right: Optional[float],
    magnitude: float = 0.62,
) -> float:
    """Turn toward the clearer forward diagonal, not an open side room."""
    amount = max(0.15, abs(float(magnitude)))
    if front_left is None and front_right is None:
        return amount
    if front_left is None:
        return -amount
    if front_right is None:
        return amount
    return amount if front_left >= front_right else -amount


@dataclass
class FrontierCluster:
    """One coherent unknown/occluded direction group (FUEL-style cluster)."""

    cluster_id: int
    yaw: float
    distance: float
    point_count: int
    average_range: float
    angular_width: float
    last_update: float
    hits: int = 1
    camera_visits: int = 0
    covered: bool = False

    def coverage_gain(self, camera_range: float, camera_fov: float) -> float:
        """Approximate unseen area if the camera looks at this cluster."""
        usable = min(max(self.average_range, 0.5), camera_range)
        # Wider angular support and longer free range => higher gain.
        span = min(self.angular_width, camera_fov)
        return usable * usable * span

    def view_cost(self, robot_yaw: float, turn_weight: float = 0.35) -> float:
        turn = abs(angle_diff(self.yaw, robot_yaw)) / math.pi
        revisit = 0.15 * self.camera_visits
        return turn_weight * turn + revisit


@dataclass
class ViewpointDecision:
    """Local refined viewpoint for the camera (FUEL hierarchical local stage)."""

    score: float
    yaw: float
    distance: float
    layer: str  # near | mid | far
    hold_seconds: float
    cluster_id: int
    reason: str
    # near: stop briefly; mid: keep driving while yawing (FUEL local refine).
    preempt_mode: str = "yaw_blend"


@dataclass
class SphereCandidate:
    """Compact mid-range return that may be a danger sphere (needs RGB confirm)."""

    yaw: float
    distance: float
    diameter: float
    point_count: int
    score: float


@dataclass
class GapTarget:
    """Navigable free-space opening (door / aisle) from LiDAR sectors."""

    yaw: float
    distance: float
    width: float
    score: float


@dataclass(frozen=True)
class RoomViewpoint:
    """A safe local viewpoint with an explicit expected information gain."""

    x: float
    y: float
    yaw: float
    expected_gain: float
    shadow_cells: int
    path_cost: float
    score: float
    reason: str


@dataclass(frozen=True)
class RoomCoverageStatus:
    """Online, sensor-only evidence used to decide whether a room is done."""

    lidar_ratio: float
    free_cells: int
    shadow_cells: int
    largest_shadow_m2: float
    wall_components: int
    obstacle_components: int
    far_wall_seen: bool
    left_wall_seen: bool
    right_wall_seen: bool
    boundary_directions_seen: int
    known_cells: int = 0
    unknown_cells: int = 0
    largest_unknown_m2: float = 0.0
    estimated_room_cells: int = 0
    boundary_complete: bool = False


def room_boundary_gate(
    status: RoomCoverageStatus,
    observed_depth: float,
    required_depth: float,
    min_directions: int = 2,
) -> bool:
    """Pure hard gate: shallow, wall-free room visits are never complete."""
    return (
        math.isfinite(observed_depth)
        and observed_depth >= required_depth
        and status.far_wall_seen
        and status.boundary_directions_seen >= max(1, int(min_directions))
    )


def room_entry_confirmed(
    door_frame_crossed: bool,
    room_depth: float,
    required_depth: float = 0.8,
    pose_healthy: bool = True,
) -> bool:
    """Confirm a real room entry from geometry, never elapsed command time."""
    return (
        bool(door_frame_crossed)
        and bool(pose_healthy)
        and math.isfinite(room_depth)
        and room_depth >= max(0.1, float(required_depth))
    )


def viewpoint_candidate_is_diverse(
    candidate: Tuple[float, float],
    visited: Sequence[Tuple[float, float]],
    door_x: float,
    door_y: float,
    inward_yaw: float,
    min_baseline: float = 1.2,
    min_lateral_span: float = 1.4,
) -> bool:
    """Return whether a supplementary viewpoint adds useful spatial parallax."""
    if not visited:
        return True
    cx, cy = candidate
    if not (math.isfinite(cx) and math.isfinite(cy)):
        return False
    baseline = min(math.hypot(cx - vx, cy - vy) for vx, vy in visited)
    if baseline < max(0.0, float(min_baseline)):
        return False

    side_x, side_y = -math.sin(inward_yaw), math.cos(inward_yaw)

    def lateral(point: Tuple[float, float]) -> float:
        return (point[0] - door_x) * side_x + (point[1] - door_y) * side_y

    lateral_values = [lateral(point) for point in visited]
    lateral_values.append(lateral(candidate))
    return max(lateral_values) - min(lateral_values) >= max(
        0.0, float(min_lateral_span)
    )


def room_viewpoint_diversity_gate(
    viewpoints: Sequence[Tuple[float, float]],
    door_x: float,
    door_y: float,
    inward_yaw: float,
    min_viewpoints: int = 2,
    min_baseline: float = 1.2,
    min_lateral_span: float = 1.4,
) -> bool:
    """Require enough viewpoints and both translational and lateral spread."""
    if len(viewpoints) < max(2, int(min_viewpoints)):
        return False
    finite = [
        (float(x), float(y))
        for x, y in viewpoints
        if math.isfinite(x) and math.isfinite(y)
    ]
    if len(finite) < max(2, int(min_viewpoints)):
        return False
    maximum_baseline = max(
        math.hypot(ax - bx, ay - by)
        for index, (ax, ay) in enumerate(finite)
        for bx, by in finite[index + 1 :]
    )
    if maximum_baseline < max(0.0, float(min_baseline)):
        return False
    side_x, side_y = -math.sin(inward_yaw), math.cos(inward_yaw)
    laterals = [
        (x - door_x) * side_x + (y - door_y) * side_y for x, y in finite
    ]
    return max(laterals) - min(laterals) >= max(0.0, float(min_lateral_span))


def shadow_debt_candidate_allowed(
    largest_shadow_m2: float,
    cleared_shadow_m2: float,
    cleared_fraction: float,
    debt_threshold_m2: float = 0.5,
    min_clear_m2: float = 0.2,
    min_clear_fraction: float = 0.25,
) -> bool:
    """Allow a low-global-gain view if it materially clears a major shadow."""
    return (
        math.isfinite(largest_shadow_m2)
        and largest_shadow_m2 > max(0.0, float(debt_threshold_m2))
        and (
            cleared_shadow_m2 >= max(0.0, float(min_clear_m2))
            or cleared_fraction >= max(0.0, float(min_clear_fraction))
        )
    )


def room_exploration_complete(
    entry_ok: bool,
    status: RoomCoverageStatus,
    observed_depth: float,
    required_depth: float,
    viewpoints: Sequence[Tuple[float, float]],
    door_x: float,
    door_y: float,
    inward_yaw: float,
    next_gain: float,
    min_lidar_ratio: float = 0.92,
    max_shadow_m2: float = 0.5,
    max_next_gain: float = 0.08,
    min_viewpoints: int = 2,
    min_baseline: float = 1.2,
    min_lateral_span: float = 1.4,
) -> bool:
    """Sensor-only final gate for a genuinely completed multi-view room."""
    return (
        bool(entry_ok)
        and room_boundary_gate(status, observed_depth, required_depth)
        and room_viewpoint_diversity_gate(
            viewpoints,
            door_x,
            door_y,
            inward_yaw,
            min_viewpoints=min_viewpoints,
            min_baseline=min_baseline,
            min_lateral_span=min_lateral_span,
        )
        and status.lidar_ratio >= min_lidar_ratio
        and status.largest_shadow_m2 <= max_shadow_m2
        and math.isfinite(next_gain)
        and next_gain < max_next_gain
    )


class RoomVisibilityGrid:
    """Door-relative 2-D free/occupied/occlusion grid for one room.

    The map is deliberately local.  It never consumes the generated layout or
    ground truth: a room is bounded by the doorway half-plane, a conservative
    local range, and the free/occupied evidence accumulated from body-frame
    LiDAR returns.  Unknown cells are not counted as explored merely because
    the robot drove nearby.  Only unknown cells in a geometric shadow behind a
    compact occupied component become coverage debt.
    """

    def __init__(
        self,
        door_x: float,
        door_y: float,
        inward_yaw: float,
        resolution: float = 0.15,
        inflation_radius: float = 0.45,
        max_depth: float = 9.5,
        half_width: float = 7.5,
        sensor_range: float = 9.0,
        wall_min_length: float = 1.2,
        wall_max_residual: float = 0.18,
        shadow_depth: float = 2.2,
    ) -> None:
        self.door_x = float(door_x)
        self.door_y = float(door_y)
        self.inward_yaw = wrap_yaw(inward_yaw)
        self.resolution = max(0.05, float(resolution))
        self.inflation_radius = max(self.resolution, float(inflation_radius))
        self.max_depth = max(1.0, float(max_depth))
        self.half_width = max(1.0, float(half_width))
        self.sensor_range = max(1.0, float(sensor_range))
        self.wall_min_length = max(0.5, float(wall_min_length))
        self.wall_max_residual = max(self.resolution, float(wall_max_residual))
        self.shadow_depth = max(self.resolution, float(shadow_depth))
        self.free: Set[Tuple[int, int]] = set()
        self.occupied: Set[Tuple[int, int]] = set()
        self._scan_poses: Deque[Tuple[float, float]] = deque(maxlen=6)

    def record_scan_pose(
        self, robot_x: float, robot_y: float, min_separation: float = 0.4
    ) -> bool:
        """Record only spatially distinct scan origins used for shadow evidence."""
        if not (math.isfinite(robot_x) and math.isfinite(robot_y)):
            return False
        separation = max(0.0, float(min_separation))
        if any(
            math.hypot(robot_x - old_x, robot_y - old_y) < separation
            for old_x, old_y in self._scan_poses
        ):
            return False
        self._scan_poses.append((float(robot_x), float(robot_y)))
        return True

    def cell_key(self, x: float, y: float) -> Tuple[int, int]:
        return (
            int(math.floor(float(x) / self.resolution)),
            int(math.floor(float(y) / self.resolution)),
        )

    def cell_center(self, key: Tuple[int, int]) -> Tuple[float, float]:
        return (
            (key[0] + 0.5) * self.resolution,
            (key[1] + 0.5) * self.resolution,
        )

    def room_coordinates(self, x: float, y: float) -> Tuple[float, float]:
        dx, dy = x - self.door_x, y - self.door_y
        depth = dx * math.cos(self.inward_yaw) + dy * math.sin(self.inward_yaw)
        lateral = -dx * math.sin(self.inward_yaw) + dy * math.cos(self.inward_yaw)
        return depth, lateral

    def inside_room(self, x: float, y: float) -> bool:
        depth, lateral = self.room_coordinates(x, y)
        return -0.25 <= depth <= self.max_depth and abs(lateral) <= self.half_width

    def _ray_cells(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> List[Tuple[int, int]]:
        distance = math.hypot(x1 - x0, y1 - y0)
        steps = max(1, int(math.ceil(distance / (self.resolution * 0.45))))
        cells: List[Tuple[int, int]] = []
        previous = None
        for index in range(steps + 1):
            ratio = index / float(steps)
            key = self.cell_key(x0 + ratio * (x1 - x0), y0 + ratio * (y1 - y0))
            if key != previous:
                cells.append(key)
                previous = key
        return cells

    @staticmethod
    def _percentile(values: Sequence[float], ratio: float) -> float:
        ordered = sorted(values)
        if not ordered:
            return 0.0
        index = min(len(ordered) - 1, max(0, int(ratio * len(ordered))))
        return float(ordered[index])

    def update_scan(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        points_xy: Sequence[Tuple[float, float]],
        angular_bins: int = 180,
    ) -> None:
        """Fuse a body-frame scan using one robust endpoint per angular bin."""
        if not points_xy:
            return
        bins: List[List[float]] = [[] for _ in range(max(36, angular_bins))]
        for px, py in points_xy:
            distance = math.hypot(px, py)
            if not math.isfinite(distance) or distance < 0.42:
                continue
            angle = math.atan2(py, px) % (2.0 * math.pi)
            index = int(angle / (2.0 * math.pi) * len(bins)) % len(bins)
            bins[index].append(min(distance, self.sensor_range))

        fused = 0
        for index, values in enumerate(bins):
            if not values:
                continue
            # High percentiles reject the A1 body/floor returns that otherwise
            # look like a ring of obstacles around the robot.
            distance = self._percentile(values, 0.72)
            local_yaw = (index + 0.5) * 2.0 * math.pi / len(bins)
            world_yaw = wrap_yaw(robot_yaw + local_yaw)
            end_x = robot_x + distance * math.cos(world_yaw)
            end_y = robot_y + distance * math.sin(world_yaw)
            ray = self._ray_cells(robot_x, robot_y, end_x, end_y)
            if len(ray) < 2:
                continue
            endpoint = ray[-1]
            for key in ray[:-1]:
                cx, cy = self.cell_center(key)
                if self.inside_room(cx, cy):
                    self.free.add(key)
            if distance < self.sensor_range * 0.98:
                cx, cy = self.cell_center(endpoint)
                if self.inside_room(cx, cy):
                    self.occupied.add(endpoint)
            fused += 1
        if fused:
            self.free.difference_update(self.occupied)
            self.record_scan_pose(robot_x, robot_y)

    def _components(
        self, cells: Iterable[Tuple[int, int]], neighbor_radius: int = 1
    ) -> List[Set[Tuple[int, int]]]:
        remaining = set(cells)
        components: List[Set[Tuple[int, int]]] = []
        radius = max(1, int(neighbor_radius))
        while remaining:
            seed = remaining.pop()
            component = {seed}
            queue = deque([seed])
            while queue:
                cx, cy = queue.popleft()
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if dx == 0 and dy == 0:
                            continue
                        nxt = (cx + dx, cy + dy)
                        if nxt in remaining:
                            remaining.remove(nxt)
                            component.add(nxt)
                            queue.append(nxt)
            components.append(component)
        return components

    def _component_geometry(
        self, component: Set[Tuple[int, int]]
    ) -> Tuple[float, float, float, float]:
        points = [self.cell_center(key) for key in component]
        mean_x = sum(point[0] for point in points) / len(points)
        mean_y = sum(point[1] for point in points) / len(points)
        sxx = sum((point[0] - mean_x) ** 2 for point in points) / len(points)
        syy = sum((point[1] - mean_y) ** 2 for point in points) / len(points)
        sxy = sum(
            (point[0] - mean_x) * (point[1] - mean_y) for point in points
        ) / len(points)
        trace = sxx + syy
        root = math.sqrt(max(0.0, (sxx - syy) ** 2 + 4.0 * sxy * sxy))
        major = max(0.0, 0.5 * (trace + root))
        minor = max(0.0, 0.5 * (trace - root))
        length = max(self.resolution, 2.0 * math.sqrt(3.0 * major))
        residual = math.sqrt(minor)
        return mean_x, mean_y, length, residual

    def occupied_components(
        self,
    ) -> Tuple[List[Set[Tuple[int, int]]], List[Set[Tuple[int, int]]]]:
        walls: List[Set[Tuple[int, int]]] = []
        obstacles: List[Set[Tuple[int, int]]] = []
        # Two-cell adjacency connects neighboring 2-degree LiDAR endpoints on
        # a wall while retaining separation between ordinary furniture.
        for component in self._components(self.occupied, neighbor_radius=2):
            if len(component) < 2:
                obstacles.append(component)
                continue
            _, _, length, residual = self._component_geometry(component)
            if length >= self.wall_min_length and residual <= self.wall_max_residual:
                walls.append(component)
            else:
                obstacles.append(component)
        return walls, obstacles

    def inflated_occupied(self) -> Set[Tuple[int, int]]:
        radius = int(math.ceil(self.inflation_radius / self.resolution))
        inflated: Set[Tuple[int, int]] = set()
        radius_sq = (self.inflation_radius / self.resolution) ** 2
        for cx, cy in self.occupied:
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx * dx + dy * dy <= radius_sq:
                        inflated.add((cx + dx, cy + dy))
        return inflated

    def _shadow_cells(self) -> Set[Tuple[int, int]]:
        if not self._scan_poses:
            return set()
        _, obstacles = self.occupied_components()
        shadows: Set[Tuple[int, int]] = set()
        # Use the latest three distinct sensing positions.  A cell exposed by
        # any later scan is removed below, so shadows naturally disappear when
        # the dog walks to a useful second viewpoint.
        for component in obstacles:
            center_x, center_y, length, _ = self._component_geometry(component)
            object_half_width = max(0.18, min(0.65, 0.5 * length + 0.12))
            for robot_x, robot_y in list(self._scan_poses)[-3:]:
                dx, dy = center_x - robot_x, center_y - robot_y
                distance = math.hypot(dx, dy)
                if distance < 0.5:
                    continue
                ux, uy = dx / distance, dy / distance
                vx, vy = -uy, ux
                along = self.resolution
                while along <= self.shadow_depth:
                    width = object_half_width + 0.10 * along
                    lateral = -width
                    while lateral <= width:
                        x = center_x + ux * along + vx * lateral
                        y = center_y + uy * along + vy * lateral
                        key = self.cell_key(x, y)
                        if self.inside_room(x, y):
                            shadows.add(key)
                        lateral += self.resolution
                    along += self.resolution
        shadows.difference_update(self.free)
        shadows.difference_update(self.occupied)
        return shadows

    def _boundary_band_value(
        self,
        values: Sequence[Tuple[float, float]],
        band_index: int,
        span_index: int,
        band_width: float,
        min_points: int = 8,
        min_span: float = 1.8,
    ) -> Optional[float]:
        """Return the strongest wall-band coordinate in door-local space."""
        bands: Dict[int, List[Tuple[float, float]]] = {}
        for value in values:
            band = int(round(value[band_index] / band_width))
            bands.setdefault(band, []).append(value)
        candidates = []
        for band, samples in bands.items():
            spans = [sample[span_index] for sample in samples]
            if len(samples) < min_points or max(spans) - min(spans) < min_span:
                continue
            candidates.append((len(samples), max(spans) - min(spans),
                               band * band_width))
        return max(candidates)[2] if candidates else None

    def boundary_geometry(self) -> Dict[str, Optional[float]]:
        """Estimate the three room limits from online occupied endpoints only."""
        points = [self.room_coordinates(*self.cell_center(key))
                  for key in self.occupied]
        far_points = [point for point in points
                      if point[0] >= 4.5 and abs(point[1]) <= self.half_width]
        left_points = [point for point in points
                       if point[1] >= 2.8 and 0.4 <= point[0] <= self.max_depth]
        right_points = [point for point in points
                        if point[1] <= -2.8 and 0.4 <= point[0] <= self.max_depth]
        band_width = max(0.30, 2.5 * self.resolution)
        return {
            "far_depth": self._boundary_band_value(
                far_points, 0, 1, band_width),
            "left_lateral": self._boundary_band_value(
                left_points, 1, 0, band_width),
            "right_lateral": self._boundary_band_value(
                right_points, 1, 0, band_width),
        }

    def room_mask(self) -> Set[Tuple[int, int]]:
        """Return the inferred room interior, including still-unknown cells.

        No mask is exposed until all three non-door walls are supported.  This
        is important: using the connected known-free component as the mask
        silently removes unknown corners from the denominator.
        """
        geometry = self.boundary_geometry()
        far = geometry["far_depth"]
        left = geometry["left_lateral"]
        right = geometry["right_lateral"]
        if far is None or left is None or right is None:
            return set()
        # Stay half a cell inside structural returns.  Furniture remains part
        # of the known numerator, while space hidden behind it remains unknown.
        depth_max = min(self.max_depth, float(far) - 0.5 * self.resolution)
        lateral_min = max(-self.half_width, float(right) + 0.5 * self.resolution)
        lateral_max = min(self.half_width, float(left) - 0.5 * self.resolution)
        if depth_max <= 0.4 or lateral_max <= lateral_min:
            return set()
        mask: Set[Tuple[int, int]] = set()
        depth = 0.5 * self.resolution
        normal_x, normal_y = math.cos(self.inward_yaw), math.sin(self.inward_yaw)
        tangent_x, tangent_y = -normal_y, normal_x
        while depth <= depth_max + 1e-9:
            lateral = lateral_min
            while lateral <= lateral_max + 1e-9:
                x = self.door_x + depth * normal_x + lateral * tangent_x
                y = self.door_y + depth * normal_y + lateral * tangent_y
                mask.add(self.cell_key(x, y))
                lateral += self.resolution
            depth += self.resolution
        return mask

    def unknown_cells(self) -> Set[Tuple[int, int]]:
        mask = self.room_mask()
        return mask - self.free - self.occupied

    def coverage_status(self) -> RoomCoverageStatus:
        shadows = self._shadow_cells()
        observed_free = {key for key in self.free if self.inside_room(*self.cell_center(key))}
        mask = self.room_mask()
        known = mask & (self.free | self.occupied)
        unknown = mask - known
        largest_unknown = max((len(component) for component in
                               self._components(unknown)), default=0)
        largest_shadow = max((len(component) for component in
                              self._components(shadows)), default=0)
        walls, obstacles = self.occupied_components()
        far_wall, left_wall, right_wall = self._boundary_evidence()
        boundary_complete = bool(far_wall and left_wall and right_wall and mask)
        return RoomCoverageStatus(
            lidar_ratio=(float(len(known)) / float(len(mask)) if mask else 0.0),
            free_cells=len(observed_free),
            shadow_cells=len(shadows),
            largest_shadow_m2=largest_shadow * self.resolution * self.resolution,
            wall_components=len(walls),
            obstacle_components=len(obstacles),
            far_wall_seen=far_wall,
            left_wall_seen=left_wall,
            right_wall_seen=right_wall,
            boundary_directions_seen=sum((far_wall, left_wall, right_wall)),
            known_cells=len(known),
            unknown_cells=len(unknown),
            largest_unknown_m2=(largest_unknown * self.resolution *
                                self.resolution),
            estimated_room_cells=len(mask),
            boundary_complete=boundary_complete,
        )

    def shadow_cells(self) -> Set[Tuple[int, int]]:
        """Return a copy of unresolved obstacle-shadow cells for diagnostics."""
        return set(self._shadow_cells())

    @staticmethod
    def _band_has_span(
        values: Sequence[Tuple[float, float]],
        band_index: int,
        span_index: int,
        band_width: float,
        min_points: int,
        min_span: float,
    ) -> bool:
        """Detect a populated, straight endpoint band without fitting truth walls."""
        bands: Dict[int, List[float]] = {}
        for value in values:
            band = int(round(value[band_index] / band_width))
            bands.setdefault(band, []).append(value[span_index])
        for samples in bands.values():
            if len(samples) >= min_points and max(samples) - min(samples) >= min_span:
                return True
        return False

    def _boundary_evidence(self) -> Tuple[bool, bool, bool]:
        """Return far/left/right wall evidence from LiDAR endpoints only.

        A compact obstacle may have many returns, so point count alone is not
        sufficient.  Boundary evidence requires endpoints to occupy a narrow
        depth/lateral band while spanning at least 1.8 m along the orthogonal
        axis.  The thresholds are conservative sensor-local limits, not room
        dimensions or simulator metadata.
        """
        points = [self.room_coordinates(*self.cell_center(key)) for key in self.occupied]
        far_points = [
            point
            for point in points
            if point[0] >= 4.5 and abs(point[1]) <= self.half_width
        ]
        left_points = [
            point
            for point in points
            if point[1] >= 2.8 and 0.4 <= point[0] <= self.max_depth
        ]
        right_points = [
            point
            for point in points
            if point[1] <= -2.8 and 0.4 <= point[0] <= self.max_depth
        ]
        band_width = max(0.30, 2.5 * self.resolution)
        far_wall = self._band_has_span(
            far_points, 0, 1, band_width, min_points=8, min_span=1.8
        )
        left_wall = self._band_has_span(
            left_points, 1, 0, band_width, min_points=8, min_span=1.8
        )
        right_wall = self._band_has_span(
            right_points, 1, 0, band_width, min_points=8, min_span=1.8
        )
        return far_wall, left_wall, right_wall

    def _nearest_key(
        self, x: float, y: float, candidates: Iterable[Tuple[int, int]]
    ) -> Optional[Tuple[int, int]]:
        best = None
        best_distance = None
        for key in candidates:
            cx, cy = self.cell_center(key)
            distance = math.hypot(cx - x, cy - y)
            if best_distance is None or distance < best_distance:
                best = key
                best_distance = distance
        return best

    def reachable_free(self, start_x: float, start_y: float) -> Set[Tuple[int, int]]:
        navigable = self.free - self.inflated_occupied()
        start = self._nearest_key(start_x, start_y, navigable)
        if start is None or math.hypot(
            self.cell_center(start)[0] - start_x,
            self.cell_center(start)[1] - start_y,
        ) > 0.9:
            return set()
        reached = {start}
        queue = deque([start])
        while queue:
            cx, cy = queue.popleft()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nxt = (cx + dx, cy + dy)
                    if nxt in navigable and nxt not in reached:
                        reached.add(nxt)
                        queue.append(nxt)
        return reached

    def nearest_safe_viewpoint(
        self,
        target_x: float,
        target_y: float,
        start_x: float,
        start_y: float,
        search_radius: float = 1.2,
    ) -> Optional[Tuple[float, float]]:
        reachable = self.reachable_free(start_x, start_y)
        # Sparse angular rays can leave one-cell cracks that disconnect a
        # perfectly clear straight route in the flood-fill graph.  Admit such
        # cells only when the full A1-inflated segment from the current pose is
        # collision-free; this preserves safety without falling back to a
        # shallow doorway viewpoint merely because of rasterization.
        navigable = self.free - self.inflated_occupied()
        blocked = self.inflated_occupied()
        line_reachable: Set[Tuple[int, int]] = set()
        for key in navigable:
            cx, cy = self.cell_center(key)
            if math.hypot(cx - target_x, cy - target_y) > search_radius:
                continue
            ray = self._ray_cells(start_x, start_y, cx, cy)
            if not any(ray_key in blocked for ray_key in ray[1:]):
                line_reachable.add(key)
        best_key = self._nearest_key(target_x, target_y, reachable | line_reachable)
        if best_key is None:
            return None
        x, y = self.cell_center(best_key)
        if math.hypot(x - target_x, y - target_y) > search_radius:
            return None
        return x, y

    def first_viewpoint(
        self, start_x: float, start_y: float, preferred_depth: float = 2.4
    ) -> Optional[Tuple[float, float]]:
        target_x = self.door_x + preferred_depth * math.cos(self.inward_yaw)
        target_y = self.door_y + preferred_depth * math.sin(self.inward_yaw)
        viewpoint = self.nearest_safe_viewpoint(
            target_x, target_y, start_x, start_y, search_radius=1.4
        )
        if viewpoint is not None:
            return viewpoint
        fallback_x = self.door_x + 1.0 * math.cos(self.inward_yaw)
        fallback_y = self.door_y + 1.0 * math.sin(self.inward_yaw)
        return self.nearest_safe_viewpoint(
            fallback_x, fallback_y, start_x, start_y, search_radius=0.8
        )

    def _line_visible(
        self, start: Tuple[int, int], target: Tuple[int, int]
    ) -> bool:
        sx, sy = self.cell_center(start)
        tx, ty = self.cell_center(target)
        ray = self._ray_cells(sx, sy, tx, ty)
        for key in ray[1:-1]:
            if key in self.occupied:
                return False
        return True

    def best_viewpoint(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        visited: Sequence[Tuple[float, float]],
        min_gain: float = 0.08,
    ) -> Optional[RoomViewpoint]:
        shadows = self._shadow_cells()
        if not shadows:
            return None
        shadow_components = self._components(shadows)
        main_shadow = max(shadow_components, key=len)
        largest_shadow_m2 = len(main_shadow) * self.resolution * self.resolution
        reachable = self.reachable_free(robot_x, robot_y)
        if not reachable:
            return None
        total_evidence = max(1, len(self.free) + len(shadows))
        best: Optional[RoomViewpoint] = None
        # A coarse lattice is enough for A1-sized motion and avoids scoring
        # thousands of almost identical 0.15 m cells.
        stride = max(1, int(round(0.45 / self.resolution)))
        for key in reachable:
            if key[0] % stride or key[1] % stride:
                continue
            x, y = self.cell_center(key)
            path_cost = math.hypot(x - robot_x, y - robot_y)
            if path_cost < 0.65 or path_cost > 3.5:
                continue
            if any(math.hypot(x - vx, y - vy) < 0.75 for vx, vy in visited):
                continue
            visible = [shadow for shadow in shadows if self._line_visible(key, shadow)]
            if not visible:
                continue
            gain = len(visible) / float(total_evidence)
            main_visible = sum(1 for shadow in main_shadow if shadow in visible)
            debt_candidate = shadow_debt_candidate_allowed(
                largest_shadow_m2,
                main_visible * self.resolution * self.resolution,
                main_visible / float(max(1, len(main_shadow))),
            )
            if gain < min_gain and not debt_candidate:
                continue
            mean_x = sum(self.cell_center(cell)[0] for cell in visible) / len(visible)
            mean_y = sum(self.cell_center(cell)[1] for cell in visible) / len(visible)
            yaw = math.atan2(mean_y - y, mean_x - x)
            yaw_cost = abs(angle_diff(yaw, robot_yaw)) / math.pi
            # Gain dominates; distance and yaw break ties and suppress loops.
            score = 0.72 * gain - 0.055 * path_cost - 0.06 * yaw_cost
            decision = RoomViewpoint(
                x=x,
                y=y,
                yaw=yaw,
                expected_gain=gain,
                shadow_cells=len(visible),
                path_cost=path_cost,
                score=score,
                reason=(
                    "major_shadow_debt" if gain < min_gain else "compact_occlusion"
                ),
            )
            if best is None or decision.score > best.score:
                best = decision
        return best

    def boundary_recovery_viewpoint(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        visited: Sequence[Tuple[float, float]],
        preferred_depth: float,
    ) -> Optional[RoomViewpoint]:
        """Choose a safe deeper/offset point when room boundaries are missing."""
        status = self.coverage_status()
        diversity_complete = room_viewpoint_diversity_gate(
            visited,
            self.door_x,
            self.door_y,
            self.inward_yaw,
        )
        if (
            status.far_wall_seen
            and status.boundary_directions_seen >= 2
            and diversity_complete
        ):
            return None
        current_depth, current_lateral = self.room_coordinates(robot_x, robot_y)
        desired_depth = min(
            self.max_depth - 0.8,
            max(preferred_depth + 0.8, current_depth + 0.9, 3.8),
        )
        if not status.left_wall_seen and not status.right_wall_seen:
            # Alternate sides across visits instead of repeatedly walking the
            # centre line, which cannot resolve long furniture shadows well.
            mean_lateral = current_lateral
            if visited:
                lateral_values = [self.room_coordinates(vx, vy)[1] for vx, vy in visited]
                mean_lateral = sum(lateral_values) / len(lateral_values)
            offsets = [1.6, -1.6, 0.0] if mean_lateral <= 0.0 else [-1.6, 1.6, 0.0]
        elif not status.left_wall_seen:
            offsets = [1.6, 0.7, -0.7]
        elif not status.right_wall_seen:
            offsets = [-1.6, -0.7, 0.7]
        else:
            mean_lateral = current_lateral
            if visited:
                lateral_values = [self.room_coordinates(vx, vy)[1] for vx, vy in visited]
                mean_lateral = sum(lateral_values) / len(lateral_values)
            offsets = [1.6, -1.6, 0.0] if mean_lateral <= 0.0 else [-1.6, 1.6, 0.0]

        best: Optional[RoomViewpoint] = None
        inward_cos = math.cos(self.inward_yaw)
        inward_sin = math.sin(self.inward_yaw)
        for lateral in offsets:
            target_x = self.door_x + desired_depth * inward_cos - lateral * inward_sin
            target_y = self.door_y + desired_depth * inward_sin + lateral * inward_cos
            safe = self.nearest_safe_viewpoint(
                target_x, target_y, robot_x, robot_y, search_radius=1.5
            )
            if safe is None:
                continue
            if not viewpoint_candidate_is_diverse(
                safe,
                visited,
                self.door_x,
                self.door_y,
                self.inward_yaw,
            ):
                continue
            path_cost = math.hypot(safe[0] - robot_x, safe[1] - robot_y)
            if path_cost < 0.65 or path_cost > 3.8:
                continue
            target_yaw = math.atan2(
                self.door_y + self.max_depth * inward_sin - safe[1],
                self.door_x + self.max_depth * inward_cos - safe[0],
            )
            missing = 3 - status.boundary_directions_seen
            forced_diversity = not diversity_complete
            score = 0.45 + 0.12 * missing - 0.045 * path_cost
            candidate = RoomViewpoint(
                x=safe[0],
                y=safe[1],
                yaw=target_yaw,
                expected_gain=max(0.08, 0.12 * missing),
                shadow_cells=status.shadow_cells,
                path_cost=path_cost,
                score=score,
                reason=(
                    "spatial_diversity" if forced_diversity else "missing_room_boundary"
                ),
            )
            if best is None or candidate.score > best.score:
                best = candidate
        return best

    def scan_headings(
        self,
        x: float,
        y: float,
        current_yaw: float,
        interest_yaws: Sequence[float] = (),
        camera_fov: float = math.radians(60.0),
    ) -> List[float]:
        """Return one monotonic interior sweep plus non-redundant interests."""
        # Two overlapping 60-degree views cover the central 120-degree room
        # sector. Danger and material occlusion headings supplement them.
        headings: List[float] = [
            self.inward_yaw - 0.52,
            self.inward_yaw + 0.52,
        ]
        headings.extend(interest_yaws)
        for component in self._components(self._shadow_cells()):
            if len(component) * self.resolution * self.resolution < 0.08:
                continue
            cx = sum(self.cell_center(key)[0] for key in component) / len(component)
            cy = sum(self.cell_center(key)[1] for key in component) / len(component)
            headings.append(math.atan2(cy - y, cx - x))
        unique: List[float] = []
        merge_angle = max(0.25, camera_fov * 0.55)
        for heading in headings:
            heading = wrap_yaw(heading)
            # Do not spend room time looking back through the doorway.
            if math.cos(angle_diff(heading, self.inward_yaw)) <= 0.02:
                continue
            if not any(abs(angle_diff(heading, existing)) < merge_angle for existing in unique):
                unique.append(heading)
        # Sort in the room-relative half-plane and make a single sweep.  Pick
        # its direction by the first-turn cost; this avoids greedy left-right
        # bouncing and saves rotation time.
        unique.sort(key=lambda item: angle_diff(item, self.inward_yaw))
        ascending = unique[:5]
        descending = list(reversed(unique[-5:]))
        asc_cost = abs(angle_diff(ascending[0], current_yaw)) if ascending else math.inf
        desc_cost = abs(angle_diff(descending[0], current_yaw)) if descending else math.inf
        return ascending if asc_cost <= desc_cost else descending


@dataclass
class FrontierInformationStructure:
    """Incremental FIS maintained from successive radar scans."""

    sector_count: int = 36
    camera_range: float = 7.5
    camera_fov: float = math.radians(60.0)
    radar_max_range: float = 20.0
    near_range: float = 4.0
    mid_range: float = 9.0
    merge_yaw: float = math.radians(25.0)
    stale_seconds: float = 8.0
    min_points: int = 3
    _next_id: int = 1
    _sectors: List[Tuple[int, float]] = field(default_factory=list)
    _clusters: Dict[int, FrontierCluster] = field(default_factory=dict)
    _camera_seen: set = field(default_factory=set)
    _lidar_covered: set = field(default_factory=set)
    _history: Deque[Tuple[float, float, float]] = field(
        default_factory=lambda: deque(maxlen=64)
    )

    def __post_init__(self) -> None:
        if not self._sectors:
            self._sectors = [(0, 0.0)] * self.sector_count

    def cell_key(self, x: float, y: float, cell_size: float = 3.0) -> Tuple[int, int]:
        return int(math.floor(x / cell_size)), int(math.floor(y / cell_size))

    def heading_bin(self, yaw: float) -> int:
        normalized = (yaw + 2.0 * math.pi) % (2.0 * math.pi)
        return int(normalized / (2.0 * math.pi) * self.sector_count) % self.sector_count

    def mark_camera_view(
        self, x: float, y: float, yaw: float, cell_size: float = 3.0
    ) -> None:
        cell = self.cell_key(x, y, cell_size)
        half = max(
            1,
            int(
                math.ceil(
                    self.camera_fov / (2.0 * math.pi) * self.sector_count / 2.0
                )
            ),
        )
        center = self.heading_bin(yaw)
        for offset in range(-half, half + 1):
            self._camera_seen.add(
                (cell[0], cell[1], (center + offset) % self.sector_count)
            )

    def mark_lidar_coverage(
        self,
        x: float,
        y: float,
        robot_yaw: float,
        ranges_by_sector: Sequence[Tuple[int, float]],
        cell_size: float = 1.5,
        max_mark_range: float = 6.0,
    ) -> None:
        """Mark map cells already cleared by LiDAR free-space (no need to walk there)."""
        for index, (count, distance) in enumerate(ranges_by_sector):
            if count < self.min_points:
                continue
            reach = min(max(distance - 0.35, 0.0), max_mark_range)
            if reach < 1.0:
                continue
            body_ang = (index + 0.5) * 2.0 * math.pi / self.sector_count
            # PointCloud yaw is atan2(y,x); convert to world.
            world_ang = wrap_yaw(robot_yaw + body_ang)
            step = cell_size * 0.8
            t = cell_size
            while t < reach:
                wx = x + t * math.cos(world_ang)
                wy = y + t * math.sin(world_ang)
                self._lidar_covered.add(self.cell_key(wx, wy, cell_size))
                t += step

    def is_lidar_covered(self, x: float, y: float, cell_size: float = 1.5) -> bool:
        return self.cell_key(x, y, cell_size) in self._lidar_covered

    def is_camera_heading_seen(
        self, x: float, y: float, yaw: float, cell_size: float = 1.5
    ) -> bool:
        cell = self.cell_key(x, y, cell_size)
        return (cell[0], cell[1], self.heading_bin(yaw)) in self._camera_seen

    def lidar_coverage_count(self) -> int:
        return len(self._lidar_covered)

    def local_coverage_ratio(
        self, x: float, y: float, radius: float = 4.0, cell_size: float = 1.5
    ) -> float:
        """Fraction of cells in a disk around the robot already lidar-covered."""
        n = max(1, int(math.ceil(radius / cell_size)))
        total = 0
        covered = 0
        for ix in range(-n, n + 1):
            for iy in range(-n, n + 1):
                cx = x + (ix + 0.5) * cell_size
                cy = y + (iy + 0.5) * cell_size
                if math.hypot(cx - x, cy - y) > radius:
                    continue
                total += 1
                if self.is_lidar_covered(cx, cy, cell_size):
                    covered += 1
        return float(covered) / float(max(total, 1))

    def best_gap_target(
        self,
        robot_yaw: float,
        prefer_side: int = 0,
        min_width: float = 0.35,
        min_range: float = 2.0,
    ) -> Optional[GapTarget]:
        """Pick the most promising free opening (door/aisle) relative to body."""
        if not self._sectors:
            return None
        open_bins = []
        for index, (count, distance) in enumerate(self._sectors):
            if count >= self.min_points and distance >= min_range:
                open_bins.append(index)
        best: Optional[GapTarget] = None
        for group in self._group_contiguous(open_bins):
            width = (len(group) / float(self.sector_count)) * 2.0 * math.pi
            if width < min_width:
                continue
            local_yaw = self._group_yaw(group)
            avg_range = sum(self._sectors[i][1] for i in group) / float(len(group))
            # Prefer side openings over dead-ahead when exploring rooms/corridor doors.
            side_bonus = abs(math.sin(local_yaw)) * 0.35
            prefer_bonus = 0.0
            if prefer_side < 0 and local_yaw > 0.4:
                prefer_bonus = 0.25
            elif prefer_side > 0 and local_yaw < -0.4:
                prefer_bonus = 0.25
            # Penalize revisiting headings already RGB-covered heavily.
            score = 0.45 * min(avg_range / 8.0, 1.0) + 0.35 * min(width / 1.2, 1.0)
            score += side_bonus + prefer_bonus
            # Mild preference to keep some forward progress.
            score += 0.10 * max(0.0, math.cos(local_yaw))
            world_yaw = wrap_yaw(robot_yaw + local_yaw)
            cand = GapTarget(
                yaw=world_yaw, distance=avg_range, width=width, score=score
            )
            if best is None or cand.score > best.score:
                best = cand
        return best

    @staticmethod
    def detect_sphere_candidates(
        points_xy: Sequence[Tuple[float, float]],
        robot_yaw: float,
        min_range: float = 1.0,
        max_range: float = 6.5,
        diameter_min: float = 0.08,
        diameter_max: float = 0.55,
    ) -> List[SphereCandidate]:
        """Cluster compact isolated returns that look like ~0.3m spheres."""
        pts = []
        for x, y in points_xy:
            r = math.hypot(x, y)
            if min_range <= r <= max_range:
                pts.append((x, y, r, math.atan2(y, x)))
        if len(pts) < 6:
            return []
        pts.sort(key=lambda item: item[3])
        # Greedy angular clustering.
        clusters: List[List[Tuple[float, float, float, float]]] = []
        current = [pts[0]]
        for item in pts[1:]:
            if abs(angle_diff(item[3], current[-1][3])) < 0.12:
                current.append(item)
            else:
                if len(current) >= 4:
                    clusters.append(current)
                current = [item]
        if len(current) >= 4:
            clusters.append(current)

        candidates: List[SphereCandidate] = []
        for group in clusters:
            xs = [p[0] for p in group]
            ys = [p[1] for p in group]
            rs = [p[2] for p in group]
            angs = [p[3] for p in group]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            dist = math.hypot(cx, cy)
            span = abs(angle_diff(max(angs), min(angs)))
            # Handle wrap: use chord diameter from std of positions.
            diam = 2.0 * math.sqrt(
                (sum((x - cx) ** 2 for x in xs) + sum((y - cy) ** 2 for y in ys))
                / max(len(group), 1)
            )
            diam = max(diam, dist * span)
            if not (diameter_min <= diam <= diameter_max):
                continue
            # Isolation: mean range of neighbors should be farther (object sticks out).
            mean_r = sum(rs) / len(rs)
            if mean_r < min_range:
                continue
            local_yaw = math.atan2(cy, cx)
            world_yaw = wrap_yaw(robot_yaw + local_yaw)
            # Prefer ~0.15-0.35m objects at mid range (danger sphere size).
            size_score = 1.0 - min(abs(diam - 0.22) / 0.25, 1.0)
            range_score = 1.0 - min(abs(mean_r - 3.0) / 3.5, 1.0)
            score = 0.55 * size_score + 0.35 * range_score + 0.10 * min(len(group) / 20.0, 1.0)
            if score < 0.35:
                continue
            candidates.append(
                SphereCandidate(
                    yaw=world_yaw,
                    distance=mean_r,
                    diameter=diam,
                    point_count=len(group),
                    score=score,
                )
            )
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[:5]

    def note_camera_visit(self, yaw: float, cluster_id: Optional[int] = None) -> None:
        """Record that the RGB camera completed a dwell on a viewpoint."""
        if cluster_id is not None and cluster_id in self._clusters:
            cluster = self._clusters[cluster_id]
            cluster.camera_visits += 1
            if cluster.camera_visits >= 2 or cluster.average_range <= self.near_range:
                cluster.covered = True
            return
        for cluster in self._clusters.values():
            if abs(angle_diff(cluster.yaw, yaw)) <= 0.5 * self.camera_fov:
                cluster.camera_visits += 1
                if cluster.camera_visits >= 2:
                    cluster.covered = True

    def update_from_radar(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        ranges_by_sector: Sequence[Tuple[int, float]],
        stamp: float,
    ) -> None:
        """Incrementally refresh sector stats and merge into frontier clusters.

        ranges_by_sector: list of (point_count, range_percentile) in robot frame.
        """
        if len(ranges_by_sector) != self.sector_count:
            # Resize gracefully if caller changes resolution.
            self.sector_count = len(ranges_by_sector)
            self._sectors = list(ranges_by_sector)
        else:
            self._sectors = list(ranges_by_sector)

        self._history.append((robot_x, robot_y, robot_yaw))
        self._expire_stale(stamp)

        # Detect contiguous free/open sectors as frontier candidates.
        open_bins: List[int] = []
        for index, (count, distance) in enumerate(self._sectors):
            if count >= self.min_points and 1.0 <= distance <= self.radar_max_range:
                open_bins.append(index)

        groups = self._group_contiguous(open_bins)
        for group in groups:
            local_yaw = self._group_yaw(group)
            global_yaw = wrap_yaw(robot_yaw + local_yaw)
            avg_range = sum(self._sectors[i][1] for i in group) / float(len(group))
            point_count = sum(self._sectors[i][0] for i in group)
            width = (len(group) / float(self.sector_count)) * 2.0 * math.pi
            self._upsert_cluster(
                yaw=global_yaw,
                distance=avg_range,
                point_count=point_count,
                average_range=avg_range,
                angular_width=width,
                stamp=stamp,
            )

    def _expire_stale(self, stamp: float) -> None:
        stale_ids = [
            cluster_id
            for cluster_id, cluster in self._clusters.items()
            if stamp - cluster.last_update > self.stale_seconds
        ]
        for cluster_id in stale_ids:
            del self._clusters[cluster_id]

    def _group_contiguous(self, bins: Sequence[int]) -> List[List[int]]:
        if not bins:
            return []
        ordered = sorted(set(bins))
        # Handle wrap-around at 0/N.
        groups: List[List[int]] = []
        current = [ordered[0]]
        for value in ordered[1:]:
            if value == current[-1] + 1:
                current.append(value)
            else:
                groups.append(current)
                current = [value]
        groups.append(current)
        if (
            len(groups) >= 2
            and groups[0][0] == 0
            and groups[-1][-1] == self.sector_count - 1
        ):
            wrapped = groups[-1] + groups[0]
            groups = groups[1:-1] + [wrapped]
        return groups

    def _group_yaw(self, group: Sequence[int]) -> float:
        # Circular mean of sector mid-angles.
        sx = 0.0
        sy = 0.0
        for index in group:
            angle = (index + 0.5) * 2.0 * math.pi / self.sector_count
            sx += math.cos(angle)
            sy += math.sin(angle)
        return math.atan2(sy, sx)

    def _upsert_cluster(
        self,
        yaw: float,
        distance: float,
        point_count: int,
        average_range: float,
        angular_width: float,
        stamp: float,
    ) -> None:
        best_id = None
        best_err = None
        for cluster_id, cluster in self._clusters.items():
            err = abs(angle_diff(yaw, cluster.yaw))
            if err <= self.merge_yaw and (best_err is None or err < best_err):
                best_id = cluster_id
                best_err = err
        if best_id is None:
            cluster_id = self._next_id
            self._next_id += 1
            self._clusters[cluster_id] = FrontierCluster(
                cluster_id=cluster_id,
                yaw=yaw,
                distance=distance,
                point_count=point_count,
                average_range=average_range,
                angular_width=angular_width,
                last_update=stamp,
            )
            return
        cluster = self._clusters[best_id]
        alpha = 0.35
        cluster.yaw = wrap_yaw(
            cluster.yaw + alpha * angle_diff(yaw, cluster.yaw)
        )
        cluster.distance = (1.0 - alpha) * cluster.distance + alpha * distance
        cluster.average_range = (
            (1.0 - alpha) * cluster.average_range + alpha * average_range
        )
        cluster.point_count = max(cluster.point_count, point_count)
        cluster.angular_width = max(cluster.angular_width, angular_width)
        cluster.last_update = stamp
        cluster.hits += 1

    def layer_for_distance(self, distance: float) -> str:
        if distance <= self.near_range:
            return "near"
        if distance <= self.mid_range:
            return "mid"
        return "far"

    def hold_time_for(self, distance: float, layer: str) -> float:
        # FUEL local refine: nearby viewpoints need less dwell; farther ones
        # need more frames because the sphere subtends fewer pixels.
        ratio = min(max(distance / self.camera_range, 0.0), 1.0)
        if layer == "near":
            return 0.25 + 0.20 * ratio
        if layer == "mid":
            return 0.40 + 0.35 * ratio
        # Far clusters are radar-tracked only; if we still turn, keep it short.
        return 0.20

    def select_camera_viewpoint(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        cell_size: float = 3.0,
    ) -> Optional[ViewpointDecision]:
        """Hierarchical local viewpoint selection for the RGB camera."""
        cell = self.cell_key(robot_x, robot_y, cell_size)
        best: Optional[ViewpointDecision] = None
        for cluster in self._clusters.values():
            if cluster.covered:
                continue
            layer = self.layer_for_distance(cluster.average_range)
            # Far layer: radar already sees it; do not spend camera time yet.
            if layer == "far":
                continue
            if cluster.average_range > self.camera_range * 1.05:
                continue
            global_bin = self.heading_bin(cluster.yaw)
            if (cell[0], cell[1], global_bin) in self._camera_seen:
                continue

            gain = cluster.coverage_gain(self.camera_range, self.camera_fov)
            cost = cluster.view_cost(robot_yaw)
            # Prefer mid-range surfaces that are still within RealSense reach,
            # then near occlusions behind furniture.
            layer_bonus = 0.20 if layer == "mid" else 0.10
            density = min(cluster.point_count / 40.0, 1.0)
            score = (
                0.55 * (gain / max(self.camera_range ** 2 * self.camera_fov, 1e-3))
                + 0.20 * density
                + layer_bonus
                - cost
            )
            if best is None or score > best.score:
                # Mid-range: refine while moving. Near: short stationary dwell
                # for occlusion clearance (furniture / doorway shadows).
                preempt = "stop_hold" if layer == "near" else "yaw_blend"
                best = ViewpointDecision(
                    score=score,
                    yaw=cluster.yaw,
                    distance=min(cluster.average_range, self.camera_range),
                    layer=layer,
                    hold_seconds=self.hold_time_for(cluster.average_range, layer),
                    cluster_id=cluster.cluster_id,
                    reason="fis_{}_{:.1f}m".format(layer, cluster.average_range),
                    preempt_mode=preempt,
                )
        return best

    def active_clusters(self) -> List[FrontierCluster]:
        return sorted(
            self._clusters.values(),
            key=lambda item: item.average_range,
        )

    def summary(self) -> Dict[str, int]:
        layers = {"near": 0, "mid": 0, "far": 0, "covered": 0}
        for cluster in self._clusters.values():
            if cluster.covered:
                layers["covered"] += 1
            else:
                layers[self.layer_for_distance(cluster.average_range)] += 1
        return layers
