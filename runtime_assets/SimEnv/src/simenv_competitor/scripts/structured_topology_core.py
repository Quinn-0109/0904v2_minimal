#!/usr/bin/env python3
"""ROS-free geometry and state logic for structured indoor exploration.

The online controller deliberately builds its corridor/door/room topology from
an occupancy grid.  There are no layout coordinates or expected room counts in
this module.  Unknown cells are never accepted as traversable space.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
try:
    import cv2
except ImportError:  # pragma: no cover - diagnosed explicitly by the detector
    cv2 = None


Point = Tuple[float, float]


def wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def angle_distance(a: float, b: float) -> float:
    """Undirected line-angle distance in [0, pi/2]."""
    delta = abs(wrap_angle(a - b))
    return min(delta, abs(math.pi - delta))


def unit(angle: float) -> Point:
    return math.cos(angle), math.sin(angle)


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1])


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def interpolate_polyline(path: Sequence[Point], spacing: float = 0.075) -> List[Point]:
    if not path:
        return []
    result = [(float(path[0][0]), float(path[0][1]))]
    for start, end in zip(path[:-1], path[1:]):
        length = distance(start, end)
        count = max(1, int(math.ceil(length / max(1e-3, float(spacing)))))
        for index in range(1, count + 1):
            alpha = index / float(count)
            result.append((
                float(start[0]) + alpha * (float(end[0]) - float(start[0])),
                float(start[1]) + alpha * (float(end[1]) - float(start[1])),
            ))
    return result


@dataclass
class OccupancyMap:
    """A nav_msgs/OccupancyGrid-like array: -1 unknown, 0 free, 100 occupied."""

    data: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.int16)
        if self.data.ndim != 2 or self.resolution <= 0.0:
            raise ValueError("invalid occupancy map")

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    def world_to_cell(self, point: Sequence[float]) -> Optional[Tuple[int, int]]:
        ix = int(math.floor((float(point[0]) - self.origin_x) / self.resolution))
        iy = int(math.floor((float(point[1]) - self.origin_y) / self.resolution))
        if 0 <= ix < self.width and 0 <= iy < self.height:
            return ix, iy
        return None

    def cell_to_world(self, cell: Sequence[int]) -> Point:
        return (
            self.origin_x + (int(cell[0]) + 0.5) * self.resolution,
            self.origin_y + (int(cell[1]) + 0.5) * self.resolution,
        )

    def state(self, point: Sequence[float]) -> int:
        cell = self.world_to_cell(point)
        return -1 if cell is None else int(self.data[cell[1], cell[0]])

    def is_free(self, point: Sequence[float]) -> bool:
        return self.state(point) == 0

    def has_occupied_near(self, point: Sequence[float], radius: float) -> bool:
        cell = self.world_to_cell(point)
        if cell is None:
            return True
        cells = int(math.ceil(float(radius) / self.resolution))
        x0, y0 = cell
        for dy in range(-cells, cells + 1):
            for dx in range(-cells, cells + 1):
                if dx * dx + dy * dy > cells * cells:
                    continue
                x, y = x0 + dx, y0 + dy
                if not (0 <= x < self.width and 0 <= y < self.height):
                    return True
                if self.data[y, x] >= 50:
                    return True
        return False

    def footprint_safe(
        self,
        point: Sequence[float],
        yaw: float,
        length: float = 0.70,
        width: float = 0.44,
        margin: float = 0.05,
        unknown_is_safe: bool = False,
    ) -> bool:
        """Check a sampled oriented A1 footprint.

        Unknown is unsafe unless the caller has separate occupancy evidence
        for a tightly bounded region (the manager uses this only where the
        robot is already standing at the beginning of a path).
        """
        half_l = 0.5 * float(length) + float(margin)
        half_w = 0.5 * float(width) + float(margin)
        step = max(0.04, min(self.resolution * 0.5, 0.08))
        c, s = math.cos(yaw), math.sin(yaw)
        along = np.arange(-half_l, half_l + step * 0.5, step)
        across = np.arange(-half_w, half_w + step * 0.5, step)
        for forward in along:
            for lateral in across:
                sample = (
                    float(point[0]) + c * forward - s * lateral,
                    float(point[1]) + s * forward + c * lateral,
                )
                state = self.state(sample)
                if state >= 50 or (state < 0 and not unknown_is_safe):
                    return False
        return True


@dataclass
class CorridorRegion:
    corridor_id: str
    centerline: List[Point]
    principal_direction: float
    estimated_width: float
    forward_extent: float
    left_wall: List[Point]
    right_wall: List[Point]
    confidence: float
    confirmed: bool = False
    observations: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Doorway:
    door_id: str
    center: Point
    normal_direction: float
    width: float
    left_frame_point: Point
    right_frame_point: Point
    corridor_side: Point
    interior_side: Point
    confidence: float
    confirmed: bool = False
    traversable: bool = False
    visited: bool = False
    attempt_count: int = 0
    observations: int = 1
    depth_confirmations: int = 0
    adjacent_region_id: Optional[str] = None
    estimated_unknown_size: float = 0.0
    blacklist_until: float = 0.0
    corridor_id: Optional[str] = None
    source_room_candidate_id: Optional[str] = None
    geometry_locked: bool = False
    last_validated_grid_sequence: int = -1
    channel_validation: dict = field(default_factory=dict)

    @property
    def normal(self) -> Point:
        return unit(self.normal_direction)

    @property
    def tangent(self) -> Point:
        nx, ny = self.normal
        return -ny, nx

    def signed_depth(self, point: Sequence[float]) -> float:
        return dot((float(point[0]) - self.center[0], float(point[1]) - self.center[1]),
                   self.normal)

    def lateral_offset(self, point: Sequence[float]) -> float:
        return dot((float(point[0]) - self.center[0], float(point[1]) - self.center[1]),
                   self.tangent)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RoomExpansionObservation:
    """Layout-free evidence that a corridor side opens into a large space."""

    candidate_id: str
    corridor_id: str
    side: int
    centroid: Point
    core_area: float
    longitudinal_min: float
    longitudinal_max: float
    lateral_expansion_depth: float
    maximum_clearance: float
    expansion_ratio: float
    geometry_confidence: float
    portal_center: Optional[Point] = None
    portal_width: float = 0.0
    portal_normal_direction: float = 0.0
    portal_confidence: float = 0.0
    observations: int = 1
    depth_confirmations: int = 0
    depth_confidence: float = 0.0
    fused_confidence: float = 0.0
    confirmed: bool = False
    door_id: Optional[str] = None
    channel_validation: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["side_name"] = "left" if self.side > 0 else "right"
        return payload


def doorway_from_room_candidate(
    candidate: RoomExpansionObservation,
    parameters: Optional["GeometryParameters"] = None,
) -> Optional[Doorway]:
    """Convert a geometrically confirmed room portal into a door hypothesis.

    This function only constructs geometry.  ``confirmed`` and ``traversable``
    deliberately remain false until the online manager has validated the full
    outside-to-inside channel against both the 2-D projection and OctoMap.
    """
    if not candidate.confirmed or candidate.portal_center is None:
        return None
    p = parameters or GeometryParameters()
    width = float(candidate.portal_width)
    if not (p.door_width_min <= width <= p.door_width_max):
        return None
    nx, ny = unit(candidate.portal_normal_direction)
    tx, ty = -ny, nx
    center = (float(candidate.portal_center[0]),
              float(candidate.portal_center[1]))
    half_width = 0.5 * width
    return Doorway(
        door_id=candidate.door_id or "candidate_door",
        center=center,
        normal_direction=float(candidate.portal_normal_direction),
        width=width,
        left_frame_point=(center[0] - half_width * tx,
                          center[1] - half_width * ty),
        right_frame_point=(center[0] + half_width * tx,
                           center[1] + half_width * ty),
        corridor_side=(center[0] - p.door_outside_distance * nx,
                       center[1] - p.door_outside_distance * ny),
        interior_side=(center[0] + p.door_inside_distance * nx,
                       center[1] + p.door_inside_distance * ny),
        confidence=float(candidate.fused_confidence),
        confirmed=False,
        traversable=False,
        observations=max(p.door_confirmation_frames, candidate.observations),
        depth_confirmations=candidate.depth_confirmations,
        adjacent_region_id="room_" + candidate.candidate_id,
        estimated_unknown_size=float(candidate.core_area),
        corridor_id=candidate.corridor_id,
        source_room_candidate_id=candidate.candidate_id,
    )


def align_candidate_door_to_corridor(
    door: Doorway,
    candidate: RoomExpansionObservation,
    corridor: CorridorRegion,
    parameters: Optional["GeometryParameters"] = None,
) -> Doorway:
    """Project a room portal onto the candidate's observed corridor side.

    A room-expansion centroid can pull a raw portal normal diagonally along the
    hall.  Using that diagonal as the room half-plane makes the downstream
    corridor look like room interior.  The corridor detector already provides
    the wall axis and width, while ``candidate.side`` identifies which wall
    opened.  Preserve the sensor-derived longitudinal station and width, but
    make the door normal perpendicular to the corridor and place its centre on
    the corresponding side wall.
    """
    if len(corridor.centerline) < 2 or candidate.side not in (-1, 1):
        return door
    p = parameters or GeometryParameters()
    axis = unit(corridor.principal_direction)
    left_normal = (-axis[1], axis[0])
    inward = (candidate.side * left_normal[0],
              candidate.side * left_normal[1])
    origin = corridor.centerline[0]
    station = dot((door.center[0] - origin[0],
                   door.center[1] - origin[1]), axis)
    centerline_point = (origin[0] + station * axis[0],
                        origin[1] + station * axis[1])
    center = (centerline_point[0] + 0.5 * corridor.estimated_width * inward[0],
              centerline_point[1] + 0.5 * corridor.estimated_width * inward[1])
    normal_direction = math.atan2(inward[1], inward[0])
    tangent = (-inward[1], inward[0])
    half_width = 0.5 * door.width
    payload = door.to_dict()
    payload.update({
        "center": center,
        "normal_direction": normal_direction,
        "left_frame_point": (center[0] - half_width * tangent[0],
                             center[1] - half_width * tangent[1]),
        "right_frame_point": (center[0] + half_width * tangent[0],
                              center[1] + half_width * tangent[1]),
        "corridor_side": (center[0] - p.door_outside_distance * inward[0],
                          center[1] - p.door_outside_distance * inward[1]),
        "interior_side": (center[0] + p.door_inside_distance * inward[0],
                          center[1] + p.door_inside_distance * inward[1]),
    })
    return Doorway(**payload)


class RoomState(str, Enum):
    UNVISITED = "UNVISITED"
    ENTERING = "ENTERING"
    EXPLORING = "EXPLORING"
    COMPLETED = "COMPLETED"
    TEMPORARILY_UNREACHABLE = "TEMPORARILY_UNREACHABLE"


@dataclass
class RoomRegion:
    region_id: str
    entry_door_id: str
    estimated_boundary: List[Point] = field(default_factory=list)
    observed_free_area: float = 0.0
    visited_free_area: float = 0.0
    unknown_volume: float = 0.0
    local_goal_count: int = 0
    state: RoomState = RoomState.UNVISITED
    maximum_inside_depth: float = 0.0
    expansion_confidence: float = 0.0
    no_goal_cycles: int = 0
    successful_goals: List[Point] = field(default_factory=list)
    map_growth_history: List[float] = field(default_factory=list)
    visited_growth_history: List[float] = field(default_factory=list)
    last_map_sequence: int = -1
    no_goal_map_sequences: List[int] = field(default_factory=list)
    unreachable_attempt_count: int = 0
    unreachable_confirmed: bool = False
    unreachable_reason: Optional[str] = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["state"] = self.state.value
        return payload


@dataclass
class GeometryParameters:
    corridor_width_min: float = 1.0
    corridor_width_max: float = 4.0
    corridor_forward_depth_min: float = 4.0
    corridor_width_variation: float = 0.45
    parallel_wall_angle_tolerance_deg: float = 15.0
    corridor_confirmation_frames: int = 3
    corridor_end_confirmation_frames: int = 3
    door_width_min: float = 0.8
    door_width_max: float = 2.0
    door_confirmation_frames: int = 3
    door_position_tolerance: float = 0.35
    door_angle_tolerance_deg: float = 15.0
    door_footprint_margin: float = 0.08
    door_outside_distance: float = 0.85
    door_inside_distance: float = 1.50
    door_inside_confirmation_samples: int = 8
    door_entry_max_attempts: int = 3
    inside_depth_min: float = 1.0
    room_goal_baseline_min: float = 0.8
    path_sample_spacing: float = 0.075


class CorridorDetector:
    """Detect a local corridor from repeated parallel-wall cross sections."""

    def __init__(self, parameters: Optional[GeometryParameters] = None) -> None:
        self.p = parameters or GeometryParameters()

    @staticmethod
    def _nearest_wall(
        grid: OccupancyMap,
        center: Point,
        normal: Point,
        side: int,
        maximum: float,
    ) -> Optional[Tuple[float, Point]]:
        step = max(grid.resolution * 0.65, 0.07)
        for lateral in np.arange(0.35, maximum + step, step):
            point = (
                center[0] + side * float(lateral) * normal[0],
                center[1] + side * float(lateral) * normal[1],
            )
            state = grid.state(point)
            if state >= 50:
                return float(lateral), point
            if state < 0:
                return None
        return None

    def _evaluate(self, grid: OccupancyMap, pose: Point, angle: float) -> Optional[dict]:
        axis = unit(angle)
        normal = (-axis[1], axis[0])
        stations = np.arange(-2.0, 6.01, 0.30)
        pairs = []
        left_points, right_points = [], []
        maximum_half = self.p.corridor_width_max * 0.70
        for station in stations:
            center = (pose[0] + float(station) * axis[0],
                      pose[1] + float(station) * axis[1])
            left = self._nearest_wall(grid, center, normal, 1, maximum_half)
            right = self._nearest_wall(grid, center, normal, -1, maximum_half)
            if left and right:
                width = left[0] + right[0]
                if self.p.corridor_width_min <= width <= self.p.corridor_width_max:
                    pairs.append((float(station), left[0], right[0]))
                    left_points.append(left[1])
                    right_points.append(right[1])
        if len(pairs) < 7:
            return None
        widths = np.asarray([item[1] + item[2] for item in pairs])
        width_std = float(np.std(widths))
        if width_std > self.p.corridor_width_variation:
            return None
        support = len(pairs) / float(len(stations))
        # Offset the centerline toward the mean wall midpoint.
        offsets = np.asarray([(item[2] - item[1]) * 0.5 for item in pairs])
        offset = float(np.median(offsets))
        center = (pose[0] + offset * normal[0], pose[1] + offset * normal[1])
        forward = 0.0
        for value in np.arange(0.0, 10.01, max(grid.resolution, 0.10)):
            sample = (center[0] + float(value) * axis[0],
                      center[1] + float(value) * axis[1])
            if grid.state(sample) != 0:
                break
            forward = float(value)
        backward = 0.0
        for value in np.arange(0.0, 5.01, max(grid.resolution, 0.10)):
            sample = (center[0] - float(value) * axis[0],
                      center[1] - float(value) * axis[1])
            if grid.state(sample) != 0:
                break
            backward = float(value)
        aspect = (forward + backward) / max(float(np.median(widths)), 0.1)
        confidence = np.clip(
            0.45 * support
            + 0.25 * min(1.0, forward / self.p.corridor_forward_depth_min)
            + 0.20 * min(1.0, aspect / 2.5)
            + 0.10 * max(0.0, 1.0 - width_std / self.p.corridor_width_variation),
            0.0, 1.0,
        )
        return {
            "angle": angle,
            "axis": axis,
            "normal": normal,
            "center": center,
            "width": float(np.median(widths)),
            "width_std": width_std,
            "forward": forward,
            "backward": backward,
            "left_wall": left_points,
            "right_wall": right_points,
            "confidence": float(confidence),
            "score": float(confidence + 0.015 * len(pairs)),
        }

    def _region_from_evaluation(
        self,
        grid: OccupancyMap,
        evaluation: dict,
        previous_direction: Optional[float] = None,
    ) -> Optional[CorridorRegion]:
        best = dict(evaluation)
        best["left_wall"] = list(evaluation["left_wall"])
        best["right_wall"] = list(evaluation["right_wall"])
        # On first acquisition orient toward the longer visible direction.
        # Once a corridor direction is established, keep it: flipping toward
        # the longer space behind the robot at an end wall would make the
        # manager drive backward and would also reverse left/right door sides.
        if previous_direction is None and best["backward"] > best["forward"]:
            best["angle"] = wrap_angle(best["angle"] + math.pi)
            best["axis"] = (-best["axis"][0], -best["axis"][1])
            best["normal"] = (-best["normal"][0], -best["normal"][1])
            best["forward"], best["backward"] = best["backward"], best["forward"]
            best["left_wall"], best["right_wall"] = best["right_wall"], best["left_wall"]
        # Full-footprint feasibility is evaluated only for the selected axis.
        # Doing it for all 36 orientation hypotheses is equivalent but makes
        # every online topology update unnecessarily expensive.
        feasible = []
        for direction, limit in ((1.0, best["forward"]),
                                 (-1.0, best["backward"])):
            extent = 0.0
            for value in np.arange(0.0, limit + max(grid.resolution, 0.10) * 0.5,
                                   max(grid.resolution, 0.10)):
                sample = (
                    best["center"][0] + direction * float(value) * best["axis"][0],
                    best["center"][1] + direction * float(value) * best["axis"][1],
                )
                if not grid.footprint_safe(sample, best["angle"]):
                    break
                extent = float(value)
            feasible.append(extent)
        best["forward"], best["backward"] = feasible
        if sum(feasible) < max(1.0, best["width"]):
            return None
        start = (best["center"][0] - best["backward"] * best["axis"][0],
                 best["center"][1] - best["backward"] * best["axis"][1])
        end = (best["center"][0] + best["forward"] * best["axis"][0],
               best["center"][1] + best["forward"] * best["axis"][1])
        return CorridorRegion(
            corridor_id="corridor_candidate",
            centerline=[start, end],
            principal_direction=wrap_angle(best["angle"]),
            estimated_width=best["width"],
            forward_extent=best["forward"],
            left_wall=best["left_wall"],
            right_wall=best["right_wall"],
            confidence=best["confidence"],
        )

    def detect(
        self,
        grid: OccupancyMap,
        pose: Sequence[float],
        previous_direction: Optional[float] = None,
    ) -> Optional[CorridorRegion]:
        if previous_direction is None:
            angles = np.linspace(0.0, math.pi, 36, endpoint=False)
        else:
            angles = [previous_direction + math.radians(delta)
                      for delta in range(-20, 21, 4)]
        evaluations = [self._evaluate(grid, (float(pose[0]), float(pose[1])), angle)
                       for angle in angles]
        evaluations = [item for item in evaluations if item]
        if not evaluations:
            return None
        return self._region_from_evaluation(
            grid, max(evaluations, key=lambda item: item["score"]),
            previous_direction)

    def detect_candidates(
        self,
        grid: OccupancyMap,
        pose: Sequence[float],
        maximum_candidates: int = 4,
        angular_separation_deg: float = 25.0,
    ) -> List[CorridorRegion]:
        """Return distinct local corridor axes for junction/branch discovery."""
        evaluations = [self._evaluate(
            grid, (float(pose[0]), float(pose[1])), angle)
            for angle in np.linspace(0.0, math.pi, 36, endpoint=False)]
        evaluations = sorted(
            (item for item in evaluations if item),
            key=lambda item: item["score"], reverse=True)
        selected: List[CorridorRegion] = []
        for evaluation in evaluations:
            if any(angle_distance(evaluation["angle"], item.principal_direction) <
                   math.radians(angular_separation_deg) for item in selected):
                continue
            region = self._region_from_evaluation(grid, evaluation, None)
            if region is not None:
                selected.append(region)
            if len(selected) >= max(1, int(maximum_candidates)):
                break
        return selected


class DoorDetector:
    """Find stable traversable gaps in the two detected corridor walls."""

    def __init__(self, parameters: Optional[GeometryParameters] = None) -> None:
        self.p = parameters or GeometryParameters()

    @staticmethod
    def _occupied_near(grid: OccupancyMap, point: Point, radius: float = 0.20) -> bool:
        cell = grid.world_to_cell(point)
        if cell is None:
            return False
        cells = max(1, int(math.ceil(radius / grid.resolution)))
        x0, y0 = cell
        for y in range(max(0, y0 - cells), min(grid.height, y0 + cells + 1)):
            for x in range(max(0, x0 - cells), min(grid.width, x0 + cells + 1)):
                if grid.data[y, x] >= 50:
                    return True
        return False

    def detect(self, grid: OccupancyMap, corridor: CorridorRegion) -> List[Doorway]:
        if len(corridor.centerline) < 2:
            return []
        angle = corridor.principal_direction
        axis = unit(angle)
        normal = (-axis[1], axis[0])
        origin = corridor.centerline[0]
        length = distance(corridor.centerline[0], corridor.centerline[-1])
        half_width = 0.5 * corridor.estimated_width
        step = max(grid.resolution, 0.10)
        stations = np.arange(0.0, length + step * 0.5, step)
        result: List[Doorway] = []
        # Use one projection cell to join the sampled wall surface.  The old
        # fixed 0.22 m probe expands to two cells at the live 0.15 m OctoMap
        # resolution, shrinking a nominal 1.2 m room opening by about 0.6 m.
        # Such an opening then looks narrower than door_width_min and is lost.
        # One cell still tolerates voxel/sampling jitter without erasing the
        # physical doorway.
        wall_probe_radius = max(0.05, 0.65 * grid.resolution)
        for side in (-1, 1):
            wall_present = []
            wall_points = []
            for station in stations:
                centerline = (origin[0] + float(station) * axis[0],
                              origin[1] + float(station) * axis[1])
                wall = (centerline[0] + side * half_width * normal[0],
                        centerline[1] + side * half_width * normal[1])
                wall_points.append(wall)
                wall_present.append(self._occupied_near(
                    grid, wall, wall_probe_radius))
            index = 1
            while index < len(stations) - 1:
                if wall_present[index]:
                    index += 1
                    continue
                start = index
                while index < len(stations) and not wall_present[index]:
                    index += 1
                end = index - 1
                if start == 0 or index >= len(stations):
                    continue
                width = float(stations[end] - stations[start] + step)
                if not self.p.door_width_min <= width <= self.p.door_width_max:
                    continue
                if not wall_present[start - 1] or not wall_present[index]:
                    continue
                middle = 0.5 * (float(stations[start]) + float(stations[end]))
                centerline = (origin[0] + middle * axis[0], origin[1] + middle * axis[1])
                center = (centerline[0] + side * half_width * normal[0],
                          centerline[1] + side * half_width * normal[1])
                outward = (side * normal[0], side * normal[1])
                corridor_side = (center[0] - 0.45 * outward[0],
                                 center[1] - 0.45 * outward[1])
                interior_side = (center[0] + 0.55 * outward[0],
                                 center[1] + 0.55 * outward[1])
                deeper = (center[0] + 1.0 * outward[0], center[1] + 1.0 * outward[1])
                # The gap itself and its corridor side must be observed free.
                # Door-behind space may still be unknown, but an unknown door
                # is not marked traversable until later observations clear it.
                if not grid.is_free(center) or not grid.is_free(corridor_side):
                    continue
                if grid.state(interior_side) >= 50 or grid.state(deeper) >= 50:
                    continue
                left_frame = wall_points[start - 1]
                right_frame = wall_points[index]
                behind_states = (grid.state(interior_side), grid.state(deeper))
                observed_behind = sum(state == 0 for state in behind_states)
                unknown_behind = sum(state < 0 for state in behind_states)
                tangent = (-outward[1], outward[0])
                unknown_samples = 0
                sample_step = max(0.30, 2.0 * grid.resolution)
                for depth in np.arange(0.5, 3.01, sample_step):
                    for lateral in np.arange(-1.5, 1.51, sample_step):
                        sample = (
                            center[0] + float(depth) * outward[0]
                            + float(lateral) * tangent[0],
                            center[1] + float(depth) * outward[1]
                            + float(lateral) * tangent[1],
                        )
                        unknown_samples += int(grid.state(sample) < 0)
                confidence = min(1.0, 0.55 + 0.15 * observed_behind + 0.05 * unknown_behind)
                footprint_traversable = all(
                    grid.footprint_safe(point, math.atan2(outward[1], outward[0]))
                    for point in (corridor_side, center, interior_side))
                result.append(Doorway(
                    door_id="door_candidate",
                    center=center,
                    normal_direction=math.atan2(outward[1], outward[0]),
                    width=width,
                    left_frame_point=left_frame,
                    right_frame_point=right_frame,
                    corridor_side=corridor_side,
                    interior_side=interior_side,
                    confidence=confidence,
                    traversable=(observed_behind == 2 and footprint_traversable and
                                 width >= 0.44 + 2.0 * self.p.door_footprint_margin),
                    estimated_unknown_size=float(unknown_samples * sample_step ** 2),
                ))
        return result


class RoomExpansionDetector:
    """Detect room-scale free-space expansion beside a tracked corridor.

    Unlike :class:`DoorDetector`, this detector does not require a perfectly
    rasterized door gap.  It first finds free regions whose clearance is wider
    than the corridor, then uses gaps in the independently sampled corridor
    wall evidence to estimate a portal.  No layout coordinates are consumed.
    """

    def __init__(self, parameters: Optional[GeometryParameters] = None) -> None:
        self.p = parameters or GeometryParameters()

    @staticmethod
    def _coordinates(grid: OccupancyMap, corridor: CorridorRegion):
        ys, xs = np.indices(grid.data.shape, dtype=np.float64)
        world_x = grid.origin_x + (xs + 0.5) * grid.resolution
        world_y = grid.origin_y + (ys + 0.5) * grid.resolution
        origin = corridor.centerline[0]
        axis = unit(corridor.principal_direction)
        normal = (-axis[1], axis[0])
        dx, dy = world_x - origin[0], world_y - origin[1]
        longitudinal = dx * axis[0] + dy * axis[1]
        lateral = dx * normal[0] + dy * normal[1]
        return world_x, world_y, longitudinal, lateral, axis, normal

    def _portal_from_wall_evidence(
        self,
        corridor: CorridorRegion,
        side: int,
        longitudinal_min: float,
        longitudinal_max: float,
    ) -> Tuple[Optional[Point], float, float]:
        axis = unit(corridor.principal_direction)
        normal = (-axis[1], axis[0])
        origin = corridor.centerline[0]
        wall = corridor.left_wall if side > 0 else corridor.right_wall
        stations = sorted(dot((point[0] - origin[0], point[1] - origin[1]), axis)
                          for point in wall)
        if len(stations) < 4:
            return None, 0.0, 0.0
        short_steps = [following - previous
                       for previous, following in zip(stations[:-1], stations[1:])
                       if 0.05 < following - previous <= 0.65]
        nominal_step = float(np.median(short_steps)) if short_steps else 0.30
        candidates = []
        for previous, following in zip(stations[:-1], stations[1:]):
            width = following - previous - nominal_step
            middle = 0.5 * (previous + following)
            if not self.p.door_width_min <= width <= max(2.5, self.p.door_width_max):
                continue
            if not longitudinal_min - 1.5 <= middle <= longitudinal_max + 1.5:
                continue
            # Prefer an aperture near the observed expansion while retaining
            # a weak score for incomplete room cores.
            interval_center = 0.5 * (longitudinal_min + longitudinal_max)
            candidates.append((abs(middle - interval_center), middle, width))
        if not candidates:
            return None, 0.0, 0.0
        _, middle, width = min(candidates)
        half_width = 0.5 * corridor.estimated_width
        center = (
            origin[0] + middle * axis[0] + side * half_width * normal[0],
            origin[1] + middle * axis[1] + side * half_width * normal[1],
        )
        confidence = min(1.0, 0.55 + 0.25 * min(width / 1.2, 1.0)
                         + 0.20 * min(len(stations) / 12.0, 1.0))
        return center, float(width), float(confidence)

    @staticmethod
    def _occupied_near(
        grid: OccupancyMap, point: Point, radius: float,
    ) -> bool:
        """Return measured occupied support only; unknown is not a door frame."""
        cell = grid.world_to_cell(point)
        if cell is None:
            return False
        cells = max(1, int(math.ceil(float(radius) / grid.resolution)))
        x0, y0 = cell
        for y in range(max(0, y0 - cells), min(grid.height, y0 + cells + 1)):
            for x in range(max(0, x0 - cells), min(grid.width, x0 + cells + 1)):
                if grid.data[y, x] >= 50:
                    return True
        return False

    def _portal_from_free_channel(
        self,
        grid: OccupancyMap,
        corridor: CorridorRegion,
        side: int,
        longitudinal_min: float,
        longitudinal_max: float,
        preferred_station: float,
        room_seed: Optional[Point] = None,
        free_labels: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[Point], float, float]:
        """Recover a portal when the corridor detector's wall points are sparse.

        This fallback does not infer an aperture from unknown space.  It scans
        the candidate's corridor-side interval and accepts only a contiguous
        band where an A1 footprint is observed free on the corridor side, in
        the wall plane, and inside the expanded free-space component.  Both
        longitudinal ends must also have measured occupied frame support.
        """
        if len(corridor.centerline) < 2 or side not in (-1, 1):
            return None, 0.0, 0.0
        axis = unit(corridor.principal_direction)
        normal = (-axis[1], axis[0])
        outward = (side * normal[0], side * normal[1])
        outward_yaw = math.atan2(outward[1], outward[0])
        origin = corridor.centerline[0]
        corridor_length = distance(corridor.centerline[0], corridor.centerline[-1])
        # A room core is defined by high-clearance cells, so furniture and the
        # narrow entrance can put its first core cell several metres beyond
        # the actual doorway. Search far enough back toward the corridor to
        # bridge that gap; free connectivity to ``room_seed`` below prevents
        # this wider interval from borrowing an unrelated neighbouring door.
        search_padding = max(1.5, 2.25 * corridor.estimated_width)
        start = max(0.0, longitudinal_min - search_padding)
        finish = min(corridor_length, longitudinal_max + search_padding)
        step = max(grid.resolution, 0.10)
        if finish - start < step:
            return None, 0.0, 0.0
        half_width = 0.5 * corridor.estimated_width
        if room_seed is not None and free_labels is None:
            _, free_labels = cv2.connectedComponents(
                (grid.data == 0).astype(np.uint8), 8)

        def component_at(point: Point, search_radius: float) -> int:
            cell = grid.world_to_cell(point)
            if cell is None or free_labels is None:
                return 0
            x0, y0 = cell
            cells = max(0, int(math.ceil(search_radius / grid.resolution)))
            best = None
            for y in range(max(0, y0 - cells),
                           min(grid.height, y0 + cells + 1)):
                for x in range(max(0, x0 - cells),
                               min(grid.width, x0 + cells + 1)):
                    label = int(free_labels[y, x])
                    if label <= 0:
                        continue
                    squared = (x - x0) ** 2 + (y - y0) ** 2
                    if best is None or squared < best[0]:
                        best = (squared, label)
            return 0 if best is None else best[1]

        room_component = (component_at(
            room_seed, max(grid.resolution, 0.5 * corridor.estimated_width))
                          if room_seed is not None else 0)
        safe = []
        stations = np.arange(start, finish + 0.5 * step, step)
        for station in stations:
            wall = (
                origin[0] + float(station) * axis[0] +
                side * half_width * normal[0],
                origin[1] + float(station) * axis[1] +
                side * half_width * normal[1],
            )
            samples = [
                (wall[0] - 0.45 * outward[0], wall[1] - 0.45 * outward[1]),
                wall,
                (wall[0] + 0.55 * outward[0], wall[1] + 0.55 * outward[1]),
            ]
            safe.append(all(
                grid.footprint_safe(sample, outward_yaw,
                                    margin=self.p.door_footprint_margin)
                for sample in samples))

        runs = []
        index = 0
        body_lateral_width = 0.44 + 2.0 * self.p.door_footprint_margin
        frame_radius = max(0.16, 1.10 * grid.resolution)
        while index < len(stations):
            if not safe[index]:
                index += 1
                continue
            first = index
            while index + 1 < len(stations) and safe[index + 1]:
                index += 1
            last = index
            first_station, last_station = (float(stations[first]),
                                           float(stations[last]))
            inferred_width = last_station - first_station + step + body_lateral_width
            if self.p.door_width_min <= inferred_width <= self.p.door_width_max:
                before = first_station - 0.5 * body_lateral_width - 0.5 * step
                after = last_station + 0.5 * body_lateral_width + 0.5 * step
                frame_points = []
                for station in (before, after):
                    frame_points.append((
                        origin[0] + station * axis[0] +
                        side * half_width * normal[0],
                        origin[1] + station * axis[1] +
                        side * half_width * normal[1],
                    ))
                if all(self._occupied_near(grid, point, frame_radius)
                       for point in frame_points):
                    middle = 0.5 * (first_station + last_station)
                    wall = (
                        origin[0] + middle * axis[0] +
                        side * half_width * normal[0],
                        origin[1] + middle * axis[1] +
                        side * half_width * normal[1],
                    )
                    inside = (wall[0] + 0.55 * outward[0],
                              wall[1] + 0.55 * outward[1])
                    if (room_seed is not None and
                            (room_component <= 0 or component_at(
                                inside, max(grid.resolution, 0.20)) !=
                             room_component)):
                        index += 1
                        continue
                    runs.append((abs(middle - preferred_station), middle,
                                 inferred_width))
            index += 1
        if not runs:
            return None, 0.0, 0.0
        _, middle, width = min(runs)
        center = (
            origin[0] + middle * axis[0] + side * half_width * normal[0],
            origin[1] + middle * axis[1] + side * half_width * normal[1],
        )
        # Lower than a clean wall-gap estimate, but strong enough to be
        # accumulated. Execution still requires independent 2-D and 3-D
        # validation in the manager.
        return center, float(width), 0.82

    def detect(self, grid: OccupancyMap,
               corridor: CorridorRegion) -> List[RoomExpansionObservation]:
        if cv2 is None:
            raise RuntimeError("OpenCV is required for room expansion detection")
        if len(corridor.centerline) < 2:
            return []
        free = (grid.data == 0).astype(np.uint8)
        _, free_labels = cv2.connectedComponents(free, 8)
        clearance = cv2.distanceTransform(free, cv2.DIST_L2, 5) * grid.resolution
        width = corridor.estimated_width
        half_width = 0.5 * width
        core_clearance = max(1.20, 0.65 * width)
        core = np.logical_and(free > 0, clearance >= core_clearance).astype(np.uint8)
        count, labels, statistics, _ = cv2.connectedComponentsWithStats(core, 8)
        world_x, world_y, longitudinal, lateral, _, normal = self._coordinates(
            grid, corridor)
        corridor_length = distance(corridor.centerline[0], corridor.centerline[-1])
        minimum_core_area = max(1.5, 0.40 * width * width)
        minimum_lateral = half_width + 0.45
        corridor_onset = 3.0 * width
        pieces = []
        for label in range(1, count):
            mask = labels == label
            area = int(statistics[label, cv2.CC_STAT_AREA]) * grid.resolution ** 2
            if area < minimum_core_area:
                continue
            lon_min = float(longitudinal[mask].min())
            lon_max = float(longitudinal[mask].max())
            lat_mean = float(lateral[mask].mean())
            if lon_max < corridor_onset or lon_min > corridor_length + width:
                continue
            if abs(lat_mean) < minimum_lateral:
                continue
            pieces.append({
                "side": 1 if lat_mean > 0.0 else -1,
                "area": area,
                "lon_min": lon_min,
                "lon_max": lon_max,
                "mask": mask,
                "max_clearance": float(clearance[mask].max()),
            })
        merge_gap = max(2.5, 1.25 * width)
        groups = []
        for piece in sorted(pieces, key=lambda item: (item["side"], item["lon_min"])):
            if (groups and groups[-1]["side"] == piece["side"] and
                    piece["lon_min"] - groups[-1]["lon_max"] <= merge_gap):
                groups[-1]["pieces"].append(piece)
                groups[-1]["lon_max"] = max(groups[-1]["lon_max"], piece["lon_max"])
            else:
                groups.append({"side": piece["side"], "lon_min": piece["lon_min"],
                               "lon_max": piece["lon_max"], "pieces": [piece]})
        observations = []
        for sequence, group in enumerate(groups, 1):
            mask = np.logical_or.reduce([piece["mask"] for piece in group["pieces"]])
            weights = np.maximum(clearance[mask], grid.resolution)
            centroid = (float(np.average(world_x[mask], weights=weights)),
                        float(np.average(world_y[mask], weights=weights)))
            core_area = float(mask.sum() * grid.resolution ** 2)
            lateral_depth = float(np.max(np.abs(lateral[mask])) - half_width)
            span = float(group["lon_max"] - group["lon_min"])
            maximum_clearance = max(piece["max_clearance"]
                                    for piece in group["pieces"])
            expansion_ratio = maximum_clearance / max(half_width, grid.resolution)
            geometry = min(
                1.0,
                0.30 + 0.18 * min(expansion_ratio / 2.0, 1.0)
                + 0.18 * min(lateral_depth / (2.0 * width), 1.0)
                + 0.18 * min(span / (2.0 * width), 1.0)
                + 0.16 * min(core_area / (3.0 * width * width), 1.0),
            )
            wall_portal = self._portal_from_wall_evidence(
                corridor, group["side"], group["lon_min"], group["lon_max"])
            preferred_station = dot(
                (centroid[0] - corridor.centerline[0][0],
                 centroid[1] - corridor.centerline[0][1]),
                unit(corridor.principal_direction))
            free_portal = self._portal_from_free_channel(
                grid, corridor, group["side"], group["lon_min"],
                group["lon_max"], preferred_station, centroid, free_labels)
            if free_portal[0] is not None:
                # Directly observed footprint clearance and connectivity are
                # stronger localization evidence than a sparse wall-point
                # gap. Preserve the independent wall support as confidence,
                # but use the free channel's centre and physical width.
                portal, portal_width = free_portal[:2]
                portal_confidence = max(
                    float(free_portal[2]), float(wall_portal[2]))
            else:
                portal, portal_width, portal_confidence = wall_portal
            outward = (group["side"] * normal[0], group["side"] * normal[1])
            observation = RoomExpansionObservation(
                candidate_id="room_expansion_{:02d}".format(sequence),
                corridor_id=corridor.corridor_id,
                side=group["side"], centroid=centroid,
                core_area=core_area,
                longitudinal_min=group["lon_min"],
                longitudinal_max=group["lon_max"],
                lateral_expansion_depth=lateral_depth,
                maximum_clearance=maximum_clearance,
                expansion_ratio=expansion_ratio,
                geometry_confidence=geometry,
                portal_center=portal,
                portal_width=portal_width,
                portal_normal_direction=math.atan2(outward[1], outward[0]),
                portal_confidence=portal_confidence,
                fused_confidence=0.85 * geometry + 0.15 * portal_confidence,
            )
            observations.append(observation)
        return observations


class TemporalTopologyTracker:
    """Merge geometry over map updates; a single frame never confirms topology."""

    def __init__(self, parameters: Optional[GeometryParameters] = None) -> None:
        self.p = parameters or GeometryParameters()
        self.corridors: List[CorridorRegion] = []
        self.doors: List[Doorway] = []
        self._corridor_sequence = 0
        self._door_sequence = 0

    def update_corridor(self, observed: Optional[CorridorRegion]) -> Optional[CorridorRegion]:
        if observed is None:
            return None
        match = None
        for corridor in self.corridors:
            if (angle_distance(corridor.principal_direction,
                               observed.principal_direction) <= math.radians(
                                   self.p.parallel_wall_angle_tolerance_deg)
                    and self._corridor_regions_connect(corridor, observed)):
                match = corridor
                break
        if match is None:
            self._corridor_sequence += 1
            observed.corridor_id = "corridor_{:02d}".format(self._corridor_sequence)
            self.corridors.append(observed)
            match = observed
        else:
            old = match.observations
            weight = 1.0 / float(old + 1)
            accumulated_endpoints = list(match.centerline) + list(observed.centerline)
            # Align the undirected observation with the stored direction.
            observed_angle = observed.principal_direction
            if abs(wrap_angle(observed_angle - match.principal_direction)) > math.pi / 2:
                observed_angle = wrap_angle(observed_angle + math.pi)
            match.principal_direction = wrap_angle(
                match.principal_direction + weight * wrap_angle(
                    observed_angle - match.principal_direction))
            match.estimated_width = (1.0 - weight) * match.estimated_width + weight * observed.estimated_width
            # forward_extent is a current local end-wall observation, not the
            # historical maximum; retaining the maximum makes corridor-end
            # confirmation impossible after traversing a long corridor.
            match.forward_extent = observed.forward_extent
            match.confidence = max(match.confidence, observed.confidence)
            axis = unit(match.principal_direction)
            origin = accumulated_endpoints[0]
            projections = [dot((point[0] - origin[0], point[1] - origin[1]), axis)
                           for point in accumulated_endpoints]
            match.centerline = [
                (origin[0] + min(projections) * axis[0],
                 origin[1] + min(projections) * axis[1]),
                (origin[0] + max(projections) * axis[0],
                 origin[1] + max(projections) * axis[1]),
            ]
            match.left_wall = observed.left_wall
            match.right_wall = observed.right_wall
            match.observations += 1
        match.confirmed = match.observations >= self.p.corridor_confirmation_frames
        return match

    @staticmethod
    def _corridor_regions_connect(a: CorridorRegion, b: CorridorRegion) -> bool:
        """Do not merge spatially separate but parallel corridor branches."""
        if not a.centerline or not b.centerline:
            return False
        axis = unit(a.principal_direction)
        normal = (-axis[1], axis[0])
        a_mid = ((a.centerline[0][0] + a.centerline[-1][0]) * 0.5,
                 (a.centerline[0][1] + a.centerline[-1][1]) * 0.5)
        b_mid = ((b.centerline[0][0] + b.centerline[-1][0]) * 0.5,
                 (b.centerline[0][1] + b.centerline[-1][1]) * 0.5)
        lateral = abs(dot((b_mid[0] - a_mid[0], b_mid[1] - a_mid[1]), normal))
        if lateral > max(0.75, 0.45 * (a.estimated_width + b.estimated_width)):
            return False
        a_s = sorted(dot((point[0] - a_mid[0], point[1] - a_mid[1]), axis)
                     for point in (a.centerline[0], a.centerline[-1]))
        b_s = sorted(dot((point[0] - a_mid[0], point[1] - a_mid[1]), axis)
                     for point in (b.centerline[0], b.centerline[-1]))
        gap = max(0.0, max(a_s[0], b_s[0]) - min(a_s[1], b_s[1]))
        return gap <= 4.0

    def update_doors(
        self,
        observations: Iterable[Doorway],
        depth_confirmed_ids: Optional[Iterable[str]] = None,
    ) -> List[Doorway]:
        depth_ids = set(depth_confirmed_ids or [])
        updated = []
        for observed in observations:
            match = None
            for door in self.doors:
                if (distance(door.center, observed.center) <= self.p.door_position_tolerance
                        and abs(wrap_angle(door.normal_direction -
                                           observed.normal_direction)) <= math.radians(
                                               self.p.door_angle_tolerance_deg)):
                    match = door
                    break
            if match is None:
                self._door_sequence += 1
                observed.door_id = "door_{:02d}".format(self._door_sequence)
                # Raw wall-gap tracks are perception candidates only.  The
                # manager is the sole authority that may set traversable=True
                # after room-expansion association plus 2-D and 3-D checks.
                if not observed.source_room_candidate_id:
                    observed.traversable = False
                self.doors.append(observed)
                match = observed
            else:
                weight = 1.0 / float(match.observations + 1)
                if not match.geometry_locked:
                    match.center = (
                        (1.0 - weight) * match.center[0] + weight * observed.center[0],
                        (1.0 - weight) * match.center[1] + weight * observed.center[1],
                    )
                    match.width = (1.0 - weight) * match.width + weight * observed.width
                    match.normal_direction = wrap_angle(
                        match.normal_direction + weight * wrap_angle(
                            observed.normal_direction - match.normal_direction))
                    match.left_frame_point = observed.left_frame_point
                    match.right_frame_point = observed.right_frame_point
                    match.corridor_side = observed.corridor_side
                    match.interior_side = observed.interior_side
                match.confidence = max(match.confidence, observed.confidence)
                if observed.source_room_candidate_id:
                    match.traversable = match.traversable or observed.traversable
                match.estimated_unknown_size = max(match.estimated_unknown_size,
                                                   observed.estimated_unknown_size)
                match.observations += 1
            if match.door_id in depth_ids:
                match.depth_confirmations += 1
                match.confidence = min(1.0, match.confidence + 0.08)
            match.confirmed = (
                match.observations >= self.p.door_confirmation_frames
                and match.confidence >= 0.65)
            updated.append(match)
        return updated


def corridor_junction_target(
    door: Doorway,
    source_corridor_id: Optional[str],
    corridors: Iterable[CorridorRegion],
    angle_tolerance_deg: float = 25.0,
    center_band_margin: float = 0.20,
) -> Optional[str]:
    """Return the other corridor reached by a wall gap, or None for a room door."""
    for corridor in corridors:
        if (not corridor.confirmed or corridor.corridor_id == source_corridor_id or
                len(corridor.centerline) < 2):
            continue
        if angle_distance(door.normal_direction,
                          corridor.principal_direction) > math.radians(
                              angle_tolerance_deg):
            continue
        start, end = corridor.centerline[0], corridor.centerline[-1]
        segment = (end[0] - start[0], end[1] - start[1])
        length_squared = dot(segment, segment)
        if length_squared < 1e-9:
            continue
        alpha = dot((door.center[0] - start[0], door.center[1] - start[1]),
                    segment) / length_squared
        if not -0.10 <= alpha <= 1.10:
            continue
        alpha_clamped = min(1.0, max(0.0, alpha))
        closest = (start[0] + alpha_clamped * segment[0],
                   start[1] + alpha_clamped * segment[1])
        if distance(door.center, closest) <= (
                0.5 * corridor.estimated_width + center_band_margin):
            return corridor.corridor_id
    return None


@dataclass
class PathValidity:
    valid: bool
    samples: int
    occupied_intersections: int
    unknown_intersections: int
    footprint_collisions: int
    door_crossings: List[dict]
    non_door_crossings: int
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def validate_path(
    grid: OccupancyMap,
    path: Sequence[Point],
    allowed_doors: Sequence[Doorway] = (),
    required_door_id: Optional[str] = None,
    spacing: float = 0.075,
    initial_unknown_footprint_grace: float = 0.0,
    check_footprint: bool = True,
) -> PathValidity:
    samples = interpolate_polyline(path, spacing)
    occupied = unknown = footprint = 0
    crossings: List[dict] = []
    non_door = 0
    travelled = 0.0
    for index, point in enumerate(samples):
        if index:
            travelled += distance(samples[index - 1], point)
        state = grid.state(point)
        occupied += int(state >= 50)
        unknown += int(state < 0)
        if index + 1 < len(samples):
            yaw = math.atan2(samples[index + 1][1] - point[1],
                             samples[index + 1][0] - point[0])
        elif index > 0:
            yaw = math.atan2(point[1] - samples[index - 1][1],
                             point[0] - samples[index - 1][0])
        else:
            yaw = 0.0
        if check_footprint:
            footprint += int(not grid.footprint_safe(
                point, yaw,
                unknown_is_safe=(travelled <=
                                 float(initial_unknown_footprint_grace) + 1e-9)))
    for previous, current in zip(samples[:-1], samples[1:]):
        for door in allowed_doors:
            a, b = door.signed_depth(previous), door.signed_depth(current)
            if a == 0.0 or b == 0.0 or a * b < 0.0:
                denominator = abs(a) + abs(b)
                alpha = 0.5 if denominator < 1e-9 else abs(a) / denominator
                intersection = (
                    previous[0] + alpha * (current[0] - previous[0]),
                    previous[1] + alpha * (current[1] - previous[1]),
                )
                lateral = abs(door.lateral_offset(intersection))
                # A doorway plane is local to its frames, not an infinite
                # building-wide boundary. Far-away mathematical crossings
                # are ignored; occupied/unknown/room-region checks cover them.
                if lateral > 0.5 * door.width + 0.45:
                    continue
                legal = lateral <= max(0.0, 0.5 * door.width - 0.05)
                crossings.append({
                    "door_id": door.door_id,
                    "point": intersection,
                    "legal": legal,
                    "direction": "enter" if b > a else "exit",
                })
                non_door += int(not legal)
    required_ok = required_door_id is None or any(
        item["door_id"] == required_door_id and item["legal"] for item in crossings)
    invalid_crossing = any(not item["legal"] for item in crossings)
    valid = (len(samples) >= 2 and occupied == 0 and unknown == 0 and
             footprint == 0 and non_door == 0 and not invalid_crossing and required_ok)
    if occupied:
        reason = "occupied_intersection"
    elif unknown:
        reason = "unknown_not_traversable"
    elif footprint:
        reason = "footprint_collision"
    elif invalid_crossing or non_door:
        reason = "non_door_boundary_crossing"
    elif not required_ok:
        reason = "required_door_not_crossed"
    elif len(samples) < 2:
        reason = "empty_path"
    else:
        reason = "valid"
    return PathValidity(valid, len(samples), occupied, unknown, footprint,
                        crossings, non_door, reason)


def door_entry_evidence(
    door: Doorway,
    trajectory: Sequence,
    inside_depth_min: float = 1.0,
    minimum_inside_samples: int = 8,
    plane_hysteresis: float = 0.08,
) -> dict:
    """Return strict physical evidence that a trajectory entered ``door``.

    A planned path or a goal-success latch is not entry evidence.  The actual
    trajectory must start on the corridor side, cross the finite door aperture,
    finish at least ``inside_depth_min`` behind the plane, and remain on the
    room side for several consecutive odometry samples.
    """
    points: List[Point] = []
    for item in trajectory:
        if isinstance(item, dict):
            if "x" not in item or "y" not in item:
                continue
            point = (float(item["x"]), float(item["y"]))
        else:
            if len(item) < 2:
                continue
            point = (float(item[0]), float(item[1]))
        if all(math.isfinite(value) for value in point):
            points.append(point)

    hysteresis = max(0.0, float(plane_hysteresis))
    required_samples = max(1, int(minimum_inside_samples))
    crossing = None
    legal_crossing = False
    saw_corridor_side = False
    previous = None
    previous_depth = None
    maximum_consecutive_inside = 0
    ending_consecutive_inside = 0
    for point in points:
        depth = door.signed_depth(point)
        if depth <= -hysteresis:
            saw_corridor_side = True
        if (crossing is None and saw_corridor_side and previous is not None and
                previous_depth is not None and previous_depth <= 0.0 <= depth):
            denominator = abs(previous_depth) + abs(depth)
            alpha = 0.5 if denominator < 1e-9 else abs(previous_depth) / denominator
            crossing = (
                previous[0] + alpha * (point[0] - previous[0]),
                previous[1] + alpha * (point[1] - previous[1]),
            )
            legal_crossing = abs(door.lateral_offset(crossing)) <= max(
                0.0, 0.5 * door.width - 0.05)
        if crossing is not None and legal_crossing and depth >= hysteresis:
            ending_consecutive_inside += 1
            maximum_consecutive_inside = max(
                maximum_consecutive_inside, ending_consecutive_inside)
        else:
            ending_consecutive_inside = 0
        previous, previous_depth = point, depth

    final_depth = door.signed_depth(points[-1]) if points else -math.inf
    confirmed = bool(
        crossing is not None and legal_crossing and
        final_depth >= float(inside_depth_min) and
        ending_consecutive_inside >= required_samples)
    return {
        "confirmed": confirmed,
        "sample_count": len(points),
        "saw_corridor_side": saw_corridor_side,
        "crossing_point": crossing,
        "legal_crossing": legal_crossing,
        "final_inside_depth": final_depth if math.isfinite(final_depth) else None,
        "maximum_consecutive_inside_samples": maximum_consecutive_inside,
        "ending_consecutive_inside_samples": ending_consecutive_inside,
        "required_inside_depth": float(inside_depth_min),
        "required_inside_samples": required_samples,
        "reason": (
            "confirmed" if confirmed else
            "no_corridor_to_room_plane_crossing" if crossing is None else
            "crossing_outside_door_aperture" if not legal_crossing else
            "insufficient_inside_depth" if final_depth < float(inside_depth_min) else
            "insufficient_consecutive_inside_samples"
        ),
    }


def _cell_clear_for_astar(
    grid: OccupancyMap,
    cell: Tuple[int, int],
    radius: float,
    unknown_is_safe: bool = False,
) -> bool:
    point = grid.cell_to_world(cell)
    if grid.state(point) != 0:
        return False
    cells = max(0, int(math.ceil(radius / grid.resolution)))
    radius_squared = max(0.0, float(radius)) ** 2
    for dy in range(-cells, cells + 1):
        for dx in range(-cells, cells + 1):
            # ``cells`` only bounds the loop.  Using it as the disk radius
            # rounds every non-integral clearance up by a full map cell (for
            # example 0.16 m became 0.30 m on a 0.15 m grid).  Compare the
            # actual metric distance between cell centres instead.
            if ((dx * grid.resolution) ** 2 +
                    (dy * grid.resolution) ** 2 > radius_squared + 1e-12):
                continue
            x, y = cell[0] + dx, cell[1] + dy
            if not (0 <= x < grid.width and 0 <= y < grid.height):
                return False
            state = int(grid.data[y, x])
            if state >= 50 or (state < 0 and not unknown_is_safe):
                return False
    return True


def astar_free_path(
    grid: OccupancyMap,
    start: Point,
    goal: Point,
    clearance_radius: float = 0.16,
    allowed_side: Optional[Tuple[Doorway, int, float]] = None,
    allow_unknown_at_start: bool = False,
) -> List[Point]:
    """Plan through observed free cells only; optionally stay on one door side."""
    start_cell, goal_cell = grid.world_to_cell(start), grid.world_to_cell(goal)
    if start_cell is None or goal_cell is None:
        return []
    # The robot's current occupancy is direct evidence that its initial
    # location is traversable, even when ray projection leaves a sliver of
    # unknown behind it.  Callers must opt in; measured obstacles are never
    # ignored and every successor uses the strict check below.
    if not _cell_clear_for_astar(
            grid, start_cell, clearance_radius,
            unknown_is_safe=allow_unknown_at_start):
        return []
    if not _cell_clear_for_astar(grid, goal_cell, clearance_radius):
        return []

    def side_allowed(cell: Tuple[int, int]) -> bool:
        if allowed_side is None:
            return True
        door, sign, tolerance = allowed_side
        return sign * door.signed_depth(grid.cell_to_world(cell)) >= -float(tolerance)

    frontier = [(distance(start_cell, goal_cell), 0.0, start_cell)]
    cost = {start_cell: 0.0}
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1),
                 (1, 1), (1, -1), (-1, 1), (-1, -1))
    while frontier:
        _, current_cost, current = heapq.heappop(frontier)
        if current == goal_cell:
            cells = [current]
            while cells[-1] != start_cell:
                cells.append(parent[cells[-1]])
            cells.reverse()
            points = [grid.cell_to_world(cell) for cell in cells]
            points[0], points[-1] = start, goal
            return points
        if current_cost > cost.get(current, math.inf) + 1e-9:
            continue
        for dx, dy in neighbors:
            nxt = current[0] + dx, current[1] + dy
            if not (0 <= nxt[0] < grid.width and 0 <= nxt[1] < grid.height):
                continue
            if not side_allowed(nxt) or not _cell_clear_for_astar(
                    grid, nxt, clearance_radius):
                continue
            step = math.sqrt(2.0) if dx and dy else 1.0
            candidate_cost = current_cost + step
            if candidate_cost >= cost.get(nxt, math.inf):
                continue
            cost[nxt] = candidate_cost
            parent[nxt] = current
            heuristic = math.hypot(nxt[0] - goal_cell[0], nxt[1] - goal_cell[1])
            heapq.heappush(frontier, (candidate_cost + heuristic,
                                     candidate_cost, nxt))
    return []


def select_forward_free_bootstrap(
    grid: OccupancyMap,
    pose: Point,
    yaw: float,
    minimum_distance: float = 1.0,
    maximum_distance: float = 4.0,
    spacing: float = 0.075,
) -> Tuple[Optional[Point], List[Point]]:
    """Select the farthest observed-free body-forward entrance seed.

    This is intentionally pose-relative and cannot encode a layout direction
    or world waypoint.  Every shorter candidate is validated with the same
    footprint/unknown rules used by normal exploration.
    """
    direction = unit(yaw)
    best_target = None
    best_path: List[Point] = []
    step = max(0.20, grid.resolution)
    for travel in np.arange(float(minimum_distance),
                            float(maximum_distance) + step * 0.5, step):
        target = (pose[0] + float(travel) * direction[0],
                  pose[1] + float(travel) * direction[1])
        direct = [pose, target]
        validity = validate_path(grid, direct, spacing=spacing)
        if not validity.valid:
            break
        best_target, best_path = target, direct
    return best_target, best_path


def corridor_end_geometry_evidence(
    grid: OccupancyMap,
    corridor: CorridorRegion,
    pose: Point,
    support_radius: float = 0.48,
) -> bool:
    """Require known occupied support ahead; unknown alone is not an end wall."""
    if not corridor.centerline:
        return False
    axis = unit(corridor.principal_direction)
    origin = corridor.centerline[0]
    station = dot((pose[0] - origin[0], pose[1] - origin[1]), axis)
    center = (origin[0] + station * axis[0], origin[1] + station * axis[1])
    probe_distance = corridor.forward_extent + max(grid.resolution, 0.10)
    probe = (center[0] + probe_distance * axis[0],
             center[1] + probe_distance * axis[1])
    cell = grid.world_to_cell(probe)
    if cell is None:
        return False
    cells = max(1, int(math.ceil(float(support_radius) / grid.resolution)))
    for dy in range(-cells, cells + 1):
        for dx in range(-cells, cells + 1):
            if dx * dx + dy * dy > cells * cells:
                continue
            x, y = cell[0] + dx, cell[1] + dy
            if 0 <= x < grid.width and 0 <= y < grid.height and grid.data[y, x] >= 50:
                return True
    return False


def plan_door_route(
    grid: OccupancyMap,
    current: Point,
    room_goal: Point,
    door: Doorway,
    entering: bool = True,
    sample_spacing: float = 0.075,
    outside_distance: float = 0.85,
    inside_distance: float = 1.50,
) -> Tuple[List[Point], PathValidity]:
    """Plan current->outside->door->inside->goal, or its exact reverse."""
    nx, ny = door.normal
    outside_distance = max(0.45, float(outside_distance))
    inside_distance = max(float(inside_distance), 1.05)
    outside = (door.center[0] - outside_distance * nx,
               door.center[1] - outside_distance * ny)
    inside = (door.center[0] + inside_distance * nx,
              door.center[1] + inside_distance * ny)
    if entering:
        first = astar_free_path(grid, current, outside, allowed_side=(door, -1, 0.08))
        last = astar_free_path(grid, inside, room_goal, allowed_side=(door, 1, 0.08))
        bridge = [outside, door.center, inside]
        route = (first + bridge[1:] + last[1:]) if first and last else []
    else:
        first = astar_free_path(grid, current, inside, allowed_side=(door, 1, 0.08))
        last = astar_free_path(grid, outside, room_goal, allowed_side=(door, -1, 0.08))
        bridge = [inside, door.center, outside]
        route = (first + bridge[1:] + last[1:]) if first and last else []
    validity = validate_path(grid, route, [door], door.door_id, sample_spacing)
    return route, validity


def room_free_region(
    grid: OccupancyMap,
    door: Doorway,
    seed: Point,
    maximum_radius: float = 12.0,
) -> List[Tuple[int, int]]:
    """Flood observed room-side free space while treating the door plane as a barrier."""
    seed_cell = grid.world_to_cell(seed)
    if seed_cell is None or grid.data[seed_cell[1], seed_cell[0]] != 0:
        return []
    queue = [seed_cell]
    visited = {seed_cell}
    for cell in queue:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = cell[0] + dx, cell[1] + dy
            if nxt in visited or not (0 <= nxt[0] < grid.width and 0 <= nxt[1] < grid.height):
                continue
            point = grid.cell_to_world(nxt)
            if grid.data[nxt[1], nxt[0]] != 0:
                continue
            if door.signed_depth(point) < 0.05:
                continue
            if distance(point, seed) > maximum_radius:
                continue
            visited.add(nxt)
            queue.append(nxt)
    return list(visited)


def update_room_geometry(
    room: RoomRegion,
    grid: OccupancyMap,
    door: Doorway,
    robot_pose: Point,
    map_sequence: Optional[int] = None,
) -> RoomRegion:
    depth = door.signed_depth(robot_pose)
    room.maximum_inside_depth = max(room.maximum_inside_depth, depth)
    seed = (door.center[0] + max(1.05, depth) * door.normal[0],
            door.center[1] + max(1.05, depth) * door.normal[1])
    region = room_free_region(grid, door, seed)
    room.observed_free_area = len(region) * grid.resolution ** 2
    if region:
        unknown_boundary = set()
        for cell in region:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                x, y = cell[0] + dx, cell[1] + dy
                if (0 <= x < grid.width and 0 <= y < grid.height and
                        grid.data[y, x] < 0):
                    unknown_boundary.add((x, y))
        # A 2-D online map cannot measure full room volume; use a documented
        # one-metre-height frontier-volume proxy for queue/local scoring.
        room.unknown_volume = len(unknown_boundary) * grid.resolution ** 2
        points = np.asarray([grid.cell_to_world(cell) for cell in region])
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        room.estimated_boundary = [
            (float(minimum[0]), float(minimum[1])),
            (float(maximum[0]), float(minimum[1])),
            (float(maximum[0]), float(maximum[1])),
            (float(minimum[0]), float(maximum[1])),
        ]
        tangent_values = np.asarray([door.lateral_offset(point) for point in points])
        lateral_width = float(tangent_values.max() - tangent_values.min())
        ratio = lateral_width / max(door.width, 0.1)
        room.expansion_confidence = float(np.clip((ratio - 1.0) / 1.0, 0.0, 1.0))
    if map_sequence is None or map_sequence != room.last_map_sequence:
        room.map_growth_history.append(room.observed_free_area)
        if len(room.map_growth_history) > 10:
            room.map_growth_history = room.map_growth_history[-10:]
        if map_sequence is not None:
            room.last_map_sequence = int(map_sequence)
    return room


def select_room_viewpoint(
    room: RoomRegion,
    grid: OccupancyMap,
    door: Doorway,
    minimum_baseline: float = 0.8,
) -> Optional[Point]:
    seed = (door.center[0] + max(1.2, room.maximum_inside_depth) * door.normal[0],
            door.center[1] + max(1.2, room.maximum_inside_depth) * door.normal[1])
    region = room_free_region(grid, door, seed)
    if not region:
        return None
    previous = room.successful_goals
    candidates = []
    for index, cell in enumerate(region):
        if index % max(1, int(0.30 / grid.resolution)):
            continue
        point = grid.cell_to_world(cell)
        if door.signed_depth(point) < 1.0:
            continue
        yaw = door.normal_direction
        # A viewpoint is also a heading-change location.  Checking only the
        # entry normal can select a wall-corner pose that is statically safe
        # but clips the wall as the body turns toward the next viewpoint.
        # Eight orientations conservatively approximate the swept A1 body.
        if any(not grid.footprint_safe(point, yaw + index * math.pi / 4.0)
               for index in range(8)):
            continue
        separation = min((distance(point, old) for old in previous), default=3.0)
        if previous and separation < minimum_baseline:
            continue
        unknown_neighbors = 0
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                x, y = cell[0] + dx, cell[1] + dy
                if 0 <= x < grid.width and 0 <= y < grid.height and grid.data[y, x] < 0:
                    unknown_neighbors += 1
        score = 2.0 * unknown_neighbors + min(separation, 3.0) + 0.2 * door.signed_depth(point)
        candidates.append((score, point))
    return max(candidates, default=(None, None), key=lambda item: item[0])[1]


def select_room_coverage_viewpoint(
    room: RoomRegion,
    grid: OccupancyMap,
    door: Doorway,
    current: Point,
    role: str,
    minimum_baseline: float = 0.8,
    minimum_clearance: float = 0.60,
) -> Optional[dict]:
    """Choose a persistent room-depth or room-breadth macro viewpoint.

    The first target moves to a clear point near the middle depth of the
    door-bounded room component.  Subsequent breadth targets maximize lateral
    separation from completed room viewpoints.  Returned paths reach the full
    macro viewpoint; they are not FUEL rolling look-ahead points.
    """
    if role not in ("primary", "breadth"):
        raise ValueError("room coverage role must be primary or breadth")
    if cv2 is None:
        return None
    seed_depth = max(1.2, door.signed_depth(current), room.maximum_inside_depth)
    seed = (door.center[0] + seed_depth * door.normal[0],
            door.center[1] + seed_depth * door.normal[1])
    region = set(room_free_region(grid, door, seed))
    if not region:
        return None
    mask = np.zeros(grid.data.shape, dtype=np.uint8)
    for x, y in region:
        mask[y, x] = 1
    clearance_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5) * grid.resolution
    depths = [door.signed_depth(grid.cell_to_world(cell)) for cell in region]
    maximum_depth = max(depths, default=0.0)
    target_depth = min(5.0, max(2.5, 0.60 * maximum_depth))
    stride = max(1, int(round(0.30 / grid.resolution)))
    previous = list(room.successful_goals)
    previous_lateral = [door.lateral_offset(point) for point in previous]
    candidates = []
    for sample_index, cell in enumerate(sorted(region)):
        if sample_index % stride:
            continue
        point = grid.cell_to_world(cell)
        depth = door.signed_depth(point)
        lateral = door.lateral_offset(point)
        clearance = float(clearance_map[cell[1], cell[0]])
        if depth < 1.25 or clearance < minimum_clearance:
            continue
        if any(not grid.footprint_safe(
                point, door.normal_direction + orientation * math.pi / 4.0)
               for orientation in range(8)):
            continue
        separation = min((distance(point, old) for old in previous), default=math.inf)
        if previous and separation < minimum_baseline:
            continue
        # Euclidean distance is sufficient for the cheap first-stage rank.
        # Full A* is intentionally deferred to a small shortlist below.
        path_distance = distance(current, point)
        unknown_neighbors = 0
        radius = max(2, int(math.ceil(0.75 / grid.resolution)))
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x, y = cell[0] + dx, cell[1] + dy
                if (0 <= x < grid.width and 0 <= y < grid.height and
                        grid.data[y, x] < 0):
                    unknown_neighbors += 1
        if role == "primary":
            score = (
                -2.5 * abs(depth - target_depth)
                - 0.25 * abs(lateral)
                + 2.0 * min(clearance, 1.5)
                + 0.04 * unknown_neighbors
                - 0.12 * path_distance)
            lateral_gain = abs(lateral)
        else:
            lateral_gain = min(
                (abs(lateral - old) for old in previous_lateral),
                default=abs(lateral))
            score = (
                3.0 * min(lateral_gain, 4.0)
                + 0.30 * min(depth, 5.0)
                + 1.5 * min(clearance, 1.5)
                + 0.06 * unknown_neighbors
                - 0.12 * path_distance)
        candidates.append((score, {
            "position": point,
            "role": role,
            "depth": depth,
            "lateral": lateral,
            "lateral_gain": lateral_gain,
            "clearance": clearance,
            "path_distance": path_distance,
            "unknown_neighbors": unknown_neighbors,
            "maximum_room_depth": maximum_depth,
            "target_primary_depth": target_depth,
        }))
    # Running A* for every free cell made a single room decision take several
    # seconds.  Geometry ranks are optimistic; authorize only a bounded
    # shortlist with the complete room-side and footprint checks.
    authorized = []
    for geometric_score, proposal in sorted(
            candidates, key=lambda item: item[0], reverse=True)[:48]:
        point = proposal["position"]
        path = astar_free_path(
            grid, current, point, clearance_radius=0.16,
            allowed_side=(door, 1, 0.08), allow_unknown_at_start=True)
        validity = validate_path(
            grid, path, spacing=0.075,
            initial_unknown_footprint_grace=0.45)
        if not path or not validity.valid:
            continue
        if any(grid.world_to_cell(path_point) not in region for path_point in path[1:]):
            continue
        exact_path_distance = sum(
            distance(a, b) for a, b in zip(path[:-1], path[1:]))
        exact_score = geometric_score - 0.12 * (
            exact_path_distance - proposal["path_distance"])
        proposal["path"] = path
        proposal["path_distance"] = exact_path_distance
        authorized.append((exact_score, proposal))
    return max(authorized, default=(None, None), key=lambda item: item[0])[1]


def room_complete(room: RoomRegion) -> bool:
    map_stable = False
    if len(room.map_growth_history) >= 3:
        recent = room.map_growth_history[-3:]
        map_stable = max(recent) - min(recent) <= 0.25
    visited_stable = False
    if len(room.visited_growth_history) >= 3:
        recent_visited = room.visited_growth_history[-3:]
        visited_stable = max(recent_visited) - min(recent_visited) <= 0.25
    return (
        room.local_goal_count >= 2
        and room.no_goal_cycles >= 3
        and map_stable
        and visited_stable
        and room.maximum_inside_depth >= 1.0
        and room.observed_free_area >= 1.0
        and room.expansion_confidence > 0.0
    )


class MissionPhase(str, Enum):
    # Adaptive exploration states.  Structure recognition changes scoring, not
    # whether exploration is allowed to run.
    EXPLORE_GENERIC = "EXPLORE_GENERIC"
    EXPLORE_CORRIDOR_AWARE = "EXPLORE_CORRIDOR_AWARE"
    EXPLORE_ROOM_LOCAL = "EXPLORE_ROOM_LOCAL"
    RECOVERY = "RECOVERY"
    CORRIDOR_INITIALIZE = "CORRIDOR_INITIALIZE"
    CORRIDOR_DISCOVERY = "CORRIDOR_DISCOVERY"
    CORRIDOR_END_CONFIRM = "CORRIDOR_END_CONFIRM"
    BUILD_ROOM_QUEUE = "BUILD_ROOM_QUEUE"
    APPROACH_DOOR = "APPROACH_DOOR"
    CROSS_DOOR = "CROSS_DOOR"
    ROOM_LOCAL_EXPLORE = "ROOM_LOCAL_EXPLORE"
    EXIT_ROOM = "EXIT_ROOM"
    RETURN_TO_CORRIDOR = "RETURN_TO_CORRIDOR"
    SELECT_NEXT_ROOM = "SELECT_NEXT_ROOM"
    GLOBAL_RECOVERY = "GLOBAL_RECOVERY"
    FINISHED = "FINISHED"


def unified_candidate_score(
        information_gain: float, local_unknown_volume: float,
        unvisited_area_bonus: float, travel_distance: float,
        collision_cost: float, revisit_penalty: float,
        corridor_progress_bonus: float = 0.0, doorway_bonus: float = 0.0,
        room_entry_bonus: float = 0.0, *, alpha: float = 0.25,
        beta: float = 5.0, gamma: float = 2.0, eta: float = 3.0,
        theta: float = 4.0, kappa: float = 5.0, distance_weight: float = 0.30,
        revisit_weight: float = 2.0) -> float:
    """Score generic and structure-enhanced candidates in one pool.

    Callers pass zero structure bonuses when evidence is absent.  In
    particular, corridor confidence is deliberately not an eligibility gate.
    Collision/path validity remains a separate mandatory filter.
    """
    return (alpha * information_gain + beta * local_unknown_volume
            + gamma * unvisited_area_bonus
            + eta * corridor_progress_bonus + theta * doorway_bonus
            + kappa * room_entry_bonus - distance_weight * travel_distance
            - revisit_weight * revisit_penalty - collision_cost)


@dataclass
class CompletionEvidence:
    all_corridor_branches_complete: bool
    door_queue_empty: bool
    rooms_resolved: bool
    large_reachable_unknown: bool
    no_candidate_cycles: int
    visited_stagnant_seconds: float
    elapsed_seconds: float


def completion_decision(evidence: CompletionEvidence) -> Tuple[bool, Optional[str]]:
    if evidence.elapsed_seconds >= 1800.0:
        return False, "safety_timeout"
    complete = (
        evidence.all_corridor_branches_complete
        and evidence.door_queue_empty
        and evidence.rooms_resolved
        and not evidence.large_reachable_unknown
        and evidence.no_candidate_cycles >= 3
        and evidence.visited_stagnant_seconds >= 60.0
    )
    return complete, "structured_exploration_complete" if complete else None


def build_room_queue(doors: Iterable[Doorway], current: Point, now: float = 0.0) -> List[str]:
    eligible = [door for door in doors if door.confirmed and door.traversable
                and not door.visited and door.blacklist_until <= now]
    eligible.sort(key=lambda door: (
        -door.estimated_unknown_size,
        -door.confidence,
        distance(current, door.center),
        door.attempt_count,
    ))
    return [door.door_id for door in eligible]


def depth_geometry_observation(
    depth_image: np.ndarray,
    horizontal_fov_deg: float = 60.0,
    minimum_depth: float = 0.4,
    maximum_depth: float = 8.0,
    center_fraction: float = 0.5,
    roi_fraction: float = 0.30,
) -> dict:
    """Lightweight depth-only geometry evidence, with no semantic model."""
    depth = np.asarray(depth_image, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        valid = np.logical_and(
            np.isfinite(depth),
            np.logical_and(depth >= minimum_depth, depth <= maximum_depth),
        )
    if depth.ndim != 2 or not np.any(valid):
        return {
            "door_confidence_depth": 0.0,
            "visible_free_depth": 0.0,
            "left_wall_distance": None,
            "right_wall_distance": None,
        }
    height, width = depth.shape
    band = depth[int(0.30 * height):int(0.75 * height)]
    band_valid = valid[int(0.30 * height):int(0.75 * height)]
    half_roi = 0.5 * float(np.clip(roi_fraction, 0.08, 0.60))
    roi_start = float(np.clip(center_fraction - half_roi, 0.0, 1.0))
    roi_end = float(np.clip(center_fraction + half_roi, 0.0, 1.0))
    if roi_end - roi_start < 0.05:
        return {
            "door_confidence_depth": 0.0,
            "visible_free_depth": 0.0,
            "left_wall_distance": None,
            "right_wall_distance": None,
        }
    sectors = []
    span = roi_end - roi_start
    for relative_start, relative_end in ((0.0, 0.28), (0.38, 0.62), (0.72, 1.0)):
        start = roi_start + relative_start * span
        end = roi_start + relative_end * span
        x0, x1 = int(start * width), int(end * width)
        values = band[:, x0:x1][band_valid[:, x0:x1]]
        sectors.append(float(np.median(values)) if values.size else None)
    left, center, right = sectors
    edge_support = sum(value is not None and center is not None and
                       center > value + 0.35 for value in (left, right))
    confidence = 0.0 if center is None else min(1.0, 0.35 + 0.25 * edge_support
                                                + 0.10 * min(center / 4.0, 1.0))
    return {
        "horizontal_fov_deg": float(horizontal_fov_deg),
        "projection_center_fraction": float(center_fraction),
        "projection_roi_fraction": float(roi_fraction),
        "door_confidence_depth": float(confidence),
        "visible_free_depth": 0.0 if center is None else center,
        "left_wall_distance": left,
        "right_wall_distance": right,
    }
