#!/usr/bin/env python3
"""Layout-free doorway crossing detection and coverage-driven room visits.

The online logic consumes only the observed floor projection and the FAST-LIO
trajectory.  Simulator layout metadata is deliberately excluded from this
module; it is reserved for offline evaluation and plotting.
"""

from __future__ import annotations

import math
import itertools
import copy
from collections import deque
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from baseline_planning_core import OccupancyGrid2D, astar_safe_path
from frontier_information_structure import RoomVisibilityGrid


Point = Tuple[float, float]


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _path_prefix_to_observed_point(
        path: Sequence[Sequence[float]], point: Sequence[float]
        ) -> List[List[float]]:
    """Keep only the path prefix that could have been traversed to ``point``.

    ENTRY execution may stop part-way through a preplanned path, or an
    asynchronous corridor-goal preemption may leave the live pose between its
    stored raster samples.  The untraversed suffix must never be reversed as
    an EXIT route: doing so first drives the robot deeper into the room.  Use
    the closest segment projection as the progress boundary, then append the
    observed pose so the reverse trace starts exactly where the robot is now.
    """
    clean = []
    for item in path or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            candidate = [float(item[0]), float(item[1])]
        except (TypeError, ValueError):
            continue
        if (not all(math.isfinite(value) for value in candidate) or
                (clean and _distance(clean[-1], candidate) <= 0.02)):
            continue
        clean.append(candidate)
    try:
        observed = [float(point[0]), float(point[1])]
    except (TypeError, ValueError, IndexError):
        return clean
    if (not all(math.isfinite(value) for value in observed) or not clean):
        return clean
    if len(clean) == 1:
        if _distance(clean[0], observed) > 0.02:
            clean.append(observed)
        return clean

    best = None
    for index in range(len(clean) - 1):
        first, second = clean[index], clean[index + 1]
        dx, dy = second[0] - first[0], second[1] - first[1]
        length_sq = dx * dx + dy * dy
        ratio = (0.0 if length_sq <= 1e-12 else max(0.0, min(1.0,
                 ((observed[0] - first[0]) * dx +
                  (observed[1] - first[1]) * dy) / length_sq)))
        projection = [first[0] + ratio * dx, first[1] + ratio * dy]
        error = _distance(projection, observed)
        # Prefer the later segment on an exact tie. A* paths should not loop,
        # but this keeps the progress estimate monotonic around raster bends.
        candidate = (error, -index, index, projection)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    segment_index, projection = best[2], best[3]
    prefix = [list(item) for item in clean[:segment_index + 1]]
    if not prefix or _distance(prefix[-1], projection) > 0.02:
        prefix.append(projection)
    if _distance(prefix[-1], observed) > 0.02:
        prefix.append(observed)
    else:
        prefix[-1] = observed
    return prefix


def exit_timeout_for_path(path_length: float,
                          config: "LightweightRoomConfig") -> float:
    """Allocate bounded time proportional to the remaining EXIT path."""
    dynamic = (float(config.exit_timeout_overhead_seconds) +
               max(0.0, float(path_length)) *
               float(config.exit_timeout_seconds_per_meter))
    lower = max(float(config.portal_timeout_seconds), dynamic)
    upper = max(float(config.portal_timeout_seconds),
                float(config.exit_timeout_max_seconds))
    return min(lower, upper)


def resume_reversed_entry_trace(points: Sequence[Sequence[float]],
                                current: Sequence[float],
                                maximum_resume_distance: float = 0.75
                                ) -> List[List[float]]:
    """Resume a verified reverse ENTRY trace near the robot.

    After a bounded EXIT timeout the robot can already be halfway through the
    breadcrumb trace. Restarting at its deep-room endpoint sends it backward;
    replacing it with a newly inflated direct portal can be unreachable.
    """
    trace = [[float(point[0]), float(point[1])] for point in points]
    if len(trace) < 2 or current is None or len(current) < 2:
        return trace
    distances = [
        math.hypot(point[0] - float(current[0]),
                   point[1] - float(current[1]))
        for point in trace]
    nearest = min(range(len(trace)), key=distances.__getitem__)
    if (nearest > 0 and nearest < len(trace) - 1 and
            distances[nearest] <= max(0.05, float(maximum_resume_distance))):
        return trace[nearest:]
    return trace


def truncate_reversed_entry_trace_at_corridor(
        points: Sequence[Sequence[float]],
        door: "EstimatedDoorway",
        minimum_corridor_depth: float = 0.60) -> List[List[float]]:
    """Stop a reverse ENTRY replay after the first proven corridor crossing.

    A fallback ENTRY A* can begin several metres before the doorway. Replaying
    its entire prefix on EXIT wastes room time and can carry the robot behind
    the next door station.  The ordered reverse trace is already physically
    traversed evidence, so its first sample at least ``minimum_corridor_depth``
    outside the detected door plane is a sufficient bounded endpoint.  The
    default 0.60 m includes the executor's endpoint tolerance while preserving
    the scheduler's required 0.30 m body-centre crossing.
    """
    trace = [[float(point[0]), float(point[1])] for point in points]
    if len(trace) < 2:
        return trace
    threshold = -max(0.05, float(minimum_corridor_depth))
    for index, point in enumerate(trace):
        if door.depth(point) <= threshold:
            # Preserve at least two ordered breadcrumbs for the preplanned-path
            # executor even when the trace starts just outside the door.
            return trace[:max(2, index + 1)]
    return trace


def _unit(angle: float) -> Point:
    return math.cos(angle), math.sin(angle)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1])


@dataclass
class LightweightRoomConfig:
    enabled: bool = False
    # Separate floors may share the same map x/y coordinates.  Keep their
    # semantic identities distinct even when the same scheduler is reused.
    room_id_prefix: str = "estimated_room_"
    trajectory_spacing: float = 0.20
    doorway_width_min: float = 0.75
    doorway_width_max: float = 2.30
    aperture_probe_range: float = 4.0
    room_depth_probe_range: float = 8.0
    room_side_probe_range: float = 7.5
    expansion_margin: float = 0.80
    confirmation_depth: float = 1.10
    # The entry waypoint is allowed to be shortened by the live occupancy
    # grid.  Keep the nominal confirmation depth as a planning target, but
    # distinguish it from the small, geometric proof that the body centre has
    # actually crossed the doorway plane.
    minimum_crossing_depth: float = 0.18
    duplicate_door_radius: float = 1.50
    goal_clearance: float = 0.38
    # A doorway staging pose only proves that the robot crossed the plane;
    # the subsequent G1 scan chooses the normal room clearance.  Allow this
    # short, SCAN-lite-verified portal segment to use the observed opening
    # rather than rejecting a real narrow projection after small map drift.
    entry_staging_clearance: float = 0.26
    entry_depth: float = 1.50
    side_depth: float = 3.00
    near_wall_margin: float = 0.65
    minimum_goal_separation: float = 0.90
    exit_offset: float = 0.90
    exit_retry_limit: int = 2
    entry_retry_limit: int = 2
    candidate_confirmation_count: int = 3
    # Corridor-side detection historically committed after one raster so a
    # fast floor-1 sweep would not pass a real door.  Keep that behaviour as
    # the default, while allowing the separately constructed floor-2 mission
    # to demand repeat evidence in its less stable post-stair map.
    corridor_side_confirmation_count: int = 1
    # A shallow staging route is useful on floor 1, where the doorway has
    # already been validated by the established map.  Floor 2 may opt out:
    # after stair handoff a false aperture can otherwise be labelled as an
    # entered room even though no verified portal exists.
    require_portal_preflight_for_entry: bool = False
    door_takeover_distance: float = 5.0
    # Occupancy evidence around a doorway often becomes complete only after the
    # lidar has moved several metres past it.  Keep enough recent corridor in
    # the search window to recover that door instead of losing the room for the
    # remainder of the sweep.
    door_lookbehind_distance: float = 6.0
    door_approach_offset: float = 0.60
    room_budget_seconds: float = 75.0
    budget_starts_after_entry: bool = False
    room_goal_timeout_seconds: float = 28.0
    portal_timeout_seconds: float = 20.0
    entry_timeout_seconds: float = 45.0
    # Reject a doorway commit whose online A* route has folded into a long
    # detour around a drifted/phantom portal.  Such a route is not a safe
    # alternative entrance: it is a false candidate that otherwise consumes
    # the whole entry progress budget before other doors can be considered.
    entry_preflight_max_path_m: float = 7.5
    entry_preflight_max_detour_ratio: float = 3.0
    # Bound expensive global searches for a drifted ENTRY candidate. Normal
    # portals succeed among the highest-ranked few endpoints.
    entry_candidate_astar_limit: int = 40
    # Floor 1 may latch a door while a fast corridor goal is still
    # decelerating.  When explicitly enabled by that floor, refresh the ENTRY
    # route from the post-preemption pose and retain only the actually reached
    # prefix for a later reverse EXIT. Upper floors keep their validated
    # behaviour through the default false value.
    entry_trace_observed_progress_only: bool = False
    door_cooldown_seconds: float = 90.0
    door_evidence_retry_seconds: float = 3.0
    exit_reserve_seconds: float = 20.0
    exit_progress_timeout_seconds: float = 8.0
    exit_timeout_seconds_per_meter: float = 4.0
    exit_timeout_overhead_seconds: float = 8.0
    exit_timeout_max_seconds: float = 60.0
    exit_anchor_skip_distance: float = 0.55
    # A fresh upper-floor map can move the re-estimated door plane after the
    # robot has entered a room.  When explicitly enabled by that isolated
    # floor manager, reaching the corridor start of the *same physically
    # traversed ENTRY trace* while reversing it is stronger exit evidence than
    # the shifted plane depth.  The default stays disabled so floor-1 and
    # ordinary portal exits retain their existing confirmation policy.
    accept_reversed_entry_trace_exit: bool = False
    reversed_entry_trace_exit_tolerance: float = 0.30
    # EXIT consists of four short doorway-relative segments. If a freshly
    # inflated grid disconnects one segment, do not search the entire floor
    # before selecting the observed ENTRY-trace fallback.
    exit_preflight_maximum_expansions: int = 800
    direct_verified_exit: bool = False
    semantic_completion_tolerance: float = 0.80
    side_goal_retry_limit: int = 2
    coverage_minimum_ratio: float = 0.90
    coverage_maximum_unknown_m2: float = 2.00
    coverage_maximum_shadow_m2: float = 1.25
    coverage_minimum_observations: int = 1
    # Planning range, not the advertised sensor maximum.  Distant floor and
    # grazing-wall returns are not reliable enough to erase room coverage
    # debt merely because an ideal two-dimensional ray reaches them.
    lidar_effective_range: float = 8.0
    maximum_center_depth: float = 3.80
    maximum_side_lateral: float = 3.00
    # Bound the optional RGB-D breadth recovery to a local baseline around
    # G1 rather than a second deep-room excursion.
    visual_breadth_lateral: float = 0.65
    visual_breadth_min_baseline: float = 1.20
    visual_two_pose_lateral_enabled: bool = False
    # Select the smallest useful set of room viewpoints from online coverage
    # debt instead of always executing G1 -> G3 -> G4.  This remains opt-in so
    # baseline/ablation launch files keep their original semantics.
    adaptive_minimal_viewpoints: bool = False
    adaptive_maximum_viewpoints: int = 3
    # In a timed visual mission, use the central view before a lateral
    # coverage view.  G1 observes the far wall and camera fan in one transit;
    # later side views remain available only when the resulting *measured*
    # coverage debt requires them.
    adaptive_prefer_central_first: bool = False
    adaptive_occlusion_shadow_trigger_m2: float = 0.55
    adaptive_minimum_gain_m2: float = 0.12
    adaptive_path_cost_weight: float = 0.32
    adaptive_set_cover_ratio: float = 0.88
    adaptive_set_cover_residual_m2: float = 0.75
    adaptive_route_return_weight: float = 0.65
    # Ground LiDAR often observes only one thin jamb in the 2-D floor
    # projection.  Disabled by default; semantic FUEL runs may enable it when
    # three-frame confidence and strong room-side expansion are also present.
    allow_single_jamb_strong_expansion: bool = False
    single_jamb_minimum_confidence: float = 0.82
    # A new doorway opposite a visited room may project through that room when
    # FAST-LIO has a small accumulated translation error.  Only suppress a
    # candidate as a duplicate when both inward normals still point to the
    # same side of the corridor.
    visited_room_overlap_min_normal_alignment: float = 0.25
    # A ray overlap between doors at different corridor stations is caused by
    # the expanding map projection, not a duplicate doorway.
    visited_room_overlap_max_door_separation: float = 2.50

    def validate(self) -> None:
        if self.trajectory_spacing <= 0.0:
            raise ValueError("room trajectory spacing must be positive")
        if not 0.4 <= self.doorway_width_min < self.doorway_width_max:
            raise ValueError("invalid doorway width interval")
        if (self.confirmation_depth <= 0.0 or
                self.minimum_crossing_depth <= 0.0 or
                self.minimum_crossing_depth > self.confirmation_depth or
                self.expansion_margin <= 0.0):
            raise ValueError("invalid room crossing thresholds")
        if (self.room_depth_probe_range <= self.entry_depth or
                self.room_side_probe_range <= self.near_wall_margin):
            raise ValueError("invalid room interior probe ranges")
        if (min(self.entry_depth, self.side_depth,
                self.entry_staging_clearance) <= 0.0 or
                self.entry_staging_clearance > self.goal_clearance):
            raise ValueError("invalid room goal depths")
        if not -1.0 <= self.visited_room_overlap_min_normal_alignment <= 1.0:
            raise ValueError("invalid visited-room overlap normal alignment")
        if self.visited_room_overlap_max_door_separation <= 0.0:
            raise ValueError("invalid visited-room overlap door separation")
        if min(self.exit_retry_limit, self.entry_retry_limit,
               self.candidate_confirmation_count,
               self.corridor_side_confirmation_count) < 1:
            raise ValueError("room retry and confirmation counts must be positive")
        if min(self.room_budget_seconds, self.room_goal_timeout_seconds,
               self.portal_timeout_seconds, self.entry_timeout_seconds,
               self.door_cooldown_seconds) <= 0.0:
            raise ValueError("room timing budgets must be positive")
        if (self.entry_preflight_max_path_m <= 0.0 or
                self.entry_preflight_max_detour_ratio <= 1.0 or
                self.entry_candidate_astar_limit < 1):
            raise ValueError("invalid entry preflight detour limits")
        if (min(self.door_takeover_distance, self.door_approach_offset) <= 0.0 or
                self.door_lookbehind_distance < 0.0):
            raise ValueError("door takeover distances must be positive")
        if (self.exit_reserve_seconds <= 0.0 or
                self.exit_progress_timeout_seconds <= 0.0 or
                self.exit_timeout_seconds_per_meter <= 0.0 or
                self.exit_timeout_overhead_seconds < 0.0 or
                self.exit_timeout_max_seconds <= 0.0 or
                self.exit_anchor_skip_distance <= 0.0 or
                self.reversed_entry_trace_exit_tolerance <= 0.0 or
                self.exit_preflight_maximum_expansions < 1 or
                self.side_goal_retry_limit < 1):
            raise ValueError("invalid room scheduling configuration")
        if not 0.0 < self.coverage_minimum_ratio <= 1.0:
            raise ValueError("invalid room coverage ratio")
        if min(self.coverage_maximum_unknown_m2,
               self.coverage_maximum_shadow_m2) < 0.0:
            raise ValueError("invalid room coverage debt limits")
        if (self.coverage_minimum_observations < 1 or
                min(self.lidar_effective_range, self.maximum_center_depth,
                    self.maximum_side_lateral) <= 0.0):
            raise ValueError("invalid room coverage observation configuration")
        if not 0.0 <= self.single_jamb_minimum_confidence <= 1.0:
            raise ValueError("invalid single-jamb confidence threshold")
        if (self.adaptive_maximum_viewpoints < 1 or
                min(self.adaptive_occlusion_shadow_trigger_m2,
                    self.adaptive_minimum_gain_m2,
                    self.adaptive_path_cost_weight) < 0.0):
            raise ValueError("invalid adaptive room viewpoint configuration")
        if (not 0.0 < self.adaptive_set_cover_ratio <= 1.0 or
                self.adaptive_set_cover_residual_m2 < 0.0 or
                self.adaptive_route_return_weight < 0.0):
            raise ValueError("invalid adaptive room set-cover configuration")


@dataclass
class EstimatedDoorway:
    door_id: str
    center: Point
    normal_direction: float
    width: float
    left_frame_point: Point
    right_frame_point: Point
    corridor_side: Point
    interior_side: Point
    confidence: float
    entered_at: float
    visited: bool = False
    completed: bool = False
    coverage_complete: bool = False
    temporarily_failed: bool = False

    @property
    def normal(self) -> Point:
        return _unit(self.normal_direction)

    @property
    def tangent(self) -> Point:
        nx, ny = self.normal
        return -ny, nx

    def depth(self, point: Sequence[float]) -> float:
        return _dot((float(point[0]) - self.center[0],
                     float(point[1]) - self.center[1]), self.normal)

    def lateral(self, point: Sequence[float]) -> float:
        return _dot((float(point[0]) - self.center[0],
                     float(point[1]) - self.center[1]), self.tangent)

    def to_dict(self) -> dict:
        return asdict(self)


def canonical_doorway_anchors(center: Point, normal_direction: float,
                              config: LightweightRoomConfig
                              ) -> Tuple[Point, Point]:
    """Return semantic anchors guaranteed to lie on opposite portal sides.

    Occupancy evidence can mature only after the robot has crossed the wall
    plane, so a trajectory sample labelled ``before`` may already be inside
    the room.  Deriving both anchors from the observed aperture centre and its
    corridor-to-room normal prevents a later EXIT from stopping short.
    """
    nx, ny = _unit(normal_direction)
    outside_depth = max(config.exit_offset, config.door_approach_offset, 0.60)
    inside_depth = max(config.entry_depth,
                       config.confirmation_depth + 0.20)
    corridor_side = (center[0] - outside_depth * nx,
                     center[1] - outside_depth * ny)
    interior_side = (center[0] + inside_depth * nx,
                     center[1] + inside_depth * ny)
    return corridor_side, interior_side


def _state(grid: OccupancyGrid2D, point: Point) -> int:
    cell = grid.world_to_cell(point)
    return -1 if cell is None else int(grid.data[cell[1], cell[0]])


def _ray_distance(grid: OccupancyGrid2D, point: Point, angle: float,
                  maximum: float) -> float:
    """Distance to the first occupied or unknown cell along one ray."""
    step = max(0.04, 0.45 * grid.resolution)
    direction = _unit(angle)
    distance = step
    while distance <= maximum + 1e-9:
        sample = (point[0] + distance * direction[0],
                  point[1] + distance * direction[1])
        if _state(grid, sample) != 0:
            return distance
        distance += step
    return maximum


def _parallel_side_extent(grid: OccupancyGrid2D, door: EstimatedDoorway,
                          anchor: Point, side_sign: float,
                          maximum: float) -> Tuple[float, List[float]]:
    """Estimate a side wall without mistaking one item of furniture for it.

    A room wall blocks every nearby ray, while a chair/cabinet normally blocks
    only part of the depth band.  The upper robust extent from five parallel
    rays therefore preserves reachable free space behind lateral obstacles.
    """
    samples = []
    nx, ny = door.normal
    angle = door.normal_direction + side_sign * 0.5 * math.pi
    for offset in (-1.20, -0.60, 0.0, 0.60, 1.20):
        origin = (anchor[0] + offset * nx, anchor[1] + offset * ny)
        if door.depth(origin) < 1.10 or _state(grid, origin) != 0:
            continue
        samples.append(_ray_distance(grid, origin, angle, maximum))
    if not samples:
        return _ray_distance(grid, anchor, angle, maximum), []
    ordered = sorted(samples)
    # Second-largest when possible rejects one raster leak while retaining a
    # clear parallel view around a central obstacle.
    extent = ordered[-2] if len(ordered) >= 3 else ordered[-1]
    return extent, samples


def aperture_width(grid: OccupancyGrid2D, point: Point,
                   travel_direction: float, maximum: float = 4.0) -> float:
    tangent = travel_direction + 0.5 * math.pi
    return (_ray_distance(grid, point, tangent, maximum) +
            _ray_distance(grid, point, tangent + math.pi, maximum))


def doorway_is_lateral_to_corridor(normal_direction: float,
                                   corridor_axis: Sequence[float],
                                   maximum_alignment: float = 0.55) -> bool:
    """Whether a crossed aperture points across, rather than along, a corridor."""
    if corridor_axis is None or len(corridor_axis) < 2:
        return True
    norm = math.hypot(float(corridor_axis[0]), float(corridor_axis[1]))
    if norm <= 1e-9:
        return True
    normal = _unit(float(normal_direction))
    alignment = abs(
        normal[0] * float(corridor_axis[0]) / norm +
        normal[1] * float(corridor_axis[1]) / norm)
    return alignment <= max(0.0, min(1.0, float(maximum_alignment)))


def semantic_corridor_branch_candidate(
        grid: OccupancyGrid2D, centerline: Point,
        corridor_axis: Sequence[float], side: int, inside_point: Point,
        now: float, config: LightweightRoomConfig,
        door_sequence: int = 1) -> Optional[EstimatedDoorway]:
    """Promote a lidar-visible side branch to a geometry-checked doorway.

    This is the semantic complement to trajectory aperture detection.  It is
    used only after the corridor scheduler has already supplied a persistent
    lateral frontier and a known-free A*/SCAN-reachable penetration point.
    Candidate centres are still found from the observed jamb aperture; no
    layout coordinate or truth room boundary is used.
    """
    if corridor_axis is None or len(corridor_axis) < 2:
        return None
    norm = math.hypot(float(corridor_axis[0]), float(corridor_axis[1]))
    if norm <= 1e-9 or int(side) == 0:
        return None
    axis = (float(corridor_axis[0]) / norm,
            float(corridor_axis[1]) / norm)
    normal = (-axis[1] * (1.0 if int(side) > 0 else -1.0),
              axis[0] * (1.0 if int(side) > 0 else -1.0))
    heading = math.atan2(normal[1], normal[0])
    relative_inside = (float(inside_point[0]) - float(centerline[0]),
                       float(inside_point[1]) - float(centerline[1]))
    penetration = _dot(relative_inside, normal)
    if penetration < max(1.05, config.door_approach_offset + 0.35):
        return None
    # Search across the online corridor wall band.  At the centreline the
    # tangent ray is long; at a real wall opening it becomes the jamb-to-jamb
    # width.  The best supported narrow aperture wins.
    depth_step = max(0.08, grid.resolution)
    station_step = max(0.10, grid.resolution)
    candidates = []
    depth = 0.45
    while depth <= min(1.75, penetration - 0.20) + 1e-9:
        station_offset = -1.20
        while station_offset <= 1.20 + 1e-9:
            center = (
                float(centerline[0]) + depth * normal[0] +
                station_offset * axis[0],
                float(centerline[1]) + depth * normal[1] +
                station_offset * axis[1])
            if _state(grid, center) != 0:
                station_offset += station_step
                continue
            left_distance = _ray_distance(
                grid, center, heading + 0.5 * math.pi,
                config.doorway_width_max)
            right_distance = _ray_distance(
                grid, center, heading - 0.5 * math.pi,
                config.doorway_width_max)
            # The branch station is where the robot happened to cross, often
            # near one jamb.  Recenter on the midpoint of the two observed
            # frame hits before storing semantic portal geometry; otherwise a
            # valid ENTRY can later produce an EXIT line grazing that jamb.
            midpoint_shift = 0.5 * (left_distance - right_distance)
            if abs(midpoint_shift) <= 0.75:
                midpoint = (center[0] + midpoint_shift * axis[0],
                            center[1] + midpoint_shift * axis[1])
                if _state(grid, midpoint) == 0:
                    center = midpoint
                    left_distance = _ray_distance(
                        grid, center, heading + 0.5 * math.pi,
                        config.doorway_width_max)
                    right_distance = _ray_distance(
                        grid, center, heading - 0.5 * math.pi,
                        config.doorway_width_max)
            width = left_distance + right_distance
            if not (config.doorway_width_min <= width <=
                    config.doorway_width_max):
                station_offset += station_step
                continue
            left = (center[0] + left_distance * axis[0],
                    center[1] + left_distance * axis[1])
            right = (center[0] - right_distance * axis[0],
                     center[1] - right_distance * axis[1])
            frame_count = int(_occupied_near(grid, left)) + int(
                _occupied_near(grid, right))
            if frame_count == 0:
                station_offset += station_step
                continue
            corridor_side = (
                center[0] - config.door_approach_offset * normal[0],
                center[1] - config.door_approach_offset * normal[1])
            if (_state(grid, corridor_side) != 0 or
                    _state(grid, inside_point) != 0):
                station_offset += station_step
                continue
            inside_depth = _dot((float(inside_point[0]) - center[0],
                                 float(inside_point[1]) - center[1]), normal)
            if inside_depth < 0.30:
                station_offset += station_step
                continue
            inside_width = aperture_width(
                grid, inside_point, heading, config.aperture_probe_range)
            required_expansion = (min(config.expansion_margin, 0.30)
                                  if frame_count == 2 else
                                  config.expansion_margin)
            if inside_width < width + required_expansion:
                station_offset += station_step
                continue
            # Prefer bilateral jambs, a centre close to the branch station,
            # and enough room-side penetration to make ENTRY resumable.
            confidence = min(1.0, 0.62 + 0.14 * frame_count +
                             0.10 * min(1.0, inside_depth) -
                             0.04 * abs(station_offset))
            asymmetry = abs(left_distance - right_distance) / max(0.1, width)
            candidates.append((
                -frame_count, asymmetry, abs(station_offset), -inside_depth,
                EstimatedDoorway(
                    door_id="estimated_door_{:02d}".format(door_sequence),
                    center=center, normal_direction=heading, width=width,
                    left_frame_point=left, right_frame_point=right,
                    corridor_side=corridor_side,
                    interior_side=(float(inside_point[0]),
                                   float(inside_point[1])),
                    confidence=confidence, entered_at=float(now))))
            station_offset += station_step
        depth += depth_step
    return min(candidates, key=lambda item: item[:4])[4] if candidates else None


def _spaced_points(trajectory: Sequence[Sequence[float]], spacing: float) -> List[Point]:
    points: List[Point] = []
    for item in trajectory:
        point = (float(item[0]), float(item[1]))
        if not points or _distance(points[-1], point) >= spacing:
            points.append(point)
    return points


class DoorwayCrossingDetector:
    """Detect a narrow aperture bracketed by wider free space on both sides."""

    def __init__(self, config: LightweightRoomConfig):
        config.validate()
        self.config = config
        self.doors: List[EstimatedDoorway] = []
        self.last_detection_path_length = 0.0

    def _duplicate(self, center: Point,
                   normal_direction: Optional[float] = None) -> bool:
        # A doorway is a physical landmark, not a successful-task record.
        # Near-coincident centres are the same portal even if traversal flips
        # the normal.  At larger separation, preserve real opposing corridor
        # doors and merge only similarly directed entrances.
        for door in self.doors:
            distance = _distance(center, door.center)
            if distance <= min(0.75, 0.5 * self.config.duplicate_door_radius):
                return True
            if distance > self.config.duplicate_door_radius:
                continue
            if normal_direction is None:
                return True
            if math.cos(float(normal_direction) -
                        door.normal_direction) >= 0.50:
                return True
        return False

    def _candidate_from_points(self, grid: OccupancyGrid2D,
                               raw_points: Sequence[Sequence[float]],
                               now: float, require_unvisited: bool = True
                               ) -> Optional[EstimatedDoorway]:
        points = _spaced_points(raw_points, self.config.trajectory_spacing)
        if len(points) < 9:
            return None
        candidates = []
        for index in range(3, len(points) - 4):
            before, after = points[index - 3], points[index + 4]
            if _distance(before, after) < 1.2:
                continue
            angle = math.atan2(after[1] - before[1], after[0] - before[0])
            widths = [aperture_width(
                grid, points[j], angle, self.config.aperture_probe_range)
                for j in range(index - 2, index + 3)]
            minimum_width = min(widths)
            minimum_index = index - 2 + widths.index(minimum_width)
            center = points[minimum_index]
            before_index = max(0, minimum_index - 3)
            after_index = min(len(points) - 1, minimum_index + 4)
            before_width = aperture_width(
                grid, points[before_index], angle,
                self.config.aperture_probe_range)
            after_width = aperture_width(
                grid, points[after_index], angle,
                self.config.aperture_probe_range)
            if not (self.config.doorway_width_min <= minimum_width <=
                    self.config.doorway_width_max):
                continue
            if min(before_width, after_width) < (
                    minimum_width + self.config.expansion_margin):
                continue
            if _distance(center, points[-1]) < self.config.confirmation_depth:
                continue
            if require_unvisited and self._duplicate(center, angle):
                continue
            tangent = _unit(angle + 0.5 * math.pi)
            left_distance = _ray_distance(
                grid, center, angle + 0.5 * math.pi,
                self.config.doorway_width_max)
            right_distance = _ray_distance(
                grid, center, angle - 0.5 * math.pi,
                self.config.doorway_width_max)
            left = (center[0] + left_distance * tangent[0],
                    center[1] + left_distance * tangent[1])
            right = (center[0] - right_distance * tangent[0],
                     center[1] - right_distance * tangent[1])
            symmetry = 1.0 - min(1.0, abs(left_distance - right_distance) /
                                 max(0.1, minimum_width))
            expansion = min(1.0, (min(before_width, after_width) - minimum_width) /
                            max(0.1, self.config.expansion_margin * 2.0))
            confidence = 0.55 + 0.20 * symmetry + 0.25 * expansion
            candidates.append((confidence, minimum_index, center, angle,
                               minimum_width, left, right, before_index,
                               after_index))
        if not candidates:
            return None
        (confidence, _, center, angle, width, left, right,
         before_index, after_index) = max(candidates)
        return EstimatedDoorway(
            door_id="estimated_door_{:02d}".format(len(self.doors) + 1),
            center=center, normal_direction=angle, width=width,
            left_frame_point=left, right_frame_point=right,
            # Preserve the physically traversed samples for ENTRY.  Exit-only
            # canonicalization happens after ROOM_ENTERED and must not alter
            # the previously proven door detector/takeover behaviour.
            corridor_side=points[before_index],
            interior_side=points[after_index],
            confidence=min(1.0, confidence), entered_at=float(now))

    def detect(self, grid: OccupancyGrid2D,
               trajectory: Sequence[Sequence[float]], now: float) -> Optional[EstimatedDoorway]:
        points = _spaced_points(trajectory, self.config.trajectory_spacing)
        if len(points) < 9:
            return None
        # Only the recent travelled geometry is relevant.  Keeping a modest
        # tail also lets a newly grown map confirm a crossing one cycle later.
        points = points[-36:]
        cumulative = [0.0]
        for first, second in zip(points[:-1], points[1:]):
            cumulative.append(cumulative[-1] + _distance(first, second))
        if cumulative[-1] <= self.last_detection_path_length + 0.35:
            return None
        door = self._candidate_from_points(grid, points, now)
        if door is None:
            return None
        self.doors.append(door)
        self.last_detection_path_length = max(self.last_detection_path_length,
                                              cumulative[-1])
        return door

    def planned_candidate(self, grid: OccupancyGrid2D,
                          path: Sequence[Sequence[float]],
                          now: float) -> Optional[EstimatedDoorway]:
        """Find a corridor-to-room aperture on a path before it is executed."""
        return self._candidate_from_points(grid, path, now)

    def corridor_side_candidate(self, grid: OccupancyGrid2D, current: Point,
                                corridor_heading: float, now: float,
                                lookahead: Optional[float] = None,
                                lookbehind: Optional[float] = None
                                ) -> Optional[EstimatedDoorway]:
        """Detect a side opening ahead or just behind the current projection.

        OctoMap evidence can mature one planning cycle after the robot passes a
        doorway.  A short look-behind window recovers that door without a full
        corridor reversal.
        """
        along = _unit(corridor_heading)
        step = max(0.10, grid.resolution)
        candidates = []
        lookahead = (self.config.door_takeover_distance if lookahead is None
                     else float(lookahead))
        lookbehind = (self.config.door_lookbehind_distance
                      if lookbehind is None else max(0.0, float(lookbehind)))
        first_index = -int(math.ceil(lookbehind / step))
        last_index = max(4, int(math.floor(lookahead / step)))
        for side_sign in (-1.0, 1.0):
            side_heading = corridor_heading + side_sign * 0.5 * math.pi
            side = _unit(side_heading)
            samples = []
            for index in range(first_index, last_index + 1):
                longitudinal = index * step
                centerline = (current[0] + longitudinal * along[0],
                              current[1] + longitudinal * along[1])
                distance = _ray_distance(
                    grid, centerline, side_heading, self.config.aperture_probe_range)
                samples.append((longitudinal, centerline, distance))
            ordered = sorted(item[2] for item in samples)
            baseline = ordered[max(0, int(0.25 * (len(ordered) - 1)))]
            openings = [item[2] >= baseline + self.config.expansion_margin
                        for item in samples]
            start = 0
            while start < len(samples):
                if not openings[start]:
                    start += 1
                    continue
                end = start
                while end + 1 < len(samples) and openings[end + 1]:
                    end += 1
                width = (end - start + 1) * step
                if (self.config.doorway_width_min <= width <=
                        self.config.doorway_width_max):
                    middle = (start + end) // 2
                    _, centerline, open_distance = samples[middle]
                    center = (centerline[0] + baseline * side[0],
                              centerline[1] + baseline * side[1])
                    corridor_side = (
                        center[0] - self.config.door_approach_offset * side[0],
                        center[1] - self.config.door_approach_offset * side[1])
                    interior_side = (center[0] + self.config.entry_depth * side[0],
                                     center[1] + self.config.entry_depth * side[1])
                    if (_state(grid, center) == 0 and
                            _state(grid, interior_side) == 0 and
                            not self._duplicate(center, side_heading)):
                        tangent = along
                        left = (center[0] + 0.5 * width * tangent[0],
                                center[1] + 0.5 * width * tangent[1])
                        right = (center[0] - 0.5 * width * tangent[0],
                                 center[1] - 0.5 * width * tangent[1])
                        expansion = min(1.0, (open_distance - baseline) /
                                        max(0.1, 2.0 * self.config.expansion_margin))
                        confidence = 0.65 + 0.30 * expansion
                        candidates.append(EstimatedDoorway(
                            door_id="estimated_door_{:02d}".format(
                                len(self.doors) + 1),
                            center=center, normal_direction=side_heading,
                            width=width, left_frame_point=left,
                            right_frame_point=right,
                            corridor_side=corridor_side,
                            interior_side=interior_side,
                            confidence=min(1.0, confidence), entered_at=float(now)))
                start = end + 1
        return max(candidates, key=lambda item: item.confidence,
                   default=None)


def _clearance(grid: OccupancyGrid2D, point: Point, maximum: float = 1.5) -> float:
    return min(_ray_distance(grid, point, angle, maximum)
               for angle in (0.0, math.pi / 4.0, math.pi / 2.0,
                             3.0 * math.pi / 4.0, math.pi,
                             5.0 * math.pi / 4.0, 3.0 * math.pi / 2.0,
                             7.0 * math.pi / 4.0))


def _occupied_near(grid: OccupancyGrid2D, point: Point,
                   radius: float = 0.30) -> bool:
    cell = grid.world_to_cell(point)
    if cell is None:
        return False
    bound = max(1, int(math.ceil(radius / grid.resolution)))
    for dy in range(-bound, bound + 1):
        for dx in range(-bound, bound + 1):
            if (dx * grid.resolution) ** 2 + (dy * grid.resolution) ** 2 > radius ** 2:
                continue
            x, y = cell[0] + dx, cell[1] + dy
            if (0 <= x < grid.width and 0 <= y < grid.height and
                    int(grid.data[y, x]) >= 50):
                return True
    return False


def portal_clearance(grid: OccupancyGrid2D, door: EstimatedDoorway,
                     config: LightweightRoomConfig) -> float:
    """Clearance that fits the measured opening but never shrinks below A1."""
    return max(0.17, min(config.goal_clearance,
                         0.5 * door.width - grid.resolution - 0.03))


def doorway_candidate_valid(grid: OccupancyGrid2D, door: EstimatedDoorway,
                            config: LightweightRoomConfig) -> Tuple[bool, str]:
    left_frame = _occupied_near(grid, door.left_frame_point)
    right_frame = _occupied_near(grid, door.right_frame_point)
    frame_count = int(left_frame) + int(right_frame)
    if frame_count == 0:
        return False, "missing_occupied_doorframe_support"
    for label, point in (("center", door.center),
                         ("corridor_side", door.corridor_side),
                         ("interior_side", door.interior_side)):
        if _state(grid, point) != 0:
            return False, label + "_not_observed_free"
    inside_width = aperture_width(
        grid, door.interior_side, door.normal_direction,
        config.aperture_probe_range)
    # Once both occupied jambs are present, a modest widening plus the portal
    # A* preflight is sufficient.  Requiring the full 0.8 m on an early map
    # rejected the real lower seed-77 door when furniture clipped the lateral
    # room-side ray.  A single jamb still needs the full configured expansion.
    strong_single_jamb = bool(
        frame_count == 1 and config.allow_single_jamb_strong_expansion and
        door.confidence >= config.single_jamb_minimum_confidence)
    required_expansion = (min(config.expansion_margin, 0.30)
                          if frame_count == 2 or strong_single_jamb
                          else config.expansion_margin)
    if inside_width < door.width + required_expansion:
        return False, "insufficient_room_side_expansion"
    if frame_count == 1:
        if (config.allow_single_jamb_strong_expansion and
                door.confidence >= config.single_jamb_minimum_confidence):
            # The caller already debounces this geometry over independent map
            # projections.  Known-free portal samples, full room-side
            # expansion and A*/SCAN preflight remain mandatory, so this does
            # not turn an unknown-space centroid into a navigation target.
            return True, "door_geometry_valid_single_jamb_strong_expansion"
        # A single jamb plus open space also occurs at the lobby-to-corridor
        # transition.  debug6 proved that accepting it creates a convincing
        # but false room in the open lobby.  Delayed candidates remain in the
        # longitudinal look-behind window while their second jamb matures.
        return False, "missing_bilateral_occupied_doorframe_support"
    return True, "door_geometry_valid_two_jambs"


def doorway_candidate_entry_ready(grid: OccupancyGrid2D,
                                  door: EstimatedDoorway) -> Tuple[bool, str]:
    """Minimal gate for a promoted local doorway candidate.

    A promoted candidate is already the result of the local opening detector.
    Do not apply a second semantic door-frame/width confirmation here: that
    used to discard real openings when one jamb was missing in the projected
    map.  Keep only the checks needed to avoid publishing an obviously
    unsafe goal; FAR/A* and SCAN-lite remain the final safety gates.
    """
    # At least one anchor must be observed free.  Any single anchor is enough
    # to start a local staging attempt; demanding specifically the corridor
    # side was brittle when the scan clipped a jamb or FAST-LIO shifted the
    # doorway raster by one or two cells.
    if all(_state(grid, point) != 0 for point in
           (door.corridor_side, door.center, door.interior_side)):
        return False, "candidate_no_observed_free_anchor"
    return True, "promoted_candidate_observed_free_anchor"


def _snap_safe_point(grid: OccupancyGrid2D, desired: Point,
                     clearance: float, door: EstimatedDoorway,
                     depth_tolerance: float,
                     search_radius: float = 0.45) -> Optional[Point]:
    cell = grid.world_to_cell(desired)
    if cell is None:
        return None
    bound = max(1, int(math.ceil(search_radius / grid.resolution)))
    desired_depth = door.depth(desired)
    candidates = []
    for dy in range(-bound, bound + 1):
        for dx in range(-bound, bound + 1):
            candidate_cell = cell[0] + dx, cell[1] + dy
            if not (0 <= candidate_cell[0] < grid.width and
                    0 <= candidate_cell[1] < grid.height and
                    int(grid.data[candidate_cell[1], candidate_cell[0]]) == 0):
                continue
            point = grid.cell_to_world(candidate_cell)
            if (abs(door.depth(point) - desired_depth) > depth_tolerance or
                    _distance(point, desired) > search_radius or
                    _clearance(grid, point) < clearance):
                continue
            candidates.append((_distance(point, desired), point))
    return min(candidates)[1] if candidates else None


def prepare_portal_path(grid: OccupancyGrid2D, door: EstimatedDoorway,
                        current: Point, entry_target: Point,
                        config: LightweightRoomConfig) -> Optional[dict]:
    clearance = portal_clearance(grid, door, config)
    resume_inside = door.depth(current) >= 0.15
    if resume_inside:
        # A partially successful commit has already crossed the mandatory
        # centre.  Retrying via corridor_side would undo progress and can
        # consume the entire room budget; continue directly to confirmation.
        desired = (entry_target,)
        tolerances = (0.45,)
    else:
        desired = (door.corridor_side, door.center, entry_target)
        tolerances = (0.35, 0.20, 0.45)
    snapped = [_snap_safe_point(grid, point, clearance, door, tolerance)
               for point, tolerance in zip(desired, tolerances)]
    if any(point is None for point in snapped):
        return None
    start = current
    combined = []
    for index, target in enumerate(snapped):
        result = astar_safe_path(
            grid, start, target, clearance, 0.15,
            allow_blocked_start=(index == 0))
        if not result.get("success"):
            return None
        path = list(result.get("path", []))
        combined.extend(path if not combined else path[1:])
        start = target
    length = sum(_distance(combined[index - 1], combined[index])
                 for index in range(1, len(combined)))
    direct = _distance(current, snapped[-1])
    # Portal anchors add a small, intentional dogleg.  A route several times
    # longer than the actual doorway distance means the floor projection is
    # trying to route around stale occupied cells, not through the opening.
    if length > max(config.entry_preflight_max_path_m,
                    config.entry_preflight_max_detour_ratio * direct + 1.0):
        return None
    return {
        "mandatory_portal_waypoints": [[point[0], point[1]] for point in snapped],
        "portal_clearance_m": clearance,
        "preflight_path": combined,
        "preflight_path_length_m": length,
        "portal_resume_inside": resume_inside,
    }


def fallback_entry_proposal(grid: OccupancyGrid2D, door: EstimatedDoorway,
                            current: Point,
                            config: LightweightRoomConfig,
                            clearance_override: Optional[float] = None
                            ) -> Optional[Dict]:
    """Find a conservative staging/entry point when the ideal portal is sparse.

    A projected doorway can be confirmed while its semantic centre or inward
    endpoint is still unknown (common with FAST-LIO drift and partial scans).
    Do not discard the doorway in that case.  Search the observed free ray from
    the corridor side toward the interior, and let A* remain the authority on
    reachability.  This never invents an unknown-space goal.
    """
    anchors = (door.corridor_side, door.center, door.interior_side)
    # Include short interpolation steps so a single known-free cell at the
    # threshold can be used as a staging point on the next map update.
    desired = []
    for start, end in ((door.corridor_side, door.center),
                       (door.center, door.interior_side)):
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            desired.append((start[0] + fraction * (end[0] - start[0]),
                            start[1] + fraction * (end[1] - start[1])))
    desired.extend(anchors)
    # This is a short crossing confirmation, not the eventual room-viewpoint.
    # Keep the regular body clearance for G1 and all other room paths, while
    # permitting a narrow but observed-free doorway cell here. SCAN-lite still
    # validates the resulting A* path before execution.
    clearance = min(portal_clearance(grid, door, config),
                    config.entry_staging_clearance)
    if clearance_override is not None:
        clearance = min(clearance, max(0.12, float(clearance_override)))
    candidates = []
    for point in desired:
        if _state(grid, point) != 0:
            point = _snap_safe_point(grid, point, clearance, door, 0.55,
                                     search_radius=0.65)
        if point is None or _state(grid, point) != 0:
            continue
        if door.depth(point) < config.minimum_crossing_depth:
            continue
        if _clearance(grid, point) < clearance:
            continue
        result = astar_safe_path(grid, current, point, clearance, 0.15,
                                 allow_blocked_start=True)
        if not result.get("success"):
            continue
        path = list(result.get("path", []))
        length = sum(_distance(path[i - 1], path[i])
                     for i in range(1, len(path)))
        direct = _distance(current, point)
        if length > max(config.entry_preflight_max_path_m,
                        config.entry_preflight_max_detour_ratio * direct + 1.0):
            continue
        # Prefer the deepest observed-free point, with distance as tie-breaker.
        candidates.append((door.depth(point), -length, point, length, result))
    if not candidates:
        return None
    _, _, point, length, path_result = max(
        candidates, key=lambda item: (item[0], item[1]))
    return {
        "position": point, "role": "ENTRY_STAGING", "score": 0.0,
        "depth": door.depth(point), "lateral": door.lateral(point),
        "clearance": _clearance(grid, point), "unknown_neighbors": 0,
        "predicted_local_gain_m2": 0.0,
        "preflight_path_length_m": length,
        "fallback_candidate": True,
        "entry_staging_fallback": True,
        "portal_preflight": None,
        "entry_staging_path_result": path_result,
    }


def prepare_exit_path(grid: OccupancyGrid2D, door: EstimatedDoorway,
                      current: Point,
                      config: LightweightRoomConfig) -> Optional[dict]:
    clearance = portal_clearance(grid, door, config)
    normal = door.normal
    # First move to the door-normal centreline while still safely inside the
    # room.  The following short normal segment makes the executor turn toward
    # the doorway before its footprint reaches the jambs.  This implements an
    # online, geometry-derived sequence:
    #   parallel-to-door alignment -> face/approach door -> centre -> corridor
    # and uses no room or doorway truth coordinates.
    alignment_depth = max(0.75, min(1.20, 0.65 * config.entry_depth))
    approach_depth = max(0.40, min(0.60, 0.40 * config.entry_depth))
    alignment = (door.center[0] + alignment_depth * normal[0],
                 door.center[1] + alignment_depth * normal[1])
    approach = (door.center[0] + approach_depth * normal[0],
                door.center[1] + approach_depth * normal[1])
    exit_point, _ = canonical_doorway_anchors(
        door.center, door.normal_direction, config)
    # Keep the short pre-door approach anchor even when the overall exit is
    # executed continuously.  It gives the gait a stable door-normal posture
    # before the footprint reaches the jambs; removing it caused subsequent
    # far-door entry preflights to become unstable in opt34.
    desired = (alignment, approach, door.center, exit_point)
    tolerances = (0.30, 0.22, 0.20, 0.35)
    snapped = [_snap_safe_point(grid, point, clearance, door, tolerance)
               for point, tolerance in zip(desired, tolerances)]
    if any(point is None for point in snapped):
        return None
    start, combined = current, []
    for index, target in enumerate(snapped):
        result = astar_safe_path(
            grid, start, target, clearance, 0.15,
            allow_blocked_start=(index == 0),
            maximum_expansions=config.exit_preflight_maximum_expansions)
        if not result.get("success"):
            return None
        path = list(result.get("path", []))
        combined.extend(path if not combined else path[1:])
        start = target
    return {
        "mandatory_portal_waypoints": [[point[0], point[1]] for point in snapped],
        "portal_clearance_m": clearance,
        "preflight_path": combined,
        "door_centerline_alignment": [snapped[0][0], snapped[0][1]],
        "door_normal_approach": [snapped[1][0], snapped[1][1]],
    }


def _unknown_neighbors(grid: OccupancyGrid2D, point: Point, radius: float = 0.9) -> int:
    cell = grid.world_to_cell(point)
    if cell is None:
        return 0
    bound = max(1, int(math.ceil(radius / grid.resolution)))
    count = 0
    for dy in range(-bound, bound + 1):
        for dx in range(-bound, bound + 1):
            if dx * dx + dy * dy > bound * bound:
                continue
            x, y = cell[0] + dx, cell[1] + dy
            if 0 <= x < grid.width and 0 <= y < grid.height:
                count += int(grid.data[y, x] < 0)
    return count


def _segment_has_occupied(grid: OccupancyGrid2D, start: Point, end: Point) -> bool:
    """Whether an observed obstacle occludes the interior of one sight line."""
    length = _distance(start, end)
    if length < 0.6:
        return False
    step = max(0.04, 0.40 * grid.resolution)
    count = max(2, int(math.ceil(length / step)))
    # Ignore endpoint neighborhoods: only an obstacle between the observation
    # point and candidate makes the candidate an occlusion-revealing goal.
    for index in range(2, count - 1):
        ratio = float(index) / float(count)
        point = (start[0] + ratio * (end[0] - start[0]),
                 start[1] + ratio * (end[1] - start[1]))
        if _state(grid, point) > 0:
            return True
    return False


def _first_occupied_distance(grid: OccupancyGrid2D, start: Point,
                             end: Point) -> Optional[float]:
    length = _distance(start, end)
    if length < 0.6:
        return None
    step = max(0.04, 0.40 * grid.resolution)
    count = max(2, int(math.ceil(length / step)))
    for index in range(1, count):
        ratio = float(index) / float(count)
        point = (start[0] + ratio * (end[0] - start[0]),
                 start[1] + ratio * (end[1] - start[1]))
        if _state(grid, point) > 0:
            return ratio * length
    return None


def select_room_goal(grid: OccupancyGrid2D, door: EstimatedDoorway,
                     current: Point, role: str,
                     completed: Sequence[Point],
                     config: LightweightRoomConfig,
                     fallback: bool = False,
                     boundary_probe: bool = False) -> Optional[Dict]:
    """Select the entry transit, room-centre view, or one mandatory side view."""
    # Side views use the stable first interior observation as their reference;
    # an occlusion-depth view is explicitly relative to the latest scan pose.
    anchor = (completed[-1] if role == "G2" and completed else
              completed[0] if completed else door.interior_side)
    left_extent, left_extent_samples = _parallel_side_extent(
        grid, door, anchor, 1.0, config.room_side_probe_range)
    right_extent, right_extent_samples = _parallel_side_extent(
        grid, door, anchor, -1.0, config.room_side_probe_range)
    raw_left_target = max(0.70, left_extent - config.near_wall_margin)
    raw_right_target = max(0.70, right_extent - config.near_wall_margin)
    # A normal coverage view remains compact.  If online occupancy still has
    # no evidence for this side wall, move farther into that half of the room
    # instead of trusting an ideal long-range ray from the centre.
    side_fraction = (0.72 if boundary_probe else
                     0.36 if fallback else 0.45)
    minimum_side_lateral = min(0.85, max(0.55,
                                         config.maximum_side_lateral))
    left_target = min(
        config.maximum_side_lateral,
        max(minimum_side_lateral, side_fraction * raw_left_target))
    right_target = -min(
        config.maximum_side_lateral,
        max(minimum_side_lateral, side_fraction * raw_right_target))
    role_targets = {
        "ENTRY": (max(config.entry_depth, config.confirmation_depth + 0.20), 0.0),
        "G1": (config.entry_depth, 0.0),
        # Keep the obstacle-shadow/depth view inside the normal room budget.
        # The former +1.0 bias routinely pushed G2 to 4.7--5 m and made the
        # return path dominate a short exploration run.
        "G2": (min(config.room_depth_probe_range - config.near_wall_margin,
                    max(config.maximum_center_depth,
                        door.depth(anchor) + 1.2)), 0.0),
        "G3": (max(config.side_depth, door.depth(anchor)), left_target),
        "G4": (max(config.side_depth, door.depth(anchor)), right_target),
    }
    if role not in role_targets:
        raise ValueError("unknown room goal role " + str(role))
    normal, tangent = door.normal, door.tangent
    desired_depth, desired_lateral = role_targets[role]
    if boundary_probe and role in ("G3", "G4"):
        # A diagonal deep/side view exposes a missing corner better than a
        # second pose close to the doorway.  Candidate endpoints remain known
        # free and must still pass clearance and A* below.
        desired_depth = min(
            config.room_depth_probe_range - config.near_wall_margin,
            max(desired_depth, config.maximum_center_depth + 0.80))
    if role in ("G2", "G3", "G4") and not completed:
        return None
    # Narrow 0.75 m doors can have only one safe centre column.  Full-cell
    # ENTRY sampling prevents a 0.25 m semantic-grid stride from skipping it.
    stride = (1 if role == "ENTRY" else
              max(1, int(round(0.25 / grid.resolution))))
    candidates = []
    for y in range(0, grid.height, stride):
        for x in range(0, grid.width, stride):
            if int(grid.data[y, x]) != 0:
                continue
            point = grid.cell_to_world((x, y))
            depth, lateral = door.depth(point), door.lateral(point)
            if (depth < 1.15 or depth > config.room_depth_probe_range or
                    abs(lateral) > config.room_side_probe_range + 0.25):
                continue
            if role == "ENTRY":
                # ENTRY is a portal transit, not an information-gain view.
                # Keep its endpoint on the door normal so clearance scoring
                # cannot turn a straight commit into a diagonal wall crossing.
                entry_lateral_limit = min(
                    0.30, max(0.12, 0.5 * door.width - config.goal_clearance))
                if abs(lateral) > entry_lateral_limit:
                    continue
            if role == "G3" and lateral < (0.55 if fallback else 0.70):
                continue
            if role == "G4" and lateral > (-0.55 if fallback else -0.70):
                continue
            occlusion_revealing = False
            if role == "G2":
                # A second depth point is justified only when an observed
                # obstacle lies between the latest scan pose and a deeper
                # reachable free cell.  A* below must find a route around it.
                if depth < door.depth(anchor) + 0.70:
                    continue
                occlusion_revealing = _segment_has_occupied(grid, anchor, point)
                if not occlusion_revealing:
                    continue
            clearance = _clearance(grid, point)
            if clearance < config.goal_clearance:
                continue
            separation = min((_distance(point, old) for old in completed),
                             default=math.inf)
            if completed and separation < config.minimum_goal_separation:
                continue
            unknown = _unknown_neighbors(grid, point)
            if role == "ENTRY":
                score = (-3.0 * abs(depth - desired_depth) - 0.8 * abs(lateral)
                         + 0.8 * min(clearance, 1.2))
            elif role == "G2":
                score = (-1.6 * abs(depth - desired_depth)
                         - 1.2 * abs(lateral)
                         + 0.07 * unknown
                         + 1.8
                         + 0.5 * min(clearance, 1.2))
            else:
                score = (-1.8 * abs(depth - desired_depth)
                         - 2.0 * abs(lateral - desired_lateral)
                         + 0.05 * unknown
                         + 0.5 * min(clearance, 1.2))
            candidates.append((score, point, depth, lateral, clearance, unknown))

    if role == "G1" and candidates:
        # Estimate the room centre from robust door-local extents.  The door
        # half-plane excludes the corridor; percentiles suppress narrow raster
        # cracks and isolated free returns.  Candidate scoring below then
        # projects an occupied geometric centre to the nearest high-clearance
        # reachable free cell.
        depths = sorted(item[2] for item in candidates)
        laterals = sorted(item[3] for item in candidates)
        low = max(0, int(0.08 * (len(depths) - 1)))
        high = min(len(depths) - 1, int(0.92 * (len(depths) - 1)))
        # Move only far enough to expose the room boundary.  Coverage, not a
        # fixed percentage of the room depth, decides whether side views are
        # needed after this observation.
        desired_depth = min(
            config.maximum_center_depth,
            depths[low] + 0.45 * (depths[high] - depths[low]))
        if boundary_probe:
            desired_depth = min(
                config.room_depth_probe_range - config.near_wall_margin,
                max(desired_depth, config.maximum_center_depth + 0.80))
        desired_lateral = 0.5 * (laterals[low] + laterals[high])
        rescored = []
        for _, point, depth, lateral, clearance, unknown in candidates:
            center_error = math.hypot(depth - desired_depth,
                                      lateral - desired_lateral)
            score = (-2.6 * center_error + 1.1 * min(clearance, 1.5)
                     + 0.01 * unknown)
            rescored.append((score, point, depth, lateral, clearance, unknown))
        candidates = rescored

    candidate_limit = (config.entry_candidate_astar_limit
                       if role == "ENTRY" else 40)
    for score, point, depth, lateral, clearance, unknown in sorted(
            candidates, reverse=True)[:candidate_limit]:
        planning_clearance = (portal_clearance(grid, door, config)
                              if role == "ENTRY" else config.goal_clearance)
        path = astar_safe_path(grid, current, point, planning_clearance, 0.15,
                               allow_blocked_start=True)
        if path.get("success"):
            path_points = list(path.get("path", []))
            path_length = sum(_distance(path_points[index - 1], path_points[index])
                              for index in range(1, len(path_points)))
            return {
                "position": point, "role": role, "score": score,
                "depth": depth, "lateral": lateral, "clearance": clearance,
                "unknown_neighbors": unknown,
                "predicted_local_gain_m2": (unknown * grid.resolution *
                                             grid.resolution),
                "preflight_path_length_m": path_length,
                "occlusion_revealing": bool(role == "G2"),
                "occlusion_anchor": [anchor[0], anchor[1]],
                "estimated_room_width": left_extent + right_extent,
                "left_wall_distance": left_extent,
                "right_wall_distance": right_extent,
                "left_wall_distance_samples": left_extent_samples,
                "right_wall_distance_samples": right_extent_samples,
                "fallback_candidate": bool(fallback),
                "boundary_probe": bool(boundary_probe),
                "side_extent_fraction": (side_fraction
                                         if role in ("G3", "G4") else None),
                "center_target": ([desired_depth, desired_lateral]
                                  if role == "G1" else None),
                "center_displacement_m": (math.hypot(
                    depth - desired_depth, lateral - desired_lateral)
                    if role == "G1" else None),
                "center_adjustment_reason": (
                    "geometric_center_blocked_or_unreachable"
                    if role == "G1" and math.hypot(
                        depth - desired_depth, lateral - desired_lateral) >
                    1.5 * grid.resolution else
                    "geometric_center_reachable" if role == "G1" else None),
                "depth_fraction_target": (0.45 if role == "G1" else None),
            }
    return None


def select_adaptive_room_goal(grid: OccupancyGrid2D,
                              door: EstimatedDoorway,
                              current: Point,
                              completed: Sequence[Point],
                              attempted_roles: Sequence[str],
                              coverage: Dict,
                              config: LightweightRoomConfig) -> Optional[Dict]:
    """Choose one next-best view from online LiDAR coverage debt.

    No room dimensions or layout labels enter this decision.  G2 is enabled
    only by a measured obstacle shadow and selects reachable observed-free
    space behind that obstacle.  G3/G4 compete by marginal frontier gain, so
    an already visible side is never visited just to satisfy a fixed list.
    """
    attempted = {str(role) for role in attempted_roles}
    observation_anchors = list(completed) or [current]
    shadow_m2 = float(coverage.get("largest_shadow_m2", 0.0) or 0.0)
    # This debt is set only by the online RGB-D coverage mapper after the
    # first doorway-anchored observation.  It carries no target identity or
    # layout information: it merely permits one LiDAR-safe side/depth view
    # when the camera has not actually covered its interior fan.
    visual_debt = bool(coverage.get("visual_deepening_requested", False))
    boundary_complete = bool(coverage.get("boundary_complete", False))
    candidates = []

    roles = []
    if ("G2" not in attempted and
            shadow_m2 >= config.adaptive_occlusion_shadow_trigger_m2):
        roles.append("G2")
    if "G1" not in attempted:
        roles.append("G1")
    roles.extend(role for role in ("G3", "G4") if role not in attempted)

    for role in roles:
        role_completed = (observation_anchors if role != "G1" else completed)
        boundary_probe = bool(
            (role == "G1" and not coverage.get("far_wall_seen", False)) or
            (role == "G3" and not coverage.get("left_wall_seen", False)) or
            (role == "G4" and not coverage.get("right_wall_seen", False)))
        proposal = select_room_goal(
            grid, door, current, role, role_completed, config,
            boundary_probe=boundary_probe)
        if proposal is None and role in ("G3", "G4"):
            proposal = select_room_goal(
                grid, door, current, role, role_completed, config,
                fallback=True, boundary_probe=boundary_probe)
        if proposal is None:
            continue
        local_gain = float(proposal.get("predicted_local_gain_m2", 0.0))
        travel = float(proposal.get("preflight_path_length_m", 0.0))
        occlusion_gain = (min(shadow_m2, 2.5)
                          if role == "G2" else 0.0)
        # The first depth view exposes room boundaries even if the current
        # frontier band is still small.  Supplementary views must pay for
        # themselves through new cells or removal of a measured shadow.
        # A first depth view is structurally useful even before a frontier
        # band is measurable.  Do not grant the same synthetic gain to G3/G4:
        # in a nearly covered room that made zero-gain side poses win solely
        # because a wall direction was not yet classified, adding distance
        # without revealing any cells.  Supplementary breadth views now need
        # measured local unknown/occlusion debt of their own.
        structural_gain = 1.0 if role == "G1" and not completed else 0.0
        visual_gain = (.20 if visual_debt and role in ("G3", "G4") else 0.0)
        marginal_gain = local_gain + occlusion_gain + structural_gain + visual_gain
        if (role != "G1" and
                marginal_gain < config.adaptive_minimum_gain_m2):
            continue
        utility = marginal_gain - config.adaptive_path_cost_weight * travel
        if role == "G2":
            utility += 0.45  # prefer clearing a real occlusion over side travel
        proposal.update({
            "adaptive_utility": utility,
            "adaptive_marginal_gain_m2": marginal_gain,
            "adaptive_local_unknown_gain_m2": local_gain,
            "adaptive_occlusion_gain_m2": occlusion_gain,
            "adaptive_visual_gain_m2": visual_gain,
            "adaptive_structural_gain": structural_gain,
            "adaptive_selection": True,
        })
        candidates.append(proposal)

    if not candidates:
        return None
    best = max(candidates, key=lambda item: (
        float(item["adaptive_utility"]),
        float(item["adaptive_marginal_gain_m2"]),
        -float(item.get("preflight_path_length_m", 0.0))))
    # When a large measured shadow has a safe route to its far side, prefer
    # that depth view if its utility is close to the nominal winner.  This is
    # the online analogue of "look behind the object", with a strict detour
    # bound so a distant G2 cannot override a much cheaper side observation.
    depth_views = [item for item in candidates if item["role"] == "G2"]
    if depth_views:
        depth_view = max(depth_views, key=lambda item: item["adaptive_utility"])
        if (float(depth_view["adaptive_utility"]) >=
                float(best["adaptive_utility"]) - 0.35):
            return depth_view
    return best


def _ray_visible_unknown_cells(grid: OccupancyGrid2D,
                               door: EstimatedDoorway,
                               viewpoint: Point,
                               config: LightweightRoomConfig) -> set:
    """Unknown floor cells conservatively visible from one safe viewpoint.

    The endpoint must be known free (enforced by ``select_room_goal``).  Rays
    stop at the first non-free cell.  In particular, an unknown endpoint is
    visible only through already observed free space; unknown cells behind it
    are not optimistically counted as scanned.  A
    two-cell sampling stride bounds runtime without changing the set-cover
    ordering at the 0.15 m map resolution.
    """
    origin = grid.world_to_cell(viewpoint)
    if origin is None:
        return set()

    def line_cells(start, end):
        x0, y0 = start
        x1, y1 = end
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        error = dx - dy
        while True:
            yield x0, y0
            if x0 == x1 and y0 == y1:
                break
            twice = 2 * error
            if twice > -dy:
                error -= dy
                x0 += sx
            if twice < dx:
                error += dx
                y0 += sy

    normal, tangent = door.normal, door.tangent
    range_sq = config.lidar_effective_range ** 2
    visible = set()
    stride = 2
    for y in range(0, grid.height, stride):
        for x in range(0, grid.width, stride):
            if int(grid.data[y, x]) >= 0:
                continue
            point = grid.cell_to_world((x, y))
            relative = (point[0] - door.center[0],
                        point[1] - door.center[1])
            depth = _dot(relative, normal)
            lateral = _dot(relative, tangent)
            if (depth < 0.0 or depth > config.room_depth_probe_range or
                    abs(lateral) > config.room_side_probe_range or
                    _distance(viewpoint, point) ** 2 > range_sq):
                continue
            blocked = False
            for cx, cy in line_cells(origin, (x, y)):
                if (cx, cy) in (origin, (x, y)):
                    continue
                if not (0 <= cx < grid.width and 0 <= cy < grid.height):
                    blocked = True
                    break
                if int(grid.data[cy, cx]) != 0:
                    blocked = True
                    break
            if not blocked:
                visible.add((x, y))
    return visible


def _coverage_cell_components(cells: Sequence[Tuple[int, int]]) -> List[set]:
    """Connected components for the two-cell-stride visibility lattice."""
    remaining = set(cells)
    components = []
    while remaining:
        seed = remaining.pop()
        component = {seed}
        queue = deque([seed])
        while queue:
            x, y = queue.popleft()
            for dx in (-2, 0, 2):
                for dy in (-2, 0, 2):
                    if dx == 0 and dy == 0:
                        continue
                    neighbour = (x + dx, y + dy)
                    if neighbour in remaining:
                        remaining.remove(neighbour)
                        component.add(neighbour)
                        queue.append(neighbour)
        components.append(component)
    return sorted(components, key=len, reverse=True)


def _generic_coverage_candidates(grid: OccupancyGrid2D,
                                 door: EstimatedDoorway,
                                 current: Point,
                                 completed: Sequence[Point],
                                 attempted_roles: Sequence[str],
                                 config: LightweightRoomConfig) -> List[Dict]:
    """Sample reachable free-side viewpoints around unresolved map debt.

    G1/G2/G3/G4 are useful labels, but they are not a complete basis for an
    irregular room.  In particular, a large unknown component can remain on
    the opposite side of furniture even when all fixed role templates fail.
    This sampler is layout-free: it uses only free cells adjacent to unknown,
    A* reachability and the same occlusion-aware ray estimator.
    """
    attempted = {str(role) for role in attempted_roles}
    available_roles = [role for role in ("G1", "G2", "G3", "G4")
                       if role not in attempted]
    if not available_roles:
        return []
    stride = max(2, int(round(0.45 / grid.resolution)))
    preliminary = []
    for y in range(0, grid.height, stride):
        for x in range(0, grid.width, stride):
            if int(grid.data[y, x]) != 0:
                continue
            point = grid.cell_to_world((x, y))
            depth, lateral = door.depth(point), door.lateral(point)
            if (depth < 1.15 or depth > config.room_depth_probe_range or
                    abs(lateral) > config.room_side_probe_range or
                    _clearance(grid, point) < config.goal_clearance or
                    min((_distance(point, old) for old in completed),
                        default=math.inf) < config.minimum_goal_separation):
                continue
            nearby_unknown = _unknown_neighbors(grid, point, radius=1.5)
            if nearby_unknown <= 0:
                continue
            direct = _distance(current, point)
            preliminary.append((nearby_unknown / (1.0 + direct), point,
                                depth, lateral, nearby_unknown))
    # Expensive global ray casting is reserved for the strongest spatially
    # distributed frontier-side samples.
    shortlisted = []
    for item in sorted(preliminary, reverse=True):
        point = item[1]
        if any(_distance(point, old[1]) < 0.75 for old in shortlisted):
            continue
        shortlisted.append(item)
        if len(shortlisted) >= 16:
            break
    proposals = []
    for _, point, depth, lateral, nearby_unknown in shortlisted:
        path = astar_safe_path(
            grid, current, point, config.goal_clearance, 0.15,
            allow_blocked_start=True)
        if not path.get("success"):
            continue
        path_points = list(path.get("path", []))
        path_length = sum(_distance(path_points[i - 1], path_points[i])
                          for i in range(1, len(path_points)))
        cells = _ray_visible_unknown_cells(grid, door, point, config)
        if not cells:
            continue
        preferred = ("G3" if lateral >= 0.0 else "G4")
        role = (preferred if preferred in available_roles else
                "G2" if "G2" in available_roles else
                available_roles[0])
        proposals.append({
            "position": point, "role": role,
            "score": len(cells) * stride_area(grid) -
                     config.adaptive_path_cost_weight * path_length,
            "depth": depth, "lateral": lateral,
            "clearance": _clearance(grid, point),
            "unknown_neighbors": nearby_unknown,
            "predicted_local_gain_m2": len(cells) * stride_area(grid),
            "preflight_path_length_m": path_length,
            "occlusion_revealing": False,
            "fallback_candidate": True,
            "coverage_cells": cells,
            "generic_coverage_debt_candidate": True,
        })
    proposals.sort(key=lambda item: (
        -len(item["coverage_cells"]), item["preflight_path_length_m"]))
    return proposals[:8]


def plan_minimum_coverage_route(grid: OccupancyGrid2D,
                                door: EstimatedDoorway,
                                current: Point,
                                completed: Sequence[Point],
                                attempted_roles: Sequence[str],
                                coverage: Dict,
                                return_anchor: Point,
                                config: LightweightRoomConfig) -> Optional[Dict]:
    """Select the fewest room views, then the shortest safe visit order.

    At most four semantic candidates are considered and no more than three
    are selected, so all subsets and permutations can be audited exactly.
    This is a local set-cover/TSP problem over online OccupancyGrid evidence;
    it does not use room truth or fixed coordinates.
    """
    attempted = {str(role) for role in attempted_roles}
    anchors = list(completed) or [current]
    inferred_unknown = _inferred_room_unknown_grid_cells(
        grid, door, config)
    shadow_m2 = float(coverage.get("largest_shadow_m2", 0.0) or 0.0)
    roles = ["G1", "G3", "G4"]
    if shadow_m2 >= config.adaptive_occlusion_shadow_trigger_m2:
        roles.insert(1, "G2")
    candidates = []
    for role in roles:
        if role in attempted:
            continue
        boundary_probe = bool(
            (role == "G1" and not coverage.get("far_wall_seen", False)) or
            (role == "G3" and not coverage.get("left_wall_seen", False)) or
            (role == "G4" and not coverage.get("right_wall_seen", False)))
        proposal = select_room_goal(
            grid, door, current, role, anchors, config,
            boundary_probe=boundary_probe)
        if proposal is None and role in ("G3", "G4"):
            proposal = select_room_goal(
                grid, door, current, role, anchors, config, fallback=True,
                boundary_probe=boundary_probe)
        if proposal is None:
            continue
        cells = _ray_visible_unknown_cells(
            grid, door, proposal["position"], config)
        if inferred_unknown:
            cells.intersection_update(inferred_unknown)
        proposal = dict(proposal)
        proposal["coverage_cells"] = cells
        proposal["predicted_visible_unknown_m2"] = (
            len(cells) * stride_area(grid))
        candidates.append(proposal)
    semantic_union = set().union(
        *(item["coverage_cells"] for item in candidates)) if candidates else set()
    unresolved_m2 = float(coverage.get("largest_unknown_m2", 0.0) or 0.0)
    if (not semantic_union or
            not bool(coverage.get("boundary_complete", False)) or
            unresolved_m2 > config.coverage_maximum_unknown_m2):
        generic = _generic_coverage_candidates(
            grid, door, current, completed, attempted, config)
        for proposal in generic:
            if inferred_unknown:
                proposal["coverage_cells"].intersection_update(
                    inferred_unknown)
                if not proposal["coverage_cells"]:
                    continue
            if any(_distance(proposal["position"], old["position"]) < 0.45
                   for old in candidates):
                continue
            candidates.append(proposal)
    if not candidates:
        return None

    candidate_union = set().union(
        *(item["coverage_cells"] for item in candidates))
    # When all three structural wall bands are visible, use the complete
    # inferred-room unknown mask as the set-cover denominator.  Falling back
    # to candidate_union is reserved for the early boundary-incomplete phase.
    # This prevents one candidate from reporting 100% merely because no other
    # candidate currently sees a still-unmapped corner.
    union = set(inferred_unknown) if inferred_unknown else candidate_union
    # If rays through the doorway have already cleared the whole observed
    # room, retain one central depth pose to physically confirm entry.  Extra
    # side points are never added without measurable coverage debt.
    if not union:
        if completed:
            return None
        central = next((item for item in candidates if item["role"] == "G1"),
                       min(candidates, key=lambda item:
                           item.get("preflight_path_length_m", math.inf)))
        central["adaptive_marginal_gain_m2"] = 1.0 if not completed else 0.0
        central["adaptive_utility"] = (
            central["adaptive_marginal_gain_m2"] -
            config.adaptive_path_cost_weight *
            central.get("preflight_path_length_m", 0.0))
        return {"ordered": [central], "coverable_cell_count": 0,
                "covered_cell_count": 0, "coverage_ratio": 1.0,
                "residual_m2": 0.0,
                "route_length_m": central.get("preflight_path_length_m", 0.0),
                "selection_reason": "physical_entry_confirmation"}

    cell_area = stride_area(grid)
    initial_components = _coverage_cell_components(union)
    largest_initial_component = (
        initial_components[0] if initial_components else set())
    for item in candidates:
        item["largest_gap_overlap_m2"] = (
            len(item["coverage_cells"] & largest_initial_component) *
            cell_area)
    required = min(
        len(union),
        max(1, int(math.ceil(config.adaptive_set_cover_ratio * len(union)))),
    )
    max_views = min(config.adaptive_maximum_viewpoints, len(candidates))
    # Missing structural directions are hard coverage debts.  Require a
    # candidate in each available missing direction without exceeding the
    # existing three-view room budget.
    missing_boundary_roles = set()
    if not coverage.get("far_wall_seen", False):
        missing_boundary_roles.add("G1")
    if not coverage.get("left_wall_seen", False):
        missing_boundary_roles.add("G3")
    if not coverage.get("right_wall_seen", False):
        missing_boundary_roles.add("G4")
    candidate_roles = {str(item["role"]) for item in candidates}
    # G2 (a verified view behind an observed obstacle) is also a valid deep
    # observation for missing far-boundary evidence.  Do not force G1 and
    # crowd that more useful depth view out of the three-point budget.
    required_boundary_roles = (
        (missing_boundary_roles - {"G1"}) & candidate_roles)
    far_boundary_required = bool(
        "G1" in missing_boundary_roles and
        candidate_roles.intersection({"G1", "G2"}))

    distance_cache = {}

    def safe_distance(start: Point, end: Point) -> Optional[float]:
        key = (round(float(start[0]), 3), round(float(start[1]), 3),
               round(float(end[0]), 3), round(float(end[1]), 3))
        if key in distance_cache:
            return distance_cache[key]
        result = astar_safe_path(
            grid, start, end, config.goal_clearance, 0.15,
            allow_blocked_start=True)
        if not result.get("success"):
            distance_cache[key] = None
            return None
        path = list(result.get("path", []))
        distance_cache[key] = sum(_distance(path[i - 1], path[i])
                                  for i in range(1, len(path)))
        return distance_cache[key]

    best = None
    best_effort = []
    for count in range(1, max_views + 1):
        feasible_at_count = []
        for subset in itertools.combinations(candidates, count):
            subset_roles = {item["role"] for item in subset}
            if len(subset_roles) != count:
                continue
            if not required_boundary_roles.issubset(subset_roles):
                continue
            if (far_boundary_required and
                    not subset_roles.intersection({"G1", "G2"})):
                continue
            covered = set().union(*(item["coverage_cells"] for item in subset))
            residual_m2 = (len(union) - len(covered)) * cell_area
            residual_components = _coverage_cell_components(union - covered)
            largest_residual_m2 = (
                len(residual_components[0]) * cell_area
                if residual_components else 0.0)
            for order in itertools.permutations(subset):
                route_length, start, valid = 0.0, current, True
                for item in order:
                    segment = safe_distance(start, item["position"])
                    if segment is None:
                        valid = False
                        break
                    route_length += segment
                    start = item["position"]
                if not valid:
                    continue
                return_distance = safe_distance(start, return_anchor)
                if return_distance is None:
                    continue
                route_cost = (route_length +
                              config.adaptive_route_return_weight *
                              return_distance)
                best_effort.append((
                    largest_residual_m2, residual_m2, route_cost,
                    -len(covered), order, route_length, return_distance,
                    covered))
                # A high global ratio must not hide one large unobserved
                # corner.  Both the ratio and the largest connected residual
                # component are hard constraints; scattered sub-threshold
                # cells are allowed without adding a wasteful viewpoint.
                if (len(covered) < required or
                        largest_residual_m2 >
                        config.adaptive_set_cover_residual_m2):
                    continue
                feasible_at_count.append((route_cost, -len(covered), order,
                                          route_length, return_distance,
                                          covered, residual_m2,
                                          largest_residual_m2))
        if feasible_at_count:
            best = min(feasible_at_count, key=lambda item: (item[0], item[1]))
            break
    if best is None:
        if not best_effort:
            return None
        # No <=3-view set can satisfy the hard residual bound.  Balance the
        # remaining largest hole against the complete route (including the
        # return anchor).  Previously this branch minimized coverage alone,
        # so a low-gain, long G2 excursion could consume most of the mission
        # even when a shorter view covered nearly as much unknown space.
        path_weight = max(0.0, float(config.adaptive_path_cost_weight))
        effort = min(best_effort, key=lambda item: (
            item[0] + path_weight * item[2],
            item[1] + path_weight * item[2],
            item[2], item[3]))
        (largest_residual_m2, residual_m2, _, _, order, route_length,
         return_distance, covered) = effort
        selection_reason = "minimum_largest_coverage_gap_best_effort"
    else:
        (_, _, order, route_length, return_distance, covered, residual_m2,
         largest_residual_m2) = best
        selection_reason = "minimum_viewpoint_set_with_gap_constraint"
    ordered = []
    for item in order:
        proposal = dict(item)
        proposal["adaptive_marginal_gain_m2"] = (
            len(item["coverage_cells"]) * cell_area)
        proposal["adaptive_utility"] = (
            proposal["adaptive_marginal_gain_m2"] -
            config.adaptive_path_cost_weight *
            proposal.get("preflight_path_length_m", 0.0))
        proposal["largest_gap_overlap_m2"] = float(
            item.get("largest_gap_overlap_m2", 0.0))
        ordered.append(proposal)
    return {
        "ordered": ordered,
        "coverable_cell_count": len(union),
        "covered_cell_count": len(covered),
        "coverage_ratio": len(covered) / float(max(1, len(union))),
        "residual_m2": residual_m2,
        "largest_residual_m2": largest_residual_m2,
        "largest_initial_gap_m2":
            len(largest_initial_component) * cell_area,
        "route_length_m": route_length,
        "return_distance_m": return_distance,
        "selection_reason": selection_reason,
        "boundary_complete_before": bool(
            coverage.get("boundary_complete", False)),
        "missing_boundary_roles": sorted(missing_boundary_roles),
        "required_boundary_roles": sorted(required_boundary_roles),
        "far_boundary_required": far_boundary_required,
    }


def stride_area(grid: OccupancyGrid2D) -> float:
    """Area represented by one cell sampled at the 2x2 ray-cast stride."""
    return 4.0 * grid.resolution * grid.resolution


def _room_visibility_from_occupancy(grid: OccupancyGrid2D,
                                    door: EstimatedDoorway,
                                    config: LightweightRoomConfig
                                    ) -> RoomVisibilityGrid:
    """Build the shared observed/unknown room mask used by score and stop."""
    visibility = RoomVisibilityGrid(
        door.center[0], door.center[1], door.normal_direction,
        resolution=grid.resolution,
        inflation_radius=config.goal_clearance,
        max_depth=config.room_depth_probe_range,
        half_width=config.room_side_probe_range,
        sensor_range=config.lidar_effective_range)
    normal, tangent = door.normal, door.tangent
    corners = []
    for depth in (-0.25, config.room_depth_probe_range):
        for lateral in (-config.room_side_probe_range,
                        config.room_side_probe_range):
            corners.append((
                door.center[0] + depth * normal[0] + lateral * tangent[0],
                door.center[1] + depth * normal[1] + lateral * tangent[1]))
    raw_x = [int(math.floor((point[0] - grid.origin_x) / grid.resolution))
             for point in corners]
    raw_y = [int(math.floor((point[1] - grid.origin_y) / grid.resolution))
             for point in corners]
    x_min = max(0, min(raw_x) - 1)
    x_max = min(grid.width - 1, max(raw_x) + 1)
    y_min = max(0, min(raw_y) - 1)
    y_max = min(grid.height - 1, max(raw_y) + 1)
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            value = int(grid.data[y, x])
            if value < 0:
                continue
            world = grid.cell_to_world((x, y))
            if not visibility.inside_room(*world):
                continue
            key = visibility.cell_key(*world)
            if value == 0:
                visibility.free.add(key)
            else:
                visibility.occupied.add(key)
    visibility.free.difference_update(visibility.occupied)
    return visibility


def _inferred_room_unknown_grid_cells(grid: OccupancyGrid2D,
                                      door: EstimatedDoorway,
                                      config: LightweightRoomConfig) -> set:
    """Return unresolved cells inside the three-wall inferred room mask.

    Candidate-ray union is not a valid coverage denominator: one viewpoint
    trivially covers 100% of the cells visible from itself while a different
    corner remains absent from every candidate ray.  This mask is inferred
    from online occupied wall bands and therefore includes those missing
    corners without using layout metadata or truth room bounds.
    """
    visibility = _room_visibility_from_occupancy(grid, door, config)
    cells = set()
    for key in visibility.unknown_cells():
        cell = grid.world_to_cell(visibility.cell_center(key))
        if cell is not None and cell[0] % 2 == 0 and cell[1] % 2 == 0:
            cells.add(cell)
    return cells


def occupancy_room_coverage(grid: OccupancyGrid2D,
                            door: EstimatedDoorway,
                            viewpoints: Sequence[Point],
                            config: LightweightRoomConfig) -> dict:
    """Measure room coverage from the observed, occlusion-aware floor map.

    This intentionally does not stamp a radius around the robot.  Known free
    cells originate from ray-traced LiDAR projection, room limits require
    occupied evidence for the far/left/right walls, and unknown cells inside
    those limits remain coverage debt.
    """
    visibility = _room_visibility_from_occupancy(grid, door, config)
    for point in viewpoints:
        visibility.record_scan_pose(float(point[0]), float(point[1]))
    status = visibility.coverage_status()
    result = asdict(status)
    result["coverage_complete"] = bool(
        status.boundary_complete and
        status.lidar_ratio >= config.coverage_minimum_ratio and
        status.largest_unknown_m2 <= config.coverage_maximum_unknown_m2 and
        status.largest_shadow_m2 <= config.coverage_maximum_shadow_m2)
    result["viewpoint_count"] = len(viewpoints)
    result["configured_lidar_range_m"] = config.lidar_effective_range
    return result




class LightweightRoomScheduler:
    """Execute minimal useful room views followed by a verified room exit."""

    def __init__(self, config: LightweightRoomConfig):
        config.validate()
        self.config = config
        self.detector = DoorwayCrossingDetector(config)
        self.active_door: Optional[EstimatedDoorway] = None
        self.completed_points: List[Point] = []
        self.next_role_index = 0
        self.events: List[dict] = []
        self.selected_goals: List[dict] = []
        self.room_count = 0
        self.exit_attempts = 0
        self.entry_attempts = 0
        self.successful_roles = set()
        self.entry_confirmed = False
        self.active_room_id: Optional[str] = None
        self.completion_mode: Optional[str] = None
        self.prepared_entry: Optional[dict] = None
        self.room_started_at: Optional[float] = None
        self.first_room_started_at: Optional[float] = None
        self.state = "CORRIDOR_SWEEP"
        self.pending_candidates: Dict[Tuple[int, int], dict] = {}
        # Local doorway tracks jitter by a few centimetres every scan.  Keep
        # a short candidate-ID quarantine in addition to the geometric door
        # cooldown so one narrow false aperture cannot monopolize the 60 s
        # room-entry budget. (zip 接入: consider_local_door_candidate A 版依赖)
        self.local_candidate_quarantine: Dict[str, float] = {}
        self.door_cooldowns: Dict[Tuple[int, int], float] = {}
        self.door_cooldown_centers: Dict[Tuple[int, int], Point] = {}
        self.role_attempts: Dict[str, int] = {}
        self.failed_roles = set()
        self.return_completed = False
        self.return_attempts = 0
        self.return_anchor: Optional[Point] = None
        self.entry_portal_waypoints: List[List[float]] = []
        self.exit_fallback_issued = False
        self.exit_blocked = False
        self.coverage_status: Optional[dict] = None
        self.adaptive_route_queue: List[dict] = []
        self.visual_deepening_requested = False
        # LiDAR can classify a compact room as covered from the portal.  That
        # is valid for mapping but not proof that the forward RGB-D camera has
        # ever observed from a safe interior pose.
        self.visual_anchor_completed = False

    def _door_key(self, point: Point) -> Tuple[int, int]:
        scale = max(0.50, self.config.duplicate_door_radius * .5)
        return int(round(point[0] / scale)), int(round(point[1] / scale))

    def _pending_key(self, point: Point) -> Tuple[int, int]:
        """Associate map-jittered detections with the same pending doorway."""
        match_radius = max(0.45, 0.50 * self.config.duplicate_door_radius)
        for key, record in self.pending_candidates.items():
            candidate = record.get("candidate")
            if candidate is not None and _distance(point, candidate.center) <= match_radius:
                return key
        return self._door_key(point)

    def _set_door_cooldown(self, point: Point, until: float) -> Tuple[int, int]:
        key = self._door_key(point)
        self.door_cooldowns[key] = float(until)
        self.door_cooldown_centers[key] = (float(point[0]), float(point[1]))
        return key

    def _door_cooldown_until(self, point: Point) -> float:
        until = -math.inf
        for key, expiry in self.door_cooldowns.items():
            center = self.door_cooldown_centers.get(key)
            if (center is not None and
                    _distance(point, center) <= self.config.duplicate_door_radius):
                until = max(until, float(expiry))
        return until

    def _set_state(self, state: str, now: float, reason: Optional[str] = None) -> None:
        if state == self.state:
            return
        self.events.append({
            "event": "ROOM_STATE_CHANGED", "elapsed_sec": round(float(now), 3),
            "from": self.state, "to": state, "reason": reason,
        })
        self.state = state

    def _room_id(self) -> str:
        return (self.active_room_id or
                "{}{:02d}".format(
                    self.config.room_id_prefix, self.room_count + 1))

    def _confirm_entry(self, point: Point, now: float,
                       source: str) -> None:
        if self.entry_confirmed:
            return
        self.entry_confirmed = True
        if self.first_room_started_at is None:
            self.first_room_started_at = float(now)
        # Use the pose that demonstrably crossed into the room as the return
        # staging point.  A doorway re-estimated after the crossing can place
        # ``interior_side`` in a newly occupied raster cell, which previously
        # caused an unbounded goal_footprint_blocked RETURN loop.
        if point is not None:
            self.return_anchor = (float(point[0]), float(point[1]))
        # Door interception and portal alignment can consume most of a short
        # visit budget before the camera is actually inside. Timed semantic
        # missions may start observation/exit timing at the confirmed
        # crossing; ENTRY keeps its own bounded timeout.
        if self.config.budget_starts_after_entry:
            self.room_started_at = float(now)
        elif self.room_started_at is None:
            self.room_started_at = float(now)
        self.room_count += 1
        self.active_room_id = "{}{:02d}".format(
            self.config.room_id_prefix, self.room_count)
        self.events.append({
            "event": "ROOM_ENTERED", "elapsed_sec": round(float(now), 3),
            "room_id": self.active_room_id,
            "door_id": self.active_door.door_id if self.active_door else None,
            "entry_source": source,
        })

    def _clear_active(self) -> None:
        self.active_door = None
        self.active_room_id = None
        self.completed_points = []
        self.next_role_index = 0
        self.exit_attempts = 0
        self.entry_attempts = 0
        self.successful_roles = set()
        self.entry_confirmed = False
        self.room_started_at = None
        self.completion_mode = None
        self.prepared_entry = None
        self.role_attempts = {}
        self.failed_roles = set()
        self.return_completed = False
        self.return_attempts = 0
        self.return_anchor = None
        self.entry_portal_waypoints = []
        self.exit_fallback_issued = False
        self.exit_blocked = False
        self.coverage_status = None
        self.adaptive_route_queue = []
        self.visual_deepening_requested = False
        self.visual_anchor_completed = False

    def _activate(self, door: EstimatedDoorway, now: float,
                  proactive: bool,
                  crossed_point: Optional[Point] = None,
                  entry_waypoints: Optional[Sequence[Sequence[float]]] = None,
                  reuse_existing: bool = False
                  ) -> EstimatedDoorway:
        # A failed doorway remains in ``detector.doors`` so later scans do
        # not create duplicate semantic landmarks.  Upper-floor recovery may
        # deliberately retry that *same* unvisited landmark after its short
        # cooldown.  Re-registering it here would create a second room record
        # at the same physical portal and corrupt the four-room count.
        if not reuse_existing:
            door.door_id = "estimated_door_{:02d}".format(
                len(self.detector.doors) + 1)
            self.detector.doors.append(door)
        elif not any(item is door for item in self.detector.doors):
            raise ValueError("reused doorway is not registered")
        door.entered_at = float(now)
        door.temporarily_failed = False
        self.active_door = door
        self.completed_points = []
        self.next_role_index = 0
        self.exit_attempts = 0
        self.entry_attempts = 0
        self.successful_roles = set()
        self.entry_confirmed = False
        self.active_room_id = None
        self.room_started_at = float(now)
        self.completion_mode = None
        self.prepared_entry = None
        self.role_attempts = {}
        self.failed_roles = set()
        self.return_completed = False
        self.return_attempts = 0
        self.entry_portal_waypoints = [
            [float(point[0]), float(point[1])] for point in (entry_waypoints or [])]
        self.exit_fallback_issued = False
        self.exit_blocked = False
        self.coverage_status = None
        self.adaptive_route_queue = []
        self.visual_deepening_requested = False
        self.visual_anchor_completed = False
        if not proactive and crossed_point is not None:
            self._confirm_entry(crossed_point, now, "verified_post_crossing")
        self._set_state("DOOR_COMMIT", now,
                        "proactive_path_intercept" if proactive else
                        "crossing_confirmed")
        self.events.append({
            "event": ("DOOR_COMMIT_STARTED" if proactive else
                      "ROOM_ACTIVATED_POST_CROSSING"),
            "elapsed_sec": round(float(now), 3),
            "room_id": self._room_id(),
            "door": door.to_dict(), "proactive": proactive,
        })
        return door

    def request_visual_deepening(self, room_id: str, reason: str) -> bool:
        """Allow bounded G1-breadth views for a real RGB-D coverage gap."""
        if (self.active_door is None or not self.entry_confirmed or
                str(room_id) != self._room_id() or self.visual_deepening_requested):
            return False
        self.visual_deepening_requested = True
        # The two-lateral-view mode uses G1 only as a geometric reference.
        # A route queued from the ENTRY map callback can still contain a
        # physical G1 waypoint; retaining it silently changes the intended
        # ENTRY -> left -> right sequence into a three-point room traverse.
        # Clear only this unexecuted adaptive queue.  The current ENTRY pose
        # remains in ``completed_points`` and is still used by all A* checks.
        if (self.config.visual_two_pose_lateral_enabled and
                str(reason) == "two_lateral_full_camera_coverage"):
            removed = [str(item.get("role"))
                       for item in self.adaptive_route_queue]
            self.adaptive_route_queue = []
        else:
            removed = []
        self.events.append({
            "event": "ROOM_VISUAL_DEEPENING_REQUESTED",
            "room_id": self._room_id(), "door_id": self.active_door.door_id,
            "reason": str(reason),
            "discarded_previsual_roles": removed,
        })
        return True

    def clear_visual_deepening(self, room_id: str, reason: str) -> None:
        if self.active_door is None or str(room_id) != self._room_id():
            return
        if self.visual_deepening_requested:
            self.events.append({
                "event": "ROOM_VISUAL_DEEPENING_CLEARED",
                "room_id": self._room_id(), "door_id": self.active_door.door_id,
                "reason": str(reason),
            })
        self.visual_deepening_requested = False

    def complete_virtual_visual_anchor(self, room_id: str) -> bool:
        """Accept the two lateral full-sweep views as the G1 visual anchor.

        In the two-lateral profile G1 is intentionally a geometric reference,
        not a physical stop.  Without this explicit completion marker the
        generic semantic scheduler inserts a redundant physical G1 after both
        side scans solely to satisfy its legacy anchor bookkeeping.
        """
        if (not self.config.visual_two_pose_lateral_enabled or
                self.active_door is None or str(room_id) != self._room_id()):
            return False
        self.visual_anchor_completed = True
        self.successful_roles.add("G1")
        self.events.append({
            "event": "ROOM_VIRTUAL_G1_ANCHOR_COMPLETED",
            "room_id": self._room_id(), "door_id": self.active_door.door_id,
            "reason": "two_lateral_full_sweeps",
        })
        return True

    def _prepare_and_activate(self, grid: OccupancyGrid2D,
                              door: EstimatedDoorway, current: Point,
                              now: float, proactive: bool,
                              allow_local_wall_break_support: bool = False,
                              reuse_existing: bool = False,
                              wall_break_clearance_override: Optional[float] = None
                              ) -> Optional[EstimatedDoorway]:
        # A false aperture at the opposite edge of a corridor can have a door
        # centre outside the duplicate radius while its inward ray still leads
        # directly into an already covered room.  Reject such alternate
        # entrances before committing locomotion.
        overlap = None
        for known in self.detector.doors:
            if not (known.visited or known.completed):
                continue
            center_separation = math.hypot(
                door.center[0] - known.center[0],
                door.center[1] - known.center[1])
            if (center_separation >
                    self.config.visited_room_overlap_max_door_separation):
                continue
            normal_alignment = _dot(door.normal, known.normal)
            # Opposing normals are two sides of a corridor, not two aperture
            # estimates of the same room.  The old ray-only test rejected the
            # missing fourth room in opt12 after a modest map-frame drift made
            # its projected inward ray cross a previously visited room.
            if (normal_alignment <
                    self.config.visited_room_overlap_min_normal_alignment):
                continue
            for probe_depth in (1.5, 3.0, 5.0):
                probe = (
                    door.center[0] + probe_depth * door.normal[0],
                    door.center[1] + probe_depth * door.normal[1])
                if (0.35 <= known.depth(probe) <=
                        self.config.room_depth_probe_range and
                        abs(known.lateral(probe)) <=
                        self.config.room_side_probe_range):
                    overlap = known
                    break
            if overlap is not None:
                break
        if overlap is not None:
            self.events.append({
                "event": "DOOR_CANDIDATE_REJECTED",
                "elapsed_sec": round(float(now), 3),
                "center": [door.center[0], door.center[1]],
                "reason": "inward_ray_overlaps_visited_room",
                "overlapping_door_id": overlap.door_id,
                "door_center_separation_m": round(float(math.hypot(
                    door.center[0] - overlap.center[0],
                    door.center[1] - overlap.center[1])), 3),
                "normal_alignment": round(float(_dot(
                    door.normal, overlap.normal)), 3),
            })
            self._set_state("CORRIDOR_SWEEP", now, "visited_room_overlap")
            return None
        # Once the local detector promotes a candidate, try it immediately.
        # The former doorway_candidate_valid() gate required bilateral jambs
        # and room-side expansion, which filtered out room doors in sparse or
        # drifted projections.  Safety is still enforced by known-free portal
        # samples, FAR/A*, and SCAN-lite during execution.
        valid, reason = doorway_candidate_entry_ready(grid, door)
        proposal = (select_room_goal(grid, door, current, "ENTRY", [], self.config)
                    if valid else None)
        portal = (prepare_portal_path(
            grid, door, current, proposal["position"], self.config)
                  if proposal is not None and
                  not proposal.get("entry_staging_fallback") else None)
        # Do not lose a confirmed side opening solely because the ideal
        # 1.15m entry cell is temporarily unknown/unreachable.  Use a local
        # free staging point and retry the inward entry after the next scan.
        if valid and (proposal is None or
                      (portal is None and
                       not proposal.get("entry_staging_fallback"))):
            proposal = fallback_entry_proposal(
                grid, door, current, self.config,
                clearance_override=wall_break_clearance_override
                if wall_break_clearance_override is not None else None)
            if (proposal is not None and
                    wall_break_clearance_override is not None):
                proposal["local_detector_verified_staging"] = True
            portal = None
        if not valid or proposal is None:
            rejection = (reason if not valid else
                         "no_safe_entry_candidate")
            # These checks depend on the still-growing occupancy projection.
            # A real doorway can fail them on the first pass and become valid a
            # few updates later.  Reserve the long cooldown for two actual
            # DOOR_COMMIT execution failures; evidence/preflight rejection gets
            # only a short debounce and remains recoverable via look-behind.
            key = self._set_door_cooldown(
                door.center,
                float(now) + self.config.door_evidence_retry_seconds)
            self.events.append({
                "event": "DOOR_CANDIDATE_REJECTED",
                "elapsed_sec": round(float(now), 3),
                "center": [door.center[0], door.center[1]],
                "reason": rejection,
                "cooldown_until": self.door_cooldowns[key],
            })
            self._set_state("CORRIDOR_SWEEP", now, "door_candidate_rejected")
            return None
        activated = self._activate(
            door, now, proactive=proactive,
            reuse_existing=reuse_existing)
        self.prepared_entry = dict(proposal, portal_preflight=portal)
        self.entry_portal_waypoints = ([list(point) for point in
                                       portal["mandatory_portal_waypoints"]]
                                      if portal is not None else [])
        return activated

    def retry_known_unvisited_door(
            self, grid: OccupancyGrid2D, current: Point, now: float,
            maximum_distance: float) -> Optional[EstimatedDoorway]:
        """Reactivate one cooled, known doorway that has never been exited.

        Door detection intentionally suppresses every portal already present
        in ``detector.doors``.  That is correct for visited rooms, but it also
        made an ENTRY failure permanent: after the cooldown, the real door
        was repeatedly observed and then discarded as a duplicate.  Reuse
        the original semantic landmark instead, while running the same live
        occupancy, portal A* and later SCAN-lite gates as a new detection.
        """
        if (not self.config.enabled or self.active_door is not None or
                grid is None or current is None):
            return None
        limit = max(0.0, float(maximum_distance))
        candidates = []
        for door in self.detector.doors:
            if door.visited or door.completed:
                continue
            cooldown_until = self._door_cooldown_until(door.center)
            if float(now) < cooldown_until:
                continue
            distance = _distance(current, door.corridor_side)
            if distance <= limit:
                candidates.append((distance, door.entered_at, door))
        if not candidates:
            return None
        distance, _, door = min(candidates, key=lambda item: item[:2])
        self.events.append({
            "event": "KNOWN_UNVISITED_DOOR_RETRY_SELECTED",
            "elapsed_sec": round(float(now), 3),
            "door_id": door.door_id,
            "center": [float(door.center[0]), float(door.center[1])],
            "distance_to_corridor_side_m": round(float(distance), 3),
            "policy": "reuse_landmark_after_cooldown",
        })
        return self._prepare_and_activate(
            grid, door, current, now, proactive=True,
            allow_local_wall_break_support=True,
            reuse_existing=True)

    def observe(self, grid: OccupancyGrid2D,
                trajectory: Sequence[Sequence[float]], now: float,
                corridor_axis: Optional[Sequence[float]] = None,
                maximum_corridor_alignment: float = 0.55
                ) -> Optional[EstimatedDoorway]:
        if not self.config.enabled or self.active_door is not None:
            return None
        door = self.detector.detect(grid, trajectory, now)
        if door is None:
            return None
        # detect() already registered the door; remove it before using the
        # common activation path, which assigns the stable id once.
        self.detector.doors.pop()
        if not doorway_is_lateral_to_corridor(
                door.normal_direction, corridor_axis,
                maximum_corridor_alignment):
            self.events.append({
                "event": "POST_CROSSING_REJECTED",
                "elapsed_sec": round(float(now), 3),
                "reason": "aperture_normal_aligned_with_corridor",
                "center": [door.center[0], door.center[1]],
                "door_normal_direction": door.normal_direction,
                "corridor_axis": list(corridor_axis),
            })
            return None
        key = self._door_key(door.center)
        cooldown_until = self._door_cooldown_until(door.center)
        if float(now) < cooldown_until:
            self.events.append({
                "event": "COOLED_DOOR_CROSSING_IGNORED",
                "elapsed_sec": round(float(now), 3),
                "center": [door.center[0], door.center[1]],
                "cooldown_until": cooldown_until,
            })
            return None
        crossed = (float(trajectory[-1][0]), float(trajectory[-1][1]))
        # A verified post-crossing has already completed ENTRY transit.  The
        # next semantic goal is still a separately selected room-centre G1.
        valid, reason = doorway_candidate_valid(grid, door, self.config)
        if not valid:
            self.events.append({
                "event": "POST_CROSSING_REJECTED",
                "elapsed_sec": round(float(now), 3), "reason": reason,
                "center": [door.center[0], door.center[1]],
            })
            return None
        crossing_points = []
        for point in trajectory:
            candidate_point = (float(point[0]), float(point[1]))
            if -1.2 <= door.depth(candidate_point) <= max(
                    self.config.confirmation_depth, 1.5):
                if (not crossing_points or
                        _distance(crossing_points[-1], candidate_point) >= 0.25):
                    crossing_points.append(candidate_point)
        return self._activate(door, now, proactive=False,
                              crossed_point=crossed,
                              entry_waypoints=crossing_points[-8:])

    def consider_planned_path(self, grid: OccupancyGrid2D,
                              path: Sequence[Sequence[float]],
                              now: float) -> Optional[EstimatedDoorway]:
        """Confirm and activate a door before executing a room-bound path."""
        if not self.config.enabled or self.active_door is not None:
            return None
        candidate = self.detector.planned_candidate(grid, path, now)
        if candidate is None:
            if self.pending_candidates:
                self.pending_candidates.clear()
                self._set_state("CORRIDOR_SWEEP", now,
                                "door_candidate_not_reconfirmed")
            return None
        key = self._pending_key(candidate.center)
        if float(now) < self._door_cooldown_until(candidate.center):
            return None
        record = self.pending_candidates.setdefault(key, {
            "count": 0, "first_seen": float(now), "candidate": candidate})
        for stale_key in list(self.pending_candidates):
            if stale_key != key:
                self.pending_candidates.pop(stale_key, None)
        record["count"] += 1
        record["candidate"] = candidate
        self._set_state("DOOR_APPROACH", now, "planned_aperture_candidate")
        required_count = int(self.config.candidate_confirmation_count)
        self.events.append({
            "event": "DOOR_CANDIDATE_CONFIRMED",
            "elapsed_sec": round(float(now), 3),
            "center": [candidate.center[0], candidate.center[1]],
            "confirmation_count": record["count"],
            "required_count": required_count,
            "confidence": candidate.confidence,
        })
        if record["count"] < self.config.candidate_confirmation_count:
            return None
        self.pending_candidates.pop(key, None)
        current = (float(path[0][0]), float(path[0][1]))
        # Corridor-side scans often lose one thin jamb in the 2-D projection.
        # The candidate has already passed three local confirmations and the
        # resulting entry still goes through known-free/A*/SCAN-lite checks.
        return self._prepare_and_activate(
            grid, candidate, current, now, proactive=True,
            allow_local_wall_break_support=True)

    def consider_corridor_door(self, grid: OccupancyGrid2D, current: Point,
                               corridor_heading: float, now: float,
                               lookahead: Optional[float] = None,
                               lookbehind: Optional[float] = None
                               ) -> Optional[EstimatedDoorway]:
        """Turn one detected side-wall opening into an entry attempt."""
        if not self.config.enabled or self.active_door is not None:
            return None
        candidate = self.detector.corridor_side_candidate(
            grid, current, corridor_heading, now,
            lookahead=lookahead, lookbehind=lookbehind)
        if candidate is None:
            if self.pending_candidates:
                self.pending_candidates.clear()
                self._set_state("CORRIDOR_SWEEP", now,
                                "door_candidate_not_reconfirmed")
            return None
        key = self._pending_key(candidate.center)
        if float(now) < self._door_cooldown_until(candidate.center):
            return None
        record = self.pending_candidates.setdefault(key, {
            "count": 0, "first_seen": float(now), "candidate": candidate})
        for stale_key in list(self.pending_candidates):
            if stale_key != key:
                self.pending_candidates.pop(stale_key, None)
        record["count"] += 1
        record["candidate"] = candidate
        self._set_state("DOOR_APPROACH", now, "corridor_side_aperture")
        # Floor 1 retains the one-raster default so a fast corridor sweep does
        # not pass a real door.  The independent floor-2 scheduler can require
        # repeat evidence because its post-stair projection is more vulnerable
        # to a wall gap being interpreted as an aperture.
        required_count = int(self.config.corridor_side_confirmation_count)
        self.events.append({
            "event": "DOOR_CANDIDATE_CONFIRMED",
            "elapsed_sec": round(float(now), 3),
            "center": [candidate.center[0], candidate.center[1]],
            "confirmation_count": record["count"],
            "required_count": required_count,
            "confidence": candidate.confidence,
            "source": "corridor_side_scan",
        })
        if record["count"] < required_count:
            return None
        self.pending_candidates.pop(key, None)
        return self._prepare_and_activate(
            grid, candidate, current, now, proactive=True,
            allow_local_wall_break_support=True)

    def consider_semantic_corridor_branch(
            self, grid: OccupancyGrid2D, current: Point,
            centerline: Point, corridor_axis: Sequence[float], side: int,
            now: float, branch_id: Optional[str] = None
            ) -> Optional[EstimatedDoorway]:
        """Debounce a side-frontier doorway without weakening safe ENTRY.

        The caller has already verified the branch endpoint with A* and
        SCAN-lite.  This method still requires an observed jamb aperture,
        room-side expansion, three independent map confirmations and the
        common portal preflight before activating the room state machine.
        """
        if not self.config.enabled or self.active_door is not None:
            return None
        candidate = semantic_corridor_branch_candidate(
            grid, centerline, corridor_axis, side, current, now, self.config,
            len(self.detector.doors) + 1)
        if candidate is None:
            return None
        key = self._pending_key(candidate.center)
        if float(now) < self._door_cooldown_until(candidate.center):
            return None
        record = self.pending_candidates.setdefault(key, {
            "count": 0, "first_seen": float(now), "candidate": candidate})
        for stale_key in list(self.pending_candidates):
            if stale_key != key:
                self.pending_candidates.pop(stale_key, None)
        record["count"] += 1
        record["candidate"] = candidate
        self._set_state("DOOR_APPROACH", now,
                        "semantic_corridor_side_frontier")
        self.events.append({
            "event": "DOOR_CANDIDATE_CONFIRMED",
            "elapsed_sec": round(float(now), 3),
            "center": [candidate.center[0], candidate.center[1]],
            "confirmation_count": record["count"],
            "required_count": 1,
            "confidence": candidate.confidence,
            "source": "semantic_corridor_side_frontier",
            "corridor_branch_id": branch_id,
        })
        self.pending_candidates.pop(key, None)
        return self._prepare_and_activate(
            grid, candidate, current, now, proactive=True)

    def consider_local_door_candidate(
            self, grid: OccupancyGrid2D, current: Point,
            evidence: dict, now: float,
            reuse_prevalidated_entry: bool = False
            ) -> Optional[EstimatedDoorway]:
        """Promote a robot-local aperture directly into the room scheduler.

        The external detector has already produced a local candidate. This
        adapter reconstructs the semantic doorway and sends an entry attempt;
        FAR/A* and SCAN-lite remain the execution safety gates.
        No global corridor station or layout coordinate is consulted.
        """
        if (not self.config.enabled or self.active_door is not None or
                not isinstance(evidence, dict) or
                not evidence.get("confirmed") or
                not evidence.get("astar_reachable") or
                not evidence.get("scan_lite_safe")):
            return None
        center = evidence.get("door_center") or []
        interior = evidence.get("entry_goal") or []
        corridor_side = evidence.get("corridor_side") or []
        try:
            center = (float(center[0]), float(center[1]))
            interior = (float(interior[0]), float(interior[1]))
            yaw = float(evidence["yaw"])
            width = float(evidence["width_m"])
            corridor_side = ((float(corridor_side[0]),
                              float(corridor_side[1]))
                             if len(corridor_side) >= 2 else
                             (center[0] - self.config.door_approach_offset *
                              math.cos(yaw),
                              center[1] - self.config.door_approach_offset *
                              math.sin(yaw)))
        except (KeyError, IndexError, TypeError, ValueError, OverflowError):
            return None
        # Local 8 m geometry can report a narrow wall crack as a doorway.
        # Those candidates passed the binary scan-lite gate but repeatedly
        # stalled at the jamb (0 waypoint progress), costing two 25 s entry
        # retries.  Require a physically traversable aperture for proactive
        # local promotion; semantic/global doors retain the normal configured
        # minimum and are not affected.
        local_width_floor = self.config.doorway_width_min
        candidate_id = str(evidence.get("candidate_id") or "")
        if candidate_id and float(now) < self.local_candidate_quarantine.get(
                candidate_id, 0.0):
            return None
        # Require a continuous observed-free normal passage from the corridor
        # side into the proposed interior point.  A* can accept a one-cell
        # doorway crack after inflation, yet the physical body still stalls at
        # the jamb.  This ray gate rejects that false aperture before it can
        # preempt the fast corridor goal.
        if (_segment_has_occupied(grid, corridor_side, interior) or
                _distance(corridor_side, interior) < 0.45):
            if candidate_id:
                self.local_candidate_quarantine[candidate_id] = float(now) + 12.0
            self.events.append({
                "event": "LOCAL_DOOR_CANDIDATE_REJECTED",
                "elapsed_sec": round(float(now), 3),
                "candidate_id": candidate_id,
                "width_m": round(width, 3) if math.isfinite(width) else None,
                "reason": "blocked_normal_passage_ray",
            })
            return None
        if (not all(math.isfinite(value) for value in
                    center + interior + corridor_side + (yaw, width)) or
                not (local_width_floor <= width <=
                     self.config.doorway_width_max) or
                self.detector._duplicate(center, yaw) or
                float(now) < self._door_cooldown_until(center)):
            if math.isfinite(width) and width < local_width_floor:
                # Debounce the same narrow aperture; the detector publishes
                # it every scan and otherwise floods the scheduler while the
                # robot is still trying to advance down the corridor.
                if candidate_id:
                    self.local_candidate_quarantine[candidate_id] = float(now) + 12.0
                self._set_door_cooldown(center, float(now) + 6.0)
                self.events.append({
                    "event": "LOCAL_DOOR_CANDIDATE_REJECTED",
                    "elapsed_sec": round(float(now), 3),
                    "candidate_id": evidence.get("candidate_id"),
                    "width_m": round(width, 3),
                    "reason": "narrow_aperture_jamb_stall_guard",
                })
            return None
        tangent = (-math.sin(yaw), math.cos(yaw))
        door = EstimatedDoorway(
            door_id="estimated_door_{:02d}".format(
                len(self.detector.doors) + 1),
            center=center, normal_direction=yaw, width=width,
            left_frame_point=(center[0] + 0.5 * width * tangent[0],
                              center[1] + 0.5 * width * tangent[1]),
            right_frame_point=(center[0] - 0.5 * width * tangent[0],
                               center[1] - 0.5 * width * tangent[1]),
            corridor_side=corridor_side, interior_side=interior,
            confidence=min(1.0, max(0.70, 0.70 + 0.05 *
                           float(evidence.get("confirmation_count", 2)))),
            entered_at=float(now))
        self.events.append({
            "event": "LOCAL_DOOR_CANDIDATE_PROMOTED",
            "elapsed_sec": round(float(now), 3),
            "candidate_id": evidence.get("candidate_id"),
            "center": list(center), "entry_goal": list(interior),
            "width_m": width,
            "source": "robot_local_8m_geometry",
        })
        # The mirrored same-station planner has already sampled the complete
        # corridor-side -> doorway -> interior normal ray and retained only an
        # observed-free, footprint-clear candidate. Running the generic
        # whole-room ENTRY search again here is both redundant and expensive:
        # speed_fix_26 spent 29.8 s synchronously in that search while the dog
        # stood still beside a confirmed opposite door. Reuse only the
        # explicitly prevalidated interior anchor. next_goal() still creates
        # the mandatory corridor-side/door/interior anchors, and the manager
        # still applies its independent SCAN-lite check before locomotion.
        if reuse_prevalidated_entry:
            activated = self._activate(door, now, proactive=True)
            self.prepared_entry = {
                "position": interior,
                "role": "ENTRY",
                "score": 0.0,
                "depth": door.depth(interior),
                "lateral": door.lateral(interior),
                "clearance": _clearance(grid, interior),
                "unknown_neighbors": 0,
                "predicted_local_gain_m2": 0.0,
                "preflight_path_length_m": _distance(current, interior),
                "fallback_candidate": False,
                "prevalidated_local_entry": True,
            }
            self.events.append({
                "event": "LOCAL_DOOR_PREVALIDATED_ENTRY_REUSED",
                "elapsed_sec": round(float(now), 3),
                "candidate_id": candidate_id,
                "door_id": door.door_id,
                "policy": "reuse_observed_free_normal_ray",
            })
            return activated
        return self._prepare_and_activate(
            grid, door, current, now, proactive=True,
            allow_local_wall_break_support=True)

    def abandon_corridor_resume(self, now: float, reason: str) -> None:
        """Release an unreachable recentering target instead of retrying it."""
        if self.active_door is None and self.state == "CORRIDOR_RESUME":
            self._set_state("CORRIDOR_SWEEP", now, reason)

    def entry_retry_pending(self) -> bool:
        """Whether a failed ENTRY still owns its configured bounded retry.

        The outer exploration manager uses this to avoid treating the first
        transient portal-footprint failure at the terminal door as final.
        The scheduler has already changed back to DOOR_COMMIT at that point,
        and ``next_goal`` will select its shallow, freshly checked staging
        path.  No extra retry is created here.
        """
        return bool(
            self.active_door is not None and
            not self.entry_confirmed and
            self.state == "DOOR_COMMIT" and
            0 < self.entry_attempts < self.config.entry_retry_limit)

    def abort_active_room(self, now: float, reason: str,
                          cooldown_seconds: Optional[float] = None) -> None:
        """Release a timed-out room so corridor exploration can resume.

        A semantic entry timeout must not leave ``active_door`` latched: that
        makes the planner return no goal and the executor hold zero velocity.
        The detected door remains in ``detector.doors`` for duplicate
        suppression, while the scheduler returns to corridor mode.
        """
        if self.active_door is None:
            return
        # An aborted ENTRY has demonstrated that this portal cannot be
        # traversed safely from the current map/pose.  Keep it out of the
        # immediate candidate pool; otherwise DOOR_COMMIT can select the very
        # same opening again and create a long, stale-map retry path.
        # A caller may grant one short retry for a transient map-refinement
        # failure.  All ordinary timeout/no-progress aborts retain the normal
        # long cooldown, preventing stale portal loops.
        cooldown = self.config.door_cooldown_seconds
        if cooldown_seconds is not None:
            try:
                requested = float(cooldown_seconds)
                if math.isfinite(requested):
                    cooldown = max(0.0, min(cooldown, requested))
            except (TypeError, ValueError):
                pass
        key = self._set_door_cooldown(
            self.active_door.center, float(now) + cooldown)
        self.events.append({
            "event": "ROOM_ABORTED",
            "elapsed_sec": round(float(now), 3),
            "door_id": self.active_door.door_id,
            "reason": str(reason),
            "cooldown_until": self.door_cooldowns.get(key),
        })
        self._clear_active()
        self._set_state("CORRIDOR_SWEEP", now, str(reason))

    def request_immediate_exit(self, now: float, reason: str) -> bool:
        """Skip optional room viewpoints after a continuity-pass entry."""
        if self.active_door is None or not self.entry_confirmed:
            return False
        self.adaptive_route_queue = []
        self.next_role_index = 3
        if self.completion_mode is None:
            self.completion_mode = "return_pass_entry_view"
        self._set_state("ROOM_RETURN", now, str(reason))
        self.events.append({
            "event": "ROOM_RETURN_PASS_IMMEDIATE_EXIT_REQUESTED",
            "elapsed_sec": round(float(now), 3),
            "room_id": self._room_id(),
            "door_id": self.active_door.door_id,
            "reason": str(reason),
        })
        return True

    def goal_in_visited_room(self, point: Point) -> bool:
        for door in self.detector.doors:
            if not (door.visited or door.completed):
                continue
            if 0.4 <= door.depth(point) <= 6.0 and abs(door.lateral(point)) <= 4.5:
                return True
        return False


    def next_goal(self, grid: OccupancyGrid2D, current: Point, now: float) -> Optional[dict]:
        if not self.config.enabled or self.active_door is None:
            return None
        elapsed = (float(now) - self.room_started_at
                   if self.room_started_at is not None else 0.0)
        if not self.entry_confirmed:
            if self.prepared_entry is not None:
                proposal = self.prepared_entry
                if self.config.entry_trace_observed_progress_only:
                    staging = (proposal.get("entry_staging_path_result") or {})
                    stale_path = list(staging.get("path") or [])
                    start_error = (_distance(current, stale_path[0])
                                   if stale_path else 0.0)
                    if start_error > max(0.60, 4.0 * grid.resolution):
                        refreshed = fallback_entry_proposal(
                            grid, self.active_door, current, self.config)
                        if refreshed is not None:
                            self.events.append({
                                "event": "ROOM_ENTRY_PATH_REFRESHED_AFTER_PREEMPTION",
                                "elapsed_sec": round(float(now), 3),
                                "room_id": self._room_id(),
                                "door_id": self.active_door.door_id,
                                "stale_start_error_m": round(start_error, 3),
                                "old_path_length_m": round(float(
                                    proposal.get("preflight_path_length_m", 0.0)
                                    or 0.0), 3),
                                "new_path_length_m": round(float(
                                    refreshed.get("preflight_path_length_m", 0.0)
                                    or 0.0), 3),
                            })
                            proposal = refreshed
            elif self.entry_attempts > 0:
                # A failed nominal ENTRY normally means the rolling 2-D
                # projection inflated the deeper interior endpoint after the
                # doorway was selected.  Retrying select_room_goal() chooses
                # the same deepest cell and repeats the failure.  Use the
                # already available, footprint/A*-checked shallow staging
                # path for the bounded retry; G1 still provides the required
                # deep room observation after the crossing is confirmed.
                proposal = fallback_entry_proposal(
                    grid, self.active_door, current, self.config)
                if proposal is not None:
                    self.events.append({
                        "event": "ROOM_ENTRY_STAGING_RETRY_SELECTED",
                        "elapsed_sec": round(float(now), 3),
                        "room_id": self._room_id(),
                        "door_id": self.active_door.door_id,
                        "attempt": self.entry_attempts + 1,
                        "position": list(proposal["position"]),
                        "depth": round(float(proposal["depth"]), 3),
                    })
            else:
                proposal = select_room_goal(
                    grid, self.active_door, current, "ENTRY", [], self.config)
            if proposal is None:
                key = self._set_door_cooldown(
                    self.active_door.center,
                    float(now) + self.config.door_cooldown_seconds)
                self.events.append({
                    "event": "DOOR_CANDIDATE_REJECTED",
                    "elapsed_sec": round(float(now), 3),
                    "door_id": self.active_door.door_id,
                    "reason": "no_safe_entry_candidate",
                    "cooldown_until": self.door_cooldowns[key],
                })
                self._clear_active()
                self._set_state("CORRIDOR_SWEEP", now, "door_candidate_rejected")
                return None
            if (self.config.require_portal_preflight_for_entry and
                    proposal.get("entry_staging_fallback")):
                # Before any physical ENTRY attempt this is only incomplete
                # map evidence, not proof that the doorway is bad.  Keep the
                # short evidence debounce so a fresh view from farther along
                # the corridor can promote the real portal.  After an actual
                # failed ENTRY, retain the normal long doorway quarantine.
                cooldown = (self.config.door_evidence_retry_seconds
                            if self.entry_attempts == 0 else
                            self.config.door_cooldown_seconds)
                key = self._set_door_cooldown(
                    self.active_door.center,
                    float(now) + cooldown)
                self.events.append({
                    "event": "DOOR_CANDIDATE_REJECTED",
                    "elapsed_sec": round(float(now), 3),
                    "door_id": self.active_door.door_id,
                    "reason": "entry_staging_fallback_disallowed",
                    "cooldown_until": self.door_cooldowns[key],
                })
                self._clear_active()
                self._set_state(
                    "CORRIDOR_SWEEP", now,
                    "unverified_entry_fallback_rejected")
                return None
            # A staging fallback already contains the exact A* path that was
            # validated for a reachable free-space pose.  It is deliberately
            # not a door-centre transit, so rebuilding nominal portal anchors
            # here makes the manager require waypoints absent from that path
            # and rejects the goal before the robot can move.
            portal = (None if proposal.get("entry_staging_fallback") else
                      (proposal.get("portal_preflight") or
                       prepare_portal_path(
                           grid, self.active_door, current,
                           proposal["position"], self.config)))
            if (self.config.require_portal_preflight_for_entry and
                    portal is None):
                cooldown = (self.config.door_evidence_retry_seconds
                            if self.entry_attempts == 0 else
                            self.config.door_cooldown_seconds)
                key = self._set_door_cooldown(
                    self.active_door.center,
                    float(now) + cooldown)
                self.events.append({
                    "event": "DOOR_CANDIDATE_REJECTED",
                    "elapsed_sec": round(float(now), 3),
                    "door_id": self.active_door.door_id,
                    "reason": "portal_preflight_unavailable",
                    "cooldown_until": self.door_cooldowns[key],
                })
                self._clear_active()
                self._set_state(
                    "CORRIDOR_SWEEP", now,
                    "unverified_portal_rejected")
                return None
            self.prepared_entry = None
            point = (tuple(portal["mandatory_portal_waypoints"][-1])
                     if portal is not None else
                     (float(proposal["position"][0]),
                      float(proposal["position"][1])))
            self._set_state("DOOR_COMMIT", now, "entry_transit")
            goal = {
                "position": [point[0], point[1], 0.0],
                "information_gain": 0.0,
                "score": float(proposal["score"]),
                "source": "lightweight_room_entry",
                "room_role": "ENTRY",
                "room_id": self._room_id(),
                "estimated_door_id": self.active_door.door_id,
                "room_goal_diagnostic": proposal,
                "execution_timeout_sec": min(
                    self.config.entry_timeout_seconds,
                    max(3.0, self.config.room_budget_seconds -
                        self.config.exit_reserve_seconds - elapsed)),
                "progress_timeout_sec": max(
                    15.0, self.config.exit_progress_timeout_seconds),
                "progressive_timeout": True,
                "room_goal_attempt": self.entry_attempts + 1,
                "mandatory_portal_waypoints": (
                    portal["mandatory_portal_waypoints"] if portal else []),
                "portal_clearance_m": (portal["portal_clearance_m"]
                                       if portal else 0.0),
                "portal_preflight_verified": bool(portal),
                "portal_resume_inside": bool(
                    portal.get("portal_resume_inside", False)
                    if portal else False),
            }
            if proposal.get("entry_staging_path_result"):
                goal["_preplanned_path_result"] = dict(
                    proposal["entry_staging_path_result"])
                goal["entry_staging_preflight_verified"] = True
            elif portal is not None and portal.get("preflight_path"):
                # ``prepare_portal_path`` has just A*-validated this exact
                # corridor-side -> centre -> interior sequence.  Reusing it
                # avoids immediately running the same grid search again;
                # manager-side SCAN-lite still validates every waypoint and
                # requests a fresh A* if the rolling map changed.
                goal["_preplanned_path_result"] = {
                    "success": True, "reason": "room_entry_astar_preflight_path",
                    "path": list(portal["preflight_path"])}
                goal["entry_preflight_reused"] = True
            self.selected_goals.append(dict(goal, planned_at=round(float(now), 3)))
            return goal
        # Stop as soon as the observed floor projection proves the room is
        # covered.  At least one successful in-room sensing pose is mandatory,
        # so a mature corridor map cannot falsely complete an unvisited room.
        coverage_viewpoints = list(self.completed_points)
        if current not in coverage_viewpoints:
            coverage_viewpoints.append(current)
        self.coverage_status = occupancy_room_coverage(
            grid, self.active_door, coverage_viewpoints, self.config)
        enough_observations = (
            len(self.successful_roles) >=
            self.config.coverage_minimum_observations)
        # The timed two-pose profile explicitly completes the virtual G1
        # anchor after its central full view is sufficient.  A small rolling
        # map residual must not then re-open the generic LiDAR set-cover
        # queue: that queue can insert physical G1 and G4 goals *after* the
        # visual contract has been completed, turning a one-view room into a
        # three-view room.  The original G3 scan remains a real interior
        # LiDAR observation; this only suppresses redundant follow-up views.
        two_pose_visual_complete = bool(
            self.config.visual_two_pose_lateral_enabled and
            self.visual_anchor_completed and
            not self.visual_deepening_requested and
            "G3" in self.successful_roles)
        if (enough_observations and
                (self.coverage_status["coverage_complete"] or
                 two_pose_visual_complete) and
                not self.visual_deepening_requested and
                self.visual_anchor_completed and
                self.completion_mode != "lidar_coverage" and
                (self.next_role_index < 3 or
                 self.config.adaptive_minimal_viewpoints)):
            pending_roles = (role for role in ("G1", "G2", "G3", "G4")
                             if role not in self.role_attempts)
            for pending_role in pending_roles:
                self.events.append({
                    "event": "ROOM_GOAL_SKIPPED",
                    "elapsed_sec": round(float(now), 3),
                    "door_id": self.active_door.door_id,
                    "role": pending_role,
                    "reason": ("two_pose_visual_complete_no_redundant_route"
                               if two_pose_visual_complete else
                               "lidar_coverage_complete"),
                })
            self.next_role_index = 3
            self.completion_mode = "lidar_coverage"
            self._set_state("ROOM_RETURN", now, "lidar_coverage_complete")
            self.events.append({
                "event": "ROOM_COVERAGE_COMPLETE",
                "elapsed_sec": round(float(now), 3),
                "room_id": self._room_id(),
                "door_id": self.active_door.door_id,
                "coverage": dict(self.coverage_status),
                "two_pose_visual_complete": two_pose_visual_complete,
            })

        # Reserve enough of the same hard visit budget for RETURN and EXIT.
        observation_deadline = max(
            0.0, self.config.room_budget_seconds -
            self.config.exit_reserve_seconds)
        observation_attempts = len({
            role for role in self.role_attempts
            if role in ("G1", "G2", "G3", "G4")})
        observation_capacity = (
            observation_attempts < self.config.adaptive_maximum_viewpoints
            if self.config.adaptive_minimal_viewpoints else
            self.next_role_index < 3)
        if (elapsed >= observation_deadline and observation_capacity and
                self.completion_mode != "lidar_coverage"):
            pending_roles = (role for role in ("G1", "G2", "G3", "G4")
                             if role not in self.role_attempts)
            for pending_role in pending_roles:
                self.failed_roles.add(pending_role)
                self.events.append({
                    "event": "ROOM_GOAL_SKIPPED",
                    "elapsed_sec": round(float(now), 3),
                    "door_id": self.active_door.door_id,
                    "role": pending_role,
                    "reason": "room_budget_exit_reservation",
                })
            self.next_role_index = 3
            if self.completion_mode is None:
                self.completion_mode = "time_budget"
            self._set_state("ROOM_RETURN", now, "exit_time_reservation")

        if (self.config.adaptive_minimal_viewpoints and
                self.next_role_index < 3):
            attempted = [role for role in self.role_attempts
                         if role in ("G1", "G2", "G3", "G4")]
            proposal = None
            # Once LiDAR coverage is complete, an outstanding RGB-D anchor
            # is the only remaining sensing debt.  Do not re-run set cover
            # and send the robot to an unrelated long G2/G3 point merely to
            # satisfy that camera requirement: use the existing G1 route (or
            # one footprint-validated G1 fallback) and exit immediately
            # afterwards.  This preserves broad LiDAR coverage and one
            # interior RGB-D view while removing a recurrent 5--7 m detour.
            if (enough_observations and
                    self.coverage_status.get("coverage_complete") and
                    not self.visual_anchor_completed and
                    "G1" not in self.role_attempts and
                    # A two-pose visual sweep uses the two occupancy/A*
                    # validated lateral views as its camera anchors.  Never
                    # insert a physical central G1 between them: it adds a
                    # detour without improving the two-baseline coverage.
                    not (self.config.visual_two_pose_lateral_enabled and
                         self.visual_deepening_requested)):
                queued_g1 = next((item for item in self.adaptive_route_queue
                                  if str(item.get("role")) == "G1"), None)
                if queued_g1 is not None:
                    self.adaptive_route_queue.remove(queued_g1)
                    proposal = dict(queued_g1)
                else:
                    visual_anchor = select_room_goal(
                        grid, self.active_door, current, "G1",
                        self.completed_points, self.config, fallback=True)
                    if visual_anchor is not None:
                        proposal = dict(visual_anchor)
                        proposal["coverage_cells"] = set()
                        proposal["adaptive_marginal_gain_m2"] = 0.0
                        proposal["adaptive_utility"] = float(
                            proposal.get("score", 0.0))
                if proposal is not None:
                    proposal["visual_observation_fallback"] = True
                    self.events.append({
                        "event": "ADAPTIVE_ROOM_COVERAGE_COMPLETE_G1_ONLY",
                        "elapsed_sec": round(float(now), 3),
                        "room_id": self._room_id(),
                        "door_id": self.active_door.door_id,
                        "role": proposal.get("role"),
                        "reason": "lidar_complete_rgbd_anchor_pending",
                    })
            while self.adaptive_route_queue:
                if proposal is not None:
                    break
                queued = self.adaptive_route_queue[0]
                cells = queued.get("coverage_cells", set())
                remaining = sum(
                    1 for x, y in cells
                    if 0 <= x < grid.width and 0 <= y < grid.height and
                    int(grid.data[y, x]) < 0)
                # The route was generated from a previous occupancy frame.
                # Once G1 has supplied a fresh LiDAR scan, a queued lateral
                # point can have *no* remaining unknown/shadow cell at all.
                # Executing it only adds a 4--6 m camera detour and cannot
                # reveal a LiDAR-suspect obstacle.  Discard it explicitly;
                # this is safer than treating an empty set as a mandatory
                # coverage waypoint.
                if not cells or remaining == 0:
                    skipped = self.adaptive_route_queue.pop(0)
                    skipped_role = str(skipped["role"])
                    self.role_attempts[skipped_role] = 0
                    self.events.append({
                        "event": "ADAPTIVE_ROOM_VIEW_SKIPPED_NO_REMAINING_LIDAR_DEBT",
                        "elapsed_sec": round(float(now), 3),
                        "room_id": self._room_id(),
                        "role": skipped_role,
                        "reason": "fresh_g1_scan_removed_candidate_gain",
                    })
                    continue
                if (cells and remaining * stride_area(grid) <
                        self.config.adaptive_minimum_gain_m2):
                    skipped = self.adaptive_route_queue.pop(0)
                    skipped_role = str(skipped["role"])
                    self.role_attempts[skipped_role] = 0
                    self.events.append({
                        "event": "ADAPTIVE_ROOM_VIEW_SKIPPED_AFTER_COVERAGE",
                        "elapsed_sec": round(float(now), 3),
                        "room_id": self._room_id(),
                        "role": skipped_role,
                        "remaining_gain_m2": remaining * stride_area(grid),
                    })
                    continue
                proposal = self.adaptive_route_queue.pop(0)
                break
            if (proposal is None and
                    len(attempted) < self.config.adaptive_maximum_viewpoints):
                # Immediately after a portal crossing the local room mask can
                # legitimately contain no known cells yet.  Running the full
                # candidate/A* set-cover search in that state is pure delay:
                # every candidate has zero measured marginal coverage and the
                # search eventually chooses the ordinary G1 entry view
                # anyway.  Take that same footprint-validated view directly
                # and let its fresh LiDAR/RGB-D frame seed the next cycle.
                # This prevents 40--60 s CPU stalls observed at a doorway on
                # a newly expanded voxel grid without weakening any later
                # coverage decision.
                no_room_observation_yet = bool(
                    not attempted and
                    int(self.coverage_status.get("known_cells", 0) or 0) == 0)
                # The timed profile permits exactly one interior viewpoint.
                # Running the full set-cover/TSP search cannot change that
                # cardinality, yet on a partly observed room it repeatedly
                # spent 8--12 s constructing coverage masks before selecting
                # a single G3 point.  Select from the same footprint-checked
                # structural candidates directly.  Prefer the central G1
                # when it is competitive (better RGB-D field of view), while
                # retaining a shorter side fallback if G1 is blocked.
                fast_single_view = bool(
                    not attempted and
                    self.config.adaptive_maximum_viewpoints <= 1)
                central_first = bool(
                    not attempted and
                    self.config.adaptive_prefer_central_first)
                if fast_single_view or central_first:
                    candidates = []
                    initial_roles = (("G3", "G4") if
                                     (self.config.visual_two_pose_lateral_enabled and
                                      self.visual_deepening_requested) else
                                     ("G1", "G3", "G4"))
                    for candidate_role in initial_roles:
                        candidate = select_room_goal(
                            grid, self.active_door, current, candidate_role,
                            self.completed_points or [current], self.config,
                            fallback=True)
                        if candidate is not None:
                            candidates.append(candidate)
                    if candidates:
                        shortest = min(
                            candidates,
                            key=lambda item: float(item.get(
                                "preflight_path_length_m", math.inf)))
                        centered = next((item for item in candidates
                                         if item.get("role") == "G1"), None)
                        virtual_g1 = None
                        if (self.config.visual_two_pose_lateral_enabled and
                                self.visual_deepening_requested):
                            reference = select_room_goal(
                                grid, self.active_door, current, "G1",
                                self.completed_points, self.config,
                                fallback=True)
                            if reference is not None:
                                virtual_g1 = reference["position"]
                        if virtual_g1 is not None:
                            # Start the visual pass at the live, central G1
                            # reference rather than immediately crossing the
                            # room to a lateral extreme.  The manager turns
                            # here first, measures *actual* RGB-D coverage,
                            # and requests the opposite G4 view only if a
                            # real blind sector remains.  Reusing the G3 role
                            # preserves the two-pose sweep bookkeeping while
                            # avoiding a compulsory two-crossing room tour.
                            proposal = dict(reference)
                            proposal["role"] = "G3"
                            proposal["central_visual_first"] = True
                            proposal["virtual_g1_reference"] = [
                                float(virtual_g1[0]), float(virtual_g1[1])]
                            proposal["virtual_g1_baseline_m"] = 0.0
                        elif (centered is not None and (central_first or
                                float(centered.get("preflight_path_length_m", math.inf))
                                <= float(shortest.get(
                                    "preflight_path_length_m", math.inf)) + 1.0)):
                            proposal = centered
                        else:
                            proposal = shortest
                if proposal is not None:
                    proposal = dict(proposal)
                    proposal["coverage_cells"] = set()
                    proposal["adaptive_marginal_gain_m2"] = 1.0
                    proposal["adaptive_utility"] = float(
                        proposal.get("score", 0.0))
                    proposal["fast_unobserved_room_entry"] = bool(
                        no_room_observation_yet)
                    self.events.append({
                        "event": "ADAPTIVE_ROOM_FAST_SINGLE_VIEW_SELECTED",
                        "elapsed_sec": round(float(now), 3),
                        "room_id": self._room_id(),
                        "door_id": self.active_door.door_id,
                        "role": proposal["role"],
                        "reason": ("no_known_room_cells_skip_set_cover"
                                   if no_room_observation_yet else
                                   ("central_first_measured_coverage_then_side_if_needed"
                                    if central_first else
                                    "single_view_skip_set_cover")),
                        "position": [float(proposal["position"][0]),
                                     float(proposal["position"][1])],
                    })
                elif (no_room_observation_yet and not
                      (self.config.visual_two_pose_lateral_enabled and
                       self.visual_deepening_requested)):
                    proposal = select_room_goal(
                        grid, self.active_door, current, "G1",
                        self.completed_points, self.config, fallback=True)
                    if proposal is not None:
                        proposal = dict(proposal)
                        proposal["coverage_cells"] = set()
                        proposal["adaptive_marginal_gain_m2"] = 1.0
                        proposal["adaptive_utility"] = float(
                            proposal.get("score", 0.0))
                        proposal["fast_unobserved_room_entry"] = True
                        self.events.append({
                            "event": "ADAPTIVE_ROOM_FAST_ENTRY_VIEW_SELECTED",
                            "elapsed_sec": round(float(now), 3),
                            "room_id": self._room_id(),
                            "door_id": self.active_door.door_id,
                            "role": "G1",
                            "reason": "no_known_room_cells_skip_set_cover",
                            "position": [float(proposal["position"][0]),
                                         float(proposal["position"][1])],
                        })
                # The first anchor view can satisfy LiDAR coverage while
                # RGB-D still has an interior fan gap.  Request exactly one
                # additional safe, map-derived side/depth view; it is not a
                # fixed room waypoint and is unavailable without that online
                # visual debt.
                if proposal is None and self.visual_deepening_requested:
                    # This is a request-local breadth baseline around the
                    # already reached G1 view.  It stays at the measured G1
                    # depth and offsets only 0.55--1.0 m laterally, with
                    # occupancy/A* validation for each endpoint.  No layout
                    # coordinate or fixed room waypoint is used.
                    visual_config = copy.copy(self.config)
                    visual_target_depth = max(
                        1.15, self.active_door.depth(current))
                    visual_config.side_depth = visual_target_depth
                    visual_config.maximum_center_depth = visual_target_depth
                    visual_config.maximum_side_lateral = min(
                        2.50, max(0.55, self.config.visual_breadth_lateral))
                    # The normal 0.90 m separation rejects exactly the
                    # compact G1 breadth baseline we want here.  Clearance
                    # and A* remain mandatory; only duplicate suppression is
                    # reduced to the requested local-view scale.
                    visual_config.minimum_goal_separation = min(
                        self.config.minimum_goal_separation,
                        max(0.40, visual_config.maximum_side_lateral * 0.70))
                    visual_candidates = []
                    for visual_role in ("G3", "G4"):
                        if visual_role in attempted:
                            continue
                        candidate = select_room_goal(
                            grid, self.active_door, current, visual_role,
                            self.completed_points or [current], visual_config,
                            fallback=True, boundary_probe=True)
                        if candidate is not None:
                            visual_candidates.append(candidate)
                    if visual_candidates:
                        # Prefer the strongest separated camera baseline.
                        # The endpoints are online occupancy/A*-validated;
                        # choosing the furthest one is what keeps a central
                        # obstacle from occluding both RGB-D turns.
                        proposal = max(
                            visual_candidates,
                            key=lambda item: (
                                _distance(item["position"], current),
                                self.active_door.depth(item["position"]),
                                float(item.get("score", 0.0))))
                        proposal = dict(proposal)
                        baseline_m = _distance(proposal["position"], current)
                        proposal["visual_baseline_m"] = float(baseline_m)
                        proposal["visual_baseline_requirement_met"] = bool(
                            baseline_m >= self.config.visual_breadth_min_baseline)
                        proposal["coverage_cells"] = set()
                        proposal["adaptive_marginal_gain_m2"] = .20
                        proposal["adaptive_utility"] = float(proposal.get("score", 0.0))
                        proposal["visual_deepening"] = True
                        self.events.append({
                            "event": "ADAPTIVE_ROOM_VISUAL_BREADTH_VIEW_SELECTED",
                            "elapsed_sec": round(float(now), 3),
                            "room_id": self._room_id(),
                            "door_id": self.active_door.door_id,
                            "role": proposal["role"],
                            "position": [float(proposal["position"][0]),
                                         float(proposal["position"][1])],
                            "depth_m": round(float(self.active_door.depth(
                                proposal["position"])), 3),
                            "requested_depth_m": round(float(visual_target_depth), 3),
                            "requested_lateral_m": round(
                                float(visual_config.maximum_side_lateral), 3),
                            "baseline_from_g1_m": round(float(baseline_m), 3),
                            "minimum_requested_baseline_m": round(float(
                                self.config.visual_breadth_min_baseline), 3),
                            "baseline_requirement_met": bool(
                                proposal["visual_baseline_requirement_met"]),
                        })
                # While the two lateral visual sweep is active, the generic
                # LiDAR set-cover queue is not allowed to reintroduce G1.
                # The visual branch above chooses only the remaining G3/G4
                # anchors, then the manager marks virtual G1 complete.
                route_plan = (None if (proposal is not None or
                              (self.config.visual_two_pose_lateral_enabled and
                               self.visual_deepening_requested)) else
                              plan_minimum_coverage_route(
                                  grid, self.active_door, current,
                                  self.completed_points, attempted,
                                  self.coverage_status,
                                  self.return_anchor or
                                  self.active_door.interior_side,
                                  self.config))
                if route_plan is not None:
                    self.adaptive_route_queue = list(route_plan["ordered"])
                    self.events.append({
                        "event": "ADAPTIVE_ROOM_COVERAGE_ROUTE_PLANNED",
                        "elapsed_sec": round(float(now), 3),
                        "room_id": self._room_id(),
                        "door_id": self.active_door.door_id,
                        "roles": [item["role"] for item in
                                  self.adaptive_route_queue],
                        "waypoints": [[item["position"][0],
                                       item["position"][1]] for item in
                                      self.adaptive_route_queue],
                        "coverable_cell_count": route_plan[
                            "coverable_cell_count"],
                        "covered_cell_count": route_plan[
                            "covered_cell_count"],
                        "coverage_ratio": route_plan["coverage_ratio"],
                        "residual_m2": route_plan["residual_m2"],
                        "largest_residual_m2": route_plan.get(
                            "largest_residual_m2"),
                        "largest_initial_gap_m2": route_plan.get(
                            "largest_initial_gap_m2"),
                        "route_length_m": route_plan["route_length_m"],
                        "return_distance_m": route_plan.get(
                            "return_distance_m"),
                        "selection_reason": route_plan["selection_reason"],
                        "boundary_complete_before": route_plan.get(
                            "boundary_complete_before"),
                        "missing_boundary_roles": route_plan.get(
                            "missing_boundary_roles", []),
                        "required_boundary_roles": route_plan.get(
                            "required_boundary_roles", []),
                        "far_boundary_required": route_plan.get(
                            "far_boundary_required", False),
                    })
                    proposal = self.adaptive_route_queue.pop(0)
            # LiDAR can already see most of a small room from the doorway and
            # therefore leave the coverage route empty.  That is sufficient
            # for mapping, but not for the forward-facing RGB-D camera: the
            # camera has not yet received an interior transit and can miss a
            # low red sphere behind furniture or outside the doorway FOV.
            # Keep the timed one-view profile, but guarantee its single
            # footprint-validated G1 observation before returning.  This is
            # deliberately limited to rooms with no previous observation, so
            # it cannot turn into an unbounded coverage sweep.
            if (proposal is None and
                    self.config.visual_two_pose_lateral_enabled and
                    self.visual_deepening_requested):
                # Map updates immediately after ENTRY can temporarily erase
                # every sampled G3/G4 cell even though the just-reached
                # interior pose is known safe.  Do not interpret that short
                # mapping gap as permission to leave the room: use the
                # successful ENTRY pose for the missing lateral camera turn,
                # let its LiDAR frame refresh the grid, and then retry the
                # normal map-derived opposite view on the next cycle.  This
                # fallback has no translation and is only available for the
                # two-view visual request, so it cannot expand exploration.
                next_lateral = next((role for role in ("G3", "G4")
                                     if role not in attempted), None)
                if next_lateral is not None:
                    proposal = {
                        "position": (float(current[0]), float(current[1])),
                        "role": next_lateral,
                        "score": 0.0,
                        "clearance": _clearance(grid, current),
                        "coverage_cells": set(),
                        "adaptive_marginal_gain_m2": 0.0,
                        "adaptive_utility": 0.0,
                        "preflight_path_length_m": 0.0,
                        "fallback_candidate": True,
                        "entry_pose_visual_fallback": True,
                        "visual_deepening": True,
                    }
                    self.events.append({
                        "event": "ADAPTIVE_ROOM_VISUAL_ENTRY_POSE_FALLBACK",
                        "elapsed_sec": round(float(now), 3),
                        "room_id": self._room_id(),
                        "door_id": self.active_door.door_id,
                        "role": next_lateral,
                        "reason": "temporary_no_safe_lateral_sample_after_entry",
                        "position": [float(current[0]), float(current[1])],
                    })
            if proposal is None and not attempted:
                visual_anchor = select_room_goal(
                    grid, self.active_door, current, "G1",
                    self.completed_points, self.config, fallback=True)
                if visual_anchor is not None:
                    proposal = dict(visual_anchor)
                    proposal["coverage_cells"] = set()
                    proposal["adaptive_marginal_gain_m2"] = 0.0
                    proposal["adaptive_utility"] = float(
                        proposal.get("score", 0.0))
                    proposal["visual_observation_fallback"] = True
                    self.events.append({
                        "event": "ADAPTIVE_ROOM_VISUAL_ANCHOR_SELECTED",
                        "elapsed_sec": round(float(now), 3),
                        "room_id": self._room_id(),
                        "door_id": self.active_door.door_id,
                        "role": "G1",
                        "reason": "no_lidar_marginal_gain_but_no_rgbd_view",
                        "position": [float(proposal["position"][0]),
                                     float(proposal["position"][1])],
                    })
            if proposal is not None:
                role = str(proposal["role"])
                point = proposal["position"]
                diagnostic = dict(proposal)
                diagnostic["coverage_cell_count"] = len(
                    diagnostic.pop("coverage_cells", set()))
                self._set_state({
                    "G1": "G1_CENTER", "G2": "G2_OCCLUSION_DEPTH",
                    "G3": "G3_LEFT", "G4": "G4_RIGHT",
                }[role], now, "adaptive_minimum_coverage_view")
                goal = {
                    "position": [point[0], point[1], 0.0],
                    "information_gain": float(
                        proposal["adaptive_marginal_gain_m2"]),
                    "score": float(proposal["adaptive_utility"]),
                    "source": "lightweight_room_semantic",
                    "room_role": role,
                    "room_id": self._room_id(),
                    "estimated_door_id": self.active_door.door_id,
                    "room_goal_diagnostic": diagnostic,
                    "execution_timeout_sec": min(
                        self.config.room_goal_timeout_seconds,
                        max(3.0, observation_deadline - elapsed)),
                    "progress_timeout_sec": min(
                        self.config.room_goal_timeout_seconds,
                        max(3.0, observation_deadline - elapsed),
                        max(15.0, self.config.exit_progress_timeout_seconds)),
                    "progressive_timeout": True,
                    "room_goal_attempt": 1,
                    "fallback_candidate": bool(
                        proposal.get("fallback_candidate", False)),
                    "adaptive_minimal_viewpoint": True,
                }
                self.role_attempts[role] = 1
                self.events.append({
                    "event": "ADAPTIVE_ROOM_VIEW_SELECTED",
                    "elapsed_sec": round(float(now), 3),
                    "room_id": self._room_id(),
                    "door_id": self.active_door.door_id,
                    "role": role,
                    "coverage_before": dict(self.coverage_status),
                    "marginal_gain_m2": proposal[
                        "adaptive_marginal_gain_m2"],
                    "path_length_m": proposal[
                        "preflight_path_length_m"],
                    "utility": proposal["adaptive_utility"],
                })
                self.selected_goals.append(
                    dict(goal, planned_at=round(float(now), 3)))
                return goal
            self.next_role_index = 3
            if self.completion_mode is None:
                self.completion_mode = (
                    "adaptive_viewpoint_budget" if
                    len(attempted) >= self.config.adaptive_maximum_viewpoints
                    else "adaptive_no_marginal_gain")
            self._set_state("ROOM_RETURN", now, self.completion_mode)
            self.events.append({
                "event": "ADAPTIVE_ROOM_OBSERVATION_STOPPED",
                "elapsed_sec": round(float(now), 3),
                "room_id": self._room_id(),
                "door_id": self.active_door.door_id,
                "reason": self.completion_mode,
                "attempted_roles": attempted,
                "coverage": dict(self.coverage_status),
            })
        roles = ("G1", "G3", "G4")
        while (not self.config.adaptive_minimal_viewpoints and
               self.next_role_index < len(roles)):
            role = roles[self.next_role_index]
            fallback = (role in ("G3", "G4") and
                        self.role_attempts.get(role, 0) > 0)
            proposal = select_room_goal(
                grid, self.active_door, current, role,
                self.completed_points, self.config, fallback=fallback)
            if proposal is None and role in ("G3", "G4") and not fallback:
                proposal = select_room_goal(
                    grid, self.active_door, current, role,
                    self.completed_points, self.config, fallback=True)
                fallback = proposal is not None
            self.next_role_index += 1
            if proposal is None:
                skip_reason = {
                    "G1": "no_safe_room_center_candidate",
                    "G3": "no_safe_left_near_wall_path",
                    "G4": "no_safe_right_near_wall_path",
                }.get(role, "no_safe_observed_free_candidate")
                self.events.append({
                    "event": "ROOM_GOAL_SKIPPED", "elapsed_sec": round(float(now), 3),
                    "door_id": self.active_door.door_id, "role": role,
                    "reason": skip_reason,
                })
                if role in ("G1", "G3", "G4"):
                    self.failed_roles.add(role)
                continue
            point = proposal["position"]
            if role == "G1":
                self._set_state("G1_CENTER", now)
            elif role == "G3":
                self._set_state("G3_LEFT", now)
            else:
                self._set_state("G4_RIGHT", now)
            goal = {
                "position": [point[0], point[1], 0.0],
                "information_gain": float(proposal["unknown_neighbors"]),
                "score": float(proposal["score"]),
                "source": "lightweight_room_semantic",
                "room_role": role,
                "room_id": self._room_id(),
                "estimated_door_id": self.active_door.door_id,
                "room_goal_diagnostic": proposal,
                "execution_timeout_sec": min(
                    self.config.room_goal_timeout_seconds,
                    max(3.0, observation_deadline - elapsed)),
                "progress_timeout_sec": min(
                    self.config.room_goal_timeout_seconds,
                    max(3.0, observation_deadline - elapsed),
                    max(15.0, self.config.exit_progress_timeout_seconds)),
                "progressive_timeout": True,
                "room_goal_attempt": self.role_attempts.get(role, 0) + 1,
                "fallback_candidate": bool(fallback),
            }
            self.role_attempts[role] = self.role_attempts.get(role, 0) + 1
            self.selected_goals.append(dict(goal, planned_at=round(float(now), 3)))
            return goal
        required_roles = {"G1", "G3", "G4"}
        semantic_complete = required_roles.issubset(self.successful_roles)
        if self.completion_mode is None:
            if semantic_complete:
                self.completion_mode = "three_point_semantic"
            else:
                self.completion_mode = "incomplete"

        if self.config.direct_verified_exit and not self.return_completed:
            # The EXIT preflight below already A*-checks current -> doorway ->
            # corridor and SCAN-lite validates every resulting waypoint.  A
            # separate RETURN endpoint duplicates this safe route and caused
            # two eight-second timeouts per room in the 300 s validation.
            self.return_completed = True
            self.events.append({
                "event": "ROOM_RETURN_FOLDED_INTO_EXIT",
                "elapsed_sec": round(float(now), 3),
                "room_id": self._room_id(),
                "door_id": self.active_door.door_id,
                "reason": "direct_astar_scan_verified_exit",
            })

        # Return from a side observation to the saved interior portal anchor
        # before issuing the short, mandatory door crossing segment.
        normal = self.active_door.normal
        alignment_depth = max(
            0.75, min(1.20, 0.65 * self.config.entry_depth))
        # RETURN ends on the inward door-normal centreline, not merely at the
        # historical entry pose.  EXIT can then approach the portal without a
        # diagonal footprint sweeping across either jamb.
        interior_anchor = (
            self.active_door.center[0] + alignment_depth * normal[0],
            self.active_door.center[1] + alignment_depth * normal[1])
        snapped_anchor = _snap_safe_point(
            grid, interior_anchor, self.config.goal_clearance,
            self.active_door, depth_tolerance=0.75, search_radius=0.90)
        return_path_check = None
        if snapped_anchor is not None:
            path_check = astar_safe_path(
                grid, current, snapped_anchor, self.config.goal_clearance,
                0.15, allow_blocked_start=True)
            if path_check.get("success"):
                interior_anchor = snapped_anchor
                return_path_check = path_check
        if (not self.return_completed and
                _distance(current, interior_anchor) <= 0.45):
            self.return_completed = True
            self.events.append({
                "event": "ROOM_RETURNED_TO_PORTAL",
                "elapsed_sec": round(float(now), 3),
                "room_id": self._room_id(),
                "door_id": self.active_door.door_id,
                "reason": "already_at_interior_anchor",
            })
        if not self.return_completed:
            if self.return_attempts >= 2:
                # Two genuine planning/execution failures are enough.  The
                # EXIT preflight still A*-checks the current pose -> mandatory
                # door centre -> corridor side, so bypassing an unusable
                # staging cell cannot create a wall shortcut.
                self.return_completed = True
                self.events.append({
                    "event": "ROOM_RETURN_BYPASSED_TO_VERIFIED_EXIT",
                    "elapsed_sec": round(float(now), 3),
                    "room_id": self._room_id(),
                    "door_id": self.active_door.door_id,
                    "attempts": self.return_attempts,
                    "reason": "return_anchor_unusable",
                })
            else:
                self.return_anchor = interior_anchor
        if not self.return_completed:
            self._set_state("ROOM_RETURN", now)
            # The observation budget may be exhausted while the final side
            # goal is still making useful progress.  Never turn that into a
            # one-second RETURN loop: getting back to the saved inside anchor
            # is a safety action and needs one realistic traversal window.
            remaining_total = max(
                self.config.portal_timeout_seconds,
                self.config.room_budget_seconds - elapsed)
            remaining_budget = max(
                self.config.portal_timeout_seconds,
                remaining_total -
                min(self.config.portal_timeout_seconds,
                    0.45 * remaining_total))
            goal = {
                "position": [interior_anchor[0], interior_anchor[1], 0.0],
                "information_gain": 0.0, "score": 0.0,
                "source": "lightweight_room_return", "room_role": "RETURN",
                "room_id": self._room_id(),
                "estimated_door_id": self.active_door.door_id,
                "room_return_attempt": self.return_attempts + 1,
                "execution_timeout_sec": remaining_budget,
                "progress_timeout_sec": min(
                    remaining_budget,
                    max(8.0, self.config.exit_progress_timeout_seconds)),
                "progressive_timeout": True,
            }
            if (return_path_check is not None and
                    return_path_check.get("path")):
                goal["_preplanned_path_result"] = {
                    "success": True,
                    "reason": "room_return_astar_preflight_path",
                    "path": list(return_path_check.get("path") or []),
                }
                goal["return_preflight_verified"] = True
            self.selected_goals.append(dict(goal, planned_at=round(float(now), 3)))
            return goal

        # Exit correction is deliberately scoped after confirmed room entry.
        # Door detection, takeover and ENTRY retain their original trajectory
        # anchors; only the final return crossing gets a guaranteed outside
        # endpoint.
        exit_point, _ = canonical_doorway_anchors(
            self.active_door.center,
            self.active_door.normal_direction,
            self.config)
        self._set_state("ROOM_EXIT", now)
        # Likewise, once at the portal, always allow a complete verified door
        # crossing even if semantic observation consumed the nominal budget.
        remaining_budget = self.config.portal_timeout_seconds
        # Upper-floor managers explicitly opt into accepting the exact,
        # endpoint-matched reverse ENTRY trace as corridor-exit proof.  That
        # trace was just physically executed and footprint-audited.  Prefer it
        # before rebuilding an A* portal from the post-scan raster: run75
        # spent 178 s synchronously expanding that stale raster even though
        # the reverse trace subsequently executed successfully in 4.8 s.
        entry_trace_corridor_started = bool(
            len(self.entry_portal_waypoints) >= 2 and
            self.active_door.depth(self.entry_portal_waypoints[0]) <= 0.0)
        prefer_reverse_entry = bool(
            self.config.accept_reversed_entry_trace_exit and
            self.entry_confirmed and
            entry_trace_corridor_started)
        exit_portal = (None if prefer_reverse_entry else prepare_exit_path(
            grid, self.active_door, current, self.config))
        if exit_portal is None:
            self.events.append({
                "event": (
                    "ROOM_EXIT_PREFLIGHT_SKIPPED_REVERSE_ENTRY_TRACE"
                    if prefer_reverse_entry else
                    "ROOM_EXIT_PREFLIGHT_FAILED"),
                "elapsed_sec": round(float(now), 3),
                "room_id": self._room_id(),
                "door_id": self.active_door.door_id,
                "entry_waypoint_count": len(self.entry_portal_waypoints),
            })
            # The fixed canonical outside offset can become occupied after the
            # room scan updates the floor projection. The detector's interior
            # and corridor-side anchors are timestamped poses from the actual
            # online doorway crossing, so use that measured trace and let
            # SCAN-lite validate it again in 3-D.
            mandatory_exit = [
                [self.active_door.interior_side[0],
                 self.active_door.interior_side[1]],
                [self.active_door.center[0], self.active_door.center[1]],
                [self.active_door.corridor_side[0],
                 self.active_door.corridor_side[1]],
            ]
            exit_point = tuple(mandatory_exit[-1])
            exit_clearance = portal_clearance(grid, self.active_door, self.config)
        else:
            mandatory_exit = exit_portal["mandatory_portal_waypoints"]
            exit_clearance = exit_portal["portal_clearance_m"]
            exit_point = tuple(mandatory_exit[-1])
        # If the current raster cannot reproduce a portal that the robot just
        # traversed on ENTRY, reverse those timestamped entry anchors instead
        # of retrying the same newly inflated door-centre cell.  PlannerBase
        # receives this as a preplanned path, while SCAN-lite still validates
        # every execution waypoint in the manager.
        reverse_entry_fallback = bool(
            exit_portal is None and entry_trace_corridor_started)
        # A failed EXIT is evidence that the *newly rebuilt* portal raster is
        # no longer reproducible from the current interior pose.  Retrying
        # that same geometry twice consumed 20--30 seconds in the far rooms
        # before eventually taking the already verified reverse ENTRY trace.
        # Switch to that physically traversed trace after the first failed
        # exit instead.  It remains bounded, has passed the entry footprint
        # audit, and still ends at the separately checked corridor anchor.
        fallback_exit = bool(
            reverse_entry_fallback or self.exit_attempts >= 1)
        # The reverse ENTRY trace is physically traversed evidence. If an
        # attempt times out after making progress, retry it from the nearest
        # remaining breadcrumb instead of switching to a newly inflated
        # direct portal that may be disconnected.
        if fallback_exit:
            self.exit_fallback_issued = True
            # Frozen at ENTRY; never append a later deep-room pose here.
            # That straight chord to the door can cross room furniture.
            observed_entry = self.entry_portal_waypoints
            reversed_entry = truncate_reversed_entry_trace_at_corridor(
                list(reversed(observed_entry)), self.active_door)
            if self.exit_attempts >= 1:
                reversed_entry = resume_reversed_entry_trace(
                    reversed_entry, current)
            # The entry portal was physically traversed and footprint-audited
            # moments earlier.  Reversing it is more reliable than asking a
            # freshly inflated voxel raster to rediscover the same narrow
            # doorway after several exit-preflight failures.
            use_reversed_entry = len(reversed_entry) >= 2
            mandatory_exit = (reversed_entry if use_reversed_entry else
                              mandatory_exit)
            # The final reverse-ENTRY waypoint is the measured corridor-side
            # pose that the robot physically reached on entry.  Do not append
            # a newly rasterized outside offset here: after a room scan that
            # last 20--40 cm is exactly where stale occupancy can block an
            # otherwise valid recovery. Normal exits retain their canonical
            # outside endpoint; this exception applies only to one fallback.
            if (not use_reversed_entry and
                    _distance(mandatory_exit[-1], exit_point) > 0.15):
                mandatory_exit.append(list(exit_point))
            exit_point = tuple(mandatory_exit[-1])
        goal = {
            "position": [exit_point[0], exit_point[1], 0.0],
            "information_gain": 0.0, "score": 0.0,
            "source": ("lightweight_room_exit_fallback" if fallback_exit else
                       "lightweight_room_exit"),
            "room_role": "EXIT",
            "room_id": self._room_id(),
            "estimated_door_id": self.active_door.door_id,
            "room_exit_attempt": self.exit_attempts + 1,
            "execution_timeout_sec": remaining_budget,
            "progress_timeout_sec": self.config.exit_progress_timeout_seconds,
            "progressive_timeout": True,
            "breadcrumb_fallback": fallback_exit,
            "mandatory_portal_waypoints": mandatory_exit,
            "portal_clearance_m": exit_clearance,
            "portal_preflight_verified": exit_portal is not None,
        }
        if (fallback_exit and
                (len(self.entry_portal_waypoints) >= 2 or
                 (exit_portal is None and self.entry_confirmed))):
            # The manager's SCAN-lite refiner recognizes this flag and keeps
            # the already traversed/observed reverse doorway trace intact.
            # Some valid entries are promoted from local geometry before a
            # complete raster portal path exists, leaving
            # ``entry_portal_waypoints`` empty.  After one failed EXIT their
            # active-door anchors are still the best bounded recovery trace;
            # without this flag the manager re-plans the same newly occupied
            # door-centre cell until the retry limit terminates the mission.
            # This is limited to one post-failure EXIT recovery and is never
            # used for ordinary room motion or new room entry.
            goal["verified_trajectory_backtrack"] = True
            goal["observed_portal_recovery"] = bool(
                not self.entry_portal_waypoints)
        if exit_portal is not None and exit_portal.get("preflight_path"):
            # Execute the exact current -> inside alignment -> door centre ->
            # corridor path that just passed occupancy preflight. Rebuilding
            # it later from mission-wide breadcrumbs caused every retry to
            # grow and re-run already completed room motion.
            goal["_preplanned_path_result"] = {
                "success": True,
                "reason": "centered_exit_preflight_path",
                "path": list(exit_portal.get("preflight_path") or []),
            }
        elif exit_portal is None:
            trace = [(float(current[0]), float(current[1]))]
            trace.extend((float(point[0]), float(point[1]))
                         for point in mandatory_exit)
            goal["_preplanned_path_result"] = {
                "success": True,
                "reason": "observed_doorway_trace_exit_fallback",
                "path": trace,
            }
            goal["observed_doorway_trace_fallback"] = True
        if fallback_exit and len(self.entry_portal_waypoints) >= 2:
            preplanned = [(float(current[0]), float(current[1]))]
            preplanned.extend((float(point[0]), float(point[1]))
                              for point in mandatory_exit)
            goal["_preplanned_path_result"] = {
                "success": True,
                "reason": "reversed_physically_traversed_entry_portal",
                "path": preplanned,
            }
            goal["reversed_entry_portal_fallback"] = True
        self.selected_goals.append(dict(goal, planned_at=round(float(now), 3)))
        return goal

    def record_result(self, goal: dict, success: bool, final_point: Optional[Point],
                      now: float, reason: Optional[str] = None) -> dict:
        if self.active_door is None or not str(goal.get("source", "")).startswith(
                "lightweight_room_"):
            return {"success": bool(success), "reason": reason,
                    "reported_success": bool(success),
                    "role": goal.get("room_role")}
        role = goal.get("room_role")
        effective_success = bool(success)
        effective_reason = reason
        if role in ("G1", "G2", "G3", "G4") and final_point is not None:
            target = goal.get("position") or []
            target_neighborhood_reached = bool(
                len(target) >= 2 and
                _distance(final_point, (float(target[0]), float(target[1]))) <=
                self.config.semantic_completion_tolerance)
            depth_valid = self.active_door.depth(final_point) >= 1.10
            lateral = self.active_door.lateral(final_point)
            side_valid = (role in ("G1", "G2") or
                          (role == "G3" and lateral >= 0.55) or
                          (role == "G4" and lateral <= -0.55))
            if not success and target_neighborhood_reached and depth_valid and side_valid:
                effective_success = True
                effective_reason = "semantic_observation_neighborhood_reached"
        elif role == "ENTRY":
            # A door-centre target can be safely shortened as the occupancy
            # map is updated during approach.  Rejecting such a target unless
            # it reaches the full interior-view depth incorrectly treats a
            # real doorway crossing as a failed entry (opt9 crossed 0.21 m
            # and was rejected by the previous 1.10 m-only condition).
            # The subsequent G1/G3/G4 goals still demand their normal deep
            # room observations; this only confirms the portal crossing.
            depth_confirmed = bool(
                final_point is not None and
                self.active_door.depth(final_point) >=
                self.config.minimum_crossing_depth)
            target = goal.get("position") or []
            # A progress-timeout can leave the robot somewhere else along a
            # long doorway approach while its projected point happens to lie
            # on the estimated room side of the door plane.  That is not a
            # crossing: accept a failed controller result only when the final
            # pose is also close to the commanded inside staging point.
            entry_target_reached = bool(
                final_point is not None and len(target) >= 2 and
                _distance(final_point, (float(target[0]), float(target[1]))) <=
                self.config.semantic_completion_tolerance)
            if success and not depth_confirmed:
                effective_success = False
                effective_reason = "entry_depth_not_confirmed"
            elif not success and depth_confirmed and entry_target_reached:
                effective_success = True
                effective_reason = "entry_confirmed_by_final_pose"
        elif role == "RETURN":
            target = goal.get("position") or []
            # The RETURN target may be relocated to a nearby observed-free
            # cell when the original interior anchor becomes inflated after
            # the room scan.  Requiring proximity only to the stale geometric
            # anchor turns a successfully executed safe return into
            # portal_anchor_not_reached.  Reaching the actual commanded target
            # is equivalent evidence that the return phase completed; EXIT is
            # still checked separately for a real door-plane crossing.
            returned = bool(
                final_point is not None and (
                    _distance(final_point,
                              self.active_door.interior_side) <= 0.55 or
                    (len(target) >= 2 and
                     _distance(final_point,
                               (float(target[0]), float(target[1]))) <=
                     self.config.semantic_completion_tolerance)))
            if success and not returned:
                effective_success = False
                effective_reason = "portal_anchor_not_reached"
            elif not success and returned:
                effective_success = True
                effective_reason = "portal_anchor_reached_by_final_pose"
        elif role == "EXIT":
            # The manager can apply a stronger live-corridor check after the
            # executor reaches an EXIT endpoint.  Never undo that rejection
            # merely because the same endpoint crossed the *estimated* door
            # plane: run67's oblique room-interior opening satisfied this
            # local depth test while still ending 2.7 m inside Room 3.
            strict_corridor_rejection = effective_reason in (
                "exit_raw_corridor_geometry_unconfirmed",
                "exit_outside_established_corridor_band",
                "exit_corridor_map_or_pose_unavailable",
            )
            crossed_to_corridor = bool(
                final_point is not None and
                # The corridor target is 0.60-0.90 m beyond the estimated
                # door plane, but the executor may legally stop 0.15 m from
                # it and the 0.15 m occupancy raster jitters the re-estimated
                # plane.  Run12 physically reached -0.3945 m and was rejected
                # by the old -0.400 m threshold.  A full 0.30 m body-centre
                # crossing remains well outside this tolerance envelope.
                self.active_door.depth(final_point) <= -0.30)
            target = goal.get("position") or []
            entry_trace_start = (self.entry_portal_waypoints[0]
                                 if self.entry_portal_waypoints else None)
            reversed_entry_trace_exit = bool(
                self.config.accept_reversed_entry_trace_exit and
                success and final_point is not None and len(target) >= 2 and
                bool(goal.get("verified_trajectory_backtrack")) and
                bool(goal.get("reversed_entry_portal_fallback")) and
                entry_trace_start is not None and
                self.active_door.depth(entry_trace_start) <= 0.0 and
                _distance(final_point,
                          (float(target[0]), float(target[1]))) <=
                self.config.reversed_entry_trace_exit_tolerance and
                _distance((float(target[0]), float(target[1])),
                          entry_trace_start) <= 0.15)
            if success and not (crossed_to_corridor or
                                reversed_entry_trace_exit):
                effective_success = False
                effective_reason = "corridor_side_not_confirmed"
            elif success and reversed_entry_trace_exit and not crossed_to_corridor:
                effective_reason = "corridor_exit_confirmed_by_reversed_entry_trace"
            elif (not success and crossed_to_corridor and
                  not strict_corridor_rejection):
                effective_success = True
                effective_reason = "corridor_exit_confirmed_by_final_pose"
        self.events.append({
            "event": "ROOM_GOAL_RESULT", "elapsed_sec": round(float(now), 3),
            "room_id": goal.get("room_id"), "door_id": self.active_door.door_id,
            "role": role, "goal": goal.get("position"),
            "success": effective_success,
            "reported_execution_success": bool(success),
            "reason": effective_reason,
            "execution_progress": goal.get("execution_result"),
        })
        if (effective_success and role in ("G1", "G2", "G3", "G4") and
                final_point is not None):
            self.completed_points.append((float(final_point[0]), float(final_point[1])))
            self.successful_roles.add(str(role))
            if role == "G1":
                self.visual_anchor_completed = True
        if role in ("G1", "G2", "G3", "G4") and not effective_success:
            # The remaining order was optimized from the preceding pose.
            # Recompute it after any execution/SCAN failure instead of blindly
            # following a now-invalid suffix.
            self.adaptive_route_queue = []
            attempts = self.role_attempts.get(str(role), 0)
            limit = (2 if role in ("G1", "G2") else
                     self.config.side_goal_retry_limit)
            if (not self.config.adaptive_minimal_viewpoints and
                    attempts < limit):
                role_index = {"G1": 0, "G3": 1, "G4": 2}[str(role)]
                self.next_role_index = min(self.next_role_index, role_index)
                self.events.append({
                    "event": "ROOM_GOAL_RETRY_SCHEDULED",
                    "elapsed_sec": round(float(now), 3),
                    "room_id": goal.get("room_id"),
                    "door_id": self.active_door.door_id,
                    "role": role,
                    "next_attempt": attempts + 1,
                    "fallback_candidate": True,
                })
            else:
                self.failed_roles.add(str(role))
        if role == "ENTRY":
            if effective_success:
                # Freeze partial-path progress while this pose still belongs
                # to ENTRY, before room observation moves away from the trace.
                if (self.entry_portal_waypoints and
                        self.config.entry_trace_observed_progress_only and
                        final_point is not None):
                    self.entry_portal_waypoints = (
                        _path_prefix_to_observed_point(
                            self.entry_portal_waypoints, final_point))
                # A local-door staging fallback has no short portal anchors,
                # but it does carry the complete A* path that began at the
                # physical corridor and was just executed into the room.
                # Preserve that path as the bounded reverse EXIT trace.  The
                # old code discarded it and, after a false EXIT confirmation,
                # could only retry the same interior door-plane endpoint.
                if not self.entry_portal_waypoints:
                    diagnostic = goal.get("room_goal_diagnostic") or {}
                    staging = diagnostic.get(
                        "entry_staging_path_result") or {}
                    traversed = []
                    for point in staging.get("path") or []:
                        if (not isinstance(point, (list, tuple)) or
                                len(point) < 2):
                            continue
                        try:
                            x, y = float(point[0]), float(point[1])
                        except (TypeError, ValueError):
                            continue
                        if not (math.isfinite(x) and math.isfinite(y)):
                            continue
                        if (not traversed or
                                _distance(traversed[-1], (x, y)) > 0.02):
                            traversed.append([x, y])
                    planned_waypoint_count = len(traversed)
                    if (self.config.entry_trace_observed_progress_only and
                            final_point is not None):
                        traversed = _path_prefix_to_observed_point(
                            traversed, final_point)
                    if len(traversed) >= 2:
                        self.entry_portal_waypoints = traversed
                        self.events.append({
                            "event": "ROOM_ENTRY_TRAVERSAL_CAPTURED",
                            "elapsed_sec": round(float(now), 3),
                            "room_id": goal.get("room_id"),
                            "door_id": self.active_door.door_id,
                            "waypoint_count": len(traversed),
                            "planned_waypoint_count": planned_waypoint_count,
                            "observed_progress_only": bool(
                                self.config.entry_trace_observed_progress_only),
                            "source": "entry_staging_astar_path",
                        })
                self._confirm_entry(final_point, now, "entry_goal_success")
                self._set_state("G1_CENTER", now, "entry_confirmed")
            else:
                self.entry_attempts += 1
                if self.entry_attempts < self.config.entry_retry_limit:
                    self.next_role_index = 0
                    self._set_state("DOOR_COMMIT", now, "entry_retry")
                else:
                    key = self._set_door_cooldown(
                        self.active_door.center,
                        float(now) + self.config.door_cooldown_seconds)
                    self.events.append({
                        "event": "DOOR_COMMIT_FAILED",
                        "elapsed_sec": round(float(now), 3),
                        "room_id": goal.get("room_id"),
                        "door_id": self.active_door.door_id,
                        "cooldown_until": self.door_cooldowns[key],
                    })
                    self._clear_active()
                    self._set_state("CORRIDOR_SWEEP", now, "door_cooldown")
        if role == "RETURN":
            self.return_attempts += int(not effective_success)
            if effective_success:
                self.return_completed = True
                self.events.append({
                    "event": "ROOM_RETURNED_TO_PORTAL",
                    "elapsed_sec": round(float(now), 3),
                    "room_id": goal.get("room_id"),
                    "door_id": self.active_door.door_id,
                    "attempts": self.return_attempts + 1,
                })
            else:
                self.events.append({
                    "event": "ROOM_RETURN_FAILED",
                    "elapsed_sec": round(float(now), 3),
                    "room_id": goal.get("room_id"),
                    "door_id": self.active_door.door_id,
                    "attempts": self.return_attempts,
                    "reason": effective_reason,
                })
        if role == "EXIT":
            self.active_door.visited = bool(effective_success)
            required_roles = {"G1", "G3", "G4"}
            coverage_complete = bool(
                self.completion_mode == "lidar_coverage" and
                self.coverage_status and
                self.coverage_status.get("coverage_complete"))
            # Occupancy rasterisation leaves a thin band of unknown cells at
            # wall/obstacle silhouettes even after a room is effectively
            # covered.  Treat only a *bounded*, high-ratio residual as
            # complete.  This is map-derived evidence (not room metadata) and
            # prevents a verified fourth-room EXIT from being followed by
            # repeated corridor rescans instead of the F1 return handoff.
            coverage = self.coverage_status or {}
            near_complete_coverage = bool(
                coverage.get("boundary_complete") and
                float(coverage.get("lidar_ratio", 0.0) or 0.0) >= 0.985 and
                float(coverage.get("largest_unknown_m2", float("inf")) or float("inf")) <= 1.0 and
                float(coverage.get("largest_shadow_m2", float("inf")) or float("inf")) <= 0.9)
            coverage_complete = bool(coverage_complete or near_complete_coverage)
            self.active_door.coverage_complete = bool(
                effective_success and coverage_complete)
            self.active_door.completed = bool(
                effective_success and (
                    coverage_complete or
                    required_roles.issubset(self.successful_roles)))
            self.active_door.temporarily_failed = bool(
                effective_success and not self.active_door.completed)
            if effective_success and not self.active_door.completed:
                # Leave safely, then continue the corridor instead of
                # immediately reacquiring the same incomplete doorway.
                self._set_door_cooldown(
                    self.active_door.center,
                    float(now) + self.config.door_cooldown_seconds)
            self.exit_attempts += int(not effective_success)
            self.events.append({
                "event": "ROOM_EXITED" if effective_success else "ROOM_EXIT_FAILED",
                "elapsed_sec": round(float(now), 3),
                "room_id": goal.get("room_id"), "door_id": self.active_door.door_id,
                "room_elapsed_sec": (round(float(now) - self.room_started_at, 3)
                                     if self.room_started_at is not None else None),
                "completed_observation_count": len(self.completed_points),
                "successful_roles": sorted(self.successful_roles),
                "required_roles": sorted(required_roles),
                "failed_roles": sorted(self.failed_roles),
                "room_complete": self.active_door.completed,
                "completion_mode": self.completion_mode or "incomplete",
                "coverage": (dict(self.coverage_status)
                             if self.coverage_status is not None else None),
                "near_complete_coverage_accepted": bool(
                    near_complete_coverage),
                "door_plane_depth": (round(self.active_door.depth(final_point), 3)
                                     if final_point is not None else None),
                "corridor_exit_confirmed": effective_success,
                "breadcrumb_fallback": bool(goal.get("breadcrumb_fallback")),
                "final_point": ([round(float(final_point[0]), 4),
                                 round(float(final_point[1]), 4)]
                                if final_point is not None else None),
            })
            if effective_success:
                self._clear_active()
                self._set_state("CORRIDOR_RESUME", now, "room_completed")
            elif self.exit_attempts >= self.config.exit_retry_limit:
                self.events.append({
                    "event": "ROOM_EXIT_BLOCKED",
                    "elapsed_sec": round(float(now), 3),
                    "room_id": goal.get("room_id"),
                    "door_id": self.active_door.door_id,
                    "attempts": self.exit_attempts,
                    "fallback_pending": not self.exit_fallback_issued,
                    "action": "abandon_door_cooldown_then_corridor_resume",
                })
                # The fallback is already the physically traversed entry
                # trace.  If the bounded retry limit is exhausted, do not
                # emit the same unreachable portal goal every scheduler
                # cycle: it previously consumed the whole mission budget in
                # one room.  Instead of terminating the whole mission with
                # the ROOM_EXIT_BLOCKED state, abandon this door (it already
                # carries the 90 s cooldown set on exit failure) and resume
                # corridor exploration so other rooms can still be visited.
                # Every door failing the same way degrades into the normal
                # corridor-exhaustion/before-F1 close-out.
                self.exit_blocked = True
                self._clear_active()
                self._set_state(
                    "CORRIDOR_SWEEP", now,
                    "exit_retry_limit_exhausted_abandon")
        return {"success": bool(effective_success),
                "reason": effective_reason,
                "reported_success": bool(success), "role": role}

    def snapshot(self) -> dict:
        return {
            "schema": "simenv_lightweight_room_recognition_v5",
            "config": asdict(self.config),
            "online_inputs": ["FAST-LIO trajectory", "observed occupancy grid"],
            "truth_used_online": False,
            "room_count": self.room_count,
            "exited_room_count": sum(door.visited for door in self.detector.doors),
            "complete_room_count": sum(door.completed for door in self.detector.doors),
            "active_door_id": (self.active_door.door_id
                               if self.active_door is not None else None),
            "active_room_id": self.active_room_id,
            "entry_confirmed": self.entry_confirmed,
            "completion_mode": self.completion_mode,
            "coverage_status": (dict(self.coverage_status)
                                if self.coverage_status is not None else None),
            "successful_roles": sorted(self.successful_roles),
            "failed_roles": sorted(self.failed_roles),
            "return_completed": self.return_completed,
            "exit_attempts": self.exit_attempts,
            "exit_fallback_issued": self.exit_fallback_issued,
            "exit_blocked": self.exit_blocked,
            "state": self.state,
            "room_started_at": self.room_started_at,
            "first_room_started_at": self.first_room_started_at,
            "pending_door_candidates": len(self.pending_candidates),
            "door_cooldowns": [
                {"key": list(key), "until": until,
                 "center": list(self.door_cooldown_centers.get(key, ())) }
                for key, until in self.door_cooldowns.items()],
            "doors": [door.to_dict() for door in self.detector.doors],
            "selected_goals": self.selected_goals,
            "events": self.events,
        }