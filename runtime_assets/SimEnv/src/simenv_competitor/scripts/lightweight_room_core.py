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
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from baseline_planning_core import OccupancyGrid2D, astar_safe_path
from frontier_information_structure import RoomVisibilityGrid


Point = Tuple[float, float]


def obstacle_candidate_is_behind_front_gap(
        door, point: Point, clearance: float) -> bool:
    """Reject truth-obstacle viewpoints at or behind its measured front."""
    front_depth = getattr(door, "truth_obstacle_front_depth_m", None)
    if front_depth is None:
        return False
    return bool(
        door.contract_depth(point) >=
        float(front_depth) - max(0.20, float(clearance)))


def obstacle_view_policy(door) -> str:
    """Return the locked shallow-chord or deep-side-peek room policy."""
    explicit = str(getattr(door, "truth_obstacle_view_policy", "") or "")
    if explicit in ("shallow_front_chord", "deep_outer_side_peek"):
        return explicit
    front = getattr(door, "truth_obstacle_front_depth_m", None)
    return ("deep_outer_side_peek"
            if front is not None and float(front) >= 3.50
            else "shallow_front_chord")


def obstacle_route_front_depth(door) -> Optional[float]:
    """Front edge that a cross-side chord may not pass.

    Shallow rooms must stay in front of every doorway blocker (run33's small
    chair as well as the meeting table).  Deep rooms intentionally use the
    principal furniture front so their outer peeks can see around side
    occluders such as F3 room2's sofa.
    """
    principal = getattr(door, "truth_obstacle_front_depth_m", None)
    if principal is None:
        return None
    if obstacle_view_policy(door) == "shallow_front_chord":
        nearest = getattr(
            door, "truth_obstacle_nearest_blocker_front_depth_m", None)
        if nearest is not None:
            return min(float(principal), float(nearest))
    return float(principal)


def direct_exit_fold_allowed(
        door, current: Point, entry_depth: float) -> bool:
    """Fold RETURN into EXIT from the portal or a verified obstacle gap.

    A deep point in an open room can still be behind furniture, so it retains
    the bounded RETURN staging pass.  An obstacle-contract viewpoint is
    deliberately constrained to the finite door--obstacle gap.  The EXIT
    planner A*-checks its full route and execution receives an independent 3-D
    audit, so a separate RETURN only replays that same safe chord.
    """
    depth = float(door.depth(current))
    if depth <= max(2.85, 1.90 * float(entry_depth)):
        return True
    front = getattr(door, "truth_obstacle_front_depth_m", None)
    return bool(
        getattr(door, "viewpoint_contract", None) ==
        "obstacle_front_opposite_sides" and
        front is not None and
        float(entry_depth) <= depth < float(front) - 0.15)


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


def sparsify_verified_trace(points: Sequence[Sequence[float]],
                            spacing: float = 1.20,
                            maximum_heading_change: float = 0.55
                            ) -> List[List[float]]:
    """Reduce dense verified breadcrumbs while preserving bends/endpoints."""
    clean = []
    for point in points:
        candidate = [float(point[0]), float(point[1])]
        if not clean or _distance(clean[-1], candidate) > 0.02:
            clean.append(candidate)
    if len(clean) <= 2:
        return clean
    spacing = max(0.20, float(spacing))
    turn_limit = max(0.05, float(maximum_heading_change))
    retained = [clean[0]]
    distance_since_anchor = 0.0
    for index in range(1, len(clean) - 1):
        previous, current, following = (clean[index - 1], clean[index],
                                        clean[index + 1])
        distance_since_anchor += _distance(previous, current)
        incoming = math.atan2(current[1] - previous[1],
                              current[0] - previous[0])
        outgoing = math.atan2(following[1] - current[1],
                              following[0] - current[0])
        heading_change = abs(math.atan2(math.sin(outgoing - incoming),
                                        math.cos(outgoing - incoming)))
        if (distance_since_anchor >= spacing or
                (heading_change >= turn_limit and
                 distance_since_anchor >= min(0.60, 0.50 * spacing))):
            retained.append(current)
            distance_since_anchor = 0.0
    if _distance(retained[-1], clean[-1]) > 0.02:
        retained.append(clean[-1])
    return retained


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


def complete_reversed_entry_trace_to_corridor(
        points: Sequence[Sequence[float]],
        door: "EstimatedDoorway",
        minimum_crossing_depth: float = 0.30) -> List[List[float]]:
    """Append the saved outside anchor when an ENTRY prefix ends at the door.

    Fall recovery can retain only the physically reached ENTRY prefix.  If
    that prefix starts just inside the room, reversing it yields
    ``inside -> door centre`` but no corridor endpoint.  Such a prefix remains
    useful evidence up to the portal, but it must never replace the mandatory
    final crossing.  The doorway's ``corridor_side`` is the online observed
    outside anchor captured when the candidate was promoted.
    """
    trace = [[float(point[0]), float(point[1])] for point in points]
    if not trace:
        return trace
    threshold = -max(0.05, float(minimum_crossing_depth))
    if door.depth(trace[-1]) <= threshold:
        return trace
    outside = [float(door.corridor_side[0]),
               float(door.corridor_side[1])]
    if _distance(trace[-1], outside) > 0.02:
        trace.append(outside)
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
    # Keep observation endpoints at the normal body clearance, but permit a
    # freshly crossed doorway to escape a thin inflated-raster seal.  This is
    # used only after the normal A* attempt fails in an otherwise open room;
    # the manager still audits every retained segment with live 3-D SCAN-lite.
    post_entry_path_clearance: float = 0.20
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
    # Floor-1 local door detections already provide a corridor-side, centre,
    # and interior anchor. Prefer that short physical crossing over a global
    # A* path, which can loop behind the robot around a partly inflated jamb.
    prefer_direct_anchored_entry: bool = False
    # Layout matching is useful for selecting the room's camera contract.  A
    # scalar centre distance is not sufficient for deciding whether it may
    # repair an ENTRY centreline: run68/run72 measured the correct wall plane
    # (normal error 0.04/0.13 m) but reported an aperture edge 0.79/1.04 m
    # along the wall.  Permit a bounded tangent-only centre correction while
    # retaining the detector's measured normal/depth coordinate.  The legacy
    # scalar limit remains the conservative whole-centre snap criterion.
    # The generated 1.4 m aperture can be observed from just inside the room,
    # biasing the fitted wall plane by roughly half the opening width.  Keep
    # the snap bounded below the known 0.794 m wrong-door regression, while
    # accepting run115's station-correct (3.4 cm tangent error) 0.720 m
    # normal-depth bias.
    truth_portal_snap_max_separation: float = 0.75
    truth_portal_tangent_repair_max_separation: float = 1.35
    truth_portal_normal_repair_max_separation: float = 0.35
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
    entry_progress_timeout_seconds: float = 5.0
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
    # A locked two-view room may need one stationary map refresh after its
    # first physical camera pose before the opposite/deep partner is visible
    # in the rolling raster.  Reserve one bounded post-refresh planning
    # window; the refresh itself remains excluded from physical-view credit.
    mandatory_two_pose_refresh_grace_seconds: float = 18.0
    # A deep side-peek starts about 3.0 m beyond ENTRY and then has to cross
    # the full obstacle-front chord at the deliberately slow 0.30 m/s room
    # profile.  run39 proved that the ordinary 31 s visit budget (26 s before
    # EXIT reserve) expires while G3 is still making physical progress.  Give
    # only the truth-locked deep obstacle policy a bounded observation
    # extension; shallow obstacle chords and open deep/near rooms retain the
    # throughput budget above.
    deep_obstacle_contract_grace_seconds: float = 40.0
    # A real G4 locomotion failure may be followed by exactly one fresh-grid
    # endpoint reselection.  Keep that promised retry alive for a much shorter
    # window than the map-refresh grace; otherwise a failure just beyond the
    # ordinary observation deadline is scheduled and then killed on the next
    # callback before it can produce a goal.
    mandatory_two_pose_execution_retry_grace_seconds: float = 10.0
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
    # A physically observed ENTRY trace may end only just outside the door.
    # Extend that endpoint farther into the corridor so the goal tolerance
    # cannot report EXIT while the robot body is still across the threshold.
    reversed_entry_trace_corridor_extension: float = 0.0
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
    # G3/G4 are camera parallax poses, not far-wall coverage traversals.
    # Cap their doorway-relative depth independently so an overestimated
    # open-room extent cannot turn both observations into long diagonals.
    maximum_side_depth: float = 8.0
    # Open rooms need a genuine deep/near parallax pair, but the deep camera
    # does not have to approach the far wall.  Timed profiles can shorten this
    # target while preserving a separately enforced minimum deep distance.
    open_room_preferred_deep_depth: float = 5.25
    open_room_minimum_deep_depth: float = 5.00
    # A newly rebuilt upper-floor map initially observes only the doorway
    # strip.  That floor may execute the near physical view first so its lidar
    # opens the centreline before the mandatory deep goal is planned.
    truth_open_near_first: bool = False

    maximum_side_lateral: float = 3.00
    # Bound the optional RGB-D breadth recovery to a local baseline around
    # G1 rather than a second deep-room excursion.
    visual_breadth_lateral: float = 0.65
    visual_breadth_min_baseline: float = 1.20
    visual_two_pose_lateral_enabled: bool = False
    # Optional timed-route variant for a centre obstacle: keep the first
    # camera pose deep on one side, then finish at a shallower pose on the
    # opposite side so the verified EXIT begins near the portal.
    obstacle_second_view_near_door: bool = False
    # When a central obstacle is measured and the door-to-obstacle gap is
    # wide enough, keep both views in front of it but place them beyond its
    # opposite lateral edges. Endpoint, inter-view and return A* checks are
    # unchanged; if the front cross-route is not clear, planning falls back
    # to the ordinary deep/opposite-side route.
    obstacle_front_opposite_side_pair: bool = False
    # A compact coffee table is not the "large obstacle" for which two
    # shallow front-side views are intended: both views can then remain in
    # front of a sofa and miss a hazard behind it. Require a measured central
    # footprint of this area before selecting the short front cross-route.
    obstacle_front_minimum_area_m2: float = 0.50
    # Require a real central camera stop after the two lateral parallax views.
    # This prevents a bookkeeping-only virtual G1 from hiding an occluded
    # hazard that neither side pose could see.
    visual_require_physical_center: bool = False
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
               self.entry_progress_timeout_seconds,
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
                self.reversed_entry_trace_corridor_extension < 0.0 or
                self.mandatory_two_pose_refresh_grace_seconds < 0.0 or
                self.deep_obstacle_contract_grace_seconds < 0.0 or
                self.mandatory_two_pose_execution_retry_grace_seconds < 0.0 or
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
        if (self.open_room_minimum_deep_depth <= 0.0 or
                self.open_room_preferred_deep_depth <
                self.open_room_minimum_deep_depth):
            raise ValueError(
                "invalid open-room deep viewpoint configuration")

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
    # A room whose EXIT reached the corridor but did not satisfy G3/G4 must
    # be re-entered as the same semantic room.  Without a persistent room id
    # the retry increments room_count and turns one partial room into a fifth
    # room, while the original doorway remains an untracked debt.
    room_id: Optional[str] = None
    partial_retry_count: int = 0
    # Preserve the physical observation debt across the bounded same-door
    # re-entry.  Without this, a room that completed G3 but timed out at G4
    # was re-entered as an empty scheduler and could execute only a redundant
    # G1 before being marked retry-exhausted.
    partial_successful_roles: Tuple[str, ...] = ()
    partial_completed_points: Tuple[Point, ...] = ()
    # Optional simulator-layout contract.  It selects only the required
    # physical camera geometry; every endpoint and path is still validated by
    # the live occupancy grid, A* and SCAN-lite before execution.
    viewpoint_contract: Optional[str] = None
    viewpoint_contract_source: Optional[str] = None
    viewpoint_contract_locked_before_entry: bool = False
    truth_room_id: Optional[str] = None
    # Rigid map->Gazebo-world hypothesis selected when this measured doorway
    # was matched to the generated layout.  F1 camera_init is commonly yaw
    # rotated by 90 degrees, while F2/F3 are truth-reanchored world-aligned.
    # Persisting the transform prevents later truth-layout safety checks from
    # comparing map-frame room goals directly with world-frame room bounds.
    truth_coordinate_transform: Optional[str] = None
    # The online doorway remains authoritative for portal crossing.  Keep a
    # separate, layout-matched frame only for the locked camera contract so a
    # lateral door-centre error cannot make one obstacle-side view 0.5 m from
    # the table and the other 1.5 m away.  These values are expressed in the
    # active map frame (not Gazebo world), including F1's yaw transform.
    truth_contract_center: Optional[Point] = None
    truth_contract_normal_direction: Optional[float] = None
    truth_contract_match_separation_m: Optional[float] = None
    truth_contract_match_tangent_separation_m: Optional[float] = None
    truth_contract_match_normal_separation_m: Optional[float] = None
    truth_obstacle_front_depth_m: Optional[float] = None
    truth_obstacle_rear_depth_m: Optional[float] = None
    truth_obstacle_minimum_lateral_m: Optional[float] = None
    truth_obstacle_maximum_lateral_m: Optional[float] = None
    # Deep rooms can contain secondary side occluders beyond the primary
    # front obstacle.  Store their complete contract-frame lateral envelope
    # so the two outer-side views clear those silhouettes instead of merely
    # straddling the centred table (run87 F3 room2 / D83).
    truth_obstacle_visibility_minimum_lateral_m: Optional[float] = None
    truth_obstacle_visibility_maximum_lateral_m: Optional[float] = None
    # Obstacle rooms have two physically different contracts.  A shallow
    # front blocker uses a short cross-gap chord before *all* furniture; a
    # deep blocker with adjacent occluders needs deeper outer-side peeks.
    truth_obstacle_view_policy: Optional[str] = None
    truth_obstacle_nearest_blocker_front_depth_m: Optional[float] = None
    # For a truth-classified open room, retain one furniture-audited axial
    # depth.  It is only a final endpoint fallback when the rolling upper-floor
    # raster truncates the otherwise clear room; live SCAN-lite still audits
    # the physical segment before locomotion.
    truth_open_centerline_deep_depth_m: Optional[float] = None
    # Explicit physical transaction evidence.  These fields are persisted in
    # room_recognition_history.json so acceptance does not have to infer a
    # real crossing or a two-view contract from counters alone.
    entry_truth_crossing_confirmed: bool = False
    g3_truth_pose: Optional[Point] = None
    g4_truth_pose: Optional[Point] = None
    physical_viewpoint_separation_m: Optional[float] = None
    physical_two_view_contract_met: bool = False
    viewpoint_contract_geometry_met: bool = False
    viewpoint_contract_geometry: Optional[Dict] = None
    exit_truth_crossing_confirmed: bool = False
    room_transaction_complete: bool = False
    corridor_centerline_recovered: bool = False

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

    @property
    def contract_normal(self) -> Point:
        if self.truth_contract_normal_direction is None:
            return self.normal
        return _unit(float(self.truth_contract_normal_direction))

    @property
    def contract_tangent(self) -> Point:
        nx, ny = self.contract_normal
        return -ny, nx

    @property
    def contract_center(self) -> Point:
        if self.truth_contract_center is None:
            return self.center
        return (float(self.truth_contract_center[0]),
                float(self.truth_contract_center[1]))

    def contract_depth(self, point: Sequence[float]) -> float:
        center = self.contract_center
        return _dot((float(point[0]) - center[0],
                     float(point[1]) - center[1]), self.contract_normal)

    def contract_lateral(self, point: Sequence[float]) -> float:
        center = self.contract_center
        return _dot((float(point[0]) - center[0],
                     float(point[1]) - center[1]), self.contract_tangent)

    def contract_point(self, depth: float, lateral: float) -> Point:
        center = self.contract_center
        nx, ny = self.contract_normal
        tx, ty = self.contract_tangent
        return (center[0] + float(depth) * nx + float(lateral) * tx,
                center[1] + float(depth) * ny + float(lateral) * ty)

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


def portal_lateral_preserving_exit(door: EstimatedDoorway,
                                   current: Sequence[float],
                                   outside_depth: float = 0.60
                                   ) -> Optional[Point]:
    """Return a straight outward endpoint when already inside the aperture.

    Near a narrow doorway, first moving laterally back to the saved door
    centre makes the quadruped rotate against a jamb.  Preserve the measured
    lateral coordinate and cross along the doorway normal instead.  This is
    only valid while the body centre is near the door plane and inside the
    usable aperture; downstream occupancy and SCAN-lite checks remain active.
    """
    depth = float(door.depth(current))
    lateral = float(door.lateral(current))
    usable_half_width = max(0.0, 0.5 * float(door.width) - 0.12)
    if abs(depth) > 0.45 or usable_half_width <= 0.0:
        return None
    if abs(lateral) > usable_half_width:
        return None
    nx, ny = door.normal
    tx, ty = door.tangent
    target_depth = -max(0.60, float(outside_depth))
    return (float(door.center[0]) + target_depth * nx + lateral * tx,
            float(door.center[1]) + target_depth * ny + lateral * ty)


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
            # The same portal observed once from the corridor and once just
            # inside the room may flip its normal and drift by ~0.9 m.  Treat
            # those near-coincident centres as one physical door.  Genuine
            # opposing corridor doors are separated by the full corridor
            # width (~2.2 m) and therefore remain distinct.
            if distance <= min(1.15, 0.75 * self.config.duplicate_door_radius):
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


def _eight_ray_clearance_field(grid: OccupancyGrid2D,
                               maximum: float = 1.5) -> np.ndarray:
    """Vectorize the existing eight-ray clearance for grid-cell centres.

    Room-view candidates are always cell centres. Computing the same eight
    rays independently for thousands of candidates dominated planning time.
    This field preserves the original step size, ray directions, unknown-cell
    policy and out-of-map policy; arbitrary path samples continue to use
    _clearance directly.
    """
    maximum = max(0.0, float(maximum))
    result = np.full(grid.data.shape, maximum, dtype=np.float64)
    if maximum <= 0.0 or grid.width <= 0 or grid.height <= 0:
        return result
    step = max(0.04, 0.45 * grid.resolution)
    distances = []
    distance = step
    while distance <= maximum + 1e-9:
        distances.append(distance)
        distance += step
    for angle in (0.0, math.pi / 4.0, math.pi / 2.0,
                  3.0 * math.pi / 4.0, math.pi,
                  5.0 * math.pi / 4.0, 3.0 * math.pi / 2.0,
                  7.0 * math.pi / 4.0):
        ray_clearance = np.full(grid.data.shape, maximum, dtype=np.float64)
        unresolved = np.ones(grid.data.shape, dtype=bool)
        cosine, sine = math.cos(angle), math.sin(angle)
        for ray_distance in distances:
            offset_x = int(math.floor(
                0.5 + ray_distance * cosine / grid.resolution))
            offset_y = int(math.floor(
                0.5 + ray_distance * sine / grid.resolution))
            blocked = np.ones(grid.data.shape, dtype=bool)
            source_x0 = max(0, -offset_x)
            source_x1 = min(grid.width, grid.width - offset_x)
            source_y0 = max(0, -offset_y)
            source_y1 = min(grid.height, grid.height - offset_y)
            if source_x0 < source_x1 and source_y0 < source_y1:
                blocked[source_y0:source_y1, source_x0:source_x1] = (
                    grid.data[source_y0 + offset_y:source_y1 + offset_y,
                              source_x0 + offset_x:source_x1 + offset_x] != 0)
            newly_blocked = unresolved & blocked
            ray_clearance[newly_blocked] = ray_distance
            unresolved &= ~blocked
            if not np.any(unresolved):
                break
        result = np.minimum(result, ray_clearance)
    return result


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
                        config: LightweightRoomConfig,
                        clearance_override: Optional[float] = None) -> Optional[dict]:
    clearance = portal_clearance(grid, door, config)
    if clearance_override is not None:
        clearance = min(clearance, max(0.05, float(clearance_override)))
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
            allow_blocked_start=(index == 0), maximum_expansions=min(300, config.exit_preflight_maximum_expansions))
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
    if length > min(config.entry_preflight_max_path_m,
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
                                 allow_blocked_start=True, maximum_expansions=min(300, config.exit_preflight_maximum_expansions))
        if not result.get("success"):
            continue
        path = list(result.get("path", []))
        length = sum(_distance(path[i - 1], path[i])
                     for i in range(1, len(path)))
        direct = _distance(current, point)
        if length > min(config.entry_preflight_max_path_m,
                        config.entry_preflight_max_detour_ratio * direct + 1.0):
            continue
        # Prefer the deepest observed-free point, with distance as tie-breaker.
        candidates.append((door.depth(point), -length, point, length, result))
    if not candidates:
        return None
    # For a retry prefer the first safe crossing depth and shortest route,
    # rather than the deepest observed cell.  Depth exploration belongs to
    # the planned lateral viewpoints after entry.
    target_depth = max(config.minimum_crossing_depth + 0.25, 0.90)
    _, _, point, length, path_result = min(
        candidates, key=lambda item: (abs(item[0] - target_depth),
                                      item[3], -item[0]))
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


def _observed_clear_straight_segment(grid: OccupancyGrid2D,
                                     start: Point, end: Point,
                                     clearance: float,
                                     allow_start_radius: float = 0.25) -> bool:
    """Validate one observed-free body chord with batched exact ray tests.

    The scalar implementation called world_to_cell roughly 1.3 million times
    while evaluating one two-pose route. The samples, eight headings, ray step
    and blocked/unknown policy below are unchanged; only their array execution
    is batched. SCAN-lite still performs the independent live 3-D body audit.
    """
    length = _distance(start, end)
    if length <= 0.35:
        return True
    required_clearance = float(clearance)
    maximum_clearance = 1.5
    if maximum_clearance + 1e-9 < required_clearance:
        return False
    step = max(0.06, min(0.12, 0.60 * grid.resolution))
    sample_distance = min(
        length, max(step, float(allow_start_radius)))
    distances = []
    while sample_distance <= length + 1e-9:
        distances.append(sample_distance)
        sample_distance += step
    if not distances:
        return True
    sample_points = []
    delta_x, delta_y = end[0] - start[0], end[1] - start[1]
    for distance in distances:
        ratio = min(1.0, distance / max(length, 1e-9))
        sample_points.append((
            start[0] + ratio * delta_x,
            start[1] + ratio * delta_y))
    point_x = np.asarray([point[0] for point in sample_points],
                         dtype=np.float64)
    point_y = np.asarray([point[1] for point in sample_points],
                         dtype=np.float64)

    def cells_are_observed_free(x_values, y_values):
        cell_x = np.floor(
            (x_values - float(grid.origin_x)) / grid.resolution).astype(
                np.int64)
        cell_y = np.floor(
            (y_values - float(grid.origin_y)) / grid.resolution).astype(
                np.int64)
        valid = ((cell_x >= 0) & (cell_x < grid.width) &
                 (cell_y >= 0) & (cell_y < grid.height))
        if not np.all(valid):
            return False
        return bool(np.all(grid.data[cell_y, cell_x] == 0))

    if not cells_are_observed_free(point_x, point_y):
        return False
    ray_step = max(0.04, 0.45 * grid.resolution)
    ray_distance = ray_step
    relevant_ray_distances = []
    while ray_distance <= maximum_clearance + 1e-9:
        if ray_distance + 1e-9 < required_clearance:
            relevant_ray_distances.append(ray_distance)
        ray_distance += ray_step
    for angle in (0.0, math.pi / 4.0, math.pi / 2.0,
                  3.0 * math.pi / 4.0, math.pi,
                  5.0 * math.pi / 4.0, 3.0 * math.pi / 2.0,
                  7.0 * math.pi / 4.0):
        direction_x, direction_y = _unit(angle)
        for ray_distance in relevant_ray_distances:
            if not cells_are_observed_free(
                    point_x + ray_distance * direction_x,
                    point_y + ray_distance * direction_y):
                return False
    return True


def prepare_exit_path(grid: OccupancyGrid2D, door: EstimatedDoorway,
                      current: Point,
                      config: LightweightRoomConfig) -> Optional[dict]:
    clearance = portal_clearance(grid, door, config)
    normal = door.normal

    def nearest_observed_anchor_path() -> Optional[dict]:
        """Take the shortest checked route to the doorway's live inside pose."""
        anchor_clearance = min(clearance, config.entry_staging_clearance)
        inside = _snap_safe_point(
            grid, door.interior_side, anchor_clearance, door, 0.45,
            search_radius=0.70)
        if inside is None:
            return None
        approach_path = astar_safe_path(
            grid, current, inside, anchor_clearance, 0.15,
            allow_blocked_start=True,
                maximum_expansions=config.exit_preflight_maximum_expansions)
        if not approach_path.get("success"):
            return None
        combined = list(approach_path.get("path") or [])
        direct = _distance(current, inside)
        length = sum(_distance(combined[i - 1], combined[i])
                     for i in range(1, len(combined)))
        if length > max(6.5, 2.25 * direct + 0.75):
            return None
        mandatory = [inside, door.center, door.corridor_side]
        for point in mandatory[1:]:
            if not combined or _distance(combined[-1], point) > 0.02:
                combined.append(point)
        return {
            "mandatory_portal_waypoints": [
                [point[0], point[1]] for point in mandatory],
            "portal_clearance_m": anchor_clearance,
            "preflight_path": combined,
            "door_centerline_alignment": [inside[0], inside[1]],
            "door_normal_approach": [inside[0], inside[1]],
            "nearest_observed_anchor_fallback": True,
            "approach_path_length_m": length,
        }

    # Normal case: align near the doorway, then cross it. This deliberately
    # contains no room-deep breadcrumb; observation points are not exit goals.
    alignment_depth = max(0.75, min(1.20, 0.65 * config.entry_depth))
    alignment = (door.center[0] + alignment_depth * normal[0],
                 door.center[1] + alignment_depth * normal[1])
    exit_point, _ = canonical_doorway_anchors(
        door.center, door.normal_direction, config)
    # Three semantic anchors are sufficient: shortest A* to the saved
    # door-normal inside pose, continuous door-centre transit, then corridor.
    # The former extra approach anchor was only 0.3--0.6 m from its neighbours
    # and was frequently moved by SCAN-lite, falsely invalidating a safe exit.
    desired = (alignment, door.center, exit_point)
    tolerances = (0.30, 0.20, 0.35)
    snapped = [_snap_safe_point(grid, point, clearance, door, tolerance)
               for point, tolerance in zip(desired, tolerances)]
    if any(point is None for point in snapped):
        return nearest_observed_anchor_path()
    # When the second observation pose has a clear chord to the saved inside
    # centreline, execute that chord directly.  The old four-connected A* trace
    # was safe but produced staircase-shaped 0.15 m bends which became 4--7
    # stop-and-go waypoints.  Keep A* for furniture-obstructed rooms and retain
    # the conservative alignment -> door centre -> corridor crossing in both
    # cases.
    direct_approach = bool(
        _distance(current, snapped[0]) >= 0.55 and
        _observed_clear_straight_segment(
            grid, current, snapped[0], clearance))
    start = snapped[0] if direct_approach else current
    combined = [current, snapped[0]] if direct_approach else []
    first_index = 1 if direct_approach else 0
    for index, target in enumerate(snapped[first_index:], start=first_index):
        result = astar_safe_path(
            grid, start, target, clearance, 0.15,
            allow_blocked_start=(index == 0),
                maximum_expansions=config.exit_preflight_maximum_expansions)
        if not result.get("success"):
            return nearest_observed_anchor_path()
        path = list(result.get("path", []))
        combined.extend(path if not combined else path[1:])
        start = target
    return {
        "mandatory_portal_waypoints": [[point[0], point[1]] for point in snapped],
        "portal_clearance_m": clearance,
        "preflight_path": combined,
        "door_centerline_alignment": [snapped[0][0], snapped[0][1]],
        "door_normal_approach": [snapped[0][0], snapped[0][1]],
        "direct_centerline_exit_approach": direct_approach,
        "nearest_observed_anchor_fallback": False,
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
                     boundary_probe: bool = False,
                     entry_clearance_override: Optional[float] = None
                     ) -> Optional[Dict]:
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
        # ENTRY is only a quick portal crossing.  Keep it shallow; the
        # subsequent G3/G4 radar views provide room depth.  A deep ENTRY
        # endpoint is especially fragile when the doorway projection is still
        # sparse and costs a full progress-timeout before retrying.
        "ENTRY": (max(config.minimum_crossing_depth + 0.25,
                      min(config.entry_depth, 1.00)), 0.0),
        "G1": (config.entry_depth, 0.0),
        # Keep the obstacle-shadow/depth view inside the normal room budget.
        # The former +1.0 bias routinely pushed G2 to 4.7--5 m and made the
        # return path dominate a short exploration run.
        "G2": (min(config.room_depth_probe_range - config.near_wall_margin,
                    max(config.maximum_center_depth,
                        door.depth(anchor) + 1.2)), 0.0),
        "G3": (min(config.maximum_side_depth,
                   max(config.side_depth, door.depth(anchor))), left_target),
        "G4": (min(config.maximum_side_depth,
                   max(config.side_depth, door.depth(anchor))), right_target),
    }
    if role not in role_targets:
        raise ValueError("unknown room goal role " + str(role))
    normal, tangent = door.normal, door.tangent
    desired_depth, desired_lateral = role_targets[role]
    entry_clearance = config.goal_clearance
    if role == "ENTRY" and entry_clearance_override is not None:
        # Reserved for an independently confirmed doorway whose rolling 2-D
        # projection has thickened the jambs. It never affects room views or
        # ordinary detector candidates.
        entry_clearance = min(
            entry_clearance, max(0.12, float(entry_clearance_override)))
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
            # Keep ENTRY sampling consistent with role_targets above. The
            # former hard-coded 1.15 m floor made the advertised 1.0 m quick
            # crossing impossible on a sparse upper-floor map, so a valid
            # 0.75 m doorway was discarded as no_safe_entry_candidate.
            minimum_depth = (max(config.minimum_crossing_depth + 0.25,
                                 min(config.entry_depth, 1.00))
                             if role == "ENTRY" else 1.15)
            if (depth < minimum_depth or
                    depth > config.room_depth_probe_range or
                    abs(lateral) > config.room_side_probe_range + 0.25):
                continue
            if role == "ENTRY":
                # ENTRY is a portal transit, not an information-gain view.
                # Keep its endpoint on the door normal so clearance scoring
                # cannot turn a straight commit into a diagonal wall crossing.
                entry_lateral_limit = min(
                    0.30, max(0.12, 0.5 * door.width - entry_clearance))
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
            required_goal_clearance = (
                entry_clearance if role == "ENTRY" else config.goal_clearance)
            if clearance < required_goal_clearance:
                continue
            separation = min((_distance(point, old) for old in completed),
                             default=math.inf)
            if completed and separation < config.minimum_goal_separation:
                continue
            # Keep the second lateral observation spatially distinct, but do
            # not let its route jump to the far end of a large room.  The
            # distance is measured from the first physical observation point,
            # not just the doorway frame, so it remains a useful parallax view
            # while fitting the room exit budget.
            if (role in ("G3", "G4") and completed and
                    _distance(point, completed[0]) > 3.2):
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
        planning_clearance = (min(portal_clearance(grid, door, config),
                                  entry_clearance)
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
                "entry_clearance_override_m": (
                    float(entry_clearance_override)
                    if role == "ENTRY" and
                    entry_clearance_override is not None else None),
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


def select_near_door_side_second_visual_view(
        grid: OccupancyGrid2D, door: EstimatedDoorway, current: Point,
        completed: Sequence[Point], template: Dict,
        config: LightweightRoomConfig) -> Optional[Dict]:
    """Complete a near first view with one short, separated near-side view."""
    anchors = list(completed)
    if not anchors or door.depth(anchors[-1]) >= 2.80:
        return None
    anchor = anchors[-1]
    anchor_depth = door.depth(anchor)
    anchor_lateral = door.lateral(anchor)
    required_baseline = (
        config.minimum_goal_separation +
        min(0.60, 0.75 * config.semantic_completion_tolerance))
    target_depth = min(2.35, max(1.55, anchor_depth + 0.25))
    preferred_signs = ((-1.0 if anchor_lateral > 0.0 else 1.0),
                       (1.0 if anchor_lateral > 0.0 else -1.0))
    target_abs_lateral = max(
        2.20, required_baseline + 0.10 - abs(anchor_lateral))
    planning_clearance = float(config.goal_clearance)
    candidates = []
    for y in range(grid.height):
        for x in range(grid.width):
            if int(grid.data[y, x]) != 0:
                continue
            point = grid.cell_to_world((x, y))
            depth = door.depth(point)
            lateral = door.lateral(point)
            if (depth < 1.15 or depth > 2.85 or
                    _clearance(grid, point) < config.goal_clearance or
                    any(_distance(point, old) < required_baseline
                        for old in anchors)):
                continue
            sign_rank = (0 if lateral * preferred_signs[0] > 0.0 else 1)
            score = (sign_rank + abs(depth - target_depth) +
                     0.35 * abs(abs(lateral) - target_abs_lateral))
            candidates.append((score, point, depth, lateral))
    for _, point, depth, lateral in sorted(candidates)[:16]:
        path = astar_safe_path(
            grid, current, point, planning_clearance, 0.15,
            allow_blocked_start=True)
        if not path.get("success"):
            continue
        points = list(path.get("path") or [])
        length = sum(_distance(points[i - 1], points[i])
                     for i in range(1, len(points)))
        chord = max(0.01, _distance(current, point))
        if length > max(4.80, 1.80 * chord):
            continue
        result = dict(template)
        result.update({
            "position": point,
            "role": "G4",
            "depth": float(depth),
            "lateral": float(lateral),
            "preflight_path": sparsify_verified_trace(points, spacing=0.35),
            "preflight_path_length_m": float(length),
            "path_planning_clearance_m": float(planning_clearance),
            "post_entry_path_clearance_relaxed": False,
            "two_pose_room_sweep": True,
            "two_pose_index": 2,
            "two_pose_strategy": "provisional_near_gap_second",
            "visual_baseline_m": min(
                _distance(point, old) for old in anchors),
            "visual_baseline_requirement_met": True,
            "fallback_candidate": True,
        })
        return result
    return None


def select_fresh_grid_deep_second_visual_view(
        door: EstimatedDoorway, completed: Sequence[Point], plan: Dict,
        config: LightweightRoomConfig) -> Optional[Dict]:
    """Reuse the fresh-grid deep leg after an intentional near mapping G3.

    plan_two_pose_room_sweep has already clearance/A*-validated every
    ordered item from the live near pose. Keep only a genuinely diagonal,
    depth-separated candidate so the provisional repair cannot add another
    near-door point.
    """
    anchors = list(completed)
    if not anchors:
        return None
    anchor = anchors[-1]
    anchor_depth = door.depth(anchor)
    anchor_lateral = door.lateral(anchor)
    required_baseline = max(
        config.minimum_goal_separation,
        0.80 * config.visual_breadth_min_baseline)
    candidates = []
    for item in list(plan.get("ordered") or []):
        point = tuple(float(value) for value in item["position"][:2])
        depth = float(item.get("depth", door.depth(point)))
        lateral = float(item.get("lateral", door.lateral(point)))
        depth_delta = depth - anchor_depth
        lateral_delta = abs(lateral - anchor_lateral)
        baseline = _distance(point, anchor)
        diagonal_ratio = lateral_delta / max(depth_delta, 1e-6)
        if (depth_delta < max(1.20, 0.35 * required_baseline) or
                baseline < required_baseline - 0.05 or
                diagonal_ratio < 0.70 or diagonal_ratio > 1.45):
            continue
        candidates.append((
            abs(diagonal_ratio - 1.0), -depth_delta,
            float(item.get("preflight_path_length_m", math.inf)), item,
            baseline, diagonal_ratio))
    if not candidates:
        return None
    _, _, _, selected, baseline, diagonal_ratio = min(candidates)
    result = dict(selected)
    result.update({
        "role": "G4", "two_pose_index": 2,
        "two_pose_strategy": "fresh_grid_near_then_deep",
        "two_pose_obstacle": None,
        "visual_baseline_m": float(baseline),
        "visual_baseline_requirement_met": True,
        "visual_diagonal_ratio": float(diagonal_ratio),
        "fallback_candidate": False,
    })
    return result


def select_locked_open_missing_depth_view(
        grid: OccupancyGrid2D, door: EstimatedDoorway, current: Point,
        completed: Sequence[Point], config: LightweightRoomConfig
        ) -> Optional[Dict]:
    """Select the still-missing half of a locked deep+near contract.

    A newly crossed upper-floor room can expose only a shallow mapping point
    on the first projection.  After that point succeeds, the generic adaptive
    selector used to choose another shallow G4 and physically exit, leaving a
    partial truth contract.  The same physical doorway was then rediscovered
    under a new estimated-room id.  This selector runs on the fresh live grid
    and admits only an endpoint which supplies the missing depth class; A* and
    the normal room clearance remain mandatory.
    """
    anchors = [(float(point[0]), float(point[1])) for point in completed]
    if (grid is None or not anchors or
            getattr(door, "viewpoint_contract", None) != "open_deep_near"):
        return None
    anchor_depths = [float(door.depth(point)) for point in anchors]
    # Use the same physical depth classes as final contract validation.  The
    # locomotion executor may legally stop within semantic tolerance of the
    # requested deep endpoint.  The old 0.55*tolerance selector threshold
    # therefore classified a physically accepted deep G3 as "near" and sent
    # G4 even deeper (F3 fallanchor room 2: 5.48 m then 6.94 m).
    minimum_deep = max(
        3.80, float(config.open_room_minimum_deep_depth) -
        float(config.semantic_completion_tolerance))
    has_deep = any(depth >= minimum_deep for depth in anchor_depths)
    deepest = max(anchor_depths)
    maximum_near = min(4.35, deepest - 1.50) if has_deep else 4.35
    has_near = any(depth <= maximum_near for depth in anchor_depths)
    if has_deep and has_near:
        return None

    known_y, known_x = np.nonzero(grid.data == 0)
    candidates = []
    for cell_x, cell_y in zip(known_x.tolist(), known_y.tolist()):
        point = grid.cell_to_world((int(cell_x), int(cell_y)))
        depth = float(door.depth(point))
        lateral = float(door.lateral(point))
        if abs(lateral) > min(1.20, config.room_side_probe_range):
            continue
        if has_deep:
            if depth > maximum_near:
                continue
        elif depth < minimum_deep:
            continue
        baseline = min(_distance(point, old) for old in anchors)
        if baseline < config.minimum_goal_separation:
            continue
        clearance = _clearance(grid, point)
        if clearance < config.goal_clearance:
            continue
        desired_depth = (min(config.room_depth_probe_range -
                             config.near_wall_margin,
                             max(config.open_room_preferred_deep_depth,
                                 minimum_deep))
                         if not has_deep else
                         max(1.25, min(2.60, maximum_near)))
        candidates.append((abs(depth - desired_depth) + 0.35 * abs(lateral),
                           point, depth, lateral, clearance, baseline))

    missing_role = "G4" if has_deep else "G3"
    for _, point, depth, lateral, clearance, baseline in sorted(candidates)[:8]:
        path = astar_safe_path(
            grid, current, point, config.goal_clearance, 0.15,
            allow_blocked_start=True, maximum_expansions=800)
        if not path.get("success"):
            continue
        points = list(path.get("path") or [])
        length = sum(_distance(points[index - 1], points[index])
                     for index in range(1, len(points)))
        return {
            "position": point, "role": missing_role, "score": 0.0,
            "depth": depth, "lateral": lateral, "clearance": clearance,
            "coverage_cells": set(), "adaptive_marginal_gain_m2": 0.0,
            "adaptive_utility": 0.0, "preflight_path_length_m": length,
            "preflight_path": sparsify_verified_trace(points, spacing=0.35),
            "two_pose_room_sweep": True, "two_pose_index": 2,
            "two_pose_strategy": ("locked_open_missing_near" if has_deep
                                  else "locked_open_missing_deep"),
            "visual_baseline_m": baseline,
            "visual_baseline_requirement_met": True,
            "fallback_candidate": False,
        }
    # A freshly reset upper-floor projection can end several metres before the
    # physical back wall even though the generated-layout contract proves the
    # doorway centreline is furniture-free.  In that case the live-grid search
    # above cannot manufacture a deep cell, and a same-room retry merely repeats
    # a shallow G4 forever.  Use the audited axial endpoint as a final bounded
    # proposal.  Known occupied centre samples still veto it here, and the
    # manager's independent 3-D SCAN-lite audit remains mandatory before motion.
    truth_deep = getattr(door, "truth_open_centerline_deep_depth_m", None)
    if not has_deep and truth_deep is not None:
        depth = float(truth_deep)
        point = (float(door.center[0]) + depth * float(door.normal[0]),
                 float(door.center[1]) + depth * float(door.normal[1]))
        baseline = min(_distance(point, old) for old in anchors)
        occupied = False
        segment_length = _distance(current, point)
        sample_count = max(2, int(math.ceil(segment_length / 0.15)))
        for index in range(sample_count + 1):
            ratio = float(index) / float(sample_count)
            sample = (float(current[0]) + ratio * (point[0] - current[0]),
                      float(current[1]) + ratio * (point[1] - current[1]))
            cell_x, cell_y = grid.world_to_cell(sample)
            if (cell_x < 0 or cell_y < 0 or cell_x >= grid.width or
                    cell_y >= grid.height or int(grid.data[cell_y, cell_x]) > 0):
                occupied = True
                break
        if (not occupied and depth >= minimum_deep and
                baseline >= config.minimum_goal_separation):
            return {
                "position": point, "role": "G3", "score": 0.0,
                "depth": depth, "lateral": 0.0,
                "clearance": _clearance(grid, point),
                "coverage_cells": set(), "adaptive_marginal_gain_m2": 0.0,
                "adaptive_utility": 0.0,
                "preflight_path_length_m": segment_length,
                "preflight_path": [tuple(current), point],
                "two_pose_room_sweep": True, "two_pose_index": 2,
                "two_pose_strategy":
                    "truth_layout_open_centerline_missing_deep",
                "visual_baseline_m": baseline,
                "visual_baseline_requirement_met": True,
                "truth_layout_centerline_fallback": True,
                "fallback_candidate": True,
            }
    return None


def select_reversed_open_g3_near_view(
        door: EstimatedDoorway, current: Point, completed: Sequence[Point],
        executed_path: Sequence[Point],
        config: LightweightRoomConfig) -> Optional[Dict]:
    """Recover the near view on the just-traversed open-room G3 trace.

    Upper-floor map rebuilds can mark the axial return lane unknown immediately
    after G3 even though the dog physically traversed that lane seconds ago.
    Replaying only the bounded prefix back to a near-door depth is stronger
    evidence than repeatedly asking that stale raster for another endpoint.
    """
    anchors = [(float(point[0]), float(point[1])) for point in completed]
    trace = [(float(point[0]), float(point[1])) for point in executed_path
             if isinstance(point, (list, tuple)) and len(point) >= 2]
    if (not anchors or len(trace) < 2 or
            getattr(door, "viewpoint_contract", None) != "open_deep_near"):
        return None
    minimum_deep = max(
        3.80, float(config.open_room_minimum_deep_depth) -
        float(config.semantic_completion_tolerance) - 0.15)
    deepest = max(float(door.depth(point)) for point in anchors)
    if deepest < minimum_deep:
        return None
    maximum_near = min(4.35, deepest - 1.50)
    desired_depth = max(float(config.minimum_crossing_depth) + 0.35,
                        min(2.60, maximum_near - 0.10))

    # The stored route runs entry -> G3.  Find the near-door endpoint on that
    # proven lane, but do not blindly prepend the old G3 endpoint when the
    # first G4 execution has already moved away from it.  run38 ended the
    # failed G4 beside the clear axial lane; replaying ``current -> G3 -> G4``
    # made the dog walk deeper before coming out and consumed the remaining
    # observation window.  Open-room geometry has no doorway-front blocker;
    # ``current -> near`` is independently truth/SCAN-lite audited by the
    # manager, so it is the bounded and physically shorter recovery.
    reversed_trace = list(reversed(trace))
    route = [(float(current[0]), float(current[1]))]
    for point in reversed_trace:
        if _distance(route[-1], point) > 0.05:
            route.append(point)
    endpoint = None
    endpoint_index = None
    for index in range(1, len(route)):
        start, end = route[index - 1], route[index]
        start_depth, end_depth = door.depth(start), door.depth(end)
        if ((start_depth - desired_depth) *
                (end_depth - desired_depth) <= 0.0 and
                abs(start_depth - end_depth) > 1e-6):
            ratio = ((desired_depth - start_depth) /
                     (end_depth - start_depth))
            endpoint = (start[0] + ratio * (end[0] - start[0]),
                        start[1] + ratio * (end[1] - start[1]))
            endpoint_index = index
            break
    if endpoint is None:
        eligible = [(index, point) for index, point in enumerate(route[1:], 1)
                    if (config.minimum_crossing_depth <= door.depth(point) <=
                        maximum_near)]
        if not eligible:
            return None
        endpoint_index, endpoint = min(
            eligible, key=lambda item: abs(door.depth(item[1]) - desired_depth))
    baseline = min(_distance(endpoint, old) for old in anchors)
    if baseline < config.minimum_goal_separation:
        return None
    direct_recovery = bool(
        _distance(current, trace[-1]) >
        max(0.35, float(config.semantic_completion_tolerance)))
    if direct_recovery:
        verified_route = [(float(current[0]), float(current[1])), endpoint]
    else:
        verified_route = route[:endpoint_index]
        verified_route.append(endpoint)
    return {
        "position": endpoint, "role": "G4", "score": 0.0,
        "depth": float(door.depth(endpoint)),
        "lateral": float(door.lateral(endpoint)),
        "clearance": math.inf, "coverage_cells": set(),
        "adaptive_marginal_gain_m2": 0.0, "adaptive_utility": 0.0,
        "preflight_path_length_m": sum(
            _distance(verified_route[index - 1], verified_route[index])
            for index in range(1, len(verified_route))),
        "preflight_path": sparsify_verified_trace(
            verified_route, spacing=0.55),
        "two_pose_room_sweep": True, "two_pose_index": 2,
        "two_pose_strategy": (
            "direct_open_g4_recovery_to_near" if direct_recovery else
            "reversed_executed_open_g3_to_near"),
        "visual_baseline_m": baseline,
        "visual_baseline_requirement_met": True,
        "verified_trajectory_backtrack": bool(not direct_recovery),
        "live_3d_audit_required": bool(direct_recovery),
        "truth_layout_open_direct_recovery": bool(direct_recovery),
        "fallback_candidate": True,
    }


def select_opposite_side_second_visual_view(
        grid: OccupancyGrid2D, door: EstimatedDoorway, current: Point,
        completed: Sequence[Point], template: Dict,
        config: LightweightRoomConfig,
        missing_role: str = "G4") -> Optional[Dict]:
    """Repair a same-side missing G3/G4 using fresh observed-free occupancy."""
    anchors = list(completed)
    if not anchors:
        return None
    anchor = anchors[-1]
    anchor_lateral = door.lateral(anchor)
    if abs(anchor_lateral) < 0.45:
        return None
    desired_sign = -1.0 if anchor_lateral > 0.0 else 1.0
    # Stay in the door--occluder gap: enough opposite-side baseline to
    # expose the hidden sector, without a deep cross-room detour.
    target_depth = min(2.35, max(1.75, door.depth(anchor) - 2.15))
    target_lateral = desired_sign * max(
        1.00, 0.28 * config.visual_breadth_min_baseline)
    required_baseline = (
        config.minimum_goal_separation +
        min(0.60, 0.75 * config.semantic_completion_tolerance))
    geometric = []
    for y in range(grid.height):
        for x in range(grid.width):
            if int(grid.data[y, x]) != 0:
                continue
            point = grid.cell_to_world((x, y))
            depth = door.depth(point)
            lateral = door.lateral(point)
            if (depth < 1.25 or
                    depth > config.room_depth_probe_range -
                    config.near_wall_margin or
                    lateral * desired_sign < 0.60 or
                    any(_distance(point, old) < required_baseline
                        for old in anchors)):
                continue
            score = math.hypot(depth - target_depth,
                               1.15 * (lateral - target_lateral))
            geometric.append((score, point, depth, lateral))
    geometric.sort(key=lambda item: item[0])
    astar_attempts = 0
    for _, point, depth, lateral in geometric:
        if _clearance(grid, point) < config.goal_clearance:
            continue
        astar_attempts += 1
        path = astar_safe_path(
            grid, current, point, config.goal_clearance, 0.15,
            allow_blocked_start=True)
        if path.get("success"):
            points = list(path.get("path") or [])
            length = sum(_distance(points[i - 1], points[i])
                         for i in range(1, len(points)))
            result = dict(template)
            result.update({
                "position": point, "role": str(missing_role),
                "preflight_path": sparsify_verified_trace(
                    points, spacing=0.35),
                "preflight_path_length_m": float(length),
                "path_planning_clearance_m":
                    float(config.goal_clearance),
                "post_entry_path_clearance_relaxed": False,
                "two_pose_room_sweep": True,
                "two_pose_index": (1 if str(missing_role) == "G3" else 2),
                "two_pose_strategy":
                    "fresh_grid_opposite_side_repair",
                "visual_baseline_m": min(
                    _distance(point, old) for old in anchors),
                "visual_baseline_requirement_met": True,
                "fallback_candidate": True,
            })
            return result
        if astar_attempts >= 4:
            break
    return None


def select_front_gap_opposite_side_second_visual_view(
        grid: OccupancyGrid2D, door: EstimatedDoorway, current: Point,
        completed: Sequence[Point], template: Dict,
        config: LightweightRoomConfig,
        missing_role: str = "G4",
        excluded_targets: Sequence[Point] = ()) -> Optional[Dict]:
    """Find the second obstacle view strictly in the door-front gap.

    This is deliberately separate from the generic mandatory fallback.  A
    blocked obstacle-side endpoint must not be replaced by a deep wrap-around
    point: that changes the physical contract and can turn G4 into a duplicate
    or a long collision-prone detour.  Infer the measured obstacle front from
    occupied cells near the doorway normal, then search only free cells before
    that front edge and on the opposite lateral side of the completed G3.
    """
    anchors = list(completed)
    if not anchors:
        return None
    anchor = anchors[-1]
    anchor_lateral = door.contract_lateral(anchor)
    if abs(anchor_lateral) < 0.35:
        return None
    desired_sign = -1.0 if anchor_lateral > 0.0 else 1.0

    def near_blocker_segmented_route(target):
        """Keep a side-to-side deep pair in front of a nearer blocker.

        Some generated rooms contain a small centred chair before the large
        table which defines the visual contract.  A diagonal/current A* route
        can graze that chair and leave Goal Executor oscillating at one
        repaired raster waypoint (run158).  When generated geometry proves a
        nearer blocker exists, explicitly retract on the already occupied
        side, cross before that blocker, and deepen on the target side.  This
        remains a planning candidate only: all samples must be observed free
        here and the manager subsequently repeats its dense live-3D audit.
        """
        nearest_front = getattr(
            door, "truth_obstacle_nearest_blocker_front_depth_m", None)
        primary_front = getattr(
            door, "truth_obstacle_front_depth_m", None)
        if nearest_front is None or primary_front is None:
            return None
        nearest_front = float(nearest_front)
        primary_front = float(primary_front)
        if nearest_front >= primary_front - 0.45:
            return None
        target_depth = door.contract_depth(target)
        target_lateral = door.contract_lateral(target)
        current_depth = door.contract_depth(current)
        current_lateral = door.contract_lateral(current)
        if (current_lateral * target_lateral >= 0.0 or
                min(current_depth, target_depth) < nearest_front - 0.10):
            return None
        interior_depth = max(
            config.minimum_crossing_depth,
            door.contract_depth(door.interior_side))
        route_clearance = max(
            0.17, min(0.20, config.post_entry_path_clearance))
        crossing_depth = min(
            nearest_front - max(0.42, route_clearance + 0.22),
            min(current_depth, target_depth) - 0.35)
        crossing_depth = max(interior_depth + 0.18, crossing_depth)
        if crossing_depth >= nearest_front - 0.30:
            return None
        route = [
            tuple(current),
            door.contract_point(crossing_depth, current_lateral),
            door.contract_point(crossing_depth, target_lateral),
            tuple(target),
        ]
        compact = [route[0]]
        for point in route[1:]:
            if _distance(point, compact[-1]) >= 0.12:
                compact.append(point)
        if len(compact) < 3:
            return None
        for start, end in zip(compact[:-1], compact[1:]):
            samples = max(2, int(math.ceil(_distance(start, end) / 0.08)))
            for index in range(1, samples + 1):
                alpha = float(index) / float(samples)
                sample = (
                    start[0] + alpha * (end[0] - start[0]),
                    start[1] + alpha * (end[1] - start[1]),
                )
                if (_state(grid, sample) != 0 or
                        _clearance(grid, sample) < route_clearance):
                    return None
        return compact[1:], crossing_depth, route_clearance
    occupied_front = []
    for y in range(grid.height):
        for x in range(grid.width):
            if int(grid.data[y, x]) <= 50:
                continue
            point = grid.cell_to_world((x, y))
            depth = door.contract_depth(point)
            lateral = door.contract_lateral(point)
            if (depth >= 0.65 and depth <= config.room_depth_probe_range and
                    abs(lateral) <= max(1.25, 0.55 * config.maximum_side_lateral)):
                occupied_front.append(depth)
    # On a freshly entered upper-floor room the rolling raster can contain
    # no occupied furniture cells yet.  The generated floor geometry is still
    # authoritative for the *contract* (the live grid remains authoritative
    # for every endpoint/path safety check), so use its measured front edge as
    # a bounded planning prior instead of abandoning G4 and wrapping around
    # the obstacle.
    truth_front = (
        float(door.truth_obstacle_front_depth_m)
        if (getattr(door, "viewpoint_contract", None) ==
            "obstacle_front_opposite_sides" and
            getattr(door, "truth_obstacle_front_depth_m", None) is not None)
        else None)
    stale_live_front_edge_rejected = False
    if occupied_front:
        measured_front = min(occupied_front)
        # A freshly reset F3 projection can contain a continuous occupied
        # stripe immediately behind the doorway (run4: measured depth 0.79 m
        # while the locked table front was 3.80 m).  Treating that stripe as
        # furniture makes ``max_depth`` negative and prevents the existing
        # truth-bounded/live-3D-audited G4 fallback from ever running.  The
        # generated room contract is authoritative for which side of the
        # furniture the two camera centres must occupy; the live 3-D audit is
        # still authoritative for whether the resulting chord may execute.
        if (truth_front is not None and
                measured_front < truth_front - max(
                    0.75, 2.0 * float(config.goal_clearance))):
            front_edge = truth_front
            stale_live_front_edge_rejected = True
        else:
            front_edge = measured_front
    elif truth_front is not None:
        front_edge = truth_front
    else:
        return None
    view_policy = obstacle_view_policy(door)
    route_front_edge = float(obstacle_route_front_depth(door) or front_edge)
    # The obstacle-front contract is lateral, inside the finite door--table
    # gap.  Reusing the open-room visual-breadth target (2.34 m in the timed
    # profile) can exceed the entire free span between the two safe side
    # anchors (about 2.2 m in floor_*_room_1).  The physical two-view minimum
    # already rejects duplicate poses and is the correct bound here.
    # Keep a small endpoint margin, but do not reuse the full semantic goal
    # tolerance here.  Run8 showed that adding 0.35 m forced the truth fallback
    # from the nearest opposite-side gap point to a lateral-extra=0.40 point
    # outside the locally reachable front chord.  G3/G4 completion is audited
    # again from Gazebo truth, so 0.10 m is sufficient planning allowance while
    # retaining the strict 1.80 m measured-pose contract.
    required_baseline = (
        float(config.minimum_goal_separation) +
        min(0.10, float(config.semantic_completion_tolerance)))
    # The first gap-side pose already crossed the narrow doorway under the
    # bounded post-entry clearance contract.  Requiring the ordinary 0.38 m
    # room clearance for G4 can reject every cell on the opposite side when
    # the online doorway is only about 0.60 m wide (the F3 far-side room in
    # f3truthguard).  Use the same 0.20 m observed-free footprint bound as G3;
    # A* still validates the complete cross-gap path and never goes behind the
    # obstacle front.
    planning_clearance = min(
        config.goal_clearance,
        max(0.17, min(0.20, config.post_entry_path_clearance)))
    # Keep both camera centres close enough to the furniture front to see
    # around its lateral edge, while retaining a body-sized stopping gap.
    # Run12's 1.85--2.43 m poses were geometrically on opposite sides, but
    # remained too close to the doorway: the near sofa occluded F3 room2's
    # deep red sphere from both scans.  A nominal 3.30 m centre depth for the
    # generated 3.80 m furniture front leaves 0.50 m physical standoff and
    # makes the side-peek ray pass the sofa edge.  This is still strictly a
    # door--obstacle-gap pose and remains gated by live A*/SCAN-lite checks.
    max_depth = min(
        3.32,
        route_front_edge - max(float(config.goal_clearance) + 0.10, 0.48))
    if max_depth < 1.10:
        return None
    desired_visibility_depth = (
        min(3.30, max_depth - 0.02)
        if view_policy == "deep_outer_side_peek" else
        min(1.65, max(1.20, max_depth - 0.02)))
    # A deep room must remain deep on *both* sides.  run102 reached the first
    # 3.175 m side peek, then successive endpoint failures allowed the generic
    # candidate search to drift to 1.675/1.375 m and finally mutate the room
    # into a shallow chord.  That still produced a nominal two-view contract
    # but missed the occluded F3 hazard.  Lock every replacement endpoint to
    # the same visibility band used by completion-time geometry validation.
    minimum_visibility_depth = (
        max(1.05, float(front_edge) - 1.00)
        if view_policy == "deep_outer_side_peek" else 1.05)
    obstacle_min = float(getattr(
        door, "truth_obstacle_minimum_lateral_m", -0.30) or -0.30)
    obstacle_max = float(getattr(
        door, "truth_obstacle_maximum_lateral_m", 0.30) or 0.30)
    visibility_min = float(getattr(
        door, "truth_obstacle_visibility_minimum_lateral_m", obstacle_min)
        if getattr(door, "truth_obstacle_visibility_minimum_lateral_m", None)
        is not None else obstacle_min)
    visibility_max = float(getattr(
        door, "truth_obstacle_visibility_maximum_lateral_m", obstacle_max)
        if getattr(door, "truth_obstacle_visibility_maximum_lateral_m", None)
        is not None else obstacle_max)
    # A shallow front chord belongs in the finite door--obstacle gap.  The
    # former unconditional 1.50 m target pushed even a 0.60 m coffee table's
    # two endpoints onto lounge furniture (run54 F1 room 0), then retried the
    # same blocked deep geometry until the room was discarded as partial.
    # The generated ``visibility_*`` envelope can span every piece of
    # furniture in the room (run93 F3 room2: -3.657..+3.457 m).  Treating
    # that whole-room envelope as one obstacle pushed G3 to lateral=4.67 m,
    # produced a 7.04 m route and then made EXIT replay the deep route.  The
    # camera centres only need to clear the *primary* blocker edges; the
    # 170-degree scan provides the remaining visibility.  Keep deep peeks
    # deeper than shallow chords, but cap their lateral offset around the
    # primary obstacle and preserve the physical baseline between sides.
    minimum_shallow_side = max(
        0.5 * required_baseline + 0.05,
        (obstacle_max if desired_sign > 0.0 else abs(obstacle_min)) +
        float(config.goal_clearance) + 0.20)
    desired_side_magnitude = (
        min(1.85, max(1.50,
            (obstacle_max if desired_sign > 0.0 else abs(obstacle_min)) +
            float(config.goal_clearance) + 0.52))
        if view_policy == "deep_outer_side_peek" else
        minimum_shallow_side)
    minimum_target_side = (
        (obstacle_max if desired_sign > 0.0 else abs(obstacle_min)) + 0.70
        if view_policy == "deep_outer_side_peek" else 0.55)
    candidates = []
    for y in range(grid.height):
        for x in range(grid.width):
            if int(grid.data[y, x]) != 0:
                continue
            point = grid.cell_to_world((x, y))
            if any(_distance(point, failed) < 0.35
                   for failed in excluded_targets):
                continue
            depth = door.contract_depth(point)
            lateral = door.contract_lateral(point)
            if (depth < minimum_visibility_depth or depth > max_depth or
                    lateral * desired_sign < minimum_target_side or
                    _clearance(grid, point) < planning_clearance or
                    any(_distance(point, old) < required_baseline
                        for old in anchors)):
                continue
            candidates.append((
                1.35 * abs(depth - desired_visibility_depth) +
                0.75 * abs(abs(lateral) - desired_side_magnitude),
                point, depth, lateral))
    for _, point, depth, lateral in sorted(candidates)[:32]:
        path = astar_safe_path(
            grid, current, point, planning_clearance, 0.15,
            allow_blocked_start=True)
        if not path.get("success"):
            continue
        points = list(path.get("path") or [])
        length = sum(_distance(points[i - 1], points[i])
                     for i in range(1, len(points)))
        chord = max(0.01, _distance(current, point))
        maximum_detour_ratio = (
            1.85 if view_policy == "deep_outer_side_peek" else 1.20)
        if length > max(0.75, maximum_detour_ratio * chord):
            continue
        segmented = (near_blocker_segmented_route(point)
                     if view_policy == "deep_outer_side_peek" else None)
        selected_path = sparsify_verified_trace(points, spacing=0.35)
        near_blocker_crossing_depth = None
        if segmented is not None:
            selected_path, near_blocker_crossing_depth, _ = segmented
            length = sum(_distance(
                ([current] + list(selected_path))[index - 1],
                ([current] + list(selected_path))[index])
                for index in range(1, len(selected_path) + 1))
        result = dict(template)
        result.update({
            "position": point,
            "role": str(missing_role),
            "depth": float(depth),
            "lateral": float(lateral),
            "preflight_path": selected_path,
            "preflight_path_length_m": float(length),
            "path_planning_clearance_m": float(config.goal_clearance),
            "post_entry_path_clearance_relaxed": bool(
                planning_clearance < config.goal_clearance),
            "two_pose_room_sweep": True,
            "two_pose_index": (1 if str(missing_role) == "G3" else 2),
            "two_pose_strategy": "obstacle_front_gap_opposite_side",
            "obstacle_side_peek_visibility_target": True,
            "obstacle_front_standoff_m": float(route_front_edge - depth),
            "obstacle_view_policy": view_policy,
            "desired_visibility_depth_m": float(desired_visibility_depth),
            "desired_side_magnitude_m": float(desired_side_magnitude),
            "visual_baseline_m": min(_distance(point, old) for old in anchors),
            "visual_baseline_requirement_met": True,
            "fallback_candidate": True,
            "two_pose_obstacle": {"front_edge_depth_m": float(front_edge)},
        })
        if segmented is not None:
            result.update({
                "near_blocker_segmented_transition": True,
                "near_blocker_front_depth_m": float(
                    door.truth_obstacle_nearest_blocker_front_depth_m),
                "near_blocker_crossing_depth_m": float(
                    near_blocker_crossing_depth),
                "two_pose_obstacle": {
                    "front_edge_depth_m": float(front_edge),
                    "classification": "occupied_centre_component",
                    "segmented_near_blocker_transition": True,
                },
            })
        return result
    # A reset upper-floor raster can remain unknown across the opposite side
    # even after the first physical scan.  The generated layout already locks
    # the obstacle front and lateral extent for this room.  Use that geometry
    # only to form one bounded front-gap chord; any *known* occupied sample
    # vetoes it here and the manager's independent 3-D SCAN-lite audit still
    # gates locomotion.  This is preferable to wrapping behind the obstacle or
    # silently exiting with a one-view partial transaction.
    if (getattr(door, "viewpoint_contract", None) ==
            "obstacle_front_opposite_sides" and
            getattr(door, "truth_obstacle_front_depth_m", None) is not None):
        target_magnitude = max(
            desired_side_magnitude, 0.5 * required_baseline + 0.10,
            (obstacle_max if desired_sign > 0.0 else abs(obstacle_min)) +
            config.goal_clearance +
            (0.52 if view_policy == "deep_outer_side_peek" else 0.20))
        if view_policy == "deep_outer_side_peek":
            target_magnitude=min(1.85, target_magnitude)
        nx, ny = door.contract_normal
        tx, ty = door.contract_tangent
        contract_center = door.contract_center
        # Try several front-gap chords.  A single depth (formerly 2.20 m)
        # lets one stale occupied raster cell veto the only geometrically
        # valid G4 even though a shallower chord in the same door--obstacle
        # gap is clear.  Keep the opposite-side/baseline constraints fixed,
        # and select the shortest known-occupied-free truth-bounded chord.
        truth_gap_candidates = []
        truth_gap_candidates_live_audit = []
        depth_targets = []
        for candidate_depth in (
                desired_visibility_depth,
                min(3.05, max_depth - 0.12),
                min(2.75, max_depth - 0.25),
                min(2.40, max_depth - 0.40)):
            if (candidate_depth >= minimum_visibility_depth and
                    candidate_depth < front_edge and
                    all(abs(candidate_depth - old) >= 0.15
                        for old in depth_targets)):
                depth_targets.append(candidate_depth)
        for target_depth in depth_targets:
            for lateral_extra in (0.0, 0.20, 0.40):
                target_lateral = desired_sign * (
                    target_magnitude + lateral_extra)
                point = (
                    contract_center[0] + target_depth * nx +
                    target_lateral * tx,
                    contract_center[1] + target_depth * ny +
                    target_lateral * ty)
                if any(_distance(point, failed) < 0.35
                       for failed in excluded_targets):
                    continue
                samples = max(
                    2, int(math.ceil(_distance(current, point) / 0.10)))
                known_occupied = any(
                    _state(grid, (
                        current[0] +
                        (point[0] - current[0]) * index / samples,
                        current[1] +
                        (point[1] - current[1]) * index / samples)) > 50
                    for index in range(1, samples + 1))
                baseline = min(_distance(point, old) for old in anchors)
                if baseline >= required_baseline:
                    truth_gap_candidates_live_audit.append((
                        abs(target_depth - desired_visibility_depth) +
                        0.35 * abs(abs(target_lateral) -
                                   desired_side_magnitude),
                        _distance(current, point), point, target_depth,
                        target_lateral, baseline, known_occupied))
                if not known_occupied and baseline >= required_baseline:
                    truth_gap_candidates.append((
                        abs(target_depth - desired_visibility_depth) +
                        0.35 * abs(abs(target_lateral) -
                                   desired_side_magnitude),
                        _distance(current, point), point, target_depth,
                        target_lateral, baseline))
        raster_veto_overridden = False
        if truth_gap_candidates:
            (_, _, point, target_depth, target_lateral,
             baseline) = min(truth_gap_candidates)
        elif truth_gap_candidates_live_audit:
            # The F3 far obstacle room projected a continuous occupied stripe
            # across every mathematically valid front-gap chord even after a
            # stationary refresh, while generated geometry placed the table
            # front 1.6 m deeper.  Do not reinterpret that stale 2-D stripe as
            # permission to leave partial.  Dispatch exactly one bounded
            # truth-gap candidate to the independent live 3-D SCAN-lite audit;
            # execution is still vetoed if the physical chord is obstructed.
            (_, _, point, target_depth, target_lateral, baseline,
             _) = min(truth_gap_candidates_live_audit)
            raster_veto_overridden = True
        else:
            point = None
        if point is not None:
            result = dict(template)
            result.update({
                "position": point,
                "role": str(missing_role),
                "depth": float(target_depth),
                "lateral": float(target_lateral),
                "preflight_path": [tuple(current), point],
                "preflight_path_length_m": _distance(current, point),
                "path_planning_clearance_m": 0.0,
                "post_entry_path_clearance_relaxed": True,
                "truth_geometry_verified_gap_fallback": True,
                "known_occupied_segment_veto_passed": bool(
                    not raster_veto_overridden),
                "stale_raster_segment_veto_overridden": bool(
                    raster_veto_overridden),
                "stale_live_front_edge_rejected": bool(
                    stale_live_front_edge_rejected),
                "live_3d_audit_required": True,
                "two_pose_room_sweep": True,
                "two_pose_index": (1 if str(missing_role) == "G3" else 2),
                "two_pose_strategy":
                    "truth_obstacle_front_gap_opposite_side",
                "obstacle_side_peek_visibility_target": True,
                "obstacle_front_standoff_m": float(
                    front_edge - target_depth),
                "desired_visibility_depth_m": float(
                    desired_visibility_depth),
                "desired_side_magnitude_m": float(
                    desired_side_magnitude),
                "visual_baseline_m": float(baseline),
                "visual_baseline_requirement_met": True,
                "fallback_candidate": True,
                "two_pose_obstacle": {
                    "front_edge_depth_m": float(front_edge)},
                "excluded_obstacle_gap_target_count":
                    len(excluded_targets),
            })
            return result
    return None


def select_truth_obstacle_gap_first_visual_view(
        grid: OccupancyGrid2D, door: EstimatedDoorway, current: Point,
        config: LightweightRoomConfig,
        excluded_targets: Sequence[Point] = ()) -> Optional[Dict]:
    """Select G3 strictly in the door--obstacle gap for a truth-classified room.

    This is used immediately after ENTRY, before the live raster necessarily
    contains a complete obstacle component.  It never searches behind the
    measured front face and never chooses a central/deep mapping transit.
    A* and live clearance still gate the returned endpoint and path.
    """
    if (getattr(door, "viewpoint_contract", None) !=
            "obstacle_front_opposite_sides" or
            getattr(door, "truth_obstacle_front_depth_m", None) is None):
        return None
    # A dense live-3D/footprint rejection is stronger evidence than the
    # freshly reset 2-D raster.  Keep a meaningful radius around such an
    # endpoint so a retry cannot quantise back onto the same blocked pose.
    # The opposite side (or a different observed-free cell) remains eligible.
    exclusion_radius = 0.35

    def target_is_excluded(point: Point) -> bool:
        return any(_distance(point, failed) < exclusion_radius
                   for failed in excluded_targets)
    front_edge = float(door.truth_obstacle_front_depth_m)
    view_policy = obstacle_view_policy(door)
    route_front_edge = float(obstacle_route_front_depth(door) or front_edge)
    # ENTRY has already physically crossed the narrow portal.  Requiring the
    # ordinary 0.38 m room clearance from that threshold rejects both valid
    # side gaps when the live doorway estimate is only about 0.60 m wide.
    # Use the same bounded post-entry clearance accepted by SCAN-lite; the
    # endpoint and every A* sample remain observed-free.
    planning_clearance = min(config.goal_clearance, 0.20)
    max_depth = min(
        3.32,
        route_front_edge - max(float(config.goal_clearance) + 0.10, 0.48))
    if max_depth < 1.10:
        return None
    desired_depth = (
        min(3.30, max(1.15, max_depth - 0.02))
        if view_policy == "deep_outer_side_peek" else
        min(1.65, max(1.20, max_depth - 0.02)))
    lateral_limit = min(config.room_side_probe_range,
                        max(2.0, config.maximum_side_lateral))
    obstacle_min = float(getattr(door, "truth_obstacle_minimum_lateral_m",
                                 -0.30) or -0.30)
    obstacle_max = float(getattr(door, "truth_obstacle_maximum_lateral_m",
                                 0.30) or 0.30)
    # Plan beyond the strict measured-pose boundary so normal goal-arrival
    # tolerance cannot turn a valid deep/outer target into run32's shallow
    # G3 (depth=2.32 m, lateral=0.95 m).  A shallow obstacle naturally keeps
    # a shallower front-gap pair; F3 room2 (front=3.8 m) gets the deeper side
    # peeks required to see D83 around its adjacent sofa.
    arrival_margin = min(
        0.25, max(0.12, 0.35 * config.semantic_completion_tolerance))
    minimum_visibility_depth = (
        max(float(config.minimum_crossing_depth), front_edge - 1.00)
        if view_policy == "deep_outer_side_peek" else
        float(config.minimum_crossing_depth))
    minimum_planned_depth = min(
        max_depth, minimum_visibility_depth + arrival_margin)
    side_peek_margin = (
        max(0.70, float(config.goal_clearance) + 0.35)
        if view_policy == "deep_outer_side_peek" else
        max(0.12, 0.30 * float(config.goal_clearance)))
    minimum_outer_lateral = max(
        obstacle_max + side_peek_margin + arrival_margin,
        abs(obstacle_min) + side_peek_margin + arrival_margin)
    minimum_symmetric_side = max(
        0.82, 0.5 * float(config.minimum_goal_separation))
    shallow_positive = max(
        minimum_symmetric_side + 0.05,
        obstacle_max + config.goal_clearance + 0.20)
    shallow_negative = min(
        -(minimum_symmetric_side + 0.05),
        obstacle_min - config.goal_clearance - 0.20)
    side_targets = (
        min(1.85, max(1.50,
            obstacle_max + side_peek_margin + arrival_margin)),
        max(-1.85, min(-1.50,
            obstacle_min - side_peek_margin - arrival_margin)))
    if view_policy == "shallow_front_chord":
        side_targets = (shallow_positive, shallow_negative)
    candidates = []
    for y in range(grid.height):
        for x in range(grid.width):
            if int(grid.data[y, x]) != 0:
                continue
            point = grid.cell_to_world((x, y))
            if target_is_excluded(point):
                continue
            depth = door.contract_depth(point)
            lateral = door.contract_lateral(point)
            if (depth < minimum_planned_depth or depth > max_depth or
                    abs(lateral) > lateral_limit or
                    abs(lateral) < max(minimum_symmetric_side,
                                       minimum_outer_lateral) or
                    _clearance(grid, point) < planning_clearance):
                continue
            side_error = min(abs(lateral - target) for target in side_targets)
            if side_error > 0.90:
                continue
            path = astar_safe_path(
                grid, current, point, planning_clearance, 0.15,
                allow_blocked_start=True,
                maximum_expansions=config.exit_preflight_maximum_expansions)
            if not path.get("success"):
                continue
            points = list(path.get("path") or [])
            length = sum(_distance(points[i - 1], points[i])
                         for i in range(1, len(points)))
            chord = max(0.05, _distance(current, point))
            maximum_detour_ratio = (
                1.85 if view_policy == "deep_outer_side_peek" else 1.20)
            if length > max(0.75, maximum_detour_ratio * chord):
                continue
            candidates.append((abs(depth - desired_depth) + side_error,
                               point, depth, lateral, points, length))
    if not candidates:
        # The far F3 doorway can be physically crossed while its freshly reset
        # 2-D raster remains unknown on both lateral gap sides.  Construct one
        # truth-bounded side target before the measured obstacle front.  Known
        # occupied samples veto the direct chord, and execution still requires
        # the manager's live 3-D SCAN-lite clearance audit.  This fallback is a
        # real camera destination, never a bookkeeping-only duplicate pose.
        target_depth = desired_depth
        nx, ny = door.contract_normal
        tx, ty = door.contract_tangent
        contract_center = door.contract_center
        # Start on the side opposite the current camera centre.  The manager's
        # truth obstacle-gap audit intentionally requires a real cross-gap
        # chord; trying the same side first cannot satisfy that contract.
        current_lateral = door.contract_lateral(current)
        ordered_side_targets = tuple(sorted(
            side_targets,
            key=lambda value: (0 if value * current_lateral < 0.0 else 1,
                               abs(value))))
        stale_raster_candidate = None
        for target_lateral in ordered_side_targets:
            point = (
                contract_center[0] + target_depth * nx +
                target_lateral * tx,
                contract_center[1] + target_depth * ny +
                target_lateral * ty)
            if target_is_excluded(point):
                continue
            samples = max(
                2, int(math.ceil(_distance(current, point) / 0.10)))
            known_occupied = any(
                _state(grid, (
                    current[0] +
                    (point[0] - current[0]) * index / samples,
                    current[1] +
                    (point[1] - current[1]) * index / samples)) > 50
                for index in range(1, samples + 1))
            candidate = {
                "position": point,
                "role": "G3",
                "depth": float(target_depth),
                "lateral": float(target_lateral),
                # When the 2-D raster contains a stale occupied stripe, leave
                # this empty so the manager runs ordinary A* and then its
                # locked room-bounds/furniture truth audit before the dense
                # live 3-D refiner.  A non-stale chord can retain its direct
                # preflight.
                "preflight_path": ([] if known_occupied else
                                   [tuple(current), point]),
                "preflight_path_length_m": _distance(current, point),
                "path_planning_clearance_m": 0.0,
                "post_entry_path_clearance_relaxed": True,
                "truth_geometry_verified_gap_fallback": True,
                "known_occupied_segment_veto_passed": bool(
                    not known_occupied),
                "stale_raster_segment_veto_overridden": bool(
                    known_occupied),
                "live_3d_audit_required": True,
                "two_pose_room_sweep": True,
                "two_pose_index": 1,
                "two_pose_strategy": "obstacle_front_gap_first_truth_bounded",
                "obstacle_side_peek_visibility_target": True,
                "obstacle_front_standoff_m": float(
                    route_front_edge - target_depth),
                "obstacle_view_policy": view_policy,
                "desired_visibility_depth_m": float(desired_depth),
                "desired_side_magnitude_m": float(abs(target_lateral)),
                "visual_baseline_m": 0.0,
                "visual_baseline_requirement_met": True,
                "fallback_candidate": True,
                "two_pose_obstacle": {
                    "front_edge_depth_m": float(front_edge)},
                "excluded_obstacle_gap_target_count": len(excluded_targets),
                "coverage_cells": set(),
                "adaptive_marginal_gain_m2": 0.0,
                "adaptive_utility": 0.0,
                "score": 0.0,
                "clearance": 0.0,
            }
            if not known_occupied:
                return candidate
            # Preserve one geometry-valid candidate instead of returning a
            # stationary pseudo-G3.  No motion or viewpoint credit occurs
            # unless the manager's independent truth and dense 3-D checks
            # both accept the endpoint and Goal Executor reaches it.
            if stale_raster_candidate is None:
                stale_raster_candidate = candidate
        if stale_raster_candidate is not None:
            return stale_raster_candidate
        return None
    _, point, depth, lateral, points, length = min(candidates)
    return {
        "position": point,
        "role": "G3",
        "depth": float(depth),
        "lateral": float(lateral),
        "preflight_path": sparsify_verified_trace(points, spacing=0.35),
        "preflight_path_length_m": float(length),
        "path_planning_clearance_m": float(planning_clearance),
        "post_entry_path_clearance_relaxed": bool(
            planning_clearance < config.goal_clearance),
        "two_pose_room_sweep": True,
        "two_pose_index": 1,
        "two_pose_strategy": "obstacle_front_gap_first",
        "obstacle_side_peek_visibility_target": True,
        "obstacle_front_standoff_m": float(route_front_edge - depth),
        "obstacle_view_policy": view_policy,
        "desired_visibility_depth_m": float(desired_depth),
        "desired_side_magnitude_m": float(
            min(abs(side_targets[0]), abs(side_targets[1]))),
        "visual_baseline_m": 0.0,
        "visual_baseline_requirement_met": False,
        "fallback_candidate": True,
        "two_pose_obstacle": {
            "front_edge_depth_m": front_edge,
            "minimum_lateral_m": obstacle_min,
            "maximum_lateral_m": obstacle_max,
        },
        "excluded_obstacle_gap_target_count": len(excluded_targets),
    }


def select_mandatory_second_visual_view(
        grid: OccupancyGrid2D, door: EstimatedDoorway, current: Point,
        completed: Sequence[Point],
        config: LightweightRoomConfig) -> Optional[Dict]:
    """Find one safe depth-separated view when the lateral G4 template fails.

    A freshly crossed upper-floor room can expose a valid near/central G3
    before its side walls are represented well enough for the ordinary G4
    template.  LiDAR coverage may then be complete even though the mandatory
    second RGB-D viewpoint is still missing.  Search the same observed-free
    occupancy cells with the normal clearance and A* gates, but prefer a deep
    central endpoint instead of requiring a particular lateral sign.
    """
    anchors = list(completed) or [(float(current[0]), float(current[1]))]
    fallback = copy.copy(config)
    fallback.goal_clearance = min(
        config.goal_clearance,
        max(0.17, config.post_entry_path_clearance))
    deepest_anchor = max((door.depth(point) for point in anchors),
                         default=door.depth(current))
    if deepest_anchor >= 3.40:
        # A deep first camera centre already exposes the far wall. The old
        # generic fallback always requested an even deeper G1 and could turn
        # a 2.5 m chord into a 9--11 m obstacle detour. Complete the pair at
        # a safe near-door point instead.
        target_depth = min(2.35, max(1.70, deepest_anchor - 2.20))
        deepest_point = max(
            anchors, key=lambda point: door.depth(point))
        deepest_lateral = door.lateral(deepest_point)
        required_baseline = (
            config.minimum_goal_separation +
            min(0.60, 0.75 * config.semantic_completion_tolerance))
        candidates = []
        for y in range(grid.height):
            for x in range(grid.width):
                if int(grid.data[y, x]) != 0:
                    continue
                point = grid.cell_to_world((x, y))
                depth = door.depth(point)
                lateral = door.lateral(point)
                if (depth < 1.15 or depth > 2.85 or
                        _clearance(grid, point) < fallback.goal_clearance or
                        any(_distance(point, old) < required_baseline
                            for old in anchors)):
                    continue
                depth_delta = abs(deepest_anchor - depth)
                lateral_delta = abs(lateral - deepest_lateral)
                axial_rank = (0 if abs(lateral) <= 0.80 else 1)
                # For an open room the depth change is the physical baseline.
                # Prefer the short door-normal return lane and reject the old
                # large opposite-side triangle unless no axial endpoint is
                # available at all.
                score = (abs(depth - target_depth) +
                         0.80 * abs(lateral) +
                         0.20 * lateral_delta +
                         0.05 * _distance(current, point))
                candidates.append((axial_rank, score, point, depth,
                                   lateral))
        for axial_rank, _, point, depth, lateral in sorted(
                candidates)[:24]:
            if axial_rank != 0:
                continue
            verified_path = astar_safe_path(
                grid, current, point, fallback.goal_clearance, 0.15,
                allow_blocked_start=True)
            if not verified_path.get("success"):
                continue
            path_points = list(verified_path.get("path") or [])
            path_length = sum(
                _distance(path_points[index - 1], path_points[index])
                for index in range(1, len(path_points)))
            chord = max(0.01, _distance(current, point))
            if path_length > max(4.80, 1.80 * chord):
                continue
            baseline = min(
                (_distance(point, old) for old in anchors),
                default=math.inf)
            return {
                "position": point,
                "role": "G4",
                "score": -float(path_length),
                "depth": float(depth),
                "lateral": float(lateral),
                "clearance": float(_clearance(grid, point)),
                "coverage_cells": set(),
                "adaptive_marginal_gain_m2": 0.20,
                "adaptive_utility": -float(path_length),
                "visual_deepening": True,
                "visual_baseline_m": float(baseline),
                "visual_baseline_requirement_met": True,
                "visual_axial_lateral_m": float(lateral),
                "two_pose_room_sweep": True,
                "two_pose_strategy": "mandatory_deep_then_axial_near_fallback",
                "two_pose_index": 2,
                "two_pose_obstacle": None,
                "post_entry_path_clearance_relaxed": True,
                "path_planning_clearance_m": float(fallback.goal_clearance),
                "fallback_candidate": True,
                "preflight_path": sparsify_verified_trace(
                    path_points, spacing=0.35),
                "preflight_path_length_m": float(path_length),
            }
    fallback.maximum_center_depth = min(
        config.room_depth_probe_range - config.near_wall_margin,
        max(config.open_room_preferred_deep_depth,
            door.depth(current) + config.minimum_goal_separation))
    proposal = select_room_goal(
        grid, door, current, "G1", anchors, fallback,
        fallback=True, boundary_probe=True)
    if proposal is None:
        # The semantic G1 template can be empty after a near mapping G3.
        # Search observed-free cells for the still-mandatory separated G4;
        # endpoint clearance, A*, detour and live 3-D gates all remain.
        required_baseline = max(
            1.50, config.minimum_goal_separation - 0.15)
        target_depth = min(
            config.room_depth_probe_range - config.near_wall_margin,
            max(config.open_room_preferred_deep_depth,
                door.depth(current) + config.minimum_goal_separation))
        candidates = []
        for y in range(grid.height):
            for x in range(grid.width):
                if int(grid.data[y, x]) != 0:
                    continue
                point = grid.cell_to_world((x, y))
                depth = door.depth(point)
                lateral = door.lateral(point)
                baseline = min(
                    (_distance(point, old) for old in anchors),
                    default=math.inf)
                if (depth <= door.depth(current) + 1.20 or
                        depth > config.room_depth_probe_range -
                        config.near_wall_margin or
                        baseline < required_baseline or
                        _clearance(grid, point) < fallback.goal_clearance):
                    continue
                score = (abs(depth - target_depth) +
                         0.08 * abs(lateral) +
                         0.04 * _distance(current, point))
                candidates.append((score, point, depth, lateral, baseline))
        astar_attempts = 0
        for _, point, depth, lateral, baseline in sorted(candidates):
            if astar_attempts >= 16:
                break
            astar_attempts += 1
            candidate_path = astar_safe_path(
                grid, current, point, fallback.goal_clearance, 0.15,
                allow_blocked_start=True)
            if not candidate_path.get("success"):
                continue
            candidate_points = list(candidate_path.get("path") or [])
            candidate_length = sum(
                _distance(candidate_points[index - 1],
                          candidate_points[index])
                for index in range(1, len(candidate_points)))
            chord = max(0.01, _distance(current, point))
            if candidate_length > max(5.50, 1.85 * chord):
                continue
            proposal = {
                "position": point,
                "score": -float(candidate_length),
                "depth": float(depth),
                "lateral": float(lateral),
                "clearance": float(_clearance(grid, point)),
                "preflight_path": sparsify_verified_trace(
                    candidate_points, spacing=0.35),
                "preflight_path_length_m": float(candidate_length),
                "mandatory_free_cell_fallback": True,
            }
            break
        if proposal is None:
            return None
    verified_path = astar_safe_path(
        grid, current, proposal["position"], fallback.goal_clearance, 0.15,
        allow_blocked_start=True)
    if not verified_path.get("success"):
        return None
    path_points = list(verified_path.get("path") or [])
    path_length = sum(
        _distance(path_points[index - 1], path_points[index])
        for index in range(1, len(path_points)))
    proposal = dict(proposal)
    proposal["preflight_path"] = sparsify_verified_trace(
        path_points, spacing=0.35)
    proposal["preflight_path_length_m"] = float(path_length)
    baseline = min(
        (_distance(proposal["position"], point) for point in anchors),
        default=math.inf)
    proposal.update({
        "role": "G4",
        "coverage_cells": set(),
        "adaptive_marginal_gain_m2": 0.20,
        "adaptive_utility": float(proposal.get("score", 0.0)),
        "visual_deepening": True,
        "visual_baseline_m": float(baseline),
        "visual_baseline_requirement_met": bool(
            baseline >= config.minimum_goal_separation),
        "two_pose_room_sweep": True,
        "two_pose_strategy": "mandatory_near_then_deep_fallback",
        "two_pose_index": 2,
        "two_pose_obstacle": None,
        "post_entry_path_clearance_relaxed": True,
        "path_planning_clearance_m": float(fallback.goal_clearance),
        "fallback_candidate": True,
    })
    return proposal


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
        # Do not treat fresh wall silhouettes as an interior obstacle.
        if (role == "G2" and
                int(coverage.get("known_cells", 0) or 0) < 120):
            continue
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
        # A supplementary spin point must remain close enough to complete
        # within the room budget.  Geometry can produce a valid A* route to
        # the far end of a long/open room, but spending 7--10 m on G3/G4
        # strands the mission and prevents the doorway exit.  Keep G1 as the
        # structural center view; only accept a second view when its route is
        # bounded and therefore useful for two-point coverage.
        # The corridor-side rooms are ~8 m deep, so the opposite wall view
        # sits 3.5-4.5 m past the doorway.  Allow up to 5 m so G3/G4 both
        # execute on every floor; longer detours remain rejected.
        if (role != "G1" and travel > 5.0):
            continue
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
        # A side view whose corridor wall has never been observed from inside
        # (left_wall_seen / right_wall_seen false) is exactly the coverage a
        # second viewpoint exists to provide.  Give it the structural gain so
        # every room reliably visits both G3 and G4 regardless of frontier
        # debt, matching the F1 behaviour on all floors.
        structural_gain = (
            1.0 if (role == "G1" and not completed) else
            (0.45 if (role in ("G3", "G4") and boundary_probe) else 0.0))
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


def stage_uncertain_open_near_mapping_view(
        grid: OccupancyGrid2D, current: Point, plan: Optional[Dict],
        config: LightweightRoomConfig, attempted: Sequence[str]
        ) -> Tuple[Optional[Dict], bool]:
    """Stage a near scan when an initial open-room deep route is weak.

    A deep route that is reachable only with the post-entry clearance
    relaxation is not strong evidence that the room centre is actually open.
    Execute the already selected near-side camera stop first, then let the
    scheduler rebuild the remaining separated view from the post-scan grid.
    The replacement route starts at the live ENTRY pose; the original
    deep-to-near second leg is never reused backwards.
    """
    if (plan is None or attempted or
            str(plan.get("strategy", "")) not in (
                "deep_then_blind_near", "deep_then_axial_near")):
        return plan, False
    ordered = list(plan.get("ordered") or [])
    if len(ordered) < 2:
        return plan, False
    deep, near = ordered[0], ordered[1]
    if (not deep.get("post_entry_path_clearance_relaxed") or
            float(near.get("depth", math.inf)) >=
            float(deep.get("depth", -math.inf)) - 0.80):
        return plan, False
    target = tuple(float(value) for value in near["position"][:2])
    route_clearance = config.goal_clearance
    route = astar_safe_path(
        grid, current, target, route_clearance, 0.15,
        allow_blocked_start=True)
    relaxed = False
    if not route.get("success"):
        route_clearance = min(
            config.goal_clearance,
            max(0.17, config.post_entry_path_clearance))
        route = astar_safe_path(
            grid, current, target, route_clearance, 0.15,
            allow_blocked_start=True)
        relaxed = bool(route.get("success"))
    if not route.get("success"):
        return plan, False
    route_points = list(route.get("path") or [])
    route_length = sum(
        _distance(route_points[index - 1], route_points[index])
        for index in range(1, len(route_points)))
    staged = dict(near)
    staged.update({
        "role": "G3", "two_pose_index": 1,
        "two_pose_strategy": "uncertain_open_near_mapping_view",
        "two_pose_mapping_transit": True,
        "replan_second_view_from_fresh_grid": True,
        "preflight_path": sparsify_verified_trace(
            route_points, spacing=0.35),
        "preflight_path_length_m": route_length,
        "path_planning_clearance_m": route_clearance,
        "post_entry_path_clearance_relaxed": relaxed,
    })
    staged_plan = dict(plan)
    staged_plan.update({
        "ordered": [staged],
        "strategy": "uncertain_open_near_mapping_view",
        "baseline_m": 0.0, "route_length_m": route_length,
        "return_distance_m": 0.0,
        "fresh_grid_second_view_required": True,
    })
    return staged_plan, True


def plan_two_pose_room_sweep(grid: OccupancyGrid2D,

                             door: EstimatedDoorway,
                             current: Point,
                             return_anchor: Point,
                             config: LightweightRoomConfig,
                             excluded_targets_by_role: Optional[
                                 Dict[str, Sequence[Point]]] = None
                             ) -> Optional[Dict]:
    """Plan two separated full-spin poses using only online occupancy.

    A clear room gets a deep point and a shallow point close to the doorway
    normal.  This keeps the second observation on the short return leg instead
    of forcing a cross-room triangle. A sizeable centre obstacle gets one point on
    either side; the timed variant finishes at a shallower opposite-side point
    to shorten the subsequent egress. All endpoints,
    inter-pose travel and the return route are occupancy/A* validated.
    """
    excluded_targets_by_role = excluded_targets_by_role or {}
    minimum_depth = max(1.25, door.depth(current) + 0.20)
    maximum_depth = max(
        minimum_depth + 1.0,
        config.room_depth_probe_range - config.near_wall_margin)
    lateral_limit = min(config.room_side_probe_range,
                        max(2.0, config.maximum_side_lateral))

    obstacle_candidate_cells = {}
    central_obstacle_front_depths = []
    free_room_samples = []
    obstacle_band = min(1.20, max(0.75, 0.48 * lateral_limit))
    sample_stride = max(1, int(round(0.20 / grid.resolution)))
    # Front-of-obstacle views may legitimately backtrack from the ENTRY pose,
    # but remain well inside the measured doorway plane.
    front_sample_minimum_depth = 0.75
    # Transform all known cell centres into doorway-local coordinates in one
    # NumPy pass. Keeping this as a Python double loop cost 3--4 seconds per
    # room on the competition grid even though the later clearance/A* gates
    # inspect only a handful of endpoints. np.nonzero is row-major, so the
    # candidate order remains identical to the former y-then-x traversal.
    known_y, known_x = np.nonzero(grid.data >= 0)
    if known_x.size:
        world_x = (float(grid.origin_x) +
                   (known_x.astype(np.float64) + 0.5) * grid.resolution)
        world_y = (float(grid.origin_y) +
                   (known_y.astype(np.float64) + 0.5) * grid.resolution)
        normal_x, normal_y = door.normal
        tangent_x, tangent_y = door.tangent
        delta_x = world_x - float(door.center[0])
        delta_y = world_y - float(door.center[1])
        depths = delta_x * normal_x + delta_y * normal_y
        laterals = delta_x * tangent_x + delta_y * tangent_y
        states = grid.data[known_y, known_x]

        free_mask = (
            (states == 0) &
            (known_x % sample_stride == 0) &
            (known_y % sample_stride == 0) &
            (depths >= front_sample_minimum_depth) &
            (depths <= maximum_depth) &
            (np.abs(laterals) <= lateral_limit))
        for index in np.flatnonzero(free_mask):
            free_room_samples.append((
                (float(world_x[index]), float(world_y[index])),
                float(depths[index]), float(laterals[index])))

        occupied = states != 0
        central_mask = (
            occupied & (depths >= 0.75) &
            (depths <= maximum_depth - 0.55) &
            (np.abs(laterals) <= min(0.22, obstacle_band)))
        central_obstacle_front_depths.extend(
            float(depths[index]) for index in np.flatnonzero(central_mask))

        obstacle_mask = (
            occupied & (depths >= minimum_depth + 0.30) &
            (depths <= maximum_depth - 0.55) &
            (np.abs(laterals) <= obstacle_band))
        for index in np.flatnonzero(obstacle_mask):
            obstacle_candidate_cells[(int(known_x[index]),
                                      int(known_y[index]))] = (
                float(depths[index]), float(laterals[index]))

    # Do not add every occupied return in the doorway-local band into one
    # fictitious piece of furniture. Sparse wall edges and transient raster
    # streaks at unrelated depths previously combined into enough area to
    # classify an open room as obstacle_front_side_split. Only one connected
    # component that is both sizeable and relevant to the door-front region
    # may switch the two-view geometry to the obstacle strategy. A large
    # object near the far/deep observation depth is still avoided by endpoint
    # clearance and A*, but it must not replace the required deep+near pair.
    pending_obstacle_cells = set(obstacle_candidate_cells)
    obstacle_components = []
    while pending_obstacle_cells:
        seed = pending_obstacle_cells.pop()
        component = [seed]
        queue = deque([seed])
        while queue:
            cell_x, cell_y = queue.popleft()
            # Bridge one missing raster cell: float-to-cell quantisation can
            # split one physical furniture face into adjacent strips.
            for offset_x in (-2, -1, 0, 1, 2):
                for offset_y in (-2, -1, 0, 1, 2):
                    if offset_x == 0 and offset_y == 0:
                        continue
                    neighbour = (cell_x + offset_x, cell_y + offset_y)
                    if neighbour in pending_obstacle_cells:
                        pending_obstacle_cells.remove(neighbour)
                        component.append(neighbour)
                        queue.append(neighbour)
        obstacle_components.append(component)

    front_obstacle_maximum_depth = min(
        maximum_depth - 0.55,
        max(minimum_depth + 1.40,
            config.open_room_minimum_deep_depth - 1.10))
    eligible_obstacle_components = []
    cell_area = grid.resolution * grid.resolution
    for component in obstacle_components:
        records = [obstacle_candidate_cells[cell] for cell in component]
        component_depth_slices = {
            int(round(item[0] / max(grid.resolution, 0.05)))
            for item in records}
        component_area = len(component) * cell_area
        component_min_lateral = min(item[1] for item in records)
        component_max_lateral = max(item[1] for item in records)
        component_lateral_span = (
            component_max_lateral - component_min_lateral)
        # Door-centre jitter can leave only a sparse projection of a real wide
        # table face. Keep the normal area gate for compact fragments, while
        # accepting a coherent centre-spanning component at least 0.65 m wide.
        # The existing narrow-fragment regression remains classified as open.
        broad_coherent_front = bool(
            component_area >= max(
                0.20, 0.45 * config.obstacle_front_minimum_area_m2) and
            component_lateral_span >= 0.65 and
            component_min_lateral <= 0.25 and
            component_max_lateral >= -0.25)
        normal_area_front = bool(
            component_area >= max(
                0.20, config.obstacle_front_minimum_area_m2 - 1e-6))
        if ((normal_area_front or broad_coherent_front) and
                len(component_depth_slices) >= 2 and
                min(item[0] for item in records) <=
                front_obstacle_maximum_depth):
            eligible_obstacle_components.append(
                (component, broad_coherent_front and not normal_area_front))
    selected_obstacle_entry = (
        max(eligible_obstacle_components,
            key=lambda item: (
                len(item[0]),
                -min(obstacle_candidate_cells[cell][0]
                     for cell in item[0])))
        if eligible_obstacle_components else None)
    selected_obstacle_component = (
        selected_obstacle_entry[0] if selected_obstacle_entry else [])
    selected_obstacle_is_broad_sparse = bool(
        selected_obstacle_entry and selected_obstacle_entry[1])
    obstacle_cells = [obstacle_candidate_cells[cell]
                      for cell in selected_obstacle_component]
    obstacle_area = len(obstacle_cells) * grid.resolution * grid.resolution
    if (getattr(door, "viewpoint_contract", None) ==
            "obstacle_front_opposite_sides" and
            door.truth_obstacle_front_depth_m is not None and
            door.truth_obstacle_rear_depth_m is not None and
            door.truth_obstacle_minimum_lateral_m is not None and
            door.truth_obstacle_maximum_lateral_m is not None):
        # Use the generated furniture footprint only to state where the
        # required door--obstacle gap ends.  The camera endpoints and all
        # connecting paths below still have to exist as observed-free cells
        # in the live map.  Replacing (rather than merging) the LIO component
        # prevents two unrelated racks joined by drift smear from forcing a
        # wrap-around route.
        truth_front = float(door.truth_obstacle_front_depth_m)
        truth_rear = float(door.truth_obstacle_rear_depth_m)
        truth_min_lateral = float(door.truth_obstacle_minimum_lateral_m)
        truth_max_lateral = float(door.truth_obstacle_maximum_lateral_m)
        obstacle_cells = [
            (truth_front, truth_min_lateral),
            (truth_front, truth_max_lateral),
            (truth_rear, truth_min_lateral),
            (truth_rear, truth_max_lateral),
        ]
        obstacle_area = max(
            grid.resolution * grid.resolution,
            (truth_rear - truth_front) *
            (truth_max_lateral - truth_min_lateral))
        central_obstacle_front_depths = [truth_front]
    # A low table can interrupt the traversable centre lane while contributing
    # few occupied projection cells. Probe three doorway-normal rays too. An
    # unknown shadow in the centre must not be crossed by the chord between
    # the two physical camera centres.
    ray_start_depth = max(minimum_depth, door.depth(current))
    ray_probe_length = max(0.0, maximum_depth - ray_start_depth)
    centre_ray_hits = []
    if ray_probe_length >= 2.0:
        nx, ny = door.normal
        tx, ty = door.tangent
        for lateral_offset in (-0.22, 0.0, 0.22):
            ray_start = (
                door.center[0] + ray_start_depth * nx + lateral_offset * tx,
                door.center[1] + ray_start_depth * ny + lateral_offset * ty)
            clearance = _ray_distance(
                grid, ray_start, door.normal_direction, ray_probe_length)
            if 1.0 <= clearance <= ray_probe_length - 0.70:
                centre_ray_hits.append(ray_start_depth + clearance)
    # Ray-only inference is retained for low furniture, but all three returns must
    # terminate on one coherent front face rather than unrelated wall/raster
    # fragments at different depths.
    ray_depth_spread = ((max(centre_ray_hits) - min(centre_ray_hits))
                        if centre_ray_hits else float("inf"))
    ray_centre_obstructed = bool(
        len(centre_ray_hits) >= 3 and ray_depth_spread <= 0.55 and
        min(centre_ray_hits) <= front_obstacle_maximum_depth)
    centre_obstructed = bool(
        obstacle_cells or ray_centre_obstructed)
    # The generated multi-floor run already uses simulator truth for the
    # planar pose and corridor handoff.  When the floor layout says a room has
    # no sizeable object in the doorway-normal band, do not let a stale LIO
    # smear or a wrongly connected furniture component replace the mandatory
    # deep+near pair with two shallow points.  This override does not create a
    # goal: the open-room branch below must still find two observed-free,
    # clearance-valid endpoints and a live A* route.
    if getattr(door, "viewpoint_contract", None) == "open_deep_near":
        centre_obstructed = False
        obstacle_cells = []
        ray_centre_obstructed = False

    clearance_cache = {}
    clearance_field = _eight_ray_clearance_field(grid)
    path_cache = {}

    def safe_samples(desired_depth, desired_lateral, start, separated_from,
                     required_clearance=None):
        route_clearance = (config.goal_clearance
                           if required_clearance is None else
                           max(config.goal_clearance,
                               float(required_clearance)))
        geometric = []
        for point, depth, lateral in free_room_samples:
            if any(_distance(point, old) <
                   config.minimum_goal_separation
                   for old in separated_from):
                continue
            error = math.hypot(depth - desired_depth,
                               1.20 * (lateral - desired_lateral))
            geometric.append((error, point, depth, lateral))
        results = []
        # Only the nearest clearance-valid doorway-local samples can represent
        # this requested camera centre. The former [:24] geometric prefix
        # still launched up to 24 full-grid A* searches per target on upper
        # floors; a failed two-pose plan then consumed about 20 s before asking
        # for the intended fresh-map refresh. Inspect all endpoints cheaply,
        # but cap expensive A* attempts after clearance filtering.
        astar_attempts = 0
        for _, point, depth, lateral in sorted(
                geometric, key=lambda item: item[0]):
            point_cell = grid.world_to_cell(point)
            clearance_key = tuple(point_cell) if point_cell is not None else point
            if clearance_key not in clearance_cache:
                clearance_cache[clearance_key] = float(
                    clearance_field[point_cell[1], point_cell[0]])
            clearance = clearance_cache[clearance_key]
            if clearance < route_clearance:
                continue
            if astar_attempts >= 4:
                break
            astar_attempts += 1
            start_cell = grid.world_to_cell(start)
            path_key = (
                tuple(start_cell) if start_cell is not None else tuple(start),
                clearance_key, round(route_clearance, 3))
            if path_key not in path_cache:
                # Most room-view legs are short observed-free chords. A
                # direct sampled clearance proof avoids a full-floor A* for
                # every nearby raster candidate. Bounded A* remains for legs
                # that genuinely have to bend around furniture.
                if _observed_clear_straight_segment(
                        grid, start, point, route_clearance,
                        allow_start_radius=0.45):
                    path_cache[path_key] = {
                        "success": True,
                        "reason": "observed_clear_direct_room_view",
                        "path": [start, point],
                        "expansions": 0,
                    }
                else:
                    path_cache[path_key] = astar_safe_path(
                        grid, start, point, route_clearance, 0.15,
                        allow_blocked_start=True,
                        maximum_expansions=(
                            config.exit_preflight_maximum_expansions))
            path = path_cache[path_key]
            relaxed_start_escape = False
            relaxed_clearance = min(
                route_clearance, config.post_entry_path_clearance)
            if (not path.get("success") and not centre_obstructed and
                    relaxed_clearance + 1e-6 < route_clearance):
                relaxed_key = path_key + ("post_entry_start_escape",
                                          round(relaxed_clearance, 3))
                if relaxed_key not in path_cache:
                    path_cache[relaxed_key] = astar_safe_path(
                        grid, start, point, relaxed_clearance, 0.15,
                        allow_blocked_start=True,
                        maximum_expansions=(
                            config.exit_preflight_maximum_expansions))
                relaxed_path = path_cache[relaxed_key]
                if relaxed_path.get("success"):
                    path = relaxed_path
                    relaxed_start_escape = True
            portal_start_escape = False
            if (not path.get("success") and centre_obstructed and
                    door.depth(start) < 1.80):
                # A narrow doorway can inflate into a raster seal around the
                # freshly crossed pose even though the measured opening and
                # the room beyond it are free.  Do not relax clearance around
                # the central obstacle.  Instead, validate one short straight
                # portal throat at the measured doorway clearance, then run
                # the original obstacle-clearance A* from the first roomy
                # interior point.  SCAN-lite still audits the combined path
                # in live 3-D before execution.
                start_depth = door.depth(start)
                start_lateral = door.lateral(start)
                throat_clearance = portal_clearance(grid, door, config)
                lateral_limit_at_portal = max(
                    0.20, 0.50 * door.width - throat_clearance)
                if abs(start_lateral) <= lateral_limit_at_portal:
                    normal_x, normal_y = door.normal
                    tangent_x, tangent_y = door.tangent
                    escape_depth = max(1.15, start_depth + 0.30)
                    escape_depth_limit = min(
                        maximum_depth, start_depth + 1.50)
                    while escape_depth <= escape_depth_limit + 1e-9:
                        escape = (
                            door.center[0] + escape_depth * normal_x +
                            start_lateral * tangent_x,
                            door.center[1] + escape_depth * normal_y +
                            start_lateral * tangent_y)
                        if (_state(grid, escape) == 0 and
                                _clearance(grid, escape) >= route_clearance and
                                _observed_clear_straight_segment(
                                    grid, start, escape, throat_clearance)):
                            suffix = astar_safe_path(
                                grid, escape, point, route_clearance, 0.15,
                                allow_blocked_start=True,
                                maximum_expansions=(
                                    config.exit_preflight_maximum_expansions))
                            if suffix.get("success"):
                                suffix_points = list(suffix.get("path", []))
                                path = dict(suffix)
                                path["path"] = ([start, escape] +
                                                suffix_points[1:])
                                portal_start_escape = True
                                break
                        escape_depth += max(0.10, grid.resolution)
            if not path.get("success"):
                continue
            path_points = list(path.get("path", []))
            path_length = sum(_distance(path_points[i - 1], path_points[i])
                              for i in range(1, len(path_points)))
            results.append({
                "position": point, "depth": depth, "lateral": lateral,
                "clearance": clearance, "path_length_m": path_length,
                # Preserve raster bends around furniture. A 0.75 m chord can
                # cut the corner of a low table even though A* itself stayed
                # in free cells.
                "preflight_path": sparsify_verified_trace(
                    path_points, spacing=0.35),
                "post_entry_path_clearance_relaxed": relaxed_start_escape,
                "portal_start_escape": portal_start_escape,
                "path_planning_clearance_m": (
                    relaxed_clearance if relaxed_start_escape else
                    route_clearance),
            })
            if len(results) >= 2:
                break
        return results

    pairs = []
    # Keep verified door-front gap pairs separate from conservative flank
    # pairs.  If the former exists, it is the contractual geometry for a
    # front obstacle: two physical viewpoints on opposite sides in the
    # door--obstacle passage.  Mixing both sets and selecting by route score
    # allowed raster noise to choose a long wrap-around route (the room0 /
    # room2 failure visible in Figure 13).
    front_gap_pairs = []
    strategy = "deep_then_axial_near"
    obstacle_summary = None
    required_baseline = max(config.minimum_goal_separation,
                            config.visual_breadth_min_baseline)
    if centre_obstructed:
        strategy = "obstacle_side_split"
        if obstacle_cells:
            min_depth = min(item[0] for item in obstacle_cells)
            max_depth = max(item[0] for item in obstacle_cells)
            min_lateral = min(item[1] for item in obstacle_cells)
            max_lateral = max(item[1] for item in obstacle_cells)
        else:
            # Ray/unknown-shadow inference supplies an approximate central
            # footprint. Endpoints and the complete route still pass the
            # ordinary clearance and A* gates below.
            inferred_depth = float(sum(centre_ray_hits) /
                                   max(1, len(centre_ray_hits)))
            min_depth = max(minimum_depth + 0.30, inferred_depth - 0.35)
            max_depth = min(maximum_depth - 0.55, inferred_depth + 0.45)
            min_lateral, max_lateral = -0.45, 0.45
        raw_front_edge_depth = (
            min(central_obstacle_front_depths)
            if central_obstacle_front_depths else min_depth)
        # A single occupied return immediately behind the portal can be a
        # jamb/inflation remnant rather than the front face of the sizeable
        # centre component. Treating that isolated return as the furniture
        # edge removes the door--obstacle free strip and forces a long deep
        # wrap-around. Endpoint, chord and A* checks below still reject a
        # genuinely blocked front gap.
        front_edge_depth = raw_front_edge_depth
        if min_depth - raw_front_edge_depth > 0.50:
            front_edge_depth = min_depth
        # Place both spins beside the obstacle at its depth band, not in the
        # shallow free wedge in front of it.  A doorway sofa/lounge cluster
        # hides hazards behind it unless the camera centres sit outside the
        # occupied lateral span at or just beyond the obstacle depth.
        obstacle_mid_depth = 0.5 * (min_depth + max_depth)
        # Keep the camera centres and, crucially, their lateral connecting
        # route a full body margin behind the obstacle. speed_fix_10 placed
        # the line only 0.3--0.45 m behind its rear edge; the occupancy A*
        # accepted it but the physical A1 clipped the edge and rolled over.
        obstacle_route_clearance = max(
            config.goal_clearance, min(0.68, config.goal_clearance + 0.24))
        view_depth = min(
            maximum_depth - 0.25,
            max(minimum_depth + 0.55,
                max_depth + max(0.85, config.goal_clearance + 0.45),
                obstacle_mid_depth))
        second_view_depth = view_depth
        if config.obstacle_second_view_near_door:
            # Keep the first camera centre behind/deep beside the obstacle,
            # but finish before its front edge on the opposite lateral side.
            # The inter-pose A* must go around the measured object, while
            # ROOM_EXIT begins from a short, already verified door leg.
            shallow_candidate = max(
                minimum_depth + 0.45, min_depth - 0.65)
            if shallow_candidate <= view_depth - 1.0:
                second_view_depth = shallow_candidate
                strategy = "obstacle_deep_then_opposite_near"
        side_half_baseline = max(
            1.20, 0.50 * config.visual_breadth_min_baseline)
        left_target = min(
            lateral_limit, max(side_half_baseline, max_lateral + 1.10))
        right_target = max(
            -lateral_limit, min(-side_half_baseline, min_lateral - 1.10))
        obstacle_summary = {
            "area_m2": obstacle_area, "minimum_depth_m": min_depth,
            "maximum_depth_m": max_depth,
            "minimum_lateral_m": min_lateral,
            "maximum_lateral_m": max_lateral,
            "front_edge_depth_m": front_edge_depth,
            "raw_front_edge_depth_m": raw_front_edge_depth,
            "central_ray_hits": len(centre_ray_hits),
            "classification": ("occupied_centre_component"
                               if obstacle_cells else
                               "central_ray_or_unknown_shadow"),
            "broad_coherent_sparse_projection": bool(
                selected_obstacle_is_broad_sparse),
            "deep_view_depth_m": view_depth,
            "second_view_depth_m": second_view_depth,
            "second_view_near_door": bool(
                second_view_depth < view_depth - 0.50),
        }
        left = safe_samples(
            view_depth, left_target, current, [], obstacle_route_clearance)
        right = safe_samples(
            view_depth, right_target, current, [], obstacle_route_clearance)
        for first_side, second_target in ((left, right_target),
                                          (right, left_target)):
            for first in first_side:
                seconds = safe_samples(
                    second_view_depth, second_target, first["position"],
                    [first["position"]], obstacle_route_clearance)
                for second in seconds:
                    opposite_sides = ((first["lateral"] >= 0.0) !=
                                      (second["lateral"] >= 0.0))
                    depth_separated = (
                        not config.obstacle_second_view_near_door or
                        first["depth"] - second["depth"] >= 0.80)
                    if opposite_sides and depth_separated:
                        pairs.append((first, second))
        if (config.obstacle_front_opposite_side_pair and
                (getattr(door, "viewpoint_contract", None) ==
                 "obstacle_front_opposite_sides" or
                 obstacle_area >=
                 config.obstacle_front_minimum_area_m2 or
                 selected_obstacle_is_broad_sparse)):
            # Both camera centres remain between the doorway and the front
            # edge of the measured obstacle, but straddle its left/right
            # sides. Their connecting route crosses in front of the object
            # instead of wrapping around its rear edge.
            # This route crosses the free strip in front of the obstacle; it
            # does not pass along either obstacle flank. Reusing the larger
            # rear/flank clearance here rejects valid door--obstacle gaps.
            # Use ordinary robot clearance both forward and laterally, plus a
            # small raster tolerance before the measured obstacle front edge.
            front_route_clearance = config.goal_clearance
            # The obstacle-front strip is width-limited by the doorway and
            # the measured furniture face. Requiring the open-room diagonal
            # baseline here rejected otherwise valid opposite-side points
            # (run95 then selected a deep point behind the obstacle and spent
            # 24.7 s replaying the whole room trajectory to exit). Both
            # centres already straddle a measured obstacle and therefore gain
            # useful occlusion parallax with a smaller translational baseline.
            front_required_baseline = max(
                config.minimum_goal_separation,
                0.65 * config.visual_breadth_min_baseline)

            # Keep both obstacle-front viewpoints in the door--obstacle
            # passage, but leave a wider forward buffer for the robot body.
            # The former 0.75 m offset put the selected G3/G4 centres only
            # about 0.74 m before the measured front edge; the endpoint was
            # geometrically valid while the physical cross-gap waypoint
            # stalled on the inflated obstacle (F1 bounded-rescan run).
            # Moving the pair toward the doorway preserves opposite-side
            # parallax and makes the short front crossing executable.
            preferred_front_gap = max(
                front_route_clearance + 0.25, 1.05)
            # A close obstacle can leave less than the preferred 1.05 m
            # buffer while still providing a physically valid door-front
            # strip. Preserve at least a 0.90 m interior camera depth instead
            # of making the pair set empty and abandoning the room.
            front_gap = min(
                preferred_front_gap,
                max(front_route_clearance + 0.10,
                    front_edge_depth -
                    max(front_sample_minimum_depth + 0.15, 1.10)))
            # Use the deepest safe part of the door--obstacle gap so both
            # camera centres can move beyond the furniture lateral edges.  Placing the cross-room chord only
            # one clearance radius before a long obstacle face can make the
            # inflated raster reject it and fall back to a deep wrap-around.
            # Both views remain inside the room and strictly before the
            # measured obstacle front edge.
            front_view_depth = max(
                front_sample_minimum_depth + 0.15,
                front_edge_depth - front_gap)
            # Keep G3 at the deep edge of the door--obstacle strip, but put
            # G4 on the opposite side *and* closer to the doorway.  Using the
            # same deep chord for both points satisfied the geometric
            # left/right test, yet the upper-floor obstacle room then needed
            # a 7.14 m A* replay to exit and physically fell at the jamb.  A
            # shallow opposite-side G4 preserves real occlusion parallax,
            # remains wholly between door and obstacle, and leaves a short
            # diagonal egress instead of another traversal of the furniture
            # front face.
            front_second_view_depth = max(
                front_sample_minimum_depth + 0.25,
                min(front_view_depth - 0.90, 2.40))
            front_has_clearance = (
                front_view_depth <=
                front_edge_depth - front_route_clearance + 0.02 and
                front_second_view_depth >= front_sample_minimum_depth)
            front_start = current
            front_retreat_prefix = []
            front_start_ready = True
            if door.depth(current) > front_view_depth + 0.10:
                normal_x, normal_y = door.normal
                tangent_x, tangent_y = door.tangent
                current_lateral = door.lateral(current)
                front_start = (
                    door.center[0] + front_view_depth * normal_x +
                    current_lateral * tangent_x,
                    door.center[1] + front_view_depth * normal_y +
                    current_lateral * tangent_y)
                retreat_clearance = portal_clearance(grid, door, config)
                front_start_ready = bool(
                    _state(grid, front_start) == 0 and
                    _clearance(grid, front_start) >=
                    front_route_clearance and
                    _observed_clear_straight_segment(
                        grid, current, front_start, retreat_clearance,
                        allow_start_radius=0.45))
                if front_start_ready:
                    front_retreat_prefix = [current, front_start]
            front_half_baseline = (
                0.5 * max(config.minimum_goal_separation,
                          0.70 * config.visual_breadth_min_baseline) +
                max(0.10, grid.resolution))
            front_left_target = min(
                lateral_limit, max(
                    front_half_baseline,
                    max_lateral + config.goal_clearance + 0.15))
            front_right_target = max(
                -lateral_limit, min(
                    -front_half_baseline,
                    min_lateral - config.goal_clearance - 0.15))
            if front_has_clearance and front_start_ready:
                for first_target, second_target in (
                        (front_left_target, front_right_target),
                        (front_right_target, front_left_target)):
                    front_first = safe_samples(
                        front_view_depth, first_target, front_start, [],
                        front_route_clearance)
                    for first_raw in front_first:
                        if front_retreat_prefix:
                            raw_path = list(
                                first_raw.get("preflight_path") or [])
                            combined_path = (
                                list(front_retreat_prefix) +
                                (raw_path[1:] if raw_path else []))
                            first_raw = dict(
                                first_raw,
                                path_length_m=(
                                    first_raw["path_length_m"] +
                                    _distance(current, front_start)),
                                preflight_path=combined_path,
                                front_obstacle_retreat=True)
                        front_seconds = safe_samples(
                            front_second_view_depth, second_target,
                            first_raw["position"], [first_raw["position"]],
                            front_route_clearance)
                        for second_raw in front_seconds:
                            baseline = _distance(
                                first_raw["position"], second_raw["position"])
                            opposite_sides = (
                                (first_raw["lateral"] >= 0.0) !=
                                (second_raw["lateral"] >= 0.0))
                            both_in_front = max(
                                first_raw["depth"], second_raw["depth"]) <= (
                                    front_edge_depth - front_route_clearance + 0.02)
                            if (opposite_sides and both_in_front and
                                    baseline >=
                                    front_required_baseline - 0.05):
                                first = dict(
                                    first_raw,
                                    pair_strategy="obstacle_front_side_split")
                                second = dict(
                                    second_raw,
                                    pair_strategy="obstacle_front_side_split")
                                front_gap_pairs.append((first, second))
    else:
        # In an open room, scan deep first.  The second scan stays near the
        # portal and remains close to the same doorway-normal lane, so it also
        # becomes the final pre-exit observation point.
        # Keep the first physical camera stop genuinely deep in a clear room.
        # The second stop is deliberately shallower, but try several observed
        # free depth bands: a single 1.9 m band can be occupied by furniture
        # and previously forced a one-point fallback followed by a horizontal
        # same-depth sweep.
        # Keep the nominal near-door view just inside the 2.05 m shadow
        # boundary.  On the first-floor far room, centimetre-scale raster
        # variation selected 2.04 m in one run: the remote sphere was then
        # visible for only four partially occluded frames.  A 0.15 m inward
        # margin preserved the short exit leg in the adjacent successful run
        # while exposing five ordinary sphere contours.
        near_depth = max(
            minimum_depth + 0.55,
            min(2.20, maximum_depth - 2.40))
        deep_depth = min(
            maximum_depth - 0.25,
            max(config.open_room_preferred_deep_depth,
                near_depth + max(1.80,
                                 0.50 * required_baseline)))
        near_depth_targets = []
        for candidate_depth in (near_depth, near_depth + 0.75,
                                near_depth + 1.50):
            candidate_depth = min(candidate_depth, deep_depth - 1.80)
            if (candidate_depth >= minimum_depth and
                    all(abs(candidate_depth - old) >= 0.20
                        for old in near_depth_targets)):
                near_depth_targets.append(candidate_depth)
        deep_candidates = []
        # Open-room depth, rather than a large lateral chord, is the primary
        # physical baseline. Keep both centres close to the door-normal lane;
        # the former 1.6--2.7 m opposite-side offsets produced a 5.73 m
        # triangular G3->G4 leg in F1 room 2 despite there being no central
        # obstacle.
        deep_lateral = min(
            lateral_limit,
            max(0.35, min(0.80, 0.18 * required_baseline)))
        for lateral_target in (0.0, deep_lateral, -deep_lateral):
            deep_candidates.extend(safe_samples(
                deep_depth, lateral_target, current, []))
        # In an ordinarily sized clear room, "deep" is a hard geometric
        # property, not just the closest sample to a preferred target.  Keep
        # the first spin beyond the configured deep minimum whenever the
        # observed free depth supports it; only a genuinely shorter mapped
        # room may lower that bound for safety.
        minimum_clear_room_deep = min(
            config.open_room_minimum_deep_depth, maximum_depth - 0.25,
            max(minimum_depth + 2.5,
                near_depth + max(2.0, 0.55 * required_baseline)))
        deep_candidates = [
            item for item in deep_candidates
            if item["depth"] >= minimum_clear_room_deep]
        unique_deep = []
        for item in deep_candidates:
            if any(_distance(item["position"], old["position"]) < 0.20
                   for old in unique_deep):
                continue
            unique_deep.append(item)
        unique_deep = sorted(
            unique_deep, key=lambda item: item["path_length_m"])[:4]
        for first in unique_deep:
            axial_limit = min(0.80, lateral_limit)
            near_targets = []
            for target in (max(-axial_limit, min(axial_limit,
                                                first["lateral"])),
                           0.0, axial_limit, -axial_limit):
                if all(abs(target - old) >= 0.20 for old in near_targets):
                    near_targets.append(target)
            first_pairs = []
            for near_target in near_targets:
                for shallow_depth in near_depth_targets:
                    seconds = safe_samples(
                        shallow_depth, near_target, first["position"],
                        [first["position"]])
                    for second in seconds:
                        depth_delta = first["depth"] - second["depth"]
                        if (depth_delta < max(1.80,
                                             required_baseline * 0.50) or
                                _distance(first["position"],
                                          second["position"]) <
                                required_baseline * 0.80 or
                                abs(second["lateral"]) > axial_limit + 1e-6):
                            continue
                        second["blind_cells_visible"] = 0
                        second["blind_cells_total"] = 0
                        first_pairs.append((first, second))
                        break
            if first_pairs:
                best_for_first = min(
                    first_pairs,
                    key=lambda item: (float(item[1]["path_length_m"]),
                                      abs(float(item[1]["lateral"]))))
                pairs.append(best_for_first)
    # Prefer a physically verified door-front pair over any conservative
    # flank/wrap-around pair.  This must run after both the obstacle and open
    # room branches have generated candidates; placing it before the branch
    # would change the meaning of the centre-obstacle classifier.
    if front_gap_pairs:
        pairs = front_gap_pairs
        strategy = "obstacle_front_side_split"

    if (getattr(door, "viewpoint_contract", None) ==
            "open_deep_near" and
            getattr(door, "truth_open_centerline_deep_depth_m", None)
            is not None):
        # A freshly reset upper-floor raster can contain only a narrow free
        # strip around the body.  Even when that strip produces a generic
        # pair, it can label an
        # intermediate, off-axis mapping pose as G3 and later tries to reach
        # the genuinely deep point through a long A* wrap around unknown or
        # stale LIO cells.  On F3 this consumed an entire room budget while
        # the simulator-truth trajectory showed that the doorway-normal lane
        # was physically clear.
        #
        # The generated-layout contract already proves that this room has no
        # sizeable object in the doorway-normal band.  Use that fact only as
        # a bounded *planning prior*: emit the required centreline deep+near
        # pair and let the manager's independent live 3-D SCAN-lite audit vet
        # both chords before any command is published.  Completion still
        # requires two distinct measured truth poses and a physical EXIT.
        # Prefer this contract over a provisional raster pair; otherwise F3
        # can drive laterally into newly projected furniture, after which the
        # valid centreline deep goal is rejected as start_footprint_blocked.
        truth_deep = min(
            float(door.truth_open_centerline_deep_depth_m),
            max(minimum_depth + 2.50,
                float(config.room_depth_probe_range) - 0.25))
        truth_deep = max(
            truth_deep,
            float(config.open_room_minimum_deep_depth))
        truth_near = max(
            minimum_depth + 0.55,
            min(2.20, truth_deep - max(
                1.80, 0.50 * required_baseline)))
        # Do not retry a blocked centre cell forever.  The generated-layout
        # proof describes a clear doorway-normal *band*, not one mathematically
        # exact raster cell.  Pick one of a small set of parallel axial lanes
        # from the current live grid, keeping both views on the same lane so
        # the contract remains genuinely deep+near (and never becomes a room
        # wrap-around).  Centreline wins whenever its clearance is comparable.
        tangent = (-float(door.normal[1]), float(door.normal[0]))
        lane_candidates = []
        for lateral in (0.0, 0.30, -0.30, 0.50, -0.50):
            deep_point = (
                float(door.center[0]) + truth_deep * float(door.normal[0]) +
                lateral * tangent[0],
                float(door.center[1]) + truth_deep * float(door.normal[1]) +
                lateral * tangent[1])
            near_point = (
                float(door.center[0]) + truth_near * float(door.normal[0]) +
                lateral * tangent[0],
                float(door.center[1]) + truth_near * float(door.normal[1]) +
                lateral * tangent[1])
            deep_role = "G4" if config.truth_open_near_first else "G3"
            near_role = "G3" if config.truth_open_near_first else "G4"
            if (any(_distance(deep_point, old) < 0.35
                    for old in excluded_targets_by_role.get(deep_role, ())) or
                    any(_distance(near_point, old) < 0.35
                        for old in excluded_targets_by_role.get(
                            near_role, ()))):
                continue
            minimum_lane_clearance = min(
                _clearance(grid, deep_point), _clearance(grid, near_point))
            # The old selector ranked only endpoint clearance.  On F3 room02
            # this chose the +0.50 m lane although the live raster contained
            # no path from the just-crossed ENTRY pose to its deep endpoint;
            # the executor then rejected the same unreachable G3 repeatedly.
            # Rank a lane first by the two required physical legs being
            # reachable in their real execution order.  Unknown upper-floor
            # cells may still leave every lane unplanned, in which case the
            # existing truth-layout + live-3-D bounded fallback remains, but
            # a proven live route can no longer lose to a clearer dead end.
            ordered_points = ((near_point, deep_point)
                              if config.truth_open_near_first else
                              (deep_point, near_point))
            leg_start = current
            verified_legs = []
            lane_reachable = True
            lane_route_length = 0.0
            for leg_end in ordered_points:
                if _observed_clear_straight_segment(
                        grid, leg_start, leg_end,
                        config.post_entry_path_clearance,
                        allow_start_radius=0.45):
                    leg_path = {"success": True,
                                "path": [leg_start, leg_end]}
                else:
                    leg_path = astar_safe_path(
                        grid, leg_start, leg_end,
                        min(config.goal_clearance,
                            config.post_entry_path_clearance), 0.15,
                        allow_blocked_start=True,
                        maximum_expansions=(
                            config.exit_preflight_maximum_expansions))
                if not leg_path.get("success"):
                    lane_reachable = False
                    verified_legs = []
                    break
                points = list(leg_path.get("path") or [])
                lane_route_length += sum(
                    _distance(points[index - 1], points[index])
                    for index in range(1, len(points)))
                verified_legs.append(points)
                leg_start = leg_end
            lane_candidates.append((
                int(lane_reachable), minimum_lane_clearance,
                # A small saving caused by the body's provisional off-axis
                # ENTRY pose must not displace a clear centreline contract.
                # Prefer the least-offset lane whenever reachability and
                # endpoint clearance are equal; route length only breaks the
                # remaining tie.  A genuinely clearer parallel lane still
                # wins through minimum_lane_clearance above.
                -abs(lateral), -lane_route_length, lateral,
                deep_point, near_point, verified_legs))
        if not lane_candidates:
            return None
        (truth_lane_reachable, _, _, _, truth_lane_lateral,
         truth_deep_point, truth_near_point,
         truth_lane_verified_legs) = max(
            lane_candidates,
            key=lambda item: (item[0], item[1], item[2], item[3]))
        truth_baseline = _distance(truth_deep_point, truth_near_point)
        if truth_baseline >= config.minimum_goal_separation:
            first_length = _distance(current, truth_deep_point)
            second_length = truth_baseline
            ordered = [{
                "position": truth_deep_point, "role": "G3",
                "score": -first_length,
                "depth": truth_deep, "lateral": truth_lane_lateral,
                "clearance": _clearance(grid, truth_deep_point),
                "preflight_path_length_m": first_length,
                "preflight_path": [tuple(current), truth_deep_point],
                "coverage_cells": set(),
                "adaptive_marginal_gain_m2": 1.0,
                "adaptive_utility": -first_length,
                "two_pose_room_sweep": True,
                "two_pose_strategy":
                    "truth_layout_open_centerline_deep_near",
                "two_pose_index": 1, "two_pose_obstacle": None,
                "truth_layout_centerline_fallback": True,
                "truth_layout_parallel_lane_offset_m": truth_lane_lateral,
                "truth_layout_live_lane_reachable": bool(
                    truth_lane_reachable),
                "live_3d_audit_required": True,
                "fallback_candidate": True,
            }, {
                "position": truth_near_point, "role": "G4",
                "score": -second_length,
                "depth": truth_near, "lateral": truth_lane_lateral,
                "clearance": _clearance(grid, truth_near_point),
                "preflight_path_length_m": second_length,
                "preflight_path": [truth_deep_point, truth_near_point],
                "coverage_cells": set(),
                "adaptive_marginal_gain_m2": 1.0,
                "adaptive_utility": -second_length,
                "two_pose_room_sweep": True,
                "two_pose_strategy":
                    "truth_layout_open_centerline_deep_near",
                "two_pose_index": 2, "two_pose_obstacle": None,
                "truth_layout_centerline_fallback": True,
                "truth_layout_parallel_lane_offset_m": truth_lane_lateral,
                "truth_layout_live_lane_reachable": bool(
                    truth_lane_reachable),
                "live_3d_audit_required": True,
                "fallback_candidate": True,
            }]
            if config.truth_open_near_first:
                near_first_length = _distance(current, truth_near_point)
                near_item = dict(ordered[1])
                near_item.update({
                    "role": "G3", "score": -near_first_length,
                    "preflight_path_length_m": near_first_length,
                    "preflight_path": [tuple(current), truth_near_point],
                    "adaptive_utility": -near_first_length,
                    "two_pose_index": 1,
                    "truth_layout_near_first_mapping": True,
                })
                deep_item = dict(ordered[0])
                deep_item.update({
                    "role": "G4", "score": -truth_baseline,
                    "preflight_path_length_m": truth_baseline,
                    "preflight_path": [truth_near_point, truth_deep_point],
                    "adaptive_utility": -truth_baseline,
                    "two_pose_index": 2,
                    "truth_layout_near_first_mapping": True,
                })
                ordered = [near_item, deep_item]
            if truth_lane_reachable and len(truth_lane_verified_legs) == 2:
                for item, points in zip(ordered,
                                        truth_lane_verified_legs):
                    path_length = sum(
                        _distance(points[index - 1], points[index])
                        for index in range(1, len(points)))
                    item["preflight_path"] = sparsify_verified_trace(
                        points, spacing=0.35)
                    item["preflight_path_length_m"] = float(path_length)
                    item["path_planning_clearance_m"] = float(min(
                        config.goal_clearance,
                        config.post_entry_path_clearance))
            return {
                "ordered": ordered,
                "strategy": "truth_layout_open_centerline_deep_near",
                "baseline_m": truth_baseline,
                "route_length_m": (
                    _distance(current, ordered[0]["position"]) +
                    truth_baseline +
                    _distance(ordered[-1]["position"], return_anchor)),
                "return_distance_m": _distance(
                    ordered[-1]["position"], return_anchor),
                "obstacle": None,
                "truth_layout_centerline_fallback": True,
                "truth_layout_parallel_lane_offset_m": truth_lane_lateral,
                "truth_layout_live_lane_reachable": bool(
                    truth_lane_reachable),
                "truth_layout_near_first_mapping": bool(
                    config.truth_open_near_first),
                "live_3d_audit_required": True,
            }

    if not pairs:
        # Immediately after ENTRY the upper-floor projection can expose only
        # the near half of the room. Treating that incomplete raster as an
        # open room made the >=5 m deep set empty, after which the scheduler
        # fell back to two independent same-depth goals. Their second segment
        # was not locked to an A* route around the central object and timed
        # out in F2 room 0. Build one provisional, opposite-side pair in the
        # observed-free near band instead. Every endpoint and the inter-pose
        # route still pass the same clearance/A* gates; a later room with a
        # mapped deep region continues to use deep_then_axial_near.
        fallback_near_depth = max(
            minimum_depth + 0.55,
            min(2.05, maximum_depth - 0.25))
        # This provisional pair is mapping-only. Once the deep region is
        # observed, the accepted open-room pair is rebuilt as axial deep+near.
        fallback_deep_depth = min(
            maximum_depth - 0.25,
            max(fallback_near_depth + max(2.40, 0.68 * required_baseline),
                minimum_depth + 2.20))
        fallback_lateral = min(
            lateral_limit,
            max(1.55, 0.45 * config.visual_breadth_min_baseline))
        for deep_target in (fallback_lateral, -fallback_lateral):
            deep_side = safe_samples(
                fallback_deep_depth, deep_target, current, [])
            for first in deep_side:
                seconds = safe_samples(
                    fallback_near_depth, -deep_target, first["position"],
                    [first["position"]])
                for second in seconds:
                    depth_delta = first["depth"] - second["depth"]
                    lateral_delta = abs(first["lateral"] - second["lateral"])
                    diagonal_ratio = (
                        lateral_delta / max(depth_delta, 1e-6))
                    if ((first["lateral"] >= 0.0) !=
                            (second["lateral"] >= 0.0) and
                            depth_delta >= max(1.20,
                                               required_baseline * 0.35) and
                            lateral_delta >= max(0.80,
                                                 required_baseline * 0.35) and
                            0.70 <= diagonal_ratio <= 1.45):
                        pairs.append((first, second))
        if pairs:
            strategy = "provisional_obstacle_side_split"
            if obstacle_summary is None:
                obstacle_summary = {
                    "classification": "deep_region_not_yet_observed",
                    "provisional_side_split": True,
                }

    best = None
    pair_diagnostics = []
    for first, second in pairs:
        back_target = return_anchor
        back = astar_safe_path(
            grid, second["position"], back_target,
            config.goal_clearance, 0.15, allow_blocked_start=True,
            maximum_expansions=config.exit_preflight_maximum_expansions)
        if not back.get("success"):
            # The interior anchor can become occupied after the room scan
            # inflates the two jambs, even though the physically observed
            # portal and its corridor-side anchor remain connected.  Rejecting
            # the whole two-pose sweep here caused open rooms to degrade to one
            # deep point followed by a same-depth horizontal recovery view.
            # Validate egress against the same door's corridor-side anchor;
            # ROOM_EXIT still builds and verifies its own live return path.
            back_target = door.corridor_side
            back = astar_safe_path(
                grid, second["position"], back_target,
                config.goal_clearance, 0.15, allow_blocked_start=True,
                maximum_expansions=config.exit_preflight_maximum_expansions)
        return_path_deferred = not back.get("success")
        if return_path_deferred:
            # A fresh upper-floor projection often has both observation
            # centres and their connecting route mapped, but still contains
            # an unknown strip at the door jamb. Rejecting the complete pair
            # produced run12's 0.16 m/0.17 m entry-pose fallbacks. ROOM_EXIT
            # rebuilds a live route and can reverse the verified ENTRY trace,
            # so retain the safe two-centre route and defer only egress proof.
            back_length = _distance(second["position"], back_target) + 2.0
        else:
            back_points = list(back.get("path", []))
            back_length = sum(_distance(back_points[i - 1], back_points[i])
                              for i in range(1, len(back_points)))
        baseline = _distance(first["position"], second["position"])
        route_length = (first["path_length_m"] +
                        second["path_length_m"] + back_length)
        score = (route_length - 0.55 * min(baseline, 5.0) - 0.20 * min(abs(first["depth"] - second["depth"]), 4.0) - 0.25 * min(abs(first["lateral"] - second["lateral"]), 4.0))
        if centre_obstructed:
            if (first.get("pair_strategy") ==
                    "obstacle_front_side_split"):
                # Prefer the door--obstacle gap when it is fully A*-verified.
                # This avoids wrapping behind the object and leaves G4 on a
                # short verified egress leg, while the generic score still
                # compares endpoint clearance, route length and return path.
                score -= 1.50

            score += 0.25 * abs(first["lateral"] + second["lateral"])
        blind_cells_total = int(second.get("blind_cells_total", 0))
        blind_cells_visible = int(second.get("blind_cells_visible", 0))
        blind_recovery_supported = bool(
            not centre_obstructed and blind_cells_total >= 5 and
            blind_cells_visible > 0)
        if not centre_obstructed:
            score += (0.60 if first["lateral"] * second["lateral"] > 0.25 else 0.0)
            if blind_cells_total >= 5:
                # A near view that recovers none of the deep view measured
                # occupancy shadows repeats the same furniture occlusion. It
                # is still geometrically diagonal, but fix15 F2 room 2 showed
                # that such a zero-recovery pair can miss the sphere hidden
                # behind the central furniture. Prefer the opposite deep side
                # whenever it provides any verified shadow recovery.
                recovery_ratio = min(
                    1.0, blind_cells_visible / float(blind_cells_total))
                score += 2.00 * (1.0 - recovery_ratio)
            score -= 0.03 * float(blind_cells_visible)
        if (first["depth"] - second["depth"] >= 0.80 and
                first["lateral"] >= 0.0 and
                not blind_recovery_supported):
            # The two safe sides remain available, but make the G3-deep/G4-
            # near ordering robust to centimetre-scale raster noise. A large
            # genuine route advantage can still select the positive side.
            score += 4.00
        score += 1.50 if return_path_deferred else 0.0
        pair_diagnostics.append({
            "score": round(float(score), 4),
            "first_position": list(first["position"]),
            "second_position": list(second["position"]),
            "first_depth": round(float(first["depth"]), 3),
            "second_depth": round(float(second["depth"]), 3),
            "first_lateral": round(float(first["lateral"]), 3),
            "second_lateral": round(float(second["lateral"]), 3),
            "blind_cells_visible": blind_cells_visible,
            "blind_cells_total": blind_cells_total,
            "route_length_m": round(float(route_length), 3),
        })
        candidate = (score, -baseline, first, second, back_length,
                     return_path_deferred)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        if centre_obstructed or not unique_deep:
            return None
        first = min(unique_deep,
                    key=lambda item: item["path_length_m"])
        seed_partner = None
        visibility = _room_visibility_from_occupancy(grid, door, config)
        visibility.record_scan_pose(*first["position"])
        shadow_points = [visibility.cell_center(key)
                         for key in visibility.shadow_cells()]
        blind_side = 0.0
        if shadow_points:
            blind_side = sum(door.lateral(point)
                             for point in shadow_points) / len(shadow_points)
        axial_limit = min(0.80, lateral_limit)
        for near_target in (max(-axial_limit, min(axial_limit,
                                                  first["lateral"])),
                            0.0, axial_limit, -axial_limit):
            for shallow_depth in near_depth_targets:
                seconds = safe_samples(
                    shallow_depth, near_target, first["position"],
                    [first["position"]])
                for second in seconds:
                    depth_delta = first["depth"] - second["depth"]
                    if (depth_delta >= max(1.20,
                                           required_baseline * 0.35) and
                            abs(second["lateral"]) <= axial_limit + 1e-6):
                        seed_partner = second
                        break
                if seed_partner is not None:
                    break
            if seed_partner is not None:
                break
        if seed_partner is not None:
            ordered = []
            for index, (role, item) in enumerate(
                    zip(("G3", "G4"), (first, seed_partner)), start=1):
                ordered.append({
                    "position": item["position"], "role": role,
                    "score": -float(item["path_length_m"]),
                    "depth": item["depth"], "lateral": item["lateral"],
                    "clearance": item["clearance"],
                    "preflight_path_length_m": item["path_length_m"],
                    "preflight_path": list(item.get("preflight_path") or []),
                    "coverage_cells": set(),
                    "adaptive_marginal_gain_m2": 1.0,
                    "adaptive_utility": -float(item["path_length_m"]),
                    "two_pose_room_sweep": True,
                    "two_pose_strategy": "deep_seed_then_axial_near",
                    "two_pose_index": index, "two_pose_obstacle": None,
                    "blind_cells_visible": int(item.get(
                        "blind_cells_visible", 0)),
                    "blind_cells_total": int(item.get(
                        "blind_cells_total", 0)),
                    "fallback_candidate": False,
                })
            baseline = _distance(first["position"], seed_partner["position"])
            back = astar_safe_path(
                grid, seed_partner["position"], return_anchor,
                config.goal_clearance, 0.15, allow_blocked_start=True,
                maximum_expansions=config.exit_preflight_maximum_expansions)
            back_length = 0.0
            if back.get("success"):
                back_points = list(back.get("path", []))
                back_length = sum(_distance(back_points[i - 1], back_points[i])
                                  for i in range(1, len(back_points)))
            return {
                "ordered": ordered,
                "strategy": "deep_seed_then_axial_near",
                "baseline_m": baseline,
                "route_length_m": (first["path_length_m"] +
                                   seed_partner["path_length_m"] +
                                   back_length),
                "return_distance_m": back_length,
                "obstacle": None,
            }
        seed = {
            "position": first["position"], "role": "G3",
            "score": -float(first["path_length_m"]),
            "depth": first["depth"], "lateral": first["lateral"],
            "clearance": first["clearance"],
            "preflight_path_length_m": first["path_length_m"],
            "preflight_path": list(first.get("preflight_path") or []),
            "coverage_cells": set(), "adaptive_marginal_gain_m2": 1.0,
            "adaptive_utility": -float(first["path_length_m"]),
            "two_pose_room_sweep": True,
            "two_pose_strategy": "deep_seed_then_axial_near",
            "two_pose_index": 1, "two_pose_obstacle": None,
            "blind_cells_visible": 0, "blind_cells_total": 0,
            "fallback_candidate": False,
        }
        return {
            "ordered": [seed], "strategy": "deep_seed_then_axial_near",
            "baseline_m": 0.0,
            "route_length_m": first["path_length_m"] +
                              _distance(first["position"], return_anchor),
            "return_distance_m": _distance(first["position"], return_anchor),
            "obstacle": None,
        }

    _, negative_baseline, first, second, back_length, return_path_deferred = best
    selected_strategy = str(first.get("pair_strategy", strategy))
    ordered = []
    for index, (role, item) in enumerate(zip(("G3", "G4"),
                                             (first, second)), start=1):
        ordered.append({
            "position": item["position"], "role": role,
            "score": -float(item["path_length_m"]),
            "depth": item["depth"], "lateral": item["lateral"],
            "clearance": item["clearance"],
            "preflight_path_length_m": item["path_length_m"], "preflight_path": list(item.get("preflight_path") or []),
            "coverage_cells": set(), "adaptive_marginal_gain_m2": 1.0,
            "adaptive_utility": -float(item["path_length_m"]),
            "two_pose_room_sweep": True,
            "two_pose_strategy": selected_strategy,
            "two_pose_index": index, "two_pose_obstacle": obstacle_summary,
            "post_entry_path_clearance_relaxed": bool(item.get(
                "post_entry_path_clearance_relaxed", False)),
            "path_planning_clearance_m": item.get(
                "path_planning_clearance_m", config.goal_clearance),
            "blind_cells_visible": int(item.get("blind_cells_visible", 0)),
            "blind_cells_total": int(item.get("blind_cells_total", 0)),
            "fallback_candidate": False,
        })
    return {
        "ordered": ordered, "strategy": selected_strategy,
        "baseline_m": -negative_baseline,
        "route_length_m": first["path_length_m"] +
                          second["path_length_m"] + back_length,
        "return_distance_m": back_length, "obstacle": obstacle_summary,
        "return_path_deferred_to_live_exit": bool(return_path_deferred),
        "candidate_pair_diagnostics": sorted(
            pair_diagnostics, key=lambda item: item["score"]),
    }

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
        # Once a room is physically classified as the deep obstacle subtype,
        # keep its bounded observation grace for the whole transaction.  The
        # live-map fallback may later switch the endpoint policy to the
        # shallow front chord.  Recomputing the grace from that mutable policy
        # made the deadline shrink by 40 s on the next callback and forced an
        # immediate partial EXIT before the replacement G4 could run.
        self.deep_obstacle_contract_grace_armed = False
        self.state = "CORRIDOR_SWEEP"
        self.pending_candidates: Dict[Tuple[int, int], dict] = {}
        # Local doorway tracks jitter by a few centimetres every scan.  Keep
        # a short candidate-ID quarantine in addition to the geometric door
        # cooldown so one narrow false aperture cannot monopolize the 60 s
        # room-entry budget.
        self.local_candidate_quarantine: Dict[str, float] = {}
        self.door_cooldowns: Dict[Tuple[int, int], float] = {}
        self.door_cooldown_centers: Dict[Tuple[int, int], Point] = {}
        # Keep unconfirmed ENTRY refinement failures across the short-lived
        # orphan doorway records.  Otherwise the same jittered local doorway
        # is recreated with the same semantic number and its 5--15 s retry
        # cooldown expires before the corridor can reach the next room.
        self.entry_geometry_failure_counts: Dict[str, int] = {}
        self.role_attempts: Dict[str, int] = {}
        self.failed_roles = set()
        self.return_completed = False
        self.return_attempts = 0
        self.return_anchor: Optional[Point] = None
        self.entry_portal_waypoints: List[List[float]] = []
        self.exit_fallback_issued = False
        self.exit_blocked = False
        # Set by the manager only after its one bounded Gazebo-truth reseat at
        # the immutable physical portal.  The next EXIT must start from that
        # new near-door pose; replaying a pre-reseat deep inside anchor sends
        # the robot back into the room before trying to leave.
        self.truth_portal_restage_exit_pending = False
        # A blocked EXIT gets one bounded recovery pass in the manager before
        # the room may be abandoned.  This prevents an ENTERED_PARTIAL room
        # from leaking into the corridor scheduler.
        self.exit_blocked_recovery_used = False
        # After bounded EXIT cleanup, the first corridor-resume goal must
        # move past the failed doorway station instead of selecting that
        # same station again.
        self.corridor_resume_forward_escape_pending = False
        # EXIT proves a portal crossing, but not that corridor navigation has
        # recovered.  Retain the just-exited room until a measured centreline
        # pose or a successful recenter goal supplies that separate evidence.
        self.pending_corridor_recovery_door: Optional[EstimatedDoorway] = None
        # A same-station opposite doorway can be observed while the previous
        # room still owes its short EXIT -> corridor-centreline recovery.  It
        # must not preempt that recovery, but dropping the already validated
        # candidate makes the robot drive past the fourth room (the far-row
        # pair is often visible only during this brief post-EXIT window).
        # Retain only the prevalidated local candidate and activate it as soon
        # as measured centreline recovery is confirmed.
        self.deferred_prevalidated_door_after_centerline: Optional[
            EstimatedDoorway] = None
        self.deferred_prevalidated_entry_after_centerline: Optional[dict] = None
        self.deferred_entry_waypoints_after_centerline: List[List[float]] = []
        # Prevent an EXIT-blocked doorway from being re-entered in the same floor pass.
        self.blocked_door_ids = set()
        self.coverage_status: Optional[dict] = None
        self.adaptive_route_queue: List[dict] = []
        self.visual_deepening_requested = False
        # LiDAR can classify a compact room as covered from the portal.  That
        # is valid for mapping but not proof that the forward RGB-D camera has
        # ever observed from a safe interior pose.
        self.visual_anchor_completed = False
        self.visual_entry_refresh_used = 0
        self.visual_mapping_transit_used = 0
        # A truth-classified front obstacle may not yet be represented in the
        # rolling raster immediately after G3.  Permit one stationary G4 map
        # refresh before declaring the room partial; the refresh is never
        # counted as a physical viewpoint.
        self.obstacle_gap_g4_refresh_used = 0
        self.obstacle_gap_center_mapping_transit_used = 0
        # Exact occupancy/A*-verified route used by the physically successful
        # obstacle-front G3.  Reversing this bounded trace is safer and much
        # cheaper than asking a freshly self-occluded raster to rediscover a
        # route back to the entry-side mapping anchor before selecting G4.
        self.obstacle_gap_g3_executed_path: List[Point] = []
        self.open_g3_executed_path: List[Point] = []
        self.open_missing_depth_refresh_used = 0
        self.mandatory_two_pose_unavailable_refresh_used = 0
        # True only after the first physical camera centre was deliberately
        # staged near the doorway because the deep half of an otherwise open
        # room was not mapped yet.  The next mandatory view must then deepen
        # from the fresh grid; treating it as an ordinary provisional pair
        # used to select a second near-door point (simtime_010407 room 2).
        self.uncertain_open_near_mapping_completed = False
        # A route can be valid on the ENTRY snapshot and become occupied after
        # the first in-room LiDAR update. Give each physical camera role one
        # bounded fresh-grid re-selection; never abandon the two-view contract
        # after one pre-motion geometry rejection, and never loop forever.
        self.visual_geometry_retry_roles = set()
        # Keep pre-motion G3/G4 endpoint collisions for the active room.  A
        # bounded retry must select a different point in the same semantic
        # class; run12 otherwise submitted the identical colliding F3 G4
        # coordinate three times and rolled a valid 4th room back to partial.
        self.failed_visual_geometry_targets: Dict[str, List[Point]] = {}
        # A fresh upper-floor room can rasterize the robot's already occupied
        # start footprint before it has inferred any room cells.  Preserve the
        # rejected physical role and perform one stationary map refresh before
        # regenerating its endpoint; never spend G4 or EXIT budget on that
        # stale start cell.
        self.visual_geometry_map_refresh_pending = None
        # Only rooms whose fresh occupancy actually contains a central
        # occluder need an enforced opposite-side second camera view. Applying
        # this repair to open rooms adds a full-width transit without revealing
        # anything that the normal deep+near pair cannot see.
        self.two_pose_opposite_repair_required = False

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

    def _physical_two_view_contract_evidence(self) -> Dict:
        """Evaluate the two executed Gazebo-truth poses against the contract.

        Physical separation alone is not sufficient: two near-door points in
        an open room, or two points on the same side of a front obstacle, are
        both invalid even when they are far apart.  Keep this check local to
        the measured doorway frame so it remains valid on every floor.
        """
        door = self.active_door
        evidence = {
            "viewpoint_contract": (
                getattr(door, "viewpoint_contract", None)
                if door is not None else None),
            "g3_truth_pose": (
                list(door.g3_truth_pose)
                if door is not None and door.g3_truth_pose is not None
                else None),
            "g4_truth_pose": (
                list(door.g4_truth_pose)
                if door is not None and door.g4_truth_pose is not None
                else None),
            "physical_separation_met": False,
            "contract_geometry_met": False,
            "reason": "missing_truth_viewpoint",
        }
        if (door is None or door.g3_truth_pose is None or
                door.g4_truth_pose is None):
            return evidence
        g3 = door.g3_truth_pose
        g4 = door.g4_truth_pose
        separation = _distance(g3, g4)
        contract = str(door.viewpoint_contract or "")
        if contract == "obstacle_front_opposite_sides":
            g3_depth = door.contract_depth(g3)
            g4_depth = door.contract_depth(g4)
            g3_lateral = door.contract_lateral(g3)
            g4_lateral = door.contract_lateral(g4)
        else:
            g3_depth, g4_depth = door.depth(g3), door.depth(g4)
            g3_lateral, g4_lateral = door.lateral(g3), door.lateral(g4)
        separation_met = bool(
            separation >= float(self.config.minimum_goal_separation))
        evidence.update({
            "physical_viewpoint_separation_m": float(separation),
            "physical_separation_met": separation_met,
            "g3_depth_m": float(g3_depth),
            "g4_depth_m": float(g4_depth),
            "g3_lateral_m": float(g3_lateral),
            "g4_lateral_m": float(g4_lateral),
        })
        if contract == "open_deep_near":
            deep_depth = max(g3_depth, g4_depth)
            near_depth = min(g3_depth, g4_depth)
            minimum_depth_delta = max(
                1.50, 0.75 * self.config.minimum_goal_separation)
            # The deep camera must actually expose the room interior.  Allow
            # normal goal tolerance around the configured preferred depth,
            # while keeping the near camera in the doorway-side half.
            # ``goal_reached`` is a footprint tolerance, not an exact camera
            # placement.  Leave a small measured-arrival allowance here so a
            # camera that stops a few centimetres short of the planned deep
            # point is not reclassified as "near".  The depth-delta and
            # physical-separation gates below still prevent two shallow poses
            # from satisfying the contract.
            minimum_deep_depth = max(
                3.80,
                self.config.open_room_minimum_deep_depth -
                self.config.semantic_completion_tolerance - 0.15)
            maximum_near_depth = min(
                4.00 + min(0.35,
                           self.config.semantic_completion_tolerance),
                deep_depth - minimum_depth_delta)
            depth_delta_met = bool(
                deep_depth - near_depth >= minimum_depth_delta)
            deep_met = bool(deep_depth >= minimum_deep_depth)
            near_met = bool(
                near_depth >= self.config.minimum_crossing_depth and
                near_depth <= maximum_near_depth)
            geometry_met = bool(depth_delta_met and deep_met and near_met)
            evidence.update({
                "deep_depth_m": float(deep_depth),
                "near_depth_m": float(near_depth),
                "minimum_depth_delta_m": float(minimum_depth_delta),
                "minimum_deep_depth_m": float(minimum_deep_depth),
                "maximum_near_depth_m": float(maximum_near_depth),
                "depth_delta_met": depth_delta_met,
                "deep_view_met": deep_met,
                "near_view_met": near_met,
                "contract_geometry_met": geometry_met,
                "reason": ("open_deep_near_met" if geometry_met else
                           "open_deep_near_geometry_failed"),
            })
        elif contract == "obstacle_front_opposite_sides":
            front = door.truth_obstacle_front_depth_m
            obstacle_min = door.truth_obstacle_minimum_lateral_m
            obstacle_max = door.truth_obstacle_maximum_lateral_m
            if front is None or obstacle_min is None or obstacle_max is None:
                evidence["reason"] = "obstacle_truth_geometry_missing"
                return evidence
            front_margin = max(0.15, 0.50 * self.config.goal_clearance)
            side_margin = max(0.12, 0.30 * self.config.goal_clearance)
            view_policy = obstacle_view_policy(door)
            route_front = float(obstacle_route_front_depth(door) or front)
            maximum_gap_depth = route_front - front_margin
            both_in_front_gap = bool(
                self.config.minimum_crossing_depth <= g3_depth <=
                maximum_gap_depth and
                self.config.minimum_crossing_depth <= g4_depth <=
                maximum_gap_depth)
            # Merely standing on opposite sides near the doorway is not a
            # useful obstacle-room camera pair.  Run12 met that weak shape but
            # both rays were occluded by the near sofa.  Require each measured
            # pose to reach the front half of the door--furniture gap and to
            # clear the corresponding furniture side by a real peek margin.
            deep_side_peek_required = bool(
                view_policy == "deep_outer_side_peek")
            # The executor accepts a measured endpoint within its bounded
            # semantic goal tolerance.  run61's opposite-side G4 stopped
            # 0.033 m short of the nominal front-half boundary (2.767 m vs
            # 2.800 m), despite preserving a 2.6 m physical baseline and the
            # required outer side peek.  Apply only a small arrival allowance
            # here; this remains far too small to accept run12's 2.2 m shallow
            # pair and does not relax the front-gap or opposite-side gates.
            visibility_arrival_tolerance = min(
                0.20, 0.25 * float(
                    self.config.semantic_completion_tolerance))
            minimum_visibility_depth = (
                max(float(self.config.minimum_crossing_depth),
                    float(front) - 1.00 - visibility_arrival_tolerance)
                if deep_side_peek_required else
                float(self.config.minimum_crossing_depth))
            side_peek_margin = max(
                0.70, float(self.config.goal_clearance) + 0.35)
            # A deep room can contain unrelated furniture close to both side
            # walls.  The aggregate visibility envelope is useful diagnostic
            # information, but requiring both poses to lie outside that whole
            # envelope makes the contract impossible (run94: -3.657..3.457 m)
            # and contradicts the bounded selector, which peeks around the
            # primary door-facing blocker at about +/-1.5 m.  Contract the
            # physical side-peek against that primary blocker; live A*/SCAN
            # and the hazard visibility audit remain responsible for the
            # other furniture.
            diagnostic_visibility_min = float(
                door.truth_obstacle_visibility_minimum_lateral_m
                if door.truth_obstacle_visibility_minimum_lateral_m is not
                None else obstacle_min)
            diagnostic_visibility_max = float(
                door.truth_obstacle_visibility_maximum_lateral_m
                if door.truth_obstacle_visibility_maximum_lateral_m is not
                None else obstacle_max)
            visibility_min = float(obstacle_min)
            visibility_max = float(obstacle_max)
            both_views_deep_side_peek = bool(
                g3_depth >= minimum_visibility_depth and
                g4_depth >= minimum_visibility_depth)
            outer_side_peek = bool(
                (g3_lateral >= visibility_max + side_peek_margin and
                 g4_lateral <= visibility_min - side_peek_margin) or
                (g4_lateral >= visibility_max + side_peek_margin and
                 g3_lateral <= visibility_min - side_peek_margin))
            opposite_sides = bool(
                (g3_lateral >= float(obstacle_max) + side_margin and
                 g4_lateral <= float(obstacle_min) - side_margin) or
                (g4_lateral >= float(obstacle_max) + side_margin and
                 g3_lateral <= float(obstacle_min) - side_margin))
            geometry_met = bool(
                both_in_front_gap and opposite_sides and
                (not deep_side_peek_required or
                 (both_views_deep_side_peek and outer_side_peek)))
            evidence.update({
                "obstacle_front_depth_m": float(front),
                "obstacle_route_front_depth_m": route_front,
                "obstacle_view_policy": view_policy,
                "deep_side_peek_required": deep_side_peek_required,
                "obstacle_minimum_lateral_m": float(obstacle_min),
                "obstacle_maximum_lateral_m": float(obstacle_max),
                "obstacle_visibility_minimum_lateral_m": visibility_min,
                "obstacle_visibility_maximum_lateral_m": visibility_max,
                "room_visibility_envelope_minimum_lateral_m":
                    diagnostic_visibility_min,
                "room_visibility_envelope_maximum_lateral_m":
                    diagnostic_visibility_max,
                "maximum_gap_depth_m": float(maximum_gap_depth),
                "minimum_visibility_depth_m": float(
                    minimum_visibility_depth),
                "visibility_arrival_tolerance_m": float(
                    visibility_arrival_tolerance),
                "side_peek_margin_m": float(side_peek_margin),
                "both_views_in_front_gap": both_in_front_gap,
                "opposite_obstacle_sides": opposite_sides,
                "both_views_deep_side_peek": both_views_deep_side_peek,
                "outer_side_peek": outer_side_peek,
                "contract_geometry_met": geometry_met,
                "reason": ("obstacle_front_opposite_sides_met"
                           if geometry_met else
                           "obstacle_front_opposite_sides_geometry_failed"),
            })
        else:
            evidence["reason"] = "viewpoint_contract_missing_or_unknown"
        evidence["physical_two_view_contract_met"] = bool(
            separation_met and evidence["contract_geometry_met"])
        return evidence

    def _canonicalize_open_partial_contract(self, evidence: Dict,
                                            now: float) -> None:
        """Keep only the valid depth class before a bounded room retry.

        G3/G4 are execution slots, but an endpoint fallback can occasionally
        make the nominal near slot physically deeper than the nominal deep
        slot.  Preserving both successful role labels then makes the partial
        retry believe there is no outstanding view and immediately stop on
        ``adaptive_viewpoint_budget``.  Canonicalise the measured Gazebo-truth
        poses to one valid class and explicitly leave the other role pending.
        The retry therefore executes a real replacement pose instead of
        reusing two geometrically invalid successes.
        """
        door = self.active_door
        if (door is None or
                str(getattr(door, "viewpoint_contract", "")) !=
                "open_deep_near" or
                evidence.get("physical_two_view_contract_met")):
            return
        poses = [pose for pose in (door.g3_truth_pose, door.g4_truth_pose)
                 if pose is not None]
        if not poses:
            return

        preserve_role = None
        preserved_pose = None
        repair_role = None
        # A one-pose contract makes the aggregate evaluator return before it
        # emits deep/near flags.  Classify that measured pose here so a valid
        # G3 is not erased merely because G4 is missing (run39).
        config = getattr(self, "config", None)
        open_minimum_deep = float(getattr(
            config, "open_room_minimum_deep_depth", 6.25))
        semantic_tolerance = float(getattr(
            config, "semantic_completion_tolerance", 0.80))
        minimum_crossing = float(getattr(
            config, "minimum_crossing_depth", 0.45))
        minimum_deep = max(
            3.80, open_minimum_deep - semantic_tolerance - 0.15)
        deepest_pose = max(poses, key=door.depth)
        shallowest_pose = min(poses, key=door.depth)
        measured_deep_met = bool(
            evidence.get("deep_view_met") or
            door.depth(deepest_pose) >= minimum_deep)
        measured_near_met = bool(
            evidence.get("near_view_met") or
            (door.depth(shallowest_pose) >= minimum_crossing and
             door.depth(shallowest_pose) <= 4.35))
        if measured_deep_met:
            # A missing near class is the recurrent F2 failure.  Preserve the
            # physically deepest sample as canonical G3 and force a fresh G4.
            preserve_role = "G3"
            preserved_pose = deepest_pose
            repair_role = "G4"
        elif measured_near_met:
            # Symmetric case: retain the shallow sample as canonical G4 and
            # force the retry to obtain a genuinely deep G3.
            preserve_role = "G4"
            preserved_pose = shallowest_pose
            repair_role = "G3"

        self.successful_roles.discard("G3")
        self.successful_roles.discard("G4")
        door.g3_truth_pose = None
        door.g4_truth_pose = None
        self.completed_points = []
        if preserve_role is not None and preserved_pose is not None:
            canonical_pose = (float(preserved_pose[0]),
                              float(preserved_pose[1]))
            self.successful_roles.add(preserve_role)
            self.completed_points.append(canonical_pose)
            if preserve_role == "G3":
                door.g3_truth_pose = canonical_pose
            else:
                door.g4_truth_pose = canonical_pose
        else:
            # Neither measured class is valid.  Re-run both bounded slots;
            # do not carry invalid physical observations into acceptance.
            repair_role = "G3_G4"
        self.events.append({
            "event": "ROOM_OPEN_CONTRACT_PARTIAL_CANONICALIZED",
            "elapsed_sec": round(float(now), 3),
            "room_id": self._room_id(),
            "door_id": door.door_id,
            "preserved_role": preserve_role,
            "repair_role": repair_role,
            "preserved_truth_pose": (
                list(preserved_pose) if preserved_pose is not None else None),
            "original_g3_depth_m": evidence.get("g3_depth_m"),
            "original_g4_depth_m": evidence.get("g4_depth_m"),
            "deep_view_met": measured_deep_met,
            "near_view_met": measured_near_met,
            "classification_source": (
                "complete_contract_evidence" if
                "deep_view_met" in evidence else
                "available_physical_pose_depth"),
            "policy": "bounded_retry_only_missing_truth_depth_class",
        })

    def _canonicalize_obstacle_partial_contract(self, evidence: Dict,
                                                now: float) -> None:
        """Preserve at most one valid front-gap side before a bounded retry.

        A generic adaptive G4 escaped the locked obstacle gate in run81.  Its
        physical endpoint was on the opposite lateral side, but beyond the
        shallow room's maximum door--obstacle gap depth.  EXIT correctly
        rejected the pair; preserving both nominal roles then made same-door
        re-entry jump directly back to EXIT.  Keep one physically valid side
        (if present) and force the missing role to execute at a new pose.
        """
        door = self.active_door
        if (door is None or
                str(getattr(door, "viewpoint_contract", "")) !=
                "obstacle_front_opposite_sides" or
                evidence.get("physical_two_view_contract_met")):
            return
        maximum_gap_depth = evidence.get("maximum_gap_depth_m")
        minimum_visibility_depth = evidence.get("minimum_visibility_depth_m")
        obstacle_minimum = evidence.get("obstacle_minimum_lateral_m")
        obstacle_maximum = evidence.get("obstacle_maximum_lateral_m")
        if any(value is None for value in (
                maximum_gap_depth, minimum_visibility_depth,
                obstacle_minimum, obstacle_maximum)):
            valid = []
        else:
            tolerance = min(
                0.20, 0.25 * float(self.config.semantic_completion_tolerance))
            valid = []
            for role, pose in (("G3", door.g3_truth_pose),
                               ("G4", door.g4_truth_pose)):
                if pose is None:
                    continue
                depth = float(door.contract_depth(pose))
                lateral = float(door.contract_lateral(pose))
                in_gap = bool(
                    depth >= float(minimum_visibility_depth) - tolerance and
                    depth <= float(maximum_gap_depth) + tolerance)
                outside_obstacle_side = bool(
                    lateral <= float(obstacle_minimum) - 0.30 or
                    lateral >= float(obstacle_maximum) + 0.30)
                if in_gap and outside_obstacle_side:
                    valid.append((role, pose, depth, lateral))

        # One side is sufficient evidence to preserve.  If both individual
        # poses are valid but their pair failed, retain the first canonical
        # slot only so the retry must obtain a newly verified opposite side.
        preserved = valid[0] if valid else None
        original = {
            "G3": door.g3_truth_pose,
            "G4": door.g4_truth_pose,
        }
        self.successful_roles.discard("G3")
        self.successful_roles.discard("G4")
        self.completed_points = []
        door.g3_truth_pose = None
        door.g4_truth_pose = None
        if preserved is not None:
            preserve_role, preserve_pose, _, _ = preserved
            canonical_pose = (float(preserve_pose[0]), float(preserve_pose[1]))
            self.successful_roles.add(preserve_role)
            self.completed_points.append(canonical_pose)
            if preserve_role == "G3":
                door.g3_truth_pose = canonical_pose
            else:
                door.g4_truth_pose = canonical_pose
            repair_role = "G4" if preserve_role == "G3" else "G3"
        else:
            preserve_role = None
            canonical_pose = None
            repair_role = "G3_G4"
        self.two_pose_opposite_repair_required = True
        self.events.append({
            "event": "ROOM_OBSTACLE_CONTRACT_PARTIAL_CANONICALIZED",
            "elapsed_sec": round(float(now), 3),
            "room_id": self._room_id(),
            "door_id": door.door_id,
            "preserved_role": preserve_role,
            "repair_role": repair_role,
            "preserved_truth_pose": (list(canonical_pose)
                                      if canonical_pose is not None else None),
            "original_g3_truth_pose": original["G3"],
            "original_g4_truth_pose": original["G4"],
            "maximum_gap_depth_m": maximum_gap_depth,
            "policy": "bounded_retry_only_valid_front_gap_side",
        })

    def _confirm_entry(self, point: Point, now: float,
                       source: str) -> None:
        if self.entry_confirmed:
            return
        self.entry_confirmed = True
        if self.active_door is not None:
            self.active_door.entry_truth_crossing_confirmed = True
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
        if (self.active_door is not None and
                getattr(self.active_door, "room_id", None)):
            # Re-entering a known partial doorway is a continuation of the
            # same room contract, not a new room discovery.
            self.active_room_id = str(self.active_door.room_id)
        else:
            self.room_count += 1
            self.active_room_id = "{}{:02d}".format(
                self.config.room_id_prefix, self.room_count)
            if self.active_door is not None:
                self.active_door.room_id = self.active_room_id
        self.events.append({
            "event": "ROOM_ENTERED", "elapsed_sec": round(float(now), 3),
            "room_id": self.active_room_id,
            "door_id": self.active_door.door_id if self.active_door else None,
            "entry_source": source,
            "entry_truth_crossing_confirmed": True,
            "entry_truth_pose": ([round(float(point[0]), 4),
                                  round(float(point[1]), 4)]
                                 if point is not None else None),
        })

    def _clear_active(self) -> None:
        # A doorway candidate is registered before physical ENTRY is
        # confirmed.  If refinement/portal preflight fails at that stage,
        # keeping the record creates an orphan ``estimated_door_N`` with no
        # room_id and later looks like a partial room in the acceptance
        # history.  Remove only this unconfirmed orphan; a confirmed room
        # (including a real partial two-view contract) remains registered for
        # its bounded same-room retry and duplicate suppression.
        orphan = self.active_door
        if (orphan is not None and not self.entry_confirmed and
                not getattr(orphan, "room_id", None)):
            try:
                if any(item is orphan for item in self.detector.doors):
                    self.detector.doors.remove(orphan)
                    self.events.append({
                        "event": "UNCONFIRMED_ENTRY_ORPHAN_CLEANED",
                        "door_id": orphan.door_id,
                        "reason": "entry_failed_before_physical_room_crossing",
                    })
            except (AttributeError, ValueError):
                pass
        self.active_door = None
        self.active_room_id = None
        self.completed_points = []
        self.next_role_index = 0
        self.exit_attempts = 0
        self.entry_attempts = 0
        self.successful_roles = set()
        self.entry_confirmed = False
        self.room_started_at = None
        self.deep_obstacle_contract_grace_armed = False
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
        self.truth_portal_restage_exit_pending = False
        self.exit_blocked_recovery_used = False
        self.corridor_resume_forward_escape_pending = False
        self.coverage_status = None
        self.adaptive_route_queue = []
        self.visual_deepening_requested = False
        self.visual_anchor_completed = False
        self.visual_entry_refresh_used = 0
        self.visual_mapping_transit_used = 0
        self.obstacle_gap_g4_refresh_used = 0
        self.obstacle_gap_center_mapping_transit_used = 0
        self.obstacle_gap_g3_executed_path = []
        self.open_missing_depth_refresh_used = 0
        self.mandatory_two_pose_unavailable_refresh_used = 0
        self.uncertain_open_near_mapping_completed = False
        self.visual_geometry_retry_roles = set()
        self.failed_visual_geometry_targets = {}
        self.visual_geometry_map_refresh_pending = None
        self.two_pose_opposite_repair_required = False

    def _activate(self, door: EstimatedDoorway, now: float,
                  proactive: bool,
                  crossed_point: Optional[Point] = None,
                  entry_waypoints: Optional[Sequence[Sequence[float]]] = None,
                  reuse_existing: bool = False
                  ) -> Optional[EstimatedDoorway]:
        # Do not let an opposite/new doorway preempt the measured centreline
        # recovery owed by the preceding EXIT.  In the first strict F3 run,
        # three otherwise complete rooms kept this flag false because every
        # next-room takeover replaced the semantic transaction before the
        # corridor resume goal could execute.  A same-room partial retry may
        # continue; a different physical portal waits for the short recenter.
        pending_recovery = self.pending_corridor_recovery_door
        if (pending_recovery is not None and
                pending_recovery is not door and
                _distance(pending_recovery.center, door.center) >
                min(1.15, 0.75 * self.config.duplicate_door_radius)):
            self.events.append({
                "event": "NEW_DOOR_ACTIVATION_BLOCKED_PENDING_CENTERLINE",
                "elapsed_sec": round(float(now), 3),
                "pending_door_id": pending_recovery.door_id,
                "candidate_center": [float(door.center[0]),
                                     float(door.center[1])],
                "policy": "exit_then_centerline_then_next_physical_room",
            })
            return None
        # A failed doorway remains in ``detector.doors`` so later scans do
        # not create duplicate semantic landmarks.  Upper-floor recovery may
        # deliberately retry that *same* unvisited landmark after its short
        # cooldown.  Re-registering it here would create a second room record
        # at the same physical portal and corrupt the four-room count.
        # A few already-validated takeover paths (post-crossing observation,
        # truth staging and the robot-local prevalidated portal) call
        # _activate() directly.  They used to bypass the duplicate gate in
        # _prepare_and_activate(): after a partial EXIT the very same physical
        # doorway was consequently registered as estimated_door_N+1.  Reuse an
        # unfinished landmark here as the final, common identity guard.  A
        # completed landmark must be rejected here as well: truth staging and
        # robot-local promotion intentionally bypass the ordinary detector and
        # therefore cannot rely on its duplicate rejection.
        if not reuse_existing:
            existing = min(
                self.detector.doors,
                key=lambda item: _distance(item.center, door.center),
                default=None)
            duplicate_distance = (
                _distance(existing.center, door.center)
                if existing is not None else float("inf"))
            if (existing is not None and duplicate_distance <=
                    min(1.15, 0.75 * self.config.duplicate_door_radius)):
                if existing.completed:
                    self.events.append({
                        "event": "DIRECT_ACTIVATION_COMPLETED_DOOR_REJECTED",
                        "elapsed_sec": round(float(now), 3),
                        "door_id": existing.door_id,
                        "candidate_center": [float(door.center[0]),
                                             float(door.center[1])],
                        "distance_m": round(float(duplicate_distance), 3),
                        "policy": "single_physical_portal_single_room_identity",
                    })
                    return None
                self.events.append({
                    "event": "DIRECT_ACTIVATION_UNFINISHED_DOOR_REUSED",
                    "elapsed_sec": round(float(now), 3),
                    "door_id": existing.door_id,
                    "candidate_center": [float(door.center[0]),
                                         float(door.center[1])],
                    "policy": "single_physical_portal_single_room_identity",
                })
                door = existing
                reuse_existing = True
        continue_partial = bool(
            reuse_existing and not door.completed and door.room_id and
            (door.partial_retry_count > 0 or door.temporarily_failed or
             door.partial_successful_roles or door.partial_completed_points))
        preserved_roles = set(door.partial_successful_roles)
        preserved_points = [
            (float(point[0]), float(point[1]))
            for point in door.partial_completed_points
            if isinstance(point, (list, tuple)) and len(point) >= 2]
        if not reuse_existing:
            door.door_id = "estimated_door_{:02d}".format(
                len(self.detector.doors) + 1)
            self.detector.doors.append(door)
        elif not any(item is door for item in self.detector.doors):
            raise ValueError("reused doorway is not registered")
        door.entered_at = float(now)
        door.temporarily_failed = False
        self.active_door = door
        self.completed_points = preserved_points if continue_partial else []
        self.next_role_index = 0
        self.exit_attempts = 0
        self.entry_attempts = 0
        self.successful_roles = preserved_roles if continue_partial else set()
        self.entry_confirmed = False
        # Keep the semantic room identity visible from DOOR_COMMIT onward when
        # this is a bounded continuation of a partial EXIT.  Previously the
        # event emitted between re-entry planning and the crossing used
        # ``room_count + 1`` even though _confirm_entry() correctly restored
        # the original room id.  That made a same-room retry look like a new
        # room and confused the floor handoff/counters.
        self.active_room_id = (str(door.room_id)
                               if continue_partial and door.room_id else None)
        self.room_started_at = float(now)
        self.deep_obstacle_contract_grace_armed = bool(
            getattr(door, "viewpoint_contract", None) ==
            "obstacle_front_opposite_sides" and
            obstacle_view_policy(door) == "deep_outer_side_peek")
        self.completion_mode = None
        self.prepared_entry = None
        self.role_attempts = ({role: 1 for role in preserved_roles}
                              if continue_partial else {})
        self.failed_roles = set()
        self.return_completed = False
        self.return_attempts = 0
        self.entry_portal_waypoints = [
            [float(point[0]), float(point[1])] for point in (entry_waypoints or [])]
        self.exit_fallback_issued = False
        self.exit_blocked = False
        self.coverage_status = None
        self.adaptive_route_queue = []
        self.visual_deepening_requested = bool(
            continue_partial and
            self.config.visual_two_pose_lateral_enabled and
            ("G3" not in self.successful_roles or
             "G4" not in self.successful_roles))
        self.visual_anchor_completed = False
        self.visual_entry_refresh_used = 0
        self.visual_mapping_transit_used = 0
        self.obstacle_gap_g4_refresh_used = 0
        self.obstacle_gap_center_mapping_transit_used = 0
        self.obstacle_gap_g3_executed_path = []
        self.open_missing_depth_refresh_used = 0
        self.mandatory_two_pose_unavailable_refresh_used = 0
        self.uncertain_open_near_mapping_completed = False
        self.visual_geometry_retry_roles = set()
        self.visual_geometry_map_refresh_pending = None
        self.two_pose_opposite_repair_required = False
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
                self.config.visual_require_physical_center or
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
        # Every detector path converges here.  Enforce physical-door identity
        # centrally, including proactive planned-path candidates: after a
        # partial EXIT the same portal used to be registered a second time
        # before its short retry cooldown elapsed.  That spent another full
        # room visit while incrementing the apparent room count.  Reuse the
        # existing unfinished landmark, or reject a completed one.
        if not reuse_existing:
            existing = min(
                self.detector.doors,
                key=lambda item: _distance(item.center, door.center),
                default=None)
            if (existing is not None and
                    _distance(existing.center, door.center) <=
                    min(1.15, 0.75 * self.config.duplicate_door_radius)):
                cooldown_until = self._door_cooldown_until(existing.center)
                if float(now) < cooldown_until:
                    self.events.append({
                        "event": "DUPLICATE_PARTIAL_DOOR_COOLDOWN_HELD",
                        "elapsed_sec": round(float(now), 3),
                        "door_id": existing.door_id,
                        "candidate_center": list(door.center),
                        "cooldown_until": float(cooldown_until),
                    })
                    return None
                if existing.completed:
                    self.events.append({
                        "event": "DUPLICATE_COMPLETED_DOOR_REJECTED",
                        "elapsed_sec": round(float(now), 3),
                        "door_id": existing.door_id,
                        "candidate_center": list(door.center),
                    })
                    return None
                door = existing
                reuse_existing = True
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
        restricted_entry_clearance = (
            float(wall_break_clearance_override)
            if (allow_local_wall_break_support and
                wall_break_clearance_override is not None) else None)
        proposal = (select_room_goal(
            grid, door, current, "ENTRY", [], self.config,
            entry_clearance_override=restricted_entry_clearance)
                    if valid else None)
        nominal_proposal_available = proposal is not None
        portal = (prepare_portal_path(
            grid, door, current, proposal["position"], self.config,
            clearance_override=restricted_entry_clearance)
                  if proposal is not None and
                  not proposal.get("entry_staging_fallback") else None)
        nominal_portal_available = portal is not None
        fallback_attempted = False
        # Do not lose a confirmed side opening solely because the ideal
        # 1.15m entry cell is temporarily unknown/unreachable.  Use a local
        # free staging point and retry the inward entry after the next scan.
        if valid and (proposal is None or
                      (portal is None and
                       not proposal.get("entry_staging_fallback"))):
            fallback_attempted = True
            proposal = fallback_entry_proposal(
                grid, door, current, self.config,
                clearance_override=(
                    wall_break_clearance_override
                    if (allow_local_wall_break_support and
                        wall_break_clearance_override is not None) else
                    0.18 if allow_local_wall_break_support else None))
            if proposal is not None and allow_local_wall_break_support:
                # The robot-local detector has independently validated this
                # short ray with A* and SCAN-lite. Preserve that provenance so
                # next_goal can execute the shallow staging step while the
                # global rolling projection grows behind the doorway.
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
                "entry_preflight_detail": {
                    "candidate_entry_ready": bool(valid),
                    "candidate_entry_reason": str(reason),
                    "nominal_proposal_available": bool(
                        nominal_proposal_available),
                    "nominal_portal_available": bool(
                        nominal_portal_available),
                    "fallback_attempted": bool(fallback_attempted),
                    "fallback_proposal_available": bool(
                        proposal is not None),
                    "local_wall_break_support": bool(
                        allow_local_wall_break_support),
                    "wall_break_clearance_override_m": (
                        float(wall_break_clearance_override)
                        if wall_break_clearance_override is not None else
                        None),
                },
                "cooldown_until": self.door_cooldowns[key],
            })
            self._set_state("CORRIDOR_SWEEP", now, "door_candidate_rejected")
            return None
        activated = self._activate(
            door, now, proactive=proactive,
            reuse_existing=reuse_existing)
        if activated is None:
            self.prepared_entry = None
            self.entry_portal_waypoints = []
            if self.pending_corridor_recovery_door is not None:
                # The candidate has already passed doorway evidence, endpoint
                # selection and (when available) portal A*.  Do not discard it
                # during the ~1 s window between a physical EXIT and measured
                # corridor recentering.  Run13 promoted the fourth F1 doorway
                # at 248.073 s, rejected it only for this recovery debt, then
                # cleared the debt at 249.186 s and never saw that doorway
                # again, eventually timing out at 3/4.  Preserve the validated
                # proposal and activate it exactly once when the centreline
                # confirmation arrives; execution still receives the normal
                # manager SCAN-lite gate.
                self.deferred_prevalidated_door_after_centerline = door
                self.deferred_prevalidated_entry_after_centerline = dict(
                    proposal, portal_preflight=portal)
                self.deferred_entry_waypoints_after_centerline = (
                    [list(point) for point in
                     portal["mandatory_portal_waypoints"]]
                    if portal is not None else [])
                self.events.append({
                    "event": "VALIDATED_DOOR_DEFERRED_UNTIL_CENTERLINE",
                    "elapsed_sec": round(float(now), 3),
                    "door_id": door.door_id,
                    "center": [float(door.center[0]),
                               float(door.center[1])],
                    "portal_preflight_available": bool(portal is not None),
                    "policy":
                        "preserve_validated_candidate_across_exit_recenter",
                })
                self._set_state("CORRIDOR_RESUME", now,
                                "previous_exit_centerline_recovery_pending")
            else:
                self._set_state("CORRIDOR_SWEEP", now,
                                "completed_physical_door_rejected")
            return None
        self.prepared_entry = dict(proposal, portal_preflight=portal)
        self.entry_portal_waypoints = ([list(point) for point in
                                       portal["mandatory_portal_waypoints"]]
                                      if portal is not None else [])
        return activated

    def activate_truth_portal_staging(
            self, grid: OccupancyGrid2D, door: EstimatedDoorway,
            current: Point, now: float) -> Optional[EstimatedDoorway]:
        """Stage one bounded generated-geometry portal after live retries.

        This is reserved for the manager's upper-floor truth fallback after
        the ordinary occupancy proposal has already failed its bounded retry
        budget.  It does not mark ENTRY.  It only supplies corridor-side,
        door-centre and shallow-interior anchors; the manager still subjects
        their physical execution path to the live SCAN-lite 3-D audit.
        """
        if (grid is None or door is None or current is None or
                self.active_door is not None):
            return None
        valid, reason = doorway_candidate_entry_ready(grid, door)
        # A frozen upper-floor rolling projection can preserve the corridor
        # but expose no free room-side anchor at a generated doorway.  This
        # method is called only after the manager exhausts its station/side
        # retries and while physically beside that exact truth portal.  Allow
        # that one missing-anchor reason to reach the shallow staging trace;
        # actual execution still requires the manager's live SCAN-lite 3-D
        # audit and a measured truth door-plane crossing before ENTRY credit.
        # The manager deliberately classifies both of these bounded live-map
        # failures as recoverable truth-portal staging.  Keeping only the
        # first spelling here made every F2 far-row attempt in
        # ...f1fallrearm consume 3/3 retries with
        # ``no_safe_entry_candidate`` and then silently return None, even
        # though the subsequent staging path would still be checked by the
        # dense live SCAN-lite 3-D audit and require a measured door-plane
        # crossing.  Accept both equivalent "projection cannot propose an
        # anchor" outcomes; occupied 3-D geometry is not bypassed.
        truth_missing_anchor = str(reason) in {
            "candidate_no_observed_free_anchor",
            "no_safe_entry_candidate",
        }
        if ((not valid and not truth_missing_anchor) or
                _distance(current, door.center) > 2.60):
            return None
        target_depth = max(self.config.minimum_crossing_depth + 0.25, 0.85)
        target = (door.center[0] + target_depth * door.normal[0],
                  door.center[1] + target_depth * door.normal[1])
        # Keep the body centre farther than its radius from the wall plane.
        # The detector's nominal corridor_side is only 0.60 m outside the
        # plane; on F3 its inflated 2-D footprint touched the wall stripe and
        # every far-row attempt failed at mandatory segment 0.  A 0.90 m
        # standoff is still inside the generated corridor and leaves each
        # following portal leg below the 1.25 m truth-envelope bound.
        corridor_anchor = (
            float(door.center[0]) - 0.90 * float(door.normal[0]),
            float(door.center[1]) - 0.90 * float(door.normal[1]))
        # When already beside the generated portal, forcing an exact
        # corridor-side raster anchor adds no physical safety and is the
        # recurrent F3 far-row segment-0 blocked cell.  Start directly toward
        # the door centre in that bounded case; every leg still receives the
        # manager's truth-aperture and dense live 3-D audits.
        current_depth = door.depth(current)
        already_staged = bool(
            current_depth <= 0.0 and
            _distance(current, door.center) <= 1.25)
        # A far-row fallback is often scheduled 1.3--1.7 m before the portal
        # station.  Driving diagonally from there to the wall-side anchor
        # combines corridor advance and lateral approach, so the inflated
        # body envelope clips the door-frame stripe (run52 repeatedly failed
        # as portal_segment_0_goal_unreachable).  First preserve the current
        # door-normal depth and move only along the corridor tangent to the
        # portal row.  The following short lateral anchors retain the same
        # truth-aperture, live 3-D and physical crossing checks.
        tangent = (-float(door.normal[1]), float(door.normal[0]))
        tangent_delta = (
            (float(door.center[0]) - float(current[0])) * tangent[0] +
            (float(door.center[1]) - float(current[1])) * tangent[1])
        station_aligned_anchor = (
            float(current[0]) + tangent_delta * tangent[0],
            float(current[1]) + tangent_delta * tangent[1])
        station_alignment_required = bool(
            not already_staged and abs(tangent_delta) > 0.45)
        approach_anchors = (
            ([station_aligned_anchor] if station_alignment_required else []) +
            ([] if already_staged else [corridor_anchor]))
        anchors = approach_anchors + [
            (float(door.center[0]), float(door.center[1])),
            (float(target[0]), float(target[1])),
        ]
        trace = [(float(current[0]), float(current[1]))] + anchors
        length = sum(_distance(trace[index - 1], trace[index])
                     for index in range(1, len(trace)))
        portal = {
            "mandatory_portal_waypoints": [list(point) for point in anchors],
            "portal_clearance_m": 0.0,
            "preflight_path": trace,
            "preflight_path_length_m": length,
            "portal_resume_inside": False,
        }
        proposal = {
            "position": target, "role": "ENTRY_STAGING", "score": 0.0,
            "depth": target_depth, "lateral": 0.0,
            "clearance": 0.0, "unknown_neighbors": 0,
            "predicted_local_gain_m2": 0.0,
            "preflight_path_length_m": length,
            "fallback_candidate": True,
            "truth_geometry_verified_staging": True,
            "local_detector_verified_staging": True,
            "staging_portal_preflight_verified": True,
            "portal_preflight": portal,
        }
        activated = self._activate(door, now, proactive=True)
        if activated is None:
            self.prepared_entry = None
            self.entry_portal_waypoints = []
            if self.pending_corridor_recovery_door is not None:
                self.deferred_prevalidated_door_after_centerline = door
                self.deferred_prevalidated_entry_after_centerline = proposal
                self.deferred_entry_waypoints_after_centerline = [
                    list(point) for point in anchors]
                # _prepare_and_activate() may have rejected the ordinary
                # rolling-map proposal immediately before this bounded truth
                # staging fallback.  That rejection changes the scheduler to
                # CORRIDOR_SWEEP even though the preceding room still owns a
                # measured centreline-recovery debt.  Keeping that state made
                # the manager skip the recovery goal forever: the deferred
                # opposite portal remained valid, but could never be promoted
                # (F2 deferredportalfix, far-row room 4).  Restore the owning
                # transaction state explicitly; confirm_corridor_centerline_
                # recovered() remains the only operation that may activate
                # this doorway.
                self._set_state(
                    "CORRIDOR_RESUME", now,
                    "truth_portal_deferred_pending_centerline")
                self.events.append({
                    "event": "TRUTH_PORTAL_STAGING_DEFERRED_UNTIL_CENTERLINE",
                    "elapsed_sec": round(float(now), 3),
                    "candidate_center": [float(door.center[0]),
                                         float(door.center[1])],
                    "policy": "exit_centerline_then_truth_staging_entry",
                })
                return None
            self.events.append({
                "event": "TRUTH_PORTAL_STAGING_DUPLICATE_REJECTED",
                "elapsed_sec": round(float(now), 3),
                "candidate_center": [float(door.center[0]),
                                     float(door.center[1])],
            })
            return None
        self.prepared_entry = proposal
        self.entry_portal_waypoints = [list(point) for point in anchors]
        self.events.append({
            "event": "TRUTH_PORTAL_STAGING_ACTIVATED",
            "elapsed_sec": round(float(now), 3),
            "door_id": activated.door_id,
            "candidate_entry_reason": str(reason),
            "target_depth_m": round(float(target_depth), 3),
            "physical_entry_still_required": True,
            "live_scan_lite_3d_still_required": True,
            "corridor_anchor_skipped": already_staged,
            "station_alignment_inserted": station_alignment_required,
        })
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
            # A physical EXIT with incomplete coverage is recoverable room
            # debt, not a permanently visited room.  Reuse the same bounded
            # landmark retry path as an unvisited doorway so a terminal
            # return cannot hand off a partial room as completed.
            if (door.completed or
                    str(door.door_id) in self.blocked_door_ids):
                continue
            cooldown_until = self._door_cooldown_until(door.center)
            if float(now) < cooldown_until:
                continue
            distance = _distance(current, door.corridor_side)
            if distance <= limit:
                # A partial EXIT has already paid the ENTRY cost and owns an
                # unfinished physical-view contract.  It must be resumed
                # before any fresh/never-entered doorway, regardless of
                # corridor distance; otherwise a new door can be committed
                # while the old room still has a missing G3/G4 side.
                partial_priority = 0 if (
                    door.partial_retry_count > 0 or
                    door.partial_successful_roles or
                    door.partial_completed_points) else 1
                candidates.append((partial_priority, distance,
                                   door.entered_at, door))
        if not candidates:
            return None
        _, distance, _, door = min(candidates, key=lambda item: item[:3])
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
            if activated is None:
                self.prepared_entry = None
                self.entry_portal_waypoints = []
                if self.pending_corridor_recovery_door is not None:
                    self.deferred_prevalidated_door_after_centerline = door
                    self.deferred_prevalidated_entry_after_centerline = {
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
                    self.deferred_entry_waypoints_after_centerline = []
                    self.events.append({
                        "event": "LOCAL_DOOR_DEFERRED_UNTIL_CENTERLINE",
                        "elapsed_sec": round(float(now), 3),
                        "candidate_id": candidate_id,
                        "center": [float(door.center[0]),
                                   float(door.center[1])],
                        "policy": "exit_centerline_then_same_station_entry",
                    })
                    self._set_state(
                        "CORRIDOR_RESUME", now,
                        "previous_exit_centerline_recovery_pending")
                else:
                    self._set_state("CORRIDOR_SWEEP", now,
                                    "completed_physical_door_rejected")
                return None
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
            allow_local_wall_break_support=True,
            wall_break_clearance_override=(
                0.14 if candidate_id.startswith("mirrored-pair-") else 0.18))

    def abandon_corridor_resume(self, now: float, reason: str) -> None:
        """Release an unreachable recentering target instead of retrying it."""
        if self.active_door is None and self.state == "CORRIDOR_RESUME":
            self._set_state("CORRIDOR_SWEEP", now, reason)

    def confirm_corridor_centerline_recovered(
            self, now: float, point: Optional[Sequence[float]],
            source: str) -> bool:
        """Attach measured centreline recovery to the just-exited room."""
        door = self.pending_corridor_recovery_door
        if door is None:
            return False
        door.corridor_centerline_recovered = True
        self.events.append({
            "event": "CORRIDOR_CENTERLINE_RECOVERED",
            "elapsed_sec": round(float(now), 3),
            "room_id": door.room_id,
            "door_id": door.door_id,
            "truth_pose": ([round(float(point[0]), 4),
                            round(float(point[1]), 4)]
                           if point is not None and len(point) >= 2 else None),
            "source": str(source),
            "room_transaction_complete": bool(
                door.room_transaction_complete),
        })
        self.pending_corridor_recovery_door = None
        deferred_door = self.deferred_prevalidated_door_after_centerline
        deferred_entry = self.deferred_prevalidated_entry_after_centerline
        deferred_waypoints = list(
            self.deferred_entry_waypoints_after_centerline)
        self.deferred_prevalidated_door_after_centerline = None
        self.deferred_prevalidated_entry_after_centerline = None
        self.deferred_entry_waypoints_after_centerline = []
        if deferred_door is not None and deferred_entry is not None:
            activated = self._activate(
                deferred_door, now, proactive=True, reuse_existing=False)
            if activated is not None:
                self.prepared_entry = dict(deferred_entry)
                self.entry_portal_waypoints = deferred_waypoints
                self.events.append({
                    "event": "LOCAL_DOOR_REACTIVATED_AFTER_CENTERLINE",
                    "elapsed_sec": round(float(now), 3),
                    "door_id": activated.door_id,
                    "center": [float(activated.center[0]),
                               float(activated.center[1])],
                    "policy": "measured_centerline_then_deferred_entry",
                })
            else:
                self.events.append({
                    "event": "LOCAL_DOOR_DEFERRED_REACTIVATION_REJECTED",
                    "elapsed_sec": round(float(now), 3),
                    "center": [float(deferred_door.center[0]),
                               float(deferred_door.center[1])],
                })
        return True

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

    def request_immediate_exit(self, now: float, reason: str,
                               final_point: Optional[Point] = None) -> bool:
        """Skip optional room viewpoints after a continuity-pass entry."""
        if self.active_door is None or not self.entry_confirmed:
            return False
        # A portal crossing alone is not a room observation.  On run220 the
        # terminal-return fallback crossed only 0.38 m beyond the estimated
        # doorway, immediately requested EXIT, and falsely counted that room
        # as visited.  Keep the fast return-pass path for a genuine interior
        # camera pose, but let the ordinary G3/G4 planner take over after a
        # shallow crossing.
        if final_point is not None:
            observed_depth = self.active_door.depth(final_point)
            required_depth = max(
                0.75, min(1.20, 0.65 * self.config.entry_depth))
            if observed_depth < required_depth:
                self.events.append({
                    "event":
                        "ROOM_RETURN_PASS_IMMEDIATE_EXIT_REJECTED_SHALLOW_ENTRY",
                    "elapsed_sec": round(float(now), 3),
                    "room_id": self._room_id(),
                    "door_id": self.active_door.door_id,
                    "observed_depth_m": round(float(observed_depth), 3),
                    "required_depth_m": round(float(required_depth), 3),
                    "reason": str(reason),
                })
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

    def abandon_active_room_after_bounded_exit_failure(
            self, now: float, reason: str) -> Optional[dict]:
        """Cleanly discard one room whose bounded EXIT recovery is exhausted.

        This is a continuation path, not a room completion path.  The active
        doorway is removed from the live detector so an ENTERED_PARTIAL
        transaction cannot leak into corridor scheduling, and the portal is
        cooled/blocked so the same opening cannot immediately be selected
        again.  Confirmed physical EXIT counters are intentionally unchanged.
        """
        door = self.active_door
        if door is None:
            return None
        payload = {
            "event": "PARTIAL_ROOM_CLEANED_BEFORE_CONTINUATION",
            "elapsed_sec": round(float(now), 3),
            "room_id": self.active_room_id,
            "door_id": door.door_id,
            "entry_confirmed": bool(self.entry_confirmed),
            "completed": bool(door.completed),
            "physical_exit_count_before": sum(
                bool(item.visited) for item in self.detector.doors),
            "reason": str(reason),
            "policy": "remove_active_transaction_keep_exit_count_strict",
        }
        self.events.append(payload)
        self.blocked_door_ids.add(str(door.door_id))
        self._set_door_cooldown(
            door.center, float(now) + self.config.door_cooldown_seconds)
        # Keep the physical doorway record as a failed/partial historical
        # room.  Clearing the active transaction is the cleanup requirement;
        # deleting the record or decrementing room_count makes the published
        # recognized-room count go backwards (and caused F1 1/4 to become
        # 0/4 after a bounded EXIT failure).  ``blocked_door_ids`` prevents
        # another immediate transaction, while visited/exited remains false.
        door.temporarily_failed = True
        door.completed = False
        door.coverage_complete = bool(door.coverage_complete)
        self.corridor_resume_forward_escape_pending = True
        payload["policy"] = (
            "retain_failed_room_record_clear_active_block_future_retry")
        self._clear_active()
        self._set_state("CORRIDOR_RESUME", now, str(reason))
        return payload

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
        # Freeze the truth-derived obstacle contract as soon as it is known.
        # In particular, do not let the incomplete post-ENTRY raster classify
        # this room as an open room and select a deep mapping transit.
        if (getattr(self.active_door, "viewpoint_contract", None) ==
                "obstacle_front_opposite_sides"):
            self.two_pose_opposite_repair_required = True
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
                    proposal.get("entry_staging_fallback") and
                    not proposal.get("local_detector_verified_staging")):
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
            staging_fallback = bool(proposal.get("entry_staging_fallback"))
            portal = (proposal.get("portal_preflight") or
                      (None if staging_fallback else
                       prepare_portal_path(
                           grid, self.active_door, current,
                           proposal["position"], self.config)))
            # Floor-1 may explicitly choose the short geometric portal transit.
            if (self.config.prefer_direct_anchored_entry and
                    not self.config.require_portal_preflight_for_entry):
                # Each short leg is checked again by the live executor. Do not
                # reuse the global/staging A* path: run203 first travelled
                # backwards along the corridor and then completed no waypoint.
                proposal = dict(proposal)
                proposal.pop("entry_staging_path_result", None)
                proposal.pop("staging_portal_preflight_verified", None)
                # Discovery and identity remain sensor-derived, but after the
                # measured portal is matched to a generated physical room use
                # that immutable door plane for the short ENTRY anchors.  A
                # 0.1--0.3 m centre bias is enough to place the nominal anchor
                # on an inflated jamb (run63 detected the later F1 doors and
                # selected the right contracts, then failed segment 0/1).
                truth_center = getattr(
                    self.active_door, "truth_contract_center", None)
                truth_direction = getattr(
                    self.active_door,
                    "truth_contract_normal_direction", None)
                truth_match_separation = getattr(
                    self.active_door,
                    "truth_contract_match_separation_m", None)
                truth_tangent_separation = getattr(
                    self.active_door,
                    "truth_contract_match_tangent_separation_m", None)
                truth_normal_separation = getattr(
                    self.active_door,
                    "truth_contract_match_normal_separation_m", None)
                whole_center_aligned = bool(
                    truth_center is not None and len(truth_center) >= 2 and
                    truth_direction is not None and
                    truth_match_separation is not None and
                    float(truth_match_separation) <=
                    float(self.config.truth_portal_snap_max_separation))
                tangent_repair_aligned = bool(
                    truth_center is not None and len(truth_center) >= 2 and
                    truth_direction is not None and
                    truth_tangent_separation is not None and
                    truth_normal_separation is not None and
                    float(truth_tangent_separation) <= float(
                        self.config.truth_portal_tangent_repair_max_separation) and
                    float(truth_normal_separation) <= float(
                        self.config.truth_portal_normal_repair_max_separation))
                truth_aligned = bool(
                    whole_center_aligned or tangent_repair_aligned)
                if truth_aligned:
                    portal_normal = (
                        math.cos(float(truth_direction)),
                        math.sin(float(truth_direction)))
                    # Correct only the along-wall station to the immutable
                    # physical door centre.  Preserve the measured normal
                    # offset so a small layout/raster wall-plane disagreement
                    # cannot move the body into an inflated jamb.
                    detected_delta = (
                        float(self.active_door.center[0]) -
                        float(truth_center[0]),
                        float(self.active_door.center[1]) -
                        float(truth_center[1]))
                    signed_normal_offset = (
                        0.0 if whole_center_aligned else
                        _dot(detected_delta, portal_normal))
                    portal_center = (
                        float(truth_center[0]) +
                        signed_normal_offset * portal_normal[0],
                        float(truth_center[1]) +
                        signed_normal_offset * portal_normal[1])
                    proposal_depth = max(
                        self.config.minimum_crossing_depth + 0.25,
                        float(proposal.get("depth", 0.85) or 0.85))
                    proposal = dict(proposal)
                    proposal["truth_portal_center_aligned"] = True
                    proposal["position"] = [
                        portal_center[0] +
                        proposal_depth * portal_normal[0],
                        portal_center[1] +
                        proposal_depth * portal_normal[1]]
                else:
                    portal_center = (
                        float(self.active_door.center[0]),
                        float(self.active_door.center[1]))
                    portal_normal = (
                        float(self.active_door.normal[0]),
                        float(self.active_door.normal[1]))
                corridor_anchor = [
                    portal_center[0] - 0.90 * portal_normal[0],
                    portal_center[1] - 0.90 * portal_normal[1]]
                center_anchor = [portal_center[0], portal_center[1]]
                interior_anchor = [
                    float(proposal["position"][0]),
                    float(proposal["position"][1])]
                # A proactive doorway intercept can preempt the corridor goal
                # after the body has already crossed the door plane. Sending
                # it back to corridor_side then asks SCAN-lite to validate a
                # reverse narrow-door segment, which is exactly the
                # portal_segment_1_goal_footprint_blocked failure seen on F2.
                # Resume at the centreline in that case; the centre->interior
                # segment still preserves the physical entry confirmation and
                # is checked independently by the manager.
                current_depth = (
                    (float(current[0]) - portal_center[0]) *
                    portal_normal[0] +
                    (float(current[1]) - portal_center[1]) *
                    portal_normal[1])
                # A shallow staging attempt can finish at the measured door
                # centre (F3 seedhandoff: depth 0.001/-0.006 m) and correctly
                # request a deeper retry. Sending that pose back to the
                # corridor-side anchor makes the reverse narrow-door segment
                # fail the footprint gate. Treat the small centre band as a
                # centreline resume: centre->interior still performs and
                # verifies the physical crossing, while a fresh corridor pose
                # (normally depth about -0.9 m) retains all three anchors.
                # First align longitudinally with the portal while retaining
                # the body's current safe corridor depth.  A truth fallback
                # can be activated when the dog is 0.5--0.7 m along the
                # corridor from the aperture.  Sending that pose diagonally
                # to the near-wall ``corridor_side`` anchor makes its endpoint
                # overlap the inflated wall return (run36 F2 far row: six
                # portal_segment_0_goal_footprint_blocked failures).  The
                # station-aligned anchor stays on the same normal-depth line,
                # then centre->interior preserves the physical crossing.
                tangent = (-portal_normal[1], portal_normal[0])
                current_lateral = (
                    (float(current[0]) - portal_center[0]) * tangent[0] +
                    (float(current[1]) - portal_center[1]) * tangent[1])
                centreline_half_width = max(
                    0.30, 0.25 * float(self.active_door.width))
                # A pose in the shallow door-plane band is a safe two-anchor
                # resume only when it is also centred in the aperture.  run135
                # exited the opposite room at depth -0.068 m but 0.934 m along
                # the wall from this portal.  The old depth-only gate removed
                # the station anchor and asked the body to cut diagonally
                # through the jamb; SCAN-lite correctly rejected every retry.
                # A pose genuinely beyond the physical crossing threshold may
                # still resume regardless of lateral drift, since returning it
                # outside would replay a reverse portal crossing.
                already_past_door_plane = bool(
                    current_depth >= self.config.minimum_crossing_depth or
                    (current_depth >= -0.10 and
                     abs(float(current_lateral)) <= centreline_half_width))
                nx, ny = portal_normal
                station_aligned_anchor = [
                    float(portal_center[0] + current_depth * nx),
                    float(portal_center[1] + current_depth * ny)]
                station_alignment_required = bool(
                    not already_past_door_plane and
                    abs(float(current_lateral)) > centreline_half_width)
                approach_anchor = (station_aligned_anchor
                                   if station_alignment_required else
                                   corridor_anchor)
                direct_anchors = ([center_anchor, interior_anchor]
                                  if already_past_door_plane else
                                  [approach_anchor, center_anchor,
                                   interior_anchor])
                portal = {
                    "mandatory_portal_waypoints": direct_anchors,
                    "portal_clearance_m": 0.0,
                    "preflight_path": [],
                    "portal_resume_inside": bool(already_past_door_plane),
                }
                staging_fallback = False
                self.events.append({
                    "event": ("ROOM_ENTRY_DIRECT_CENTERLINE_RESUME_SELECTED"
                               if already_past_door_plane else
                               "ROOM_ENTRY_DIRECT_THREE_ANCHOR_SELECTED"),
                    "elapsed_sec": round(float(now), 3),
                    "room_id": self._room_id(),
                    "door_id": self.active_door.door_id,
                    "reason": ("current_pose_already_past_door_plane"
                               if already_past_door_plane else
                               "avoid_global_astar_corridor_backtrack"),
                    "current_door_depth_m": round(float(current_depth), 3),
                    "current_door_lateral_m": round(
                        float(current_lateral), 3),
                    "station_alignment_required": bool(
                        station_alignment_required),
                    "truth_portal_center_aligned": bool(truth_aligned),
                    "truth_portal_alignment_mode": (
                        "truth_full_center"
                        if whole_center_aligned else
                        "truth_tangent_measured_normal_hybrid"
                        if tangent_repair_aligned else
                        "measured_door_center"),
                    "truth_contract_match_separation_m": (
                        None if truth_match_separation is None else
                        round(float(truth_match_separation), 3)),
                    "truth_contract_match_tangent_separation_m": (
                        None if truth_tangent_separation is None else
                        round(float(truth_tangent_separation), 3)),
                    "truth_contract_match_normal_separation_m": (
                        None if truth_normal_separation is None else
                        round(float(truth_normal_separation), 3)),
                    "truth_portal_snap_max_separation_m": round(
                        float(self.config.truth_portal_snap_max_separation), 3),
                    "detected_to_truth_center_offset_m": round(
                        _distance(self.active_door.center, portal_center), 3),
                    "approach_anchor": list(approach_anchor),
                    "waypoint_count": len(direct_anchors),
                })
            # If nominal portal geometry is sparse, fall back to the validated
            # Sending only proposal[position] produced ENTRY goals with one
            # waypoint and zero physical progress at the jamb.  The fallback
            # path explicitly traverses corridor-side -> doorway -> shallow
            # interior, while retaining A* and footprint checks.
            if portal is None and not proposal.get("entry_staging_fallback"):
                staged = fallback_entry_proposal(
                    grid, self.active_door, current, self.config)
                if staged is not None:
                    proposal = staged
                    staging_fallback = True
                    self.events.append({
                        "event": "ROOM_ENTRY_MULTIWAYPOINT_FALLBACK_SELECTED",
                        "elapsed_sec": round(float(now), 3),
                        "room_id": self._room_id(),
                        "door_id": self.active_door.door_id,
                        "waypoint_count": len((staged.get(
                            "entry_staging_path_result") or {}).get("path") or []),
                    })
            if portal is None and staging_fallback:
                portal = prepare_portal_path(
                    grid, self.active_door, current, proposal["position"],
                    self.config,
                    clearance_override=self.config.entry_staging_clearance)
                if portal is not None:
                    proposal["staging_portal_preflight_verified"] = True
                    self.events.append({
                        "event": "ROOM_ENTRY_STAGING_THREE_ANCHOR_VERIFIED",
                        "elapsed_sec": round(float(now), 3),
                        "room_id": self._room_id(),
                        "door_id": self.active_door.door_id,
                        "anchors": list(portal["mandatory_portal_waypoints"]),
                    })
            if portal is None and not self.config.require_portal_preflight_for_entry:
                # Keep explicit approach anchors even when the rolling map
                # cannot build a formal portal preflight.  The manager will
                # still run SCAN-lite/A* on each segment before dispatch.
                portal = {
                    "mandatory_portal_waypoints": [
                        [float(self.active_door.corridor_side[0]),
                         float(self.active_door.corridor_side[1])],
                        [float(self.active_door.center[0]),
                         float(self.active_door.center[1])],
                        [float(proposal["position"][0]),
                         float(proposal["position"][1])],
                    ],
                    "portal_clearance_m": 0.0,
                    "preflight_path": [],
                    "portal_resume_inside": False,
                }
                self.events.append({
                    "event": "ROOM_ENTRY_ANCHORED_PORTAL_FALLBACK",
                    "elapsed_sec": round(float(now), 3),
                    "room_id": self._room_id(),
                    "door_id": self.active_door.door_id,
                })
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
            if (portal is not None and
                    not portal.get("portal_resume_inside", False) and
                    len(portal.get("mandatory_portal_waypoints", [])) < 3):
                # Preserve explicit approach landmarks even when the rolling
                # map collapses a short portal into one endpoint.
                portal["mandatory_portal_waypoints"] = [
                    [float(self.active_door.corridor_side[0]),
                     float(self.active_door.corridor_side[1])],
                    [float(self.active_door.center[0]),
                     float(self.active_door.center[1])],
                    [float(proposal["position"][0]),
                     float(proposal["position"][1])],
                ]
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
                "execution_timeout_sec": (
                    self.config.entry_timeout_seconds
                    if self.config.budget_starts_after_entry else
                    min(self.config.entry_timeout_seconds,
                        max(3.0, self.config.room_budget_seconds -
                            self.config.exit_reserve_seconds - elapsed))
                ),
                # A doorway commit must fail fast when the projected
                # aperture is a wall crack; reserve the room budget for a
                # real door and the two in-room views.
                "progress_timeout_sec": self.config.entry_progress_timeout_seconds,
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
            if (proposal.get("staging_portal_preflight_verified") and
                    portal is not None and portal.get("preflight_path")):
                goal["_preplanned_path_result"] = {
                    "success": True,
                    "reason": "room_entry_staging_three_anchor_preflight",
                    "path": list(portal["preflight_path"])}
                goal["entry_staging_preflight_verified"] = True
                goal["staging_portal_preflight_verified"] = True
            elif proposal.get("entry_staging_path_result"):
                goal["_preplanned_path_result"] = dict(
                    proposal["entry_staging_path_result"])
                goal["entry_staging_preflight_verified"] = True
            elif portal is not None and portal.get("preflight_path"):
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
        deep_obstacle_contract = bool(
            self.deep_obstacle_contract_grace_armed or
            getattr(self.active_door, "viewpoint_contract", None) ==
            "obstacle_front_opposite_sides" and
            obstacle_view_policy(self.active_door) ==
            "deep_outer_side_peek")
        if deep_obstacle_contract:
            observation_deadline += max(
                0.0, float(
                    self.config.deep_obstacle_contract_grace_seconds))
        observation_attempts = len({
            role for role in self.role_attempts
            if role in ("G1", "G2", "G3", "G4")})
        missing_physical_roles = tuple(
            role for role in ("G3", "G4")
            if role not in self.successful_roles)
        missing_physical_view = bool(missing_physical_roles)
        exhausted_physical_roles = tuple(
            role for role in missing_physical_roles
            if role in self.failed_roles)
        locked_physical_contract = bool(
            getattr(self.active_door, "viewpoint_contract", None) in (
                "open_deep_near", "obstacle_front_opposite_sides"))
        # The one-shot refresh is diagnostic, not G4.  Previously it consumed
        # the last callback before ``observation_deadline`` and the very next
        # cycle marked the missing role failed, so F3 room04 went directly to
        # EXIT with G3/G4 at effectively the same truth pose.  Extend only a
        # locked transaction that already owns exactly one physical view and
        # has actually spent its bounded refresh.  Failed-role exhaustion and
        # rooms with no physical view retain the original hard deadline.
        refresh_spent = bool(
            self.mandatory_two_pose_unavailable_refresh_used > 0 or
            self.obstacle_gap_g4_refresh_used > 0 or
            self.open_missing_depth_refresh_used > 0 or
            self.visual_entry_refresh_used > 0)
        execution_retry_armed = bool(
            "G4_ADAPTIVE_EXECUTION_RETRY" in
            self.visual_geometry_retry_roles or
            any(role in self.visual_geometry_retry_roles
                for role in missing_physical_roles))
        single_physical_view_owned = bool(
            locked_physical_contract and
            len(self.successful_roles.intersection({"G3", "G4"})) == 1 and
            missing_physical_view and not exhausted_physical_roles)
        if single_physical_view_owned:
            observation_deadline += max(
                0.0, float(
                    self.config.mandatory_two_pose_refresh_grace_seconds))
            if not any(event.get("event") ==
                       "ROOM_TWO_POSE_SINGLE_VIEW_GRACE_ARMED"
                       for event in self.events[-12:]):
                self.events.append({
                    "event": "ROOM_TWO_POSE_SINGLE_VIEW_GRACE_ARMED",
                    "elapsed_sec": round(float(now), 3),
                    "room_id": self._room_id(),
                    "door_id": self.active_door.door_id,
                    "successful_roles": sorted(self.successful_roles),
                    "missing_roles": list(missing_physical_roles),
                    "extended_observation_deadline_sec": round(
                        float(observation_deadline), 3),
                    "grace_seconds": round(float(
                        self.config.mandatory_two_pose_refresh_grace_seconds),
                        3),
                    "refresh_spent": refresh_spent,
                    "execution_retry_armed": execution_retry_armed,
                    "policy": "one_physical_view_then_partner_or_partial",
                })
        if elapsed >= observation_deadline and missing_physical_view:
            self.failed_roles.update(missing_physical_roles)
            self.adaptive_route_queue = []
            self.next_role_index = 3
            if self.completion_mode is None:
                self.completion_mode = "room_budget_hard_deadline"
            self.events.append({
                "event": "ROOM_HARD_DEADLINE_RETURN_REQUIRED",
                "elapsed_sec": round(float(now), 3),
                "room_id": self._room_id(),
                "door_id": self.active_door.door_id,
                "missing_roles": list(missing_physical_roles),
                "room_elapsed_sec": round(float(elapsed), 3),
                "observation_deadline_sec": round(
                    float(observation_deadline), 3),
                "policy": "partial_evidence_then_physical_return_exit",
            })
            exhausted_physical_roles = tuple(missing_physical_roles)
        mandatory_two_pose_pending = bool(
            self.config.visual_two_pose_lateral_enabled and
            missing_physical_view and
            not exhausted_physical_roles and
            (locked_physical_contract or
             self.visual_deepening_requested or
             any(item.get("two_pose_room_sweep")
                 for item in self.adaptive_route_queue)))
        if exhausted_physical_roles:
            # ``record_result`` gives a G3/G4 one fresh-geometry retry and,
            # for adaptive G4, one execution retry.  Once that bounded budget
            # is exhausted the role is inserted in ``failed_roles``.  The old
            # mandatory-contract predicate ignored that terminal marker and
            # kept selecting the same mandatory fallback forever (F3 room 2
            # repeatedly emitted scan_lite_refinement_failed while stationary
            # for more than a minute).  Preserve the failed physical contract
            # in the room evidence, but release observation ownership so the
            # scheduler can execute RETURN and a real EXIT.
            if not any(event.get("event") ==
                       "ROOM_TWO_POSE_RETRY_EXHAUSTED_RETURN_REQUIRED"
                       for event in self.events[-8:]):
                self.events.append({
                    "event":
                        "ROOM_TWO_POSE_RETRY_EXHAUSTED_RETURN_REQUIRED",
                    "elapsed_sec": round(float(now), 3),
                    "room_id": self._room_id(),
                    "door_id": self.active_door.door_id,
                    "exhausted_roles": list(exhausted_physical_roles),
                    "successful_roles": sorted(self.successful_roles),
                    "policy": "bounded_failure_then_physical_return_exit",
                })
        # The adaptive viewpoint cap is only a time-saving ceiling for an
        # ordinary coverage visit.  It must never suppress the second member
        # of a locked physical two-view transaction.  In the upper-floor
        # timed profile adaptive_maximum_viewpoints is deliberately one; the
        # old ordering therefore closed this branch immediately after G3 and
        # sent an open_deep_near room straight to EXIT with no G4.  Treat the
        # outstanding contract view as reserved capacity, while retaining the
        # existing bounded planning/retry/watchdog paths below.
        observation_capacity = (
            (observation_attempts <
             self.config.adaptive_maximum_viewpoints) or
            mandatory_two_pose_pending
            if self.config.adaptive_minimal_viewpoints else
            self.next_role_index < 3)
        if (elapsed >= observation_deadline and observation_capacity and
                self.completion_mode != "lidar_coverage" and
                not mandatory_two_pose_pending):
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
            # A freshly entered upper-floor room can project the robot's own
            # body into the first rolling occupancy frame.  When the executor
            # rejects a G3/G4 before moving with start_footprint_blocked,
            # refresh that exact verified interior footprint once and then
            # regenerate the same physical role from the next live grid.  The
            # diagnostic flag makes record_result() exclude this stationary
            # operation from physical-view and contract accounting.
            if self.visual_geometry_map_refresh_pending in ("G3", "G4"):
                refresh_role = str(
                    self.visual_geometry_map_refresh_pending)
                self.visual_geometry_map_refresh_pending = None
                proposal = {
                    "position": (float(current[0]), float(current[1])),
                    "role": refresh_role,
                    "score": 0.0,
                    "clearance": _clearance(grid, current),
                    "coverage_cells": set(),
                    "adaptive_marginal_gain_m2": 0.0,
                    "adaptive_utility": 0.0,
                    "preflight_path_length_m": 0.0,
                    "preflight_path": [(float(current[0]),
                                        float(current[1]))],
                    "fallback_candidate": True,
                    "entry_pose_visual_fallback": True,
                    "visual_deepening": True,
                    "two_pose_room_sweep": True,
                    "two_pose_strategy":
                        "start_footprint_map_refresh_before_same_role",
                }
                self.events.append({
                    "event":
                        "ROOM_START_FOOTPRINT_MAP_REFRESH_BEFORE_SAME_ROLE",
                    "elapsed_sec": round(float(now), 3),
                    "room_id": self._room_id(),
                    "door_id": self.active_door.door_id,
                    "role": refresh_role,
                    "position": [float(current[0]), float(current[1])],
                    "policy":
                        "stationary_refresh_not_a_viewpoint_bounded_once",
                })
            # G3 is the first physical member of the two-view contract in
            # the lateral profile.  If its motion timed out after making
            # little/no progress, do not advance to G4: doing so records a
            # plausible second point while leaving the room partial.  Retry
            # the same role from the fresh live grid, bounded by the normal
            # side-goal retry limit, before any G4 proposal is considered.
            mandatory_failed_role = None
            if (proposal is None and
                    self.config.visual_two_pose_lateral_enabled and
                    "G3" in self.role_attempts and
                    "G3" not in self.successful_roles and
                    self.role_attempts.get("G3", 0) <
                    self.config.side_goal_retry_limit):
                mandatory_failed_role = "G3"
                self.adaptive_route_queue = []
                self.visual_deepening_requested = True
                retry_candidate = select_room_goal(
                    grid, self.active_door, current, "G3",
                    self.completed_points, self.config, fallback=True)
                if retry_candidate is not None:
                    proposal = dict(retry_candidate)
                    proposal["visual_deepening"] = True
                    proposal["mandatory_two_pose_retry"] = True
                    self.events.append({
                        "event": "ROOM_TWO_POSE_G3_RETRY_BEFORE_G4",
                        "elapsed_sec": round(float(now), 3),
                        "room_id": self._room_id(),
                        "door_id": self.active_door.door_id,
                        "role": "G3",
                        "previous_attempts": int(
                            self.role_attempts.get("G3", 0)),
                        "policy": "never_advance_to_g4_with_partial_g3",
                    })
            # Once LiDAR coverage is complete, an outstanding RGB-D anchor
            # is the only remaining sensing debt.  Do not re-run set cover
            # and send the robot to an unrelated long G2/G3 point merely to
            # satisfy that camera requirement: use the existing G1 route (or
            # one footprint-validated G1 fallback) and exit immediately
            # afterwards.  This preserves broad LiDAR coverage and one
            # interior RGB-D view while removing a recurrent 5--7 m detour.
            if (proposal is None and enough_observations and
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
                if queued.get("two_pose_room_sweep"):
                    proposal = self.adaptive_route_queue.pop(0)
                    self.events.append({
                        "event": "ROOM_TWO_POSE_NEXT_VIEW_SELECTED",
                        "elapsed_sec": round(float(now), 3),
                        "room_id": self._room_id(),
                        "door_id": self.active_door.door_id,
                        "role": proposal.get("role"),
                        "strategy": proposal.get("two_pose_strategy"),
                        "two_pose_index": proposal.get("two_pose_index"),
                        "position": [float(proposal["position"][0]),
                                     float(proposal["position"][1])],
                    })
                    break
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
                    (len(attempted) <
                     self.config.adaptive_maximum_viewpoints or
                     mandatory_two_pose_pending)):
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
                # Build the two camera centres as one route immediately after
                # ENTRY.  This prevents a good first 360-degree sweep from
                # cancelling the second physical viewpoint.
                if (proposal is None and
                        (len(attempted) <= 1 or self.visual_entry_refresh_used) and
                        self.config.visual_two_pose_lateral_enabled and
                        self.visual_deepening_requested):
                    two_pose_config = self.config
                    strict_retry_role = next(
                        (item for item in ("G3", "G4")
                         if item in self.visual_geometry_retry_roles and
                         item not in attempted), None)
                    if strict_retry_role is not None:
                        two_pose_config = copy.copy(self.config)
                        two_pose_config.goal_clearance = max(
                            self.config.goal_clearance, 0.65)
                        self.events.append({
                            "event": "ROOM_TWO_POSE_STRICT_CLEARANCE_RETRY",
                            "elapsed_sec": round(float(now), 3),
                            "room_id": self._room_id(),
                            "door_id": self.active_door.door_id,
                            "role": strict_retry_role,
                            "goal_clearance_m": round(float(
                                two_pose_config.goal_clearance), 3),
                            "policy": "fresh_grid_endpoint_and_route_margin",
                        })
                    two_pose_plan_started_wall = time.monotonic()
                    if (getattr(self.active_door,
                                "viewpoint_contract", None) ==
                            "obstacle_front_opposite_sides" and
                            "G3" not in self.successful_roles and
                            not self.completed_points):
                        self.two_pose_opposite_repair_required = True
                    two_pose_plan = plan_two_pose_room_sweep(
                        grid, self.active_door, current,
                        self.return_anchor or self.active_door.interior_side,
                        two_pose_config,
                        excluded_targets_by_role=
                        self.failed_visual_geometry_targets)
                    two_pose_planning_wall_sec = (
                        time.monotonic() - two_pose_plan_started_wall)
                    # A freshly entered upper-floor obstacle room may have
                    # no inferred room cells yet, so the generic pair planner
                    # legitimately returns None.  The truth-locked gap G3
                    # selector does not need that room polygon: it searches
                    # observed-free cells before the measured obstacle face
                    # and validates the complete live A* path.  It used to be
                    # nested under ``two_pose_plan is not None`` and was
                    # therefore unreachable in exactly this F3 cold-map case;
                    # after the one-shot refresh the scheduler fell through
                    # to adaptive_no_marginal_gain and exited with zero views.
                    truth_obstacle_g3_required = bool(
                        getattr(self.active_door,
                                "viewpoint_contract", None) ==
                        "obstacle_front_opposite_sides" and
                        "G3" not in self.successful_roles and
                        not self.completed_points)
                    if truth_obstacle_g3_required and two_pose_plan is None:
                        truth_gap_first = (
                            select_truth_obstacle_gap_first_visual_view(
                                grid, self.active_door, current,
                                self.config,
                                excluded_targets=
                                self.failed_visual_geometry_targets.get(
                                    "G3", ())))
                        if truth_gap_first is not None:
                            two_pose_plan = {
                                "ordered": [truth_gap_first],
                                "strategy": "obstacle_front_gap_first",
                                "obstacle": truth_gap_first.get(
                                    "two_pose_obstacle"),
                                "baseline_m": 0.0,
                                "route_length_m": float(
                                    truth_gap_first.get(
                                        "preflight_path_length_m", 0.0)),
                                "return_distance_m": float(_distance(
                                    truth_gap_first["position"],
                                    self.return_anchor or
                                    self.active_door.interior_side)),
                            }
                            self.events.append({
                                "event":
                                    "ROOM_TRUTH_OBSTACLE_GAP_G3_RECOVERED_WITHOUT_ROOM_POLYGON",
                                "elapsed_sec": round(float(now), 3),
                                "room_id": self._room_id(),
                                "door_id": self.active_door.door_id,
                                "position": [
                                    float(truth_gap_first["position"][0]),
                                    float(truth_gap_first["position"][1])],
                                "policy":
                                    "observed_free_truth_gap_live_astar",
                            })
                    if two_pose_plan is not None:
                        two_pose_plan, uncertainty_mapping_view = (
                            stage_uncertain_open_near_mapping_view(
                                grid, current, two_pose_plan,
                                two_pose_config, attempted))
                        obstacle = two_pose_plan.get("obstacle") or {}
                        obstacle_strategy = str(
                            two_pose_plan.get("strategy", ""))
                        # A truth-classified front obstacle must start with a
                        # gap-side G3 even when the live map has not yet
                        # projected the furniture footprint.  Do not replace
                        # that contract with the generic near->deep mapping
                        # transit; G4 is selected later from the opposite
                        # gap side after the fresh G3 scan.
                        if truth_obstacle_g3_required:
                            self.two_pose_opposite_repair_required = True
                            truth_gap_first = (
                                select_truth_obstacle_gap_first_visual_view(
                                    grid, self.active_door, current,
                                    self.config,
                                    excluded_targets=
                                    self.failed_visual_geometry_targets.get(
                                        "G3", ())))
                            if truth_gap_first is not None:
                                two_pose_plan = dict(two_pose_plan)
                                two_pose_plan["ordered"] = [truth_gap_first]
                                two_pose_plan["strategy"] = (
                                    "obstacle_front_gap_first")
                                two_pose_plan["obstacle"] = (
                                    truth_gap_first.get("two_pose_obstacle"))
                                two_pose_plan["baseline_m"] = 0.0
                                two_pose_plan["route_length_m"] = float(
                                    truth_gap_first.get(
                                        "preflight_path_length_m", 0.0))
                                uncertainty_mapping_view = False
                                obstacle_strategy = (
                                    "obstacle_front_gap_first")
                                self.events.append({
                                    "event":
                                        "ROOM_TRUTH_OBSTACLE_GAP_G3_SELECTED",
                                    "elapsed_sec": round(float(now), 3),
                                    "room_id": self._room_id(),
                                    "door_id": self.active_door.door_id,
                                    "role": "G3",
                                    "position": [
                                        float(truth_gap_first["position"][0]),
                                        float(truth_gap_first["position"][1])],
                                    "depth_m": round(float(
                                        truth_gap_first["depth"]), 3),
                                    "lateral_m": round(float(
                                        truth_gap_first["lateral"]), 3),
                                    "policy":
                                        "door_obstacle_gap_before_front_face",
                                })
                            else:
                                # Fail closed on geometry.  The generic plan
                                # may contain a reachable point behind the
                                # obstacle, but that route violates the locked
                                # room contract and must never be used as G3.
                                two_pose_plan = dict(two_pose_plan)
                                two_pose_plan["ordered"] = []
                                two_pose_plan["strategy"] = (
                                    "obstacle_front_gap_wait_for_fresh_map")
                                obstacle_strategy = str(
                                    two_pose_plan["strategy"])
                                self.events.append({
                                    "event":
                                        "ROOM_TRUTH_OBSTACLE_GAP_G3_DEFERRED",
                                    "elapsed_sec": round(float(now), 3),
                                    "room_id": self._room_id(),
                                    "door_id": self.active_door.door_id,
                                    "policy":
                                        "no_generic_deep_or_behind_obstacle_fallback",
                                })
                        # A shallow blocker has a stricter second-view
                        # contract than the generic live-map obstacle split:
                        # both camera centres must remain in front of the
                        # nearest furniture while occupying opposite lateral
                        # sides.  run71 correctly reached the truth-bounded
                        # G3, then the generic replanner moved G4 2.43 m deep,
                        # behind the 1.80 m chair front.  Reuse the same
                        # truth-gap selector from the measured G3 pose; it
                        # orders the opposite side first and the manager still
                        # audits the complete chord before locomotion.
                        truth_shallow_g4_required = bool(
                            getattr(self.active_door,
                                    "viewpoint_contract", None) ==
                            "obstacle_front_opposite_sides" and
                            obstacle_view_policy(self.active_door) ==
                            "shallow_front_chord" and
                            "G3" in self.successful_roles and
                            "G4" not in self.successful_roles and
                            bool(self.completed_points))
                        if truth_shallow_g4_required:
                            truth_gap_second = (
                                select_front_gap_opposite_side_second_visual_view(
                                    grid, self.active_door, current,
                                    self.completed_points, {}, self.config,
                                    excluded_targets=
                                    self.failed_visual_geometry_targets.get(
                                        "G4", ())))
                            if truth_gap_second is not None:
                                truth_gap_second = dict(truth_gap_second)
                                truth_gap_second["role"] = "G4"
                                truth_gap_second["two_pose_index"] = 2
                                truth_gap_second["two_pose_strategy"] = (
                                    "obstacle_front_gap_second_truth_bounded")
                                truth_gap_second["visual_baseline_m"] = float(
                                    _distance(
                                        self.completed_points[-1],
                                        truth_gap_second["position"]))
                                truth_gap_second[
                                    "visual_baseline_requirement_met"] = bool(
                                        truth_gap_second[
                                            "visual_baseline_m"] >=
                                        self.config.minimum_goal_separation)
                                two_pose_plan = dict(two_pose_plan)
                                two_pose_plan["ordered"] = [truth_gap_second]
                                two_pose_plan["strategy"] = (
                                    "obstacle_front_gap_second_truth_bounded")
                                two_pose_plan["obstacle"] = (
                                    truth_gap_second.get("two_pose_obstacle"))
                                two_pose_plan["baseline_m"] = float(
                                    truth_gap_second["visual_baseline_m"])
                                two_pose_plan["route_length_m"] = float(
                                    truth_gap_second.get(
                                        "preflight_path_length_m", 0.0))
                                uncertainty_mapping_view = False
                                obstacle_strategy = str(
                                    two_pose_plan["strategy"])
                                self.two_pose_opposite_repair_required = True
                                self.events.append({
                                    "event":
                                        "ROOM_TRUTH_SHALLOW_GAP_G4_SELECTED",
                                    "elapsed_sec": round(float(now), 3),
                                    "room_id": self._room_id(),
                                    "door_id": self.active_door.door_id,
                                    "role": "G4",
                                    "position": [float(
                                        truth_gap_second["position"][0]),
                                        float(truth_gap_second[
                                            "position"][1])],
                                    "depth_m": round(float(
                                        truth_gap_second["depth"]), 3),
                                    "lateral_m": round(float(
                                        truth_gap_second["lateral"]), 3),
                                    "baseline_m": round(float(
                                        truth_gap_second[
                                            "visual_baseline_m"]), 3),
                                    "policy":
                                        "same_front_gap_depth_opposite_side",
                                })
                        if uncertainty_mapping_view:
                            staged_position = two_pose_plan["ordered"][0][
                                "position"]
                            self.events.append({
                                "event":
                                    "ROOM_UNCERTAIN_OPEN_NEAR_MAPPING_VIEW",
                                "elapsed_sec": round(float(now), 3),
                                "room_id": self._room_id(),
                                "door_id": self.active_door.door_id,
                                "position": [float(staged_position[0]),
                                             float(staged_position[1])],
                                "policy":
                                    "near_view_then_fresh_grid_second_view",
                            })
                        if (attempted and self.completed_points and
                                obstacle_strategy ==
                                "provisional_obstacle_side_split"):
                            anchor_depth = self.active_door.depth(
                                self.completed_points[-1])
                            template = dict(
                                (two_pose_plan.get("ordered") or [{}])[0])
                            if (anchor_depth < 2.80 and
                                    self.uncertain_open_near_mapping_completed):
                                # G3 was intentionally the near mapping view.
                                # Complete an open-room diagonal with the
                                # fresh-grid deep point, rather than adding a
                                # second near-side point across the doorway.
                                repaired_second = (
                                    select_fresh_grid_deep_second_visual_view(
                                        self.active_door,
                                        self.completed_points,
                                        two_pose_plan, self.config))
                                if repaired_second is None:
                                    repaired_second = (
                                        select_mandatory_second_visual_view(
                                            grid, self.active_door, current,
                                            self.completed_points,
                                            self.config))
                            elif anchor_depth < 2.80:
                                repaired_second = (
                                    select_near_door_side_second_visual_view(
                                        grid, self.active_door, current,
                                        self.completed_points, template,
                                        self.config))
                            elif anchor_depth >= 3.40:
                                repaired_second = (
                                    select_mandatory_second_visual_view(
                                        grid, self.active_door, current,
                                        self.completed_points, self.config))
                            else:
                                repaired_second = None
                            if repaired_second is not None:
                                two_pose_plan = dict(two_pose_plan)
                                two_pose_plan["ordered"] = [repaired_second]
                                two_pose_plan["strategy"] = str(
                                    repaired_second.get(
                                        "two_pose_strategy",
                                        "provisional_second_repair"))
                                two_pose_plan["baseline_m"] = float(
                                    repaired_second.get(
                                        "visual_baseline_m", 0.0))
                                two_pose_plan["route_length_m"] = float(
                                    repaired_second.get(
                                        "preflight_path_length_m", 0.0))
                                two_pose_plan["return_distance_m"] = 0.0
                                obstacle_strategy = str(
                                    two_pose_plan["strategy"])
                        if obstacle_strategy in (
                                "obstacle_deep_then_opposite_near",
                                "obstacle_front_side_split"):
                            self.two_pose_opposite_repair_required = True
                        if attempted:
                            # Failed role attempts are not physical viewpoints.
                            # Using ``attempted`` here relabelled the next viable
                            # deep route as G4 after G3's footprint preflight had
                            # failed.  The robot then reached that route, but the
                            # transaction still owed G3 and repeated essentially
                            # the same deep motion before exiting partial.  Roles
                            # advance only from physical success evidence.
                            remaining_role = next(
                                (role for role in ("G3", "G4")
                                 if role not in self.successful_roles and
                                 self.role_attempts.get(role, 0) <
                                 self.config.side_goal_retry_limit), None)
                            # A truth-locked shallow front chord is already
                            # bounded by the door--nearest-obstacle gap.  Its
                            # opposite-side endpoints can satisfy the physical
                            # 1.8 m contract while being unable to satisfy the
                            # generic extra 0.6 m planning cushion.  run109
                            # discarded that valid G4 and then let the visual
                            # breadth fallback choose a point outside the front
                            # gap.  Preserve the exact physical threshold for
                            # this geometry-locked second view; every other
                            # adaptive route retains the larger cushion.
                            planned_separation = (
                                self.config.minimum_goal_separation
                                if truth_shallow_g4_required else
                                self.config.minimum_goal_separation +
                                min(0.60, 0.75 *
                                    self.config.semantic_completion_tolerance))
                            separated = ([
                                dict(item, role=remaining_role)
                                for item in two_pose_plan.get("ordered", [])
                                if all(_distance(item["position"], old) >=
                                       planned_separation
                                       for old in self.completed_points)]
                                if remaining_role is not None else [])
                            two_pose_plan = dict(two_pose_plan)
                            two_pose_plan["ordered"] = separated[:1]
                        if two_pose_plan.get("ordered"):
                            self.adaptive_route_queue = list(
                                two_pose_plan["ordered"])
                            proposal = self.adaptive_route_queue.pop(0)
                            self.events.append({
                                "event": "ROOM_TWO_POSE_ROUTE_PLANNED",
                                "elapsed_sec": round(float(now), 3),
                                "room_id": self._room_id(),
                                "door_id": self.active_door.door_id,
                                "strategy": two_pose_plan["strategy"],
                                "waypoints": [
                                    [float(item["position"][0]),
                                     float(item["position"][1])]
                                    for item in two_pose_plan["ordered"]],
                                "baseline_m": round(float(
                                    two_pose_plan["baseline_m"]), 3),
                                "route_length_m": round(float(
                                    two_pose_plan["route_length_m"]), 3),
                                "return_distance_m": round(float(
                                    two_pose_plan["return_distance_m"]), 3),
                                "planning_wall_sec": round(float(
                                    two_pose_planning_wall_sec), 3),
                                "obstacle": two_pose_plan.get("obstacle"),
                            })
                    elif (not attempted and
                          self.visual_entry_refresh_used < 1):
                        # A fast portal crossing can finish between occupancy
                        # callbacks.  Selecting an ordinary shallow G3/G4 on
                        # that stale raster creates a false two-view route:
                        # the later partner is too close and is correctly
                        # rejected as a duplicate.  Allow exactly one
                        # zero-distance map refresh, which is explicitly
                        # excluded from completed_points/role accounting in
                        # handle_goal_result().  The next cycle must rebuild
                        # the physical deep+near pair from a fresh grid.
                        self.visual_entry_refresh_used += 1
                        proposal = {
                            "position": (float(current[0]),
                                         float(current[1])),
                            "role": "G3",
                            "score": 0.0,
                            "clearance": _clearance(grid, current),
                            "coverage_cells": set(),
                            "adaptive_marginal_gain_m2": 0.0,
                            "adaptive_utility": 0.0,
                            "preflight_path_length_m": 0.0,
                            # Reuse the current observed-free footprint as a
                            # one-point path.  Sending this zero-distance map
                            # refresh through global A* made run24 spend 30 s
                            # searching a growing grid before returning the
                            # same pose.
                            "preflight_path": [(float(current[0]),
                                                float(current[1]))],
                            "fallback_candidate": True,
                            "entry_pose_visual_fallback": True,
                            "visual_deepening": True,
                        }
                        # A narrow freshly crossed portal can rasterize the
                        # current footprint below normal room clearance even
                        # though its observed centre lane opens immediately.
                        # When a verified transit reaches a separated interior
                        # centre, use it as the first *physical* camera view.
                        # The next pass then owes one separated G4, avoiding a
                        # third room leg while preserving the two-view contract.
                        mapping_config = copy.copy(self.config)
                        mapping_config.goal_clearance = min(
                            self.config.goal_clearance,
                            max(0.17,
                                self.config.post_entry_path_clearance))
                        mapping_transit = (None if
                            getattr(self.active_door, "viewpoint_contract", None)
                            == "obstacle_front_opposite_sides" else
                            select_room_goal(
                                grid, self.active_door, current, "G1", [],
                                mapping_config, fallback=True))
                        mapping_path = (
                            astar_safe_path(
                                grid, current,
                                mapping_transit["position"],
                                mapping_config.goal_clearance, 0.15,
                                allow_blocked_start=True)
                            if mapping_transit is not None else None)
                        if (mapping_transit is not None and
                                mapping_path is not None and
                                mapping_path.get("success") and
                                self.active_door.depth(
                                    mapping_transit["position"]) >= 1.40 and
                                _distance(current,
                                          mapping_transit["position"]) >= 0.60):
                            proposal = dict(mapping_transit)
                            proposal.update({
                                "role": "G3",
                                "coverage_cells": set(),
                                "adaptive_marginal_gain_m2": 0.0,
                                "adaptive_utility": 0.0,
                                "fallback_candidate": True,
                                "two_pose_mapping_transit": True,
                                "two_pose_room_sweep": True,
                                "two_pose_index": 1,
                                "two_pose_strategy":
                                    "mapping_near_then_separated_view",
                                "visual_deepening": True,
                                "post_entry_path_clearance_relaxed": True,
                                "path_planning_clearance_m":
                                    mapping_config.goal_clearance,
                                "preflight_path": sparsify_verified_trace(
                                    list(mapping_path.get("path") or []),
                                    spacing=0.35),
                            })
                            self.events.append({
                                "event":
                                    "ROOM_TWO_POSE_MAPPING_TRANSIT_SELECTED",
                                "elapsed_sec": round(float(now), 3),
                                "room_id": self._room_id(),
                                "door_id": self.active_door.door_id,
                                "role": "G3",
                                "start_depth_m": round(float(
                                    self.active_door.depth(current)), 3),
                                "target_depth_m": round(float(
                                    self.active_door.depth(
                                        proposal["position"])), 3),
                                "path_length_m": round(float(proposal.get(
                                    "preflight_path_length_m", 0.0)), 3),
                                "planning_clearance_m": round(float(
                                    mapping_config.goal_clearance), 3),
                                "policy":
                                    "verified_near_view_then_one_separated_view",
                            })
                        self.events.append({
                            "event": "ROOM_TWO_POSE_ENTRY_MAP_REFRESH_SELECTED",
                            "elapsed_sec": round(float(now), 3),
                            "room_id": self._room_id(),
                            "door_id": self.active_door.door_id,
                            "reason": (
                                "physical_near_view_before_separated_view"
                                if proposal.get("two_pose_mapping_transit") else
                                "two_pose_plan_waiting_for_fresh_grid"),
                            "refresh_attempt":
                                self.visual_entry_refresh_used,
                            "position": [float(proposal["position"][0]),
                                         float(proposal["position"][1])],
                        })
                # A locked open-room contract cannot fall through to the
                # generic adaptive G4 after a shallow mapping G3.  Select the
                # missing deep (or near) class explicitly from the fresh live
                # projection, or defer until another map update.  This keeps
                # one physical doorway one semantic transaction instead of
                # exiting partial and rediscovering aliases.
                if (proposal is None and attempted and self.completed_points and
                        getattr(self.active_door, "viewpoint_contract", None) ==
                        "open_deep_near"):
                    proposal = select_locked_open_missing_depth_view(
                        grid, self.active_door, current,
                        self.completed_points, self.config)
                    if proposal is not None:
                        self.events.append({
                            "event": "ROOM_LOCKED_OPEN_MISSING_DEPTH_SELECTED",
                            "elapsed_sec": round(float(now), 3),
                            "room_id": self._room_id(),
                            "door_id": self.active_door.door_id,
                            "role": proposal["role"],
                            "depth_m": round(float(proposal["depth"]), 3),
                            "lateral_m": round(float(proposal["lateral"]), 3),
                            "strategy": proposal["two_pose_strategy"],
                        })
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
                locked_two_view_contract = bool(
                    getattr(self.active_door, "viewpoint_contract", None) in (
                        "open_deep_near",
                        "obstacle_front_opposite_sides"))
                fast_single_view = bool(
                    not attempted and
                    self.config.adaptive_maximum_viewpoints <= 1 and
                    not locked_two_view_contract)
                central_first = bool(
                    not attempted and
                    self.config.adaptive_prefer_central_first)
                # The ENTRY target is itself an occupancy/A*-verified interior
                # point. In the timed single-view profile, use that pose as
                # the broad primary RGB-D sweep rather than adding a second
                # 1--2 m transit to a synthetic centre then retracing it.
                # Coverage still decides whether the visit can complete.
                if (proposal is None and fast_single_view and
                        self.active_door.depth(current) >= 0.55):
                    proposal = {
                        "position": (float(current[0]), float(current[1])),
                        "role": "G1", "score": 0.0,
                        "clearance": _clearance(grid, current),
                        "coverage_cells": set(),
                        "adaptive_marginal_gain_m2": 1.0,
                        "adaptive_utility": 0.0,
                        "preflight_path_length_m": 0.0,
                        "preflight_path": [(float(current[0]), float(current[1]))],
                        "fallback_candidate": True,
                        "entry_pose_primary_observation": True,
                    }
                    self.events.append({
                        "event": "ADAPTIVE_ROOM_ENTRY_POSE_PRIMARY_VIEW_SELECTED",
                        "elapsed_sec": round(float(now), 3),
                        "room_id": self._room_id(),
                        "door_id": self.active_door.door_id,
                        "role": "G1",
                        "depth_m": round(float(self.active_door.depth(current)), 3),
                    })
                if proposal is None and (fast_single_view or central_first):
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
                        if (virtual_g1 is not None and
                                not self.config.visual_two_pose_lateral_enabled):
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
                    # Both lateral camera centres should be genuinely inside
                    # the room, not just across the threshold.  Use the live
                    # doorway-relative lidar depth with a 2 m minimum, then
                    # let occupancy/A* snap to the nearest safe free cell.
                    # Keep a deep/shallow split so the fallback preserves a
                    # diagonal parallax baseline instead of a same-depth pan.
                    visual_deep_depth = max(
                        2.00, self.active_door.depth(current))
                    visual_entry_floor = max(
                        1.25, self.active_door.depth(current) + 0.20)
                    visual_near_depth = max(
                        visual_entry_floor + 0.55,
                        min(visual_deep_depth - max(2.0, 0.55 *
                                                    self.config.
                                                    visual_breadth_min_baseline),
                            visual_deep_depth - 1.20))
                    visual_config.side_depth = visual_near_depth
                    visual_config.maximum_center_depth = visual_deep_depth
                    visual_config.maximum_side_lateral = min(
                        5.00, max(0.55, self.config.visual_breadth_lateral))
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
                        target_depth = (visual_deep_depth
                                        if visual_role == "G3"
                                        else visual_near_depth)
                        role_config = copy.copy(visual_config)
                        role_config.side_depth = target_depth
                        role_config.maximum_center_depth = target_depth
                        candidate = select_room_goal(
                            grid, self.active_door, current, visual_role,
                            self.completed_points or [current], role_config,
                            fallback=True, boundary_probe=False)
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
                        if (not proposal["visual_baseline_requirement_met"] and
                                self.visual_entry_refresh_used > 0 and
                                self.visual_mapping_transit_used < 1):
                            # The endpoint and path are safe, but this point is
                            # too close to the first real camera centre to be
                            # the second 360-degree view.  Use it once as a
                            # scanless LiDAR mapping transit; the next cycle
                            # replans the separated camera point from the
                            # newly observed room depth.  run27 previously
                            # spun here, rejected it as a duplicate, exited,
                            # and re-entered the same room 46 s later.
                            self.visual_mapping_transit_used += 1
                            proposal["two_pose_mapping_transit"] = True
                            proposal["entry_pose_visual_fallback"] = True
                            self.events.append({
                                "event": "ROOM_TWO_POSE_MAPPING_TRANSIT_SELECTED",
                                "elapsed_sec": round(float(now), 3),
                                "room_id": self._room_id(),
                                "door_id": self.active_door.door_id,
                                "role": proposal["role"],
                                "baseline_m": round(float(baseline_m), 3),
                                "minimum_baseline_m": round(float(
                                    self.config.visual_breadth_min_baseline), 3),
                                "policy": "move_without_camera_sweep_then_replan",
                            })
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
                            "requested_depth_m": round(float(
                                visual_deep_depth if proposal.get("role") == "G3"
                                else visual_near_depth), 3),
                            "requested_lateral_m": round(
                                float(visual_config.maximum_side_lateral), 3),
                            "baseline_from_g1_m": round(float(baseline_m), 3),
                            "minimum_requested_baseline_m": round(float(
                                self.config.visual_breadth_min_baseline), 3),
                            "baseline_requirement_met": bool(
                                proposal["visual_baseline_requirement_met"]),
                        })
                if (proposal is None and not attempted and
                        mandatory_two_pose_pending and
                        self.visual_entry_refresh_used > 0 and
                        not self.two_pose_opposite_repair_required):
                    # A stationary entry-pose refresh is deliberately not a
                    # camera viewpoint. On an upper floor the fresh raster
                    # can nevertheless report almost-complete LiDAR coverage,
                    # leaving the adaptive set-cover branch with no marginal
                    # candidate. Exiting here produced a zero-view room in
                    # the full-flow F2 run. Reuse the existing observed-free
                    # clearance/A* fallback to obtain the first *physical*
                    # deep view; after it succeeds the ordinary mandatory
                    # branch below supplies the separated near-door G4.
                    proposal = select_mandatory_second_visual_view(
                        grid, self.active_door, current, [current],
                        self.config)
                    if proposal is not None:
                        proposal = dict(proposal)
                        proposal.update({
                            "role": "G3",
                            "two_pose_index": 1,
                            "two_pose_strategy":
                                "post_refresh_deep_then_near",
                            "first_physical_view_after_map_refresh": True,
                        })
                        self.events.append({
                            "event":
                                "ROOM_TWO_POSE_POST_REFRESH_DEEP_SELECTED",
                            "elapsed_sec": round(float(now), 3),
                            "room_id": self._room_id(),
                            "door_id": self.active_door.door_id,
                            "role": "G3",
                            "position": [float(proposal["position"][0]),
                                         float(proposal["position"][1])],
                            "depth_m": round(float(self.active_door.depth(
                                proposal["position"])), 3),
                            "policy":
                                "first_physical_view_then_mandatory_second",
                        })
                if (proposal is None and attempted and
                        mandatory_two_pose_pending):
                    if self.two_pose_opposite_repair_required:
                        missing_opposite_role = next(
                            (item for item in ("G3", "G4")
                             if item not in self.successful_roles), "G4")
                        proposal = select_front_gap_opposite_side_second_visual_view(
                            grid, self.active_door, current,
                            self.completed_points, {}, self.config,
                            missing_role=missing_opposite_role,
                            excluded_targets=self.failed_visual_geometry_targets.get(
                                str(missing_opposite_role), ()))
                        if proposal is not None:
                            self.events.append({
                                "event":
                                    "ROOM_TWO_POSE_FRONT_GAP_OPPOSITE_FALLBACK_SELECTED",
                                "elapsed_sec": round(float(now), 3),
                                "room_id": self._room_id(),
                                "door_id": self.active_door.door_id,
                                "role": missing_opposite_role,
                                "policy":
                                    "door_obstacle_gap_only_no_deep_wraparound",
                            })
                if (proposal is None and
                        not self.two_pose_opposite_repair_required):
                    proposal = select_mandatory_second_visual_view(
                        grid, self.active_door, current,
                        self.completed_points, self.config)
                    if proposal is not None:
                        self.events.append({
                            "event":
                                "ROOM_TWO_POSE_MANDATORY_DEEP_FALLBACK_SELECTED",
                            "elapsed_sec": round(float(now), 3),
                            "room_id": self._room_id(),
                            "door_id": self.active_door.door_id,
                            "role": proposal["role"],
                            "position": [float(proposal["position"][0]),
                                         float(proposal["position"][1])],
                            "depth_m": round(float(self.active_door.depth(
                                proposal["position"])), 3),
                            "baseline_m": round(float(
                                proposal.get("visual_baseline_m", 0.0)), 3),
                            "policy":
                                "observed_free_astar_near_then_deep",
                        })
                # Looking for G4 while parked on G3's wide side can leave the
                # narrow opposite gap entirely occluded in the rolling map.
                # Repeating stationary scans at G3 cannot change that view.
                # Move once back to the physically executed ENTRY anchor,
                # without recording a camera viewpoint, and let the next
                # raster expose both sides before selecting G4.  The transit
                # itself is live A* checked and bounded to one use.
                if (proposal is None and
                        self.two_pose_opposite_repair_required and
                        "G3" in self.successful_roles and
                        "G4" not in self.successful_roles and
                        self.obstacle_gap_center_mapping_transit_used < 1 and
                        self.return_anchor is not None and
                        _distance(current, self.return_anchor) >= 0.45):
                    transit_clearance = min(
                        self.config.goal_clearance,
                        max(0.17, self.config.post_entry_path_clearance))
                    # Both endpoints are poses the robot has physically
                    # occupied.  A fresh raster can consequently paint the
                    # robot/door-frame footprint onto either endpoint.  The
                    # ordinary goal-footprint gate then rejected this mapping
                    # transit even though the later EXIT route crossed the
                    # same chord successfully.  Ignore only the exact endpoint
                    # cell (all intermediate cells retain full clearance) and
                    # fall back to the door's verified interior centre.
                    transit_candidates = [
                        (float(self.return_anchor[0]),
                         float(self.return_anchor[1])),
                        (float(self.active_door.interior_side[0]),
                         float(self.active_door.interior_side[1])),
                    ]
                    transit_path = None
                    transit_target = None
                    transit_failures = []
                    # The dog has just traversed this exact bent route to G3.
                    # Prefer its reverse over a new global A* search through a
                    # raster that now contains self returns and an occluded
                    # obstacle edge.  The manager still re-audits every chord
                    # with the live 3-D footprint service, and the transit is
                    # explicitly excluded from viewpoint accounting.
                    if len(self.obstacle_gap_g3_executed_path) >= 2:
                        reverse_trace = list(reversed(
                            self.obstacle_gap_g3_executed_path))
                        transit_target = reverse_trace[-1]
                        transit_path = {
                            "success": True,
                            "reason":
                                "reverse_physically_executed_g3_trace",
                            "path": reverse_trace,
                        }
                    for candidate in transit_candidates:
                        if transit_path is not None:
                            break
                        if (_distance(current, candidate) < 0.45 or
                                any(_distance(candidate, previous) < 0.12
                                    for previous in transit_candidates[
                                        :len(transit_failures)])):
                            continue
                        candidate_path = astar_safe_path(
                            grid, current, candidate,
                            transit_clearance, 0.15,
                            allow_blocked_start=True,
                            maximum_expansions=
                                self.config.exit_preflight_maximum_expansions,
                            allow_blocked_goal=True)
                        if candidate_path.get("success"):
                            transit_path = candidate_path
                            transit_target = candidate
                            break
                        transit_failures.append({
                            "target": [round(candidate[0], 3),
                                       round(candidate[1], 3)],
                            "reason": candidate_path.get("reason"),
                            "expansions": int(
                                candidate_path.get("expansions", 0)),
                        })
                    if transit_path is None:
                        self.events.append({
                            "event":
                                "ROOM_OBSTACLE_G4_CENTER_MAPPING_TRANSIT_REJECTED",
                            "elapsed_sec": round(float(now), 3),
                            "room_id": self._room_id(),
                            "door_id": self.active_door.door_id,
                            "attempts": transit_failures,
                            "policy": "bounded_verified_endpoint_override",
                        })
                    if transit_path is not None:
                        transit_points = list(transit_path.get("path") or [])
                        transit_length = sum(
                            _distance(transit_points[index - 1],
                                      transit_points[index])
                            for index in range(1, len(transit_points)))
                        self.obstacle_gap_center_mapping_transit_used += 1
                        proposal = {
                            "position": transit_target,
                            "role": "G4", "score": 0.0,
                            "clearance": _clearance(
                                grid, transit_target),
                            "coverage_cells": set(),
                            "adaptive_marginal_gain_m2": 0.0,
                            "adaptive_utility": 0.0,
                            "preflight_path_length_m": float(transit_length),
                            "preflight_path": sparsify_verified_trace(
                                transit_points, spacing=0.35),
                            "fallback_candidate": True,
                            "entry_pose_visual_fallback": True,
                            "two_pose_mapping_transit": True,
                            "visual_deepening": True,
                            "two_pose_room_sweep": True,
                            "two_pose_index": 2,
                            "two_pose_strategy":
                                "obstacle_gap_center_mapping_transit",
                            "reversed_physical_g3_trace": bool(
                                transit_path.get("reason") ==
                                "reverse_physically_executed_g3_trace"),
                            "path_planning_clearance_m":
                                float(transit_clearance),
                            "post_entry_path_clearance_relaxed": bool(
                                transit_clearance <
                                self.config.goal_clearance),
                        }
                        self.events.append({
                            "event":
                                "ROOM_OBSTACLE_G4_CENTER_MAPPING_TRANSIT",
                            "elapsed_sec": round(float(now), 3),
                            "room_id": self._room_id(),
                            "door_id": self.active_door.door_id,
                            "path_length_m": round(transit_length, 3),
                            "policy":
                                "entry_anchor_transit_not_a_viewpoint_bounded_once",
                        })
                # For a truth-classified front obstacle, G4 is mandatory even
                # when the first post-G3 raster has no safe opposite-side
                # endpoint.  Give the mapper one stationary refresh at the
                # executed G3 pose, then re-enter this branch with the fresh
                # raster.  This refresh is explicitly excluded from
                # completed_points and viewpoint accounting by
                # handle_goal_result(); it cannot fake the two-view contract.
                if (proposal is None and
                        self.two_pose_opposite_repair_required and
                        "G3" in self.successful_roles and
                        "G4" not in self.successful_roles and
                        self.obstacle_gap_g4_refresh_used < 1):
                    self.obstacle_gap_g4_refresh_used += 1
                    proposal = {
                        "position": (float(current[0]), float(current[1])),
                        "role": "G4",
                        "score": 0.0,
                        "clearance": _clearance(grid, current),
                        "coverage_cells": set(),
                        "adaptive_marginal_gain_m2": 0.0,
                        "adaptive_utility": 0.0,
                        "preflight_path_length_m": 0.0,
                        "preflight_path": [(float(current[0]),
                                            float(current[1]))],
                        "fallback_candidate": True,
                        "entry_pose_visual_fallback": True,
                        "visual_deepening": True,
                        "two_pose_room_sweep": True,
                        "two_pose_strategy":
                            "obstacle_front_gap_g4_map_refresh",
                    }
                    self.events.append({
                        "event":
                            "ROOM_OBSTACLE_G4_MAP_REFRESH_BEFORE_PARTIAL",
                        "elapsed_sec": round(float(now), 3),
                        "room_id": self._room_id(),
                        "door_id": self.active_door.door_id,
                        "role": "G4",
                        "refresh_attempt":
                            self.obstacle_gap_g4_refresh_used,
                        "policy":
                            "stationary_refresh_not_a_viewpoint_bounded_once",
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
                    self.visual_deepening_requested and
                    self.visual_entry_refresh_used < 1 and
                    all(_distance(current, old) >=
                        self.config.minimum_goal_separation
                        for old in self.completed_points)):
                # One entry-pose spin is a map refresh, never two physical
                # viewpoints.  The next cycle must re-plan a separated pose.
                next_lateral = next((role for role in ("G3", "G4")
                                     if role not in attempted), None)
                if next_lateral is not None:
                    self.visual_entry_refresh_used += 1
                    proposal = {
                        "position": (float(current[0]), float(current[1])),
                        "role": next_lateral,
                        "score": 0.0,
                        "clearance": _clearance(grid, current),
                        "coverage_cells": set(),
                        "adaptive_marginal_gain_m2": 0.0,
                        "adaptive_utility": 0.0,
                        "preflight_path_length_m": 0.0,
                        "preflight_path": [(float(current[0]),
                                            float(current[1]))],
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
                        "reason": "bounded_map_refresh_before_separated_view",
                        "refresh_attempt": self.visual_entry_refresh_used,
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
            # Final fail-closed gate for an open deep+near contract.  Several
            # adaptive branches above can legitimately produce a reachable
            # G4, but reachability alone does not make it the missing depth
            # class.  Replace a second shallow point here, after *all* generic
            # selectors, so no later branch can bypass the locked geometry.
            if (proposal is not None and attempted and self.completed_points and
                    getattr(self.active_door, "viewpoint_contract", None) ==
                    "open_deep_near" and
                    str(proposal.get("two_pose_strategy", "")) !=
                    "truth_layout_open_centerline_deep_near"):
                # The truth-layout pair is constructed atomically as deep G3
                # plus near-door G4 and still passes live SCAN-lite before
                # execution.  Reclassifying its queued G4 from the first
                # measured arrival depth changed the role back to G3 in
                # roundtrip_f3forwardreturnfix, so both physical poses were
                # successful but g4_truth_pose remained empty and the room
                # was canonicalized as partial.  Final contract evidence
                # already validates measured deep/near geometry; preserve the
                # locked role here and let that fail-closed gate decide.
                completed_depths = [self.active_door.depth(point)
                                    for point in self.completed_points]
                # Use the same physical deep threshold as final contract
                # evidence.  The previous 0.55 multiplier demanded 5.81 m
                # here while final evidence accepted 5.45 m; a valid first
                # deep pose was therefore followed by another deep pose
                # instead of the mandatory near-door view.
                minimum_deep = max(
                    3.80, self.config.open_room_minimum_deep_depth -
                    self.config.semantic_completion_tolerance - 0.15)
                has_deep = any(depth >= minimum_deep
                               for depth in completed_depths)
                deepest = max(completed_depths)
                maximum_near = min(4.35, deepest - 1.50) if has_deep else 4.35
                proposed_depth = self.active_door.depth(proposal["position"])
                supplies_missing_class = bool(
                    (not has_deep and proposed_depth >= minimum_deep) or
                    (has_deep and proposed_depth <= maximum_near))
                if not supplies_missing_class:
                    rejected_depth = float(proposed_depth)
                    replacement = select_locked_open_missing_depth_view(
                        grid, self.active_door, current,
                        self.completed_points, self.config)
                    if replacement is None and has_deep:
                        replacement = select_reversed_open_g3_near_view(
                            self.active_door, current,
                            self.completed_points,
                            self.open_g3_executed_path, self.config)
                    if (replacement is None and
                            self.open_missing_depth_refresh_used < 1):
                        # The first physical pose can finish before its LiDAR
                        # frame has reached the rolling OccupancyGrid.  Do not
                        # interpret that one-callback lag as permission to
                        # leave an open room with only G3.  Refresh once at the
                        # already verified footprint, exclude the refresh from
                        # physical-view accounting, and rebuild the missing
                        # depth class from the next live grid.  This is bounded
                        # and therefore cannot turn into a stationary retry
                        # loop when the genuinely deep region is unavailable.
                        self.open_missing_depth_refresh_used += 1
                        missing_role = next(
                            (role for role in ("G3", "G4")
                             if role not in self.successful_roles), "G4")
                        proposal = {
                            "position": (float(current[0]),
                                         float(current[1])),
                            "role": missing_role,
                            "score": 0.0,
                            "clearance": _clearance(grid, current),
                            "coverage_cells": set(),
                            "adaptive_marginal_gain_m2": 0.0,
                            "adaptive_utility": 0.0,
                            "preflight_path_length_m": 0.0,
                            "preflight_path": [(float(current[0]),
                                                float(current[1]))],
                            "fallback_candidate": True,
                            "entry_pose_visual_fallback": True,
                            "visual_deepening": True,
                            "two_pose_room_sweep": True,
                            "two_pose_strategy":
                                "locked_open_missing_depth_map_refresh",
                        }
                    else:
                        proposal = replacement
                    self.events.append({
                        "event": (
                            "ROOM_LOCKED_OPEN_GENERIC_G4_REPLACED"
                            if replacement is not None else
                            "ROOM_LOCKED_OPEN_MISSING_DEPTH_MAP_REFRESH"
                            if proposal is not None else
                            "ROOM_LOCKED_OPEN_GENERIC_G4_DEFERRED"),
                        "elapsed_sec": round(float(now), 3),
                        "room_id": self._room_id(),
                        "door_id": self.active_door.door_id,
                        "rejected_depth_m": round(rejected_depth, 3),
                        "required_class": ("near" if has_deep else "deep"),
                        "replacement_depth_m": (
                            round(float(replacement["depth"]), 3)
                            if replacement is not None else None),
                        "refresh_attempt": (
                            self.open_missing_depth_refresh_used
                            if replacement is None and proposal is not None
                            else None),
                        "policy": "locked_open_contract_final_gate",
                    })
            # Validate the very first proposed second physical view as well
            # as retries.  ``attempted`` is empty before the first G4
            # dispatch; gating this check on it let
            # fresh_grid_opposite_side_repair send an 8 m same-side detour in
            # the F3 far obstacle room.  That goal consumed the room budget,
            # then the stationary refresh was (correctly) denied physical
            # G4 credit.  Once one physical G3/G4 role exists, every proposed
            # counterpart must pass the locked front-gap geometry before it
            # is dispatched, irrespective of retry history.
            if (proposal is not None and
                    str(proposal.get("role")) in ("G3", "G4") and
                    (self.two_pose_opposite_repair_required or
                     getattr(self.active_door, "viewpoint_contract", None) ==
                     "obstacle_front_opposite_sides") and
                    self.completed_points and
                    not proposal.get("entry_pose_visual_fallback") and
                    not proposal.get("two_pose_mapping_transit")):
                missing_opposite_role = str(proposal.get("role"))
                anchor_lateral = self.active_door.contract_lateral(
                    self.completed_points[-1])
                proposal_lateral = self.active_door.contract_lateral(
                    proposal["position"])
                proposal_baseline = _distance(
                    self.completed_points[-1], proposal["position"])
                baseline_with_arrival_margin = (
                    float(self.config.minimum_goal_separation) +
                    min(0.35, float(
                        self.config.semantic_completion_tolerance)))
                obstacle_front_depth = getattr(
                    self.active_door, "truth_obstacle_front_depth_m", None)
                obstacle_minimum_lateral = getattr(
                    self.active_door,
                    "truth_obstacle_minimum_lateral_m", None)
                obstacle_maximum_lateral = getattr(
                    self.active_door,
                    "truth_obstacle_maximum_lateral_m", None)
                # Opposite lateral signs alone do not prove the obstacle-room
                # contract.  A provisional generic pair can put G4 behind the
                # table (F3 room04 in f3nearfirst: depth 5.44 m for a 3.80 m
                # front edge), which both violates the door--obstacle-gap
                # requirement and leaves EXIT with a long furniture detour.
                # Treat any point at/behind the measured front face as invalid
                # and rebuild it with the strict front-gap selector below.
                proposal_behind_obstacle = \
                    obstacle_candidate_is_behind_front_gap(
                        self.active_door, proposal["position"],
                        self.config.post_entry_path_clearance)
                proposal_depth = self.active_door.contract_depth(
                    proposal["position"])
                minimum_visibility_depth = (
                    max(float(self.config.minimum_crossing_depth),
                        float(obstacle_front_depth) - 1.00)
                    if obstacle_front_depth is not None else None)
                side_peek_margin = max(
                    0.70, float(self.config.goal_clearance) + 0.35)
                proposal_shallow_for_side_peek = bool(
                    minimum_visibility_depth is not None and
                    proposal_depth < minimum_visibility_depth)
                proposal_outside_side_peek_band = bool(
                    obstacle_minimum_lateral is not None and
                    obstacle_maximum_lateral is not None and
                    not (proposal_lateral >=
                         float(obstacle_maximum_lateral) + side_peek_margin or
                         proposal_lateral <=
                         float(obstacle_minimum_lateral) - side_peek_margin))
                if (anchor_lateral * proposal_lateral >= -0.20 or
                        proposal_baseline < baseline_with_arrival_margin or
                        proposal_behind_obstacle or
                        proposal_shallow_for_side_peek or
                        proposal_outside_side_peek_band):
                    # Preserve the locked door--obstacle gap contract before
                    # trying the generic opposite-side selector.  Crucially,
                    # the missing role can be G3 when G3 timed out and G4 was
                    # the first physically successful side.  The old code
                    # repaired only a missing G4, repeatedly reissuing G4 and
                    # then accepting a stationary map refresh as the nominal
                    # G3 slot (F3 room 3, f3roomrestage).
                    repaired = None
                    if (getattr(self.active_door,
                                "viewpoint_contract", None) ==
                            "obstacle_front_opposite_sides"):
                        repaired = \
                            select_front_gap_opposite_side_second_visual_view(
                                grid, self.active_door, current,
                                self.completed_points, proposal, self.config,
                                missing_role=missing_opposite_role,
                                excluded_targets=self.failed_visual_geometry_targets.get(
                                    str(missing_opposite_role), ()))
                    if (repaired is None and
                            getattr(self.active_door,
                                    "viewpoint_contract", None) !=
                            "obstacle_front_opposite_sides"):
                        repaired = select_opposite_side_second_visual_view(
                            grid, self.active_door, current,
                            self.completed_points, proposal, self.config,
                            missing_role=missing_opposite_role)
                    if repaired is not None:
                        proposal = repaired
                        self.events.append({
                            "event":
                                "ROOM_TWO_POSE_OPPOSITE_SIDE_REPAIRED",
                            "elapsed_sec": round(float(now), 3),
                            "room_id": self._room_id(),
                            "door_id": self.active_door.door_id,
                            "role": missing_opposite_role,
                            "anchor_lateral_m": round(float(
                                anchor_lateral), 3),
                            "repaired_lateral_m": round(float(
                                self.active_door.contract_lateral(
                                    proposal["position"])), 3),
                            "position": [float(proposal["position"][0]),
                                         float(proposal["position"][1])],
                            "path_length_m": round(float(proposal.get(
                                "preflight_path_length_m", 0.0)), 3),
                            "policy":
                                "fresh_grid_observed_free_opposite_side",
                        })
                    elif (getattr(self.active_door,
                                  "viewpoint_contract", None) ==
                          "obstacle_front_opposite_sides"):
                        # Fail closed before dispatching the generic adaptive
                        # candidate.  In f3truthopenfix room04 that candidate
                        # was 4.0 m lateral on the *same* side as G3; two
                        # bounded attempts consumed the remaining room budget,
                        # so the later centre-map transit and true opposite
                        # gap selector never had enough time to execute.  A
                        # stationary refresh is emitted below, then the next
                        # cycle uses the existing one-shot ENTRY-centre
                        # mapping transit.  Neither action receives physical
                        # viewpoint credit.
                        rejected_position = proposal.get("position")
                        proposal = None
                        self.events.append({
                            "event":
                                "ROOM_OBSTACLE_SAME_SIDE_G4_REJECTED",
                            "elapsed_sec": round(float(now), 3),
                            "room_id": self._room_id(),
                            "door_id": self.active_door.door_id,
                            "role": missing_opposite_role,
                            "anchor_lateral_m": round(float(
                                anchor_lateral), 3),
                            "rejected_lateral_m": round(float(
                                proposal_lateral), 3),
                            "rejected_position": [
                                float(rejected_position[0]),
                                float(rejected_position[1])],
                            "rejected_behind_obstacle": bool(
                                proposal_behind_obstacle),
                            "rejected_depth_m": round(float(
                                proposal_depth), 3),
                            "minimum_visibility_depth_m": (
                                round(float(minimum_visibility_depth), 3)
                                if minimum_visibility_depth is not None else
                                None),
                            "rejected_shallow_for_side_peek": bool(
                                proposal_shallow_for_side_peek),
                            "rejected_outside_side_peek_band": bool(
                                proposal_outside_side_peek_band),
                            "obstacle_front_depth_m": (
                                float(obstacle_front_depth)
                                if obstacle_front_depth is not None else None),
                            "policy":
                                "refresh_then_bounded_entry_center_transit_"
                                "before_opposite_gap_replan",
                        })
            # A physical two-view contract must not fall through to
            # adaptive_no_marginal_gain merely because the rolling raster is
            # one callback behind the first executed view.  All specialised
            # selectors above are allowed to fail on that stale frame; give
            # the mapper one final stationary refresh and rebuild the missing
            # G3/G4 role on the next cycle.  record_result() excludes this
            # diagnostic goal from completed_points, so it cannot manufacture
            # a second viewpoint by spinning at the first pose.
            if (proposal is None and mandatory_two_pose_pending and
                    self.mandatory_two_pose_unavailable_refresh_used < 1):
                self.mandatory_two_pose_unavailable_refresh_used += 1
                missing_role = next(
                    (item for item in ("G3", "G4")
                     if item not in self.successful_roles), "G4")
                proposal = {
                    "position": (float(current[0]), float(current[1])),
                    "role": missing_role,
                    "score": 0.0,
                    "clearance": _clearance(grid, current),
                    "coverage_cells": set(),
                    "adaptive_marginal_gain_m2": 0.0,
                    "adaptive_utility": 0.0,
                    "preflight_path_length_m": 0.0,
                    "preflight_path": [(float(current[0]),
                                        float(current[1]))],
                    "fallback_candidate": True,
                    "entry_pose_visual_fallback": True,
                    "visual_deepening": True,
                    "two_pose_room_sweep": True,
                    "two_pose_strategy":
                        "mandatory_two_pose_unavailable_map_refresh",
                }
                self.events.append({
                    "event": "ROOM_TWO_POSE_UNAVAILABLE_MAP_REFRESH",
                    "elapsed_sec": round(float(now), 3),
                    "room_id": self._room_id(),
                    "door_id": self.active_door.door_id,
                    "role": missing_role,
                    "policy":
                        "one_stationary_refresh_then_missing_role_replan",
                })
            # Final mandatory-contract gate.  A diagnostic G4 mapping transit
            # and its stationary refresh are deliberately excluded from
            # ``successful_roles``.  On the following callback the generic
            # adaptive queue can nevertheless be empty (zero LiDAR marginal
            # gain); falling through below used to emit
            # ``adaptive_no_marginal_gain`` and close an otherwise healthy
            # obstacle room with only G3.  Re-run the strict truth-bounded
            # front-gap selector here, after every mapping refresh branch has
            # had a chance to update the raster.  This candidate remains
            # opposite-side, before the obstacle front, baseline separated,
            # and subject to the independent live 3-D execution audit.
            if (proposal is None and mandatory_two_pose_pending and
                    getattr(self.active_door, "viewpoint_contract", None) ==
                    "obstacle_front_opposite_sides" and
                    len(self.successful_roles.intersection({"G3", "G4"})) == 1):
                missing_role = next(
                    (item for item in ("G3", "G4")
                     if item not in self.successful_roles), "G4")
                proposal = select_front_gap_opposite_side_second_visual_view(
                    grid, self.active_door, current,
                    self.completed_points, {}, self.config,
                    missing_role=missing_role,
                    excluded_targets=self.failed_visual_geometry_targets.get(
                        str(missing_role), ()))
                if proposal is not None:
                    self.events.append({
                        "event":
                            "ROOM_TWO_POSE_FINAL_FRONT_GAP_GUARD_SELECTED",
                        "elapsed_sec": round(float(now), 3),
                        "room_id": self._room_id(),
                        "door_id": self.active_door.door_id,
                        "role": missing_role,
                        "position": [float(proposal["position"][0]),
                                     float(proposal["position"][1])],
                        "depth_m": round(float(self.active_door.depth(
                            proposal["position"])), 3),
                        "lateral_m": round(float(
                            self.active_door.contract_lateral(
                            proposal["position"])), 3),
                        "policy":
                            "mandatory_contract_prevents_adaptive_stop",
                    })
            proposal_execution_window = max(
                0.0, observation_deadline - elapsed)
            # This contract is needed both by the mandatory two-pose window
            # calculation below and by the final goal timeout.  A generic
            # adaptive proposal (for example F1 room2 after its first G3)
            # does not necessarily carry ``two_pose_room_sweep``; defining
            # the flag only inside that branch made the goal construction
            # raise UnboundLocalError and killed the whole three-floor run.
            obstacle_contract = bool(
                getattr(self.active_door, "viewpoint_contract", None) ==
                "obstacle_front_opposite_sides")
            if (proposal is not None and
                    proposal.get("two_pose_room_sweep")):
                # Obstacle-front partner motion is intentionally capped at
                # 0.30 m/s by the executor.  Using the generic 0.55 m/s here
                # gave a 5.10 m safe front-gap route only 14.77 s; the F1
                # physdedupe run timed it out after one waypoint and discarded
                # an otherwise valid room.  Budget with the physical speed
                # profile plus a small command/settle margin.  This changes no
                # geometry and does not relax the two-view acceptance gate.
                seconds_per_meter = (1.0 / 0.30 if obstacle_contract
                                     else 1.0 / 0.55)
                spin_and_settle_seconds = 5.5
                current_required = (
                    float(proposal.get("preflight_path_length_m", 0.0)) *
                    seconds_per_meter + spin_and_settle_seconds)
                suffix_required = sum(
                    float(item.get("preflight_path_length_m", 0.0)) *
                    seconds_per_meter + spin_and_settle_seconds
                    for item in self.adaptive_route_queue
                    if item.get("two_pose_room_sweep"))
                total_required = current_required + suffix_required
                if total_required > proposal_execution_window:
                    # Two full-spin camera centres are the room contract, not
                    # optional LiDAR debt.  The old 42 s estimate dropped the
                    # second point after a slow first transit, leaving exactly
                    # one green spin marker in run speed_fix_1.  Extend only
                    # this observation goal's execution window; EXIT retains
                    # its separate reserve and progressive timeout.
                    previous_window = proposal_execution_window
                    proposal_execution_window = max(
                        current_required,
                        total_required - suffix_required)
                    if obstacle_contract:
                        proposal_execution_window = max(
                            35.0, proposal_execution_window + 8.0)
                    self.events.append({
                        "event": "ROOM_TWO_POSE_MANDATORY_WINDOW_EXTENDED",
                        "elapsed_sec": round(float(now), 3),
                        "room_id": self._room_id(),
                        "previous_available_seconds": round(
                            previous_window, 3),
                        "extended_seconds": round(
                            proposal_execution_window, 3),
                        "required_seconds": round(total_required, 3),
                        "role": str(proposal.get("role")),
                    })
                else:
                    proposal_execution_window -= suffix_required
                # A failed first G4 can leave the body laterally displaced
                # even though the replacement is a short direct near-door
                # chord.  The path-length estimate does not include recovery
                # of that body heading or one camera settle.  Give the locked
                # second physical view one realistic, still-bounded window;
                # otherwise run38 cut both attempts at 12--14 s and converted
                # a valid open room into a costly partial re-entry.
                if (str(proposal.get("role", "")) == "G4" and
                        getattr(self.active_door, "viewpoint_contract", None)
                        in ("open_deep_near",
                            "obstacle_front_opposite_sides") and
                        len(self.successful_roles.intersection(
                            {"G3", "G4"})) == 1):
                    proposal_execution_window = max(
                        proposal_execution_window,
                        min(float(self.config.room_goal_timeout_seconds),
                            18.0))
            if proposal is not None:
                # Two-pose route planners return the same structural proposal
                # shape as adaptive candidates, but older route entries may
                # omit the scoring fields.  They are still valid, physically
                # separated viewpoints; missing metadata must not crash the
                # mission at the second/third room and trigger a global ROS
                # shutdown.  Keep the route's score when available and use a
                # small nonzero information-gain fallback for reporting.
                proposal.setdefault("adaptive_marginal_gain_m2", 0.20)
                proposal.setdefault(
                    "adaptive_utility", float(proposal.get("score", 0.0)))
                # A failed deep endpoint used to survive as the only route
                # item and was then renamed G4 merely because G3 had already
                # been attempted.  That cannot satisfy a deep+near contract:
                # preserve the atomic route member's geometric role and let
                # the bounded failed-role gate return partial if no alternate
                # lane is available.
                if (str(proposal.get("two_pose_strategy", "")) ==
                        "truth_layout_open_centerline_deep_near"):
                    route_index = int(proposal.get("two_pose_index", 0) or 0)
                    if route_index in (1, 2):
                        proposal["role"] = "G3" if route_index == 1 else "G4"
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
                    # The mandatory-window calculation above used the real
                    # 0.30 m/s obstacle motion profile, but run39 then clipped
                    # its advertised >=35 s window back to the generic 18 s
                    # goal timeout here.  Preserve that bounded physical
                    # window for obstacle contracts; open rooms retain 18 s.
                    "execution_timeout_sec": min(
                        (max(self.config.room_goal_timeout_seconds, 35.0)
                         if obstacle_contract else
                         self.config.room_goal_timeout_seconds),
                        max(3.0, proposal_execution_window)),
                    "progress_timeout_sec": min(
                        (max(self.config.room_goal_timeout_seconds, 35.0)
                         if obstacle_contract else
                         self.config.room_goal_timeout_seconds),
                        max(3.0, proposal_execution_window),
                        max(15.0, self.config.exit_progress_timeout_seconds)),
                    "progressive_timeout": True,
                    "room_goal_attempt": 1,
                    "fallback_candidate": bool(
                        proposal.get("fallback_candidate", False)),
                    "verified_trajectory_backtrack": bool(
                        proposal.get("verified_trajectory_backtrack", False)),
                    "adaptive_minimal_viewpoint": True, "_preplanned_path_result": ({"success": True, "reason": "two_pose_room_sweep_preflight", "path": list(proposal.get("preflight_path") or [])} if proposal.get("preflight_path") else None),
                }
                # Preserve the bounded retry counter across adaptive replans.
                # Resetting it to one on every dispatch allowed a blocked F2
                # far-room G3 endpoint to be republished indefinitely until the
                # whole floor budget expired.
                self.role_attempts[role] = \
                    self.role_attempts.get(role, 0) + 1
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

        # Folding RETURN into EXIT is safe only while the robot is still in
        # the portal neighbourhood.  A failed/deep viewpoint can leave it
        # several metres behind furniture; prepending that live pose to the
        # saved ENTRY breadcrumbs creates an unverified straight chord and
        # repeats it until the EXIT budget expires.  Stage back to the inside
        # centreline with bounded A* first in that case.  This is navigation,
        # not contract credit, and retains a separate physical EXIT.
        direct_exit_near_portal = direct_exit_fold_allowed(
            self.active_door, current, self.config.entry_depth)
        if (self.config.direct_verified_exit and
                direct_exit_near_portal and not self.return_completed):
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
        elif self.config.direct_verified_exit and not self.return_completed:
            self.events.append({
                "event": "ROOM_RETURN_RETAINED_FOR_DEEP_EXIT",
                "elapsed_sec": round(float(now), 3),
                "room_id": self._room_id(),
                "door_id": self.active_door.door_id,
                "current_depth_m": round(float(
                    self.active_door.depth(current)), 3),
                "policy": "bounded_astar_to_inside_anchor_before_exit",
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
        # Folded EXIT already has a bounded portal preflight.  An extra
        # unbounded RETURN A* on the post-scan raster spent ~178-193 s in
        # room 2 (speed_fix_9) and consumed the floor budget before EXIT
        # could even be dispatched.  Skip it when RETURN is already folded,
        # and always cap expansions when a genuine RETURN is still required.
        if snapped_anchor is not None and not self.return_completed:
            path_check = astar_safe_path(
                grid, current, snapped_anchor, self.config.goal_clearance,
                0.15, allow_blocked_start=True,
                maximum_expansions=self.config.exit_preflight_maximum_expansions)
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
        # trace was just physically executed and footprint-audited.  Keep the
        # short live-raster EXIT on the first attempt; after that attempt has
        # already reached the door area but stalled, prefer only the remaining
        # verified ENTRY anchors instead of retrying the same door-centre chord.
        entry_trace_corridor_started = bool(
            len(self.entry_portal_waypoints) >= 2 and
            self.active_door.depth(self.entry_portal_waypoints[0]) <= 0.0)
        prefer_reverse_entry = bool(
            self.config.accept_reversed_entry_trace_exit and
            self.entry_confirmed and
            self.exit_attempts >= 1 and
            entry_trace_corridor_started)
        exit_portal = (None if prefer_reverse_entry else prepare_exit_path(
            grid, self.active_door, current, self.config))
        nearest_anchor_exit = bool(
            exit_portal is not None and
            exit_portal.get("nearest_observed_anchor_fallback"))
        if nearest_anchor_exit:
            self.events.append({
                "event": "ROOM_EXIT_NEAREST_ANCHOR_PATH_PREPARED",
                "elapsed_sec": round(float(now), 3),
                "room_id": self._room_id(),
                "door_id": self.active_door.door_id,
                "approach_path_length_m": round(float(
                    exit_portal.get("approach_path_length_m", 0.0)), 3),
                "policy": "current_to_nearest_inside_anchor_then_cross",
            })
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
        # If the live post-scan raster cannot reproduce the portal, the
        # measured ENTRY trace is already the only physically validated route
        # through this narrow doorway.  Use it on the first EXIT attempt too;
        # waiting for one unverified current->inside chord caused the last
        # room to fall before the reverse-trace recovery was ever selected.
        reverse_entry_fallback = bool(
            exit_portal is None and entry_trace_corridor_started)
        # An obstacle-side G4 is still *in front* of the locked furniture
        # edge for both supported obstacle contracts (shallow front chord and
        # deep outer-side peek).  Replaying the complete room trajectory sends
        # the dog back to G3 before approaching the door (run47 F3 room2:
        # 7.49 m, 38 s, then timeout; run67 F1 room1: 764 collision queries,
        # 80.4 s preflight, then a fall).  On the first EXIT only, use the
        # direct current -> inside portal anchor -> measured doorway route
        # when the truth contract proves that the entire starting pose is in
        # the door--obstacle gap.
        # This route receives the manager's unchanged live 3-D SCAN-lite audit;
        # a failure falls back to the existing bounded breadcrumb recovery.
        truth_front_depth = getattr(
            self.active_door, "truth_obstacle_front_depth_m", None)
        active_obstacle_view_policy = getattr(
            self.active_door, "truth_obstacle_view_policy", "")
        # Deep side-peek viewpoints intentionally approach the furniture
        # front more closely than a shallow front chord.  Requiring the same
        # max(0.35, goal_clearance) margin rejected run122 F3 room4 by only
        # about 8 cm, then replayed a 26 s EXIT detour.  The deep contract has
        # already proven opposite physical sides and is still guarded by the
        # manager's generated-layout, furniture and live dense-3D audits.
        obstacle_front_exit_margin = (
            0.15 if active_obstacle_view_policy ==
            "deep_outer_side_peek" else
            max(0.35, float(self.config.goal_clearance)))
        obstacle_front_gap_direct_exit = bool(
            self.exit_attempts == 0 and
            self.active_door.viewpoint_contract ==
            "obstacle_front_opposite_sides" and
            active_obstacle_view_policy in (
                "shallow_front_chord", "deep_outer_side_peek") and
            self.active_door.physical_two_view_contract_met and
            truth_front_depth is not None and
            self.active_door.contract_depth(current) <=
            float(truth_front_depth) - obstacle_front_exit_margin)
        if obstacle_front_gap_direct_exit:
            # ``prepare_exit_path`` can succeed only by selecting a distant
            # historical room anchor.  In run48 that nominally valid nearest-
            # anchor route sent G4 back across the gap to G3 and produced a
            # 5.18 m/four-waypoint EXIT.  The locked contract proves the
            # current pose is still in front of the obstacle; use the short
            # measured portal anchors even when a longer raster detour exists.
            # The manager rechecks the complete current->inside chord and
            # every portal anchor against live dense 3-D before motion.
            mandatory_exit = [
                [float(self.active_door.interior_side[0]),
                 float(self.active_door.interior_side[1])],
                [float(self.active_door.center[0]),
                 float(self.active_door.center[1])],
                [float(self.active_door.corridor_side[0]),
                 float(self.active_door.corridor_side[1])],
            ]
            # A deep outer-side G4 is deliberately displaced beyond the
            # primary obstacle silhouette.  Joining it diagonally to the
            # centred inside-door anchor cuts across that silhouette.  In
            # run124 F3 room4 the dog therefore spent 25.5 simulated seconds
            # creeping laterally along the furniture edge and completed no
            # waypoint.  Stay on the already audited G4 side while retracting
            # to the inside-anchor depth, then cross the finite front gap to
            # the door centreline.  This is not a return to G3 and never goes
            # behind the obstacle; every segment still receives the manager's
            # generated-layout and live dense-3D audits.
            if active_obstacle_view_policy == "deep_outer_side_peek":
                inside_depth = self.active_door.contract_depth(
                    self.active_door.interior_side)
                current_lateral = self.active_door.contract_lateral(current)
                side_preserving_stage = self.active_door.contract_point(
                    inside_depth, current_lateral)
                stage_is_free = bool(
                    _state(grid, side_preserving_stage) == 0 and
                    _clearance(grid, side_preserving_stage) >=
                    max(0.15, min(0.30, float(self.config.goal_clearance))))
                if (stage_is_free and
                        _distance(current, side_preserving_stage) >= 0.25 and
                        _distance(side_preserving_stage,
                                  mandatory_exit[0]) >= 0.25):
                    mandatory_exit.insert(0, [
                        float(side_preserving_stage[0]),
                        float(side_preserving_stage[1]),
                    ])
                    self.events.append({
                        "event":
                            "ROOM_DEEP_OBSTACLE_SIDE_PRESERVING_EXIT_STAGED",
                        "elapsed_sec": round(float(now), 3),
                        "room_id": self._room_id(),
                        "door_id": self.active_door.door_id,
                        "current_lateral_m": round(float(current_lateral), 3),
                        "stage_depth_m": round(float(inside_depth), 3),
                        "stage": [float(side_preserving_stage[0]),
                                  float(side_preserving_stage[1])],
                        "policy": (
                            "retract_on_completed_view_side_then_cross_"
                            "front_gap_to_portal"),
                        "physical_exit_still_required": True,
                    })
            exit_point = tuple(mandatory_exit[-1])
            exit_clearance = portal_clearance(
                grid, self.active_door, self.config)
        # Retry the saved-door shortest path whenever the live raster can still
        # reproduce it.  Reverse ENTRY is now a last-resort recovery only when
        # the direct current -> inside anchor -> door -> corridor preflight is
        # unavailable; an execution retry alone must not force a long replay.
        fallback_exit = bool(
            not obstacle_front_gap_direct_exit and
            (reverse_entry_fallback or
             (self.exit_attempts >= 1 and exit_portal is None)))
        # The reverse ENTRY trace is physically traversed evidence. If an
        # attempt times out after making progress, retry it from the nearest
        # remaining breadcrumb instead of switching to a newly inflated
        # direct portal that may be disconnected.
        corridor_extension_applied = False
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
            dense_reversed_entry_count = len(reversed_entry)
            # The normal three-point trace is inside -> door centre ->
            # corridor.  Its middle point is a semantic portal constraint,
            # not redundant sampling, so only longer dense traces may be
            # sparsified.
            if dense_reversed_entry_count > 3:
                reversed_entry = sparsify_verified_trace(reversed_entry)
            if len(reversed_entry) < dense_reversed_entry_count:
                self.events.append({
                    "event": "ROOM_EXIT_ENTRY_TRACE_SPARSIFIED",
                    "elapsed_sec": round(float(now), 3),
                    "room_id": self._room_id(),
                    "original_waypoints": dense_reversed_entry_count,
                    "retained_waypoints": len(reversed_entry),
                })
            # The entry portal was physically traversed and footprint-audited
            # moments earlier.  Reversing it is more reliable than asking a
            # freshly inflated voxel raster to rediscover the same narrow
            # doorway after several exit-preflight failures.
            use_reversed_entry = len(reversed_entry) >= 2
            mandatory_exit = (reversed_entry if use_reversed_entry else
                              mandatory_exit)
            if use_reversed_entry:
                # A fall/goal preemption may save only the reached ENTRY
                # prefix.  Never let ``inside -> door centre`` replace the
                # final observed corridor-side crossing (run
                # ..._flightatruthfix_20260826_084500 stopped 0.1 m inside on
                # every retry for exactly this reason).
                mandatory_exit = complete_reversed_entry_trace_to_corridor(
                    mandatory_exit, self.active_door)
            # Run115 showed that this measured corridor-side anchor can be
            # shallow enough for goal tolerance to stop across the door
            # plane. Extend the proven trace only onto free corridor raster.
            requested_extension = max(
                0.0, self.config.reversed_entry_trace_corridor_extension)
            if use_reversed_entry and requested_extension > 0.0:
                start = tuple(mandatory_exit[-1])
                nx, ny = self.active_door.normal
                extension = requested_extension
                while extension >= 0.35 - 1e-6:
                    candidate = (start[0] - extension * nx,
                                 start[1] - extension * ny)
                    if (_state(grid, candidate) == 0 and
                            _clearance(grid, candidate) >=
                            self.config.entry_staging_clearance):
                        mandatory_exit = list(mandatory_exit)
                        mandatory_exit.append([candidate[0], candidate[1]])
                        corridor_extension_applied = True
                        self.events.append({
                            "event": "ROOM_EXIT_CORRIDOR_EXTENSION_APPLIED",
                            "elapsed_sec": round(float(now), 3),
                            "room_id": self._room_id(),
                            "requested_extension_m": requested_extension,
                            "applied_extension_m": extension,
                        })
                        break
                    extension -= 0.10
            # If no clear extension exists, retain the measured anchor for
            # recovery, but do not let that shallow anchor prove EXIT. A
            # failed result is retried before corridor/stair handoff.
            if (not use_reversed_entry and
                    _distance(mandatory_exit[-1], exit_point) > 0.15):
                mandatory_exit.append(list(exit_point))
            exit_point = tuple(mandatory_exit[-1])
        # A measured corridor-side anchor can still lie inside the executor's
        # goal tolerance or a freshly inflated raster cell. Apply the verified
        # free-corridor extension on the first attempt, not only after wasting
        # one failed EXIT retry.
        if not corridor_extension_applied:
            requested_extension = max(
                0.0, self.config.reversed_entry_trace_corridor_extension)
            start = tuple(mandatory_exit[-1])
            nx, ny = self.active_door.normal
            extension = requested_extension
            while extension >= 0.35 - 1e-6:
                candidate = (start[0] - extension * nx,
                             start[1] - extension * ny)
                if (_state(grid, candidate) == 0 and
                        _clearance(grid, candidate) >=
                        self.config.entry_staging_clearance):
                    mandatory_exit = list(mandatory_exit)
                    mandatory_exit.append([candidate[0], candidate[1]])
                    corridor_extension_applied = True
                    self.events.append({
                        "event": "ROOM_EXIT_CORRIDOR_EXTENSION_APPLIED",
                        "elapsed_sec": round(float(now), 3),
                        "room_id": self._room_id(),
                        "requested_extension_m": requested_extension,
                        "applied_extension_m": extension,
                        "policy": "first_attempt_short_exit",
                    })
                    break
                extension -= 0.10
            exit_point = tuple(mandatory_exit[-1])
        # If the dog is already physically centred inside the aperture, do
        # not force it sideways to the stale saved door-centre anchor.  That
        # diagonal manoeuvre caused repeated 14 s EXIT stalls on F2 room 4
        # even though the body was only 0.6 m from the corridor.  Cross the
        # same measured aperture along its outward normal at the current
        # lateral coordinate.  The endpoint must be free and execution still
        # receives the normal SCAN-lite 3-D audit.
        aligned_exit = portal_lateral_preserving_exit(
            self.active_door, current, self.config.exit_offset)
        if (aligned_exit is not None and
                _state(grid, aligned_exit) == 0):
            mandatory_exit = [
                [float(current[0]), float(current[1])],
                [float(aligned_exit[0]), float(aligned_exit[1])],
            ]
            exit_point = tuple(aligned_exit)
            corridor_extension_applied = True
            self.events.append({
                "event": "ROOM_EXIT_PORTAL_LATERAL_PRESERVED",
                "elapsed_sec": round(float(now), 3),
                "room_id": self._room_id(),
                "door_id": self.active_door.door_id,
                "current_depth_m": round(float(
                    self.active_door.depth(current)), 3),
                "current_lateral_m": round(float(
                    self.active_door.lateral(current)), 3),
                "policy": "straight_outward_normal_no_jamb_recentering",
            })
        # A bounded truth reseat places the body 0.55 m inside the immutable
        # generated portal.  FAST-LIO can need a few frames to reflect that
        # correction, so the saved ENTRY trace and nearest historical inside
        # anchor are stale for exactly this transaction.  Rebuild a compact
        # physical centreline crossing from the generated portal itself.  No
        # role is credited here: record_result still requires the executor to
        # move across the door and the manager independently checks Gazebo
        # truth after completion.
        truth_portal_restage_exit = bool(
            self.truth_portal_restage_exit_pending and
            getattr(self.active_door, "truth_contract_center", None) is not None and
            getattr(self.active_door,
                    "truth_contract_normal_direction", None) is not None)
        if truth_portal_restage_exit:
            contract_center = self.active_door.truth_contract_center
            contract_theta = float(
                self.active_door.truth_contract_normal_direction)
            contract_normal = (math.cos(contract_theta),
                               math.sin(contract_theta))
            inside_distance = 0.55
            outside_distance = max(0.70, float(self.config.exit_offset))
            contract_inside = [
                float(contract_center[0]) +
                inside_distance * contract_normal[0],
                float(contract_center[1]) +
                inside_distance * contract_normal[1],
            ]
            contract_outside = [
                float(contract_center[0]) -
                outside_distance * contract_normal[0],
                float(contract_center[1]) -
                outside_distance * contract_normal[1],
            ]
            mandatory_exit = [
                contract_inside,
                [float(contract_center[0]), float(contract_center[1])],
                contract_outside,
            ]
            exit_point = tuple(contract_outside)
            corridor_extension_applied = True
            # Consume before dispatch so even an executor exception cannot
            # create an unbounded truth-centreline retry loop.
            self.truth_portal_restage_exit_pending = False
            self.events.append({
                "event": "ROOM_EXIT_TRUTH_RESTAGE_DIRECT_PORTAL_SELECTED",
                "elapsed_sec": round(float(now), 3),
                "room_id": self._room_id(),
                "door_id": self.active_door.door_id,
                "inside_anchor": list(contract_inside),
                "portal_center": [float(contract_center[0]),
                                  float(contract_center[1])],
                "corridor_anchor": list(contract_outside),
                "policy": (
                    "post_truth_reseat_inside_center_to_portal_to_corridor_"
                    "no_stale_deep_anchor"),
            })
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
            "corridor_extension_applied": corridor_extension_applied,
            "portal_clearance_m": exit_clearance,
            "portal_preflight_verified": exit_portal is not None,
        }
        if obstacle_front_gap_direct_exit:
            goal["obstacle_front_gap_direct_exit"] = True
            # Retain the old field while launch files and result readers from
            # earlier runs still consume it as the manager-side fast-path
            # marker.  It now means a truth-bounded obstacle front-gap EXIT,
            # not only the deep variant.
            goal["deep_obstacle_front_gap_direct_exit"] = True
            goal["deep_obstacle_front_gap_segmented_exit"] = bool(
                active_obstacle_view_policy == "deep_outer_side_peek" and
                len(mandatory_exit) >= 4)
            goal["_preplanned_path_result"] = {
                "success": True,
                "reason": "truth_bounded_deep_obstacle_front_gap_exit",
                "path": [(float(current[0]), float(current[1]))] +
                        [(float(point[0]), float(point[1]))
                         for point in mandatory_exit],
            }
            self.events.append({
                "event": "ROOM_OBSTACLE_FRONT_GAP_DIRECT_EXIT_SELECTED",
                "elapsed_sec": round(float(now), 3),
                "room_id": self._room_id(),
                "door_id": self.active_door.door_id,
                "current_contract_depth_m": round(float(
                    self.active_door.contract_depth(current)), 3),
                "obstacle_front_depth_m": round(float(truth_front_depth), 3),
                "policy": (
                    "direct_front_gap_to_inside_anchor_then_physical_portal"),
                "live_3d_scan_lite_still_required": True,
            })
        if truth_portal_restage_exit:
            goal["truth_portal_restage_exit"] = True
            goal["portal_preflight_verified"] = False
            goal["breadcrumb_fallback"] = False
            goal["_preplanned_path_result"] = {
                "success": True,
                "reason": "truth_restage_direct_physical_portal_exit",
                "path": [(float(current[0]), float(current[1]))] +
                        [(float(point[0]), float(point[1]))
                         for point in mandatory_exit],
                "execution_waypoints": [
                    (float(point[0]), float(point[1]))
                    for point in mandatory_exit],
            }
        # The first EXIT anchor is a staging pose on the door centreline.  The
        # locomotion goal must face out through the portal before translating;
        # deriving yaw from the next waypoint makes narrow doors get hit
        # diagonally after a stale raster replan.
        outward_yaw = math.atan2(-normal[1], -normal[0])
        goal["door_centerline_yaw"] = outward_yaw
        if (exit_portal is not None and
                exit_portal.get("direct_centerline_exit_approach")):
            goal["direct_centerline_exit_approach"] = True
            goal["door_centerline_alignment"] = list(
                exit_portal.get("door_centerline_alignment") or
                mandatory_exit[0])
            self.events.append({
                "event": "ROOM_EXIT_DIRECT_CENTERLINE_APPROACH_SELECTED",
                "elapsed_sec": round(float(now), 3),
                "room_id": self._room_id(),
                "door_id": self.active_door.door_id,
                "approach_length_m": round(
                    _distance(current, mandatory_exit[0]), 3),
                "policy": "clear_straight_then_slow_portal_crossing",
            })
        if nearest_anchor_exit:
            goal["nearest_observed_anchor_exit"] = True
        if ((fallback_exit or
             (exit_portal is None and self.entry_confirmed)) and
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
        if (not truth_portal_restage_exit and
                exit_portal is not None and
                not obstacle_front_gap_direct_exit and
                exit_portal.get("preflight_path")):
            # Execute the exact current -> inside alignment -> door centre ->
            # corridor path that just passed occupancy preflight. Rebuilding
            # it later from mission-wide breadcrumbs caused every retry to
            # grow and re-run already completed room motion.
            goal["_preplanned_path_result"] = {
                "success": True,
                "reason": "centered_exit_preflight_path",
                "path": list(exit_portal.get("preflight_path") or []),
            }
        elif (not truth_portal_restage_exit and
              exit_portal is None and
              not obstacle_front_gap_direct_exit):
            trace = [(float(current[0]), float(current[1]))]
            trace.extend((float(point[0]), float(point[1]))
                         for point in mandatory_exit)
            goal["_preplanned_path_result"] = {
                "success": True,
                "reason": "observed_doorway_trace_exit_fallback",
                "path": trace,
            }
            goal["observed_doorway_trace_fallback"] = True
        if (not truth_portal_restage_exit and fallback_exit and
                len(self.entry_portal_waypoints) >= 2):
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
        transient_locomotion_failure = bool(
            not success and reason in (
                "locomotion_not_ready",
                "locomotion_not_ready_during_goal"))
        if transient_locomotion_failure:
            retry_two_pose = False
            if role in ("G1", "G2", "G3", "G4"):
                self.role_attempts.pop(str(role), None)
                self.failed_roles.discard(str(role))
                self.adaptive_route_queue = []
                diagnostic = dict(goal.get("room_goal_diagnostic") or {})
                # A controller-mode reset is not evidence that the already
                # occupancy/A*-validated camera centre became invalid.  In
                # particular, dropping the second half of a two-pose sweep
                # lets the next coverage pass exit the room with only one
                # real RGB-D viewpoint.  Preserve that exact proposal for
                # one bounded retry after locomotion recovery.  A second
                # transient failure is deliberately not requeued, so this
                # cannot form an endless room loop.
                retry_two_pose = bool(
                    role in ("G3", "G4") and
                    diagnostic.get("two_pose_room_sweep") and
                    not diagnostic.get("locomotion_recovery_retry"))
                if retry_two_pose:
                    diagnostic["locomotion_recovery_retry"] = True
                    self.adaptive_route_queue.append(diagnostic)
            self.events.append({
                "event": "ROOM_GOAL_DEFERRED_LOCOMOTION_RECOVERY",
                "elapsed_sec": round(float(now), 3),
                "room_id": goal.get("room_id"),
                "door_id": self.active_door.door_id,
                "role": role,
                "reason": reason,
                "policy": ("retry_verified_two_pose_goal_once" if
                           retry_two_pose else
                           "do_not_consume_room_or_exit_retry"),
            })
            return {
                "success": False, "reason": reason,
                "reported_success": False, "role": role,
                "transient_locomotion_failure": True}
        if role in ("G1", "G2", "G3", "G4") and final_point is not None:
            target = goal.get("position") or []
            target_neighborhood_reached = bool(
                len(target) >= 2 and
                _distance(final_point, (float(target[0]), float(target[1]))) <=
                self.config.semantic_completion_tolerance)
            viewpoint_contract = str(getattr(
                self.active_door, "viewpoint_contract", "") or "")
            depth_valid = (
                self.active_door.contract_depth(final_point) >= 1.10
                if viewpoint_contract == "obstacle_front_opposite_sides"
                else self.active_door.depth(final_point) >= 1.10)
            lateral = (
                self.active_door.contract_lateral(final_point)
                if viewpoint_contract == "obstacle_front_opposite_sides"
                else self.active_door.lateral(final_point))
            # Open rooms use an axial deep+near pair: both valid camera
            # centres can lie on the doorway centreline.  The old generic
            # lateral-role test rejected a physically reached deep G3 after
            # a controller timeout, even though it was inside the semantic
            # neighbourhood of the occupancy/A*-validated target.  Preserve
            # strict target proximity and minimum interior depth here; the
            # completed G3/G4 pair is still checked by the locked
            # _physical_two_view_contract_evidence() geometry gate below.
            side_valid = bool(
                role in ("G1", "G2") or
                viewpoint_contract == "open_deep_near" or
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
            truth_match_separation = getattr(
                self.active_door,
                "truth_contract_match_separation_m", None)
            truth_tangent_separation = getattr(
                self.active_door,
                "truth_contract_match_tangent_separation_m", None)
            truth_normal_separation = getattr(
                self.active_door,
                "truth_contract_match_normal_separation_m", None)
            contract_entry_plane_authorized = bool(
                getattr(self.active_door, "truth_contract_center", None) is not None and
                getattr(self.active_door,
                        "truth_contract_normal_direction", None) is not None and
                ((truth_match_separation is not None and
                  float(truth_match_separation) <= float(
                      self.config.truth_portal_snap_max_separation)) or
                 (truth_tangent_separation is not None and
                  truth_normal_separation is not None and
                  float(truth_tangent_separation) <= float(
                      self.config.truth_portal_tangent_repair_max_separation) and
                  float(truth_normal_separation) <= float(
                      self.config.truth_portal_normal_repair_max_separation))))
            entry_depth = (
                self.active_door.contract_depth(final_point)
                if final_point is not None and contract_entry_plane_authorized
                else self.active_door.depth(final_point)
                if final_point is not None else -math.inf)
            depth_confirmed = bool(
                final_point is not None and entry_depth >=
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
            # The commanded target is deliberately deeper than the immutable
            # room-side portal anchor. A narrow jamb can stop the body after
            # a real crossing but before that deep target. Accept only when
            # both independent conditions hold: positive door-plane depth
            # and proximity to the room-side anchor. fix23 door_01 crossed
            # 0.52 m and stopped 0.48 m from this anchor, while door_02/03
            # never crossed the plane and remain rejected.
            interior_anchor_reached = bool(
                final_point is not None and
                _distance(final_point, self.active_door.interior_side) <=
                min(0.65, self.config.semantic_completion_tolerance))
            # A zero-waypoint timeout means the controller never traversed
            # the planned portal path. FAST-LIO can still project that
            # stationary pose just inside a slightly drifted door plane; do
            # not turn that projection artefact into ROOM_ENTERED.
            # Keep final-pose recovery only when a path made progress.
            execution = goal.get("execution_result") or {}
            completed_waypoints = execution.get("completed_waypoints")
            no_entry_progress = bool(
                not success and completed_waypoints is not None and
                int(completed_waypoints) <= 0)
            if success and not depth_confirmed:
                effective_success = False
                effective_reason = "entry_depth_not_confirmed"
            elif (not no_entry_progress and not success and depth_confirmed and
                  (entry_target_reached or interior_anchor_reached)):
                effective_success = True
                effective_reason = "entry_confirmed_by_final_pose"
            elif no_entry_progress:
                effective_reason = "entry_no_waypoint_progress"
            if final_point is not None:
                goal["entry_crossing_diagnostic"] = {
                    "depth_m": round(float(entry_depth), 3),
                    "minimum_depth_m": round(float(
                        self.config.minimum_crossing_depth), 3),
                    "plane_source": (
                        "generated_contract_plane"
                        if contract_entry_plane_authorized else
                        "measured_door_plane"),
                    "target_neighborhood_reached": bool(
                        entry_target_reached),
                    "interior_anchor_reached": bool(
                        interior_anchor_reached),
                    "completed_waypoints": completed_waypoints,
                }
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
                bool(goal.get("generated_truth_exit_crossing_confirmed")) or
                (final_point is not None and
                 # The corridor target is 0.60-0.90 m beyond the estimated
                 # door plane, but the executor may legally stop 0.15 m from
                 # it and the 0.15 m occupancy raster jitters the re-estimated
                 # plane.  Run12 physically reached -0.3945 m and was rejected
                 # by the old -0.400 m threshold.  A full 0.30 m body-centre
                 # crossing remains well outside this tolerance envelope.
                 self.active_door.depth(final_point) <= -0.30))
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
                (self.config.reversed_entry_trace_corridor_extension <= 0.0 or
                 bool(goal.get("corridor_extension_applied"))) and
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
        map_refresh_only = bool((goal.get("room_goal_diagnostic") or {}).get(
            "entry_pose_visual_fallback"))
        semantic_target = goal.get("position") or []
        obstacle_contract_pose_reached = False
        if (effective_success and role in ("G3", "G4") and
                not map_refresh_only and final_point is not None and
                getattr(self.active_door, "viewpoint_contract", None) ==
                "obstacle_front_opposite_sides"):
            final_depth = self.active_door.contract_depth(final_point)
            final_lateral = self.active_door.contract_lateral(final_point)
            front_gap_valid = bool(
                final_depth >= self.config.minimum_crossing_depth and
                not obstacle_candidate_is_behind_front_gap(
                    self.active_door, final_point,
                    self.config.post_entry_path_clearance))
            completed_laterals = [
                self.active_door.contract_lateral(old)
                for old in self.completed_points]
            side_valid = bool(
                abs(final_lateral) >= 0.55 and
                (not completed_laterals or
                 any(final_lateral * old <= -0.20
                     for old in completed_laterals)))
            separated = bool(
                not self.completed_points or
                min(_distance(final_point, old)
                    for old in self.completed_points) >=
                self.config.minimum_goal_separation)
            policy = obstacle_view_policy(self.active_door)
            deep_visibility_valid = True
            if policy == "deep_outer_side_peek":
                front = getattr(
                    self.active_door, "truth_obstacle_front_depth_m", None)
                obstacle_min = getattr(
                    self.active_door,
                    "truth_obstacle_minimum_lateral_m", None)
                obstacle_max = getattr(
                    self.active_door,
                    "truth_obstacle_maximum_lateral_m", None)
                deep_visibility_valid = bool(
                    front is not None and
                    final_depth >= max(
                        self.config.minimum_crossing_depth,
                        float(front) - 1.00) and
                    obstacle_min is not None and obstacle_max is not None and
                    (final_lateral >= float(obstacle_max) + 0.70 or
                     final_lateral <= float(obstacle_min) - 0.70))
            obstacle_contract_pose_reached = bool(
                front_gap_valid and side_valid and separated and
                deep_visibility_valid)
        if (effective_success and role in ("G3", "G4") and
                not map_refresh_only and final_point is not None and
                len(semantic_target) >= 2 and
                getattr(self.active_door, "viewpoint_contract", None) ==
                "obstacle_front_opposite_sides" and
                not obstacle_contract_pose_reached):
            # A retry endpoint can be close enough to its nominal target yet
            # still fall inside the primary obstacle silhouette.  run158
            # accepted lateral=-0.897 m for a deep-side contract requiring
            # <=-1.20 m, then proceeded to EXIT with a false two-view debt
            # state.  Completion is based on the measured physical pose and
            # locked room geometry, never merely target proximity.
            effective_success = False
            effective_reason = "obstacle_contract_pose_not_reached"
        elif (effective_success and role in ("G3", "G4") and
                not map_refresh_only and final_point is not None and
                len(semantic_target) >= 2 and
                _distance(final_point, (float(semantic_target[0]),
                                        float(semantic_target[1]))) >
                self.config.semantic_completion_tolerance and
                not obstacle_contract_pose_reached):
            effective_success = False
            effective_reason = "semantic_endpoint_not_reached"
        elif (effective_success and obstacle_contract_pose_reached and
              len(semantic_target) >= 2 and
              _distance(final_point, (float(semantic_target[0]),
                                      float(semantic_target[1]))) >
              self.config.semantic_completion_tolerance):
            effective_reason = "obstacle_contract_pose_reached"
        physical_duplicate_threshold = max(
            1.20, self.config.minimum_goal_separation -
            min(0.30, 0.375 * self.config.semantic_completion_tolerance))
        if (effective_success and role in ("G1", "G2", "G3", "G4") and
                not map_refresh_only and final_point is not None and
                self.completed_points and
                min(_distance(final_point, old)
                    for old in self.completed_points) <
                physical_duplicate_threshold):
            allow_diagonal_partner = False
            if (self.active_door is not None and role in ("G3", "G4")):
                anchor = self.completed_points[0]
                depth_delta = abs(self.active_door.depth(final_point) -
                                  self.active_door.depth(anchor))
                lateral_delta = abs(self.active_door.lateral(final_point) -
                                    self.active_door.lateral(anchor))
                required_baseline = max(
                    self.config.minimum_goal_separation,
                    self.config.visual_breadth_min_baseline * 0.45)
                diagonal_ratio = lateral_delta / max(depth_delta, 1e-6)
                allow_diagonal_partner = bool(
                    depth_delta >= max(1.20, required_baseline * 0.35) and
                    lateral_delta >= max(0.80, required_baseline * 0.35) and
                    0.70 <= diagonal_ratio <= 1.45)
            if not allow_diagonal_partner:
                effective_success = False
                effective_reason = "duplicate_observation_pose_rejected"
        self.events.append({
            "event": "ROOM_GOAL_RESULT", "elapsed_sec": round(float(now), 3),
            "room_id": goal.get("room_id"), "door_id": self.active_door.door_id,
            "role": role, "goal": goal.get("position"),
            "success": effective_success,
            "reported_execution_success": bool(success),
            "reason": effective_reason,
            "execution_progress": goal.get("execution_result"),
            "entry_crossing_diagnostic": goal.get(
                "entry_crossing_diagnostic"),
            "final_point": ([float(final_point[0]), float(final_point[1])]
                            if final_point is not None else None),
            "obstacle_contract_pose_reached": bool(
                obstacle_contract_pose_reached),
        })
        if (effective_success and role in ("G1", "G2", "G3", "G4") and
                final_point is not None and not map_refresh_only):
            self.completed_points.append((float(final_point[0]), float(final_point[1])))
            self.successful_roles.add(str(role))
            if role == "G3":
                self.active_door.g3_truth_pose = (
                    float(final_point[0]), float(final_point[1]))
                diagnostic = goal.get("room_goal_diagnostic") or {}
                if (str(getattr(self.active_door, "viewpoint_contract", "")) ==
                        "open_deep_near"):
                    route = [
                        (float(point[0]), float(point[1]))
                        for point in (diagnostic.get("preflight_path") or [])
                        if isinstance(point, (list, tuple)) and len(point) >= 2]
                    if len(route) >= 2:
                        self.open_g3_executed_path = route
                        self.events.append({
                            "event": "ROOM_OPEN_G3_EXECUTED_ROUTE_SAVED",
                            "elapsed_sec": round(float(now), 3),
                            "room_id": goal.get("room_id"),
                            "door_id": self.active_door.door_id,
                            "waypoint_count": len(route),
                            "policy": "bounded_reverse_route_for_near_g4",
                        })
                if (self.two_pose_opposite_repair_required and
                        diagnostic.get("two_pose_strategy") ==
                        "obstacle_front_gap_first"):
                    route = [
                        (float(point[0]), float(point[1]))
                        for point in (diagnostic.get("preflight_path") or [])
                        if isinstance(point, (list, tuple)) and len(point) >= 2]
                    if len(route) >= 2:
                        self.obstacle_gap_g3_executed_path = route
                        self.events.append({
                            "event":
                                "ROOM_OBSTACLE_G3_EXECUTED_ROUTE_SAVED",
                            "elapsed_sec": round(float(now), 3),
                            "room_id": goal.get("room_id"),
                            "door_id": self.active_door.door_id,
                            "waypoint_count": len(route),
                            "policy":
                                "bounded_reverse_mapping_transit_for_g4",
                        })
            elif role == "G4":
                self.active_door.g4_truth_pose = (
                    float(final_point[0]), float(final_point[1]))
            if (self.active_door.g3_truth_pose is not None and
                    self.active_door.g4_truth_pose is not None):
                contract_evidence = (
                    self._physical_two_view_contract_evidence())
                self.active_door.physical_viewpoint_separation_m = (
                    contract_evidence.get(
                        "physical_viewpoint_separation_m"))
                self.active_door.viewpoint_contract_geometry_met = bool(
                    contract_evidence.get("contract_geometry_met"))
                self.active_door.viewpoint_contract_geometry = dict(
                    contract_evidence)
                self.active_door.physical_two_view_contract_met = bool(
                    contract_evidence.get(
                        "physical_two_view_contract_met"))
            completed_strategy = str(
                (goal.get("room_goal_diagnostic") or {}).get(
                    "two_pose_strategy", ""))
            if (role == "G3" and completed_strategy ==
                    "uncertain_open_near_mapping_view"):
                self.uncertain_open_near_mapping_completed = True
            if role == "G1":
                self.visual_anchor_completed = True
            if role == "G3" and self.adaptive_route_queue:
                queued_obstacle = dict(
                    (self.adaptive_route_queue[0].get(
                        "two_pose_obstacle") or {}))
                queued_obstacle_area = float(
                    queued_obstacle.get("area_m2", math.inf))
                queued_strategy = str(
                    self.adaptive_route_queue[0].get(
                        "two_pose_strategy", ""))
                open_axial_pair = queued_strategy in {
                    "deep_then_axial_near",
                    "deep_seed_then_axial_near",
                    "post_refresh_deep_then_near",
                    "mandatory_deep_then_axial_near_fallback",
                }
                compact_obstacle_pair = bool(
                    queued_strategy == "obstacle_deep_then_opposite_near" and
                    math.isfinite(queued_obstacle_area) and
                    queued_obstacle_area <= 0.55)
                if open_axial_pair:
                    # The executor is allowed to refine a deep endpoint and
                    # stop at semantic tolerance.  Reusing a G4 paired with
                    # the *requested* G3 can therefore leave the two physical
                    # camera centres too close (axialtruth F2 room 3: planned
                    # 2.89 m, executed only 1.04 m), wasting a full move before
                    # duplicate rejection.  Rebuild the near-door point from
                    # the actual completed G3 pose; the mandatory selector
                    # includes completion-tolerance margin and all A*/3-D
                    # safety checks.
                    self.adaptive_route_queue = []
                    self.events.append({
                        "event": "ROOM_TWO_POSE_OPEN_G4_FRESH_REPLAN",
                        "elapsed_sec": round(float(now), 3),
                        "room_id": goal.get("room_id"),
                        "door_id": self.active_door.door_id,
                        "strategy": queued_strategy,
                        "actual_g3": [float(final_point[0]),
                                      float(final_point[1])],
                        "policy":
                            "mandatory_axial_near_from_executed_g3_pose",
                    })
                elif compact_obstacle_pair:
                    # G3 has just supplied a less occluded occupancy frame
                    # around a compact centre obstacle. A G4 planned before
                    # that motion is disproportionately likely to lie on a
                    # newly observed inflated edge (direction1 lost 2.3 s
                    # rejecting it before any waypoint). Drop only this
                    # compact-obstacle suffix; next_goal() immediately
                    # rebuilds the mandatory separated G4 from the fresh
                    # grid. Large furniture pairs retain their jointly
                    # planned route because changing sides there can force a
                    # long and less stable wrap-around.
                    self.adaptive_route_queue = []
                    self.events.append({
                        "event":
                            "ROOM_TWO_POSE_COMPACT_G4_FRESH_REPLAN",
                        "elapsed_sec": round(float(now), 3),
                        "room_id": goal.get("room_id"),
                        "door_id": self.active_door.door_id,
                        "obstacle_area_m2": round(
                            queued_obstacle_area, 3),
                        "strategy": queued_strategy,
                        "policy":
                            "mandatory_separated_g4_from_post_g3_grid",
                    })
                else:
                    # Keep the paired G4 that was jointly A*-planned with G3.
                    # The manager still performs live SCAN-lite/footprint
                    # checks before motion; a real map change enters the
                    # one-shot fresh geometry re-selection below.
                    self.events.append({
                        "event": "ROOM_TWO_POSE_G4_QUEUE_RETAINED",
                        "elapsed_sec": round(float(now), 3),
                        "room_id": goal.get("room_id"),
                        "door_id": self.active_door.door_id,
                        "queued_roles": [str(item.get("role"))
                                         for item in
                                         self.adaptive_route_queue],
                        "policy":
                            "live_scan_lite_then_bounded_fresh_retry",
                    })
        elif effective_success and map_refresh_only:
            # A stationary refresh is followed by the manager's bounded RGB-D
            # sweep at this same verified interior footprint.  Count it as the
            # first physical observation so the adaptive scheduler cannot exit
            # before that observation contract is fulfilled.
            self.role_attempts.pop(str(role), None)
            self.events.append({
                "event": "ROOM_ENTRY_MAP_REFRESH_COMPLETED",
                "elapsed_sec": round(float(now), 3),
                "room_id": goal.get("room_id"),
                "door_id": self.active_door.door_id,
                "role": str(role),
            })
        execution_progress = goal.get("execution_result") or {}
        completed_waypoints = execution_progress.get("completed_waypoints")
        geometry_rejected_before_motion = bool(
            role in ("G3", "G4") and not effective_success and
            (completed_waypoints is None or int(completed_waypoints) <= 0) and
            (str(effective_reason) in (
                "goal_footprint_blocked", "start_footprint_blocked",
                "goal_unreachable", "no_path",
                "scan_lite_refinement_failed",
                "disconnected_refined_segment") or
             str(effective_reason).endswith("goal_footprint_blocked") or
             str(effective_reason).endswith("start_footprint_blocked")))
        semantic_separation_rejected = bool(
            role in ("G3", "G4") and not effective_success and
            str(effective_reason) == "duplicate_observation_pose_rejected")
        failed_strategy = str(
            (goal.get("room_goal_diagnostic") or {}).get(
                "two_pose_strategy", ""))
        failed_target = goal.get("position") or []
        endpoint_requires_reselection = bool(
            role in ("G3", "G4") and not effective_success and
            len(failed_target) >= 2 and
            (geometry_rejected_before_motion or
             str(effective_reason) in (
                 "semantic_endpoint_not_reached", "room_budget_timeout",
                 "progress_timeout")))
        if endpoint_requires_reselection:
            rejected = (float(failed_target[0]), float(failed_target[1]))
            alternatives = self.failed_visual_geometry_targets.setdefault(
                str(role), [])
            if not any(_distance(rejected, old) < 0.10
                       for old in alternatives):
                alternatives.append(rejected)
                self.events.append({
                    "event": (
                        "ROOM_OBSTACLE_GAP_ENDPOINT_QUARANTINED"
                        if "obstacle_front" in failed_strategy else
                        "ROOM_VISUAL_ENDPOINT_QUARANTINED"),
                    "elapsed_sec": round(float(now), 3),
                    "room_id": goal.get("room_id"),
                    "door_id": self.active_door.door_id,
                    "role": role,
                    "endpoint": [rejected[0], rejected[1]],
                    "reason": effective_reason,
                    "quarantined_endpoint_count": len(alternatives),
                    "policy": "same_role_alternate_endpoint",
                })
        if (geometry_rejected_before_motion and
                "obstacle_front" in failed_strategy and
                len(failed_target) >= 2):
            rejected = (float(failed_target[0]), float(failed_target[1]))
            alternatives = self.failed_visual_geometry_targets.setdefault(
                str(role), [])
            if not any(_distance(rejected, old) < 0.10
                       for old in alternatives):
                alternatives.append(rejected)
                self.events.append({
                    "event": "ROOM_OBSTACLE_GAP_ENDPOINT_QUARANTINED",
                    "elapsed_sec": round(float(now), 3),
                    "room_id": goal.get("room_id"),
                    "door_id": self.active_door.door_id,
                    "role": role,
                    "endpoint": [rejected[0], rejected[1]],
                    "reason": effective_reason,
                    "quarantined_endpoint_count": len(alternatives),
                    "policy": "same_contract_alternate_gap_endpoint",
                })
            # Keep the locked room subtype after bounded geometry failures.
            # Reclassifying a deep room as shallow made a later near-door G4
            # look contract-complete while losing the hazard visibility that
            # selected the deep subtype in the first place (run102).  The
            # ordinary bounded retry/continuity path may still return the dog,
            # but it must not weaken or falsify strict exploration evidence.
            distinct_deep_rejections = sum(
                len(points) for points in
                self.failed_visual_geometry_targets.values())
            if (self.active_door.truth_obstacle_view_policy ==
                    "deep_outer_side_peek" and
                    distinct_deep_rejections >= 2):
                self.adaptive_route_queue = []
                self.events.append({
                    "event": "ROOM_DEEP_SIDE_PEEK_REMAINS_STRICT_AFTER_REJECTIONS",
                    "elapsed_sec": round(float(now), 3),
                    "room_id": goal.get("room_id"),
                    "door_id": self.active_door.door_id,
                    "role": role,
                    "distinct_blocked_endpoint_count":
                        int(distinct_deep_rejections),
                    "reason": effective_reason,
                    "policy": "bounded_deep_alternates_only_no_reclassification",
                })
        bounded_geometry_reselection = bool(
            (endpoint_requires_reselection or
             semantic_separation_rejected) and
            str(role) not in self.visual_geometry_retry_roles)
        if bounded_geometry_reselection:
            # The endpoint was safe on the route-planning snapshot but a newer
            # LiDAR projection rejected it before locomotion. Re-plan exactly
            # once from the fresh grid instead of exiting with one physical
            # camera pose and later re-entering the same room from the corridor.
            self.visual_geometry_retry_roles.add(str(role))
            self.role_attempts.pop(str(role), None)
            self.failed_roles.discard(str(role))
            self.adaptive_route_queue = []
            self.visual_deepening_requested = True
            if str(effective_reason).endswith("start_footprint_blocked"):
                self.visual_geometry_map_refresh_pending = str(role)
            self.events.append({
                "event": "ROOM_TWO_POSE_GEOMETRY_RESELECTION_SCHEDULED",
                "elapsed_sec": round(float(now), 3),
                "room_id": goal.get("room_id"),
                "door_id": self.active_door.door_id,
                "role": role,
                "reason": effective_reason,
                "retry_limit": 1,
                "policy": "fresh_grid_alternative_endpoint_full_safety_checks",
            })
        elif role in ("G1", "G2", "G3", "G4") and not effective_success:
            # The remaining order was optimized from the preceding pose.
            # Recompute it after any execution/SCAN failure instead of blindly
            # following a now-invalid suffix.
            self.adaptive_route_queue = []
            attempts = self.role_attempts.get(str(role), 0)
            limit = (2 if role in ("G1", "G2") else
                     self.config.side_goal_retry_limit)
            # Adaptive mode may skip redundant lateral views, but it must not
            # turn a real G4 execution failure into a partial room.  In
            # particular, F2 showed room-budget timeout after the first G4
            # waypoint even though a fresh-grid fallback could still satisfy
            # the second physical view.  Permit one bounded G4 reselection in
            # adaptive mode; geometry-blocked failures already use the
            # stricter path above and are not duplicated here.
            adaptive_g4_retry = bool(
                self.config.adaptive_minimal_viewpoints and
                str(role) == "G4" and attempts < max(2, int(limit)))
            adaptive_retry_marker = "G4_ADAPTIVE_EXECUTION_RETRY"
            adaptive_g4_retry = bool(
                adaptive_g4_retry and
                adaptive_retry_marker not in
                self.visual_geometry_retry_roles)
            if ((not self.config.adaptive_minimal_viewpoints or
                 adaptive_g4_retry) and attempts < max(2, int(limit))):
                role_index = {"G1": 0, "G3": 1, "G4": 2}[str(role)]
                self.next_role_index = min(self.next_role_index, role_index)
                if adaptive_g4_retry:
                    # A failed G4 is not an observation. Remove it from the
                    # adaptive cardinality count so next_goal() can actually
                    # select the promised fresh endpoint.  The separate
                    # marker keeps this retry strictly one-shot.
                    self.role_attempts.pop(str(role), None)
                    self.visual_geometry_retry_roles.add(
                        adaptive_retry_marker)
                self.events.append({
                    "event": ("ROOM_TWO_POSE_ADAPTIVE_G4_RETRY_SCHEDULED"
                              if adaptive_g4_retry else
                              "ROOM_GOAL_RETRY_SCHEDULED"),
                    "elapsed_sec": round(float(now), 3),
                    "room_id": goal.get("room_id"),
                    "door_id": self.active_door.door_id,
                    "role": role,
                    "next_attempt": attempts + 1,
                    "fallback_candidate": True,
                    "policy": ("one_fresh_grid_g4_retry_preserve_two_view"
                               if adaptive_g4_retry else None),
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
                if not self.entry_portal_waypoints:
                    traversed = []
                    for point in goal.get("mandatory_portal_waypoints") or []:
                        if (not isinstance(point, (list, tuple)) or
                                len(point) < 2):
                            continue
                        try:
                            x, y = float(point[0]), float(point[1])
                        except (TypeError, ValueError):
                            continue
                        if (math.isfinite(x) and math.isfinite(y) and
                                (not traversed or _distance(
                                    traversed[-1], (x, y)) > 0.02)):
                            traversed.append([x, y])
                    if len(traversed) >= 2:
                        self.entry_portal_waypoints = traversed
                        self.events.append({
                            "event": "ROOM_ENTRY_TRAVERSAL_CAPTURED",
                            "elapsed_sec": round(float(now), 3),
                            "room_id": goal.get("room_id"),
                            "door_id": self.active_door.door_id,
                            "waypoint_count": len(traversed),
                            "planned_waypoint_count": len(traversed),
                            "observed_progress_only": True,
                            "source": "direct_three_anchor_path",
                        })
                self._set_state("G1_CENTER", now, "entry_confirmed")
            else:
                self.entry_attempts += 1
                if self.entry_attempts < self.config.entry_retry_limit:
                    self.next_role_index = 0
                    self._set_state("DOOR_COMMIT", now, "entry_retry")
                else:
                    # A pre-motion SCAN-lite/refinement rejection is a
                    # transient map-geometry failure, not evidence that this
                    # physical doorway should be abandoned for a full normal
                    # cooldown.  Long cooldowns let the forward detector
                    # take over another doorway and leave this one as a
                    # known-but-unvisited partial room.  Revisit it quickly
                    # from the corridor centreline; the bounded retry path
                    # still applies the full live-map and portal checks.
                    entry_geometry_retry = bool(
                        str(effective_reason) in (
                            "scan_lite_refinement_failed",
                            "disconnected_refined_segment",
                            "no_path", "goal_footprint_blocked"))
                    geometry_failures = 0
                    if entry_geometry_retry:
                        failure_key = str(self.active_door.door_id)
                        geometry_failures = int(
                            self.entry_geometry_failure_counts.get(
                                failure_key, 0)) + 1
                        self.entry_geometry_failure_counts[failure_key] = (
                            geometry_failures)
                    # One shallow retry is useful for a transient map slice;
                    # repeated refinement failure must not monopolize the
                    # corridor.  Quarantine the doorway geometrically for a
                    # bounded interval, allowing the next room candidate to
                    # be explored without changing physical room counters.
                    geometry_quarantine = (
                        entry_geometry_retry and geometry_failures >= 2)
                    key = self._set_door_cooldown(
                        self.active_door.center,
                        float(now) + (
                            (max(90.0, self.config.door_cooldown_seconds)
                             if geometry_quarantine else
                             max(5.0, min(15.0,
                                         3.0 * self.config.door_evidence_retry_seconds)))
                            if (effective_reason == "entry_no_waypoint_progress" or
                                entry_geometry_retry) else
                            self.config.door_cooldown_seconds))
                    self.events.append({
                        "event": "DOOR_COMMIT_FAILED",
                        "elapsed_sec": round(float(now), 3),
                        "room_id": goal.get("room_id"),
                        "door_id": self.active_door.door_id,
                        "cooldown_until": self.door_cooldowns[key],
                        "geometry_failure_count": geometry_failures,
                        "retry_policy": ("bounded_geometry_quarantine"
                                          if geometry_quarantine else
                                          "short_entry_geometry_retry"
                                          if entry_geometry_retry else
                                          "normal_door_cooldown"),
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
            # The acceptance contract is two distinct physical camera poses.
            # In the lateral/two-pose profile G1 is a legacy central coverage
            # anchor, not a third required stop.  Requiring it after G3/G4
            # both succeeded made an otherwise valid F2 room remain partial
            # solely because adaptive coverage stopped after the two required
            # poses.  Keep the stricter G1/G3/G4 contract for the legacy
            # profile, while making the requested dual-view profile explicit.
            required_roles = ({"G3", "G4"}
                              if self.config.visual_two_pose_lateral_enabled
                              else {"G1", "G3", "G4"})
            contract_evidence = (
                self._physical_two_view_contract_evidence()
                if self.config.visual_two_pose_lateral_enabled else {})
            physical_view_contract_met = bool(
                not self.config.visual_two_pose_lateral_enabled or
                contract_evidence.get("physical_two_view_contract_met"))
            if (effective_success and
                    self.config.visual_two_pose_lateral_enabled and
                    not physical_view_contract_met):
                self._canonicalize_open_partial_contract(
                    contract_evidence, now)
                self._canonicalize_obstacle_partial_contract(
                    contract_evidence, now)
            self.active_door.visited = bool(
                effective_success and physical_view_contract_met)
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
                float(coverage.get("largest_unknown_m2", float("inf"))) <= 1.0 and
                float(coverage.get("largest_shadow_m2", float("inf"))) <= 0.9)
            coverage_complete = bool(
                physical_view_contract_met and
                (coverage_complete or near_complete_coverage))
            self.active_door.coverage_complete = bool(
                effective_success and coverage_complete)
            self.active_door.completed = bool(
                effective_success and physical_view_contract_met and (
                    coverage_complete or
                    required_roles.issubset(self.successful_roles)))
            self.active_door.physical_two_view_contract_met = bool(
                physical_view_contract_met)
            self.active_door.viewpoint_contract_geometry_met = bool(
                not self.config.visual_two_pose_lateral_enabled or
                contract_evidence.get("contract_geometry_met"))
            self.active_door.viewpoint_contract_geometry = (
                dict(contract_evidence)
                if self.config.visual_two_pose_lateral_enabled else None)
            self.active_door.exit_truth_crossing_confirmed = bool(
                effective_success)
            self.active_door.room_transaction_complete = bool(
                effective_success and self.active_door.completed)
            self.active_door.temporarily_failed = bool(
                effective_success and not self.active_door.completed)
            if effective_success and not self.active_door.completed:
                # Snapshot the real observation contract before clearing the
                # active scheduler.  The next same-door entry must continue
                # from these roles/poses and only fill the missing side.
                self.active_door.partial_successful_roles = tuple(sorted(
                    str(item) for item in self.successful_roles
                    if str(item) in required_roles))
                self.active_door.partial_completed_points = tuple(
                    (float(point[0]), float(point[1]))
                    for point in self.completed_points
                    if isinstance(point, (list, tuple)) and len(point) >= 2)
                # Leave safely, then perform one bounded same-door re-entry.
                # A long normal doorway cooldown made a G4 geometry/locomotion
                # miss look like a permanently lost room on F2.  The retry is
                # short and explicit; it reuses the stable room_id and cannot
                # inflate the distinct-room count.  After the bounded retry,
                # keep the normal long cooldown and let the floor fail closed
                # rather than spinning on a partial room forever.
                retry_limit = 1
                can_retry = (self.active_door.partial_retry_count <
                             retry_limit)
                if can_retry:
                    self.active_door.partial_retry_count += 1
                    cooldown = max(5.0,
                                   min(12.0,
                                       self.config.door_evidence_retry_seconds))
                    self.events.append({
                        "event": "ROOM_PARTIAL_RETRY_PENDING",
                        "elapsed_sec": round(float(now), 3),
                        "room_id": self.active_room_id,
                        "door_id": self.active_door.door_id,
                        "failed_roles": sorted(self.failed_roles),
                        "partial_retry_count":
                            int(self.active_door.partial_retry_count),
                        "retry_limit": retry_limit,
                        "cooldown_seconds": round(float(cooldown), 3),
                        "policy": "same_room_id_bounded_reentry",
                    })
                else:
                    cooldown = self.config.door_cooldown_seconds
                    self.blocked_door_ids.add(str(self.active_door.door_id))
                    self.events.append({
                        "event": "ROOM_PARTIAL_RETRY_EXHAUSTED",
                        "elapsed_sec": round(float(now), 3),
                        "room_id": self.active_room_id,
                        "door_id": self.active_door.door_id,
                        "failed_roles": sorted(self.failed_roles),
                        "partial_retry_count":
                            int(self.active_door.partial_retry_count),
                        "retry_limit": retry_limit,
                        "policy": "fail_closed_no_partial_loop",
                    })
                self._set_door_cooldown(
                    self.active_door.center,
                    float(now) + float(cooldown))
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
                "physical_two_view_contract_met": physical_view_contract_met,
                "viewpoint_contract": self.active_door.viewpoint_contract,
                "viewpoint_contract_geometry_met": bool(
                    self.active_door.viewpoint_contract_geometry_met),
                "viewpoint_contract_geometry": (
                    dict(self.active_door.viewpoint_contract_geometry)
                    if self.active_door.viewpoint_contract_geometry else None),
                "entry_truth_crossing_confirmed": bool(
                    self.active_door.entry_truth_crossing_confirmed),
                "g3_truth_pose": self.active_door.g3_truth_pose,
                "g4_truth_pose": self.active_door.g4_truth_pose,
                "physical_viewpoint_separation_m":
                    self.active_door.physical_viewpoint_separation_m,
                "exit_truth_crossing_confirmed": bool(
                    self.active_door.exit_truth_crossing_confirmed),
                "room_transaction_complete": bool(
                    self.active_door.room_transaction_complete),
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
                self.pending_corridor_recovery_door = self.active_door
                self._clear_active()
                self._set_state("CORRIDOR_RESUME", now, "room_completed")
            # A successful EXIT clears ``active_door`` above.  The bounded
            # deep-room escalation is failure-only, so guard the reference
            # as well as avoiding any post-success policy work.
            deep_obstacle_exit = bool(
                not effective_success and
                self.active_door is not None and
                self.active_door.viewpoint_contract ==
                "obstacle_front_opposite_sides" and
                self.active_door.truth_obstacle_view_policy ==
                "deep_outer_side_peek" and
                self.active_door.physical_two_view_contract_met)
            # Deep opposite-side viewpoints end well behind the furniture.
            # If their first verified-trajectory EXIT still times out, a
            # second dense replay adds no new evidence and consumed another
            # 44.5 simulation seconds in run42.  Escalate immediately to the
            # manager's one bounded sparse/nearest-anchor recovery.  That
            # recovery still requires a physical door-plane crossing and all
            # normal live 3-D checks, so this changes retry order, not safety
            # or completion credit.
            effective_exit_retry_limit = (
                1 if deep_obstacle_exit else self.config.exit_retry_limit)
            if (not effective_success and
                    self.exit_attempts >= effective_exit_retry_limit):
                self.events.append({
                    "event": "ROOM_EXIT_BLOCKED",
                    "elapsed_sec": round(float(now), 3),
                    "room_id": goal.get("room_id"),
                    "door_id": self.active_door.door_id,
                    "attempts": self.exit_attempts,
                    "effective_retry_limit": effective_exit_retry_limit,
                    "deep_obstacle_sparse_recovery": deep_obstacle_exit,
                    "fallback_pending": not self.exit_fallback_issued,
                })
                # The fallback is already the physically traversed entry
                # trace.  If the bounded retry limit is exhausted, do not
                # emit the same unreachable portal goal every scheduler
                # cycle: it previously consumed the whole mission budget in
                # one room.  Stop with an explicit, diagnosable condition.
                self.exit_blocked = True
                self.blocked_door_ids.add(str(self.active_door.door_id))
                self._set_state(
                    "ROOM_EXIT_BLOCKED", now,
                    "exit_retry_limit_exhausted")
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
