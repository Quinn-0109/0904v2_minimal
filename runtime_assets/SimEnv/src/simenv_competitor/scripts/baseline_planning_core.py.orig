#!/usr/bin/env python3
"""ROS-free, room-agnostic 2-D occupancy-grid planning primitives.

The A* implementation is the generic observed-free planner extracted from the
structured exploration stack.  Keeping it here prevents the baseline runtime
from importing any door, portal, room, or structured state-machine code.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


Point = Tuple[float, float]


def corridor_local_door_context_allowed(
        raw_is_corridor: bool, entry_forward_locked: bool,
        locked_axis_progress: float, locked_axis_lateral_error: float,
        minimum_progress: float = 2.0,
        maximum_lateral_error: float = 0.85) -> bool:
    """Allow strong doorway evidence after a proven corridor-entry advance.

    A side doorway temporarily removes one of the wall returns used by the
    strict corridor-width classifier.  The strict result remains sufficient,
    while its fallback is deliberately narrow: the entry-forward phase must
    already be locked, and the robot must have continued forward on that
    locked line.  Consequently an opening in the initial lobby cannot create
    the corridor/room phase.
    """
    if raw_is_corridor:
        return True
    try:
        progress = float(locked_axis_progress)
        lateral = float(locked_axis_lateral_error)
        required = float(minimum_progress)
        tolerance = float(maximum_lateral_error)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(
        entry_forward_locked and
        all(math.isfinite(value) for value in
            (progress, lateral, required, tolerance)) and
        required > 0.0 and tolerance > 0.0 and
        progress >= required and lateral <= tolerance)


def corridor_line_lateral_error(point: Sequence[float],
                                origin: Sequence[float],
                                axis: Sequence[float]) -> float:
    """Return the perpendicular distance from ``point`` to a 2-D line."""
    direction = np.asarray(axis[:2], dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9 or not math.isfinite(norm):
        return math.inf
    direction /= norm
    offset = (np.asarray(point[:2], dtype=float) -
              np.asarray(origin[:2], dtype=float))
    return abs(float(direction[0] * offset[1] -
                     direction[1] * offset[0]))


def room_exit_endpoint_tolerance_allowed(
        final_point: Sequence[float], commanded_point: Sequence[float],
        door_depth: float, commanded_in_corridor_band: bool,
        endpoint_tolerance: float = 0.30,
        maximum_corridor_side_depth: float = -0.30) -> bool:
    """Accept executor tolerance around a corridor-valid EXIT endpoint.

    A small angular error in a long-lived corridor station axis grows into a
    large lateral error at far rooms. The commanded endpoint can still be
    inside the established corridor band while the executor legally stops a
    few centimetres beyond that band. This helper keeps the exception narrow:
    the target itself must be in-band, the final pose must have crossed the
    active door to its corridor side, and it must be close to that target.
    """
    if (not commanded_in_corridor_band or len(final_point) < 2 or
            len(commanded_point) < 2):
        return False
    try:
        final = (float(final_point[0]), float(final_point[1]))
        target = (float(commanded_point[0]), float(commanded_point[1]))
        depth = float(door_depth)
        tolerance = float(endpoint_tolerance)
        depth_limit = float(maximum_corridor_side_depth)
    except (TypeError, ValueError, OverflowError):
        return False
    if (not all(math.isfinite(value) for value in
                final + target + (depth, tolerance, depth_limit)) or
            tolerance <= 0.0 or depth_limit > 0.0):
        return False
    return bool(
        depth <= depth_limit and
        math.hypot(final[0] - target[0],
                   final[1] - target[1]) <= tolerance)


def rebase_corridor_line_laterally(origin: Sequence[float],
                                   axis: Sequence[float],
                                   observed_center: Sequence[float]) -> Point:
    """Move a corridor line sideways without changing station coordinates.

    The station axis and its longitudinal origin remain immutable.  Only the
    component perpendicular to the axis is corrected, so remembered doorway
    stations keep exactly the same coordinate while future transit goals use
    the newly observed physical centreline.
    """
    direction = np.asarray(axis[:2], dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9 or not math.isfinite(norm):
        raise ValueError("corridor axis must be finite and non-zero")
    direction /= norm
    base = np.asarray(origin[:2], dtype=float)
    observed = np.asarray(observed_center[:2], dtype=float)
    delta = observed - base
    lateral = delta - float(np.dot(delta, direction)) * direction
    rebased = base + lateral
    return float(rebased[0]), float(rebased[1])


def corridor_live_start_override_allowed(
        current: Sequence[float], trajectory: Sequence[Sequence[float]],
        corridor_anchor: Optional[Sequence[float]],
        corridor_axis: Optional[Sequence[float]], registration_healthy: bool,
        maximum_lateral_error: float = 0.85) -> bool:
    """Validate a tightly scoped SCAN-lite override at the live pose only.

    A rolling voxel map can contain a return inside the robot's footprint
    even though the robot is physically standing there.  The override is
    allowed only on an established corridor centre band, with healthy
    registration and a continuous recent odometry trace.  It says nothing
    about the path ahead; every later sample keeps the normal 3-D check.
    """
    if (not registration_healthy or corridor_anchor is None or
            corridor_axis is None or len(current) < 2 or len(trajectory) < 2):
        return False
    try:
        point = (float(current[0]), float(current[1]))
        anchor = (float(corridor_anchor[0]), float(corridor_anchor[1]))
        axis = (float(corridor_axis[0]), float(corridor_axis[1]))
        norm = math.hypot(axis[0], axis[1])
        if norm <= 1e-6 or not all(math.isfinite(value) for value in
                                   point + anchor + axis):
            return False
        axis = (axis[0] / norm, axis[1] / norm)
        recent = [(float(sample[1]), float(sample[2]))
                  for sample in trajectory[-20:] if len(sample) >= 3]
    except (TypeError, ValueError, IndexError, OverflowError):
        return False
    if len(recent) < 2 or math.hypot(
            recent[-1][0] - point[0], recent[-1][1] - point[1]) > 0.12:
        return False
    if any(not all(math.isfinite(value) for value in sample)
           for sample in recent):
        return False
    if any(math.hypot(b[0] - a[0], b[1] - a[1]) > 0.75
           for a, b in zip(recent[:-1], recent[1:])):
        return False
    offset = (point[0] - anchor[0], point[1] - anchor[1])
    lateral = abs(axis[0] * offset[1] - axis[1] * offset[0])
    return lateral <= float(maximum_lateral_error)


def orient_axis_away_from_origin(axis: Sequence[float],
                                 current: Sequence[float],
                                 origin: Sequence[float],
                                 minimum_displacement: float = 1.0
                                 ) -> Tuple[float, float]:
    """Give an unoriented corridor axis the mission-forward sign.

    PCA and symmetric wall geometry identify an axis but cannot identify its
    sign.  The displacement from the mission start is online odometry evidence
    and prevents the first lateral/backward frontier from flipping an already
    reached corridor toward the explored lobby.
    """
    direction = np.asarray(axis[:2], dtype=float)
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm < 1e-9:
        return 1.0, 0.0
    direction /= norm
    displacement = (np.asarray(current[:2], dtype=float) -
                    np.asarray(origin[:2], dtype=float))
    if (float(np.linalg.norm(displacement)) >=
            max(0.0, float(minimum_displacement)) and
            float(np.dot(direction, displacement)) < 0.0):
        direction = -direction
    return float(direction[0]), float(direction[1])


def corridor_axis_motion_consistency(axis: Sequence[float],
                                     observed_motion: Sequence[float],
                                     maximum_deviation_rad: float,
                                     minimum_motion: float = 2.0) -> dict:
    """Check a semantic corridor axis against online travelled motion.

    Parallel-wall geometry is symmetric and can be corrupted by a sparse
    rolling raster near an open doorway.  Once the robot has travelled a few
    metres from the floor mission origin, that displacement provides an
    independent, layout-free direction check.  Short motion deliberately
    remains inconclusive so ordinary initial corridor acquisition is not
    suppressed.
    """
    try:
        direction = np.asarray(axis[:2], dtype=float)
        motion = np.asarray(observed_motion[:2], dtype=float)
        axis_norm = float(np.linalg.norm(direction))
        motion_norm = float(np.linalg.norm(motion))
        limit = float(maximum_deviation_rad)
        required_motion = max(0.0, float(minimum_motion))
    except (TypeError, ValueError, IndexError, OverflowError):
        return {"consistent": False, "reason": "invalid_axis_or_motion"}
    if (not all(math.isfinite(value) for value in
                (axis_norm, motion_norm, limit, required_motion)) or
            axis_norm < 1e-9):
        return {"consistent": False, "reason": "invalid_axis_or_motion"}
    if motion_norm < required_motion:
        return {
            "consistent": True,
            "reason": "insufficient_motion_for_gate",
            "motion_distance_m": motion_norm,
            "deviation_rad": None,
        }
    direction /= axis_norm
    motion /= max(1e-9, motion_norm)
    alignment = float(np.clip(np.dot(direction, motion), -1.0, 1.0))
    deviation = math.acos(alignment)
    return {
        "consistent": bool(deviation <= max(0.0, limit)),
        "reason": ("axis_matches_observed_motion" if
                   deviation <= max(0.0, limit) else
                   "axis_diverges_from_observed_motion"),
        "motion_distance_m": motion_norm,
        "deviation_rad": deviation,
        "alignment": alignment,
    }


def corridor_wall_midpoint(pose: Sequence[float], axis: Sequence[float],
                           left_clearance: float,
                           right_clearance: float) -> Point:
    """Project one pose to the midpoint of its bilateral corridor walls."""
    tangent = (-float(axis[1]), float(axis[0]))
    offset = 0.5 * (float(left_clearance) - float(right_clearance))
    return (float(pose[0]) + offset * tangent[0],
            float(pose[1]) + offset * tangent[1])


def corridor_width_profile_is_uniform(widths: Sequence[float],
                                      maximum_width: float = 4.2,
                                      maximum_spread: float = 0.9,
                                      required_samples: int = 3) -> bool:
    """Require the long, approximately equal-width signature of a corridor."""
    values = [float(value) for value in widths
              if math.isfinite(float(value)) and float(value) > 0.0]
    return bool(
        len(values) >= int(required_samples) and
        max(values) <= float(maximum_width) and
        max(values) - min(values) <= float(maximum_spread))


def corridor_portal_from_reachable_path(
        path: Sequence[Point], corridor_origin: Point,
        corridor_axis: Sequence[float], side: int,
        minimum_inside_lateral: float = 1.35) -> Optional[dict]:
    """Find the real side-opening station used by an observed-free path.

    A room frontier may lie metres along the corridor from its only reachable
    doorway. Projecting the frontier itself onto the corridor invents a false
    portal and can command a wall crossing. The first A* sample reaching
    room-side depth supplies the actual online portal station and a local
    just-inside target without using layout truth.
    """
    if not path or corridor_axis is None or len(corridor_axis) < 2:
        return None
    norm = math.hypot(float(corridor_axis[0]), float(corridor_axis[1]))
    if norm <= 1e-9 or int(side) == 0 or minimum_inside_lateral <= 0.0:
        return None
    axis = (float(corridor_axis[0]) / norm,
            float(corridor_axis[1]) / norm)
    normal = (-axis[1], axis[0])
    side_sign = 1.0 if int(side) > 0 else -1.0
    origin = (float(corridor_origin[0]), float(corridor_origin[1]))
    for index, raw_point in enumerate(path):
        point = (float(raw_point[0]), float(raw_point[1]))
        delta = (point[0] - origin[0], point[1] - origin[1])
        lateral = side_sign * (
            delta[0] * normal[0] + delta[1] * normal[1])
        if lateral + 1e-9 < float(minimum_inside_lateral):
            continue
        station = delta[0] * axis[0] + delta[1] * axis[1]
        centerline = (origin[0] + station * axis[0],
                      origin[1] + station * axis[1])
        return {
            "path_index": index,
            "station": station,
            "centerline": centerline,
            "inside_point": point,
            "inside_lateral": lateral,
        }
    return None


def recent_reverse_trajectory_anchors(
        trajectory: Sequence[Sequence[float]], target: Point,
        current: Point, spacing: float = 0.75,
        since: Optional[float] = None,
        match_tolerance: float = 0.35) -> List[Point]:
    """Return a bounded reverse path to the latest visit near ``target``.

    Room-exit recovery must not use the earliest nearest pose in the complete
    mission trajectory: every retry would then append the previous room path
    and grow indefinitely. Restrict samples to the current room visit and
    prefer the most recent current/target matches so each retry is trimmed to
    only its still-unexecuted suffix.
    """
    if spacing <= 0.0 or not trajectory:
        return []
    indexed = [
        (index, sample) for index, sample in enumerate(trajectory)
        if len(sample) >= 3 and
        (since is None or float(sample[0]) + 1e-9 >= float(since))
    ]
    if not indexed:
        return []

    def distance(sample, point):
        return math.hypot(float(sample[1]) - float(point[0]),
                          float(sample[2]) - float(point[1]))

    current_matches = [item for item in indexed
                       if distance(item[1], current) <= match_tolerance]
    if current_matches:
        end_index = current_matches[-1][0]
    else:
        # Prefer recency when localization jitter produces equal distances.
        end_index = min(
            indexed, key=lambda item: (distance(item[1], current), -item[0]))[0]
    eligible = [item for item in indexed if item[0] <= end_index]
    target_matches = [item for item in eligible
                      if distance(item[1], target) <= match_tolerance]
    if target_matches:
        target_index = target_matches[-1][0]
    else:
        target_index = min(
            eligible, key=lambda item: (distance(item[1], target), -item[0]))[0]
    if target_index > end_index:
        return []
    samples = list(trajectory[target_index:end_index + 1])[::-1]
    anchors: List[Point] = []
    last = (float(current[0]), float(current[1]))
    for sample in samples:
        point = (float(sample[1]), float(sample[2]))
        if math.hypot(point[0] - last[0], point[1] - last[1]) >= spacing:
            anchors.append(point)
            last = point
    final = (float(target[0]), float(target[1]))
    if (not anchors or
            math.hypot(anchors[-1][0] - final[0],
                       anchors[-1][1] - final[1]) > 0.15):
        anchors.append(final)
    return anchors


def drop_reached_waypoint_prefix(waypoints, current, tolerance):
    """Drop the contiguous execution-path prefix already reached."""
    remaining = [tuple(point) for point in waypoints]
    if current is None or len(current) < 2:
        return remaining, 0
    completed = 0
    threshold = max(0.0, float(tolerance))
    while remaining and math.hypot(
            float(remaining[0][0]) - float(current[0]),
            float(remaining[0][1]) - float(current[1])) <= threshold:
        remaining.pop(0)
        completed += 1
    return remaining, completed


def execution_result_matches(result, goal_stamp, point,
                             stamp_tolerance=1e-4,
                             position_tolerance=0.05):
    """Match an asynchronous executor result to one published waypoint."""
    if not isinstance(result, dict) or len(point) < 2:
        return False
    try:
        stamp = float(result["goal_stamp"])
        goal = result["goal"]
        x = float(goal["x"])
        y = float(goal["y"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return (abs(stamp - float(goal_stamp)) <= float(stamp_tolerance) and
            abs(x - float(point[0])) <= float(position_tolerance) and
            abs(y - float(point[1])) <= float(position_tolerance))


def corridor_side_frontier_eligible(
        longitudinal: float, lateral: float,
        minimum_lateral: float, maximum_longitudinal: float) -> bool:
    """Return side-debt eligibility independent of corridor-axis sign.

    A PCA corridor axis represents an unoriented line: ``axis`` and ``-axis``
    are geometrically identical.  Therefore a local door/side frontier must
    not be rejected merely because the stored vector points opposite current
    travel.  Branch visit memory handles genuine rear revisits separately.
    """
    return (
        abs(float(lateral)) >= max(0.0, float(minimum_lateral)) and
        abs(float(longitudinal)) <= max(0.0, float(maximum_longitudinal)))


@dataclass
class OccupancyGrid2D:
    """OccupancyGrid-like raster: -1 unknown, 0 free, >=50 occupied."""

    data: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.int16)
        if self.data.ndim != 2 or self.resolution <= 0.0:
            raise ValueError("invalid occupancy grid")

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    def world_to_cell(self, point: Sequence[float]) -> Optional[Tuple[int, int]]:
        x = int(math.floor((float(point[0]) - self.origin_x) / self.resolution))
        y = int(math.floor((float(point[1]) - self.origin_y) / self.resolution))
        return (x, y) if 0 <= x < self.width and 0 <= y < self.height else None

    def cell_to_world(self, cell: Sequence[int]) -> Point:
        return (self.origin_x + (int(cell[0]) + 0.5) * self.resolution,
                self.origin_y + (int(cell[1]) + 0.5) * self.resolution)


def _cell_clear(grid: OccupancyGrid2D, cell: Tuple[int, int],
                clearance_radius: float, allow_unknown: bool = False,
                ignored_cell: Optional[Tuple[int, int]] = None) -> bool:
    radius = max(0.0, float(clearance_radius))
    bound = int(math.ceil(radius / grid.resolution))
    for dy in range(-bound, bound + 1):
        for dx in range(-bound, bound + 1):
            if ((dx * grid.resolution) ** 2 +
                    (dy * grid.resolution) ** 2 > radius * radius + 1e-12):
                continue
            x, y = cell[0] + dx, cell[1] + dy
            if not (0 <= x < grid.width and 0 <= y < grid.height):
                return False
            if ignored_cell is not None and (x, y) == ignored_cell:
                continue
            state = int(grid.data[y, x])
            if state >= 50 or (state < 0 and not allow_unknown):
                return False
    return True


def astar_safe_path(grid: OccupancyGrid2D, start: Point, goal: Point,
                    clearance_radius: float = 0.30,
                    reached_tolerance: float = 0.30,
                    allow_blocked_start: bool = False,
                    maximum_expansions: int = 0) -> dict:
    """Return a safe observed-free A* result with an explicit failure reason."""
    if math.hypot(goal[0] - start[0], goal[1] - start[1]) <= reached_tolerance:
        return {"success": True, "reason": "already_at_goal", "path": [goal]}
    start_cell = grid.world_to_cell(start)
    goal_cell = grid.world_to_cell(goal)
    if start_cell is None:
        return {"success": False, "reason": "start_outside_map", "path": []}
    if goal_cell is None:
        return {"success": False, "reason": "goal_outside_map", "path": []}
    if (not allow_blocked_start and
            not _cell_clear(grid, start_cell, clearance_radius, allow_unknown=True)):
        return {"success": False, "reason": "start_footprint_blocked", "path": []}
    if not _cell_clear(grid, goal_cell, clearance_radius):
        return {"success": False, "reason": "goal_footprint_blocked", "path": []}
    if start_cell == goal_cell:
        return {"success": True, "reason": "already_at_goal", "path": [goal]}

    def heuristic(cell: Tuple[int, int]) -> float:
        return math.hypot(cell[0] - goal_cell[0], cell[1] - goal_cell[1])

    frontier = [(heuristic(start_cell), 0.0, start_cell)]
    costs = {start_cell: 0.0}
    parents: Dict[Tuple[int, int], Tuple[int, int]] = {}
    clear_cache: Dict[Tuple[int, int], bool] = {}
    expansions = 0
    # Four-connected filling prevents two free areas from becoming one
    # component through a single diagonal gap at an occupied wall corner.
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))
    while frontier:
        _, current_cost, current = heapq.heappop(frontier)
        expansions += 1
        if maximum_expansions > 0 and expansions > maximum_expansions:
            return {
                "success": False,
                "reason": "maximum_expansions",
                "path": [],
                "expansions": expansions,
            }
        if current == goal_cell:
            cells = [current]
            while cells[-1] != start_cell:
                cells.append(parents[cells[-1]])
            cells.reverse()
            path = [grid.cell_to_world(cell) for cell in cells]
            path[0], path[-1] = start, goal
            reason = "path_found_start_override" if allow_blocked_start else "path_found"
            return {
                "success": True,
                "reason": reason,
                "path": path,
                "expansions": expansions,
            }
        if current_cost > costs.get(current, math.inf) + 1e-9:
            continue
        for dx, dy in neighbors:
            nxt = current[0] + dx, current[1] + dy
            if not (0 <= nxt[0] < grid.width and 0 <= nxt[1] < grid.height):
                continue
            clear = clear_cache.get(nxt)
            if clear is None:
                clear = _cell_clear(
                    grid, nxt, clearance_radius,
                    ignored_cell=(
                        start_cell if allow_blocked_start else None))
                clear_cache[nxt] = clear
            if not clear:
                continue
            step = math.sqrt(2.0) if dx and dy else 1.0
            candidate = current_cost + step
            if candidate >= costs.get(nxt, math.inf):
                continue
            costs[nxt] = candidate
            parents[nxt] = current
            heapq.heappush(frontier, (candidate + heuristic(nxt), candidate, nxt))
    return {
        "success": False,
        "reason": "goal_unreachable",
        "path": [],
        "expansions": expansions,
    }


def nearest_reachable_goal(
        grid: OccupancyGrid2D, start: Point, desired: Point,
        clearance_radius: float = 0.20, reached_tolerance: float = 0.15,
        search_radius: float = 1.50, maximum_candidates: int = 64,
        preferred_points: Sequence[Point] = (),
        excluded_points: Sequence[Point] = ()) -> dict:
    """Relocate a stale goal to the nearest safe, A*-reachable map cell.

    Saved corridor-entry coordinates can become occupied after later LiDAR
    updates or footprint inflation.  A return operation should preserve the
    intended vicinity, not fail the complete exploration mission because one
    historical coordinate is no longer a valid goal.
    """
    radius = max(0.0, float(search_radius))
    limit = max(1, int(maximum_candidates))
    desired_cell = grid.world_to_cell(desired)
    if desired_cell is None:
        return {
            "success": False, "reason": "desired_outside_map", "path": [],
            "target": None, "candidates_tested": 0,
        }

    candidates = []
    cells_seen = set()

    def add_candidate(point: Sequence[float], preferred_rank: int) -> None:
        cell = grid.world_to_cell(point)
        if cell is None or cell in cells_seen:
            return
        if math.hypot(float(point[0]) - desired[0],
                      float(point[1]) - desired[1]) > radius + 1e-9:
            return
        if any(math.hypot(float(point[0]) - float(failed[0]),
                          float(point[1]) - float(failed[1])) <=
               max(grid.resolution, reached_tolerance)
               for failed in excluded_points if len(failed) >= 2):
            return
        if not _cell_clear(grid, cell, clearance_radius):
            return
        cells_seen.add(cell)
        world = (float(point[0]), float(point[1]))
        candidates.append((
            math.hypot(world[0] - desired[0], world[1] - desired[1]),
            preferred_rank,
            math.hypot(world[0] - start[0], world[1] - start[1]),
            cell,
            world,
        ))

    # Try the exact historical point first when its footprint is still valid,
    # followed by corridor-aware hints supplied by the runtime.
    add_candidate(desired, -2)
    for index, point in enumerate(preferred_points):
        if len(point) >= 2:
            add_candidate(point, -1 + index)

    bound = int(math.ceil(radius / grid.resolution))
    for dy in range(-bound, bound + 1):
        for dx in range(-bound, bound + 1):
            cell = desired_cell[0] + dx, desired_cell[1] + dy
            if not (0 <= cell[0] < grid.width and 0 <= cell[1] < grid.height):
                continue
            point = grid.cell_to_world(cell)
            add_candidate(point, len(preferred_points) + 1)

    # Exact entry first, then explicit corridor-axis hints, then generic cells
    # nearest the entry.  Determine reachability with one connected-component
    # traversal instead of running a full A* up to ``limit`` times.
    candidates.sort(key=lambda item: (item[1], item[0], item[2]))
    candidates = candidates[:limit]
    start_cell = grid.world_to_cell(start)
    if start_cell is None:
        return {
            "success": False, "reason": "start_outside_map", "path": [],
            "target": None, "candidates_tested": 0,
        }
    frontier = deque([start_cell])
    parents = {start_cell: None}
    clear_cache = {}
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1),
                 (1, 1), (1, -1), (-1, 1), (-1, -1))
    while frontier:
        current = frontier.popleft()
        for dx, dy in neighbors:
            nxt = current[0] + dx, current[1] + dy
            if nxt in parents or not (
                    0 <= nxt[0] < grid.width and
                    0 <= nxt[1] < grid.height):
                continue
            clear = clear_cache.get(nxt)
            if clear is None:
                clear = _cell_clear(
                    grid, nxt, clearance_radius, ignored_cell=start_cell)
                clear_cache[nxt] = clear
            if not clear:
                continue
            parents[nxt] = current
            frontier.append(nxt)

    for tested, (displacement, _, _, cell, target) in enumerate(
            candidates, 1):
        if cell not in parents:
            continue
        cells = [cell]
        while parents[cells[-1]] is not None:
            cells.append(parents[cells[-1]])
        cells.reverse()
        path = [grid.cell_to_world(item) for item in cells]
        path[0], path[-1] = start, target
        return {
            "success": True,
            "reason": ("original_goal_reachable"
                       if displacement <= 1e-9 else
                       "relocated_goal_reachable"),
            "path": path,
            "target": target,
            "displacement": displacement,
            "candidates_tested": tested,
        }
    return {
        "success": False,
        "reason": "goal_unreachable" if candidates else "no_safe_candidate",
        "path": [],
        "target": None,
        "candidates_tested": len(candidates),
    }


def side_region_component_match(
        grid: OccupancyGrid2D, target: Point,
        reference_points: Sequence[Point], axis: Sequence[float],
        centerline_origin: Point, side: int,
        minimum_lateral: float = 0.75,
        snap_radius: float = 0.75,
        maximum_cells: int = 100000) -> dict:
    """Match a side target to previously visited free-space components.

    The corridor strip is removed before flood filling, so two rooms connected
    only through the corridor remain distinct components.  All inputs are
    online FAST-LIO/map coordinates; no room polygons or simulator truth are
    involved.
    """
    norm = math.hypot(float(axis[0]), float(axis[1]))
    if norm <= 1e-9:
        return {
            "matched": False, "reason": "invalid_axis",
            "matched_indices": [], "component_cells": 0,
        }
    axis_x, axis_y = float(axis[0]) / norm, float(axis[1]) / norm
    normal_x, normal_y = -axis_y, axis_x
    stable_side = 1 if int(side) >= 0 else -1
    threshold = max(0.0, float(minimum_lateral))
    target_cell = grid.world_to_cell(target)
    if target_cell is None:
        return {
            "matched": False, "reason": "target_outside_map",
            "matched_indices": [], "component_cells": 0,
        }

    def belongs(cell: Tuple[int, int]) -> bool:
        if int(grid.data[cell[1], cell[0]]) != 0:
            return False
        point = grid.cell_to_world(cell)
        lateral = (
            (point[0] - float(centerline_origin[0])) * normal_x +
            (point[1] - float(centerline_origin[1])) * normal_y)
        return stable_side * lateral >= threshold

    def nearest_belonging(cell: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        if belongs(cell):
            return cell
        bound = int(math.ceil(max(0.0, float(snap_radius)) /
                              grid.resolution))
        alternatives = []
        for dy in range(-bound, bound + 1):
            for dx in range(-bound, bound + 1):
                candidate = cell[0] + dx, cell[1] + dy
                if not (0 <= candidate[0] < grid.width and
                        0 <= candidate[1] < grid.height):
                    continue
                distance = math.hypot(dx, dy) * grid.resolution
                if (distance <= float(snap_radius) + 1e-9 and
                        belongs(candidate)):
                    alternatives.append((distance, candidate))
        return min(alternatives, default=(None, None),
                   key=lambda item: item[0])[1]

    target_cell = nearest_belonging(target_cell)
    if target_cell is None:
        return {
            "matched": False, "reason": "target_not_in_side_free_space",
            "matched_indices": [], "component_cells": 0,
        }
    reference_cells: Dict[Tuple[int, int], List[int]] = {}
    for index, point in enumerate(reference_points):
        if len(point) < 2:
            continue
        cell = grid.world_to_cell(point)
        cell = nearest_belonging(cell) if cell is not None else None
        if cell is not None:
            reference_cells.setdefault(cell, []).append(index)
    if not reference_cells:
        return {
            "matched": False, "reason": "no_mapped_reference",
            "matched_indices": [], "component_cells": 0,
        }

    queue = deque([target_cell])
    visited = {target_cell}
    matches = []
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1),
                 (1, 1), (1, -1), (-1, 1), (-1, -1))
    limit = max(1, int(maximum_cells))
    while queue and len(visited) <= limit:
        cell = queue.popleft()
        matches.extend(reference_cells.get(cell, ()))
        for dx, dy in neighbors:
            nxt = cell[0] + dx, cell[1] + dy
            if (nxt in visited or not
                    (0 <= nxt[0] < grid.width and
                     0 <= nxt[1] < grid.height) or
                    not belongs(nxt)):
                continue
            visited.add(nxt)
            queue.append(nxt)
    matches = sorted(set(matches))
    return {
        "matched": bool(matches),
        "reason": ("visited_component_overlap" if matches else
                   "distinct_side_component"),
        "matched_indices": matches,
        "component_cells": len(visited),
        "truncated": bool(queue),
    }


def execution_waypoints(path: Sequence[Point], spacing: float = 0.75) -> List[Point]:
    """Resample a polyline while always retaining its final target.

    A* normally supplies dense grid points, while FAR visualization paths can
    contain only the endpoints of a long segment.  Interpolating long
    segments is required so downstream footprint checks validate the route,
    not only its destination.
    """
    if len(path) <= 1:
        return list(path)
    step = max(0.1, float(spacing))
    selected: List[Point] = []
    previous = (float(path[0][0]), float(path[0][1]))
    # A* paths are normally sampled at the occupancy-grid resolution (0.15 m
    # in SimEnv), which is smaller than the execution spacing.  Sampling each
    # grid edge independently therefore used to discard every intermediate
    # point and retain only the goal.  That silently changed a route through a
    # doorway into one diagonal current-pose -> room-goal chord.  Carry the
    # residual arclength across polyline edges so dense A* geometry survives.
    distance_to_sample = step
    for raw_point in path[1:]:
        point = (float(raw_point[0]), float(raw_point[1]))
        segment_start = previous
        segment_length = math.hypot(
            point[0] - segment_start[0], point[1] - segment_start[1])
        while segment_length + 1e-9 >= distance_to_sample:
            ratio = distance_to_sample / max(segment_length, 1e-9)
            candidate = (
                segment_start[0] + (point[0] - segment_start[0]) * ratio,
                segment_start[1] + (point[1] - segment_start[1]) * ratio,
            )
            if (not selected or
                    math.hypot(candidate[0] - selected[-1][0],
                               candidate[1] - selected[-1][1]) > 1e-6):
                selected.append(candidate)
            segment_start = candidate
            segment_length = math.hypot(
                point[0] - segment_start[0], point[1] - segment_start[1])
            distance_to_sample = step
        distance_to_sample -= segment_length
        previous = point
    goal = (float(path[-1][0]), float(path[-1][1]))
    if (not selected or
            math.hypot(
                selected[-1][0] - goal[0],
                selected[-1][1] - goal[1]) > 1e-6):
        selected.append(goal)
    return selected


def segmented_execution_waypoints(
        path: Sequence[Point], anchors: Sequence[Point],
        spacing: float = 0.90, tolerance: float = 0.08) -> List[Point]:
    """Resample a path while retaining ordered portal anchors.

    In particular, the first portal anchor may also be ``path[0]`` when the
    robot already completed RETURN. The former loop inspected only path[1:]
    and falsely reported that mandatory anchor as lost.
    """
    if not path:
        return []
    points = [(float(point[0]), float(point[1])) for point in path]
    required = [(float(point[0]), float(point[1])) for point in anchors]
    selected: List[Point] = []

    def extend_unique(items: Sequence[Point]) -> None:
        for point in items:
            if (not selected or
                    math.hypot(point[0] - selected[-1][0],
                               point[1] - selected[-1][1]) > 1e-6):
                selected.append(point)

    anchor_index = 0
    segment = [points[0]]
    if (required and
            math.hypot(points[0][0] - required[0][0],
                       points[0][1] - required[0][1]) <= tolerance):
        extend_unique([points[0]])
        anchor_index = 1
    for point in points[1:]:
        segment.append(point)
        if (anchor_index < len(required) and
                math.hypot(point[0] - required[anchor_index][0],
                           point[1] - required[anchor_index][1]) <= tolerance):
            extend_unique(execution_waypoints(segment, spacing))
            segment = [point]
            anchor_index += 1
    if len(segment) > 1:
        extend_unique(execution_waypoints(segment, spacing))
    return selected


@dataclass
class DepthBreadthConfig:
    """Room-truth-free budget for one spatial exploration region."""

    enabled: bool = False
    region_radius: float = 5.0
    exclusion_radius: float = 4.0
    maximum_goals: int = 3
    maximum_path_length: float = 10.0
    maximum_seconds: float = 90.0
    coverage_enabled: bool = True
    coverage_minimum_ratio: float = 0.88
    coverage_maximum_unknown_m2: float = 1.50
    coverage_minimum_known_m2: float = 6.0
    coverage_stable_cycles: int = 2
    coverage_minimum_entry_depth: float = 1.25
    coverage_minimum_post_entry_goals: int = 1

    def validate(self) -> None:
        if self.region_radius <= 0.0 or self.exclusion_radius <= 0.0:
            raise ValueError("invalid depth-breadth region radius")
        if self.maximum_goals < 1:
            raise ValueError("invalid depth-breadth goal budget")
        if self.maximum_path_length <= 0.0 or self.maximum_seconds <= 0.0:
            raise ValueError("invalid depth-breadth budget")
        if not 0.0 < self.coverage_minimum_ratio <= 1.0:
            raise ValueError("invalid regional coverage ratio")
        if min(self.coverage_maximum_unknown_m2,
               self.coverage_minimum_known_m2) < 0.0:
            raise ValueError("invalid regional coverage area")
        if self.coverage_stable_cycles < 1:
            raise ValueError("invalid regional coverage stability count")
        if self.coverage_minimum_entry_depth < 0.0:
            raise ValueError("invalid regional coverage entry depth")
        if self.coverage_minimum_post_entry_goals < 0:
            raise ValueError("invalid regional post-entry goal count")


def local_region_coverage(grid: OccupancyGrid2D, center: Point,
                          radius: float,
                          observed_cells: Optional[Sequence[Tuple[int, int]]] = None) -> dict:
    """Occlusion-aware coverage of one local traversable region.

    The denominator is the free-or-unknown component reachable from a known
    free seed without crossing an observed occupied cell.  Thus an unobserved
    space behind an opening remains debt, while unknown space behind a mapped
    wall is excluded.  This never stamps a geometric sensor-radius disk as
    covered.
    """
    radius = max(grid.resolution, float(radius))
    cx = int(math.floor((float(center[0]) - grid.origin_x) / grid.resolution))
    cy = int(math.floor((float(center[1]) - grid.origin_y) / grid.resolution))
    bound = int(math.ceil(radius / grid.resolution))
    radius_sq = radius * radius

    def inside(cell: Tuple[int, int]) -> bool:
        if not (0 <= cell[0] < grid.width and 0 <= cell[1] < grid.height):
            return False
        world = grid.cell_to_world(cell)
        return ((world[0] - center[0]) ** 2 +
                (world[1] - center[1]) ** 2 <= radius_sq)

    seed = None
    seed_distance = math.inf
    search = max(2, int(math.ceil(0.75 / grid.resolution)))
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            cell = cx + dx, cy + dy
            if (inside(cell) and int(grid.data[cell[1], cell[0]]) == 0):
                distance = dx * dx + dy * dy
                if distance < seed_distance:
                    seed, seed_distance = cell, distance
    if seed is None:
        return {
            "coverage_ratio": 0.0, "known_free_cells": 0,
            "unknown_cells": 0, "largest_unknown_m2": 0.0,
            "frontier_cells": 0, "domain_cells": 0,
            "known_free_m2": 0.0, "coverage_complete": False,
            "reason": "no_known_free_seed",
        }

    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1),
                 (1, 1), (1, -1), (-1, 1), (-1, -1))
    domain = {seed}
    queue = deque([seed])
    while queue:
        cell = queue.popleft()
        for dx, dy in neighbors:
            nxt = cell[0] + dx, cell[1] + dy
            if nxt in domain or not inside(nxt):
                continue
            # Occupied evidence is the occlusion boundary.
            if int(grid.data[nxt[1], nxt[0]]) >= 50:
                continue
            domain.add(nxt)
            queue.append(nxt)
    map_known_free = {cell for cell in domain
                      if int(grid.data[cell[1], cell[0]]) == 0}
    post_entry_mode = observed_cells is not None
    if post_entry_mode:
        observed = {
            (int(cell[0]), int(cell[1])) for cell in observed_cells
            if (int(cell[0]), int(cell[1])) in domain
        }
        # Only rays emitted after physical entry may pay the region's coverage
        # debt.  Cells mapped remotely from the corridor remain debt even
        # though they are already free in the cumulative occupancy grid.
        known_free = map_known_free & observed
    else:
        known_free = map_known_free
    unknown = domain - known_free
    frontier_unknown = {
        cell for cell in unknown
        if any((cell[0] + dx, cell[1] + dy) in known_free
               for dx, dy in neighbors)
    }
    remaining = set(unknown)
    largest_unknown = 0
    while remaining:
        component_size = 0
        component_queue = deque([remaining.pop()])
        while component_queue:
            cell = component_queue.popleft()
            component_size += 1
            for dx, dy in neighbors:
                nxt = cell[0] + dx, cell[1] + dy
                if nxt in remaining:
                    remaining.remove(nxt)
                    component_queue.append(nxt)
        largest_unknown = max(largest_unknown, component_size)
    cell_area = grid.resolution * grid.resolution
    ratio = (float(len(known_free)) / float(len(domain))
             if domain else 0.0)
    return {
        "coverage_ratio": ratio,
        "known_free_cells": len(known_free),
        "unknown_cells": len(unknown),
        "largest_unknown_m2": largest_unknown * cell_area,
        "frontier_cells": len(frontier_unknown),
        "domain_cells": len(domain),
        "known_free_m2": len(known_free) * cell_area,
        "map_known_free_cells": len(map_known_free),
        "post_entry_observed_cells": len(known_free) if post_entry_mode else None,
        "coverage_source": (
            "post_entry_lidar_rays" if post_entry_mode else
            "cumulative_occupancy_map"),
        "coverage_complete": False,
        "reason": "measured",
    }


@dataclass
class CorridorSideBranch:
    """Persistent door-free side branch at one corridor station."""

    branch_id: str
    station: float
    side: int
    entry_target: Point
    frontier: Optional[Point] = None
    state: str = "UNENTERED"
    observations: int = 1
    attempt_count: int = 0
    last_seen: float = 0.0
    cooldown_until: float = 0.0
    # Lateral penetration of the best online known-free entry target.  This
    # prevents a later shallow frontier observation from degrading a branch
    # that already has a safely reachable target farther through the opening.
    entry_quality: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class CorridorBranchScheduler:
    """Pair left/right traversable regions along a corridor backbone.

    This is deliberately not a door detector. A branch is registered from a
    reachable lateral frontier and grouped only by its longitudinal station
    and side. After visiting one side, the opposite side at the same station
    outranks corridor-forward work.
    """

    def __init__(self, station_tolerance: float = 1.75,
                 retry_cooldown: float = 45.0,
                 entry_target_tolerance: float = 0.90):
        if (station_tolerance <= 0.0 or retry_cooldown < 0.0 or
                entry_target_tolerance <= 0.0):
            raise ValueError("invalid corridor branch scheduler configuration")
        self.station_tolerance = float(station_tolerance)
        self.retry_cooldown = float(retry_cooldown)
        self.entry_target_tolerance = float(entry_target_tolerance)
        self.branches: List[CorridorSideBranch] = []
        self.active_branch_id: Optional[str] = None
        self.active_station: Optional[float] = None
        self.aliases: Dict[str, str] = {}
        self._sequence = 0

    def _find(self, branch_id: Optional[str]) -> Optional[CorridorSideBranch]:
        while branch_id in self.aliases:
            branch_id = self.aliases[branch_id]
        return next((item for item in self.branches
                     if item.branch_id == branch_id), None)

    def find(self, branch_id: Optional[str]) -> Optional[CorridorSideBranch]:
        """Resolve a current or merged historical branch ID."""
        return self._find(branch_id)

    def merge_ids(self, keep_id: str,
                  duplicate_id: str) -> Optional[CorridorSideBranch]:
        """Merge explicitly matched map components while preserving memory."""
        keep, duplicate = self._find(keep_id), self._find(duplicate_id)
        if keep is None:
            return duplicate
        if duplicate is None or keep.branch_id == duplicate.branch_id:
            return keep
        if keep.side != duplicate.side:
            return keep
        return self._merge(keep, duplicate)

    def _same_region(self, first: CorridorSideBranch,
                     second: CorridorSideBranch) -> bool:
        if first.side != second.side:
            return False
        station_close = (
            abs(first.station - second.station) <= self.station_tolerance)
        target_close = (
            math.hypot(first.entry_target[0] - second.entry_target[0],
                       first.entry_target[1] - second.entry_target[1]) <=
            self.entry_target_tolerance)
        return station_close or target_close

    def _merge(self, keep: CorridorSideBranch,
               duplicate: CorridorSideBranch) -> CorridorSideBranch:
        """Merge two observations while retaining visit/cooldown memory."""
        if keep.branch_id == duplicate.branch_id:
            return keep
        total = max(1, keep.observations + duplicate.observations)
        keep.station = (
            keep.station * keep.observations +
            duplicate.station * duplicate.observations) / total
        if (duplicate.entry_quality > keep.entry_quality + 1e-9 or
                (abs(duplicate.entry_quality - keep.entry_quality) <= 1e-9 and
                 duplicate.last_seen >= keep.last_seen)):
            keep.entry_target = duplicate.entry_target
            if duplicate.frontier is not None:
                keep.frontier = duplicate.frontier
            keep.entry_quality = duplicate.entry_quality
        keep.last_seen = max(keep.last_seen, duplicate.last_seen)
        keep.observations = total
        keep.attempt_count += duplicate.attempt_count
        keep.cooldown_until = max(
            keep.cooldown_until, duplicate.cooldown_until)
        state_rank = {
            "UNENTERED": 0, "FAILED_COOLDOWN": 1,
            "ENTERED_PARTIAL": 2, "COVERED": 3,
        }
        if state_rank.get(duplicate.state, -1) > \
                state_rank.get(keep.state, -1):
            keep.state = duplicate.state
        self.aliases[duplicate.branch_id] = keep.branch_id
        for old_id, target_id in list(self.aliases.items()):
            if target_id == duplicate.branch_id:
                self.aliases[old_id] = keep.branch_id
        if self.active_branch_id == duplicate.branch_id:
            self.active_branch_id = keep.branch_id
        self.branches.remove(duplicate)
        if self.active_branch_id == keep.branch_id:
            self.active_station = keep.station
        return keep

    def _consolidate(self, preferred: CorridorSideBranch
                     ) -> CorridorSideBranch:
        """Reconcile IDs after averaged station/target estimates drift closer."""
        changed = True
        while changed:
            changed = False
            for branch in list(self.branches):
                if (branch.branch_id != preferred.branch_id and
                        self._same_region(preferred, branch)):
                    preferred = self._merge(preferred, branch)
                    changed = True
                    break
        return preferred

    def register(self, station: float, side: int, entry_target: Point,
                 now: float, frontier: Optional[Point] = None,
                 entry_quality: Optional[float] = None) -> CorridorSideBranch:
        side = 1 if int(side) >= 0 else -1
        observation = CorridorSideBranch(
            branch_id="", station=float(station), side=side,
            entry_target=(float(entry_target[0]), float(entry_target[1])))
        matches = [
            item for item in self.branches
            if self._same_region(item, observation)]
        match = min(matches, key=lambda item: int(
            item.branch_id.rsplit("_", 1)[-1])) if matches else None
        if match is None:
            self._sequence += 1
            match = CorridorSideBranch(
                branch_id="side_branch_{:02d}".format(self._sequence),
                station=float(station), side=side,
                entry_target=(float(entry_target[0]), float(entry_target[1])),
                frontier=(float(frontier[0]), float(frontier[1]))
                if frontier is not None else None,
                last_seen=float(now),
                entry_quality=max(0.0, float(entry_quality or 0.0)))
            self.branches.append(match)
        else:
            for duplicate in matches:
                if duplicate.branch_id != match.branch_id:
                    match = self._merge(match, duplicate)
            count = match.observations + 1
            match.station = (
                match.station * match.observations + float(station)) / count
            quality = (match.entry_quality if entry_quality is None else
                       max(0.0, float(entry_quality)))
            # Physical-entry observations (quality=None) remain authoritative.
            # Frontier-only updates replace the endpoint only when they offer
            # at least as much safe lateral penetration.
            if entry_quality is None or quality >= match.entry_quality - 1e-9:
                match.entry_target = (
                    float(entry_target[0]), float(entry_target[1]))
                match.frontier = (
                    (float(frontier[0]), float(frontier[1]))
                    if frontier is not None else match.frontier)
                match.entry_quality = quality
            match.observations = count
            match.last_seen = float(now)
        return self._consolidate(match)

    def select(self, candidate_ids: Sequence[str], current_station: float,
               now: float) -> Optional[CorridorSideBranch]:
        allowed = []
        for branch_id in candidate_ids:
            branch = self._find(branch_id)
            if branch is None or branch.state == "COVERED":
                continue
            if branch.state in ("ENTERED_PARTIAL", "FAILED_COOLDOWN") and \
                    float(now) < branch.cooldown_until:
                continue
            allowed.append(branch)
        if not allowed:
            return None
        # A remembered partial region must never outrank any discovered branch
        # that has not been physically entered.  If that new branch is not in
        # the current candidate set, return None so ordinary corridor FUEL
        # progress can move the robot toward it instead of revisiting a room.
        if any(branch.state == "UNENTERED" for branch in self.branches):
            allowed = [branch for branch in allowed
                       if branch.state == "UNENTERED"]
            if not allowed:
                return None

        def priority(branch: CorridorSideBranch) -> Tuple[float, ...]:
            paired = (
                self.active_station is not None and
                abs(branch.station - self.active_station) <=
                self.station_tolerance)
            state_rank = {
                "UNENTERED": 0.0, "ENTERED_PARTIAL": 1.0,
                "FAILED_COOLDOWN": 2.0,
            }.get(branch.state, 3.0)
            # New work precedes revisits. Within the same visit state, clear
            # the paired opposite side and avoid rearward corridor motion.
            delta = branch.station - float(current_station)
            rear_penalty = 1.0 if delta < -self.station_tolerance else 0.0
            return (state_rank, 0.0 if paired else 1.0,
                    rear_penalty, abs(delta), -branch.observations)

        return min(allowed, key=priority)

    def mark_entered(self, branch_id: str) -> None:
        branch = self._find(branch_id)
        if branch is None:
            return
        branch.state = "ENTERED_PARTIAL"
        branch.attempt_count += 1
        self.active_branch_id = branch.branch_id
        self.active_station = branch.station

    def mark_entry_failed(self, branch_id: str, now: float) -> Optional[str]:
        """Cooldown a side endpoint rejected before physical entry.

        A stale frontier can remain A*-reachable while SCAN-lite correctly
        rejects its final chord.  Without persisting that failure, the
        remembered-branch queue immediately republishes the identical point
        until the mission-wide consecutive-failure limit aborts exploration.
        """
        branch = self._find(branch_id)
        if branch is None:
            return None
        branch.state = "FAILED_COOLDOWN"
        branch.attempt_count += 1
        branch.cooldown_until = float(now) + self.retry_cooldown
        if self.active_branch_id == branch.branch_id:
            self.active_branch_id = None
        return branch.state

    def mark_exit(self, now: float, coverage_complete: bool,
                  failed: bool = False) -> Optional[str]:
        branch = self._find(self.active_branch_id)
        if branch is None:
            self.active_branch_id = None
            return None
        if coverage_complete:
            branch.state = "COVERED"
            branch.cooldown_until = 0.0
        else:
            branch.state = "FAILED_COOLDOWN" if failed else "ENTERED_PARTIAL"
            branch.cooldown_until = float(now) + self.retry_cooldown
        self.active_branch_id = None
        return branch.state

    def snapshot(self) -> List[dict]:
        return [item.to_dict() for item in self.branches]


class DepthBreadthScheduler:
    """Alternate bounded local depth/breadth work with a global region switch.

    A region begins at the current pose.  Moving farther than ``region_radius``
    naturally starts a new region (normal corridor progress).  Otherwise one
    primary/depth target and a bounded number of breadth targets are allowed.
    When a budget is exhausted, the caller asks the global planner for a goal
    outside the exclusion circle, or returns to ``entry_anchor``.
    """

    def __init__(self, config: DepthBreadthConfig):
        config.validate()
        self.config = config
        self.anchor: Optional[Point] = None
        self.entry_anchor: Optional[Point] = None
        self.started_at = 0.0
        self.goal_count = 0
        self.path_length = 0.0
        self.escape_requested = False
        self.escape_reason: Optional[str] = None
        self.region_index = 0
        self.coverage_status: Optional[dict] = None
        self.coverage_streak = 0
        self.entry_return_pending = False
        self.entry_return_completed = False
        self.entry_return_failed = False
        self.entry_observed = False
        self.post_entry_goal_count = 0
        self.physical_entry_anchor: Optional[Point] = None
        self.physical_entry_started_at: Optional[float] = None

    def _begin(self, anchor: Point, entry: Point, now: float) -> None:
        self.anchor = (float(anchor[0]), float(anchor[1]))
        self.entry_anchor = (float(entry[0]), float(entry[1]))
        self.started_at = float(now)
        self.goal_count = 0
        self.path_length = 0.0
        self.escape_requested = False
        self.escape_reason = None
        self.coverage_status = None
        self.coverage_streak = 0
        self.entry_return_pending = False
        self.entry_return_completed = False
        self.entry_return_failed = False
        self.entry_observed = False
        self.post_entry_goal_count = 0
        self.physical_entry_anchor = None
        self.physical_entry_started_at = None
        self.region_index += 1

    def mark_region_entry(self, entry: Point, inside: Point, now: float) -> str:
        """Start a region only after a corridor-to-open-area transition.

        ``entry`` is the last corridor pose and ``inside`` is the first pose
        classified in the wider area.  No door detector or layout truth is
        involved.  Remote LiDAR rays through an opening therefore cannot by
        themselves create the physical-entry evidence required for completion.
        """
        self._begin(inside, entry, now)
        self.entry_observed = True
        self.physical_entry_anchor = (
            float(entry[0]), float(entry[1]))
        self.physical_entry_started_at = float(now)
        return "physical_region_entry_observed"

    def mark_corridor_transit(self) -> None:
        """End physical-entry evidence after returning to the corridor."""
        self.entry_observed = False
        self.post_entry_goal_count = 0
        self.physical_entry_anchor = None
        self.physical_entry_started_at = None

    def ensure_started(self, pose: Point, now: float) -> None:
        if self.anchor is None:
            self._begin(pose, pose, now)

    def phase(self, now: float) -> str:
        if not self.config.enabled:
            return "DISABLED"
        if self.escape_requested:
            return "GLOBAL_BREADTH_SWITCH"
        if self.goal_count == 0:
            return "LOCAL_DEPTH"
        return "LOCAL_BREADTH"

    def update_budget(self, now: float) -> None:
        if not self.config.enabled or self.anchor is None or self.escape_requested:
            return
        budget_started = (
            self.physical_entry_started_at
            if self.entry_observed and
            self.physical_entry_started_at is not None else
            self.started_at)
        elapsed = max(0.0, float(now) - budget_started)
        if self.goal_count >= self.config.maximum_goals:
            self.escape_requested, self.escape_reason = True, "goal_budget"
        elif self.path_length >= self.config.maximum_path_length:
            self.escape_requested, self.escape_reason = True, "path_budget"
        elif elapsed >= self.config.maximum_seconds and self.goal_count > 0:
            self.escape_requested, self.escape_reason = True, "time_budget"

    def update_coverage(self, status: dict, pose: Point, now: float) -> bool:
        """Request a switch after physical entry and stable sensor coverage."""
        self.coverage_status = dict(status)
        physical_entry = self.physical_entry_anchor
        entry_depth = (
            math.hypot(float(pose[0]) - physical_entry[0],
                       float(pose[1]) - physical_entry[1])
            if physical_entry is not None else 0.0)
        entry_requirement_met = bool(
            self.entry_observed and
            entry_depth >= self.config.coverage_minimum_entry_depth and
            self.post_entry_goal_count >=
            self.config.coverage_minimum_post_entry_goals)
        self.coverage_status.update({
            "physical_entry_observed": self.entry_observed,
            "entry_depth_m": entry_depth,
            "post_entry_goal_count": self.post_entry_goal_count,
            "entry_requirement_met": entry_requirement_met,
        })
        complete = bool(
            self.config.coverage_enabled and entry_requirement_met and
            float(status.get("coverage_ratio", 0.0)) >=
            self.config.coverage_minimum_ratio and
            float(status.get("largest_unknown_m2", math.inf)) <=
            self.config.coverage_maximum_unknown_m2 and
            float(status.get("known_free_m2", 0.0)) >=
            self.config.coverage_minimum_known_m2)
        self.coverage_status["coverage_complete"] = complete
        self.coverage_streak = self.coverage_streak + 1 if complete else 0
        # A map-confirmed re-entry has higher priority than coverage
        # bookkeeping: preserve its pending return-to-entry action. Normal
        # budget exits may still be upgraded to coverage-complete.
        if (complete and self.coverage_streak >=
                self.config.coverage_stable_cycles and
                self.escape_reason != "coverage_complete" and
                self.escape_reason != "visited_region_reentry"):
            self.escape_requested = True
            self.escape_reason = "coverage_complete"
            self.entry_return_pending = bool(
                physical_entry is not None and
                math.hypot(pose[0] - physical_entry[0],
                           pose[1] - physical_entry[1]) > 0.45)
            self.entry_return_completed = not self.entry_return_pending
            return True
        return False

    def record_entry_return(self) -> str:
        self.entry_return_pending = False
        self.entry_return_completed = True
        self.entry_return_failed = False
        return "coverage_entry_return_completed"

    def record_entry_return_failed(self) -> str:
        # The saved pose can become occupied after additional LiDAR evidence.
        # Keep the completed-region exclusion, but let the global planner find
        # another reachable way out instead of retrying one stale coordinate.
        self.entry_return_pending = False
        self.entry_return_completed = False
        self.entry_return_failed = True
        return "coverage_entry_return_bypassed_unreachable"

    def request_entry_return(self, reason: str) -> str:
        """Force a prompt return after map evidence identifies a re-entry."""
        if not self.entry_observed or self.entry_anchor is None:
            return "entry_return_not_available"
        self.escape_requested = True
        self.escape_reason = str(reason)
        self.entry_return_pending = True
        self.entry_return_completed = False
        self.entry_return_failed = False
        return "entry_return_requested_" + str(reason)

    def record_success(self, start: Point, end: Point, path_length: float,
                       now: float, was_escape: bool = False) -> str:
        if not self.config.enabled:
            return "disabled"
        self.ensure_started(start, now)
        assert self.anchor is not None
        physical_entry_already_observed = self.entry_observed
        physical_entry_anchor = self.physical_entry_anchor
        physical_entry_started_at = self.physical_entry_started_at
        if was_escape:
            self._begin(end, start, now)
            # A global switch can move between local patches inside one wide
            # entered area.  Do not erase the fact that the robot physically
            # crossed into that area merely because the circular anchor moved.
            self.entry_observed = physical_entry_already_observed
            self.physical_entry_anchor = physical_entry_anchor
            self.physical_entry_started_at = physical_entry_started_at
            return "global_breadth_switch_completed"
        if math.hypot(end[0] - self.anchor[0], end[1] - self.anchor[1]) > \
                self.config.region_radius:
            self._begin(end, start, now)
            self.entry_observed = physical_entry_already_observed
            self.physical_entry_anchor = physical_entry_anchor
            self.physical_entry_started_at = physical_entry_started_at
            self.goal_count = 1
            self.post_entry_goal_count = (
                1 if physical_entry_already_observed else 0)
            self.path_length = max(0.0, float(path_length))
            self.update_budget(now)
            return "spatial_region_advanced"
        self.goal_count += 1
        if self.entry_observed:
            self.post_entry_goal_count += 1
        self.path_length += max(0.0, float(path_length))
        self.update_budget(now)
        return "local_goal_completed"

    def exclusion(self, now: float) -> Optional[dict]:
        self.update_budget(now)
        if not self.config.enabled or not self.escape_requested or self.anchor is None:
            return None
        return {
            "center": self.anchor,
            "radius": self.config.exclusion_radius,
            "reason": self.escape_reason,
            # A budget-limited partial visit must also leave through the
            # original physical entry, not through the most recent 5 m
            # subregion anchor deep inside the room. It remains incomplete
            # and may be revisited later; this only prevents local trapping.
            "entry_anchor": (
                self.physical_entry_anchor
                if self.entry_observed and
                self.physical_entry_anchor is not None
                else self.entry_anchor),
            "region_index": self.region_index,
            "coverage": self.coverage_status,
            "entry_return_pending": self.entry_return_pending,
            "entry_return_failed": self.entry_return_failed,
            "physical_entry_observed": self.entry_observed,
            "physical_entry_started_at": self.physical_entry_started_at,
            "post_entry_goal_count": self.post_entry_goal_count,
        }
