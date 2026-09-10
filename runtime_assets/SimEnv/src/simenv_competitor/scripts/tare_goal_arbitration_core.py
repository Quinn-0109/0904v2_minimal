#!/usr/bin/env python3
"""ROS-free commitment policy for rolling Official TARE waypoints."""

from collections import deque
from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple


Point = Tuple[float, float, float]


def planar_distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.hypot(float(first[0]) - float(second[0]),
                      float(first[1]) - float(second[1]))


def wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def goal_is_backward(pose: Sequence[float], yaw: float,
                     goal: Sequence[float]) -> bool:
    direction = math.atan2(float(goal[1]) - float(pose[1]),
                           float(goal[0]) - float(pose[0]))
    return abs(wrap_angle(direction - float(yaw))) > 0.5 * math.pi


@dataclass
class ArbitrationConfig:
    min_goal_commit_time: float = 12.0
    preemption_ratio: float = 1.30
    preemption_absolute_gain: float = 3.0
    backward_penalty: float = 5.0
    active_region_radius: float = 4.0
    local_exploration_depth: float = 3.0
    minimum_unknown_area_to_continue: float = 3.0
    minimum_frontier_cells_to_continue: int = 8
    maximum_failures: int = 3
    max_supersede_per_minute: int = 5
    forced_commit_duration: float = 15.0
    same_goal_distance: float = 0.45
    reached_distance: float = 0.45
    frontier_no_growth_updates: int = 3
    failed_goal_blacklist_duration: float = 20.0


@dataclass
class GoalMetrics:
    unknown_area: float = 0.0
    frontier_cells: int = 0


@dataclass
class ArbitrationDecision:
    accepted: bool
    reason: str
    new_score: float
    current_score: Optional[float]
    distance_to_new_goal: float
    distance_to_current_goal: Optional[float]
    is_backward_goal: bool
    active_region_id: Optional[str]
    supersede_count_last_60s: int
    forced_commit_mode: bool


def map_goal_metrics(grid, goal: Sequence[float], radius: float) -> GoalMetrics:
    """Measure unknown area and free/unknown boundary cells near a goal."""
    if grid is None:
        return GoalMetrics()
    center = grid.world_to_cell(goal)
    if center is None:
        return GoalMetrics()
    cell_radius = max(1, int(math.ceil(float(radius) / grid.resolution)))
    radius_sq = float(radius) ** 2
    unknown = 0
    frontier = 0
    for y in range(max(0, center[1] - cell_radius),
                   min(grid.height, center[1] + cell_radius + 1)):
        for x in range(max(0, center[0] - cell_radius),
                       min(grid.width, center[0] + cell_radius + 1)):
            world = grid.cell_to_world((x, y))
            if ((world[0] - float(goal[0])) ** 2 +
                    (world[1] - float(goal[1])) ** 2 > radius_sq):
                continue
            state = int(grid.data[y, x])
            if state < 0:
                unknown += 1
                continue
            if state != 0:
                continue
            for nx, ny in ((x + 1, y), (x - 1, y),
                           (x, y + 1), (x, y - 1)):
                if (0 <= nx < grid.width and 0 <= ny < grid.height and
                        int(grid.data[ny, nx]) < 0):
                    frontier += 1
                    break
    return GoalMetrics(
        unknown_area=float(unknown) * grid.resolution ** 2,
        frontier_cells=frontier)


def goal_score(pose: Sequence[float], yaw: float, goal: Sequence[float],
               metrics: GoalMetrics, backward_penalty: float,
               exempt_backward: bool = False) -> Tuple[float, bool]:
    """Map-derived comparison score; it never proposes an exploration goal."""
    backward = goal_is_backward(pose, yaw, goal)
    score = (10.0 +
             min(20.0, max(0.0, metrics.unknown_area)) +
             min(5.0, 0.05 * max(0, metrics.frontier_cells)) -
             0.35 * planar_distance(pose, goal))
    if backward and not exempt_backward:
        score -= float(backward_penalty)
    return score, backward


class TareGoalArbitrationPolicy:
    """Stateful accept/reject policy for TARE's rolling waypoint stream."""

    def __init__(self, config: Optional[ArbitrationConfig] = None):
        self.config = config or ArbitrationConfig()
        self.current_goal: Optional[Point] = None
        self.current_started_at = 0.0
        self.current_failures = 0
        self.active_region_id: Optional[str] = None
        self.active_region_center: Optional[Point] = None
        self.active_region_start_pose: Optional[Point] = None
        self.active_region_complete = True
        self._region_sequence = 0
        self._frontier_peak = 0
        self._frontier_no_growth = 0
        self.supersede_times = deque()
        self.forced_commit_until = 0.0
        self.failed_goals = deque()

    def _trim_supersedes(self, now: float) -> None:
        while self.supersede_times and self.supersede_times[0] < now - 60.0:
            self.supersede_times.popleft()

    def _trim_failed_goals(self, now: float) -> None:
        while self.failed_goals and self.failed_goals[0][1] <= now:
            self.failed_goals.popleft()

    def _failed_goal_blacklisted(self, goal: Point, now: float) -> bool:
        self._trim_failed_goals(now)
        return any(
            planar_distance(goal, failed_goal) <
            self.config.same_goal_distance
            for failed_goal, _ in self.failed_goals)

    def _start_region(self, goal: Point, pose: Point,
                      metrics: GoalMetrics) -> None:
        self._region_sequence += 1
        self.active_region_id = "active_region_{:04d}".format(
            self._region_sequence)
        self.active_region_center = goal
        self.active_region_start_pose = pose
        self.active_region_complete = False
        self._frontier_peak = metrics.frontier_cells
        self._frontier_no_growth = 0

    def update_region(self, grid, pose: Optional[Point]) -> None:
        if (self.active_region_complete or self.active_region_center is None or
                pose is None):
            return
        metrics = map_goal_metrics(
            grid, self.active_region_center,
            self.config.active_region_radius)
        if metrics.frontier_cells > self._frontier_peak:
            self._frontier_peak = metrics.frontier_cells
            self._frontier_no_growth = 0
        else:
            self._frontier_no_growth += 1
        depth = planar_distance(pose, self.active_region_start_pose)
        self.active_region_complete = bool(
            metrics.frontier_cells <
            self.config.minimum_frontier_cells_to_continue or
            metrics.unknown_area <
            self.config.minimum_unknown_area_to_continue or
            self.current_failures >= self.config.maximum_failures or
            (depth >= self.config.local_exploration_depth and
             self._frontier_no_growth >=
             self.config.frontier_no_growth_updates))

    def mark_failure(self, now: float = 0.0) -> Optional[Point]:
        if self.current_goal is not None:
            self.current_failures += 1
        if self.current_failures >= self.config.maximum_failures:
            failed_goal = self.current_goal
            if failed_goal is not None:
                self.failed_goals.append((
                    failed_goal,
                    float(now) + self.config.failed_goal_blacklist_duration))
            self.current_goal = None
            self.current_failures = 0
            self.active_region_complete = True
            return failed_goal
        return None

    def mark_execution_result(self, success: bool,
                              pose: Optional[Point],
                              now: float = 0.0) -> Optional[Point]:
        if self.current_goal is None:
            return None
        if not success:
            return self.mark_failure(now)
        if (pose is not None and
                planar_distance(pose, self.current_goal) <=
                self.config.reached_distance):
            self.current_goal = None
            self.current_failures = 0
        return None

    def _accept(self, goal: Point, pose: Point, metrics: GoalMetrics,
                now: float, replacing: bool) -> None:
        if replacing:
            self.supersede_times.append(now)
            self._trim_supersedes(now)
            if (len(self.supersede_times) >
                    self.config.max_supersede_per_minute):
                self.forced_commit_until = now + \
                    self.config.forced_commit_duration
        same_region = (
            not self.active_region_complete and
            self.active_region_center is not None and
            planar_distance(goal, self.active_region_center) <=
            self.config.active_region_radius)
        if not same_region:
            self._start_region(goal, pose, metrics)
        self.current_goal = goal
        self.current_started_at = now
        self.current_failures = 0

    def decide(self, goal: Point, pose: Point, yaw: float, grid, now: float,
               recovery: bool = False) -> ArbitrationDecision:
        self._trim_supersedes(now)
        self._trim_failed_goals(now)
        self.update_region(grid, pose)
        new_metrics = map_goal_metrics(
            grid, goal, self.config.active_region_radius)
        failure_override = self.current_failures > 0 or bool(recovery)
        new_score, backward = goal_score(
            pose, yaw, goal, new_metrics, self.config.backward_penalty,
            exempt_backward=failure_override)
        distance_new = planar_distance(pose, goal)
        current_score = None
        distance_current = None

        def decision(accepted: bool, reason: str) -> ArbitrationDecision:
            return ArbitrationDecision(
                accepted, reason, new_score, current_score, distance_new,
                distance_current, backward, self.active_region_id,
                len(self.supersede_times), now < self.forced_commit_until)

        if self._failed_goal_blacklisted(goal, now):
            return decision(False, "failed_goal_blacklisted")

        if self.current_goal is None:
            self._accept(goal, pose, new_metrics, now, replacing=False)
            return decision(True, "no_active_goal")

        distance_current = planar_distance(pose, self.current_goal)
        if planar_distance(goal, self.current_goal) < \
                self.config.same_goal_distance:
            return decision(False, "same_current_goal")

        current_metrics = map_goal_metrics(
            grid, self.current_goal, self.config.active_region_radius)
        current_score, _ = goal_score(
            pose, yaw, self.current_goal, current_metrics,
            self.config.backward_penalty,
            exempt_backward=failure_override)

        if now < self.forced_commit_until and not failure_override:
            return decision(False, "forced_commit_mode")
        if (now - self.current_started_at <
                self.config.min_goal_commit_time and not failure_override):
            return decision(False, "min_goal_commit_time")
        outside_active_region = (
            not self.active_region_complete and
            self.active_region_center is not None and
            planar_distance(goal, self.active_region_center) >
            self.config.active_region_radius)
        if outside_active_region and not failure_override:
            return decision(False, "active_region_commitment")
        improved = (
            new_score > current_score * self.config.preemption_ratio or
            new_score - current_score >
            self.config.preemption_absolute_gain)
        if not improved and not failure_override:
            return decision(False, "insufficient_score_gain")

        self._accept(goal, pose, new_metrics, now, replacing=True)
        return decision(
            True, "current_goal_failed" if failure_override else
            "preemption_margin_met")
