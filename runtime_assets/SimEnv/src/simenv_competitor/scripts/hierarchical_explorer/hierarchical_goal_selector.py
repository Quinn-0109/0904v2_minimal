"""Region-level goal selection inspired by TARE's hierarchy."""

from dataclasses import asdict, dataclass
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from exploration_graph import ExplorationGraph, ExplorationNode
from planner_interface import PlannerBase, PlannerResult

Point = Tuple[float, float]


@dataclass
class SelectorConfig:
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 1.5
    revisit_lambda: float = 1.0
    coverage_threshold: float = 0.90
    unknown_ratio_threshold: float = 0.05
    local_coverage_radius: float = 6.0
    region_enter_tolerance: float = 1.5
    entry_return_tolerance: float = 0.50
    minimum_information_gain: float = 0.20
    minimum_new_region_seconds: float = 35.0
    nominal_speed: float = 0.30
    recent_goal_distance: float = 0.75
    failure_cooldown_seconds: float = 30.0

    def validate(self) -> None:
        if not (0.0 < self.coverage_threshold <= 1.0):
            raise ValueError("invalid coverage threshold")
        if not (0.0 <= self.unknown_ratio_threshold < 1.0):
            raise ValueError("invalid unknown ratio threshold")
        if min(
                self.local_coverage_radius, self.region_enter_tolerance,
                self.entry_return_tolerance, self.minimum_new_region_seconds,
                self.nominal_speed, self.recent_goal_distance,
                self.failure_cooldown_seconds) <= 0.0:
            raise ValueError("invalid hierarchical selector configuration")


@dataclass
class GoalDecision:
    kind: str
    goal: Optional[Point]
    region_id: Optional[str] = None
    frontier_id: Optional[str] = None
    score: Optional[float] = None
    information_gain: float = 0.0
    travel_cost: float = float("inf")
    connectivity_gain: float = 0.0
    revisit_penalty: float = 0.0
    coverage_ratio: float = 0.0
    reason: str = ""
    path_result: Optional[PlannerResult] = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["goal"] = list(self.goal) if self.goal is not None else None
        if self.path_result is not None:
            payload["path_result"] = {
                "success": self.path_result.success,
                "path": [list(point) for point in self.path_result.path],
                "travel_cost": self.path_result.travel_cost,
                "reason": self.path_result.reason,
                "metadata": self.path_result.metadata,
            }
        return payload


class HierarchicalGoalSelector:
    def __init__(self, planner: PlannerBase,
                 config: SelectorConfig = None):
        self.planner = planner
        self.config = config or SelectorConfig()
        self.config.validate()
        self.active_region_id: Optional[str] = None
        self.region_entry_pose: Optional[Point] = None
        self.region_enter_time: Optional[float] = None
        self.latest_coverage = 0.0
        self.returning_to_entry = False
        self.region_entry_confirmed = False
        self.local_goal_count = 0
        self.recent_goals: List[Point] = []
        self.failed_regions: Dict[str, Tuple[int, float]] = {}

    def local_coverage(self, grid, center: Sequence[float]) -> dict:
        cell = grid.world_to_cell((float(center[0]), float(center[1])))
        if cell is None:
            return {
                "coverage_ratio": 0.0, "known_cells": 0,
                "unknown_cells": 0, "domain_cells": 0,
            }
        radius = max(1, int(math.ceil(
            self.config.local_coverage_radius / grid.resolution)))
        known = unknown = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if math.hypot(dx, dy) > radius:
                    continue
                x, y = cell[0] + dx, cell[1] + dy
                if not (0 <= x < grid.width and 0 <= y < grid.height):
                    continue
                value = int(grid.data[y, x])
                if value == 0:
                    known += 1
                elif value < 0:
                    unknown += 1
        domain = known + unknown
        return {
            "coverage_ratio": known / float(domain) if domain else 0.0,
            "known_cells": known,
            "unknown_cells": unknown,
            "domain_cells": domain,
            "known_area": known * grid.resolution ** 2,
            "unknown_area": unknown * grid.resolution ** 2,
        }

    def _connectivity_gain(self, graph: ExplorationGraph,
                           region: ExplorationNode) -> float:
        gain = 0.0
        for node_id in graph.neighbours(region.id):
            node = graph.nodes.get(node_id)
            if node is None:
                continue
            if node.type == "corridor":
                gain += 0.5
                for second_id in graph.neighbours(node.id):
                    second = graph.nodes.get(second_id)
                    if (second is not None and
                            second.type == "region" and
                            not second.visited):
                        gain += 1.0
            elif node.type == "frontier_cluster":
                gain += min(2.0, node.information_gain)
        return gain

    def _score_region(self, graph: ExplorationGraph,
                      region: ExplorationNode,
                      robot_pose: Sequence[float], grid
                      ) -> Optional[GoalDecision]:
        goal = region.goal or region.position
        path = self.planner.plan(robot_pose, goal, grid)
        if not path.success:
            return None
        connectivity = self._connectivity_gain(graph, region)
        failure_count = self.failed_regions.get(
            region.id, (0, 0.0))[0]
        revisit = float(region.visited_count + failure_count)
        score = (
            self.config.alpha * float(region.information_gain) -
            self.config.beta * path.travel_cost +
            self.config.gamma * connectivity -
            self.config.revisit_lambda * revisit)
        frontier_id = None
        if region.frontier_ids:
            frontier_id = max(
                region.frontier_ids,
                key=lambda item: (
                    graph.nodes[item].information_gain
                    if item in graph.nodes else 0.0))
        return GoalDecision(
            kind="enter_region",
            goal=(float(goal[0]), float(goal[1])),
            region_id=region.id,
            frontier_id=frontier_id,
            score=score,
            information_gain=region.information_gain,
            travel_cost=path.travel_cost,
            connectivity_gain=connectivity,
            revisit_penalty=revisit,
            coverage_ratio=region.coverage_ratio,
            reason="highest_region_score",
            path_result=path,
        )

    def _recent_duplicate(self, goal: Sequence[float]) -> bool:
        return any(
            math.hypot(float(goal[0]) - point[0],
                       float(goal[1]) - point[1]) <
            self.config.recent_goal_distance
            for point in self.recent_goals[-20:])

    def _local_region_goal(self, graph: ExplorationGraph,
                           region: ExplorationNode,
                           robot_pose: Sequence[float], grid
                           ) -> Optional[GoalDecision]:
        failure = self.failed_regions.get(region.id)
        if (failure is not None and
                time.monotonic() - failure[1] <
                self.config.failure_cooldown_seconds):
            return None
        candidates = []
        for frontier_id in region.frontier_ids:
            frontier = graph.nodes.get(frontier_id)
            if (frontier is None or not frontier.reachable or
                    frontier.goal is None or
                    self._recent_duplicate(frontier.goal)):
                continue
            path = self.planner.plan(robot_pose, frontier.goal, grid)
            if not path.success:
                continue
            score = (
                self.config.alpha * frontier.information_gain -
                self.config.beta * path.travel_cost)
            candidates.append((score, frontier, path))
        if not candidates:
            return None
        score, frontier, path = max(candidates, key=lambda item: item[0])
        return GoalDecision(
            kind="local_region_frontier",
            goal=frontier.goal,
            region_id=region.id,
            frontier_id=frontier.id,
            score=score,
            information_gain=frontier.information_gain,
            travel_cost=path.travel_cost,
            coverage_ratio=self.latest_coverage,
            reason="active_region_uncovered_frontier",
            path_result=path,
        )

    def _return_entry(self, robot_pose: Sequence[float], grid,
                      reason: str) -> GoalDecision:
        assert self.region_entry_pose is not None
        path = self.planner.plan(
            robot_pose, self.region_entry_pose, grid)
        return GoalDecision(
            kind="return_region_entry",
            goal=self.region_entry_pose,
            region_id=self.active_region_id,
            coverage_ratio=self.latest_coverage,
            travel_cost=path.travel_cost,
            reason=reason,
            path_result=path,
        )

    def select(self, graph: ExplorationGraph,
               robot_pose: Sequence[float], grid,
               remaining_seconds: float) -> GoalDecision:
        unknown_ratio = graph.unknown_ratio(grid)
        if unknown_ratio <= self.config.unknown_ratio_threshold:
            return GoalDecision(
                "return_home", None, reason="unknown_ratio_complete")
        if self.active_region_id is not None:
            region = graph.nodes.get(self.active_region_id)
            if region is None:
                self.active_region_id = None
                self.region_entry_pose = None
                self.returning_to_entry = False
            else:
                if (self.region_entry_pose is not None and
                        math.hypot(
                            float(robot_pose[0]) -
                            self.region_entry_pose[0],
                            float(robot_pose[1]) -
                            self.region_entry_pose[1]) >=
                        self.config.region_enter_tolerance):
                    self.region_entry_confirmed = True
                coverage = self.local_coverage(grid, region.position)
                self.latest_coverage = max(
                    region.coverage_ratio,
                    float(coverage["coverage_ratio"]))
                exhausted = not region.frontier_ids
                coverage_complete = (
                    self.latest_coverage >=
                    self.config.coverage_threshold and
                    self.region_entry_confirmed and
                    self.local_goal_count >= 1)
                if coverage_complete or exhausted:
                    if (self.region_entry_pose is not None and
                            math.hypot(
                                float(robot_pose[0]) -
                                self.region_entry_pose[0],
                                float(robot_pose[1]) -
                                self.region_entry_pose[1]) >
                            self.config.entry_return_tolerance):
                        self.returning_to_entry = True
                        return self._return_entry(
                            robot_pose, grid,
                            ("region_coverage_complete"
                             if coverage_complete else
                             "region_frontier_exhausted"))
                    completed_id = self.active_region_id
                    self.active_region_id = None
                    self.region_entry_pose = None
                    self.returning_to_entry = False
                    return GoalDecision(
                        "region_complete", None,
                        region_id=completed_id,
                        coverage_ratio=self.latest_coverage,
                        reason=(
                            "entry_reached_after_coverage"
                            if coverage_complete else
                            "entry_reached_after_frontier_exhausted"))
                local = self._local_region_goal(
                    graph, region, robot_pose, grid)
                if local is not None:
                    return local
                self.returning_to_entry = True
                return self._return_entry(
                    robot_pose, grid, "no_local_frontier")
        candidates = [
            node for node in graph.nodes.values()
            if (node.type == "region" and not node.visited and
                node.reachable and node.goal is not None and
                (node.id not in self.failed_regions or
                 time.monotonic() -
                 self.failed_regions[node.id][1] >=
                 self.config.failure_cooldown_seconds) and
                node.information_gain >=
                self.config.minimum_information_gain)
        ]
        decisions = [
            decision for decision in (
                self._score_region(graph, node, robot_pose, grid)
                for node in candidates)
            if decision is not None
        ]
        if not decisions:
            return GoalDecision(
                "return_home", None,
                reason="no_reachable_unvisited_region")
        selected = max(
            decisions,
            key=lambda decision: float(decision.score))
        required = (
            selected.travel_cost / self.config.nominal_speed +
            self.config.minimum_new_region_seconds)
        if remaining_seconds < required:
            return GoalDecision(
                "return_home", None,
                reason="insufficient_time_for_new_region",
                travel_cost=selected.travel_cost)
        return selected

    def record_goal_result(self, decision: GoalDecision,
                           success: bool,
                           robot_pose: Sequence[float], now: float,
                           start_pose: Optional[Sequence[float]] = None) -> dict:
        update = {
            "region_id": decision.region_id,
            "coverage_ratio": decision.coverage_ratio,
            "entered": False,
            "exited": False,
            "success": bool(success),
            "time": float(now),
            "coverage_eligible": self.region_entry_confirmed,
        }
        if success and decision.goal is not None:
            self.recent_goals.append(
                (float(decision.goal[0]), float(decision.goal[1])))
        if (not success and decision.region_id is not None):
            count = self.failed_regions.get(
                decision.region_id, (0, 0.0))[0]
            self.failed_regions[decision.region_id] = (
                count + 1, time.monotonic())
        if success and decision.kind == "enter_region":
            self.active_region_id = decision.region_id
            self.region_entry_pose = (
                float(robot_pose[0]), float(robot_pose[1]))
            self.region_enter_time = float(now)
            self.latest_coverage = decision.coverage_ratio
            self.returning_to_entry = False
            self.region_entry_confirmed = False
            self.local_goal_count = 0
            update["entered"] = True
        elif success and decision.kind == "local_region_frontier":
            self.local_goal_count += 1
        elif success and decision.kind == "return_region_entry":
            update["exited"] = True
            update["coverage_ratio"] = self.latest_coverage
            update["coverage_eligible"] = self.region_entry_confirmed
            self.active_region_id = None
            self.region_entry_pose = None
            self.returning_to_entry = False
            self.region_entry_confirmed = False
            self.local_goal_count = 0
        return update

    def state(self) -> dict:
        return {
            "active_region_id": self.active_region_id,
            "region_entry_pose": (
                list(self.region_entry_pose)
                if self.region_entry_pose is not None else None),
            "region_enter_time": self.region_enter_time,
            "latest_coverage": self.latest_coverage,
            "returning_to_entry": self.returning_to_entry,
            "region_entry_confirmed": self.region_entry_confirmed,
            "local_goal_count": self.local_goal_count,
            "recent_goal_count": len(self.recent_goals),
            "failed_regions": {
                region_id: count
                for region_id, (count, _) in self.failed_regions.items()
            },
        }
