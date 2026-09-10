"""A* implementation of the hierarchical planner contract."""

import math
from typing import Sequence

from baseline_planning_core import astar_safe_path

from .planner_base import PlannerBase, PlannerResult


class AStarPlanner(PlannerBase):
    def __init__(self, clearance: float = 0.30,
                 reached_tolerance: float = 0.15,
                 maximum_expansions: int = 100000,
                 allow_blocked_start: bool = False):
        self.clearance = float(clearance)
        self.reached_tolerance = float(reached_tolerance)
        self.maximum_expansions = max(0, int(maximum_expansions))
        self.allow_blocked_start = bool(allow_blocked_start)

    def plan(self, start: Sequence[float], goal: Sequence[float],
             occupancy_map) -> PlannerResult:
        raw = astar_safe_path(
            occupancy_map,
            (float(start[0]), float(start[1])),
            (float(goal[0]), float(goal[1])),
            self.clearance, self.reached_tolerance,
            allow_blocked_start=self.allow_blocked_start,
            maximum_expansions=self.maximum_expansions)
        path = [(float(point[0]), float(point[1]))
                for point in raw.get("path", [])]
        cost = sum(math.hypot(b[0] - a[0], b[1] - a[1])
                   for a, b in zip(path[:-1], path[1:]))
        if raw.get("success") and len(path) == 1:
            cost = math.hypot(
                float(goal[0]) - float(start[0]),
                float(goal[1]) - float(start[1]))
        return PlannerResult(
            success=bool(raw.get("success")),
            path=path,
            travel_cost=cost if raw.get("success") else float("inf"),
            reason=str(raw.get("reason", "astar_failed")),
            metadata={
                "backend": "astar",
                "expansions": int(raw.get("expansions", 0)),
                "allow_blocked_start": self.allow_blocked_start,
            },
        )
