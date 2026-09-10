"""FAR-compatible planner boundary.

The current backend intentionally delegates to A*.  Future FAR integration
only needs to replace ``plan`` while exploration graph and selector APIs stay
unchanged.
"""

from typing import Sequence

from .astar_planner import AStarPlanner
from .planner_base import PlannerBase, PlannerResult


class FarPlannerInterface(PlannerBase):
    def __init__(self, fallback: AStarPlanner = None,
                 backend: str = "astar"):
        self.fallback = fallback or AStarPlanner()
        self.backend = str(backend)

    def plan(self, start: Sequence[float], goal: Sequence[float],
             occupancy_map) -> PlannerResult:
        result = self.fallback.plan(start, goal, occupancy_map)
        result.metadata.update({
            "interface": "far_compatible",
            "requested_backend": self.backend,
            "active_backend": "astar",
        })
        return result

    def plan_nav_path(self, current_pose: Sequence[float],
                      goal_pose: Sequence[float], occupancy_map,
                      frame_id: str = "camera_init"):
        result = self.plan(current_pose, goal_pose, occupancy_map)
        return result, self.as_nav_path(result, frame_id)
