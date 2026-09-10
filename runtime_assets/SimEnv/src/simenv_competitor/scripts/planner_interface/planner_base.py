"""Planner abstraction kept independent from exploration goal selection."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

Point = Tuple[float, float]


@dataclass
class PlannerResult:
    success: bool
    path: List[Point] = field(default_factory=list)
    travel_cost: float = float("inf")
    reason: str = "not_planned"
    metadata: Dict[str, Any] = field(default_factory=dict)


class PlannerBase(ABC):
    """Common contract for A*, FAR, or another future route planner."""

    @abstractmethod
    def plan(self, start: Sequence[float], goal: Sequence[float],
             occupancy_map: Any) -> PlannerResult:
        raise NotImplementedError

    def estimate_cost(self, start: Sequence[float], goal: Sequence[float],
                      occupancy_map: Any) -> float:
        result = self.plan(start, goal, occupancy_map)
        return result.travel_cost if result.success else float("inf")

    def as_nav_path(self, result: PlannerResult, frame_id: str,
                    stamp: Optional[Any] = None) -> Any:
        """Convert a result to nav_msgs/Path when ROS messages are available."""
        try:
            import rospy
            from geometry_msgs.msg import PoseStamped
            from nav_msgs.msg import Path
        except ImportError:
            return {
                "frame_id": frame_id,
                "path": [[float(point[0]), float(point[1])]
                         for point in result.path],
            }
        message = Path()
        message.header.frame_id = frame_id
        message.header.stamp = stamp or rospy.Time.now()
        for index, point in enumerate(result.path):
            pose = PoseStamped()
            pose.header = message.header
            pose.header.seq = index
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.orientation.w = 1.0
            message.poses.append(pose)
        return message
