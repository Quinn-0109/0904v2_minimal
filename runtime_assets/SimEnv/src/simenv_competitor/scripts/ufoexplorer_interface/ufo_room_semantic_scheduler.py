#!/usr/bin/env python3
"""Online corridor-side room queue and bounded three-view room missions.

The node never reads truth/layout data.  A room candidate is admitted only
after the observed occupancy projection shows a side aperture followed by a
larger reachable free region.  During a room mission it temporarily owns the
UFO execution path, then returns through the saved entry portal and releases
control back to native UFOExplorer.
"""

import json
import math
import os
import sys
import threading
import time

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from std_msgs.msg import String

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_SCRIPT_DIR = os.path.dirname(SCRIPT_DIR)
for module_dir in (SCRIPT_DIR, PARENT_SCRIPT_DIR):
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

from hierarchical_ros_utils import occupancy_from_message
from lightweight_room_core import LightweightRoomConfig, LightweightRoomScheduler
from planner_interface import AStarPlanner


class UfoRoomSemanticScheduler:
    def __init__(self):
        self._lock = threading.RLock()
        self._frame = rospy.get_param("~world_frame", "map")
        self._output_dir = os.path.abspath(rospy.get_param("~output_dir", "/tmp/ufo"))
        self._log_path = os.path.join(
            self._output_dir, "logs", "ufo_room_semantic_history.jsonl")
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        self._pose = None
        self._grid = None
        self._map_generation = 0
        self._last_detection_generation = -1
        self._corridor = None
        self._active_goal = None
        self._goal_started = None
        self._last_path_publish = -1e9

        config = LightweightRoomConfig(
            enabled=bool(rospy.get_param("~enabled", True)),
            doorway_width_min=float(rospy.get_param("~doorway_width_min", 0.75)),
            doorway_width_max=float(rospy.get_param("~doorway_width_max", 2.30)),
            expansion_margin=float(rospy.get_param("~expansion_margin", 0.80)),
            confirmation_depth=float(rospy.get_param("~confirmation_depth", 1.10)),
            entry_depth=float(rospy.get_param("~entry_depth", 1.50)),
            side_depth=float(rospy.get_param("~side_depth", 3.00)),
            minimum_goal_separation=float(rospy.get_param(
                "~minimum_goal_separation", 1.00)),
            candidate_confirmation_count=int(rospy.get_param(
                "~candidate_confirmation_count", 3)),
            room_budget_seconds=float(rospy.get_param("~room_budget_seconds", 75.0)),
            coverage_minimum_observations=int(rospy.get_param(
                "~coverage_minimum_observations", 3)),
            maximum_center_depth=float(rospy.get_param(
                "~maximum_center_depth", 3.80)),
            maximum_side_lateral=float(rospy.get_param(
                "~maximum_side_lateral", 3.00)),
        )
        config.validate()
        self._scheduler = LightweightRoomScheduler(config)
        self._astar = AStarPlanner(
            clearance=float(rospy.get_param("~astar_clearance", 0.30)),
            reached_tolerance=0.15, maximum_expansions=100000,
            allow_blocked_start=True)
        self._path_pub = rospy.Publisher(
            "/simenv/ufo_room_mission_path", Path, queue_size=2, latch=True)
        self._status_pub = rospy.Publisher(
            "/simenv/ufo_room_mission_status", String, queue_size=10, latch=True)
        rospy.Subscriber(rospy.get_param("~odom_topic", "/simenv/ufo/odometry"),
                         Odometry, self._on_odom, queue_size=20)
        rospy.Subscriber(rospy.get_param(
            "~map_topic", "/simenv/voxel_floor_projection"),
            OccupancyGrid, self._on_map, queue_size=2)
        rospy.Subscriber("/simenv/ufo_corridor_semantics", String,
                         self._on_corridor, queue_size=5)
        rospy.Subscriber("/simenv/goal_execution_result", String,
                         self._on_execution_result, queue_size=10)
        rospy.Subscriber("/simenv/ufoexplorer_replan_request", String,
                         self._on_replan, queue_size=10)
        rospy.Timer(rospy.Duration(0.5), self._on_timer)
        self._publish_status("CORRIDOR_SWEEP", active=False, reason="startup")

    def _record(self, event, **extra):
        snapshot = self._scheduler.snapshot()
        payload = {
            "timestamp": time.time(), "event": event,
            "robot_pose": list(self._pose) if self._pose else None,
            "corridor": self._corridor,
            "active_goal": self._active_goal,
            "room_state": snapshot.get("state"),
            "active_room_id": snapshot.get("active_room_id"),
            "pending_room_candidates": snapshot.get("pending_door_candidates"),
            "successful_roles": snapshot.get("successful_roles"),
            "exit_anchor": (list(self._scheduler.active_door.corridor_side)
                            if self._scheduler.active_door is not None else None),
        }
        payload.update(extra)
        with open(self._log_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")

    def _publish_status(self, state, active=None, **extra):
        if active is None:
            active = self._scheduler.active_door is not None
        payload = {
            "state": state, "active": bool(active),
            "room_id": self._scheduler.active_room_id,
            "active_role": (self._active_goal or {}).get("room_role"),
            "queue_size": len(self._scheduler.pending_candidates),
            "completed_room_count": self._scheduler.snapshot()["complete_room_count"],
        }
        payload.update(extra)
        self._status_pub.publish(String(data=json.dumps(payload, sort_keys=True)))

    def _on_odom(self, message):
        with self._lock:
            self._pose = (message.pose.pose.position.x,
                          message.pose.pose.position.y,
                          message.pose.pose.position.z)

    def _on_map(self, message):
        try:
            grid = occupancy_from_message(message)
        except (IndexError, TypeError, ValueError):
            return
        with self._lock:
            self._grid = grid
            self._map_generation += 1

    def _on_corridor(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        with self._lock:
            self._corridor = payload

    @staticmethod
    def _append_unique(target, points):
        for point in points:
            item = (float(point[0]), float(point[1]))
            if not target or math.hypot(item[0] - target[-1][0],
                                        item[1] - target[-1][1]) > 0.05:
                target.append(item)

    def _plan_goal_path(self, goal):
        start = self._pose[:2]
        checkpoints = list(goal.get("mandatory_portal_waypoints") or [])
        target = goal.get("position") or []
        if len(target) < 2:
            return None
        if not checkpoints or math.hypot(
                float(checkpoints[-1][0]) - float(target[0]),
                float(checkpoints[-1][1]) - float(target[1])) > 0.10:
            checkpoints.append(target[:2])
        points = []
        cursor = start
        for checkpoint in checkpoints:
            result = self._astar.plan(cursor, checkpoint, self._grid)
            if not result.success:
                self._record("ROOM_PATH_REJECTED", checkpoint=list(checkpoint),
                             planner_reason=result.reason)
                return None
            self._append_unique(points, result.path)
            cursor = (float(checkpoint[0]), float(checkpoint[1]))
        if len(points) < 2:
            return None
        message = Path()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = self._frame
        for index, point in enumerate(points):
            pose = PoseStamped()
            pose.header = message.header
            pose.header.seq = index
            pose.pose.position.x, pose.pose.position.y = point
            successor = points[min(index + 1, len(points) - 1)]
            predecessor = points[max(0, index - 1)]
            yaw = math.atan2(successor[1] - predecessor[1],
                             successor[0] - predecessor[0])
            pose.pose.orientation.z = math.sin(0.5 * yaw)
            pose.pose.orientation.w = math.cos(0.5 * yaw)
            message.poses.append(pose)
        return message

    def _issue_goal(self, goal, continuation=False):
        if not continuation:
            self._active_goal = goal
            self._goal_started = time.monotonic()
        path = self._plan_goal_path(goal)
        if path is None:
            self._finish_goal(False, "astar_room_path_unreachable")
            return
        self._last_path_publish = time.monotonic()
        self._path_pub.publish(path)
        self._publish_status(self._scheduler.state, active=True,
                             reason=("local_horizon_continue" if continuation
                                     else "room_goal_issued"),
                             waypoint_count=len(path.poses),
                             target=goal.get("position"))
        self._record("ROOM_PATH_ISSUED", continuation=continuation,
                     path_waypoint_count=len(path.poses),
                     path_role=goal.get("room_role"))

    def _request_next_goal(self):
        if self._grid is None or self._pose is None:
            return
        goal = self._scheduler.next_goal(
            self._grid, self._pose[:2], time.monotonic())
        if goal is None:
            if self._scheduler.active_door is None:
                self._active_goal = None
                self._publish_status("CORRIDOR_RESUME", active=False,
                                     reason="room_mission_released")
            else:
                self._publish_status(self._scheduler.state, active=True,
                                     reason="room_goal_temporarily_unavailable")
            return
        self._issue_goal(goal)

    def _finish_goal(self, success, reason):
        goal = self._active_goal
        if goal is None:
            return
        final_point = self._pose[:2] if self._pose is not None else None
        outcome = self._scheduler.record_result(
            goal, bool(success), final_point, time.monotonic(), reason)
        self._record("ROOM_GOAL_RESULT", outcome=outcome)
        self._active_goal = None
        self._goal_started = None
        if self._scheduler.active_door is None:
            self._publish_status("CORRIDOR_RESUME", active=False,
                                 reason=outcome.get("reason") or reason)
        else:
            self._request_next_goal()

    def _target_reached(self):
        if self._active_goal is None or self._pose is None:
            return False
        target = self._active_goal.get("position") or []
        if len(target) < 2:
            return False
        # Do not release room ownership merely because the robot is near the
        # corridor-side exit anchor.  Goal Executor's normal tolerance is
        # 0.15 m; 0.25 m only absorbs odometry/callback timing skew.
        tolerance = 0.25
        return math.hypot(self._pose[0] - float(target[0]),
                          self._pose[1] - float(target[1])) <= tolerance

    def _on_execution_result(self, message):
        try:
            result = json.loads(message.data)
        except (TypeError, ValueError):
            return
        with self._lock:
            if self._active_goal is None:
                return
            if result.get("success") and self._target_reached():
                self._finish_goal(True, result.get("reason", "goal_reached"))
            elif not result.get("success"):
                self._finish_goal(False, result.get("reason", "executor_failed"))

    def _on_replan(self, message):
        with self._lock:
            if self._active_goal is None:
                return
            reason = message.data or "REPLAN_REQUESTED"
            if reason == "LOCAL_PATH_COMPLETE":
                if self._target_reached():
                    self._finish_goal(True, "room_macro_target_reached")
                else:
                    self._issue_goal(self._active_goal, continuation=True)
            elif reason in ("EXECUTOR_PATH_FAILED", "SCAN_REJECTED",
                            "UFO_PATH_BLOCKED"):
                self._finish_goal(False, reason)

    def _on_timer(self, _event):
        with self._lock:
            if self._pose is None or self._grid is None:
                return
            if self._active_goal is not None:
                if self._target_reached():
                    self._finish_goal(True, "target_reached_by_odometry")
                    return
                timeout = float(self._active_goal.get(
                    "execution_timeout_sec", 30.0))
                if (self._goal_started is not None and
                        time.monotonic() - self._goal_started > timeout):
                    self._finish_goal(False, "room_macro_goal_timeout")
                return
            if self._scheduler.active_door is not None:
                self._request_next_goal()
                return
            corridor = self._corridor or {}
            # Detection is evaluated once per actual map projection, so three
            # confirmations mean three independent LiDAR/map observations.
            if (not corridor.get("confirmed") or
                    not corridor.get("robot_inside") or
                    self._map_generation == self._last_detection_generation):
                return
            heading = corridor.get("heading")
            if heading is None:
                return
            self._last_detection_generation = self._map_generation
            door = self._scheduler.consider_corridor_door(
                self._grid, self._pose[:2], float(heading), time.monotonic())
            if door is None:
                self._publish_status(self._scheduler.state, active=False,
                                     reason="side_room_scan")
                return
            self._record("SIDE_ROOM_QUEUED", doorway=door.to_dict(),
                         evidence="side_aperture_plus_reachable_expansion")
            self._publish_status("ROOM_ENTRY", active=True,
                                 reason="confirmed_side_room_queued",
                                 doorway=door.to_dict())
            self._request_next_goal()


if __name__ == "__main__":
    rospy.init_node("ufo_room_semantic_scheduler")
    UfoRoomSemanticScheduler()
    rospy.spin()
