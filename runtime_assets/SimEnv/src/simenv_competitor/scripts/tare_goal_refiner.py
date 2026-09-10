#!/usr/bin/env python3
"""Validate official TARE goals without changing exploration semantics.

TARE is the only exploration decision maker.  This node either forwards its
arbitrated waypoint unchanged or makes a small collision-only endpoint
repair.  It never selects another exploration-path vertex and never creates
corridor, doorway, room-entry, return, frontier, or recovery goals.
"""

import json
import math
import os
import sys
import threading

import rospy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid, Odometry

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from hierarchical_ros_utils import atomic_json, occupancy_from_message
from planner_interface import AStarPlanner
from simenv_competitor.srv import CheckTwinCylinder, CheckTwinCylinderRequest
from tare_goal_refinement_core import (
    TareGoalSafetyConfig, goal_has_no_known_collision, goal_is_observed_free,
    locally_shift_goal_to_free_space)


class TareGoalRefiner:
    """A safety filter, not an exploration planner."""

    def __init__(self):
        self._lock = threading.RLock()
        self._grid = None
        self._pose = None
        self._history = []
        self._sequence = 0
        self._last_logged = None

        self._output_dir = os.path.abspath(rospy.get_param(
            "~output_dir", "/tmp/simenv/tare_far"))
        self._log_path = os.path.join(
            self._output_dir, "logs", "tare_goal_refinement.json")
        self._config = TareGoalSafetyConfig(
            goal_clearance=float(rospy.get_param(
                "~goal_clearance", 0.32)),
            local_shift_max=float(rospy.get_param(
                "~local_shift_max", 0.75)),
            local_shift_step=float(rospy.get_param(
                "~local_shift_step", 0.15)))
        self._log_goal_distance = float(rospy.get_param(
            "~log_goal_distance", 0.10))
        self._allow_unknown_tare_goal = bool(rospy.get_param(
            "~allow_unknown_tare_goal", True))
        self._astar = AStarPlanner(
            clearance=float(rospy.get_param("~astar_clearance", 0.30)),
            reached_tolerance=float(rospy.get_param(
                "~astar_reached_tolerance", 0.15)),
            maximum_expansions=int(rospy.get_param(
                "~astar_maximum_expansions", 100000)),
            allow_blocked_start=bool(rospy.get_param(
                "~astar_allow_blocked_start", True)))

        self._scan_service = rospy.ServiceProxy(
            rospy.get_param(
                "~scan_service",
                "/tare_far_voxel_mapper/check_twin_cylinder"),
            CheckTwinCylinder)
        self._publisher = rospy.Publisher(
            rospy.get_param(
                "~refined_goal_topic", "/refined_exploration_goal"),
            PointStamped, queue_size=5)

        rospy.Subscriber(
            rospy.get_param("~tare_goal_topic", "/tare/way_point"),
            PointStamped, self._on_goal, queue_size=5)
        rospy.Subscriber(
            rospy.get_param(
                "~map_topic", "/simenv/voxel_floor_projection"),
            OccupancyGrid, self._on_map, queue_size=1)
        rospy.Subscriber(
            rospy.get_param(
                "~odom_topic", "/tare/state_estimation_at_scan"),
            Odometry, self._on_odom, queue_size=20)
        rospy.on_shutdown(self._write_log)
        self._write_log()
        rospy.loginfo(
            "TARE goal safety filter ready: input=%s output=%s "
            "policy=validation_and_local_repair_only",
            rospy.resolve_name(rospy.get_param(
                "~tare_goal_topic", "/tare/way_point")),
            rospy.resolve_name(rospy.get_param(
                "~refined_goal_topic", "/refined_exploration_goal")))

    def _on_map(self, message):
        try:
            grid = occupancy_from_message(message)
        except (IndexError, TypeError, ValueError):
            return
        with self._lock:
            self._grid = grid

    def _on_odom(self, message):
        point = message.pose.pose.position
        with self._lock:
            self._pose = (
                float(point.x), float(point.y), float(point.z))

    def _voxel_safe(self, point, robot_pose):
        yaw = math.atan2(
            point[1] - robot_pose[1], point[0] - robot_pose[0])
        request = CheckTwinCylinderRequest()
        request.pose_x, request.pose_y = float(point[0]), float(point[1])
        request.pose_z = float(robot_pose[2])
        request.yaw = yaw
        request.front_offset = 0.06675
        request.rear_offset = -0.06675
        request.radius = 0.11772 + 0.08
        request.min_height = -0.057 - 0.08
        request.max_height = 0.057 + 0.08
        request.clearance_search_radius = request.radius + 0.50
        try:
            response = self._scan_service(request)
        except rospy.ServiceException:
            return False, "voxel_service_unavailable"
        if not response.map_available:
            return False, "voxel_map_unavailable"
        if response.occupied_collision:
            return False, "voxel_collision"
        if response.unknown_queries:
            return True, "voxel_safe_unknown_overlap"
        return True, "voxel_safe"

    def _safe(self, grid, point, pose, allow_unknown=False):
        occupancy_safe = (
            goal_has_no_known_collision(
                grid, point, self._config.goal_clearance)
            if allow_unknown else
            goal_is_observed_free(
                grid, point, self._config.goal_clearance))
        if not occupancy_safe:
            return False, (
                "occupancy_known_collision_or_outside_map"
                if allow_unknown else "occupancy_not_observed_free")
        return self._voxel_safe(point, pose)

    def _reachable(self, grid, point, pose):
        result = self._astar.plan(pose, point, grid)
        return bool(result.success), str(result.reason)

    def _safe_and_reachable(self, grid, point, pose):
        reachable, route = self._reachable(grid, point, pose)
        if not reachable:
            return False, route
        return self._safe(grid, point, pose)

    def _publish(self, original, goal):
        output = PointStamped()
        output.header = original.header
        output.header.stamp = rospy.Time.now()
        output.point.x = float(goal[0])
        output.point.y = float(goal[1])
        output.point.z = float(goal[2])
        self._publisher.publish(output)

    def _on_goal(self, message):
        original = (float(message.point.x), float(message.point.y),
                    float(message.point.z))
        with self._lock:
            grid, pose = self._grid, self._pose

        record = {
            "tare_original_goal": list(original),
            "refined_goal": None,
            "reason": "goal_rejected",
            "refined": False,
            "published": False,
            "source": "official_tare_waypoint",
        }
        if grid is None or pose is None:
            record["reason"] = "validation_inputs_unavailable"
        else:
            valid, validation = self._safe(
                grid, original, pose,
                allow_unknown=self._allow_unknown_tare_goal)
            record["original_validation"] = validation
            reachable, reachability = (
                self._reachable(grid, original, pose)
                if valid else (False, "safety_rejected"))
            record["original_reachability"] = reachability
            if valid and reachable:
                record.update({
                    "refined_goal": list(original),
                        "reason": "tare_goal_valid_forwarded",
                        "published": True,
                        "unknown_endpoint_policy": (
                            "allow_only_if_astar_reachable_then_far_scan_lite"
                            if self._allow_unknown_tare_goal else
                            "require_observed_free"),
                })
            else:
                shifted = locally_shift_goal_to_free_space(
                    grid, pose, original, self._config,
                    validator=lambda point: self._safe_and_reachable(
                        grid, point, pose)[0])
                if shifted is not None:
                    record.update({
                        "refined_goal": list(shifted),
                        "reason": "tare_goal_local_free_space_shift",
                        "refined": True,
                        "published": True,
                        "shift_distance_m": math.hypot(
                            shifted[0] - original[0],
                            shifted[1] - original[1]),
                        "source": "official_tare_waypoint_local_repair",
                    })
                else:
                    record["reason"] = "no_valid_local_free_space_repair"

        if record["published"]:
            self._publish(message, record["refined_goal"])
        self._record(record, pose)

    def _record(self, record, pose):
        key = (
            tuple(round(value, 2)
                  for value in record["tare_original_goal"][:2]),
            tuple(round(value, 2)
                  for value in (record["refined_goal"] or [])[:2]),
            record["reason"])
        if self._last_logged is not None:
            original_delta = math.hypot(
                key[0][0] - self._last_logged[0][0],
                key[0][1] - self._last_logged[0][1])
            if (original_delta < self._log_goal_distance and
                    key[1:] == self._last_logged[1:]):
                return
        self._last_logged = key
        self._sequence += 1
        record = dict(record)
        record.update({
            "sequence": self._sequence,
            "stamp": rospy.Time.now().to_sec(),
            "robot_pose": list(pose) if pose is not None else None,
        })
        with self._lock:
            self._history.append(record)
            self._write_log()
        rospy.loginfo(
            "TARE goal safety result: %s -> %s (%s)",
            record["tare_original_goal"][:2],
            ((record["refined_goal"] or [])[:2]), record["reason"])

    def _write_log(self):
        atomic_json(self._log_path, {
            "schema": "simenv_tare_goal_safety_filter_v2",
            "policy": (
                "TARE owns exploration semantics; only endpoint validation, "
                "and local free-space repair are permitted"),
            "records": list(self._history),
        })


if __name__ == "__main__":
    rospy.init_node("tare_goal_refiner")
    TareGoalRefiner()
    rospy.spin()
