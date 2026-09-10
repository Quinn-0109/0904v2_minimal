#!/usr/bin/env python3
"""Commit safe local prefixes of UFOExplorer paths to FAR/Goal Executor."""

import json
import math
import os
import sys
import threading
import time

import rospy
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from std_msgs.msg import String

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_SCRIPT_DIR = os.path.dirname(SCRIPT_DIR)
for module_dir in (SCRIPT_DIR, PARENT_SCRIPT_DIR):
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

from hierarchical_ros_utils import occupancy_from_message
from simenv_competitor.srv import CheckTwinCylinder, CheckTwinCylinderRequest
from ufo_path_core import (crop_horizon, distance, resample_path,
                           select_execution_waypoints, trim_path_to_pose)


class UfoPathAdapter:
    def __init__(self):
        self._lock = threading.RLock()
        self._output_dir = os.path.abspath(rospy.get_param("~output_dir", "/tmp/ufo"))
        self._log_path = os.path.join(
            self._output_dir, "logs", "ufoexplorer_goal_history.jsonl")
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        self._frame = rospy.get_param("~world_frame", "map")
        self._spacing = float(rospy.get_param("~waypoint_spacing", 0.45))
        self._min_distance = float(rospy.get_param(
            "~min_effective_waypoint_distance", 0.6))
        self._horizon = float(rospy.get_param(
            "~max_local_execution_horizon", 3.0))
        self._pose = None
        self._grid = None
        self._waypoints = []
        self._active = None
        self._pending_path = None
        self._path_id = 0
        self._goal_pub = rospy.Publisher(
            rospy.get_param("~output_goal_topic", "/ufo_local_exploration_goal"),
            PointStamped, queue_size=2)
        self._local_path_pub = rospy.Publisher(
            "/simenv/ufo_local_execution_path", Path, queue_size=2, latch=True)
        self._waypoint_pub = rospy.Publisher(
            "/simenv/ufo_executed_waypoint", PoseStamped, queue_size=10)
        self._status_pub = rospy.Publisher(
            "/simenv/ufo_path_adapter_status", String, queue_size=10, latch=True)
        self._replan_pub = rospy.Publisher(
            "/simenv/ufoexplorer_replan_request", String, queue_size=10)
        self._scan = rospy.ServiceProxy(rospy.get_param(
            "~scan_service", "/tare_far_voxel_mapper/check_twin_cylinder"),
            CheckTwinCylinder)
        rospy.Subscriber("/simenv/ufo_exploration_path",
                         Path, self._on_path, queue_size=2)
        rospy.Subscriber(rospy.get_param("~odom_topic", "/simenv/ufo/odometry"),
                         Odometry, self._on_odom, queue_size=20)
        rospy.Subscriber(rospy.get_param(
            "~map_topic", "/simenv/voxel_floor_projection"),
            OccupancyGrid, self._on_map, queue_size=2)
        rospy.Subscriber("/simenv/goal_execution_result",
                         String, self._on_result, queue_size=20)
        rospy.Timer(rospy.Duration(0.5), self._on_timer)

    def _on_timer(self, _event):
        with self._lock:
            if (self._pending_path is not None and self._pose is not None and
                    self._grid is not None and self._active is None and
                    not self._waypoints):
                pending, self._pending_path = self._pending_path, None
                self._on_path(pending)

    def _status(self, state, **extra):
        payload = dict(state=state, path_id=self._path_id,
                       remaining_waypoints=len(self._waypoints),
                       active_waypoint=self._active)
        payload.update(extra)
        self._status_pub.publish(String(data=json.dumps(payload, sort_keys=True)))

    def _record(self, **payload):
        payload.update(timestamp=time.time(), selected_path_id=self._path_id,
                       robot_pose=list(self._pose) if self._pose else None)
        with open(self._log_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")

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

    def _is_grid_free(self, point):
        if self._grid is None:
            return False
        cell = self._grid.world_to_cell(point)
        return cell is not None and int(self._grid.data[cell[1], cell[0]]) == 0

    def _repair_free(self, point):
        if self._is_grid_free(point):
            return point
        if self._grid is None:
            return None
        step = self._grid.resolution
        for radius_cells in range(1, max(2, int(0.6 / step)) + 1):
            candidates = []
            for dx in range(-radius_cells, radius_cells + 1):
                for dy in (-radius_cells, radius_cells):
                    candidates.append((point[0] + dx * step,
                                       point[1] + dy * step))
            for dy in range(-radius_cells + 1, radius_cells):
                for dx in (-radius_cells, radius_cells):
                    candidates.append((point[0] + dx * step,
                                       point[1] + dy * step))
            for candidate in candidates:
                if self._is_grid_free(candidate):
                    return candidate
        return None

    def _scan_safe(self, point, yaw):
        request = CheckTwinCylinderRequest()
        request.pose_x, request.pose_y = point[0], point[1]
        request.pose_z = self._pose[2] if self._pose else 0.45
        request.yaw = yaw
        request.front_offset, request.rear_offset = 0.20, -0.20
        request.radius = 0.32
        request.min_height, request.max_height = -0.25, 0.55
        request.clearance_search_radius = 0.55
        try:
            response = self._scan(request)
            return bool(response.map_available and not response.occupied_collision), response.status
        except rospy.ServiceException:
            return False, "SCAN_SERVICE_ERROR"

    def _make_path(self, waypoints, header):
        message = Path()
        message.header = header
        for x, y, yaw in waypoints:
            pose = PoseStamped()
            pose.header = header
            pose.pose.position.x, pose.pose.position.y = x, y
            pose.pose.orientation.z = math.sin(yaw * 0.5)
            pose.pose.orientation.w = math.cos(yaw * 0.5)
            message.poses.append(pose)
        return message

    def _on_path(self, message):
        with self._lock:
            if self._pose is None or self._grid is None:
                self._pending_path = message
                self._status("WAITING_FOR_POSE_OR_MAP")
                return
            if self._active is not None or self._waypoints:
                self._pending_path = message
                self._status("PATH_COMMIT", reason="current_local_path_active")
                return
            points = [(p.pose.position.x, p.pose.position.y)
                      for p in message.poses]
            trimmed = trim_path_to_pose(points, self._pose[:2])
            local = crop_horizon(resample_path(trimmed, self._spacing),
                                 self._horizon)
            candidates = select_execution_waypoints(
                local, self._pose[:2], self._min_distance)
            if not candidates:
                self._path_id += 1
                self._status("PATH_TOO_SHORT", reason="WAYPOINT_TOO_CLOSE")
                self._record(ufo_raw_output_type="path",
                             ufo_raw_path=[list(p) for p in points],
                             trimmed_path=[list(p) for p in trimmed],
                             ufo_raw_path_length=sum(distance(a, b)
                                 for a, b in zip(points[:-1], points[1:])),
                             trimmed_path_length=sum(distance(a, b)
                                 for a, b in zip(trimmed[:-1], trimmed[1:])),
                             selected_waypoints=[], scan_lite_status="NOT_RUN",
                             executor_status="NOT_SENT", fallback_used=False,
                             reason="WAYPOINT_TOO_CLOSE", rejected_waypoints=[])
                self._replan_pub.publish(String(data="UFO_PATH_TOO_SHORT"))
                return
            safe = []
            rejected = []
            for x, y, yaw in candidates:
                repaired = self._repair_free((x, y))
                if repaired is None:
                    rejected.append(dict(point=[x, y], reason="NOT_FREE"))
                    break
                safe_status, reason = self._scan_safe(repaired, yaw)
                if not safe_status:
                    rejected.append(dict(point=list(repaired), reason=reason))
                    break
                safe.append((repaired[0], repaired[1], yaw))
            self._path_id += 1
            if not safe:
                self._status("SCAN_REJECTED", rejected=rejected)
                self._record(ufo_raw_output_type="path",
                             ufo_raw_path=[list(p) for p in points],
                             trimmed_path=[list(p) for p in trimmed],
                             ufo_raw_path_length=sum(distance(a, b)
                                 for a, b in zip(points[:-1], points[1:])),
                             trimmed_path_length=sum(distance(a, b)
                                 for a, b in zip(trimmed[:-1], trimmed[1:])),
                             selected_waypoints=[], scan_lite_status="REJECTED",
                             executor_status="NOT_SENT", fallback_used=False,
                             reason="SCAN_REJECTED", rejected_waypoints=rejected)
                self._replan_pub.publish(String(data="SCAN_REJECTED"))
                return
            self._waypoints = safe
            header = message.header
            header.frame_id = header.frame_id or self._frame
            self._local_path_pub.publish(self._make_path(safe, header))
            self._record(ufo_raw_output_type="path",
                         ufo_raw_path=[list(p) for p in points],
                         trimmed_path=[list(p) for p in trimmed],
                         ufo_raw_path_length=sum(distance(a, b)
                             for a, b in zip(points[:-1], points[1:])),
                         trimmed_path_length=sum(distance(a, b)
                             for a, b in zip(trimmed[:-1], trimmed[1:])),
                         selected_waypoints=[list(p) for p in safe],
                         scan_lite_status="ACCEPTED",
                         executor_status="QUEUED", fallback_used=False,
                         reason="LOCAL_PATH_COMMITTED",
                         rejected_waypoints=rejected)
            self._publish_next()

    def _publish_next(self):
        if self._active is not None or not self._waypoints:
            return
        self._active = self._waypoints.pop(0)
        message = PointStamped()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = self._frame
        message.point.x, message.point.y = self._active[0], self._active[1]
        message.point.z = self._pose[2] if self._pose else 0.45
        self._goal_pub.publish(message)
        self._status("EXECUTING")

    def _on_result(self, message):
        try:
            result = json.loads(message.data)
        except (TypeError, ValueError):
            return
        with self._lock:
            if self._active is None:
                return
            completed = self._active
            self._active = None
            if result.get("success"):
                pose = PoseStamped()
                pose.header.stamp = rospy.Time.now()
                pose.header.frame_id = self._frame
                pose.pose.position.x = completed[0]
                pose.pose.position.y = completed[1]
                pose.pose.position.z = self._pose[2] if self._pose else 0.0
                pose.pose.orientation.z = math.sin(completed[2] * 0.5)
                pose.pose.orientation.w = math.cos(completed[2] * 0.5)
                self._waypoint_pub.publish(pose)
                self._record(ufo_raw_output_type="execution",
                             selected_waypoints=[list(completed)],
                             scan_lite_status="ACCEPTED",
                             executor_status="SUCCEEDED",
                             fallback_used=False, reason=result.get("reason"))
                if self._waypoints:
                    self._publish_next()
                elif self._pending_path is not None:
                    pending, self._pending_path = self._pending_path, None
                    self._on_path(pending)
                else:
                    self._replan_pub.publish(String(data="LOCAL_PATH_COMPLETE"))
            else:
                self._waypoints = []
                self._record(ufo_raw_output_type="execution",
                             selected_waypoints=[list(completed)],
                             scan_lite_status="ACCEPTED",
                             executor_status="FAILED",
                             fallback_used=False, reason=result.get("reason"))
                self._replan_pub.publish(String(data="EXECUTOR_PATH_FAILED"))


if __name__ == "__main__":
    rospy.init_node("ufo_path_adapter")
    UfoPathAdapter()
    rospy.spin()
