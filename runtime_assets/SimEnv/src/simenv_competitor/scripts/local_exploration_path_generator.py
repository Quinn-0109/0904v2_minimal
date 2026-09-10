#!/usr/bin/env python3
"""Generate and commit short exploration waypoint sequences.

Hybrid TARE/FUEL goals are treated as directional intent.  This node samples
an observed-free local path, publishes the complete nav_msgs/Path for
inspection, and sends its waypoints one at a time through the unchanged
Official FAR -> A* fallback -> SCAN-lite -> Goal Executor chain.
"""

import json
import math
import os
import sys
import threading
import time

import rospy
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from std_msgs.msg import Bool, String

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from hierarchical_ros_utils import occupancy_from_message
from local_exploration_path_core import (
    LocalPathConfig, generate_candidate_paths, path_commit_released,
    planar_distance)
from planner_interface import AStarPlanner, FarPlannerInterface
from simenv_competitor.srv import CheckTwinCylinder, CheckTwinCylinderRequest


class LocalExplorationPathGenerator:
    PATH_IDLE = "PATH_IDLE"
    PATH_COMMIT = "PATH_COMMIT"

    def __init__(self):
        self._lock = threading.RLock()
        self._pose = None
        self._yaw = 0.0
        self._frame = rospy.get_param("~world_frame", "map")
        self._grid = None
        self._grid_ready_stamp = None
        self._locomotion_ready = False
        self._frontiers = []
        self._latest_target = None
        self._pending_target = None
        self._active_region_id = None
        self._active_region_point = None
        self._visited = []
        self._last_visited_stamp = 0.0
        self._mode = self.PATH_IDLE
        self._current_path = None
        self._waypoint_index = 0
        self._executed_waypoint_count = 0
        self._commit_started = 0.0
        self._commit_start_pose = None
        self._planning = False
        self._path_sequence = 0
        self._last_result_sequence = None
        self._last_plan_attempt = 0.0
        self._failure_blacklist = []

        self._config = LocalPathConfig(
            alpha=float(rospy.get_param("~alpha", 1.0)),
            beta=float(rospy.get_param("~beta", 0.4)),
            gamma=float(rospy.get_param("~gamma", 1.0)),
            delta=float(rospy.get_param("~delta", 0.5)),
            eta=float(rospy.get_param("~eta", 1.0)),
            mu=float(rospy.get_param("~mu", 1.5)),
            waypoint_spacing=float(rospy.get_param(
                "~waypoint_spacing", 0.75)),
            waypoint_count=int(rospy.get_param("~waypoint_count", 4)),
            minimum_waypoints=int(rospy.get_param(
                "~minimum_waypoints", 2)),
            sample_free_search_radius=float(rospy.get_param(
                "~sample_free_search_radius", 0.45)),
            free_clearance=float(rospy.get_param(
                "~free_clearance", 0.30)),
            sensor_range=float(rospy.get_param("~sensor_range", 3.0)),
            risk_radius=float(rospy.get_param("~risk_radius", 0.75)),
            revisit_radius=float(rospy.get_param(
                "~revisit_radius", 0.65)),
            backward_penalty=float(rospy.get_param(
                "~backward_penalty", 5.0)))
        self._min_commit_waypoints = int(rospy.get_param(
            "~min_commit_waypoints", 3))
        self._min_commit_distance = float(rospy.get_param(
            "~min_commit_distance", 1.5))
        self._min_commit_time = float(rospy.get_param(
            "~min_commit_time", 12.0))
        self._minimum_remaining_gain = float(rospy.get_param(
            "~minimum_remaining_information_gain", 1.0))
        self._waypoint_tolerance = float(rospy.get_param(
            "~waypoint_tolerance", 0.35))
        self._planning_retry_period = float(rospy.get_param(
            "~planning_retry_period", 3.0))
        self._autonomous_start_delay = float(rospy.get_param(
            "~autonomous_start_delay", 3.0))
        self._failure_blacklist_duration = float(rospy.get_param(
            "~failure_blacklist_duration", 20.0))
        self._maximum_frontier_directions = int(rospy.get_param(
            "~maximum_frontier_directions", 3))

        astar = AStarPlanner(
            clearance=float(rospy.get_param("~astar_clearance", 0.30)),
            reached_tolerance=float(rospy.get_param(
                "~astar_reached_tolerance", 0.15)),
            maximum_expansions=int(rospy.get_param(
                "~astar_maximum_expansions", 100000)),
            allow_blocked_start=True)
        self._planner = FarPlannerInterface(astar, backend="official_far_downstream")
        self._scan_service = rospy.ServiceProxy(
            rospy.get_param(
                "~scan_service",
                "/tare_far_voxel_mapper/check_twin_cylinder"),
            CheckTwinCylinder)

        output_dir = os.path.abspath(rospy.get_param(
            "~output_dir", "/tmp/simenv/local_exploration"))
        self._log_path = os.path.join(
            output_dir, "logs", "local_exploration_path_history.jsonl")
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        with open(self._log_path, "w", encoding="utf-8"):
            pass

        self._goal_publisher = rospy.Publisher(
            rospy.get_param(
                "~output_goal_topic", "/local_exploration_goal"),
            PointStamped, queue_size=3)
        self._path_publisher = rospy.Publisher(
            rospy.get_param(
                "~output_path_topic", "/simenv/local_exploration_path"),
            Path, queue_size=2, latch=True)
        self._json_publisher = rospy.Publisher(
            rospy.get_param(
                "~output_waypoint_list_topic",
                "/simenv/local_exploration_waypoints"),
            String, queue_size=2, latch=True)

        rospy.Subscriber(
            rospy.get_param("~map_topic", "/simenv/voxel_floor_projection"),
            OccupancyGrid, self._on_map, queue_size=2)
        rospy.Subscriber(
            rospy.get_param("~odom_topic", "/Odometry"),
            Odometry, self._on_odom, queue_size=40)
        rospy.Subscriber(
            rospy.get_param("~intent_goal_topic", "/hybrid_exploration_goal"),
            PointStamped, self._on_intent_goal, queue_size=20)
        rospy.Subscriber(
            rospy.get_param("~frontier_topic", "/simenv/frontier_clusters"),
            String, self._on_frontiers, queue_size=5)
        rospy.Subscriber(
            rospy.get_param("~far_status_topic", "/simenv/tare_far_status"),
            String, self._on_far_status, queue_size=20)
        rospy.Subscriber(
            rospy.get_param(
                "~execution_result_topic", "/simenv/goal_execution_result"),
            String, self._on_execution_result, queue_size=20)
        rospy.Subscriber("/locomotion_ready", Bool,
                         self._on_locomotion_ready, queue_size=2)
        rospy.Timer(rospy.Duration(0.5), self._on_timer)
        rospy.loginfo(
            "Local exploration path generator ready: %d x %.2fm, commit=%d/%.1fm/%.1fs",
            self._config.waypoint_count, self._config.waypoint_spacing,
            self._min_commit_waypoints, self._min_commit_distance,
            self._min_commit_time)

    @staticmethod
    def _decode(message):
        try:
            value = json.loads(message.data)
            return value if isinstance(value, dict) else {}
        except (TypeError, ValueError):
            return {}

    @staticmethod
    def _now():
        return rospy.Time.now().to_sec()

    def _on_map(self, message):
        try:
            grid = occupancy_from_message(message)
        except (IndexError, TypeError, ValueError):
            return
        with self._lock:
            self._grid = grid
            if self._grid_ready_stamp is None:
                self._grid_ready_stamp = self._now()

    def _on_locomotion_ready(self, message):
        with self._lock:
            self._locomotion_ready = bool(message.data)

    def _on_odom(self, message):
        p, q = message.pose.pose.position, message.pose.pose.orientation
        pose = (float(p.x), float(p.y), float(p.z))
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        advance = False
        with self._lock:
            self._pose, self._yaw = pose, yaw
            self._frame = message.header.frame_id or self._frame
            now = self._now()
            if (not self._visited or
                    planar_distance(pose, self._visited[-1]) >= 0.35 or
                    now - self._last_visited_stamp >= 1.0):
                self._visited.append(pose)
                self._visited = self._visited[-2000:]
                self._last_visited_stamp = now
            if self._current_path is not None:
                waypoints = self._current_path.waypoints
                if (self._waypoint_index < len(waypoints) and
                        planar_distance(
                            pose, waypoints[self._waypoint_index].point(pose[2])) <=
                        self._waypoint_tolerance):
                    advance = True
        if advance:
            self._advance_waypoint("pose_reached")

    def _on_frontiers(self, message):
        data = self._decode(message)
        clusters = data.get("clusters", [])
        parsed = []
        for item in clusters:
            if not isinstance(item, dict) or not item.get("reachable", True):
                continue
            goal = item.get("goal", item.get("center"))
            if not isinstance(goal, (list, tuple)) or len(goal) < 2:
                continue
            parsed.append({
                "id": str(item.get("id", "frontier")),
                "point": (float(goal[0]), float(goal[1]), 0.0),
                "unknown_area": float(item.get("unknown_area", 0.0)),
            })
        parsed.sort(key=lambda item: item["unknown_area"], reverse=True)
        with self._lock:
            self._frontiers = parsed

    def _region_for_target_locked(self, target):
        nearest = min(
            self._frontiers, default=None,
            key=lambda item: planar_distance(target, item["point"]))
        if nearest is not None and planar_distance(target, nearest["point"]) <= 2.0:
            return nearest["id"], nearest["point"]
        region_id = "intent_{:+d}_{:+d}".format(
            int(round(target[0] / 2.0)), int(round(target[1] / 2.0)))
        return region_id, target

    def _on_intent_goal(self, message):
        point = (float(message.point.x), float(message.point.y),
                 float(message.point.z))
        if not all(math.isfinite(value) for value in point):
            return
        plan = False
        with self._lock:
            self._latest_target = point
            self._pending_target = point
            if self._mode == self.PATH_IDLE:
                plan = True
            elif self._commit_released_locked(self._now()):
                plan = True
        if plan:
            self._schedule_plan("new_intent_goal")

    def _commit_released_locked(self, now):
        if self._mode != self.PATH_COMMIT or self._current_path is None:
            return True
        distance = (planar_distance(self._pose, self._commit_start_pose)
                    if self._pose is not None and
                    self._commit_start_pose is not None else 0.0)
        remaining_gain = sum(
            item.expected_gain
            for item in self._current_path.waypoints[self._waypoint_index:])
        return path_commit_released(
            self._executed_waypoint_count, distance,
            now - self._commit_started, remaining_gain,
            self._min_commit_waypoints, self._min_commit_distance,
            self._min_commit_time, self._minimum_remaining_gain)

    def _directional_targets_locked(self):
        targets = []
        if self._active_region_point is not None:
            targets.append({
                "point": self._active_region_point,
                "source": "active_region",
                "active_region_id": self._active_region_id,
            })
        if self._pending_target is not None:
            region_id, region_point = self._region_for_target_locked(
                self._pending_target)
            targets.append({
                "point": self._pending_target,
                "source": "tare_fuel_intent",
                "active_region_id": region_id,
            })
            if self._active_region_id is None:
                self._active_region_id = region_id
                self._active_region_point = region_point
        for item in self._frontiers[:self._maximum_frontier_directions]:
            targets.append({
                "point": item["point"],
                "source": "frontier_cluster",
                "active_region_id": item["id"],
            })
        return targets

    def _scan_check(self, point, yaw):
        now = self._now()
        with self._lock:
            self._failure_blacklist = [item for item in self._failure_blacklist
                                       if item[1] > now]
            if any(planar_distance(point, item[0]) < 0.60
                   for item in self._failure_blacklist):
                return False, "temporarily_blacklisted", 1.0
        request = CheckTwinCylinderRequest()
        request.pose_x, request.pose_y = float(point[0]), float(point[1])
        request.pose_z = float(point[2])
        request.yaw = float(yaw)
        request.front_offset = 0.06675
        request.rear_offset = -0.06675
        request.radius = 0.11772 + 0.12
        request.min_height = -0.057 - 0.08
        request.max_height = 0.057 + 0.08
        request.clearance_search_radius = request.radius + 0.50
        try:
            response = self._scan_service(request)
        except rospy.ServiceException:
            return False, "scan_service_unavailable", 1.0
        if not response.map_available:
            return False, "scan_map_unavailable", 1.0
        if response.occupied_collision:
            return False, "scan_occupied_collision", 1.0
        risk = min(1.0, float(response.unknown_queries) / 20.0)
        return True, ("scan_safe" if not response.unknown_queries else
                      "scan_safe_unknown_nearby"), risk

    def _schedule_plan(self, reason):
        with self._lock:
            if self._planning:
                return
            self._planning = True
        threading.Thread(
            target=self._plan_worker, args=(reason,), daemon=True).start()

    def _plan_worker(self, reason):
        with self._lock:
            pose, yaw, grid = self._pose, self._yaw, self._grid
            visited = list(self._visited)
            targets = self._directional_targets_locked()
            active_region_id = self._active_region_id
            self._path_sequence += 1
            sequence = self._path_sequence
        if pose is None or grid is None:
            with self._lock:
                self._planning = False
            return
        selected, candidates = generate_candidate_paths(
            grid, pose, yaw, self._planner, visited, targets,
            self._config, self._scan_check, recovery=False,
            path_prefix="local_{:04d}".format(sequence))
        # If the active region still has useful safe gain, keep its candidate
        # ahead of unrelated corridor/behind targets without modifying score.
        active_candidates = [
            item for item in candidates
            if (not item.rejected and active_region_id is not None and
                item.active_region_id == active_region_id and
                item.total_information_gain >= self._minimum_remaining_gain)]
        if active_candidates:
            selected = max(active_candidates, key=lambda item: item.final_score)
        with self._lock:
            self._planning = False
            self._last_plan_attempt = self._now()
            if selected is None:
                self._mode = self.PATH_IDLE
                self._record_locked(
                    None, candidates, "no_safe_candidate_path", reason)
                return
            if self._active_region_id is not None and not active_candidates:
                self._active_region_id = None
                self._active_region_point = None
            if selected.active_region_id is not None:
                self._active_region_id = selected.active_region_id
                region_target = next((
                    item.get("point") for item in targets
                    if item.get("active_region_id") == selected.active_region_id
                ), None)
                if region_target is not None:
                    self._active_region_point = tuple(region_target)
            self._current_path = selected
            self._waypoint_index = 0
            self._executed_waypoint_count = 0
            self._commit_started = self._now()
            self._commit_start_pose = pose
            self._mode = self.PATH_COMMIT
            self._pending_target = None
            self._publish_path_locked(selected)
            self._publish_current_waypoint_locked("path_selected")
            self._record_locked(selected, candidates, "PATH_COMMIT", reason)

    def _publish_path_locked(self, selected):
        message = Path()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = self._frame
        for index, waypoint in enumerate(selected.waypoints):
            pose = PoseStamped()
            pose.header = message.header
            pose.header.seq = index
            pose.pose.position.x = waypoint.x
            pose.pose.position.y = waypoint.y
            pose.pose.position.z = self._pose[2] if self._pose else 0.0
            pose.pose.orientation.z = math.sin(0.5 * waypoint.yaw)
            pose.pose.orientation.w = math.cos(0.5 * waypoint.yaw)
            message.poses.append(pose)
        self._path_publisher.publish(message)
        self._json_publisher.publish(String(data=json.dumps({
            "path_id": selected.path_id,
            "active_region_id": selected.active_region_id,
            "waypoints": [item.to_dict() for item in selected.waypoints],
        }, sort_keys=True)))

    def _publish_current_waypoint_locked(self, reason):
        if (self._current_path is None or
                self._waypoint_index >= len(self._current_path.waypoints)):
            return
        waypoint = self._current_path.waypoints[self._waypoint_index]
        message = PointStamped()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = self._frame
        message.point.x, message.point.y = waypoint.x, waypoint.y
        message.point.z = self._pose[2] if self._pose else 0.0
        self._goal_publisher.publish(message)
        rospy.loginfo(
            "Local path %s waypoint %d/%d -> [%.2f, %.2f] (%s)",
            self._current_path.path_id, self._waypoint_index + 1,
            len(self._current_path.waypoints), waypoint.x, waypoint.y, reason)

    def _advance_waypoint(self, reason):
        replan = False
        with self._lock:
            if self._current_path is None:
                return
            if (reason == "pose_reached" and self._pose is not None and
                    self._waypoint_index <
                    len(self._current_path.waypoints) and
                    planar_distance(
                        self._pose,
                        self._current_path.waypoints[
                            self._waypoint_index].point(self._pose[2])) >
                    self._waypoint_tolerance):
                return
            self._waypoint_index += 1
            self._executed_waypoint_count += 1
            self._record_locked(
                self._current_path, [], "waypoint_executed", reason)
            if self._waypoint_index >= len(self._current_path.waypoints):
                self._mode = self.PATH_IDLE
                self._current_path = None
                replan = True
            else:
                self._publish_current_waypoint_locked("previous_waypoint_reached")
        if replan:
            self._schedule_plan("local_path_completed")

    def _fail_current_path(self, reason):
        with self._lock:
            if self._current_path is None:
                return
            if self._waypoint_index < len(self._current_path.waypoints):
                failed = self._current_path.waypoints[self._waypoint_index]
                self._failure_blacklist.append((
                    failed.point(self._pose[2] if self._pose else 0.0),
                    self._now() + self._failure_blacklist_duration))
            self._record_locked(
                self._current_path, [], "path_failed", reason)
            self._current_path = None
            self._mode = self.PATH_IDLE
            self._waypoint_index = 0
        self._schedule_plan("path_failure_" + reason)

    def _on_far_status(self, message):
        data = self._decode(message)
        state = str(data.get("state", ""))
        if state in ("planning_failed", "scan_rejected_path"):
            self._fail_current_path(state)

    def _on_execution_result(self, message):
        data = self._decode(message)
        sequence = data.get("goal_sequence")
        with self._lock:
            if sequence is not None and sequence == self._last_result_sequence:
                return
            self._last_result_sequence = sequence
        if not bool(data.get("success", False)):
            self._fail_current_path(
                str(data.get("reason", "execution_failed")))

    def _on_timer(self, _event):
        schedule = None
        with self._lock:
            now = self._now()
            ready = bool(
                self._locomotion_ready and self._pose is not None and
                self._grid is not None and self._grid_ready_stamp is not None and
                now - self._grid_ready_stamp >= self._autonomous_start_delay)
            if not ready or self._planning:
                return
            if (self._mode == self.PATH_IDLE and
                    now - self._last_plan_attempt >= self._planning_retry_period):
                schedule = "idle_autonomous_local_path"
            elif (self._pending_target is not None and
                  self._commit_released_locked(now)):
                schedule = "commit_released_for_pending_intent"
        if schedule:
            self._schedule_plan(schedule)

    def _record_locked(self, selected, candidates, commit_status, reason):
        pose = self._pose
        record = {
            "timestamp": self._now(),
            "robot_pose": list(pose) if pose is not None else None,
            "selected_path_id": selected.path_id if selected else None,
            "waypoints": ([item.to_dict() for item in selected.waypoints]
                          if selected else []),
            "total_information_gain": (
                selected.total_information_gain if selected else 0.0),
            "path_length": selected.path_length if selected else 0.0,
            "revisit_cost": selected.revisit_cost if selected else 0.0,
            "turning_cost": selected.turning_cost if selected else 0.0,
            "risk_cost": selected.risk_cost if selected else 0.0,
            "forward_progress": selected.forward_progress if selected else 0.0,
            "final_score": selected.final_score if selected else None,
            "active_region_id": self._active_region_id,
            "commit_status": commit_status,
            "executed_waypoint_count": self._executed_waypoint_count,
            "current_waypoint_index": self._waypoint_index,
            "reason_for_replan": reason,
            "information_cells": (
                [list(item) for item in selected.information_cells]
                if selected and commit_status == "PATH_COMMIT" else []),
            "rejected_candidate_paths": [
                item.to_dict() for item in candidates
                if item.rejected or item is not selected],
        }
        try:
            with open(self._log_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(
                    record, ensure_ascii=False, sort_keys=True) + "\n")
        except OSError as error:
            rospy.logwarn_throttle(
                5.0, "Could not write local exploration path log: %s", error)


if __name__ == "__main__":
    rospy.init_node("local_exploration_path_generator")
    LocalExplorationPathGenerator()
    rospy.spin()
