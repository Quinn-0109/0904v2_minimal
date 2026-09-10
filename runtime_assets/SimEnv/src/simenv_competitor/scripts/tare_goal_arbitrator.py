#!/usr/bin/env python3
"""Commitment and preemption arbitration for Official TARE waypoints.

This node never invents an exploration goal.  It accepts a raw TARE goal,
keeps the already accepted goal, or rejects the update.  Endpoint repair and
all collision/reachability checks remain in tare_goal_refiner and FAR/SCAN.
"""

import json
import math
import os
import sys
import threading

import rospy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import String

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from hierarchical_ros_utils import occupancy_from_message
from tare_goal_arbitration_core import (
    ArbitrationConfig, TareGoalArbitrationPolicy, planar_distance)


class TareGoalArbitrator:
    def __init__(self):
        self._lock = threading.RLock()
        self._pose = None
        self._yaw = 0.0
        self._grid = None
        self._recovery = False
        self._navigation_goal = None
        self._last_result_sequence = None

        config = ArbitrationConfig(
            min_goal_commit_time=float(rospy.get_param(
                "~min_goal_commit_time", 12.0)),
            preemption_ratio=float(rospy.get_param(
                "~preemption_ratio", 1.30)),
            preemption_absolute_gain=float(rospy.get_param(
                "~preemption_absolute_gain", 3.0)),
            backward_penalty=float(rospy.get_param(
                "~backward_penalty", 5.0)),
            active_region_radius=float(rospy.get_param(
                "~active_region_radius", 4.0)),
            local_exploration_depth=float(rospy.get_param(
                "~local_exploration_depth", 3.0)),
            minimum_unknown_area_to_continue=float(rospy.get_param(
                "~minimum_unknown_area_to_continue", 3.0)),
            minimum_frontier_cells_to_continue=int(rospy.get_param(
                "~minimum_frontier_cells_to_continue", 8)),
            maximum_failures=int(rospy.get_param(
                "~maximum_failures", 3)),
            max_supersede_per_minute=int(rospy.get_param(
                "~max_supersede_per_minute", 5)),
            forced_commit_duration=float(rospy.get_param(
                "~forced_commit_duration", 15.0)),
            same_goal_distance=float(rospy.get_param(
                "~same_goal_distance", 0.45)),
            reached_distance=float(rospy.get_param(
                "~reached_distance", 0.45)),
            failed_goal_blacklist_duration=float(rospy.get_param(
                "~failed_goal_blacklist_duration", 20.0)))
        self._policy = TareGoalArbitrationPolicy(config)
        self._navigation_match_distance = float(rospy.get_param(
            "~navigation_match_distance", 1.0))

        output_dir = os.path.abspath(rospy.get_param(
            "~output_dir", "/tmp/simenv/tare_far"))
        self._log_path = os.path.join(
            output_dir, "logs", "tare_goal_arbitration.jsonl")
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        # A run directory must not inherit arbitration decisions from an old
        # process that happened to reuse the same output path.
        with open(self._log_path, "w", encoding="utf-8"):
            pass

        self._publisher = rospy.Publisher(
            rospy.get_param(
                "~arbitrated_goal_topic", "/tare/arbitrated_way_point"),
            PointStamped, queue_size=3)
        self._status_publisher = rospy.Publisher(
            rospy.get_param(
                "~arbitration_status_topic", "/tare/arbitration_status"),
            String, queue_size=10, latch=True)
        rospy.Subscriber(
            rospy.get_param("~tare_goal_topic", "/tare/way_point"),
            PointStamped, self._on_tare_goal, queue_size=20)
        rospy.Subscriber(
            rospy.get_param(
                "~odom_topic", "/tare/state_estimation_at_scan"),
            Odometry, self._on_odom, queue_size=30)
        rospy.Subscriber(
            rospy.get_param(
                "~map_topic", "/simenv/voxel_floor_projection"),
            OccupancyGrid, self._on_map, queue_size=2)
        rospy.Subscriber(
            rospy.get_param(
                "~far_status_topic", "/simenv/tare_far_status"),
            String, self._on_far_status, queue_size=20)
        rospy.Subscriber(
            rospy.get_param(
                "~execution_result_topic",
                "/simenv/goal_execution_result"),
            String, self._on_execution_result, queue_size=20)
        rospy.Subscriber(
            rospy.get_param(
                "~current_navigation_goal_topic",
                "/hybrid_exploration_goal"),
            PointStamped, self._on_navigation_goal, queue_size=10)
        rospy.loginfo(
            "TARE goal arbitrator ready: commit=%.1fs ratio=%.2f "
            "absolute_gain=%.1f log=%s",
            config.min_goal_commit_time, config.preemption_ratio,
            config.preemption_absolute_gain, self._log_path)

    @staticmethod
    def _decode(message):
        try:
            value = json.loads(message.data)
            return value if isinstance(value, dict) else {}
        except (TypeError, ValueError):
            return {}

    @staticmethod
    def _point(message):
        return (float(message.point.x), float(message.point.y),
                float(message.point.z))

    def _on_odom(self, message):
        point = message.pose.pose.position
        q = message.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        pose = (float(point.x), float(point.y), float(point.z))
        with self._lock:
            self._pose, self._yaw = pose, yaw

    def _on_map(self, message):
        try:
            grid = occupancy_from_message(message)
        except (IndexError, TypeError, ValueError):
            return
        with self._lock:
            self._grid = grid
            self._policy.update_region(grid, self._pose)

    def _on_navigation_goal(self, message):
        point = self._point(message)
        if all(math.isfinite(value) for value in point):
            with self._lock:
                self._navigation_goal = point

    def _navigation_matches_tare_locked(self):
        return bool(
            self._navigation_goal is not None and
            self._policy.current_goal is not None and
            planar_distance(self._navigation_goal,
                            self._policy.current_goal) <=
            self._navigation_match_distance)

    def _on_far_status(self, message):
        data = self._decode(message)
        state = str(data.get("state", "")).lower()
        with self._lock:
            if state in ("waiting_for_far", "executing"):
                goal = data.get("goal")
                if isinstance(goal, (list, tuple)) and len(goal) >= 2:
                    self._navigation_goal = (
                        float(goal[0]), float(goal[1]),
                        float(goal[2]) if len(goal) > 2 else 0.0)
                self._recovery = False
            elif ("recovery" in state or "return" in state):
                self._recovery = True
            invalidated = None
            if (state in ("planning_failed", "scan_rejected_path") and
                    self._navigation_matches_tare_locked()):
                invalidated = self._policy.mark_failure(
                    rospy.Time.now().to_sec())
        if invalidated is not None:
            self._publish_invalidation(invalidated, state)

    def _on_execution_result(self, message):
        data = self._decode(message)
        sequence = data.get("goal_sequence")
        with self._lock:
            if sequence is not None and sequence == self._last_result_sequence:
                return
            self._last_result_sequence = sequence
            invalidated = None
            if self._navigation_matches_tare_locked():
                invalidated = self._policy.mark_execution_result(
                    bool(data.get("success", False)), self._pose,
                    rospy.Time.now().to_sec())
        if invalidated is not None:
            self._publish_invalidation(
                invalidated, str(data.get("reason", "execution_failed")))

    def _publish_invalidation(self, goal, cause):
        now = rospy.Time.now().to_sec()
        record = {
            "timestamp": now,
            "event": "current_goal_invalidated",
            "robot_pose": list(self._pose) if self._pose is not None else None,
            "tare_goal": list(goal),
            "current_goal": list(goal),
            "accepted": False,
            "reason": "current_goal_failure_limit",
            "failure_cause": cause,
            "new_score": None,
            "current_score": None,
            "distance_to_new_goal": (
                planar_distance(self._pose, goal)
                if self._pose is not None else None),
            "distance_to_current_goal": (
                planar_distance(self._pose, goal)
                if self._pose is not None else None),
            "is_backward_goal": None,
            "active_region_id": self._policy.active_region_id,
            "supersede_count_last_60s": len(
                self._policy.supersede_times),
        }
        self._write_record(record)
        self._status_publisher.publish(String(data=json.dumps({
            "event": "current_goal_invalidated",
            "stamp": now,
            "goal": list(goal),
            "reason": "current_goal_failure_limit",
            "failure_cause": cause,
        })))
        rospy.logwarn(
            "TARE goal invalidated after repeated failures: [%.2f, %.2f] (%s)",
            goal[0], goal[1], cause)

    def _write_record(self, record):
        try:
            with open(self._log_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(
                    record, ensure_ascii=False, sort_keys=True) + "\n")
        except OSError as error:
            rospy.logwarn_throttle(
                5.0, "Could not write TARE arbitration log: %s", error)

    def _on_tare_goal(self, message):
        goal = self._point(message)
        if not all(math.isfinite(value) for value in goal):
            return
        now = rospy.Time.now().to_sec()
        with self._lock:
            pose, yaw, grid = self._pose, self._yaw, self._grid
            current_before = self._policy.current_goal
            if pose is None or grid is None:
                record = {
                    "timestamp": now,
                    "robot_pose": list(pose) if pose is not None else None,
                    "tare_goal": list(goal),
                    "current_goal": (list(current_before)
                                     if current_before is not None else None),
                    "accepted": False,
                    "reason": "arbitration_inputs_unavailable",
                    "new_score": None,
                    "current_score": None,
                    "distance_to_new_goal": None,
                    "distance_to_current_goal": None,
                    "is_backward_goal": None,
                    "active_region_id": self._policy.active_region_id,
                    "supersede_count_last_60s": len(
                        self._policy.supersede_times),
                }
                decision = None
            else:
                decision = self._policy.decide(
                    goal, pose, yaw, grid, now, recovery=self._recovery)
                record = {
                    "timestamp": now,
                    "robot_pose": list(pose),
                    "tare_goal": list(goal),
                    "current_goal": (list(current_before)
                                     if current_before is not None else None),
                    "accepted": decision.accepted,
                    "reason": decision.reason,
                    "new_score": decision.new_score,
                    "current_score": decision.current_score,
                    "distance_to_new_goal": decision.distance_to_new_goal,
                    "distance_to_current_goal": (
                        decision.distance_to_current_goal),
                    "is_backward_goal": decision.is_backward_goal,
                    "active_region_id": decision.active_region_id,
                    "supersede_count_last_60s": (
                        decision.supersede_count_last_60s),
                    "forced_commit_mode": decision.forced_commit_mode,
                }
        self._write_record(record)
        self._status_publisher.publish(String(data=json.dumps({
            "event": "arbitration_decision",
            "stamp": now,
            "goal": list(goal),
            "accepted": bool(decision is not None and decision.accepted),
            "reason": record["reason"],
        })))
        if decision is not None and decision.accepted:
            output = PointStamped()
            output.header = message.header
            output.header.stamp = rospy.Time.now()
            output.point = message.point
            self._publisher.publish(output)
            rospy.loginfo(
                "TARE goal accepted: [%.2f, %.2f] reason=%s score=%.2f",
                goal[0], goal[1], decision.reason, decision.new_score)
        else:
            rospy.loginfo_throttle(
                2.0, "TARE goal held: [%.2f, %.2f] reason=%s",
                goal[0], goal[1], record["reason"])


if __name__ == "__main__":
    rospy.init_node("tare_goal_arbitrator")
    TareGoalArbitrator()
    rospy.spin()
