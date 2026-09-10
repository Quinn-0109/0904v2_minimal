#!/usr/bin/env python3
"""Diagnose zero-velocity stalls and request cause-specific recovery."""

import json
import math
import os
import threading
import time

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String


class VelocityWatchdog:
    def __init__(self):
        self._lock = threading.Lock()
        output_dir = os.path.abspath(rospy.get_param("~output_dir", "/tmp/ufo"))
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        self._log = os.path.join(output_dir, "logs", "velocity_watchdog.jsonl")
        self._timeout = float(rospy.get_param("~stall_timeout", 5.0))
        self._linear = float(rospy.get_param("~linear_threshold", 0.03))
        self._angular = float(rospy.get_param("~angular_threshold", 0.05))
        self._displacement = float(rospy.get_param(
            "~displacement_threshold", 0.05))
        self._cmd = None
        self._desired = None
        self._pose = None
        self._window_pose = None
        self._quiet_since = None
        self._complete = False
        self._locomotion_ready = False
        self._ufo, self._adapter, self._executor = {}, {}, {}
        self._last_report = -1e9
        self._status_pub = rospy.Publisher(
            "/simenv/velocity_watchdog_status", String, queue_size=10, latch=True)
        self._replan_pub = rospy.Publisher(
            "/simenv/ufoexplorer_replan_request", String, queue_size=10)
        rospy.Subscriber("/cmd_vel", Twist, self._on_cmd, queue_size=20)
        rospy.Subscriber(rospy.get_param(
            "~desired_cmd_vel_topic", "/simenv/desired_cmd_vel"),
            Twist, self._on_desired, queue_size=20)
        rospy.Subscriber(rospy.get_param("~odom_topic", "/simenv/ufo/odometry"),
                         Odometry, self._on_odom, queue_size=20)
        rospy.Subscriber("/simenv/mission_complete",
                         Bool, self._on_complete, queue_size=2)
        rospy.Subscriber("/locomotion_ready",
                         Bool, self._on_locomotion, queue_size=2)
        rospy.Subscriber("/simenv/ufoexplorer_status",
                         String, self._on_ufo, queue_size=10)
        rospy.Subscriber("/simenv/ufo_path_adapter_status",
                         String, self._on_adapter, queue_size=10)
        rospy.Subscriber("/simenv/goal_executor_status",
                         String, self._on_executor, queue_size=10)
        rospy.Timer(rospy.Duration(0.25), self._tick)

    @staticmethod
    def _decode(message):
        try:
            return json.loads(message.data)
        except (TypeError, ValueError):
            return {}

    def _on_cmd(self, msg): self._cmd = msg
    def _on_desired(self, msg): self._desired = msg
    def _on_complete(self, msg): self._complete = bool(msg.data)
    def _on_locomotion(self, msg): self._locomotion_ready = bool(msg.data)
    def _on_ufo(self, msg): self._ufo = self._decode(msg)
    def _on_adapter(self, msg): self._adapter = self._decode(msg)
    def _on_executor(self, msg): self._executor = self._decode(msg)

    def _on_odom(self, message):
        self._pose = (message.pose.pose.position.x,
                      message.pose.pose.position.y)

    def _reason(self):
        ufo_state = self._ufo.get("state", "")
        adapter_state = self._adapter.get("state", "")
        if not self._locomotion_ready:
            return "LOCOMOTION_NOT_READY"
        if ufo_state == "UFO_NO_PATH":
            return "UFO_NO_PATH"
        if ufo_state == "UFO_PATH_EMPTY":
            return "UFO_PATH_EMPTY"
        if ufo_state == "UFO_PATH_TOO_SHORT":
            return "UFO_PATH_TOO_SHORT"
        if adapter_state == "SCAN_REJECTED":
            return "SCAN_REJECTED"
        if adapter_state == "WAITING_FOR_POSE_OR_MAP":
            return "EXECUTOR_WAITING"
        if adapter_state == "PATH_COMMIT" and not self._adapter.get("active_waypoint"):
            return "WAYPOINT_TOO_CLOSE"
        if self._ufo.get("fallback_active"):
            return "PLANNER_FALLBACK_ACTIVE"
        if self._desired is not None and self._cmd is not None:
            desired = abs(self._desired.linear.x) + abs(self._desired.angular.z)
            actual = abs(self._cmd.linear.x) + abs(self._cmd.angular.z)
            if desired > 0.08 and actual < 0.03:
                return "SAFETY_GUARD_ZEROED"
        if self._executor.get("state") in ("waiting", "idle"):
            return "EXECUTOR_WAITING"
        return "UNKNOWN_STALL"

    @staticmethod
    def _twist_dict(msg):
        return None if msg is None else dict(
            linear_x=msg.linear.x, linear_y=msg.linear.y,
            angular_z=msg.angular.z)

    def _tick(self, _event):
        if self._complete or self._pose is None or self._cmd is None:
            self._quiet_since = None
            return
        quiet = (math.hypot(self._cmd.linear.x, self._cmd.linear.y) < self._linear
                 and abs(self._cmd.angular.z) < self._angular)
        now = time.monotonic()
        if not quiet:
            self._quiet_since, self._window_pose = None, None
            return
        if self._quiet_since is None:
            self._quiet_since, self._window_pose = now, self._pose
            return
        displacement = math.hypot(self._pose[0] - self._window_pose[0],
                                  self._pose[1] - self._window_pose[1])
        if now - self._quiet_since < self._timeout or displacement >= self._displacement:
            return
        if now - self._last_report < self._timeout:
            return
        self._last_report = now
        reason = self._reason()
        payload = dict(timestamp=time.time(), event="STALLED",
                       stall_reason=reason, robot_pose=list(self._pose),
                       quiet_seconds=now - self._quiet_since,
                       displacement=displacement,
                       desired_cmd_vel=self._twist_dict(self._desired),
                       cmd_vel=self._twist_dict(self._cmd),
                       ufo_status=self._ufo, adapter_status=self._adapter,
                       executor_status=self._executor)
        with open(self._log, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
        self._status_pub.publish(String(data=json.dumps(payload, sort_keys=True)))
        if reason in ("UFO_NO_PATH", "UFO_PATH_EMPTY", "UFO_PATH_TOO_SHORT",
                      "WAYPOINT_TOO_CLOSE", "SCAN_REJECTED", "UNKNOWN_STALL"):
            self._replan_pub.publish(String(data=reason))


if __name__ == "__main__":
    rospy.init_node("velocity_watchdog")
    VelocityWatchdog()
    rospy.spin()
