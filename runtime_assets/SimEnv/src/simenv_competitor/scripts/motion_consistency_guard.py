#!/usr/bin/env python3
"""Detect motion-time FAST-LIO scale inconsistency without correcting pose."""

import json
import math
import os
import threading
import time
from collections import deque

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Int32, String


class MotionConsistencyGuard:
    def __init__(self):
        output_dir = os.path.abspath(rospy.get_param("~output_dir"))
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        self.log_path = os.path.join(
            output_dir, "logs", "motion_consistency_guard.jsonl")
        self.windows = tuple(float(v) for v in rospy.get_param(
            "~windows_seconds", [5.0, 10.0, 20.0]))
        self.ratio_min = float(rospy.get_param("~minimum_distance_ratio", 0.5))
        self.ratio_max = float(rospy.get_param("~maximum_distance_ratio", 1.8))
        self.minimum_command_distance = float(rospy.get_param(
            "~minimum_command_distance", 0.30))
        self.minimum_drift = float(rospy.get_param(
            "~minimum_drift_magnitude", 0.35))
        self.rate = float(rospy.get_param("~rate", 2.0))
        self.started = time.monotonic()
        self.lock = threading.RLock()
        self.pose = None
        self.cmd = (0.0, 0.0, 0.0)
        self.desired = (0.0, 0.0, 0.0)
        self.ready = False
        self.active_goal_id = -1
        self.last_result = None
        self.last_imu_time = None
        self.last_imu_linear_acceleration = None
        # time, x, y, commanded speed, desired speed, locomotion ready
        self.samples = deque()
        self.status_pub = rospy.Publisher(
            "/simenv/motion_consistency_status", String,
            queue_size=2, latch=True)
        self.degraded_pub = rospy.Publisher(
            "/simenv/localization_degraded", Bool, queue_size=2, latch=True)
        rospy.Subscriber("/Odometry", Odometry, self._on_odom, queue_size=50)
        rospy.Subscriber("/cmd_vel", Twist, self._on_cmd, queue_size=20)
        rospy.Subscriber("/simenv/desired_cmd_vel", Twist,
                         self._on_desired, queue_size=20)
        rospy.Subscriber("/locomotion_ready", Bool,
                         self._on_ready, queue_size=5)
        rospy.Subscriber("/simenv/active_goal_id", Int32,
                         self._on_goal, queue_size=5)
        rospy.Subscriber("/simenv/goal_execution_result", String,
                         self._on_result, queue_size=5)
        rospy.Subscriber(rospy.get_param("~imu_topic", "/imu/data"), Imu,
                         self._on_imu, queue_size=20)
        rospy.Timer(rospy.Duration(1.0 / max(0.2, self.rate)), self._tick)

    def _on_cmd(self, msg):
        with self.lock:
            self.cmd = (float(msg.linear.x), float(msg.linear.y),
                        float(msg.angular.z))

    def _on_desired(self, msg):
        with self.lock:
            self.desired = (float(msg.linear.x), float(msg.linear.y),
                            float(msg.angular.z))

    def _on_ready(self, msg):
        with self.lock:
            self.ready = bool(msg.data)

    def _on_goal(self, msg):
        with self.lock:
            self.active_goal_id = int(msg.data)

    def _on_result(self, msg):
        try:
            value = json.loads(msg.data)
        except (TypeError, ValueError):
            value = {"raw": msg.data}
        with self.lock:
            self.last_result = value

    def _on_imu(self, msg):
        acceleration = msg.linear_acceleration
        with self.lock:
            self.last_imu_time = time.monotonic()
            self.last_imu_linear_acceleration = (
                float(acceleration.x), float(acceleration.y),
                float(acceleration.z))

    def _on_odom(self, msg):
        now = time.monotonic()
        p = msg.pose.pose.position
        pose = (float(p.x), float(p.y))
        if not all(math.isfinite(v) for v in pose):
            return
        with self.lock:
            self.pose = pose
            command_speed = math.hypot(self.cmd[0], self.cmd[1])
            desired_speed = math.hypot(self.desired[0], self.desired[1])
            self.samples.append((now, pose[0], pose[1], command_speed,
                                 desired_speed, self.ready))
            cutoff = now - max(self.windows) - 2.0
            while self.samples and self.samples[0][0] < cutoff:
                self.samples.popleft()

    @staticmethod
    def _integrate(samples, index):
        distance = 0.0
        for a, b in zip(samples[:-1], samples[1:]):
            dt = max(0.0, min(0.25, b[0] - a[0]))
            if a[5]:
                distance += max(0.0, a[index]) * dt
        return distance

    def _window(self, samples, now, seconds):
        values = [item for item in samples if item[0] >= now - seconds]
        if len(values) < 2:
            return {"window_sec": seconds, "available": False,
                    "degraded": False}
        path_distance = sum(math.hypot(b[1] - a[1], b[2] - a[2])
                            for a, b in zip(values[:-1], values[1:]))
        endpoint_dx = values[-1][1] - values[0][1]
        endpoint_dy = values[-1][2] - values[0][2]
        endpoint_distance = math.hypot(endpoint_dx, endpoint_dy)
        command_distance = self._integrate(values, 3)
        desired_distance = self._integrate(values, 4)
        reference = command_distance if command_distance >= 0.05 else desired_distance
        ratio = path_distance / reference if reference > 1e-6 else None
        drift = abs(path_distance - reference)
        degraded = bool(
            reference >= self.minimum_command_distance and
            drift >= self.minimum_drift and
            (ratio < self.ratio_min or ratio > self.ratio_max))
        axis = "x" if abs(endpoint_dx) >= abs(endpoint_dy) else "y"
        return {
            "window_sec": seconds, "available": True,
            "fastlio_path_distance_m": round(path_distance, 4),
            "fastlio_endpoint_distance_m": round(endpoint_distance, 4),
            "command_integral_distance_m": round(command_distance, 4),
            "desired_integral_distance_m": round(desired_distance, 4),
            "distance_ratio": None if ratio is None else round(ratio, 4),
            "drift_axis": axis,
            "drift_magnitude_m": round(drift, 4),
            "degraded": degraded,
        }

    def _tick(self, _event):
        now = time.monotonic()
        with self.lock:
            samples = list(self.samples)
            pose, ready = self.pose, self.ready
            goal_id, result = self.active_goal_id, self.last_result
            imu_time = self.last_imu_time
            imu_acceleration = self.last_imu_linear_acceleration
        windows = [self._window(samples, now, value) for value in self.windows]
        degraded_windows = [item for item in windows if item["degraded"]]
        degraded = bool(degraded_windows)
        worst = (max(degraded_windows,
                     key=lambda item: item["drift_magnitude_m"])
                 if degraded_windows else None)
        payload = {
            "timestamp": time.time(),
            "elapsed_sec": round(now - self.started, 3),
            "robot_pose": None if pose is None else {"x": pose[0], "y": pose[1]},
            "locomotion_ready": ready, "active_goal_id": goal_id,
            "localization_degraded": degraded,
            "drift_axis": None if worst is None else worst["drift_axis"],
            "drift_magnitude_m": 0.0 if worst is None else worst["drift_magnitude_m"],
            "trigger_window_sec": None if worst is None else worst["window_sec"],
            "windows": windows,
            "recent_goal_result_reason": (
                result.get("reason") if isinstance(result, dict) else None),
            "imu_available": bool(
                imu_time is not None and now - imu_time <= 1.0),
            "imu_linear_acceleration": imu_acceleration,
            "action": "warning_only_no_pose_correction",
        }
        self.degraded_pub.publish(Bool(data=degraded))
        self.status_pub.publish(String(data=json.dumps(payload, sort_keys=True)))
        try:
            with open(self.log_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, sort_keys=True) + "\n")
        except OSError as error:
            rospy.logwarn_throttle(10.0, "motion consistency log failed: %s", error)


if __name__ == "__main__":
    rospy.init_node("motion_consistency_guard")
    MotionConsistencyGuard()
    rospy.spin()
