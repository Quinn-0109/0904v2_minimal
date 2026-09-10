#!/usr/bin/env python3
"""Passive Gazebo-only stability audit for the Step 2.3 single-goal test.

This node never publishes motion commands and is not part of the online
executor. Ground truth is used only to verify the unchanged RL controller.
"""

import json
import math
import os
import threading

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


def roll_pitch(q):
    sinr = 2.0 * (q.w * q.x + q.y * q.z)
    cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr, cosr)
    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    return roll, pitch


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class GoalStabilityMonitor:
    def __init__(self):
        self._path = os.path.abspath(rospy.get_param("~output_file"))
        self._tilt_limit = float(rospy.get_param("~tilt_limit", 0.35))
        self._height_limit = float(rospy.get_param("~height_limit", 0.24))
        self._lock = threading.Lock()
        self._moving = False
        self._samples = []
        self._start = rospy.Time.now()
        rospy.Subscriber("/cmd_vel", Twist, self._on_command, queue_size=10)
        rospy.Subscriber("/ground_truth/base_w", Odometry, self._on_pose, queue_size=30)
        rospy.on_shutdown(self._write)

    def _on_command(self, message):
        moving = math.hypot(message.linear.x, message.linear.y) > 0.03 or abs(message.angular.z) > 0.03
        with self._lock:
            self._moving = self._moving or moving

    def _on_pose(self, message):
        with self._lock:
            if not self._moving:
                return
        p, q = message.pose.pose.position, message.pose.pose.orientation
        roll, pitch = roll_pitch(q)
        sample = {
            "t": round((rospy.Time.now() - self._start).to_sec(), 4),
            "x": float(p.x), "y": float(p.y), "z": float(p.z),
            "yaw": yaw_from_quaternion(q), "roll": roll, "pitch": pitch,
        }
        with self._lock:
            self._samples.append(sample)
            if len(self._samples) > 20000:
                self._samples = self._samples[-20000:]

    def _write(self):
        with self._lock:
            samples = list(self._samples)
        max_roll = max((abs(item["roll"]) for item in samples), default=None)
        max_pitch = max((abs(item["pitch"]) for item in samples), default=None)
        min_height = min((item["z"] for item in samples), default=None)
        passed = bool(
            samples and max_roll <= self._tilt_limit
            and max_pitch <= self._tilt_limit and min_height >= self._height_limit
        )
        displacement = None
        if samples:
            displacement = math.hypot(samples[-1]["x"] - samples[0]["x"],
                                      samples[-1]["y"] - samples[0]["y"])
        payload = {
            "schema": "simenv_goal_executor_stability_v1",
            "passive_test_only": True, "passed": passed,
            "sample_count": len(samples), "max_abs_roll_rad": max_roll,
            "max_abs_pitch_rad": max_pitch, "minimum_height_m": min_height,
            "ground_truth_start": samples[0] if samples else None,
            "ground_truth_end": samples[-1] if samples else None,
            "ground_truth_displacement_m": displacement,
            "thresholds": {
                "maximum_tilt_rad": self._tilt_limit,
                "minimum_height_m": self._height_limit,
            },
        }
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        temporary = self._path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, self._path)


if __name__ == "__main__":
    rospy.init_node("goal_stability_monitor")
    GoalStabilityMonitor()
    rospy.spin()
