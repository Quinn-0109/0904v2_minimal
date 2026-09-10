#!/usr/bin/env python3
"""Deterministic, sensor-only motion excitation for LIO benchmarking."""

import json
import math
import os

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def angle_diff(target, source):
    return math.atan2(math.sin(target - source), math.cos(target - source))


class LocalizationMotionBenchmark:
    def __init__(self):
        self._speed = float(rospy.get_param("~forward_speed", 0.40))
        self._yaw_rate = float(rospy.get_param("~yaw_rate", 0.38))
        self._drive1 = float(rospy.get_param("~drive1_seconds", 20.0))
        self._drive2 = float(rospy.get_param("~drive2_seconds", 10.0))
        self._drive3 = float(rospy.get_param("~drive3_seconds", 10.0))
        self._drive4 = float(rospy.get_param("~drive4_seconds", 10.0))
        self._profile = rospy.get_param("~profile", "square")
        self._start_delay = float(rospy.get_param("~start_delay", 3.0))
        self._phase_timeout = float(rospy.get_param("~spin_timeout", 24.0))
        self._done_file = rospy.get_param(
            "~done_file", os.path.join(os.getcwd(), "results", "localization_motion_done.json")
        )
        self._imu_yaw = None
        self._last_raw_yaw = None
        self._unwrapped_yaw = 0.0
        self._heading = None
        self._phase = "wait"
        self._phase_started = rospy.Time.now()
        self._started = self._phase_started
        self._turn_origin = 0.0
        self._finished = False
        self._pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self._done_pub = rospy.Publisher(
            "/simenv/mission_complete", Bool, queue_size=1, latch=True
        )
        rospy.Subscriber("/trunk_imu", Imu, self._on_imu, queue_size=100)
        rospy.Timer(rospy.Duration(0.05), self._on_timer)
        rospy.on_shutdown(lambda: self._pub.publish(Twist()))

    def _on_imu(self, message):
        raw = yaw_from_quaternion(message.orientation)
        if self._last_raw_yaw is not None:
            self._unwrapped_yaw += angle_diff(raw, self._last_raw_yaw)
        self._last_raw_yaw = raw
        self._imu_yaw = raw
        if self._heading is None:
            self._heading = raw

    def _set_phase(self, phase):
        self._phase = phase
        self._phase_started = rospy.Time.now()
        if phase.startswith("spin"):
            self._turn_origin = self._unwrapped_yaw
        rospy.loginfo("LIO benchmark phase=%s", phase)

    def _drive(self, duration, next_phase):
        command = Twist()
        error = 0.0 if self._heading is None or self._imu_yaw is None else angle_diff(
            self._heading, self._imu_yaw
        )
        command.linear.x = self._speed if abs(error) < 0.35 else 0.12
        command.angular.z = max(-0.35, min(0.35, 1.8 * error))
        self._pub.publish(command)
        if (rospy.Time.now() - self._phase_started).to_sec() >= duration:
            self._set_phase(next_phase)

    def _spin(self, target_angle, next_phase):
        command = Twist()
        command.angular.z = self._yaw_rate
        self._pub.publish(command)
        elapsed = (rospy.Time.now() - self._phase_started).to_sec()
        turned = self._unwrapped_yaw - self._turn_origin
        if turned >= target_angle or elapsed >= self._phase_timeout:
            if self._imu_yaw is not None:
                self._heading = self._imu_yaw
            self._set_phase(next_phase)

    def _finish(self):
        if self._finished:
            return
        self._finished = True
        self._pub.publish(Twist())
        payload = {
            "schema": "simenv_localization_motion_v1",
            "duration_sec": round((rospy.Time.now() - self._started).to_sec(), 3),
            "final_phase": self._phase,
            "unwrapped_yaw_rad": round(self._unwrapped_yaw, 4),
        }
        os.makedirs(os.path.dirname(os.path.abspath(self._done_file)), exist_ok=True)
        with open(self._done_file, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        self._done_pub.publish(Bool(data=True))
        rospy.loginfo("LIO motion benchmark complete")

    def _on_timer(self, _event):
        if self._finished or self._imu_yaw is None:
            return
        elapsed = (rospy.Time.now() - self._phase_started).to_sec()
        if self._phase == "wait":
            self._pub.publish(Twist())
            if elapsed >= self._start_delay:
                self._set_phase("drive1")
        elif self._phase == "drive1":
            self._drive(self._drive1, "spin90_a" if self._profile == "square" else "spin360")
        elif self._phase == "spin90_a":
            self._spin(math.pi / 2.0, "drive2")
        elif self._phase == "spin360":
            self._spin(2.0 * math.pi, "drive2")
        elif self._phase == "drive2":
            self._drive(self._drive2, "spin90_b" if self._profile == "square" else "spin180")
        elif self._phase == "spin90_b":
            self._spin(math.pi / 2.0, "drive3")
        elif self._phase == "spin180":
            self._spin(math.pi, "drive3")
        elif self._phase == "drive3":
            self._drive(self._drive3, "spin90_c" if self._profile == "square" else "settle")
        elif self._phase == "spin90_c":
            self._spin(math.pi / 2.0, "drive4")
        elif self._phase == "drive4":
            self._drive(self._drive4, "spin90_d")
        elif self._phase == "spin90_d":
            self._spin(math.pi / 2.0, "settle")
        elif self._phase == "settle":
            self._pub.publish(Twist())
            if elapsed >= 3.0:
                self._finish()


if __name__ == "__main__":
    rospy.init_node("localization_motion_benchmark")
    LocalizationMotionBenchmark()
    rospy.spin()
