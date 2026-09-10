#!/usr/bin/env python3
"""Runtime speed/clearance audit for controlled high-speed trials."""

from __future__ import annotations

import json
import math
import os
import sys
import threading
from collections import deque

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud
from std_msgs.msg import Bool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion_safety import boost_clearance_for_speed


def _roll_pitch(q):
    sinr = 2.0 * (q.w * q.x + q.y * q.z)
    cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr, cosr)
    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    return roll, pitch


def _percentile(values, ratio):
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, int(ratio * len(ordered)))])


class SpeedAudit:
    def __init__(self):
        self._lock = threading.RLock()
        self._output = os.path.abspath(
            rospy.get_param(
                "~output_file",
                os.path.join(os.getcwd(), "results", "compliant_speed_audit.json"),
            )
        )
        self._target = float(rospy.get_param("~target_max_speed", 0.62))
        self._clearance_required = float(
            rospy.get_param(
                "~boost_clearance_m", boost_clearance_for_speed(self._target)
            )
        )
        self._started = rospy.Time.now()
        self._command = 0.0
        self._gt_speed = 0.0
        self._est_speed = 0.0
        self._front = None
        self._healthy = True
        self._hold = False
        self._fall_events = 0
        self._roll = 0.0
        self._pitch = 0.0
        self._max_roll = 0.0
        self._max_pitch = 0.0
        self._last_gt = None
        self._last_est = None
        self._samples = deque(maxlen=5000)
        self._last_write = rospy.Time(0)

        rospy.Subscriber("/cmd_vel", Twist, self._on_command, queue_size=20)
        rospy.Subscriber("/ground_truth/base_w", Odometry, self._on_gt, queue_size=20)
        rospy.Subscriber("/state_estimation", Odometry, self._on_est, queue_size=20)
        rospy.Subscriber("/scan", PointCloud, self._on_scan, queue_size=2)
        rospy.Subscriber("/trunk_imu", Imu, self._on_imu, queue_size=20)
        rospy.Subscriber(
            "/simenv/localization_healthy", Bool, self._on_health, queue_size=2
        )
        rospy.Subscriber("/simenv/cmd_vel_hold", Bool, self._on_hold, queue_size=2)
        rospy.Timer(rospy.Duration(0.1), self._on_timer)
        rospy.on_shutdown(self._write)
        rospy.loginfo(
            "Speed audit armed target=%.2fm/s boost_clearance=%.2fm",
            self._target,
            self._clearance_required,
        )

    @staticmethod
    def _pose_speed(message, previous):
        stamp = message.header.stamp.to_sec()
        if stamp <= 0.0:
            stamp = rospy.Time.now().to_sec()
        p = message.pose.pose.position
        current = (stamp, float(p.x), float(p.y))
        if previous is None:
            return 0.0, current
        dt = current[0] - previous[0]
        distance = math.hypot(current[1] - previous[1], current[2] - previous[2])
        # Gazebo GT can publish at 500 Hz.  Keep the older pose until enough
        # time has accumulated instead of replacing it with every sub-20 ms
        # sample (which would otherwise make the reported speed permanently 0).
        if dt < 0.02:
            return None, previous
        if dt > 1.5 or distance > 2.5:
            return 0.0, current
        return distance / dt, current

    def _on_command(self, message):
        with self._lock:
            self._command = float(message.linear.x)

    def _on_gt(self, message):
        with self._lock:
            speed, previous = self._pose_speed(message, self._last_gt)
            self._last_gt = previous
            if speed is not None:
                self._gt_speed = speed

    def _on_est(self, message):
        with self._lock:
            speed, previous = self._pose_speed(message, self._last_est)
            self._last_est = previous
            if speed is not None:
                self._est_speed = speed

    def _on_scan(self, message):
        ranges = []
        for point in message.points:
            distance = math.hypot(float(point.x), float(point.y))
            if distance < 0.42 or distance > 12.0:
                continue
            if abs(math.degrees(math.atan2(float(point.y), float(point.x)))) <= 22.0:
                ranges.append(distance)
        with self._lock:
            self._front = _percentile(ranges, 0.82) if ranges else None

    def _on_imu(self, message):
        roll, pitch = _roll_pitch(message.orientation)
        with self._lock:
            self._roll = roll
            self._pitch = pitch
            self._max_roll = max(self._max_roll, abs(roll))
            self._max_pitch = max(self._max_pitch, abs(pitch))

    def _on_health(self, message):
        with self._lock:
            self._healthy = bool(message.data)

    def _on_hold(self, message):
        value = bool(message.data)
        with self._lock:
            if value and not self._hold:
                self._fall_events += 1
            self._hold = value

    def _on_timer(self, _event):
        now = rospy.Time.now()
        with self._lock:
            self._samples.append(
                {
                    "t": round((now - self._started).to_sec(), 3),
                    "cmd_mps": round(self._command, 4),
                    "gt_mps": round(self._gt_speed, 4),
                    "est_mps": round(self._est_speed, 4),
                    "front_m": None if self._front is None else round(self._front, 3),
                    "healthy": self._healthy,
                    "hold": self._hold,
                    "roll_rad": round(self._roll, 4),
                    "pitch_rad": round(self._pitch, 4),
                }
            )
        if (now - self._last_write).to_sec() >= 2.0:
            self._last_write = now
            self._write()

    @staticmethod
    def _sample_percentile(values, ratio):
        return _percentile(values, ratio) if values else 0.0

    def _payload(self):
        with self._lock:
            samples = list(self._samples)
            fall_events = self._fall_events
            max_roll = self._max_roll
            max_pitch = self._max_pitch
        dt = 0.1
        command = [abs(item["cmd_mps"]) for item in samples]
        gt = [item["gt_mps"] for item in samples]
        est = [item["est_mps"] for item in samples]
        moving_gt = [value for value in gt if value >= 0.03]
        fast_command = 0.8 * self._target
        high_samples = [item for item in samples if abs(item["cmd_mps"]) >= fast_command]
        front_values = [
            item["front_m"] for item in high_samples if item["front_m"] is not None
        ]
        summary = {
            "duration_sec": round(samples[-1]["t"], 3) if samples else 0.0,
            "sample_count": len(samples),
            "command_max_mps": round(max(command, default=0.0), 4),
            "command_p95_mps": round(self._sample_percentile(command, 0.95), 4),
            "gt_max_mps": round(max(gt, default=0.0), 4),
            "gt_p95_mps": round(self._sample_percentile(gt, 0.95), 4),
            "gt_moving_average_mps": round(
                sum(moving_gt) / max(1, len(moving_gt)), 4
            ),
            "est_max_mps": round(max(est, default=0.0), 4),
            "command_high_speed_sec": round(len(high_samples) * dt, 3),
            "gt_high_speed_sec": round(
                sum(value >= fast_command for value in gt) * dt, 3
            ),
            "min_front_during_high_speed_m": (
                None if not front_values else round(min(front_values), 3)
            ),
            "localization_unhealthy_sec": round(
                sum(not item["healthy"] for item in samples) * dt, 3
            ),
            "collision_proxy_sec": round(
                sum(
                    abs(item["cmd_mps"]) > 0.10
                    and item["front_m"] is not None
                    and item["front_m"] < 0.70
                    for item in samples
                )
                * dt,
                3,
            ),
            "fall_hold_events": fall_events,
            "max_abs_roll_rad": round(max_roll, 4),
            "max_abs_pitch_rad": round(max_pitch, 4),
            "uncontrolled_speed_spike": max(gt, default=0.0)
            > self._target * 1.35 + 0.15,
        }
        return {
            "schema": "simenv_speed_audit_v1",
            "runtime_only": False,
            "start_stamp": round(self._started.to_sec(), 6),
            "target_max_speed_mps": self._target,
            "boost_clearance_m": self._clearance_required,
            "summary": summary,
            "samples": samples,
        }

    def _write(self):
        try:
            payload = self._payload()
            os.makedirs(os.path.dirname(self._output), exist_ok=True)
            temporary = self._output + ".tmp"
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            os.replace(temporary, self._output)
        except Exception as error:  # noqa: BLE001 - audit is best effort
            rospy.logwarn_throttle(2.0, "Could not write speed audit: %s", error)


if __name__ == "__main__":
    rospy.init_node("speed_audit")
    SpeedAudit()
    rospy.spin()
