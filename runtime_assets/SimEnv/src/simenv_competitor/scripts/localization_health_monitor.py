#!/usr/bin/env python3
"""Runtime-only health monitor for LIO odometry.

Uses allowed odometry, IMU and command streams. Ground truth is intentionally
absent; GT comparison belongs to the offline benchmark evaluator.
"""

import json
import math
import os
import threading
from collections import deque

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, String


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def angle_diff(target, source):
    return math.atan2(math.sin(target - source), math.cos(target - source))


class LocalizationHealthMonitor:
    def __init__(self):
        self._lock = threading.Lock()
        self._output = rospy.get_param(
            "~output_file", os.path.join(os.getcwd(), "results", "localization_health.json")
        )
        self._odom_topic = rospy.get_param("~odom_topic", "/Odometry")
        self._imu_topic = rospy.get_param("~imu_topic", "/trunk_imu")
        self._max_speed = float(rospy.get_param("~max_speed_mps", 1.5))
        self._max_jump = float(rospy.get_param("~max_jump_m", 0.45))
        self._max_yaw_disagreement = float(
            rospy.get_param("~max_yaw_disagreement_rad", 0.40)
        )
        self._stagnant_seconds = float(rospy.get_param("~stagnant_seconds", 1.5))
        self._cmd_motion_min = float(rospy.get_param("~cmd_motion_min", 0.15))
        self._minimum_samples = int(rospy.get_param("~minimum_samples", 20))
        self._rolling_window_s = float(rospy.get_param("~rolling_window_s", 5.0))
        self._recovery_seconds = float(rospy.get_param("~recovery_seconds", 2.0))
        self._odom_stale_seconds = float(
            rospy.get_param("~odom_stale_seconds", 0.40)
        )
        self._last_pose = None
        self._last_stamp = None
        self._last_motion_stamp = None
        self._imu_yaw = None
        self._yaw_offset = None
        self._last_cmd = Twist()
        self._samples = 0
        self._bad_samples = 0
        self._jump_count = 0
        self._speed_count = 0
        self._yaw_count = 0
        self._nonfinite_count = 0
        self._stagnant_events = 0
        self._stagnant_active = False
        self._quality_window = deque()
        self._stable_since = None
        self._degraded_since = None
        self._max_degraded_seconds = 0.0
        self._ever_healthy = False
        self._fault_since = None
        self._max_fault_seconds = 0.0
        self._max_step = 0.0
        self._max_speed_seen = 0.0
        self._max_yaw_error = 0.0
        self._healthy = False
        self._started = rospy.Time.now()
        self._last_status_reason = "warming"
        self._healthy_pub = rospy.Publisher(
            "/simenv/localization_healthy", Bool, queue_size=1, latch=True
        )
        self._status_pub = rospy.Publisher(
            "/simenv/localization_status", String, queue_size=1, latch=True
        )
        self._healthy_pub.publish(Bool(data=False))
        rospy.Subscriber(self._odom_topic, Odometry, self._on_odom, queue_size=50)
        rospy.Subscriber(self._imu_topic, Imu, self._on_imu, queue_size=100)
        rospy.Subscriber("/cmd_vel", Twist, self._on_cmd, queue_size=20)
        rospy.Timer(rospy.Duration(0.2), self._on_timer)
        rospy.Timer(rospy.Duration(2.0), lambda _event: self._write())
        rospy.on_shutdown(self._write)

    def _on_cmd(self, message):
        with self._lock:
            self._last_cmd = message

    def _on_imu(self, message):
        with self._lock:
            self._imu_yaw = yaw_from_quaternion(message.orientation)

    def _on_odom(self, message):
        stamp = message.header.stamp
        if stamp == rospy.Time(0):
            stamp = rospy.Time.now()
        pose = message.pose.pose
        x, y = float(pose.position.x), float(pose.position.y)
        yaw = yaw_from_quaternion(pose.orientation)
        finite = all(math.isfinite(value) for value in (x, y, yaw))
        with self._lock:
            self._samples += 1
            bad = not finite
            reasons = []
            if not finite:
                self._nonfinite_count += 1
                reasons.append("nonfinite")
            if finite and self._imu_yaw is not None:
                if self._yaw_offset is None:
                    self._yaw_offset = angle_diff(yaw, self._imu_yaw)
                imu_aligned = self._imu_yaw + self._yaw_offset
                yaw_error = abs(angle_diff(yaw, imu_aligned))
                self._max_yaw_error = max(self._max_yaw_error, yaw_error)
                if yaw_error > self._max_yaw_disagreement:
                    self._yaw_count += 1
                    reasons.append("yaw_disagreement")
                    bad = True
            if finite and self._last_pose is not None and self._last_stamp is not None:
                dt = max(1e-3, (stamp - self._last_stamp).to_sec())
                step = math.hypot(x - self._last_pose[0], y - self._last_pose[1])
                speed = step / dt
                self._max_step = max(self._max_step, step)
                self._max_speed_seen = max(self._max_speed_seen, speed)
                if step > self._max_jump:
                    self._jump_count += 1
                    reasons.append("jump")
                    bad = True
                if speed > self._max_speed and step > 0.05:
                    self._speed_count += 1
                    reasons.append("speed")
                    bad = True
                if step > 0.015:
                    self._last_motion_stamp = stamp
            elif finite:
                self._last_motion_stamp = stamp
            if finite:
                self._last_pose = (x, y, yaw)
                self._last_stamp = stamp
            if bad:
                self._bad_samples += 1
                self._last_status_reason = "+".join(reasons)
            self._quality_window.append((stamp, bad))

    def _on_timer(self, _event):
        now = rospy.Time.now()
        with self._lock:
            command_active = (
                abs(self._last_cmd.linear.x) + abs(self._last_cmd.linear.y)
                >= self._cmd_motion_min
            )
            stagnant = False
            if command_active and self._last_motion_stamp is not None:
                stagnant = (now - self._last_motion_stamp).to_sec() > self._stagnant_seconds
            if stagnant:
                if not self._stagnant_active:
                    self._stagnant_events += 1
                self._last_status_reason = "stagnant"
            self._stagnant_active = stagnant
            # During Gazebo startup simulated time can be less than the
            # rolling window; rospy.Time rejects negative values.
            cutoff = rospy.Time(max(0.0, now.to_sec() - self._rolling_window_s))
            while self._quality_window and self._quality_window[0][0] < cutoff:
                self._quality_window.popleft()
            recent_count = len(self._quality_window)
            recent_bad = sum(1 for _, bad in self._quality_window if bad)
            recent_bad_ratio = recent_bad / float(max(recent_count, 1))
            odom_stale = (
                self._last_stamp is None
                or (now - self._last_stamp).to_sec() > self._odom_stale_seconds
            )
            ready = self._samples >= self._minimum_samples
            candidate_healthy = (
                ready
                and recent_count > 0
                and recent_bad_ratio <= 0.02
                and not stagnant
                and not odom_stale
            )
            if candidate_healthy:
                if self._stable_since is None:
                    self._stable_since = now
                stable_seconds = (now - self._stable_since).to_sec()
            else:
                self._stable_since = None
                stable_seconds = 0.0
            healthy = (
                candidate_healthy
                and stable_seconds >= self._recovery_seconds
            )
            if self._ever_healthy and not candidate_healthy:
                if self._fault_since is None:
                    self._fault_since = now
                self._max_fault_seconds = max(
                    self._max_fault_seconds,
                    (now - self._fault_since).to_sec(),
                )
            elif candidate_healthy and self._fault_since is not None:
                self._max_fault_seconds = max(
                    self._max_fault_seconds,
                    (now - self._fault_since).to_sec(),
                )
                self._fault_since = None
            if self._ever_healthy and not healthy:
                if self._degraded_since is None:
                    self._degraded_since = now
                self._max_degraded_seconds = max(
                    self._max_degraded_seconds,
                    (now - self._degraded_since).to_sec(),
                )
            elif healthy and self._degraded_since is not None:
                self._max_degraded_seconds = max(
                    self._max_degraded_seconds,
                    (now - self._degraded_since).to_sec(),
                )
                self._degraded_since = None
            if healthy:
                self._ever_healthy = True
            if candidate_healthy and not healthy:
                self._last_status_reason = "recovering"
            elif odom_stale and ready:
                self._last_status_reason = "odom_stale"
            elif not ready:
                self._last_status_reason = "warming"
            if healthy:
                self._last_status_reason = "ok"
            changed = healthy != self._healthy
            self._healthy = healthy
            payload = self._payload(now)
            payload["rolling_bad_ratio"] = round(recent_bad_ratio, 6)
            payload["rolling_samples"] = recent_count
            payload["odom_stale"] = odom_stale
            payload["stable_seconds"] = round(stable_seconds, 3)
        if changed:
            self._healthy_pub.publish(Bool(data=healthy))
        self._status_pub.publish(String(data=json.dumps(payload, sort_keys=True)))

    def _payload(self, now=None):
        if now is None:
            now = rospy.Time.now()
        return {
            "schema": "simenv_localization_health_v1",
            "runtime_only": True,
            "odom_topic": self._odom_topic,
            "imu_topic": self._imu_topic,
            "duration_sec": round((now - self._started).to_sec(), 3),
            "healthy": self._healthy,
            "ever_healthy": self._ever_healthy,
            "reason": self._last_status_reason,
            "samples": self._samples,
            "bad_samples": self._bad_samples,
            "bad_ratio": round(self._bad_samples / float(max(self._samples, 1)), 6),
            "jump_count": self._jump_count,
            "speed_violation_count": self._speed_count,
            "yaw_disagreement_count": self._yaw_count,
            "nonfinite_count": self._nonfinite_count,
            "stagnant_events": self._stagnant_events,
            "stagnant_active": self._stagnant_active,
            "max_degraded_seconds": round(self._max_degraded_seconds, 3),
            "max_fault_seconds": round(self._max_fault_seconds, 3),
            "active_fault_seconds": round(
                0.0
                if self._fault_since is None
                else (now - self._fault_since).to_sec(),
                3,
            ),
            "active_degraded_seconds": round(
                0.0
                if self._degraded_since is None
                else (now - self._degraded_since).to_sec(),
                3,
            ),
            "max_step_m": round(self._max_step, 4),
            "max_speed_mps": round(self._max_speed_seen, 4),
            "max_yaw_error_rad": round(self._max_yaw_error, 4),
        }

    def _write(self):
        with self._lock:
            payload = self._payload()
        directory = os.path.dirname(os.path.abspath(self._output))
        os.makedirs(directory, exist_ok=True)
        temporary = self._output + ".tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, self._output)


if __name__ == "__main__":
    rospy.init_node("localization_health_monitor")
    LocalizationHealthMonitor()
    rospy.spin()
