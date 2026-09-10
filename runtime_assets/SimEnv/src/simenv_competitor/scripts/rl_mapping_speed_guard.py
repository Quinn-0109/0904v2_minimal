#!/usr/bin/env python3
"""Fail-safe cmd_vel owner for the official-RL Room0 SLAM diagnostic."""

import json
import math
import os
import sys
import threading
import time

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String
from tf.transformations import euler_from_quaternion

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion_safety import longitudinal_slew  # noqa: E402
from room0_mapping_core import rl_guard_command  # noqa: E402


class RlMappingSpeedGuard:
    def __init__(self):
        self._output_dir = os.path.abspath(rospy.get_param("~output_dir"))
        os.makedirs(self._output_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._desired = Twist()
        self._desired_received = 0.0
        self._pose_received = 0.0
        self._pose = None
        self._armed = False
        self._ever_armed = False
        self._aborted = False
        self._abort_reason = ""
        self._recovery_hold = False
        self._recovery_clear_since = None
        self._last_linear = 0.0
        self._last_lateral = 0.0
        self._last_tick = time.monotonic()
        self._start = time.monotonic()
        self._samples = []
        self._events = []
        self._written = False
        self._maximum_linear = float(rospy.get_param("~maximum_linear", 0.35))
        self._maximum_lateral = float(rospy.get_param("~maximum_lateral", 0.35))
        self._maximum_yaw = float(rospy.get_param("~maximum_yaw", 0.35))
        self._acceleration = float(rospy.get_param("~acceleration", 0.20))
        self._deceleration = float(rospy.get_param("~deceleration", 0.80))
        self._warning_tilt = float(rospy.get_param("~warning_tilt", 0.22))
        self._abort_tilt = float(rospy.get_param("~abort_tilt", 0.35))
        self._minimum_height = float(rospy.get_param("~minimum_height", 0.27))
        self._recovery_tilt = float(rospy.get_param("~recovery_tilt", 0.14))
        self._recovery_release_tilt = float(
            rospy.get_param("~recovery_release_tilt", 0.08)
        )
        self._recovery_minimum_height = float(
            rospy.get_param("~recovery_minimum_height", 0.285)
        )
        self._recovery_release_seconds = float(
            rospy.get_param("~recovery_release_seconds", 0.8)
        )

        self._output = rospy.Publisher("/cmd_vel", Twist, queue_size=5)
        self._status = rospy.Publisher(
            "/simenv/rl_motion_status", String, queue_size=2, latch=True
        )
        self._abort = rospy.Publisher(
            "/simenv/rl_motion_abort", String, queue_size=1, latch=True
        )
        rospy.Subscriber(
            "/simenv/desired_cmd_vel", Twist, self._on_desired, queue_size=10
        )
        rospy.Subscriber(
            "/simenv/rl_motion_armed", Bool, self._on_armed, queue_size=2
        )
        self._pose_topic = rospy.get_param("~pose_topic", "/ground_truth/base_w")
        rospy.Subscriber(self._pose_topic, Odometry, self._on_pose, queue_size=20)
        rospy.Timer(rospy.Duration(0.05), self._on_timer)
        rospy.on_shutdown(self._shutdown)
        rospy.loginfo(
            "Official-RL Room0 speed guard owns /cmd_vel (v=%.2f, w=%.2f)",
            self._maximum_linear, self._maximum_yaw,
        )

    def _on_desired(self, message):
        with self._lock:
            self._desired = message
            self._desired_received = time.monotonic()

    def _on_armed(self, message):
        with self._lock:
            self._armed = bool(message.data)
            self._ever_armed = self._ever_armed or self._armed

    def _on_pose(self, message):
        pose = message.pose.pose
        q = pose.orientation
        roll, pitch, _ = euler_from_quaternion((q.x, q.y, q.z, q.w))
        twist = message.twist.twist
        actual_linear = math.hypot(twist.linear.x, twist.linear.y)
        actual_yaw = float(twist.angular.z)
        with self._lock:
            self._pose = (
                float(pose.position.z), float(roll), float(pitch),
                float(actual_linear), actual_yaw,
            )
            self._pose_received = time.monotonic()

    def _publish_abort(self, reason):
        payload = {"aborted": True, "reason": reason, "t": time.monotonic() - self._start}
        self._abort.publish(String(data=json.dumps(payload, sort_keys=True)))

    def _on_timer(self, _event):
        now = time.monotonic()
        with self._lock:
            desired = self._desired
            desired_age = now - self._desired_received if self._desired_received else math.inf
            pose_age = now - self._pose_received if self._pose_received else math.inf
            pose = self._pose
            armed = self._armed
            aborted = self._aborted
            previous_reason = self._abort_reason
        z, roll, pitch, actual_linear, actual_yaw = (
            pose if pose is not None
            else (math.nan, math.nan, math.nan, math.nan, math.nan)
        )
        tilt = max(abs(roll), abs(pitch)) if all(
            math.isfinite(value) for value in (roll, pitch)
        ) else math.inf
        if not armed:
            self._recovery_hold = False
            self._recovery_clear_since = None
        elif not aborted:
            if (
                not self._recovery_hold
                and (tilt > self._recovery_tilt or z < self._recovery_minimum_height)
            ):
                self._recovery_hold = True
                self._recovery_clear_since = None
                with self._lock:
                    self._events.append({
                        "t": now - self._start, "event": "stability_hold",
                        "tilt": tilt, "z": z,
                    })
                rospy.logwarn("RL stability hold: tilt=%.3f z=%.3f", tilt, z)
            if self._recovery_hold:
                recovered = (
                    tilt < self._recovery_release_tilt
                    and z > self._recovery_minimum_height + 0.01
                )
                if recovered:
                    self._recovery_clear_since = self._recovery_clear_since or now
                    if now - self._recovery_clear_since >= self._recovery_release_seconds:
                        self._recovery_hold = False
                        self._recovery_clear_since = None
                        with self._lock:
                            self._events.append({
                                "t": now - self._start, "event": "stability_recovered"
                            })
                        rospy.loginfo("RL stability recovered; releasing cmd_vel hold")
                else:
                    self._recovery_clear_since = None
        linear, yaw, reason, new_abort = rl_guard_command(
            desired.linear.x, desired.angular.z,
            armed=armed, desired_age=desired_age, pose_age=pose_age,
            z=z, roll=roll, pitch=pitch, aborted=aborted,
            stability_hold=self._recovery_hold,
            warning_tilt=self._warning_tilt, abort_tilt=self._abort_tilt,
            minimum_height=self._minimum_height,
            maximum_linear=self._maximum_linear, maximum_yaw=self._maximum_yaw,
        )
        if new_abort and not aborted:
            with self._lock:
                self._aborted = True
                self._abort_reason = reason
                self._events.append({"t": now - self._start, "event": "abort", "reason": reason})
            self._publish_abort(reason)
            rospy.logerr("Official-RL motion abort: %s", reason)
        elif aborted:
            reason = previous_reason or reason

        lateral = max(-self._maximum_lateral, min(
            self._maximum_lateral, float(desired.linear.y)
        ))
        if reason == "tilt_limited":
            lateral = max(-0.10, min(0.10, lateral))

        dt = max(0.0, now - self._last_tick)
        self._last_tick = now
        immediate_zero = reason not in ("normal", "tilt_limited")
        applied_linear = longitudinal_slew(
            self._last_linear, linear, dt,
            acceleration=self._acceleration,
            deceleration=self._deceleration,
            immediate_zero=immediate_zero,
        )
        applied_lateral = longitudinal_slew(
            self._last_lateral, lateral, dt,
            acceleration=self._acceleration,
            deceleration=self._deceleration,
            immediate_zero=immediate_zero,
        )
        self._last_linear = applied_linear
        self._last_lateral = applied_lateral
        command = Twist()
        command.linear.x = applied_linear
        command.linear.y = applied_lateral
        command.angular.z = yaw if not immediate_zero else 0.0
        self._output.publish(command)

        sample = {
            "t": round(now - self._start, 4), "armed": armed,
            "desired_linear": float(desired.linear.x),
            "desired_lateral": float(desired.linear.y),
            "desired_yaw": float(desired.angular.z),
            "output_linear": float(command.linear.x),
            "output_lateral": float(command.linear.y),
            "output_yaw": float(command.angular.z),
            "actual_linear": None if not math.isfinite(actual_linear) else actual_linear,
            "actual_yaw": None if not math.isfinite(actual_yaw) else actual_yaw,
            "z": None if not math.isfinite(z) else z,
            "roll": None if not math.isfinite(roll) else roll,
            "pitch": None if not math.isfinite(pitch) else pitch,
            "reason": reason,
        }
        with self._lock:
            self._samples.append(sample)
        self._status.publish(String(data=json.dumps(sample, sort_keys=True)))

    @staticmethod
    def _write_json(path, payload):
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

    def _write_artifacts(self):
        with self._lock:
            if self._written:
                return
            self._written = True
            samples = list(self._samples)
            events = list(self._events)
            aborted = self._aborted
            abort_reason = self._abort_reason
            ever_armed = self._ever_armed
        armed_samples = [sample for sample in samples if sample["armed"] and sample["z"] is not None]
        max_roll = max((abs(sample["roll"]) for sample in armed_samples), default=None)
        max_pitch = max((abs(sample["pitch"]) for sample in armed_samples), default=None)
        min_height = min((sample["z"] for sample in armed_samples), default=None)
        report = {
            "schema": "simenv_room0_rl_motion_stability_v1",
            "passed": bool(ever_armed and armed_samples and not aborted),
            "ever_armed": ever_armed,
            "aborted": aborted,
            "abort_reason": abort_reason,
            "max_abs_roll_rad": max_roll,
            "max_abs_pitch_rad": max_pitch,
            "minimum_height_m": min_height,
            "sample_count": len(samples),
            "armed_sample_count": len(armed_samples),
            "events": events,
        }
        self._write_json(os.path.join(self._output_dir, "motion_stability.json"), report)
        self._write_json(os.path.join(self._output_dir, "speed_audit.json"), {
            "schema": "simenv_room0_rl_speed_audit_v1", "samples": samples,
        })
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            timed = [sample for sample in samples if sample["z"] is not None]
            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            if timed:
                ts = [sample["t"] for sample in timed]
                axes[0].plot(ts, [sample["desired_linear"] for sample in timed], label="desired")
                axes[0].plot(ts, [sample["output_linear"] for sample in timed], label="guarded")
                axes[0].plot(ts, [sample["actual_linear"] for sample in timed], label="actual")
                axes[0].set_ylabel("linear [m/s]")
                axes[0].legend()
                axes[1].plot(ts, [sample["roll"] for sample in timed], label="roll")
                axes[1].plot(ts, [sample["pitch"] for sample in timed], label="pitch")
                axes[1].axhline(self._abort_tilt, color="r", ls="--", lw=0.8)
                axes[1].axhline(-self._abort_tilt, color="r", ls="--", lw=0.8)
                axes[1].set_ylabel("attitude [rad]")
                axes[1].legend()
                axes[2].plot(ts, [sample["z"] for sample in timed], label="body z")
                axes[2].axhline(self._minimum_height, color="r", ls="--", lw=0.8)
                axes[2].set_ylabel("z [m]")
                axes[2].set_xlabel("time [s]")
            for axis in axes:
                axis.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(os.path.join(self._output_dir, "motion_stability.png"), dpi=180)
            plt.close(fig)
        except Exception as error:
            rospy.logerr("Failed to render RL motion stability plot: %s", error)

    def _shutdown(self):
        try:
            for _ in range(5):
                self._output.publish(Twist())
                time.sleep(0.02)
        finally:
            self._write_artifacts()


if __name__ == "__main__":
    rospy.init_node("rl_mapping_speed_guard")
    RlMappingSpeedGuard()
    rospy.spin()
