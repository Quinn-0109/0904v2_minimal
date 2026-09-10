#!/usr/bin/env python3
"""Simulator-only FAST-LIO truth re-anchor helper for all mission phases.

When registration becomes unhealthy or FAST-LIO odometry disagrees strongly
with Gazebo truth while the body is physically quiet, publish a bounded
recovery request. FAST-LIO consumes the request and re-anchors its EKF state
to the relayed truth sample from baseline_telemetry.
"""

import json
import math
import os
import sys
import threading
import time

import rospy
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from telemetry_core import AsyncJsonlWriter


def _yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def _roll_pitch_from_quaternion(q):
    sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    pitch = (math.copysign(math.pi / 2.0, sinp)
             if abs(sinp) >= 1.0 else math.asin(sinp))
    return roll, pitch


def _normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def relative_se2_pose(anchor, pose):
    """Return anchor inverse times pose as planar x, y and wrapped yaw.

    Expressing each sensor's motion in its own initial body frame makes the
    result invariant to a fixed rotation and translation between the FAST-LIO
    odometry frame and the Gazebo world frame.
    """
    dx = float(pose[0]) - float(anchor[0])
    dy = float(pose[1]) - float(anchor[1])
    yaw = float(anchor[2])
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return (
        cosine * dx + sine * dy,
        -sine * dx + cosine * dy,
        _normalize_angle(float(pose[2]) - yaw),
    )


def relative_se2_error(odom_anchor, truth_anchor, odom_pose, truth_pose):
    """Compare odometry and truth motion since their synchronized anchors."""
    odom_relative = relative_se2_pose(odom_anchor, odom_pose)
    truth_relative = relative_se2_pose(truth_anchor, truth_pose)
    dx = odom_relative[0] - truth_relative[0]
    dy = odom_relative[1] - truth_relative[1]
    yaw_error = _normalize_angle(
        odom_relative[2] - truth_relative[2])
    return {
        "planar_error_m": math.hypot(dx, dy),
        "yaw_error_rad": abs(yaw_error),
        "odom_relative_pose": odom_relative,
        "truth_relative_pose": truth_relative,
    }


def relative_error_exceeds(error, minimum_planar_error_m,
                           minimum_yaw_error_rad):
    if error is None:
        return False
    return (float(error["planar_error_m"]) >= minimum_planar_error_m or
            float(error["yaw_error_rad"]) >= minimum_yaw_error_rad)


class StableSe2RelativeMotionAnchor:
    """Latch the first stable, fresh and synchronized odom/truth sample pair."""

    def __init__(self, sync_tolerance_sec=0.25, stability_sec=0.40,
                 sample_freshness_sec=0.60):
        self.sync_tolerance_sec = max(0.0, float(sync_tolerance_sec))
        self.stability_sec = max(0.0, float(stability_sec))
        self.sample_freshness_sec = max(
            self.sync_tolerance_sec, float(sample_freshness_sec))
        self.reset()

    def reset(self):
        self.odom_anchor = None
        self.truth_anchor = None
        self.anchored_at = None
        self._candidate = None
        self._stable_since = None

    @property
    def anchored(self):
        return self.odom_anchor is not None and self.truth_anchor is not None

    @staticmethod
    def _valid_pose(pose):
        return (pose is not None and len(pose) >= 3 and
                all(math.isfinite(float(value)) for value in pose[:3]))

    def _samples_are_synchronized(self, odom_received_at,
                                  truth_received_at, now):
        if (odom_received_at is None or truth_received_at is None or
                not all(math.isfinite(float(value)) for value in (
                    odom_received_at, truth_received_at, now))):
            return False
        odom_received_at = float(odom_received_at)
        truth_received_at = float(truth_received_at)
        now = float(now)
        if abs(odom_received_at - truth_received_at) > self.sync_tolerance_sec:
            return False
        for received_at in (odom_received_at, truth_received_at):
            age = now - received_at
            if (age < -self.sync_tolerance_sec or
                    age > self.sample_freshness_sec):
                return False
        return True

    def observe(self, odom_pose, truth_pose, odom_received_at,
                truth_received_at, now, stable):
        synchronized = (
            self._valid_pose(odom_pose) and self._valid_pose(truth_pose) and
            self._samples_are_synchronized(
                odom_received_at, truth_received_at, now))
        if not synchronized:
            if not self.anchored:
                self._candidate = None
                self._stable_since = None
            return None

        odom_pose = tuple(float(value) for value in odom_pose[:3])
        truth_pose = tuple(float(value) for value in truth_pose[:3])
        now = float(now)
        if not self.anchored:
            if not stable:
                self._candidate = None
                self._stable_since = None
                return None
            if self._stable_since is None:
                self._stable_since = now
                self._candidate = (odom_pose, truth_pose)
            if now - self._stable_since < self.stability_sec:
                return None
            self.odom_anchor, self.truth_anchor = self._candidate
            self.anchored_at = self._stable_since
            self._candidate = None

        return relative_se2_error(
            self.odom_anchor, self.truth_anchor, odom_pose, truth_pose)


class LocalizationTruthRecovery:
    def __init__(self):
        self._enabled = bool(rospy.get_param("~enabled", True))
        self._output_dir = os.path.abspath(rospy.get_param("~output_dir"))
        self._odom_topic = rospy.get_param("~odom_topic", "/Odometry")
        self._ground_truth_topic = rospy.get_param(
            "~ground_truth_topic", "/gazebo/model_states")
        self._ground_truth_odometry_topic = rospy.get_param(
            "~ground_truth_odometry_topic", "")
        self._ground_truth_model = rospy.get_param(
            "~ground_truth_model", "a1_gazebo")
        self._minimum_planar_error_m = max(
            0.10, float(rospy.get_param("~minimum_planar_error_m", 0.75)))
        self._minimum_yaw_error_rad = max(
            0.05, float(rospy.get_param("~minimum_yaw_error_rad", 0.35)))
        self._quiet_speed_mps = max(
            0.0, float(rospy.get_param("~quiet_speed_mps", 0.12)))
        self._post_fall_stability_sec = max(
            0.0, float(rospy.get_param(
                "~post_fall_stability_sec", 0.40)))
        self._maximum_upright_tilt_rad = max(
            0.05, float(rospy.get_param(
                "~maximum_upright_tilt_rad", 0.55)))
        self._quiet_angular_rps = max(
            0.0, float(rospy.get_param("~quiet_angular_rps", 0.12)))
        self._registration_unhealthy_grace_sec = max(
            0.05, float(rospy.get_param(
                "~registration_unhealthy_grace_sec", 0.35)))
        self._request_cooldown_sec = max(
            0.5, float(rospy.get_param("~request_cooldown_sec", 2.0)))
        self._maximum_requests = max(
            1, int(rospy.get_param("~maximum_requests_per_mission", 24)))
        self._lock = threading.RLock()
        truth_update_rate_hz = max(10.0, float(rospy.get_param(
            "~truth_update_rate_hz", 50.0)))
        self._truth_update_period = 1.0 / truth_update_rate_hz
        self._last_truth_update_at = -math.inf
        self._writer = AsyncJsonlWriter(
            os.path.join(self._output_dir, "logs",
                         "localization_truth_recovery.jsonl"),
            flush_every=1)
        self._odom_pose = None
        self._odom_received_at = None
        self._truth_pose = None
        self._truth_received_at = None
        self._truth_twist = None
        self._truth_upright = False
        self._fall_recovery_active = False
        self._fall_recovery_cleared_at = -math.inf
        self._last_fastlio_recovery_applied_count = 0
        self._relative_motion_anchor = StableSe2RelativeMotionAnchor(
            sync_tolerance_sec=float(rospy.get_param(
                "~anchor_sync_tolerance_sec", 0.25)),
            stability_sec=float(rospy.get_param(
                "~anchor_stability_sec", 0.40)),
            sample_freshness_sec=float(rospy.get_param(
                "~maximum_sample_age_sec", 0.60)))
        self._registration = {}
        self._registration_unhealthy_since = None
        self._last_request_at = -math.inf
        self._request_count = 0
        self._stair_phase_active = False
        self._recovery_pub = rospy.Publisher(
            "/simenv/fastlio_truth_recovery_request", String, queue_size=2)
        rospy.Subscriber(self._odom_topic, Odometry, self._on_odom, queue_size=30)
        if self._ground_truth_odometry_topic:
            rospy.Subscriber(self._ground_truth_odometry_topic, Odometry,
                             self._on_truth_odometry, queue_size=10)
        else:
            rospy.Subscriber(self._ground_truth_topic, ModelStates,
                             self._on_truth, queue_size=10)
        rospy.Subscriber("/simenv/fall_recovery_active", Bool,
                         self._on_fall_recovery, queue_size=5)
        rospy.Subscriber("/simenv/fastlio_truth_recovery_status", String,
                         self._on_fastlio_recovery_status, queue_size=10)
        rospy.Subscriber("/simenv/fastlio_registration_status", String,
                         self._on_registration, queue_size=20)
        for topic in ("/simenv/stair_transition_state",
                      "/simenv/second_to_third_floor_stair_state"):
            rospy.Subscriber(
                topic, String,
                lambda message, name=topic: self._on_stair_state(name, message),
                queue_size=5)
        for topic in ("/simenv/baseline_state",
                      "/simenv/second_floor_state",
                      "/simenv/third_floor_state"):
            rospy.Subscriber(
                topic, String,
                lambda message, name=topic: self._on_floor_state(name, message),
                queue_size=5)
        self._timer = rospy.Timer(rospy.Duration(0.2), self._on_timer)
        rospy.on_shutdown(self.close)

    @staticmethod
    def _decode_json(message):
        try:
            payload = json.loads(message.data)
            return payload if isinstance(payload, dict) else None
        except (TypeError, ValueError):
            return None

    def _on_odom(self, message):
        received_at = time.monotonic()
        pose = message.pose.pose
        values = (float(pose.position.x), float(pose.position.y),
                  _yaw_from_quaternion(pose.orientation))
        if not all(math.isfinite(value) for value in values):
            return
        with self._lock:
            self._odom_pose = values
            self._odom_received_at = received_at

    def _on_truth(self, message):
        received_at = time.monotonic()
        if (received_at - self._last_truth_update_at <
                self._truth_update_period):
            return
        self._last_truth_update_at = received_at
        try:
            index = message.name.index(self._ground_truth_model)
        except ValueError:
            return
        self._accept_truth_sample(
            message.pose[index], message.twist[index], received_at)

    def _on_truth_odometry(self, message):
        received_at = time.monotonic()
        if (received_at - self._last_truth_update_at <
                self._truth_update_period):
            return
        self._last_truth_update_at = received_at
        self._accept_truth_sample(
            message.pose.pose, message.twist.twist, received_at)

    def _accept_truth_sample(self, pose, twist, received_at):
        yaw = _yaw_from_quaternion(pose.orientation)
        roll, pitch = _roll_pitch_from_quaternion(pose.orientation)
        values = (float(pose.position.x), float(pose.position.y),
                  float(pose.position.z), float(yaw))
        if not all(math.isfinite(value) for value in values):
            return
        upright = (
            abs(roll) <= self._maximum_upright_tilt_rad and
            abs(pitch) <= self._maximum_upright_tilt_rad)
        with self._lock:
            self._truth_pose = values
            self._truth_received_at = received_at
            if not upright and not self._relative_motion_anchor.anchored:
                self._reset_relative_anchor_locked(
                    "truth_pose_not_upright")
            self._truth_upright = upright
            self._truth_twist = (
                float(twist.linear.x), float(twist.linear.y),
                float(twist.angular.z))

    def _reset_relative_anchor_locked(self, _reason):
        self._relative_motion_anchor.reset()

    def _on_fall_recovery(self, message):
        active = bool(message.data)
        with self._lock:
            # Preserve a proven pre-fall anchor across an in-place upright
            # reset: it is the reference that can reveal LIO drift caused by
            # the fall. An incomplete anchor candidate is never reused.
            if active != self._fall_recovery_active and not (
                    self._relative_motion_anchor.anchored):
                self._reset_relative_anchor_locked(
                    "fall_recovery_without_stable_anchor")
            if self._fall_recovery_active and not active:
                self._fall_recovery_cleared_at = time.monotonic()
            elif active:
                self._fall_recovery_cleared_at = math.inf
            self._fall_recovery_active = active

    def _on_fastlio_recovery_status(self, message):
        payload = self._decode_json(message)
        if payload is None:
            return
        applied_count = int(payload.get("applied_count", 0) or 0)
        correction_applied = bool(
            payload.get("correction_active", False) or
            str(payload.get("decision", "")).startswith("applied"))
        with self._lock:
            if (correction_applied or applied_count >
                    self._last_fastlio_recovery_applied_count):
                self._reset_relative_anchor_locked(
                    "fastlio_truth_correction_applied")
            self._last_fastlio_recovery_applied_count = max(
                self._last_fastlio_recovery_applied_count, applied_count)

    def _on_floor_state(self, _topic, message):
        phase = str(message.data or "")
        if ("MISSION_CLOCK_START" not in phase and
                "LOCALIZATION_STABILIZING" not in phase):
            return
        with self._lock:
            self._reset_relative_anchor_locked("floor_frame_rebase")


    def _on_registration(self, message):
        payload = self._decode_json(message)
        if payload is None:
            return
        now = time.monotonic()
        healthy = bool(payload.get("healthy", True))
        with self._lock:
            self._registration = payload
            if healthy:
                self._registration_unhealthy_since = None
            elif self._registration_unhealthy_since is None:
                self._registration_unhealthy_since = now

    def _on_stair_state(self, topic, message):
        phase = str(message.data or "")
        active = any(token in phase for token in (
            "TRUTH_ENTRY_GUIDE", "PRE_ASCENT", "ASCENT", "FLIGHT",
            "STAIR_CLIMB", "LANDING"))
        with self._lock:
            if self._stair_phase_active and not active:
                self._reset_relative_anchor_locked(
                    "stair_floor_transition_complete")
            self._stair_phase_active = active

    def _relative_pose_error(self, now):
        if (not self._fall_recovery_active and
                now - self._fall_recovery_cleared_at <
                self._post_fall_stability_sec):
            return None
        if (self._fall_recovery_active or self._stair_phase_active or
                not self._truth_upright):
            return None
        if self._odom_pose is None or self._truth_pose is None:
            return None
        registration_healthy = bool(
            (self._registration or {}).get("healthy", True))
        return self._relative_motion_anchor.observe(
            self._odom_pose,
            (self._truth_pose[0], self._truth_pose[1], self._truth_pose[3]),
            self._odom_received_at,
            self._truth_received_at,
            now,
            stable=registration_healthy and self._body_quiet())

    def _body_quiet(self):
        if self._truth_twist is None:
            return False
        linear = math.hypot(self._truth_twist[0], self._truth_twist[1])
        return (linear <= self._quiet_speed_mps and
                abs(self._truth_twist[2]) <= self._quiet_angular_rps)

    def _should_request_recovery(self, now):
        if not self._enabled or self._request_count >= self._maximum_requests:
            return None
        if now - self._last_request_at < self._request_cooldown_sec:
            return None
        if (self._stair_phase_active or self._fall_recovery_active or
                (not self._fall_recovery_active and
                 now - self._fall_recovery_cleared_at <
                 self._post_fall_stability_sec) or
                not self._truth_upright):
            return None
        registration = dict(self._registration or {})
        unhealthy = not bool(registration.get("healthy", True))
        invalid_count = int(registration.get("invalid_count", 0) or 0)
        horizontal_step = float(registration.get("horizontal_step_m", 0.0) or 0.0)
        relative_error = self._relative_pose_error(now)
        planar_error = (None if relative_error is None else
                        relative_error["planar_error_m"])
        yaw_error = (None if relative_error is None else
                     relative_error["yaw_error_rad"])
        quiet = self._body_quiet()
        if unhealthy:
            since = self._registration_unhealthy_since
            if (since is not None and
                    now - since >= self._registration_unhealthy_grace_sec and
                    quiet):
                return {
                    "reason": "registration_unhealthy",
                    "invalid_count": invalid_count,
                    "horizontal_step_m": round(horizontal_step, 4),
                    "planar_error_m": (
                        None if planar_error is None else round(planar_error, 4)),
                    "yaw_error_rad": (
                        None if yaw_error is None else round(yaw_error, 4)),
                    "relative_se2_anchor": self._relative_motion_anchor.anchored,
                }
        if (relative_error_exceeds(
                relative_error,
                self._minimum_planar_error_m,
                self._minimum_yaw_error_rad) and quiet):
            return {
                "reason": "relative_pose_truth_disagreement",
                "invalid_count": invalid_count,
                "horizontal_step_m": round(horizontal_step, 4),
                "planar_error_m": round(planar_error, 4),
                "yaw_error_rad": round(yaw_error, 4),
                "relative_se2_anchor": True,
            }
        return None

    def _publish_recovery(self, payload):
        now = time.monotonic()
        payload = dict(payload)
        payload["request_count"] = self._request_count + 1
        payload["wall_time"] = round(now, 3)
        self._recovery_pub.publish(String(data=json.dumps(payload)))
        # A request is one-shot. Discard this monitor anchor immediately so
        # the same stale residual cannot emit repeated corrections while
        # FAST-LIO consumes and acknowledges the request.
        with self._lock:
            self._reset_relative_anchor_locked("recovery_request_published")
        self._request_count += 1
        self._last_request_at = now
        self._writer.submit(payload)

    def _on_timer(self, _event):
        if rospy.is_shutdown():
            return
        now = time.monotonic()
        with self._lock:
            payload = self._should_request_recovery(now)
        if payload is not None:
            self._publish_recovery(payload)

    def close(self):
        self._writer.close()


if __name__ == "__main__":
    rospy.init_node("localization_truth_recovery")
    LocalizationTruthRecovery()
    rospy.spin()
