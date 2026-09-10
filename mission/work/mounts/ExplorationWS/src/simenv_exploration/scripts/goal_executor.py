#!/usr/bin/env python3
"""Execute one FUEL-lite PoseStamped goal through the official A1 RL interface."""

import json
import math
import os
import sys
import tempfile
import threading
import time

import rospy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32, String

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from goal_executor_core import (  # noqa: E402
    ProgressWatchdog, command_for_goal, quaternion_yaw, slew,
)


class GoalExecutor:
    def __init__(self):
        # Goal replacement and all command/result side effects are serialized.
        # A timer callback for goal N must never publish a velocity, zero, or
        # latched result after goal N+1 has been accepted.
        self._lock = threading.RLock()
        # JSON serialization is deliberately separate from the motion-state
        # lock. Rewriting a long sample history must not block the 20 Hz
        # command timer or acceptance of the next goal.
        self._log_write_lock = threading.Lock()
        self._log_revision = 0
        self._last_written_log_revision = -1
        self._output_dir = os.path.abspath(rospy.get_param("~output_dir", "results/goal_executor/latest"))
        os.makedirs(self._output_dir, exist_ok=True)
        log_file = os.path.basename(rospy.get_param(
            "~log_file", "goal_execution_log.json"))
        self._log_path = os.path.join(self._output_dir, log_file)
        self._odom_topic = rospy.get_param("~odom_topic", "/Odometry")
        self._goal_topic = rospy.get_param("~goal_topic", "/exploration_goal")
        self._cancel_topic = rospy.get_param(
            "~cancel_topic", "/simenv/cancel_exploration_goal")
        self._command_topic = rospy.get_param("~command_topic", "/cmd_vel")
        self._goal_tolerance = float(rospy.get_param("~goal_tolerance", 0.30))
        self._minimum_speed = float(rospy.get_param("~minimum_speed", 0.30))
        self._maximum_speed = float(rospy.get_param("~maximum_speed", 0.35))
        self._nominal_maximum_speed = self._maximum_speed
        # The semantic manager publishes a per-goal limit.  Its cap is a
        # launch parameter so a timed mission can raise it deliberately;
        # never silently reinstate a lower hard-coded cap here.
        self._dynamic_speed_cap = max(
            self._minimum_speed,
            float(rospy.get_param("~dynamic_speed_cap", self._maximum_speed)))
        self._distance_gain = float(rospy.get_param("~distance_gain", 0.40))
        self._heading_gain = float(rospy.get_param("~heading_gain", 0.35))
        self._maximum_yaw_rate = float(rospy.get_param("~maximum_yaw_rate", 0.20))
        self._lateral_speed_limit = float(rospy.get_param(
            "~lateral_speed_limit", self._maximum_speed))
        self._heading_slowdown_threshold = float(rospy.get_param(
            "~heading_slowdown_threshold", 0.45))
        self._turning_speed_limit = float(rospy.get_param(
            "~turning_speed_limit", min(0.65, self._maximum_speed)))
        # MotionConsistencyGuard compares commanded travel against FAST-LIO
        # odometry and also receives IMU samples.  It is not a pose resetter;
        # use its short degradation signal to reduce excitation while the
        # LiDAR/IMU estimator regains correspondences.
        self._localization_speed_cap = float(rospy.get_param(
            "~localization_degraded_speed_cap", 0.85))
        self._acceleration = float(rospy.get_param("~acceleration", 0.20))
        self._deceleration = float(rospy.get_param("~deceleration", 0.60))
        # A semantic doorway may replace an in-flight corridor goal.  Stopping
        # a fast gait by publishing one zero Twist tipped the robot in run20.
        # Keep ownership of the old goal until both the commanded and measured
        # speeds have settled, then acknowledge the manager's cancellation.
        self._preemption_deceleration = float(rospy.get_param(
            "~preemption_deceleration", 0.55))
        self._preemption_command_speed = float(rospy.get_param(
            "~preemption_settle_command_speed", 0.08))
        self._preemption_measured_speed = float(rospy.get_param(
            "~preemption_settle_measured_speed", 0.18))
        self._preemption_settle_seconds = float(rospy.get_param(
            "~preemption_settle_seconds", 0.30))
        self._preemption_timeout = float(rospy.get_param(
            "~preemption_timeout", 4.0))
        self._odom_timeout = float(rospy.get_param("~odom_timeout", 0.50))
        self._goal_timeout = float(rospy.get_param("~goal_timeout", 180.0))
        self._maximum_pose_jump = float(rospy.get_param("~maximum_pose_jump", 0.75))
        self._maximum_vertical_pose_jump = float(rospy.get_param(
            "~maximum_vertical_pose_jump", 0.45))
        self._minimum_pose_z = float(rospy.get_param("~minimum_pose_z", -0.80))
        self._maximum_pose_z = float(rospy.get_param("~maximum_pose_z", 1.20))
        self._shutdown_on_result = bool(rospy.get_param("~shutdown_on_result", False))
        self._registration_abort_invalid_count = max(
            1, int(rospy.get_param(
                "~registration_abort_invalid_count", 3)))
        self._arm_rl_guard = bool(rospy.get_param("~arm_rl_guard", False))
        self._watchdog = ProgressWatchdog(
            rospy.get_param("~progress_timeout", 8.0),
            rospy.get_param("~minimum_progress", 0.05),
        )
        self._maximum_stall_recoveries = int(rospy.get_param(
            "~maximum_stall_recoveries", 1))
        self._stall_recovery_seconds = float(rospy.get_param(
            "~stall_recovery_seconds", 0.6))
        self._stall_recovery_speed = float(rospy.get_param(
            "~stall_recovery_speed", 0.16))
        # Optional external-driver arbitration: while Bool true is published
        # on this topic, the executor publishes no /cmd_vel at all (not even
        # the idle zero) and rejects new goals with reason
        # "external_driver_active".  Empty topic disables the mechanism.
        self._external_pause_topic = str(rospy.get_param(
            "~external_pause_topic", "")).strip()

        self._pose = None
        self._pose_frame = ""
        self._pose_wall_time = 0.0
        self._measured_speed = math.inf
        self._previous_pose = None
        self._goal = None
        self._start_pose = None
        self._goal_start = None
        self._goal_frame = ""
        self._goal_sequence = 0
        self._goal_stamp = 0.0
        self._goal_generation = 0
        self._pose_fault = ""
        self._registration_healthy = True
        self._registration_invalid_count = 0
        self._localization_degraded = False
        self._samples = []
        self._events = []
        self._finished = False
        self._last_command = [0.0, 0.0, 0.0]
        self._last_tick = time.monotonic()
        self._locomotion_ready = False
        self._stall_recoveries = 0
        self._stall_recovery_until = None
        self._stall_recovery_command = [0.0, 0.0]
        self._rescan = None
        self._pending_cancel = None
        self._external_paused = False

        self._command_pub = rospy.Publisher(self._command_topic, Twist, queue_size=5)
        self._desired_command_pub = rospy.Publisher(
            "/simenv/desired_cmd_vel", Twist, queue_size=5)
        self._status_pub = rospy.Publisher(
            "/simenv/goal_execution_status", String, queue_size=2, latch=True
        )
        self._reached_pub = rospy.Publisher(
            "/simenv/goal_reached", Bool, queue_size=1, latch=True
        )
        self._compat_reached_pub = rospy.Publisher(
            "/goal_reached", Bool, queue_size=1, latch=True
        )
        self._result_pub = rospy.Publisher(
            "/simenv/goal_execution_result", String, queue_size=1, latch=True
        )
        self._rescan_result_pub = rospy.Publisher(
            "/simenv/local_rescan_result", String, queue_size=2, latch=True)
        self._armed_pub = rospy.Publisher(
            "/simenv/rl_motion_armed", Bool, queue_size=1, latch=True
        ) if self._arm_rl_guard else None
        rospy.Subscriber(self._odom_topic, Odometry, self._on_odom, queue_size=20)
        rospy.Subscriber(self._goal_topic, PoseStamped, self._on_goal, queue_size=2)
        rospy.Subscriber(self._cancel_topic, String, self._on_cancel, queue_size=5)
        rospy.Subscriber("/simenv/local_rescan_request", String,
                         self._on_rescan_request, queue_size=2)
        rospy.Subscriber("/simenv/goal_speed_limit", Float32,
                         self._on_speed_limit, queue_size=2)
        rospy.Subscriber("/simenv/goal_distance_gain", Float32,
                         self._on_distance_gain, queue_size=2)
        rospy.Subscriber("/simenv/goal_lateral_speed_limit", Float32,
                         self._on_lateral_speed_limit, queue_size=2)
        rospy.Subscriber("/simenv/goal_tolerance", Float32,
                         self._on_goal_tolerance, queue_size=2)
        rospy.Subscriber("/locomotion_ready", Bool,
                         self._on_locomotion_ready, queue_size=2)
        rospy.Subscriber("/simenv/fastlio_registration_status", String,
                         self._on_registration_status, queue_size=5)
        if self._external_pause_topic:
            rospy.Subscriber(self._external_pause_topic, Bool,
                             self._on_external_pause, queue_size=2)
        rospy.Subscriber("/simenv/localization_degraded", Bool,
                         self._on_localization_degraded, queue_size=2)
        # F1 retains the launch-time z envelope.  The isolated F2 manager
        # explicitly translates it only after corridor-entry handoff.
        rospy.Subscriber("/simenv/goal_executor_floor_context", String,
                         self._on_floor_context, queue_size=2)
        rospy.Timer(rospy.Duration(0.05), self._on_timer)
        rospy.on_shutdown(self._on_shutdown)
        self._write_log("waiting_for_goal", success=False, final=False)
        rospy.loginfo(
            "Goal executor ready: goal=%s odom=%s command=%s speed=%.2f..%.2f",
            self._goal_topic, self._odom_topic, self._command_topic,
            self._minimum_speed, self._maximum_speed,
        )

    @staticmethod
    def _validated_floor_context(payload):
        """Return validated upper-floor z bounds, or None for malformed data."""
        if not isinstance(payload, dict):
            return None
        try:
            floor_index = int(payload.get("floor_index", -1))
        except (TypeError, ValueError, OverflowError):
            return None
        expected_modes = {
            1: "second_floor_exploration",
            2: "third_floor_exploration",
        }
        if payload.get("mode") != expected_modes.get(floor_index):
            return None
        try:
            minimum = float(payload["minimum_pose_z"])
            maximum = float(payload["maximum_pose_z"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if (not math.isfinite(minimum) or not math.isfinite(maximum) or
                minimum >= maximum or maximum - minimum < 0.50 or
                maximum - minimum > 5.0 or minimum < -5.0 or maximum > 10.0):
            return None
        return minimum, maximum

    def _on_floor_context(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        bounds = self._validated_floor_context(payload)
        if bounds is None:
            rospy.logwarn("Ignored invalid goal-executor floor context")
            return
        with self._lock:
            self._minimum_pose_z, self._maximum_pose_z = bounds
            # Force acquisition of one fresh F2 odometry sample.  This avoids
            # exposing the last accepted F1 pose while bounds are switching;
            # the next sample must still pass every ordinary safety check.
            self._pose = None
            self._previous_pose = None
            self._pose_wall_time = 0.0
            self._pose_fault = ""
            self._events.append({
                "event": "second_floor_pose_bounds_activated",
                "wall_time": time.time(),
                "minimum_pose_z": bounds[0],
                "maximum_pose_z": bounds[1],
            })
        rospy.loginfo(
            "Second-floor executor pose bounds activated: %.2f..%.2f m",
            bounds[0], bounds[1])

    @staticmethod
    def _pose_dict(pose):
        if pose is None:
            return None
        return {"x": pose[0], "y": pose[1], "z": pose[2], "yaw": pose[3]}

    @staticmethod
    def _atomic_json(path, payload):
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=os.path.basename(path) + ".", suffix=".tmp",
            dir=directory)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2,
                          sort_keys=True)
                stream.write("\n")
            os.replace(temporary, path)
        finally:
            # os.replace removes the temporary name on success. On an I/O
            # failure, do not leave stale multi-megabyte histories behind.
            if os.path.exists(temporary):
                try:
                    os.unlink(temporary)
                except OSError:
                    pass

    def _on_odom(self, message):
        p, q = message.pose.pose.position, message.pose.pose.orientation
        pose = (float(p.x), float(p.y), float(p.z),
                quaternion_yaw(q.x, q.y, q.z, q.w))
        now = time.monotonic()
        twist = message.twist.twist
        measured_speed = math.sqrt(
            float(twist.linear.x) ** 2 + float(twist.linear.y) ** 2 +
            float(twist.linear.z) ** 2)
        fault = ""
        if not all(math.isfinite(value) for value in pose):
            fault = "localization_nonfinite"
        elif pose[2] < self._minimum_pose_z or pose[2] > self._maximum_pose_z:
            fault = "localization_vertical_fault"
        with self._lock:
            if not fault and self._previous_pose is not None:
                elapsed = max(1e-3, now - self._pose_wall_time)
                jump = math.hypot(pose[0] - self._previous_pose[0],
                                  pose[1] - self._previous_pose[1])
                vertical_jump = abs(pose[2] - self._previous_pose[2])
                # Only reject an instantaneous discontinuity, not valid motion
                # after a delayed callback.
                if elapsed < 0.25 and jump > self._maximum_pose_jump:
                    fault = "localization_jump"
                elif (elapsed < 0.25 and
                      vertical_jump > self._maximum_vertical_pose_jump):
                    fault = "localization_vertical_fault"
            if not fault:
                self._pose = pose
                self._pose_frame = message.header.frame_id
                self._previous_pose = pose
                self._pose_wall_time = now
                self._measured_speed = (measured_speed if
                                        math.isfinite(measured_speed) else
                                        math.inf)
                self._pose_fault = ""
            else:
                self._pose_fault = fault

    def _on_goal(self, message):
        p, q = message.pose.position, message.pose.orientation
        goal = (float(p.x), float(p.y), float(p.z),
                quaternion_yaw(q.x, q.y, q.z, q.w))
        if not all(math.isfinite(value) for value in goal):
            rospy.logerr("Rejected non-finite exploration goal")
            return
        now = time.monotonic()
        with self._lock:
            if self._external_paused:
                self._events.append({
                    "event": "goal_rejected_external_driver_active",
                    "wall_time": time.time(),
                    "goal_sequence": int(message.header.seq),
                })
                self._publish_zero()
                self._result_pub.publish(String(data=json.dumps({
                    "success": False,
                    "reason": "external_driver_active",
                    "distance_error_m": (math.hypot(
                        goal[0] - self._pose[0], goal[1] - self._pose[1])
                        if self._pose is not None else None),
                    "duration_sec": 0.0,
                    "goal": self._pose_dict(goal),
                    "end_pose": self._pose_dict(self._pose),
                    "goal_sequence": int(message.header.seq),
                    "goal_stamp": float(message.header.stamp.to_sec()),
                    "goal_frame": message.header.frame_id,
                    "zero_command_stall_recovery_count": 0,
                }, sort_keys=True)))
                self._write_log("external_driver_active", success=False,
                                final=False)
                rospy.logwarn("Exploration goal rejected: "
                              "external driver owns /cmd_vel")
                return
            if not self._locomotion_ready:
                self._events.append({
                    "event": "goal_rejected_locomotion_not_ready",
                    "wall_time": time.time(),
                    "goal_sequence": int(message.header.seq),
                })
                self._publish_zero()
                # Rejection must be observable by the manager.  Previously it
                # only appeared in the executor JSON log, leaving the manager
                # waiting for the full per-goal timeout (181 s in run20).
                self._result_pub.publish(String(data=json.dumps({
                    "success": False,
                    "reason": "locomotion_not_ready",
                    "distance_error_m": (math.hypot(
                        goal[0] - self._pose[0], goal[1] - self._pose[1])
                        if self._pose is not None else None),
                    "duration_sec": 0.0,
                    "goal": self._pose_dict(goal),
                    "end_pose": self._pose_dict(self._pose),
                    "goal_sequence": int(message.header.seq),
                    "goal_stamp": float(message.header.stamp.to_sec()),
                    "goal_frame": message.header.frame_id,
                    "zero_command_stall_recovery_count": 0,
                }, sort_keys=True)))
                self._write_log("locomotion_not_ready", success=False, final=False)
                rospy.logwarn("Exploration goal rejected: /locomotion_ready is false")
                return
            pose = self._pose
            pose_frame = self._pose_frame
            # /Odometry and FUEL normally both use camera_init. An empty goal
            # frame is accepted as the odometry frame for ROS compatibility.
            self._goal = goal
            self._goal_frame = message.header.frame_id
            self._goal_sequence = int(message.header.seq)
            self._goal_stamp = float(message.header.stamp.to_sec())
            self._goal_generation += 1
            generation = self._goal_generation
            self._start_pose = pose
            self._goal_start = now
            self._samples = []
            self._events.append({
                "event": "goal_received", "wall_time": time.time(),
                "goal_sequence": self._goal_sequence,
                "goal_stamp": self._goal_stamp,
                "goal_generation": generation,
            })
            self._finished = False
            self._stall_recoveries = 0
            self._stall_recovery_until = None
            self._stall_recovery_command = [0.0, 0.0]
            self._pending_cancel = None
            self._last_tick = now
            initial_distance = math.hypot(goal[0] - pose[0], goal[1] - pose[1]) if pose else math.inf
            self._watchdog.reset(now, initial_distance)
        self._reached_pub.publish(Bool(data=False))
        self._compat_reached_pub.publish(Bool(data=False))
        if self._armed_pub is not None:
            self._armed_pub.publish(Bool(data=True))
        self._write_log("executing", success=False, final=False)
        rospy.loginfo(
            "Exploration goal received: frame=%s position=(%.3f, %.3f, %.3f)",
            message.header.frame_id or pose_frame or "odom", goal[0], goal[1], goal[2],
        )

    def _on_locomotion_ready(self, message):
        with self._lock:
            self._locomotion_ready = bool(message.data)
            if not self._locomotion_ready:
                self._publish_zero()

    def _on_external_pause(self, message):
        with self._lock:
            self._external_paused = bool(message.data)

    def _on_registration_status(self, message):
        # FAST-LIO publishes this only after its first registration cycle. A
        # zero-correspondence interval must stop the current command before
        # predicted odometry can be mistaken for a valid exploration path.
        try:
            payload = json.loads(message.data)
            healthy = bool(payload.get("healthy", True))
            invalid_count = int(payload.get("invalid_count", 0) or 0)
        except (TypeError, ValueError):
            return
        with self._lock:
            self._registration_healthy = healthy
            self._registration_invalid_count = (
                0 if healthy else max(
                    invalid_count, self._registration_invalid_count + 1))

    def _on_localization_degraded(self, message):
        with self._lock:
            self._localization_degraded = bool(message.data)

    def _on_speed_limit(self, message):
        value = float(message.data)
        if not math.isfinite(value) or value <= 0.0:
            return
        with self._lock:
            # Keep dynamic speed updates bounded by the calibrated flat-ground
            # cap while retaining the manager's phase-specific limit.
            self._maximum_speed = max(
                self._minimum_speed, min(self._dynamic_speed_cap, value))

    def _on_distance_gain(self, message):
        """Apply a bounded, phase-specific approach gain."""
        value = float(message.data)
        if not math.isfinite(value) or value <= 0.0:
            return
        with self._lock:
            # This changes only how quickly speed approaches the existing
            # calibrated cap; it cannot enlarge the cap itself.
            self._distance_gain = max(0.10, min(2.00, value))

    def _on_lateral_speed_limit(self, message):
        value = float(message.data)
        if not math.isfinite(value) or value <= 0.0:
            return
        with self._lock:
            # The semantic manager publishes this only after choosing a
            # phase-specific, A*/SCAN-lite validated goal.  Corridor motion
            # retains its conservative cap; short room maneuvers may use the
            # faster value without globally re-enabling high-speed side slip.
            self._lateral_speed_limit = min(self._dynamic_speed_cap, value)

    def _on_goal_tolerance(self, message):
        value = float(message.data)
        if not math.isfinite(value):
            return
        with self._lock:
            self._goal_tolerance = max(0.08, min(0.50, value))

    def _on_cancel(self, message):
        """Cancel only the exact goal named by the manager.

        rospy owns and rewrites Header.seq during serialization, so the
        manager-generated wall-clock stamp plus goal coordinates form the
        stable identity.  Header.seq is retained only for diagnostics.
        """
        try:
            payload = json.loads(message.data)
            stamp = float(payload["goal_stamp"])
            goal_x = float(payload["goal_x"])
            goal_y = float(payload["goal_y"])
            sequence = int(payload.get("goal_sequence", -1))
            reason = str(payload.get("reason", "manager_cancel"))
        except (KeyError, TypeError, ValueError, OverflowError):
            rospy.logwarn_throttle(5.0, "Ignored malformed goal cancellation")
            return
        with self._lock:
            identity_match = (
                self._goal is not None and
                abs(stamp - self._goal_stamp) <= 1e-6 and
                abs(goal_x - self._goal[0]) <= 0.05 and
                abs(goal_y - self._goal[1]) <= 0.05)
            if not identity_match:
                self._events.append({
                    "event": "stale_cancel_ignored",
                    "wall_time": time.time(),
                    "cancel_goal_sequence": sequence,
                    "cancel_goal_stamp": stamp,
                    "cancel_goal_x": goal_x,
                    "cancel_goal_y": goal_y,
                })
                return
            generation = self._goal_generation
            cancel_reason = "goal_cancelled_by_manager:" + reason
            if reason == "confirmed_local_door_preemption":
                # The manager does not publish the doorway goal until this
                # cancellation gets its result.  That result is deliberately
                # delayed until a controlled stop has completed.
                if self._pending_cancel is None:
                    self._pending_cancel = {
                        "generation": generation,
                        "reason": cancel_reason,
                        "started": time.monotonic(),
                        "below_since": None,
                    }
                    self._events.append({
                        "event": "goal_preemption_deceleration_started",
                        "wall_time": time.time(),
                        "goal_generation": generation,
                        "command_speed": math.hypot(
                            self._last_command[0], self._last_command[1]),
                        "measured_speed": self._measured_speed,
                    })
                return
        self._finish(False, cancel_reason, generation)

    def _on_rescan_request(self, message):
        """Execute a rotation-only sensing action while navigation is idle."""
        try:
            payload = json.loads(message.data)
            request_id = str(payload["request_id"])
            visual_sweep = bool(payload.get("visual_sweep", False))
            full_visual_sweep = bool(payload.get("full_visual_sweep", False))
            room_id = payload.get("room_id")
            requested_angle = abs(float(payload.get("angle_rad", 2.0 * math.pi)))
            requested_speed = abs(float(payload.get("angular_speed", 0.16)))
            # Mapping rescans retain their conservative full-turn envelope.
            # A room RGB-D sweep is explicitly bounded and may use a short
            # half-turn at the validated yaw rate, without taking ownership
            # of translation or altering the main planner route.
            angle = (max(0.15, min(2.0 * math.pi if full_visual_sweep else math.pi,
                                    requested_angle)) if visual_sweep else
                     max(math.pi, min(2.0 * math.pi, requested_angle)))
            # RGB-D frames are sampled continuously during a visual sweep;
            # one rad/s remains below the controller yaw envelope and cuts a
            # 360-degree inspection from 10.5 s to about 6.3 s.
            speed = (max(0.10, min(1.00, requested_speed)) if visual_sweep else
                     max(0.08, min(0.25, requested_speed)))
            timeout = max(2.0 if visual_sweep else 10.0,
                          float(payload.get("timeout_sec", angle / speed + 5.0)))
            direction = -1.0 if float(payload.get("direction", 1.0)) < 0.0 else 1.0
        except (KeyError, TypeError, ValueError, OverflowError):
            rospy.logwarn("Ignored malformed local rescan request")
            return
        now = time.monotonic()
        with self._lock:
            if not self._locomotion_ready or (self._goal is not None and not self._finished):
                accepted = False
            else:
                self._rescan = {
                    "request_id": request_id, "started": now,
                    "duration": angle / speed, "timeout": timeout,
                    "speed": direction * speed, "angle": angle,
                    "visual_sweep": visual_sweep,
                    "full_visual_sweep": full_visual_sweep,
                    "room_id": room_id,
                }
                accepted = True
        if not accepted:
            # R28: 拒绝曾完全静默,manager 侧无从得知 rescan 为何失败,
            # 只能空转死锁。必须显式记录。
            rospy.logwarn("Local rescan %s rejected: %s", request_id,
                          ("locomotion not ready"
                           if not self._locomotion_ready
                           else "executor busy"))
            self._rescan_result_pub.publish(String(data=json.dumps({
                "request_id": request_id, "success": False,
                "reason": "executor_busy_or_locomotion_not_ready",
                "visual_sweep": visual_sweep, "room_id": room_id,
            }, sort_keys=True)))
        else:
            rospy.loginfo("Local rescan %s accepted: %.0f deg at %.2f rad/s",
                          request_id, math.degrees(angle), speed)

    def _publish_zero(self, expected_generation=None):
        with self._lock:
            if (expected_generation is not None and
                    expected_generation != self._goal_generation):
                return False
            self._last_command = [0.0, 0.0, 0.0]
            zero = Twist()
            # roslaunch can close publishers before a rospy.Timer callback
            # observes shutdown.  Zeroing is best-effort at that point; do
            # not emit a misleading thread traceback during normal teardown.
            try:
                self._desired_command_pub.publish(zero)
                self._command_pub.publish(zero)
            except rospy.ROSException:
                return False
            return True

    def _publish_stall_backoff(self, expected_generation=None):
        """Back away slowly along the command that led into the obstruction."""
        with self._lock:
            if (expected_generation is not None and
                    expected_generation != self._goal_generation):
                return False
            command = Twist()
            command.linear.x = self._stall_recovery_command[0]
            command.linear.y = self._stall_recovery_command[1]
            self._last_command = [
                command.linear.x, command.linear.y, 0.0]
            self._desired_command_pub.publish(command)
            self._command_pub.publish(command)
            return True

    def _finish(self, success, reason, expected_generation=None):
        with self._lock:
            # A timer callback may have snapshotted the previous goal before a
            # new subscriber callback replaced it.  Such a stale callback is
            # never allowed to finish or stop the new goal.
            if (expected_generation is not None and
                    expected_generation != self._goal_generation):
                return
            if self._finished:
                return
            self._finished = True
            self._pending_cancel = None
            self._events.append({"event": reason, "wall_time": time.time()})
            goal = self._goal
            end = self._pose
            started = self._goal_start
            goal_sequence = self._goal_sequence
            goal_stamp = self._goal_stamp
            goal_frame = self._goal_frame
            distance_error = None
            if goal is not None and end is not None:
                distance_error = math.hypot(goal[0] - end[0], goal[1] - end[1])
            # Keep the lock until every actuator/latch side effect for this
            # generation is complete.  _on_goal can only accept the next goal
            # afterwards, so an old finish cannot stop or overwrite it.
            self._publish_zero(expected_generation)
            if self._armed_pub is not None:
                self._armed_pub.publish(Bool(data=False))
            self._reached_pub.publish(Bool(data=success))
            self._compat_reached_pub.publish(Bool(data=success))
            self._result_pub.publish(String(data=json.dumps({
                "success": bool(success), "reason": reason,
                "distance_error_m": distance_error,
                "duration_sec": None if started is None else time.monotonic() - started,
                "goal": self._pose_dict(goal), "end_pose": self._pose_dict(end),
                "goal_sequence": goal_sequence, "goal_stamp": goal_stamp,
                "goal_frame": goal_frame,
                "zero_command_stall_recovery_count": self._stall_recoveries,
            }, sort_keys=True)))
            # Snapshot while this generation is still protected, then perform
            # the potentially slow JSON write after releasing the motion lock.
            log_payload = self._build_log_payload_locked(
                reason, success=success, final=True)
        self._write_log_payload(log_payload)
        if success:
            rospy.loginfo("Exploration goal reached within %.2f m", self._goal_tolerance)
        else:
            rospy.logerr("Exploration goal stopped safely: %s", reason)
        if self._shutdown_on_result:
            rospy.signal_shutdown(reason)

    def _on_timer(self, _event):
        now = time.monotonic()
        with self._lock:
            goal, pose = self._goal, self._pose
            pose_age = now - self._pose_wall_time if self._pose_wall_time else math.inf
            goal_start = self._goal_start
            pose_fault = self._pose_fault
            registration_healthy = self._registration_healthy
            registration_invalid_count = self._registration_invalid_count
            localization_degraded = self._localization_degraded
            pose_frame = self._pose_frame
            goal_frame = self._goal_frame
            finished = self._finished
            generation = self._goal_generation
            recovery_until = self._stall_recovery_until
            rescan = dict(self._rescan) if self._rescan is not None else None
            pending_cancel = (dict(self._pending_cancel)
                              if self._pending_cancel is not None else None)
            locomotion_ready = self._locomotion_ready
            measured_speed = self._measured_speed
            external_paused = self._external_paused
        if external_paused:
            # An external driver owns /cmd_vel for its whole window: suppress
            # every executor output (including the idle zero) so the 20 Hz
            # arbitration messages cannot fight the guide driver.
            return
        if rescan is not None:
            elapsed = now - rescan["started"]
            if elapsed >= rescan["duration"] or elapsed >= rescan["timeout"]:
                success = elapsed >= rescan["duration"]
                with self._lock:
                    if (self._rescan is not None and
                            self._rescan["request_id"] == rescan["request_id"]):
                        self._rescan = None
                self._publish_zero()
                self._rescan_result_pub.publish(String(data=json.dumps({
                    "request_id": rescan["request_id"], "success": success,
                    "reason": "local_rescan_complete" if success else "local_rescan_timeout",
                    "duration_sec": elapsed, "angle_rad": rescan["angle"],
                    "visual_sweep": bool(rescan.get("visual_sweep", False)),
                    "room_id": rescan.get("room_id"),
                }, sort_keys=True)))
                return
            command = Twist()
            command.angular.z = rescan["speed"]
            with self._lock:
                self._last_command = [0.0, 0.0, command.angular.z]
                self._desired_command_pub.publish(command)
                self._command_pub.publish(command)
            self._status_pub.publish(String(data=json.dumps({
                "state": "LOCAL_RESCAN", "request_id": rescan["request_id"],
                "elapsed_sec": elapsed, "target_angle_rad": rescan["angle"],
            }, sort_keys=True)))
            return
        if (pending_cancel is not None and
                pending_cancel["generation"] == generation and not finished):
            if not locomotion_ready:
                self._finish(False,
                             "locomotion_not_ready_during_preemption",
                             generation)
                return
            dt = max(0.0, now - self._last_tick)
            self._last_tick = now
            command = Twist()
            command.linear.x = slew(
                self._last_command[0], 0.0, dt,
                self._preemption_deceleration,
                self._preemption_deceleration)
            command.linear.y = slew(
                self._last_command[1], 0.0, dt,
                self._preemption_deceleration,
                self._preemption_deceleration)
            command.angular.z = slew(
                self._last_command[2], 0.0, dt,
                self._preemption_deceleration,
                self._preemption_deceleration)
            command_speed = math.hypot(command.linear.x, command.linear.y)
            command_settled = (
                command_speed <= self._preemption_command_speed and
                abs(command.angular.z) <= self._preemption_command_speed)
            measured_settled = (
                measured_speed <= self._preemption_measured_speed)
            finish_reason = None
            with self._lock:
                active = (generation == self._goal_generation and
                          not self._finished and
                          self._pending_cancel is not None)
                if not active:
                    return
                self._last_command = [command.linear.x, command.linear.y,
                                      command.angular.z]
                self._desired_command_pub.publish(command)
                self._command_pub.publish(command)
                active_cancel = self._pending_cancel
                if command_settled and measured_settled:
                    if active_cancel["below_since"] is None:
                        active_cancel["below_since"] = now
                    elif (now - active_cancel["below_since"] >=
                          self._preemption_settle_seconds):
                        finish_reason = active_cancel["reason"]
                else:
                    active_cancel["below_since"] = None
                if (finish_reason is None and
                        now - active_cancel["started"] >=
                        self._preemption_timeout):
                    # End at zero even if noisy odometry never enters the
                    # settle band.  The command has still been ramped down for
                    # the complete stability interval.
                    finish_reason = active_cancel["reason"]
                    self._events.append({
                        "event": "goal_preemption_settle_timeout",
                        "wall_time": time.time(),
                        "command_speed": command_speed,
                        "measured_speed": measured_speed,
                    })
            self._status_pub.publish(String(data=json.dumps({
                "state": "PREEMPTION_DECELERATION",
                "command_speed": command_speed,
                "measured_speed": measured_speed,
                "elapsed_sec": now - pending_cancel["started"],
            }, sort_keys=True)))
            if finish_reason is not None:
                self._finish(False, finish_reason, generation)
            return
        if goal is None or finished:
            self._publish_zero(generation)
            return
        if pose_fault:
            self._finish(False, pose_fault, generation)
            return
        if (not registration_healthy and
                registration_invalid_count >=
                self._registration_abort_invalid_count):
            self._finish(False, "fastlio_registration_lost", generation)
            return
        if not registration_healthy:
            # Match the exploration manager's debounce contract. A single
            # delayed high-speed scan stops motion but preserves the active
            # goal so it can resume after registration recovers.
            self._publish_zero(generation)
            return
        if pose is None:
            if goal_start is not None and now - goal_start > self._odom_timeout:
                self._finish(False, "localization_missing", generation)
            else:
                self._publish_zero(generation)
            return
        if pose_age > self._odom_timeout:
            self._finish(False, "localization_stale", generation)
            return
        if goal_frame and pose_frame and goal_frame != pose_frame:
            self._finish(False, "goal_odometry_frame_mismatch", generation)
            return
        if goal_start is not None and now - goal_start > self._goal_timeout:
            self._finish(False, "goal_timeout", generation)
            return
        if recovery_until is not None:
            if now < recovery_until:
                self._publish_stall_backoff(generation)
                return
            with self._lock:
                if generation == self._goal_generation:
                    self._stall_recovery_until = None
                    self._stall_recovery_command = [0.0, 0.0]
                    self._watchdog.reset(now, math.hypot(
                        goal[0] - pose[0], goal[1] - pose[1]))
        effective_maximum_speed = self._maximum_speed
        if localization_degraded:
            effective_maximum_speed = max(
                self._minimum_speed,
                min(effective_maximum_speed, self._localization_speed_cap))
        try:
            vx, vy, wz, distance, heading_error = command_for_goal(
                pose[0], pose[1], pose[3], goal[0], goal[1],
                goal_tolerance=self._goal_tolerance,
                minimum_speed=self._minimum_speed,
                maximum_speed=effective_maximum_speed,
                distance_gain=self._distance_gain,
                heading_gain=self._heading_gain,
                maximum_yaw_rate=self._maximum_yaw_rate,
                lateral_speed_limit=self._lateral_speed_limit,
                heading_slowdown_threshold=self._heading_slowdown_threshold,
                turning_speed_limit=self._turning_speed_limit,
            )
        except ValueError as error:
            self._finish(False, str(error), generation)
            return
        if distance < self._goal_tolerance:
            self._finish(True, "goal_reached", generation)
            return
        if self._watchdog.update(now, distance):
            with self._lock:
                if self._stall_recoveries < self._maximum_stall_recoveries:
                    self._stall_recoveries += 1
                    self._stall_recovery_until = now + self._stall_recovery_seconds
                    magnitude = math.hypot(vx, vy)
                    if magnitude > 1e-6:
                        self._stall_recovery_command = [
                            -self._stall_recovery_speed * vx / magnitude,
                            -self._stall_recovery_speed * vy / magnitude]
                    else:
                        self._stall_recovery_command = [
                            -self._stall_recovery_speed, 0.0]
                    self._events.append({
                        "event": "active_backoff_stall_recovery",
                        "recovery": self._stall_recoveries,
                        "command_linear_x":
                            self._stall_recovery_command[0],
                        "command_linear_y":
                            self._stall_recovery_command[1],
                        "wall_time": time.time(),
                    })
                    self._publish_stall_backoff(generation)
                    rospy.logwarn(
                        "Goal stalled; active backoff %d/%d for %.1fs",
                        self._stall_recoveries, self._maximum_stall_recoveries,
                        self._stall_recovery_seconds)
                    return
            self._finish(False, "goal_unreachable_no_progress", generation)
            return

        dt = max(0.0, now - self._last_tick)
        self._last_tick = now
        command = Twist()
        command.linear.x = slew(self._last_command[0], vx, dt,
                                self._acceleration, self._deceleration)
        command.linear.y = slew(self._last_command[1], vy, dt,
                                self._acceleration, self._deceleration)
        command.angular.z = slew(self._last_command[2], wz, dt,
                                 self._acceleration, self._deceleration)
        sample = {
            "t": round(now - goal_start, 3), "x": pose[0], "y": pose[1],
            "yaw": pose[3], "distance_to_goal": distance,
            "heading_error": heading_error, "linear_x": command.linear.x,
            "linear_y": command.linear.y, "angular_z": command.angular.z,
        }
        with self._lock:
            # Goal callbacks and timer callbacks run concurrently in rospy.
            # Discard a command computed from a goal that has since been
            # replaced, including its telemetry sample.
            if generation != self._goal_generation or self._finished:
                return
            self._last_command = [command.linear.x, command.linear.y,
                                  command.angular.z]
            self._desired_command_pub.publish(command)
            self._command_pub.publish(command)
            self._samples.append(sample)
            if len(self._samples) > 10000:
                self._samples = self._samples[-10000:]
        self._status_pub.publish(String(data=json.dumps(sample, sort_keys=True)))

    def _build_log_payload_locked(self, state, success, final):
        """Snapshot executor state; caller must hold the motion-state lock."""
        now = time.monotonic()
        self._log_revision += 1
        log_revision = self._log_revision
        goal, start, end = self._goal, self._start_pose, self._pose
        started = self._goal_start
        goal_sequence = self._goal_sequence
        goal_stamp = self._goal_stamp
        samples, events = list(self._samples), list(self._events)
        distance_error = None
        if goal is not None and end is not None:
            distance_error = math.hypot(goal[0] - end[0], goal[1] - end[1])
        return {
            "schema": "simenv_goal_execution_log_v1", "state": state,
            "log_revision": log_revision,
            "success": bool(success), "final": bool(final),
            "goal_topic": self._goal_topic, "odometry_topic": self._odom_topic,
            "command_topic": self._command_topic, "cancel_topic": self._cancel_topic,
            "goal_frame": self._goal_frame,
            "goal_sequence": goal_sequence, "goal_stamp": goal_stamp,
            "goal": self._pose_dict(goal), "start_pose": self._pose_dict(start),
            "end_pose": self._pose_dict(end), "distance_error_m": distance_error,
            "duration_sec": None if started is None else max(0.0, now - started),
            "parameters": {
                "goal_tolerance_m": self._goal_tolerance,
                "minimum_speed_mps": self._minimum_speed,
                "maximum_speed_mps": self._maximum_speed,
                "preemption_deceleration_mps2":
                    self._preemption_deceleration,
                "preemption_timeout_sec": self._preemption_timeout,
                "odometry_timeout_sec": self._odom_timeout,
                "progress_timeout_sec": self._watchdog.timeout,
                "maximum_stall_recoveries": self._maximum_stall_recoveries,
                "stall_recovery_seconds": self._stall_recovery_seconds,
            },
            "events": events, "samples": samples,
        }

    def _write_log_payload(self, payload):
        with self._log_write_lock:
            revision = int(payload.get("log_revision", -1))
            # A completed old-goal snapshot may finish JSON serialization
            # after the next goal has already produced a newer snapshot.
            # Never let that stale state overwrite the active-goal log.
            if revision <= self._last_written_log_revision:
                return
            self._atomic_json(self._log_path, payload)
            self._last_written_log_revision = revision

    def _write_log(self, state, success, final):
        with self._lock:
            payload = self._build_log_payload_locked(
                state, success=success, final=final)
        self._write_log_payload(payload)

    def _on_shutdown(self):
        self._publish_zero()
        if self._armed_pub is not None:
            try:
                self._armed_pub.publish(Bool(data=False))
            except rospy.ROSException:
                pass
        with self._lock:
            finished = self._finished
        if not finished:
            self._write_log("ros_shutdown", success=False, final=True)


if __name__ == "__main__":
    rospy.init_node("goal_executor")
    GoalExecutor()
    rospy.spin()
