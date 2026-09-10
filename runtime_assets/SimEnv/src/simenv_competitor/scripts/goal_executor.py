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
    ProgressWatchdog, bounded_dynamic_minimum_speed, command_for_goal,
    enforce_executable_translation_floor, navigation_stability_envelope,
    pose_from_truth_delta, quaternion_yaw,
    rotation_only_exit_alignment_yaw_rate, slew,
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
        self._nominal_minimum_speed = self._minimum_speed
        # This is a physical gait floor, not the corridor cruise floor.  A
        # room manager may lower the latter for the final aligned portal leg.
        self._dynamic_minimum_speed_floor = max(
            0.01, min(self._nominal_minimum_speed, float(rospy.get_param(
                "~dynamic_minimum_speed_floor", 0.15))))
        self._maximum_speed = float(rospy.get_param("~maximum_speed", 0.35))
        self._nominal_maximum_speed = self._maximum_speed
        # The semantic manager publishes a per-goal limit.  Its cap is a
        # launch parameter so a timed mission can raise it deliberately;
        # never silently reinstate a lower hard-coded cap here.
        self._dynamic_speed_cap = max(
            self._minimum_speed,
            float(rospy.get_param("~dynamic_speed_cap", self._maximum_speed)))
        # Upper-floor truth handoff keeps navigation usable while FAST-LIO
        # registration is sparse, but the nominal 2 m/s flat-floor cap is too
        # aggressive during room turns and can trigger a fall/reset.  Keep a
        # separate, bounded cap for F2/F3; the manager can still publish a
        # lower per-goal limit through the normal dynamic-speed topic.
        self._upper_floor_maximum_speed = max(
            self._minimum_speed,
            float(rospy.get_param(
                "~upper_floor_maximum_speed", self._maximum_speed)))
        self._distance_gain = float(rospy.get_param("~distance_gain", 0.40))
        self._heading_gain = float(rospy.get_param("~heading_gain", 0.35))
        self._maximum_yaw_rate = float(rospy.get_param("~maximum_yaw_rate", 0.20))
        self._absolute_maximum_yaw_rate = max(
            self._maximum_yaw_rate,
            min(1.20, float(rospy.get_param(
                "~absolute_maximum_yaw_rate", 0.75))))
        self._lateral_speed_limit = float(rospy.get_param(
            "~lateral_speed_limit", self._maximum_speed))
        self._heading_slowdown_threshold = float(rospy.get_param(
            "~heading_slowdown_threshold", 0.45))
        # Disabled for ordinary goals.  The exploration manager enables this
        # only for a physical room EXIT, where translating before the body is
        # aligned with the outward door normal can wedge the quadruped on a
        # narrow jamb.
        self._translation_heading_gate = math.inf
        self._translation_heading_aligned = False
        # The plane policy has a measurable positive-yaw deadband below
        # roughly 0.35 rad/s.  EXIT's rotate-before-translate gate used to
        # command heading_gain * error (often 0.16--0.25 rad/s), so the body
        # stopped turning just outside the gate and consumed the EXIT
        # watchdog.  This floor-independent lower bound is active only while
        # the manager has enabled the finite EXIT heading gate.
        self._translation_heading_minimum_yaw_rate = max(
            0.20, min(self._absolute_maximum_yaw_rate,
                      float(rospy.get_param(
                          "~translation_heading_minimum_yaw_rate", 0.42))))
        # EXIT already holds vx=vy=0 until the outward door-normal heading is
        # latched. Use the old proven 0.75 rad/s envelope only for a large
        # error in that rotation-only phase; the ordinary manager cap becomes
        # authoritative again before portal translation begins.
        self._translation_heading_rotation_yaw_rate = max(
            self._translation_heading_minimum_yaw_rate,
            min(self._absolute_maximum_yaw_rate, float(rospy.get_param(
                "~translation_heading_rotation_yaw_rate", 0.75))))
        self._translation_heading_fast_rotation_error = max(
            0.35, float(rospy.get_param(
                "~translation_heading_fast_rotation_error_rad", 0.70)))
        # F3 truth-degraded corridor navigation also uses a rotate-before-
        # translate gate, but those goals normally have no locked target
        # heading and therefore never enter the EXIT-only gate below.  run145
        # consequently spent more than 130 simulation seconds asking the
        # plane policy for 0.14--0.24 rad/s while its measured yaw stayed near
        # zero.  Keep this separate parameter so the physical deadband floor
        # applies to the F3 fallback without changing ordinary holonomic
        # navigation on F1/F2.
        self._truth_rotate_minimum_yaw_rate = max(
            0.20, min(self._absolute_maximum_yaw_rate,
                      float(rospy.get_param(
                          "~truth_rotate_minimum_yaw_rate", 0.42))))
        self._goal_truth_relative_navigation = False
        self._turning_speed_limit = float(rospy.get_param(
            "~turning_speed_limit", min(0.65, self._maximum_speed)))
        # Normal goal navigation may request a fast yaw while the previous
        # command still contains a fast lateral gait. In run17 this exact
        # combination (vy=0.55 m/s, wz=1.10 rad/s) produced a 2.51 rad/s
        # physical yaw spike and a 1.38 rad roll despite healthy localization.
        # Brake translation first, then retain the original fast in-place
        # turn. Visual and mapping rescans use their separate rotation-only
        # branch and never pass through this envelope.
        self._navigation_large_heading_error = max(
            0.10, float(rospy.get_param(
                "~navigation_large_heading_error_rad", 0.55)))
        self._navigation_high_yaw_rate = max(
            0.10, float(rospy.get_param(
                "~navigation_high_yaw_rate_radps", 0.80)))
        self._navigation_fast_turn_translation_threshold = max(
            0.02, float(rospy.get_param(
                "~navigation_fast_turn_translation_threshold_mps", 0.10)))
        self._navigation_braking_yaw_rate = max(
            0.10, float(rospy.get_param(
                "~navigation_braking_yaw_rate_radps", 0.55)))
        self._navigation_medium_heading_error = max(
            0.05, min(self._navigation_large_heading_error,
                      float(rospy.get_param(
                          "~navigation_medium_heading_error_rad", 0.25))))
        self._navigation_turning_translation_limit = max(
            0.10, float(rospy.get_param(
                "~navigation_turning_translation_limit_mps", 0.75)))
        self._navigation_turning_lateral_limit = max(
            0.05, float(rospy.get_param(
                "~navigation_turning_lateral_limit_mps", 0.35)))
        # MotionConsistencyGuard compares commanded travel against FAST-LIO
        # odometry and also receives IMU samples.  It is not a pose resetter;
        # use its short degradation signal to reduce excitation while the
        # LiDAR/IMU estimator regains correspondences.
        self._localization_speed_cap = float(rospy.get_param(
            "~localization_degraded_speed_cap", 0.85))
        self._goal_degraded_speed_cap = self._localization_speed_cap
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
        self._direct_floor_height_mode_pending = False
        self._direct_floor_launch_pose_z_bounds = None
        self._shutdown_on_result = bool(rospy.get_param("~shutdown_on_result", False))
        self._registration_abort_invalid_count = max(
            1, int(rospy.get_param(
                "~registration_abort_invalid_count", 30)))
        # On upper floors FAST-LIO can remain scan-frozen in sparse landing
        # geometry while its simulator-relative motion guard still publishes
        # a fresh, bounded stabilized pose.  Only that explicit phase-gated
        # guard may replace the ordinary registration health requirement.
        self._upper_floor_truth_guard_timeout = max(
            0.10, float(rospy.get_param(
                "~upper_floor_truth_guard_timeout_sec", 2.50)))
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
        self._maximum_visual_sweep_speed = max(
            0.10, float(rospy.get_param(
                "~maximum_visual_sweep_angular_speed", 1.60)))
        # Treat the last few degrees as the normal closed-loop arrival band,
        # not as a failed camera action.  The A1 yaw gait decelerates near the
        # watchdog boundary; run104 physically turned 3.620/3.665 rad but the
        # strict comparison reported a timeout.  That false failure prevented
        # the manager from crediting G3's fan, so G4 reused G3's stale edge
        # hint and scanned the same occluded side again.  This tolerance is
        # deliberately small (about 4.6 deg) and never applies to mapping
        # rescans or translational room credit.
        self._visual_sweep_completion_tolerance = min(
            0.15, max(0.0, float(rospy.get_param(
                "~visual_sweep_completion_tolerance_rad", 0.08))))
        self._maximum_mapping_rescan_speed = max(
            0.10, float(rospy.get_param(
                "~maximum_mapping_rescan_angular_speed", 1.20)))
        self._visual_candidate_slow_speed = max(
            0.10, float(rospy.get_param(
                "~visual_candidate_slow_angular_speed", 0.55)))
        self._visual_candidate_slow_seconds = max(
            0.0, float(rospy.get_param(
                "~visual_candidate_slow_seconds", 0.90)))

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
        self._target_heading = None
        self._pose_fault = ""
        self._registration_healthy = True
        self._registration_invalid_count = 0
        self._upper_floor_context_active = False
        self._upper_floor_index = -1
        self._upper_floor_truth_guard = None
        self._upper_floor_truth_guard_received_at = -math.inf
        self._truth_pose = None
        self._truth_pose_received_at = -math.inf
        self._truth_odom_anchor = None
        self._truth_world_anchor = None
        # True while F3 motion is being carried by the truth-relative guard
        # because FAST-LIO is still publishing a previous-floor height.  A
        # registration-health bit alone is insufficient in this state: its
        # planar/yaw stream can look healthy while the body is completing a
        # large post-rescan turn at a narrow doorway.
        self._truth_vertical_rebase_active = False
        self._localization_degraded = False
        self._samples = []
        self._events = []
        self._finished = False
        self._last_command = [0.0, 0.0, 0.0]
        self._navigation_stability_mode = "unrestricted"
        self._last_tick = time.monotonic()
        self._locomotion_ready = False
        self._direct_f3_rl_ready = False
        self._stall_recoveries = 0
        self._stall_recovery_until = None
        self._stall_recovery_command = [0.0, 0.0]
        self._rescan = None
        self._pending_cancel = None
        ownership_topics = rospy.get_param(
            "~stair_command_ownership_state_topics", [
                "/simenv/stair_transition_state",
                "/simenv/second_to_third_floor_stair_state",
                "/simenv/third_to_first_floor_stair_state"])
        if isinstance(ownership_topics, str):
            ownership_topics = [item.strip() for item in
                                ownership_topics.split(",") if item.strip()]
        self._stair_command_ownership_topics = list(ownership_topics)
        self._stair_command_states = {}
        self._stair_command_owned = False

        self._command_pub = rospy.Publisher(self._command_topic, Twist, queue_size=5)
        self._desired_command_pub = rospy.Publisher(
            "/simenv/desired_cmd_vel", Twist, queue_size=5)
        self._status_pub = rospy.Publisher(
            "/simenv/goal_execution_status", String, queue_size=2, latch=True
        )
        self._floor_context_status_pub = rospy.Publisher(
            "/simenv/goal_executor_floor_context_status", String,
            queue_size=2, latch=True)
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
        rospy.Subscriber("/simenv/red_ball_observations", String,
                         self._on_red_ball_observation, queue_size=20)
        rospy.Subscriber("/simenv/goal_speed_limit", Float32,
                         self._on_speed_limit, queue_size=2)
        rospy.Subscriber("/simenv/goal_minimum_speed", Float32,
                         self._on_minimum_speed, queue_size=2)
        rospy.Subscriber("/simenv/goal_distance_gain", Float32,
                         self._on_distance_gain, queue_size=2)
        rospy.Subscriber("/simenv/goal_heading_gain", Float32,
                         self._on_heading_gain, queue_size=2)
        rospy.Subscriber("/simenv/goal_target_heading", Float32,
                         self._on_target_heading, queue_size=2)
        rospy.Subscriber("/simenv/goal_translation_heading_gate", Float32,
                         self._on_translation_heading_gate, queue_size=2)
        rospy.Subscriber("/simenv/goal_truth_relative_navigation", Bool,
                         self._on_goal_truth_relative_navigation,
                         queue_size=2)
        rospy.Subscriber("/simenv/goal_maximum_yaw_rate", Float32,
                         self._on_maximum_yaw_rate, queue_size=2)
        rospy.Subscriber("/simenv/goal_lateral_speed_limit", Float32,
                         self._on_lateral_speed_limit, queue_size=2)
        rospy.Subscriber("/simenv/goal_tolerance", Float32,
                         self._on_goal_tolerance, queue_size=2)
        rospy.Subscriber("/locomotion_ready", Bool,
                         self._on_locomotion_ready, queue_size=2)
        rospy.Subscriber("/simenv/direct_f3_rl_ready", Bool,
                         self._on_direct_f3_rl_ready, queue_size=2)
        rospy.Subscriber("/simenv/fastlio_registration_status", String,
                         self._on_registration_status, queue_size=5)
        rospy.Subscriber("/simenv/fastlio_f2_truth_guard_status", String,
                         self._on_upper_floor_truth_guard, queue_size=5)
        rospy.Subscriber("/simenv/second_floor_truth_odometry", Odometry,
                         self._on_truth_odometry, queue_size=10)
        rospy.Subscriber("/simenv/localization_degraded", Bool,
                         self._on_localization_degraded, queue_size=2)
        rospy.Subscriber("/simenv/goal_degraded_speed_cap", Float32,
                         self._on_goal_degraded_speed_cap, queue_size=2)
        # F1 retains the launch-time z envelope.  The isolated F2 manager
        # explicitly translates it only after corridor-entry handoff.
        rospy.Subscriber("/simenv/goal_executor_floor_context", String,
                         self._on_floor_context, queue_size=2)
        rospy.Subscriber("/simenv/baseline_state", String,
                         self._on_floor_local_baseline_state, queue_size=2)
        # THIRD_FLOOR_REACHED must keep the stair controller command ownership
        # through landing recovery. Release it only at the explicit F3 handoff.
        rospy.Subscriber("/simenv/third_floor_state", String,
                         self._on_third_floor_state, queue_size=2)
        for topic in self._stair_command_ownership_topics:
            rospy.Subscriber(
                topic, String, self._on_stair_command_state,
                callback_args=topic, queue_size=5)
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
            self._direct_floor_height_mode_pending = False
            self._direct_floor_launch_pose_z_bounds = None
            self._upper_floor_context_active = True
            self._upper_floor_index = int(payload.get("floor_index", -1))
            # Force acquisition of one fresh F2 odometry sample.  This avoids
            # exposing the last accepted F1 pose while bounds are switching;
            # the next sample must still pass every ordinary safety check.
            self._pose = None
            self._previous_pose = None
            self._pose_wall_time = 0.0
            self._pose_fault = ""
            self._truth_odom_anchor = None
            self._truth_world_anchor = None
            self._truth_vertical_rebase_active = False
            self._events.append({
                "event": "second_floor_pose_bounds_activated",
                "wall_time": time.time(),
                "minimum_pose_z": bounds[0],
                "maximum_pose_z": bounds[1],
            })
            guard = dict(self._upper_floor_truth_guard or {})
            guard_age = (time.monotonic() -
                         self._upper_floor_truth_guard_received_at)
            guard_authorized = self._truth_guard_authorizes_upper_floor_motion(
                True, guard, guard_age,
                self._upper_floor_truth_guard_timeout)
        self._floor_context_status_pub.publish(String(data=json.dumps({
            "active": True,
            "floor_index": int(payload.get("floor_index", -1)),
            "mode": str(payload.get("mode", "")),
            "request_id": str(payload.get("request_id", "")),
            "minimum_pose_z": bounds[0],
            "maximum_pose_z": bounds[1],
            "truth_guard_authorized": bool(guard_authorized),
            "truth_guard_transport_age_sec": (
                round(guard_age, 3) if math.isfinite(guard_age) else None),
        }, sort_keys=True)))
        rospy.loginfo(
            "Second-floor executor pose bounds activated: %.2f..%.2f m",
            bounds[0], bounds[1])

    def _on_floor_local_baseline_state(self, message):
        """Authorize truth-relative navigation for direct F2/F3 launches.

        A full stair mission supplies an explicit floor-context handshake.
        A floor-local throughput launch has no stair manager, but its launch-
        time z envelope is already restricted to the requested upper floor.
        Require both pieces of evidence so an F1 or active stair transition
        can never be switched into upper-floor navigation by this fallback.
        """
        try:
            payload = json.loads(message.data)
            floor_number = int(payload.get("floor_number", -1))
        except (AttributeError, TypeError, ValueError):
            return
        if floor_number not in (2, 3):
            return
        with self._lock:
            if self._minimum_pose_z <= 1.0:
                return
            launch_minimum_pose_z = self._minimum_pose_z
            launch_maximum_pose_z = self._maximum_pose_z
            observed_pose_z = None
            if self._pose is not None and len(self._pose) >= 3:
                try:
                    candidate_pose_z = float(self._pose[2])
                    if math.isfinite(candidate_pose_z):
                        observed_pose_z = candidate_pose_z
                except (TypeError, ValueError):
                    pass
            floor_index = floor_number - 1
            if (self._upper_floor_context_active and
                    self._upper_floor_index == floor_index):
                return
            self._upper_floor_context_active = True
            self._upper_floor_index = floor_index
            # Historical direct upper-floor launches published floor-local z
            # near zero, while the current direct FAST-LIO launch publishes
            # absolute building z. Infer the mode only from a pose already
            # accepted by the deliberately upper-floor launch envelope. If no
            # such pose exists, preserve the legacy floor-local fallback.
            absolute_height_mode = (
                observed_pose_z is not None and
                launch_minimum_pose_z - 0.20 <= observed_pose_z <=
                launch_maximum_pose_z + 0.20)
            if absolute_height_mode:
                self._minimum_pose_z = launch_minimum_pose_z
                self._maximum_pose_z = launch_maximum_pose_z
                self._direct_floor_height_mode_pending = False
                self._direct_floor_launch_pose_z_bounds = None
                odom_height_mode = "absolute"
            elif observed_pose_z is not None:
                self._minimum_pose_z = -0.80
                self._maximum_pose_z = 1.20
                self._direct_floor_height_mode_pending = False
                self._direct_floor_launch_pose_z_bounds = None
                odom_height_mode = "floor_local"
            else:
                # The latched baseline-state message normally arrives before
                # FAST-LIO publishes its first odometry sample. Temporarily
                # accept the union of both proven launch envelopes, then the
                # first finite sample below immediately selects one strict
                # mode. This is direct-launch-only and is never used during a
                # stair transition.
                self._minimum_pose_z = min(-0.80, launch_minimum_pose_z)
                self._maximum_pose_z = max(1.20, launch_maximum_pose_z)
                self._direct_floor_height_mode_pending = True
                self._direct_floor_launch_pose_z_bounds = (
                    launch_minimum_pose_z, launch_maximum_pose_z)
                odom_height_mode = "pending_first_odometry"
            self._pose = None
            self._previous_pose = None
            self._pose_wall_time = 0.0
            self._pose_fault = ""
            self._truth_odom_anchor = None
            self._truth_world_anchor = None
            self._truth_vertical_rebase_active = False
            self._events.append({
                "event": "floor_local_upper_context_activated",
                "wall_time": time.time(),
                "floor_number": floor_number,
                "launch_minimum_pose_z": launch_minimum_pose_z,
                "launch_maximum_pose_z": launch_maximum_pose_z,
                "observed_pose_z": observed_pose_z,
                "odom_height_mode": odom_height_mode,
                "local_minimum_pose_z": self._minimum_pose_z,
                "local_maximum_pose_z": self._maximum_pose_z,
            })
        rospy.loginfo(
            "Direct F%d executor context activated in %s odometry-height mode",
            floor_number, odom_height_mode)

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
        finite_pose = all(math.isfinite(value) for value in pose)
        if finite_pose:
            with self._lock:
                bounds = self._direct_floor_launch_pose_z_bounds
                if self._direct_floor_height_mode_pending and bounds is not None:
                    launch_minimum_pose_z, launch_maximum_pose_z = bounds
                    if (launch_minimum_pose_z - 0.20 <= pose[2] <=
                            launch_maximum_pose_z + 0.20):
                        self._minimum_pose_z = launch_minimum_pose_z
                        self._maximum_pose_z = launch_maximum_pose_z
                        odom_height_mode = "absolute"
                    elif -0.80 <= pose[2] <= 1.20:
                        self._minimum_pose_z = -0.80
                        self._maximum_pose_z = 1.20
                        odom_height_mode = "floor_local"
                    else:
                        odom_height_mode = None
                    if odom_height_mode is not None:
                        self._direct_floor_height_mode_pending = False
                        self._direct_floor_launch_pose_z_bounds = None
                        self._events.append({
                            "event": "direct_floor_height_mode_resolved",
                            "wall_time": time.time(),
                            "observed_pose_z": pose[2],
                            "odom_height_mode": odom_height_mode,
                            "minimum_pose_z": self._minimum_pose_z,
                            "maximum_pose_z": self._maximum_pose_z,
                        })
                minimum_pose_z = self._minimum_pose_z
                maximum_pose_z = self._maximum_pose_z
        else:
            minimum_pose_z = self._minimum_pose_z
            maximum_pose_z = self._maximum_pose_z
        if not finite_pose:
            fault = "localization_nonfinite"
        elif pose[2] < minimum_pose_z or pose[2] > maximum_pose_z:
            fault = "localization_vertical_fault"
        with self._lock:
            # During the F2->F3 truth handoff FAST-LIO can still publish the
            # previous floor's absolute z while the truth-relative motion
            # guard is already valid at the new floor.  The timer below
            # replaces the planar pose with the truth-relative displacement,
            # but rejecting this callback here used to terminate every F3
            # corridor goal before that replacement could happen.  Permit
            # only this explicitly authorized, fresh F3 handoff sample; all
            # ordinary upper-floor and all F1/F2 samples retain the strict
            # height envelope.
            truth_guard_authorized = (
                self._upper_floor_context_active and
                self._upper_floor_index == 2 and
                self._truth_guard_authorizes_upper_floor_motion(
                    True, self._upper_floor_truth_guard or {},
                    now - self._upper_floor_truth_guard_received_at,
                    self._upper_floor_truth_guard_timeout) and
                self._truth_pose is not None and
                now - self._truth_pose_received_at <=
                self._upper_floor_truth_guard_timeout)
            if fault == "localization_vertical_fault" and truth_guard_authorized:
                if self._pose_fault != "localization_vertical_fault":
                    self._events.append({
                        "event": "f3_truth_vertical_odom_rebased",
                        "wall_time": time.time(),
                        "observed_pose_z": pose[2],
                        "configured_minimum_pose_z": minimum_pose_z,
                        "configured_maximum_pose_z": maximum_pose_z,
                    })
                    rospy.logwarn(
                        "F3 truth guard rebasing stale odometry height %.3f "
                        "outside %.3f..%.3f",
                        pose[2], minimum_pose_z, maximum_pose_z)
                fault = ""
                self._truth_vertical_rebase_active = True
            elif not fault:
                self._truth_vertical_rebase_active = False
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
                if (self._upper_floor_context_active and
                        self._truth_odom_anchor is None and
                        self._truth_pose is not None and
                        now - self._truth_pose_received_at <=
                        self._upper_floor_truth_guard_timeout):
                    self._truth_odom_anchor = pose
                    self._truth_world_anchor = self._truth_pose
            else:
                self._pose_fault = fault

    def _on_truth_odometry(self, message):
        p, q = message.pose.pose.position, message.pose.pose.orientation
        pose = (float(p.x), float(p.y), float(p.z),
                quaternion_yaw(q.x, q.y, q.z, q.w))
        if not all(math.isfinite(value) for value in pose):
            return
        now = time.monotonic()
        with self._lock:
            self._truth_pose = pose
            self._truth_pose_received_at = now
            if (self._upper_floor_context_active and
                    self._truth_odom_anchor is None and self._pose is not None):
                self._truth_odom_anchor = self._pose
                self._truth_world_anchor = pose

    def _on_goal(self, message):
        p, q = message.pose.position, message.pose.orientation
        goal = (float(p.x), float(p.y), float(p.z),
                quaternion_yaw(q.x, q.y, q.z, q.w))
        if not all(math.isfinite(value) for value in goal):
            rospy.logerr("Rejected non-finite exploration goal")
            return
        now = time.monotonic()
        with self._lock:
            # ``direct_f3_rl_ready`` is only a bounded handoff/keepalive
            # token.  It is latched by the F3 bridge while the controller is
            # reacquiring its walking state and therefore must never replace
            # the controller's real locomotion-ready edge when accepting a
            # motion goal.  A stale latched token otherwise allows a goal to
            # keep publishing /cmd_vel after the low-level controller has
            # cleared /locomotion_ready (the F3 first-goal failure pattern).
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
            # Re-anchor truth-relative upper-floor navigation for every new
            # goal.  FAST-LIO may accumulate a large planar/yaw drift after
            # the floor handoff; keeping the one floor-start anchor makes a
            # freshly planned deep-room goal belong to a different frame than
            # the pose used by the executor.  Pairing the current odometry and
            # current Gazebo truth at goal acceptance preserves the planner's
            # requested local displacement physically, while subsequent goal
            # progress remains independent of registration drift.
            if ((self._upper_floor_context_active or
                 self._goal_truth_relative_navigation) and
                    pose is not None and
                    self._truth_pose is not None and
                    now - self._truth_pose_received_at <=
                    self._upper_floor_truth_guard_timeout):
                self._truth_odom_anchor = tuple(pose)
                self._truth_world_anchor = tuple(self._truth_pose)
                self._events.append({
                    "event": ("semantic_truth_anchor_refreshed_for_goal"
                              if self._goal_truth_relative_navigation else
                              "upper_floor_truth_anchor_refreshed_for_goal"),
                    "wall_time": time.time(),
                    "floor_index": int(self._upper_floor_index),
                    "goal_sequence": int(message.header.seq),
                    "odom_anchor": [float(value) for value in pose],
                    "truth_anchor": [float(value)
                                     for value in self._truth_pose],
                })
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
            self._navigation_stability_mode = "unrestricted"
            # The rotate-before-translate alignment latch belongs to one
            # waypoint only and must not leak across portal segments.
            self._translation_heading_aligned = False
            self._last_tick = now
            initial_distance = math.hypot(goal[0] - pose[0], goal[1] - pose[1]) if pose else math.inf
            self._watchdog.reset(self._progress_clock_now(), initial_distance)
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

    def _on_direct_f3_rl_ready(self, message):
        with self._lock:
            self._direct_f3_rl_ready = bool(message.data)
            if not self._direct_f3_rl_ready:
                self._publish_zero()

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

    def _on_upper_floor_truth_guard(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        with self._lock:
            self._upper_floor_truth_guard = payload
            self._upper_floor_truth_guard_received_at = time.monotonic()

    @staticmethod
    def _truth_guard_authorizes_upper_floor_motion(
            upper_floor_context, payload, age_seconds, timeout_seconds):
        return bool(
            upper_floor_context and isinstance(payload, dict) and
            age_seconds <= timeout_seconds and
            payload.get("enabled") and payload.get("active") and
            payload.get("f2_context") and payload.get("truth_received") and
            not payload.get("truth_stale"))

    def _on_localization_degraded(self, message):
        with self._lock:
            self._localization_degraded = bool(message.data)

    def _on_goal_degraded_speed_cap(self, message):
        value = float(message.data)
        if not math.isfinite(value) or value <= 0.0:
            return
        with self._lock:
            self._goal_degraded_speed_cap = max(
                self._minimum_speed, min(self._dynamic_speed_cap, value))

    @staticmethod
    def _stair_state_owns_command(state):
        """Return whether a stair controller exclusively owns /cmd_vel."""
        state = str(state or "").strip().upper()
        if state in ("", "WAIT_F1", "FIRST_FLOOR_FINALIZING",
                     "STAIR_POLICY_READY"):
            return False
        terminal_markers = (
            "HANDOFF_COMPLETE", "HANDOFF_TIMEOUT", "ASCENT_TIMEOUT",
            "POLICY_FAILED", "ENTRY_NOT_REACHED", "HANDOFF_NOT_REACHED",
            "FALL_DETECTED",
            "GT_GUIDE_FAILED", "THIRD_FLOOR_EXPLORATION_START")
        return not any(marker in state for marker in terminal_markers)

    def _on_stair_command_state(self, message, topic):
        state = str(message.data or "").strip()
        with self._lock:
            previous = self._stair_command_owned
            self._stair_command_states[str(topic)] = state
            self._stair_command_owned = any(
                self._stair_state_owns_command(item)
                for item in self._stair_command_states.values())
            if self._stair_command_owned != previous:
                self._last_command = [0.0, 0.0, 0.0]
                self._events.append({
                    "event": ("stair_command_ownership_acquired" if
                              self._stair_command_owned else
                              "stair_command_ownership_released"),
                    "wall_time": time.time(),
                    "topic": str(topic),
                    "state": state,
                    "all_stair_states": dict(self._stair_command_states),
                })
                rospy.loginfo(
                    "Goal Executor stair command ownership %s: %s=%s",
                    "acquired" if self._stair_command_owned else "released",
                    topic, state)
            # Once the round-trip descent starts, the F3 exploration frame no
            # longer owns localization.  Keeping its truth guard latched made
            # every F3->F2->F1 odometry callback look like a stale F3 height
            # and repeatedly re-anchored it throughout run52.  Release only
            # on an active descent phase (never on WAIT_F3), after the stair
            # controller has acquired command ownership.
            descent_started = bool(
                str(topic) == "/simenv/third_to_first_floor_stair_state" and
                state.strip().upper().startswith("STAIR_DESCENT_") and
                self._stair_command_owned)
            if descent_started and self._upper_floor_context_active:
                self._upper_floor_context_active = False
                self._upper_floor_index = -1
                self._upper_floor_truth_guard = None
                self._upper_floor_truth_guard_received_at = -math.inf
                self._truth_odom_anchor = None
                self._truth_world_anchor = None
                self._truth_vertical_rebase_active = False
                self._events.append({
                    "event": "f3_truth_context_released_for_descent",
                    "wall_time": time.time(),
                    "topic": str(topic),
                    "state": state,
                })

    def _on_third_floor_state(self, message):
        state = str(message.data or "").strip().upper()
        if state != "THIRD_FLOOR_EXPLORATION_START":
            return
        # Replace only the F2->F3 latch. F1->F2 and active recovery states
        # retain their existing command-ownership semantics.
        self._on_stair_command_state(
            message, "/simenv/second_to_third_floor_stair_state")

    def _on_speed_limit(self, message):
        value = float(message.data)
        if not math.isfinite(value) or value <= 0.0:
            return
        with self._lock:
            # Keep dynamic speed updates bounded by the calibrated flat-ground
            # cap while retaining the manager's phase-specific limit.
            self._maximum_speed = max(
                self._minimum_speed, min(self._dynamic_speed_cap, value))

    def _on_minimum_speed(self, message):
        """Apply a bounded per-goal speed floor for short waypoint chains."""
        value = float(message.data)
        if not math.isfinite(value) or value <= 0.0:
            return
        with self._lock:
            self._minimum_speed = bounded_dynamic_minimum_speed(
                value, self._dynamic_minimum_speed_floor,
                self._dynamic_speed_cap)
            self._maximum_speed = max(
                self._minimum_speed, self._maximum_speed)

    def _on_distance_gain(self, message):
        """Apply a bounded, phase-specific approach gain."""
        value = float(message.data)
        if not math.isfinite(value) or value <= 0.0:
            return
        with self._lock:
            # This changes only how quickly speed approaches the existing
            # calibrated cap; it cannot enlarge the cap itself.
            self._distance_gain = max(0.10, min(2.00, value))

    def _on_heading_gain(self, message):
        """Apply yaw correction only for the manager-selected motion phase."""
        value = float(message.data)
        if not math.isfinite(value):
            return
        with self._lock:
            self._heading_gain = max(0.0, min(2.0, value))

    def _on_target_heading(self, message):
        value = float(message.data)
        with self._lock:
            self._target_heading = (None if not math.isfinite(value) else
                                    math.atan2(math.sin(value), math.cos(value)))

    def _on_translation_heading_gate(self, message):
        """Set an optional rotate-before-translate gate for the next goal."""
        value = float(message.data)
        with self._lock:
            self._translation_heading_gate = (
                math.inf if not math.isfinite(value) else
                max(0.08, min(0.50, value)))
            if not math.isfinite(value):
                self._translation_heading_aligned = False

    def _on_goal_truth_relative_navigation(self, message):
        """Select a manager-authorized truth-relative pose for the next goal.

        The manager publishes this only after a locked open-room G3/G4 chord
        has passed truth room-bound and expanded-furniture validation.  This
        does not authorize arbitrary F1 truth motion and is reset explicitly
        before every ordinary waypoint by the manager.
        """
        with self._lock:
            self._goal_truth_relative_navigation = bool(message.data)


    def _on_maximum_yaw_rate(self, message):
        """Apply a bounded yaw-rate limit for the selected motion phase."""
        value = float(message.data)
        if not math.isfinite(value) or value <= 0.0:
            return
        with self._lock:
            self._maximum_yaw_rate = max(0.05, min(1.60, value))
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
            requested_speed = abs(float(payload.get("angular_speed", 1.20)))
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
            speed = (max(0.10, min(self._maximum_visual_sweep_speed,
                                   requested_speed)) if visual_sweep else
                     max(0.10, min(self._maximum_mapping_rescan_speed,
                                   requested_speed)))
            timeout = max(2.0 if visual_sweep else 10.0,
                          float(payload.get("timeout_sec", angle / speed + 5.0)))
            direction = -1.0 if float(payload.get("direction", 1.0)) < 0.0 else 1.0
        except (KeyError, TypeError, ValueError, OverflowError):
            rospy.logwarn("Ignored malformed local rescan request")
            return
        now = time.monotonic()
        ros_started = self._ros_sim_time_now()
        with self._lock:
            if (not self._locomotion_ready or
                    self._rescan is not None or
                    (self._goal is not None and not self._finished)):
                accepted = False
            else:
                self._rescan = {
                    "request_id": request_id, "started": now,
                    # Mission acceptance is based on Gazebo/ROS simulation
                    # time. Keep monotonic time exclusively as the paused-
                    # simulator watchdog and persist both clocks.
                    "ros_started": ros_started,
                    "duration": angle / speed, "timeout": timeout,
                    "speed": direction * speed, "angle": angle,
                    "visual_sweep": visual_sweep,
                    "full_visual_sweep": full_visual_sweep,
                    "room_id": room_id,
                    # A commanded-rate duration is not proof that the body
                    # physically turned through the requested angle.  The RL
                    # gait accelerates into yaw and run29 measured about
                    # 1.85 rad/s for a 2.0 rad/s command, leaving a possible
                    # unseen sector.  Integrate stabilized odometry yaw for
                    # visual sweeps and use time only as the watchdog.
                    "last_yaw": (float(self._pose[3])
                                 if self._pose is not None and
                                 len(self._pose) >= 4 else None),
                    "turned_angle": 0.0,
                    "candidate_slow_until": -math.inf,
                }
                accepted = True
        if not accepted:
            self._rescan_result_pub.publish(String(data=json.dumps({
                "request_id": request_id, "success": False,
                "reason": "executor_busy_or_locomotion_not_ready",
                "visual_sweep": visual_sweep, "room_id": room_id,
            }, sort_keys=True)))
        else:
            rospy.loginfo("Local rescan %s accepted: %.0f deg at %.2f rad/s",
                          request_id, math.degrees(angle), speed)

    def _on_red_ball_observation(self, message):
        """Slow only the small angular sector containing a red candidate.

        A distant 30 cm sphere can cross the RGB image in only one reliable
        frame at the normal 2 rad/s sweep.  Holding the entire revolution at
        the old 1.6 rad/s costs every room; instead, retain fast scanning and
        briefly slow after an actual strict detector observation so the
        existing multi-frame/yaw-diversity tracker can confirm or reject it.
        """
        try:
            payload = json.loads(message.data)
            room_id = str(payload.get("room_id") or "")
            confidence = float(payload.get("confidence", 0.0))
            pixel = payload.get("pixel") or {}
            strict = (str(pixel.get("shape_gate", "strict")) == "strict" and
                      not bool(payload.get("profile_relaxed", False)))
        except (TypeError, ValueError, AttributeError):
            return
        if not strict or confidence < 0.90:
            return
        now = time.monotonic()
        with self._lock:
            if (self._rescan is None or
                    not self._rescan.get("visual_sweep") or
                    str(self._rescan.get("room_id") or "") != room_id):
                return
            self._rescan["candidate_slow_until"] = max(
                float(self._rescan.get("candidate_slow_until", -math.inf)),
                now + self._visual_candidate_slow_seconds)

    
    @staticmethod
    def _ros_sim_time_now():
        """Return ROS/Gazebo simulation time, or ``None`` before ROS init."""
        try:
            value = float(rospy.Time.now().to_sec())
        except Exception:
            # Unit construction and early roslaunch callbacks can run before
            # rospy has initialized its time source.  Wall time remains the
            # watchdog, but must never be mislabeled as simulation time.
            return None
        return value if math.isfinite(value) and value >= 0.0 else None

    @staticmethod
    def _progress_clock_now():
        """Use simulation time for motion-progress budgets when available."""
        try:
            value = float(rospy.Time.now().to_sec())
        except (AttributeError, TypeError, ValueError):
            value = math.nan
        return value if math.isfinite(value) and value > 0.0 else time.monotonic()

    def _publish_zero(self, expected_generation=None):
        with self._lock:
            if getattr(self, "_stair_command_owned", False):
                return False
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
            upper_floor_context = self._upper_floor_context_active
            upper_floor_index = self._upper_floor_index
            truth_guard = dict(self._upper_floor_truth_guard or {})
            truth_guard_age = (
                now - self._upper_floor_truth_guard_received_at)
            truth_pose = self._truth_pose
            truth_pose_age = now - self._truth_pose_received_at
            truth_odom_anchor = self._truth_odom_anchor
            truth_world_anchor = self._truth_world_anchor
            truth_vertical_rebase_active = \
                self._truth_vertical_rebase_active
            localization_degraded = self._localization_degraded
            pose_frame = self._pose_frame
            goal_frame = self._goal_frame
            finished = self._finished
            generation = self._goal_generation
            recovery_until = self._stall_recovery_until
            rescan = dict(self._rescan) if self._rescan is not None else None
            pending_cancel = (dict(self._pending_cancel)
                              if self._pending_cancel is not None else None)
            # The direct F3 token is diagnostic/keepalive state only.  Motion
            # and rescans require the authoritative low-level latch.
            locomotion_ready = bool(self._locomotion_ready)
            measured_speed = self._measured_speed
            last_command = tuple(self._last_command)
            target_heading = self._target_heading
            goal_truth_relative_navigation = bool(
                self._goal_truth_relative_navigation)
            translation_heading_gate = getattr(
                self, "_translation_heading_gate", math.inf)
            translation_heading_aligned = getattr(
                self, "_translation_heading_aligned", False)
            stair_command_owned = self._stair_command_owned
        if stair_command_owned:
            # Another publisher owns /cmd_vel for the complete physical stair
            # transition. Silence both motion and zero commands: publishing a
            # zero here at 20 Hz breaks the stair policy's continuous gait.
            self._last_tick = now
            return
        if rescan is not None:
            elapsed = now - rescan["started"]
            turned_angle = float(rescan.get("turned_angle", 0.0))
            if rescan.get("visual_sweep"):
                with self._lock:
                    live_rescan = self._rescan
                    live_yaw = (float(self._pose[3])
                                if self._pose is not None and
                                len(self._pose) >= 4 else None)
                    if (live_rescan is not None and live_yaw is not None and
                            live_rescan.get("last_yaw") is not None):
                        previous_yaw = float(live_rescan["last_yaw"])
                        yaw_delta = math.atan2(
                            math.sin(live_yaw - previous_yaw),
                            math.cos(live_yaw - previous_yaw))
                        direction = (-1.0 if float(live_rescan["speed"]) < 0.0
                                     else 1.0)
                        directed_delta = direction * yaw_delta
                        # Ignore reverse compliance and reject a single
                        # implausible localization jump; normal 20 Hz motion
                        # at 2 rad/s contributes about 0.10 rad per tick.
                        if 0.0 < directed_delta <= 0.35:
                            live_rescan["turned_angle"] = float(
                                live_rescan.get("turned_angle", 0.0)) + \
                                directed_delta
                        live_rescan["last_yaw"] = live_yaw
                        turned_angle = float(live_rescan["turned_angle"])
            completed = (
                turned_angle >= max(
                    0.0, float(rescan["angle"]) -
                    self._visual_sweep_completion_tolerance)
                if rescan.get("visual_sweep") else
                elapsed >= rescan["duration"])
            if completed or elapsed >= rescan["timeout"]:
                success = bool(completed)
                ros_finished = self._ros_sim_time_now()
                ros_started = rescan.get("ros_started")
                ros_duration = (
                    max(0.0, float(ros_finished) - float(ros_started))
                    if ros_finished is not None and ros_started is not None
                    else None)
                with self._lock:
                    if (self._rescan is not None and
                            self._rescan["request_id"] == rescan["request_id"]):
                        self._rescan = None
                self._publish_zero()
                self._rescan_result_pub.publish(String(data=json.dumps({
                    "request_id": rescan["request_id"], "success": success,
                    "reason": "local_rescan_complete" if success else "local_rescan_timeout",
                    # ``duration_sec`` is retained for backward-compatible
                    # watchdog diagnostics. Acceptance/reporting consumes the
                    # explicit ROS simulation-time fields below.
                    "duration_sec": elapsed,
                    "wall_clock_seconds": elapsed,
                    "start_ros_sim_s": ros_started,
                    "end_ros_sim_s": ros_finished,
                    "duration_ros_sim_s": ros_duration,
                    "angle_rad": rescan["angle"],
                    "measured_turned_angle_rad": turned_angle,
                    "visual_sweep": bool(rescan.get("visual_sweep", False)),
                    "room_id": rescan.get("room_id"),
                }, sort_keys=True)))
                return
            command = Twist()
            nominal_speed = float(rescan["speed"])
            if now < float(rescan.get("candidate_slow_until", -math.inf)):
                command.angular.z = math.copysign(
                    min(abs(nominal_speed), self._visual_candidate_slow_speed),
                    nominal_speed)
            else:
                command.angular.z = nominal_speed
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
        if not locomotion_ready:
            # The low-level controller clears this latch when the robot falls
            # or leaves RL locomotion.  Publishing one zero in the subscriber
            # callback is insufficient: the next 20 Hz timer tick used to
            # resume the stale goal and drive against predicted FAST-LIO
            # odometry.  Finish the generation while holding zero instead.
            self._finish(False, "locomotion_not_ready_during_goal", generation)
            return
        if pose_fault:
            self._finish(False, pose_fault, generation)
            return
        truth_guard_authorized = (
            self._truth_guard_authorizes_upper_floor_motion(
                upper_floor_context, truth_guard, truth_guard_age,
                self._upper_floor_truth_guard_timeout))
        semantic_truth_authorized = bool(
            goal_truth_relative_navigation and
            truth_pose is not None and truth_odom_anchor is not None and
            truth_world_anchor is not None and
            truth_pose_age <= self._upper_floor_truth_guard_timeout)
        truth_navigation_authorized = bool(
            truth_guard_authorized or semantic_truth_authorized)
        if (not registration_healthy and not truth_navigation_authorized and
                registration_invalid_count >=
                self._registration_abort_invalid_count):
            self._finish(False, "fastlio_registration_lost", generation)
            return
        if not registration_healthy and not truth_navigation_authorized:
            # Match the exploration manager's debounce contract. A single
            # delayed high-speed scan stops motion but preserves the active
            # goal so it can resume after registration recovers.
            self._publish_zero(generation)
            return
        if truth_navigation_authorized:
            if not registration_healthy:
                rospy.logwarn_throttle(
                    5.0, "Upper-floor truth motion guard is carrying navigation "
                    "through sparse/frozen scan registration")
            if (truth_pose is not None and truth_odom_anchor is not None and
                    truth_world_anchor is not None and
                    truth_pose_age <= self._upper_floor_truth_guard_timeout):
                try:
                    pose = pose_from_truth_delta(
                        truth_odom_anchor, truth_world_anchor, truth_pose)
                    pose_age = truth_pose_age
                    rospy.loginfo_throttle(
                        5.0, "Upper-floor navigation and goal arrival are using "
                        "truth-relative displacement in the anchored odometry frame")
                except ValueError:
                    pass
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
                    self._watchdog.reset(self._progress_clock_now(), math.hypot(
                        goal[0] - pose[0], goal[1] - pose[1]))
        effective_maximum_speed = self._maximum_speed
        if upper_floor_context and truth_guard_authorized:
            effective_maximum_speed = min(
                effective_maximum_speed, self._upper_floor_maximum_speed)
        if localization_degraded:
            effective_maximum_speed = max(
                self._minimum_speed,
                min(effective_maximum_speed, self._goal_degraded_speed_cap))
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
            nominal_vx, nominal_vy, nominal_wz = vx, vy, wz
            if target_heading is not None:
                heading_error = math.atan2(
                    math.sin(target_heading - pose[3]),
                    math.cos(target_heading - pose[3]))
                wz = max(-self._maximum_yaw_rate,
                         min(self._maximum_yaw_rate,
                             self._heading_gain * heading_error))
            vx, vy, wz, stability_mode = navigation_stability_envelope(
                vx, vy, wz, heading_error, last_command,
                large_heading_error=self._navigation_large_heading_error,
                high_yaw_rate=self._navigation_high_yaw_rate,
                fast_turn_translation_threshold=(
                    self._navigation_fast_turn_translation_threshold),
                braking_yaw_rate=self._navigation_braking_yaw_rate,
                medium_heading_error=self._navigation_medium_heading_error,
                turning_translation_limit=(
                    self._navigation_turning_translation_limit),
                turning_lateral_limit=(
                    self._navigation_turning_lateral_limit),
            )
        except ValueError as error:
            self._finish(False, str(error), generation)
            return
        # With scan registration frozen on F3, simultaneous lateral and yaw
        # commands amplify the landing policies inconsistent lateral response.
        # Use a bounded rotate-then-forward controller in this truth-authorized
        # fallback only. Normal F3 localization and every F1/F2 command retain
        # the established holonomic controller.
        f3_frozen_registration_fallback = bool(
            upper_floor_index == 2 and not registration_healthy)
        f3_truth_heading_locked_goal = bool(
            upper_floor_index == 2 and target_heading is not None)
        if (truth_guard_authorized and
                (f3_frozen_registration_fallback or
                 (upper_floor_index == 2 and
                  truth_vertical_rebase_active) or
                 f3_truth_heading_locked_goal)):
            # A long F3 corridor target can follow a 170-degree doorway
            # observation.  Translating at the generic 0.55 rad threshold
            # made the quadruped arc into the room-side wall even though the
            # goal and truth-relative pose were correct.  Finish the physical
            # turn first, then use body-forward motion only.
            if abs(heading_error) > 0.20:
                vx = 0.0
                vy = 0.0
                # A rotate-first contract is ineffective if its yaw command
                # sits inside the plane-policy deadband.  Unlike an EXIT,
                # corridor-resume goals do not carry a finite translation
                # heading gate, so enforce the executable floor here before
                # slew limiting.  The absolute cap remains authoritative.
                minimum_yaw = min(
                    self._absolute_maximum_yaw_rate,
                    self._truth_rotate_minimum_yaw_rate)
                wz = math.copysign(max(abs(wz), minimum_yaw),
                                   heading_error)
            else:
                # Once the body is aligned, retain a small lateral correction.
                # EXIT fixes yaw to the outward door normal; forcing vy=0 for
                # the entire goal makes any residual jamb-centre offset
                # geometrically uncorrectable.  The dog then reaches the door
                # plane, stalls 0.3--0.4 m beside the corridor endpoint, and
                # burns every EXIT retry.  This bounded correction is only
                # enabled after the rotate-first gate has completed.
                vy = max(-0.12, min(0.12, vy))
        # EXIT waypoints carry an explicit outward-normal heading.  In the
        # failed deepdebtrepair run the generic blended controller translated
        # at ~0.45 rad heading error and stopped 0.3 m inside room1's narrow
        # doorway.  Enforce the manager-selected gate on every floor.  Once
        # aligned, retain the ordinary holonomic correction so a small
        # centreline offset can still be removed before crossing.
        if (target_heading is not None and
                math.isfinite(translation_heading_gate)):
            # A single threshold chattered at F1 room1: the body reached the
            # strict 0.16-rad gate, began moving, then ordinary gait yaw
            # oscillation crossed back to 0.18--0.20 rad and stopped it until
            # the EXIT watchdog expired.  Latch strict initial alignment and
            # release only on a materially unsafe deviation.  This preserves
            # rotate-before-translate without repeatedly braking in the jamb.
            release_gate = min(
                0.55, max(translation_heading_gate + 0.14,
                          2.0 * translation_heading_gate))
            if translation_heading_aligned:
                translation_heading_aligned = bool(
                    abs(heading_error) <= release_gate)
            elif abs(heading_error) <= translation_heading_gate:
                translation_heading_aligned = True
            with self._lock:
                if generation == self._goal_generation:
                    self._translation_heading_aligned = bool(
                        translation_heading_aligned)
            if not translation_heading_aligned:
                vx = 0.0
                vy = 0.0
                # Preserve the sign selected from the physical heading error,
                # and command a rate the gait can actually execute.  Existing
                # latch/release hysteresis prevents chatter after alignment.
                if abs(heading_error) > 1e-4:
                    minimum_yaw = min(
                        getattr(self, "_absolute_maximum_yaw_rate",
                                max(self._maximum_yaw_rate, 0.42)),
                        getattr(self,
                                "_translation_heading_minimum_yaw_rate",
                                0.42))
                    wz = rotation_only_exit_alignment_yaw_rate(
                        heading_error,
                        nominal_yaw_rate=max(abs(wz), minimum_yaw),
                        minimum_yaw_rate=minimum_yaw,
                        fast_yaw_rate=getattr(
                            self, "_translation_heading_rotation_yaw_rate",
                            minimum_yaw),
                        fast_error_threshold=getattr(
                            self,
                            "_translation_heading_fast_rotation_error", 0.70),
                        absolute_yaw_rate=getattr(
                            self, "_absolute_maximum_yaw_rate",
                            max(abs(wz), minimum_yaw)))
            # run146 aligned the deep-room EXIT correctly, but the F3 truth
            # envelope reduced the remaining lateral correction to 0.12 m/s,
            # below the plane policy's measured translational deadband. A
            # finite gate is published only for EXIT and cleared with NaN for
            # every other role. Lift only an already nonzero, aligned command;
            # a safety or alignment stop at zero remains authoritative.
            vx, vy, translation_floor_applied = \
                enforce_executable_translation_floor(
                    vx, vy, self._minimum_speed,
                    enabled=translation_heading_aligned)
            if translation_floor_applied:
                stability_mode = "aligned_exit_translation_floor"
        # ``command_for_goal`` already returns a body-frame command.  The
        # F3 landing may leave the body yaw near pi, so a north-bound world
        # goal naturally has a negative body-y component.  Do not apply a
        # second F3-only sign flip here: that sends the dog south toward the
        # stair lobby and was observed as a repeated 3 m probe timeout.
        if distance < self._goal_tolerance:
            self._finish(True, "goal_reached", generation)
            return
        if self._watchdog.update(self._progress_clock_now(), distance):
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
            "nominal_linear_x": nominal_vx,
            "nominal_linear_y": nominal_vy,
            "nominal_angular_z": nominal_wz,
            "navigation_stability_mode": stability_mode,
        }
        with self._lock:
            # Goal callbacks and timer callbacks run concurrently in rospy.
            # Discard a command computed from a goal that has since been
            # replaced, including its telemetry sample.
            if generation != self._goal_generation or self._finished:
                return
            self._last_command = [command.linear.x, command.linear.y,
                                  command.angular.z]
            if stability_mode != self._navigation_stability_mode:
                self._events.append({
                    "event": "navigation_stability_mode_changed",
                    "wall_time": time.time(),
                    "from": self._navigation_stability_mode,
                    "to": stability_mode,
                    "heading_error": heading_error,
                    "nominal_translation_mps": math.hypot(
                        nominal_vx, nominal_vy),
                    "nominal_yaw_rate_radps": nominal_wz,
                })
                self._navigation_stability_mode = stability_mode
            self._desired_command_pub.publish(command)
            self._command_pub.publish(command)
            self._samples.append(sample)
            if len(self._samples) > 10000:
                self._samples = self._samples[-10000:]
        rospy.loginfo_throttle(
            2.0,
            "[MOTION] v=%.3f m/s (vx=%.3f vy=%.3f) wz=%.3f rad/s "
            "goal_dist=%.2f m heading_err=%.2f rad",
            math.hypot(command.linear.x, command.linear.y),
            command.linear.x, command.linear.y, command.angular.z,
            distance, heading_error)
        if stability_mode == "brake_before_fast_turn":
            rospy.logwarn_throttle(
                1.0, "[MOTION SAFETY] braking translation before %.2f "
                "rad/s navigation turn (heading error %.2f rad)",
                nominal_wz, heading_error)
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
                "navigation_large_heading_error_rad":
                    self._navigation_large_heading_error,
                "navigation_high_yaw_rate_radps":
                    self._navigation_high_yaw_rate,
                "navigation_fast_turn_translation_threshold_mps":
                    self._navigation_fast_turn_translation_threshold,
                "navigation_braking_yaw_rate_radps":
                    self._navigation_braking_yaw_rate,
                "navigation_medium_heading_error_rad":
                    self._navigation_medium_heading_error,
                "navigation_turning_translation_limit_mps":
                    self._navigation_turning_translation_limit,
                "navigation_turning_lateral_limit_mps":
                    self._navigation_turning_lateral_limit,
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
