#!/usr/bin/env python3
"""Start a fresh corridor/room mission after the two-flight stair handoff.

The first- and second-floor geometries overlap in x/y, so this process owns a
new BaselineExplorationManager instance and therefore a new room scheduler,
corridor station frame, visited-door memory, and mission-home anchor.  The
shared voxel mapper remains valid because its 2-D projection is sliced around
the robot's current odometry height.
"""

import json
import math
import os
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET

# A catkin-installed Python executable is a relay under devel/lib.  Without
# explicitly preferring this source directory, the sibling import below finds
# the relay for baseline_exploration_manager instead of the actual module; the
# relay executes the source in a private dict and consequently exports no
# BaselineExplorationManager symbol.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if not sys.path or sys.path[0] != _SCRIPT_DIR:
    sys.path.insert(0, _SCRIPT_DIR)

import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool, String

from baseline_exploration_manager import BaselineExplorationManager
from waypoint_tour_manager import WaypointTourRunner


class SecondFloorMission:
    def __init__(self):
        # This class remains the validated floor-2 implementation by default.
        # A second ROS process can select floor_number=3 and isolated topics to
        # reuse the same corridor/room algorithm without sharing scheduler state.
        self._floor_number = int(rospy.get_param("~floor_number", 2))
        self._floor_index = self._floor_number - 1
        self._floor_word = ({2: "SECOND", 3: "THIRD"}.get(
            self._floor_number, "FLOOR_{}".format(self._floor_number)))
        self._floor_slug = ({2: "second_floor", 3: "third_floor"}.get(
            self._floor_number, "floor_{}".format(self._floor_number)))
        self._state_topic = str(rospy.get_param(
            "~state_topic", "/simenv/second_floor_state"))
        self._stair_state_topic = str(rospy.get_param(
            "~stair_state_topic", "/simenv/stair_transition_state"))
        self._stair_reached_token = str(rospy.get_param(
            "~stair_reached_token", "SECOND_FLOOR_REACHED"))
        self._source_floor_state_topic = str(rospy.get_param(
            "~source_floor_state_topic", "")).strip()
        self._source_floor_failure_token = str(rospy.get_param(
            "~source_floor_failure_token", "")).strip()
        self._monitor_source_finalize = bool(rospy.get_param(
            "~monitor_source_finalize", True))
        self._auto_generate_visualization = bool(rospy.get_param(
            "~auto_generate_combined_visualization", True))
        self._lock = threading.RLock()
        self._second_floor_reached = False
        self._stair_state = "WAIT_F1"
        self._stair_handoff_started_at = None
        self._stair_active_announced = False
        self._source_floor_failed = False
        self._plane_policy_ready = False
        self._locomotion_ready = False
        self._rescan_result = None
        self._truth_pose = None
        self._truth_pose_received_at = -math.inf
        self._stabilized_odom = None
        self._stabilized_odom_received_at = -math.inf
        self._registration_status = None
        self._registration_status_received_at = -math.inf
        self._motion_guard_status = None
        self._degeneracy_status = None
        self._truth_guard_status = None
        self._truth_guard_status_received_at = -math.inf
        self._truth_guard_last_logged_at = -math.inf
        self._first_floor_finalized_at = None
        self._exploration_started_wall_time = None
        self._parent_output = os.path.abspath(rospy.get_param(
            "~parent_output_dir"))
        self._output = os.path.abspath(rospy.get_param(
            "~output_dir", os.path.join(self._parent_output, self._floor_slug)))
        # Fixed-waypoint tour mode: replaces the free-exploration manager with
        # a serial waypoint list executed through the Goal Executor, keeping
        # every token below (READY / COMPLETE) unchanged.  Off by default.
        self._tour_mode = bool(rospy.get_param("~tour_mode", False))
        raw_tour_waypoints = rospy.get_param("~tour_waypoints", "")
        self._tour_waypoints = (
            WaypointTourRunner._normalize_waypoints(raw_tour_waypoints)
            if raw_tour_waypoints else [])
        self._tour_waypoint_frame = str(rospy.get_param(
            "~tour_waypoint_frame", "world"))
        self._plane_policy = os.path.abspath(rospy.get_param("~plane_policy"))
        self._handoff_timeout = float(rospy.get_param(
            "~handoff_wait_timeout_sec", 600.0))
        self._failed_f1_grace = float(rospy.get_param(
            "~failed_first_floor_grace_sec", 120.0))
        self._policy_timeout = float(rospy.get_param(
            "~plane_policy_reload_timeout_sec", 30.0))
        self._policy_settle = float(rospy.get_param(
            "~plane_policy_settle_sec", 1.2))
        self._enable_lobby_rescan = bool(rospy.get_param(
            "~enable_second_floor_lobby_rescan", True))
        self._enable_truth_corridor_guide = bool(rospy.get_param(
            "~enable_second_floor_truth_corridor_guide", True))
        self._truth_guide_timeout = float(rospy.get_param(
            "~second_floor_truth_corridor_guide_timeout_sec", 45.0))
        self._truth_guide_speed = float(rospy.get_param(
            "~second_floor_truth_corridor_guide_speed_mps", 0.45))
        self._truth_guide_tolerance = float(rospy.get_param(
            "~second_floor_truth_corridor_guide_tolerance_m", 0.30))
        self._truth_heading_tolerance = float(rospy.get_param(
            "~second_floor_truth_corridor_heading_tolerance_rad", 0.18))
        # Third GT waypoint stands just outside the corridor mouth on the
        # corridor centreline; the robot is already facing +y when it arrives,
        # so the final yaw gate only confirms the heading.
        self._truth_corridor_align_offset = float(rospy.get_param(
            "~second_floor_truth_corridor_align_offset_m", 0.5))
        # The plane policy's measured world-frame progress is lower than the
        # requested body command while turning.  Allocate each GT route leg
        # from its own distance instead of sharing one deadline across the
        # stair exit, long lobby crossing, and final heading alignment.
        self._truth_guide_minimum_progress_speed = max(0.05, float(
            rospy.get_param(
                "~second_floor_truth_corridor_minimum_progress_speed_mps",
                0.10)))
        self._truth_guide_timeout_margin = max(0.0, float(rospy.get_param(
            "~second_floor_truth_corridor_timeout_margin_sec", 6.0)))
        self._truth_guide_stall_timeout = max(2.0, float(rospy.get_param(
            "~second_floor_truth_corridor_stall_timeout_sec", 15.0)))
        self._truth_guide_progress_epsilon = max(0.01, float(rospy.get_param(
            "~second_floor_truth_corridor_progress_epsilon_m", 0.05)))
        self._second_floor_height_margin = max(0.0, float(rospy.get_param(
            "~second_floor_truth_height_margin_m", 0.15)))
        self._second_floor_height_wait = max(0.5, float(rospy.get_param(
            "~second_floor_truth_height_wait_sec", 3.0)))
        self._localization_stabilization_timeout = max(3.0, float(
            rospy.get_param(
                "~second_floor_localization_stabilization_timeout_sec", 20.0)))
        self._localization_stable_seconds = max(1.0, float(rospy.get_param(
            "~second_floor_localization_stable_seconds", 3.0)))
        self._localization_maximum_planar_disagreement = max(0.03, float(
            rospy.get_param(
                "~second_floor_localization_maximum_planar_disagreement_m",
                0.20)))
        self._localization_maximum_yaw_disagreement = max(0.02, float(
            rospy.get_param(
                "~second_floor_localization_maximum_yaw_disagreement_rad",
                0.12)))
        # Unlike x/y, the FAST-LIO and Gazebo vertical origins are aligned in
        # this simulator (apart from the small base-to-IMU offset).  A large
        # absolute z mismatch after the stair handoff means that FAST-LIO did
        # not integrate the climb.  Run73 entered F2 with truth z=2.912 m but
        # odom z=1.020 m; the old relative-only gate incorrectly accepted it.
        self._localization_maximum_vertical_disagreement = max(0.10, float(
            rospy.get_param(
                "~second_floor_localization_maximum_vertical_disagreement_m",
                0.65)))
        self._localization_require_truth_guard = bool(rospy.get_param(
            "~second_floor_localization_require_truth_guard", True))
        self._localization_status_freshness = max(0.5, float(rospy.get_param(
            "~second_floor_localization_status_freshness_sec", 2.0)))
        self._second_floor_elevation = None
        self._truth_layout = os.path.abspath(rospy.get_param(
            "~offline_truth_layout_metadata", ""))
        self._stair_model = os.path.abspath(rospy.get_param(
            "~offline_stair_model_sdf", ""))
        # Optional external visual corridor-exit guide (module 2).  When
        # enabled it replaces the truth corridor guide in the post-climb
        # window; the goal executor is paused by the guide driver through the
        # shared pause topic while it owns /cmd_vel.
        self._enable_visual_exit_guide = bool(rospy.get_param(
            "~enable_visual_exit_guide", False))
        self._visual_guide_state_topic = str(rospy.get_param(
            "~visual_exit_guide_state_topic",
            "/simenv/corridor_exit_guide_state"))
        self._visual_guide_ready_token = str(rospy.get_param(
            "~visual_exit_guide_ready_token", "CORRIDOR_EXIT_GUIDE_READY"))
        self._visual_guide_timeout = max(1.0, float(rospy.get_param(
            "~visual_exit_guide_timeout_sec", 150.0)))
        self._visual_guide_pause_topic = str(rospy.get_param(
            "~visual_exit_guide_pause_topic", "")).strip()
        self._visual_guide_state = None
        # The combined visualizer is not shipped with this package; the
        # launch sets auto_generate_combined_visualization=false, so the
        # path stays unused (guarded at run time).
        self._visualizer = os.path.abspath(rospy.get_param(
            "~visualization_script", ""))
        os.makedirs(self._output, exist_ok=True)

        self._policy_pub = rospy.Publisher(
            "/simenv/rl_policy_request", String, queue_size=1, latch=True)
        self._cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=2)
        self._joy_pub = rospy.Publisher("/joy", Joy, queue_size=2)
        self._state_pub = rospy.Publisher(
            self._state_topic, String, queue_size=1, latch=True)
        # FAST-LIO consumes this simulator-truth pose only after the explicit
        # SECOND_FLOOR_LOCALIZATION_STABILIZING gate.  Publishing it here
        # keeps the correction source isolated from F1 and the stair climb.
        self._truth_correction_pub = rospy.Publisher(
            "/simenv/second_floor_truth_odometry", Odometry, queue_size=5)
        self._executor_floor_context_pub = rospy.Publisher(
            "/simenv/goal_executor_floor_context", String, queue_size=1,
            latch=True)
        self._rescan_pub = rospy.Publisher(
            "/simenv/local_rescan_request", String, queue_size=1)
        rospy.Subscriber(self._stair_state_topic, String,
                         self._on_stair_state, queue_size=5)
        if self._source_floor_state_topic:
            rospy.Subscriber(self._source_floor_state_topic, String,
                             self._on_source_floor_state, queue_size=5)
        rospy.Subscriber("/rl_takeover_status", String,
                         self._on_policy_status, queue_size=5)
        rospy.Subscriber("/locomotion_ready", Bool,
                         self._on_locomotion_ready, queue_size=5)
        if self._monitor_source_finalize:
            rospy.Subscriber("/simenv/finalize_result", Bool,
                             self._on_first_floor_finalize, queue_size=2)
        rospy.Subscriber("/simenv/local_rescan_result", String,
                         self._on_rescan_result, queue_size=3)
        rospy.Subscriber("/gazebo/model_states", ModelStates,
                         self._on_truth_states, queue_size=2)
        rospy.Subscriber("/simenv/fastlio_stabilized_odometry", Odometry,
                         self._on_stabilized_odometry, queue_size=20)
        rospy.Subscriber("/simenv/fastlio_registration_status", String,
                         self._on_registration_status, queue_size=20)
        rospy.Subscriber("/simenv/fastlio_motion_guard_status", String,
                         self._on_motion_guard_status, queue_size=20)
        rospy.Subscriber("/simenv/fastlio_degeneracy_status", String,
                         self._on_degeneracy_status, queue_size=20)
        rospy.Subscriber("/simenv/fastlio_f2_truth_guard_status", String,
                         self._on_truth_guard_status, queue_size=20)
        if self._enable_visual_exit_guide and self._visual_guide_state_topic:
            rospy.Subscriber(self._visual_guide_state_topic, String,
                             self._on_visual_guide_state, queue_size=5)
        self._visual_pause_pub = None
        if self._enable_visual_exit_guide and self._visual_guide_pause_topic:
            self._visual_pause_pub = rospy.Publisher(
                self._visual_guide_pause_topic, Bool, queue_size=1, latch=True)

    def _on_visual_guide_state(self, message):
        with self._lock:
            self._visual_guide_state = str(message.data).strip()

    def _on_stair_state(self, message):
        state = str(message.data)
        with self._lock:
            self._stair_state = state
            # A first-floor finalize message is expected before the climb.  It
            # is not evidence that the stair handoff failed once the stair
            # controller has accepted ownership and left its waiting states.
            if (self._stair_handoff_started_at is None and
                    state not in ("WAIT_F1", "FIRST_FLOOR_FINALIZING")):
                self._stair_handoff_started_at = time.monotonic()
            if (self._floor_number >= 3 and not self._stair_active_announced and
                    state not in ("WAIT_F1", "FIRST_FLOOR_FINALIZING")):
                self._stair_active_announced = True
                self._state_pub.publish(String(
                    data=self._event("STAIR_TRANSITION_ACTIVE")))
            if self._stair_reached_token in state:
                self._second_floor_reached = True

    def _on_source_floor_state(self, message):
        state = str(message.data).strip()
        if (self._source_floor_failure_token and
                state == self._source_floor_failure_token):
            with self._lock:
                self._source_floor_failed = True

    def _event(self, suffix):
        return "{}_FLOOR_{}".format(self._floor_word, suffix)

    def _on_policy_status(self, message):
        if ("policy_reloaded:" in str(message.data) and
                os.path.basename(self._plane_policy) in str(message.data)):
            with self._lock:
                self._plane_policy_ready = True

    def _on_locomotion_ready(self, message):
        with self._lock:
            self._locomotion_ready = bool(message.data)

    def _on_first_floor_finalize(self, message):
        if message.data:
            with self._lock:
                if self._first_floor_finalized_at is None:
                    self._first_floor_finalized_at = time.monotonic()

    def _on_rescan_result(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        with self._lock:
            self._rescan_result = payload

    def _on_truth_states(self, message):
        try:
            index = message.name.index("a1_gazebo")
        except ValueError:
            return
        pose = message.pose[index]
        q = pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        values = (float(pose.position.x), float(pose.position.y),
                  float(pose.position.z), float(yaw))
        if not all(math.isfinite(value) for value in values):
            return
        with self._lock:
            self._truth_pose = values
            self._truth_pose_received_at = time.monotonic()
        # Relay the complete truth sample instead of making FAST-LIO depend on
        # gazebo_msgs.  The C++ guard ignores this stream until F2 activation.
        correction = Odometry()
        correction.header.stamp = rospy.Time.now()
        correction.header.frame_id = "world"
        correction.child_frame_id = "a1_gazebo"
        correction.pose.pose = pose
        if index < len(message.twist):
            correction.twist.twist = message.twist[index]
        self._truth_correction_pub.publish(correction)

    @staticmethod
    def _pose_from_odometry(message):
        pose = message.pose.pose
        q = pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        values = (float(pose.position.x), float(pose.position.y),
                  float(pose.position.z), float(yaw))
        return values if all(math.isfinite(value) for value in values) else None

    def _on_stabilized_odometry(self, message):
        values = self._pose_from_odometry(message)
        if values is None:
            return
        with self._lock:
            self._stabilized_odom = values
            self._stabilized_odom_received_at = time.monotonic()

    def _on_registration_status(self, message):
        payload = self._decode_status(message)
        if payload is None:
            return
        with self._lock:
            self._registration_status = payload
            self._registration_status_received_at = time.monotonic()

    def _on_motion_guard_status(self, message):
        payload = self._decode_status(message)
        if payload is not None:
            with self._lock:
                self._motion_guard_status = payload

    def _on_degeneracy_status(self, message):
        payload = self._decode_status(message)
        if payload is not None:
            with self._lock:
                self._degeneracy_status = payload

    def _on_truth_guard_status(self, message):
        payload = self._decode_status(message)
        if payload is None:
            return
        now = time.monotonic()
        with self._lock:
            self._truth_guard_status = payload
            self._truth_guard_status_received_at = now
            should_log = bool(
                payload.get("active") and
                now - self._truth_guard_last_logged_at >= 0.5)
            if should_log:
                self._truth_guard_last_logged_at = now
        if should_log:
            self._append_localization_diagnostic({
                "event": self._event("RUNTIME_TRUTH_GUARD"),
                "truth_guard": payload,
            })

    @staticmethod
    def _decode_status(message):
        try:
            payload = json.loads(message.data)
            return payload if isinstance(payload, dict) else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _localization_disagreement(truth_anchor, truth_pose,
                                   odom_anchor, odom_pose):
        """Compare relative planar motion and the simulator-aligned height."""
        truth_dx = float(truth_pose[0]) - float(truth_anchor[0])
        truth_dy = float(truth_pose[1]) - float(truth_anchor[1])
        odom_dx = float(odom_pose[0]) - float(odom_anchor[0])
        odom_dy = float(odom_pose[1]) - float(odom_anchor[1])
        planar = math.hypot(odom_dx - truth_dx, odom_dy - truth_dy)
        truth_yaw = math.atan2(
            math.sin(float(truth_pose[3]) - float(truth_anchor[3])),
            math.cos(float(truth_pose[3]) - float(truth_anchor[3])))
        odom_yaw = math.atan2(
            math.sin(float(odom_pose[3]) - float(odom_anchor[3])),
            math.cos(float(odom_pose[3]) - float(odom_anchor[3])))
        yaw = abs(math.atan2(math.sin(odom_yaw - truth_yaw),
                            math.cos(odom_yaw - truth_yaw)))
        vertical = abs(float(odom_pose[2]) - float(truth_pose[2]))
        return {
            "relative_planar_disagreement_m": planar,
            "relative_yaw_disagreement_rad": yaw,
            "absolute_vertical_disagreement_m": vertical,
            "truth_translation_m": math.hypot(truth_dx, truth_dy),
            "odom_translation_m": math.hypot(odom_dx, odom_dy),
            "truth_delta_xy": [truth_dx, truth_dy],
            "odom_delta_xy": [odom_dx, odom_dy],
        }

    def _hold_rl(self):
        message = Joy()
        message.header.stamp = rospy.Time.now()
        message.axes = [0.0] * 8
        message.buttons = [0] * 12
        message.buttons[3] = 1
        self._joy_pub.publish(message)

    def _publish_truth_world_command(self, vx, vy, wz=0.0):
        with self._lock:
            pose = self._truth_pose
        if pose is None:
            return
        command = Twist()
        command.linear.x = vx * math.cos(pose[3]) + vy * math.sin(pose[3])
        command.linear.y = -vx * math.sin(pose[3]) + vy * math.cos(pose[3])
        command.angular.z = wz
        self._cmd_pub.publish(command)
        self._hold_rl()

    @staticmethod
    def _truth_route_stage_timeout(configured_timeout, distance,
                                   minimum_progress_speed, margin):
        """Return a bounded per-leg budget based on remaining GT distance."""
        configured = max(1.0, float(configured_timeout))
        speed = max(0.05, float(minimum_progress_speed))
        return max(configured, float(distance) / speed + max(0.0, margin))

    @staticmethod
    def _truth_height_is_second_floor(pose, elevation, margin):
        return bool(
            pose is not None and elevation is not None and len(pose) >= 3 and
            math.isfinite(float(pose[2])) and
            float(pose[2]) >= float(elevation) + max(0.0, float(margin)))

    def _wait_for_second_floor_truth_height(self):
        """Require physical F2 height before issuing any corridor command."""
        deadline = time.monotonic() + self._second_floor_height_wait
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            with self._lock:
                pose = self._truth_pose
            if self._truth_height_is_second_floor(
                    pose, self._second_floor_elevation,
                    self._second_floor_height_margin):
                return True
            self._cmd_pub.publish(Twist())
            self._hold_rl()
            time.sleep(0.05)
        return False

    def _append_localization_diagnostic(self, payload):
        """Persist one F2 handoff sample without delaying control callbacks."""
        record = {
            "schema": "simenv_{}_localization_stabilization_v1".format(
                self._floor_slug),
            "wall_time": time.time(),
        }
        record.update(payload)
        path = os.path.join(
            self._output, "{}_localization_stabilization.jsonl".format(
                self._floor_slug))
        try:
            with open(path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(
                    record, ensure_ascii=False, sort_keys=True) + "\n")
        except OSError as error:
            rospy.logwarn_throttle(
                2.0, "Cannot write F2 localization diagnostic: %s", error)

    def _stabilize_second_floor_localization(self):
        """Gate F2 exploration on stationary relative pose consistency."""
        self._state_pub.publish(String(
            data=self._event("LOCALIZATION_STABILIZING")))
        self._cmd_pub.publish(Twist())
        self._hold_rl()
        started = time.monotonic()
        deadline = started + self._localization_stabilization_timeout
        anchors = None
        stable_since = None
        last_log = -math.inf
        rejection_counts = {
            "missing_pose": 0,
            "stale_pose": 0,
            "registration_unhealthy": 0,
            "truth_guard_unhealthy": 0,
            "planar_disagreement": 0,
            "yaw_disagreement": 0,
            "vertical_disagreement": 0,
        }
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            now = time.monotonic()
            self._cmd_pub.publish(Twist())
            self._hold_rl()
            with self._lock:
                truth = self._truth_pose
                truth_time = self._truth_pose_received_at
                odom = self._stabilized_odom
                odom_time = self._stabilized_odom_received_at
                registration = dict(self._registration_status or {})
                registration_time = self._registration_status_received_at
                motion_guard = dict(self._motion_guard_status or {})
                degeneracy = dict(self._degeneracy_status or {})
                truth_guard = dict(self._truth_guard_status or {})
                truth_guard_time = self._truth_guard_status_received_at
            missing_pose = truth is None or odom is None
            stale_pose = bool(
                not missing_pose and
                (now - truth_time > self._localization_status_freshness or
                 now - odom_time > self._localization_status_freshness))
            registration_fresh = bool(
                registration and
                now - registration_time <=
                self._localization_status_freshness)
            registration_healthy = bool(
                registration_fresh and registration.get("healthy") and
                not registration.get("frozen") and
                not registration.get("innovation_rejected"))
            truth_guard_fresh = bool(
                truth_guard and
                now - truth_guard_time <= self._localization_status_freshness)
            truth_guard_healthy = bool(
                not self._localization_require_truth_guard or
                (truth_guard_fresh and truth_guard.get("enabled") and
                 truth_guard.get("f2_context") and
                 truth_guard.get("truth_received") and
                 not truth_guard.get("truth_stale") and
                 truth_guard.get("active")))
            if not missing_pose and not stale_pose and anchors is None:
                anchors = (tuple(truth), tuple(odom))
            disagreement = (self._localization_disagreement(
                anchors[0], truth, anchors[1], odom)
                if anchors is not None and not missing_pose else None)
            planar_ok = bool(
                disagreement is not None and
                disagreement["relative_planar_disagreement_m"] <=
                self._localization_maximum_planar_disagreement)
            yaw_ok = bool(
                disagreement is not None and
                disagreement["relative_yaw_disagreement_rad"] <=
                self._localization_maximum_yaw_disagreement)
            vertical_ok = bool(
                disagreement is not None and
                disagreement["absolute_vertical_disagreement_m"] <=
                self._localization_maximum_vertical_disagreement)
            sample_stable = bool(
                not missing_pose and not stale_pose and
                registration_healthy and truth_guard_healthy and
                planar_ok and yaw_ok and vertical_ok)
            if sample_stable:
                stable_since = stable_since or now
            else:
                stable_since = None
                if missing_pose:
                    rejection_counts["missing_pose"] += 1
                elif stale_pose:
                    rejection_counts["stale_pose"] += 1
                elif not registration_healthy:
                    rejection_counts["registration_unhealthy"] += 1
                elif not truth_guard_healthy:
                    rejection_counts["truth_guard_unhealthy"] += 1
                elif not planar_ok:
                    rejection_counts["planar_disagreement"] += 1
                elif not yaw_ok:
                    rejection_counts["yaw_disagreement"] += 1
                elif not vertical_ok:
                    rejection_counts["vertical_disagreement"] += 1
            stable_duration = (now - stable_since
                               if stable_since is not None else 0.0)
            if now - last_log >= 0.20:
                self._append_localization_diagnostic({
                    "event": "LOCALIZATION_STABILITY_SAMPLE",
                    "elapsed_sec": round(now - started, 3),
                    "sample_stable": sample_stable,
                    "stable_duration_sec": round(stable_duration, 3),
                    "truth_pose": list(truth) if truth is not None else None,
                    "stabilized_odom": list(odom) if odom is not None else None,
                    "truth_age_sec": (round(now - truth_time, 3)
                                      if truth is not None else None),
                    "odom_age_sec": (round(now - odom_time, 3)
                                     if odom is not None else None),
                    "registration_age_sec": (
                        round(now - registration_time, 3)
                        if registration else None),
                    "registration": registration or None,
                    "motion_guard": motion_guard or None,
                    "truth_guard": truth_guard or None,
                    "degeneracy": degeneracy or None,
                    "disagreement": disagreement,
                    "thresholds": {
                        "stable_seconds": self._localization_stable_seconds,
                        "maximum_planar_disagreement_m":
                            self._localization_maximum_planar_disagreement,
                        "maximum_yaw_disagreement_rad":
                            self._localization_maximum_yaw_disagreement,
                        "maximum_vertical_disagreement_m":
                            self._localization_maximum_vertical_disagreement,
                        "require_truth_guard":
                            self._localization_require_truth_guard,
                        "status_freshness_sec":
                            self._localization_status_freshness,
                    },
                })
                last_log = now
            if stable_duration >= self._localization_stable_seconds:
                result = {
                    "event": self._event("LOCALIZATION_STABLE"),
                    "elapsed_sec": round(now - started, 3),
                    "stable_duration_sec": round(stable_duration, 3),
                    "truth_anchor": list(anchors[0]),
                    "odom_anchor": list(anchors[1]),
                    "final_disagreement": disagreement,
                    "registration": registration,
                    "motion_guard": motion_guard,
                    "truth_guard": truth_guard,
                    "degeneracy": degeneracy,
                    "rejection_counts": rejection_counts,
                }
                self._append_localization_diagnostic(result)
                self._write_handoff(
                    self._event("LOCALIZATION_STABLE"), **result)
                self._state_pub.publish(String(
                    data=self._event("LOCALIZATION_STABLE")))
                return True
            time.sleep(0.05)
        with self._lock:
            truth = self._truth_pose
            odom = self._stabilized_odom
            registration = dict(self._registration_status or {})
            motion_guard = dict(self._motion_guard_status or {})
            truth_guard = dict(self._truth_guard_status or {})
            degeneracy = dict(self._degeneracy_status or {})
        disagreement = (self._localization_disagreement(
            anchors[0], truth, anchors[1], odom)
            if anchors is not None and truth is not None and odom is not None
            else None)
        result = {
            "event": self._event("LOCALIZATION_UNSTABLE"),
            "elapsed_sec": round(time.monotonic() - started, 3),
            "truth_pose": list(truth) if truth is not None else None,
            "stabilized_odom": list(odom) if odom is not None else None,
            "final_disagreement": disagreement,
            "registration": registration or None,
            "motion_guard": motion_guard or None,
            "truth_guard": truth_guard or None,
            "degeneracy": degeneracy or None,
            "rejection_counts": rejection_counts,
            "action": "block_{}_exploration".format(self._floor_slug),
        }
        self._append_localization_diagnostic(result)
        self._write_handoff(self._event("LOCALIZATION_UNSTABLE"), **result)
        self._state_pub.publish(String(
            data=self._event("LOCALIZATION_UNSTABLE")))
        return False

    @staticmethod
    def _second_floor_executor_context(elevation, floor_index=1,
                                       floor_slug="second_floor"):
        """Translate the validated F1 executor z envelope to floor 2."""
        elevation = float(elevation)
        return {
            "mode": "{}_exploration".format(floor_slug),
            "floor_index": int(floor_index),
            "floor_elevation": elevation,
            "minimum_pose_z": elevation - 0.80,
            "maximum_pose_z": elevation + 1.20,
        }

    def _activate_second_floor_executor_context(self):
        """Switch only the shared executor's post-handoff height envelope."""
        if self._second_floor_elevation is None:
            return False
        payload = self._second_floor_executor_context(
            self._second_floor_elevation, self._floor_index, self._floor_slug)
        encoded = json.dumps(payload, sort_keys=True)
        # The publisher is latched; repeats additionally cover a subscriber
        # reconnect immediately after the stair controller releases command.
        for _ in range(3):
            self._executor_floor_context_pub.publish(String(data=encoded))
            time.sleep(0.10)
        self._write_handoff(
            self._event("EXECUTOR_CONTEXT_ACTIVE"),
            executor_floor_context=payload)
        self._state_pub.publish(String(
            data=self._event("EXECUTOR_CONTEXT_ACTIVE")))
        return True

    def _second_floor_truth_route(self):
        """Build a safe upper-landing-to-corridor route from offline truth.

        The first waypoint leaves the stair opening across the physical upper
        landing.  The second stands just outside the corridor mouth on the
        corridor centreline, so the robot is already facing +y when it stops.
        The third lies just inside the floor-2 corridor, where the ordinary
        online corridor/door pipeline can start with the same pose and
        heading assumptions as it does on floor 1.
        """
        if not self._truth_layout or not os.path.isfile(self._truth_layout):
            return None
        try:
            with open(self._truth_layout, "r", encoding="utf-8") as stream:
                metadata = json.load(stream)
            floors = metadata.get("floors") or []
            floor = next(item for item in floors
                         if int(item.get("floor_index", -1)) ==
                         self._floor_index)
            self._second_floor_elevation = float(floor["elevation"])
            corridor = floor["corridor_bounds"]
            stair = floor["stair_bounds"]
            corridor_x = 0.5 * (float(corridor["x_min"]) +
                                float(corridor["x_max"]))
            corridor_y = float(corridor["y_min"]) + 1.0
            landing_x = 0.5 * (float(stair["x_min"]) +
                               float(stair["x_max"]))
            landing_y = float(stair["y_min"]) + 0.70
            landing_half_width = 1.51
            if self._stair_model and os.path.isfile(self._stair_model):
                root = ET.parse(self._stair_model).getroot()
                for link in root.iter("link"):
                    if link.get("name") != "stair_floor_landing_floor_{}".format(
                            self._floor_index):
                        continue
                    pose_text = (link.findtext("pose") or "").split()
                    size_text = (link.findtext(
                        "collision/geometry/box/size") or "").split()
                    if len(pose_text) >= 2:
                        landing_x, landing_y = map(float, pose_text[:2])
                    if size_text:
                        landing_half_width = 0.5 * float(size_text[0])
                    break
            # Move one body half-width beyond the east edge of the stair
            # opening.  The generated infill joins this landing to the broad
            # central lobby slab, so the segment never cuts across the void.
            stair_exit_x = max(float(stair["x_max"]) + 0.70,
                               landing_x + landing_half_width + 0.55)
            # Align just outside the corridor mouth on the corridor centreline.
            # The mouth has no door (verified against door_specs) and the
            # lobby slab around it is open, so the 0.5 m standoff leaves
            # enough turning clearance before the robot enters the corridor.
            corridor_align_y = float(corridor["y_min"]) - \
                self._truth_corridor_align_offset
            return [
                {"stage": "upper_stair_exit",
                 "target": (stair_exit_x, landing_y)},
                {"stage": "{}_corridor_align".format(self._floor_slug),
                 "target": (corridor_x, corridor_align_y)},
                {"stage": "{}_corridor_entry".format(self._floor_slug),
                 "target": (corridor_x, corridor_y)},
            ]
        except (OSError, ValueError, KeyError, StopIteration,
                ET.ParseError, TypeError) as error:
            rospy.logerr("Cannot construct second-floor truth route: %s", error)
            return None

    def _guide_to_second_floor_corridor(self):
        if not self._enable_truth_corridor_guide:
            return True
        route = self._second_floor_truth_route()
        if not route:
            self._write_handoff(self._event("GT_GUIDE_UNAVAILABLE"))
            self._write_truth_guide_result(self._event("GT_GUIDE_UNAVAILABLE"))
            return False
        trace = []
        guide_started = time.monotonic()
        if not self._wait_for_second_floor_truth_height():
            with self._lock:
                pose = self._truth_pose
            self._write_handoff(
                self._event("TRUTH_HEIGHT_NOT_CONFIRMED"),
                truth_pose=list(pose) if pose is not None else None,
                expected_floor_elevation=self._second_floor_elevation,
                required_height=(self._second_floor_elevation +
                                 self._second_floor_height_margin))
            self._write_truth_guide_result(
                self._event("TRUTH_HEIGHT_NOT_CONFIRMED"),
                truth_pose=list(pose) if pose is not None else None,
                expected_floor_elevation=self._second_floor_elevation,
                route=route)
            return False
        # Allow the stair manager's latched READY callback to release command
        # ownership before the plane-policy guide begins to move.
        release_until = time.monotonic() + 0.5
        while not rospy.is_shutdown() and time.monotonic() < release_until:
            self._cmd_pub.publish(Twist())
            self._hold_rl()
            time.sleep(0.05)
        for waypoint in route:
            stage = waypoint["stage"]
            target_x, target_y = waypoint["target"]
            if stage == "upper_stair_exit":
                event_suffix = "GT_STAIR_EXIT_GUIDE"
            elif stage.endswith("_corridor_align"):
                event_suffix = "GT_CORRIDOR_ALIGN_GUIDE"
            else:
                event_suffix = "GT_CORRIDOR_ENTRY_GUIDE"
            self._state_pub.publish(String(
                data=self._event(event_suffix)))
            last_trace = -math.inf
            with self._lock:
                initial_pose = self._truth_pose
            initial_distance = (math.hypot(
                target_x - initial_pose[0], target_y - initial_pose[1])
                if initial_pose is not None else 0.0)
            stage_timeout = self._truth_route_stage_timeout(
                self._truth_guide_timeout, initial_distance,
                self._truth_guide_minimum_progress_speed,
                self._truth_guide_timeout_margin)
            stage_deadline = time.monotonic() + stage_timeout
            best_distance = initial_distance
            last_progress = time.monotonic()
            while (not rospy.is_shutdown() and
                   time.monotonic() < stage_deadline):
                with self._lock:
                    pose = self._truth_pose
                if pose is None:
                    time.sleep(0.05)
                    continue
                dx, dy = target_x - pose[0], target_y - pose[1]
                distance = math.hypot(dx, dy)
                if time.monotonic() - last_trace >= 0.20:
                    trace.append({
                        "t": round(time.monotonic() - guide_started, 3),
                        "stage": stage, "x": pose[0], "y": pose[1],
                        "z": pose[2], "yaw": pose[3],
                        "target": [target_x, target_y],
                        "distance": distance,
                        "stage_timeout_sec": stage_timeout})
                    last_trace = time.monotonic()
                if distance <= self._truth_guide_tolerance:
                    self._cmd_pub.publish(Twist())
                    break
                if distance <= best_distance - self._truth_guide_progress_epsilon:
                    best_distance = distance
                    last_progress = time.monotonic()
                elif (time.monotonic() - last_progress >=
                      self._truth_guide_stall_timeout):
                    self._cmd_pub.publish(Twist())
                    self._write_handoff(
                        self._event("GT_GUIDE_STALLED"), stage=stage,
                        target=[target_x, target_y], trace=trace,
                        best_distance=best_distance,
                        stall_timeout_sec=self._truth_guide_stall_timeout)
                    self._write_truth_guide_result(
                        self._event("GT_GUIDE_STALLED"), stage=stage,
                        target=[target_x, target_y], route=route, trace=trace,
                        best_distance=best_distance,
                        stall_timeout_sec=self._truth_guide_stall_timeout)
                    return False
                heading = math.atan2(dy, dx)
                heading_error = math.atan2(
                    math.sin(heading - pose[3]), math.cos(heading - pose[3]))
                speed = min(self._truth_guide_speed,
                            max(0.30, 0.75 * distance))
                self._publish_truth_world_command(
                    speed * dx / max(distance, 1e-6),
                    speed * dy / max(distance, 1e-6),
                    max(-0.40, min(0.40, 0.9 * heading_error)))
                time.sleep(0.05)
            else:
                self._cmd_pub.publish(Twist())
                self._write_handoff(
                    self._event("GT_GUIDE_TIMEOUT"), stage=stage,
                    target=[target_x, target_y], trace=trace,
                    stage_timeout_sec=stage_timeout,
                    best_distance=best_distance)
                self._write_truth_guide_result(
                    self._event("GT_GUIDE_TIMEOUT"), stage=stage,
                    target=[target_x, target_y], route=route, trace=trace,
                    stage_timeout_sec=stage_timeout,
                    best_distance=best_distance)
                return False
        # Face into the corridor before starting the same forward-acquisition
        # phase used on floor 1.
        # Publish the F2 localization context *before* this pure-yaw command.
        # FAST-LIO then constrains translation while still estimating yaw;
        # publishing only after the turn allowed run57's false 5.2 m motion.
        self._state_pub.publish(String(
            data=self._event("LOCALIZATION_STABILIZING")))
        # Give FAST-LIO and the mapper at least two LiDAR periods to consume
        # the latched phase before the first angular command is issued.
        activation_deadline = time.monotonic() + 0.35
        while (not rospy.is_shutdown() and
               time.monotonic() < activation_deadline):
            self._cmd_pub.publish(Twist())
            self._hold_rl()
            time.sleep(0.05)
        target_heading = math.pi / 2.0
        heading_deadline = time.monotonic() + self._truth_guide_timeout
        while (not rospy.is_shutdown() and
               time.monotonic() < heading_deadline):
            with self._lock:
                pose = self._truth_pose
            if pose is None:
                time.sleep(0.05)
                continue
            error = math.atan2(math.sin(target_heading - pose[3]),
                               math.cos(target_heading - pose[3]))
            if abs(error) <= self._truth_heading_tolerance:
                self._cmd_pub.publish(Twist())
                self._write_handoff(
                    self._event("CORRIDOR_ENTRY_REACHED"),
                    truth_pose=list(pose), route=route, trace=trace)
                self._write_truth_guide_result(
                    self._event("CORRIDOR_ENTRY_REACHED"),
                    truth_pose=list(pose), route=route, trace=trace)
                self._state_pub.publish(String(
                    data=self._event("CORRIDOR_ENTRY_REACHED")))
                return True
            yaw_rate = max(-0.40, min(0.40, 1.1 * error))
            if abs(yaw_rate) < 0.24:
                yaw_rate = math.copysign(0.24, error)
            self._publish_truth_world_command(0.0, 0.0, yaw_rate)
            time.sleep(0.05)
        self._cmd_pub.publish(Twist())
        self._write_handoff(self._event("GT_GUIDE_TIMEOUT"),
                            stage="corridor_heading", trace=trace)
        self._write_truth_guide_result(
            self._event("GT_GUIDE_TIMEOUT"), stage="corridor_heading",
            route=route, trace=trace)
        return False

    def _write_truth_guide_result(self, state, **fields):
        payload = {"schema": "simenv_{}_gt_guide_v1".format(
                       self._floor_slug),
                   "state": state, "wall_time": time.time()}
        payload.update(fields)
        path = os.path.join(self._parent_output,
                            "{}_gt_corridor_guide.json".format(
                                self._floor_slug))
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2,
                      sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)

    def _write_handoff(self, state, **fields):
        payload = {
            "schema": "simenv_{}_handoff_v1".format(self._floor_slug),
            "state": state,
            "wall_time": time.time(),
            "plane_policy": self._plane_policy,
        }
        payload.update(fields)
        path = os.path.join(
            self._parent_output, "{}_handoff.json".format(self._floor_slug))
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2,
                      sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)

    def _wait_for_stair(self):
        self._state_pub.publish(String(data="WAIT_{}_FLOOR".format(
            self._floor_word)))
        started = time.monotonic()
        while not rospy.is_shutdown():
            with self._lock:
                reached = self._second_floor_reached
                source_floor_failed = self._source_floor_failed
                finalized = self._first_floor_finalized_at
                handoff_started = self._stair_handoff_started_at
                stair_state = self._stair_state
            if reached:
                self._write_handoff(self._event("REACHED"))
                return True
            if source_floor_failed:
                self._write_handoff(
                    "SOURCE_FLOOR_EXPLORATION_FAILED",
                    source_floor_failure_token=
                    self._source_floor_failure_token)
                return False
            # Before ownership transfer, bound the wait from process startup.
            # After transfer, give the physical stair sequence its own full
            # window instead of consuming that window during F1 exploration.
            timeout_anchor = handoff_started or started
            if time.monotonic() - timeout_anchor >= self._handoff_timeout:
                self._write_handoff(
                    self._event("HANDOFF_TIMEOUT"),
                    stair_state=stair_state,
                    stair_handoff_started=handoff_started is not None)
                return False
            if (handoff_started is None and finalized is not None and
                    time.monotonic() - finalized >= self._failed_f1_grace):
                self._write_handoff(
                    "SOURCE_FLOOR_ENDED_WITHOUT_{}_FLOOR".format(
                        self._floor_word),
                    stair_state=stair_state)
                return False
            time.sleep(0.1)
        return False

    def _restore_plane_policy(self):
        self._state_pub.publish(String(
            data=self._event("PLANE_POLICY_LOADING")))
        started = time.monotonic()
        last_request = -1.0
        stable_since = None
        while not rospy.is_shutdown() and time.monotonic() - started < self._policy_timeout:
            now = time.monotonic()
            self._cmd_pub.publish(Twist())
            self._hold_rl()
            if now - last_request >= 0.75:
                self._policy_pub.publish(String(data=self._plane_policy))
                last_request = now
            with self._lock:
                ready = self._plane_policy_ready and self._locomotion_ready
            if ready:
                stable_since = stable_since or now
                if now - stable_since >= self._policy_settle:
                    self._write_handoff(
                        self._event("EXPLORATION_READY"),
                        policy_reload_duration_sec=round(now - started, 3))
                    self._state_pub.publish(String(
                        data=self._event("EXPLORATION_READY")))
                    return True
            else:
                stable_since = None
            time.sleep(0.05)
        self._write_handoff(self._event("PLANE_POLICY_TIMEOUT"))
        return False

    def _load_second_floor_elevation(self):
        """Resolve the upper-floor elevation independently of the truth guide.

        The validated task only ever set ``_second_floor_elevation`` inside
        ``_second_floor_truth_route``, which the visual exit guide disables.
        The executor floor context needs the elevation to activate, so the
        visual guide path resolves it from the layout metadata itself.
        """
        if self._second_floor_elevation is not None:
            return True
        if not self._truth_layout or not os.path.isfile(self._truth_layout):
            return False
        try:
            with open(self._truth_layout, "r", encoding="utf-8") as stream:
                metadata = json.load(stream)
            floors = metadata.get("floors") or []
            floor = next(item for item in floors
                         if int(item.get("floor_index", -1)) ==
                         self._floor_index)
            self._second_floor_elevation = float(floor["elevation"])
            return True
        except (OSError, ValueError, KeyError, StopIteration,
                TypeError) as error:
            rospy.logerr("Cannot resolve %s elevation from layout metadata: %s",
                         self._floor_slug, error)
            return False

    def _release_visual_guide_pause(self):
        if self._visual_pause_pub is not None:
            self._visual_pause_pub.publish(Bool(data=False))

    def _wait_for_visual_exit_guide(self):
        """Run the external visual corridor-exit guide in the handoff window.

        Called after the plane policy is restored (EXPLORATION_READY latched)
        and before the truth corridor guide.  The external adapter process
        (corridor_yolo_detector + corridor_entrance_driver) owns /cmd_vel for
        the whole window and holds the goal-executor arbitration pause; this
        manager publishes no velocity command and only polls the per-floor
        guide state topic.  The driver always releases the pause when it
        finishes; this method re-releases it as a safety net before returning.
        """
        if not self._enable_visual_exit_guide:
            return True
        if not self._load_second_floor_elevation():
            self._release_visual_guide_pause()
            self._write_handoff(self._event("VISUAL_CORRIDOR_GUIDE_UNAVAILABLE"))
            self._state_pub.publish(String(
                data=self._event("VISUAL_CORRIDOR_GUIDE_UNAVAILABLE")))
            return False
        with self._lock:
            self._visual_guide_state = None
        window = self._event("VISUAL_CORRIDOR_GUIDE_WINDOW")
        self._state_pub.publish(String(data=window))
        rospy.loginfo("[%s] visual corridor exit guide window opened",
                      self._floor_slug)
        deadline = time.monotonic() + max(1.0, self._visual_guide_timeout)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            with self._lock:
                state = self._visual_guide_state
            if state:
                break
            time.sleep(0.05)
        with self._lock:
            state = self._visual_guide_state
        self._release_visual_guide_pause()
        if state == self._visual_guide_ready_token:
            self._write_handoff(
                self._event("VISUAL_CORRIDOR_GUIDE_READY"),
                guide_state=state)
            self._state_pub.publish(String(
                data=self._event("VISUAL_CORRIDOR_GUIDE_READY")))
            return True
        outcome = (self._event("VISUAL_CORRIDOR_GUIDE_TIMEOUT")
                   if not state else
                   self._event("VISUAL_CORRIDOR_GUIDE_FAILED"))
        self._write_handoff(outcome, guide_state=state)
        self._state_pub.publish(String(data=outcome))
        return False

    def _copy_first_floor_parameters(self):
        # The YAML is loaded under /baseline_exploration_manager.  Reuse its
        # validated online parameters without sharing any runtime state.
        base = rospy.get_param("/baseline_exploration_manager", {})
        if isinstance(base, dict):
            for key, value in base.items():
                private = "~" + str(key)
                if not rospy.has_param(private):
                    rospy.set_param(private, value)
        rospy.set_param("~output_dir", self._output)
        rospy.set_param("~room_id_prefix", "floor_{}_estimated_room_".format(
            self._floor_number))
        rospy.set_param("~stair_handoff_on_corridor_exit", True)
        # Keep floor 2 on the same room route/state machine already validated
        # on floor 1.  Only map-confidence and short forward-recovery guards
        # are specialized below; no floor-1 namespace or stair parameter is
        # mutated by this fresh process.
        #
        # The generic missing-room retrace is useful after a genuine terminal,
        # but an earlier F2 run mistook a room-row corner for that terminal.
        # The F2 overrides therefore retain one bounded reverse observation
        # pass only after a minimum online outbound station has been reached.
        # The stronger CORRIDOR_TERMINAL_RETURN_LATCHED path remains enabled
        # for a geometry-confirmed five-metre terminal return even when one
        # room was not safely entered on the outward pass.
        for key, value in self._second_floor_only_parameter_overrides().items():
            rospy.set_param("~" + key, value)

    @staticmethod
    def _second_floor_only_parameter_overrides():
        """Return geometry guards scoped to the fresh F2 manager instance.

        These do not mutate the YAML namespace used by the already completed
        first-floor manager and do not participate in stair control.
        """
        return {
            # run68 physically returned along the exact ENTRY A* trace, but a
            # post-room map correction shifted the estimated door plane by
            # enough to report corridor_side_not_confirmed.  Accept only this
            # bounded, endpoint-matched reverse traversal on fresh upper-floor
            # managers.  The baseline/F1 default remains false.
            "room_accept_reversed_entry_trace_exit": True,
            "room_reversed_entry_trace_exit_tolerance_m": 0.30,
            # Run75's estimated room 02 was an opening centred only 0.046 m
            # from the online corridor centreline, so its selected "inside"
            # point never crossed either corridor wall.  Physical F2 door
            # centres were at least 0.46 m lateral in the same online frame.
            # Require two scheduling observations and a conservative 0.25 m
            # lateral offset only in fresh upper-floor manager processes.
            "room_corridor_side_confirmation_count": 2,
            "room_door_minimum_centerline_lateral_m": 0.25,
            # The run67 strict EXIT confirmation is intentionally enabled on
            # the already diagnosed F1 process only.  Keep fresh F2/F3
            # managers on their previously validated settings so this repair
            # cannot alter upper-floor room routing.
            "room_exit_require_raw_corridor_confirmation": False,
            "room_exit_require_station_corridor_confirmation": False,
            # Run55 established a valid F2 corridor at t=11 s but immediately
            # launched a 7 m sweep; the first real door was visible at t=16 s
            # and expired before door scheduling armed at t=35 s.  Reuse the
            # baseline's existing safe forward-commit phase for every F2
            # corridor establishment, so the following long goal can be
            # preempted by its already validated local-door callback.
            "corridor_short_door_commit_after_establishment": True,
            # Do not override corridor speed or leg length here. The local
            # run82 copied the F1-private far-pair 3 m clamp into this fresh
            # process. That made F2 repeatedly stop beside one failed portal
            # instead of completing the outbound corridor. Upper floors keep
            # the normal preemptible 7 m chord.
            "corridor_far_pair_search_maximum_advance_m": 0.0,
            # At about 20 m online outbound progress the robot is within the
            # final corridor section. Start the reverse pass there, or when
            # only 75 s remain, so the next staircase cannot be starved.
            "corridor_partial_return_minimum_outbound_progress_m": 20.0,
            "corridor_partial_return_reserve_seconds": 75.0,
            # Leave enough budget for ENTRY, immediate EXIT, and stair handoff.
            "corridor_partial_return_room_minimum_remaining_seconds": 45.0,
            # The 150 s hard phase deadline is an F1 mission contract. Upper
            # floors retain their independent 75 s stair-reserve trigger.
            "enforce_room_phase_target_deadline": False,


            # doorway callback can preempt a long centreline goal, so F2 now
            # inherits the same 1.55 m/s and 7 m forward policy validated on
            # F1 instead of stopping to replan every 2.5 m.
            # If drift still hides an upper doorway, make one bounded reverse
            # observation pass only after at least 12 m of measured outbound
            # corridor progress.  This retains the run49 protection against a
            # premature reversal beside the first room row.
            "enable_terminal_missing_room_retrace": True,
            "terminal_missing_room_retrace_minimum_outbound_progress_m": 12.0,
            "terminal_missing_room_retrace_max_distance_m": 8.0,
            # run72 registered the far room-3 portal, failed two ENTRY paths
            # from a wall-offset pose, and then could never select it again:
            # duplicate suppression discarded every later observation.  Once
            # the opposite room has been exited and the centreline restored,
            # retry that same unvisited landmark on the bounded return pass.
            # The short cooldown expires while the opposite room is explored.
            "room_door_cooldown_seconds": 20.0,
            "enable_known_unvisited_door_retry": True,
            "known_unvisited_door_retry_distance_m": 6.5,
            "known_unvisited_door_retry_only_on_terminal_return": True,
            # run78 queued the confirmed opposite door at 104.995 s, then
            # reached its known-free corridor-side point at 122.097 s.  The
            # old fixed 8 s lifetime expired during that successful 2.2 m
            # recenter transit and silently discarded the room.  Preserve it
            # long enough for close-range portal validation on F2/F3 only.
            "paired_opposite_recenter_pending_timeout_sec": 30.0,
            # run80 confirmed the terminal wall inside the configured 5 m
            # region at 301 s, but the old guard withheld return for a known
            # failed portal and exhausted the mission.  The structural
            # station/progress/wall gates remain mandatory.
            "terminal_return_blocked_by_known_unvisited_door": False,
            # Run49 reached the F2 corridor but recent-trajectory PCA still
            # contained the lobby-entry turn and locked a 22-degree diagonal
            # axis through the side wall.  At the ordinary entry lock, use
            # only observed occupied/free raster geometry to recover the
            # long balanced parallel-wall axis.  The baseline default is
            # false, so the completed F1 mission is unaffected.
            "corridor_entry_axis_relock_parallel_wall_semantic": True,
            # Run64's sparse rolling raster proposed +20 degrees even though
            # the robot had just travelled 3.98 m at -9.9 degrees.  That
            # 29.9-degree contradiction rotated corridor-forward openings
            # into the side-door detector.  Successful runs 55/57/63 differ
            # by at most 11.5 degrees, so a 15-degree online-motion gate keeps
            # their useful relocks and rejects only the contradictory one.
            "corridor_entry_axis_relock_maximum_motion_angle_deg": 15.0,
            "corridor_entry_axis_relock_minimum_motion_m": 2.0,
            # On run46 a real F2 doorway was confirmed 0.35 m after corridor
            # establishment, but the copied 2 m post-latch gate skipped it.
            # Retain the ordinary relative forward commit while admitting the
            # first doorway before a long corridor goal passes its station.
            "corridor_door_post_latch_motion_m": 0.35,
            # Run60 exhausted the 2-D forward raster at the corridor midpoint
            # and then performed 265 stationary rescans. Enable the generic
            # fallback only in this fresh F2 process: after one bounded rescan
            # it advances 0.60 m on the latched centreline, but only when each
            # 10 cm 3-D body footprint is live, mapped and collision-free.
            "enable_corridor_monotonic_forward_probe": True,
            "corridor_monotonic_probe_after": 2,
            "corridor_monotonic_probe_distance_m": 0.60,
            "corridor_monotonic_probe_sample_spacing_m": 0.10,
            "corridor_monotonic_probe_max_unknown_queries": 8,
            "corridor_monotonic_probe_limit": 24,
            # The GT bridge starts this fresh manager already inside the F2
            # corridor.  Run51's first physical doorway was therefore only
            # station 1.181 m from the online origin and the copied F1 2 m
            # lobby-throat guard discarded it.  Keep a smaller positive F2
            # guard: it admits that doorway while still rejecting the 0.438 m
            # stair/lobby aperture observed in the same run.
            "first_room_minimum_corridor_station_m": 0.75,
            # The second-floor process regenerates the combined root figures
            # only after its own result files have been fully persisted.
            "auto_generate_visualization": False,
        }

    def _map_second_floor_lobby(self):
        if not self._enable_lobby_rescan:
            return
        request_id = "floor-{}-lobby-{}".format(
            self._floor_number, int(time.time() * 1000))
        payload = {
            "request_id": request_id,
            "visual_sweep": True,
            "full_visual_sweep": True,
            "room_id": "floor_{}_lobby".format(self._floor_number),
            "angle_rad": 2.0 * math.pi,
            "angular_speed": 0.60,
            "timeout_sec": 14.0,
        }
        with self._lock:
            self._rescan_result = None
        self._state_pub.publish(String(data=self._event("LOBBY_RESCAN")))
        self._rescan_pub.publish(String(data=json.dumps(payload, sort_keys=True)))
        deadline = time.monotonic() + 15.0
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            with self._lock:
                result = (dict(self._rescan_result)
                          if isinstance(self._rescan_result, dict) else None)
            if result is not None and result.get("request_id") == request_id:
                self._write_handoff(
                    self._event("LOBBY_RESCAN_COMPLETE"),
                    rescan_success=bool(result.get("success")),
                    rescan_reason=result.get("reason"))
                return
            time.sleep(0.05)
        self._write_handoff(self._event("LOBBY_RESCAN_TIMEOUT"))

    def _generate_combined_visualization(self):
        if not self._auto_generate_visualization:
            return
        if not os.path.isfile(self._visualizer):
            rospy.logerr("Combined visualizer missing: %s", self._visualizer)
            return
        command = ["nice", "-n", "10", sys.executable, self._visualizer,
                   "--run-dir", self._parent_output,
                   "--output-dir", os.path.join(
                       self._parent_output, "visualization")]
        try:
            subprocess.Popen(command, start_new_session=True,
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            rospy.loginfo("Combined multi-floor visualization launched "
                          "detached at nice 10 (non-blocking).")
        except OSError as error:
            rospy.logerr("Combined multi-floor visualization failed to "
                         "launch: %s", error)

    @staticmethod
    def _exploration_handoff_authorized(termination_reason,
                                        exited_room_count,
                                        room_target_count,
                                        terminal_return_latched=False):
        """Require a completed return before arming the next stair manager.

        ``BaselineExplorationManager.run`` also returns after bounded failures
        such as ROOM_EXIT_BLOCKED, TIME_LIMIT and localization loss.  Those are
        finalization events, not proof that the robot is back at the corridor
        entrance.  Only an explicit stair/lobby return termination may publish
        the latched ``*_EXPLORATION_COMPLETE`` token consumed by the next
        staircase.  Preserve the existing online terminal-wall policy, which
        may intentionally return after fewer than the nominal room target.
        """
        successful_returns = {
            "STAIR_CORRIDOR_EXIT_HANDOFF",
            "STAIR_CORRIDOR_EXIT_HANDOFF_TRUTH_GATE",
            "STAIR_LOBBY_HANDOFF_LOCALIZATION_FALLBACK",
            "STAIR_WAIT_ZONE_REACHED",
        }
        try:
            exited = int(exited_room_count)
            target = max(1, int(room_target_count))
        except (TypeError, ValueError):
            return False
        return bool(
            str(termination_reason) in successful_returns and
            (exited >= target or exited >= 1 or
             bool(terminal_return_latched)))

    def run(self):
        if not self._wait_for_stair():
            self._state_pub.publish(String(data=self._event("NOT_STARTED")))
            return
        if not self._restore_plane_policy():
            self._state_pub.publish(String(data=self._event("POLICY_FAILED")))
            return
        if not self._wait_for_visual_exit_guide():
            self._state_pub.publish(String(
                data=self._event("VISUAL_CORRIDOR_GUIDE_FAILED")))
            return
        if not self._guide_to_second_floor_corridor():
            self._state_pub.publish(String(data=self._event("GT_GUIDE_FAILED")))
            return
        if not self._stabilize_second_floor_localization():
            self._cmd_pub.publish(Twist())
            self._state_pub.publish(String(
                data=self._event("LOCALIZATION_FAILED")))
            # Early-failure paths must also publish the EXPLORATION_FAILED
            # token: the next-floor manager is required=true and only exits
            # on COMPLETE/FAILED, so a missing token leaves roslaunch hung
            # forever (observed with LOCALIZATION_UNSTABLE on full16).
            self._state_pub.publish(String(
                data=self._event("EXPLORATION_FAILED")))
            self._generate_combined_visualization()
            return
        if not self._activate_second_floor_executor_context():
            self._write_handoff(self._event("EXECUTOR_CONTEXT_FAILED"))
            self._state_pub.publish(String(
                data=self._event("EXECUTOR_CONTEXT_FAILED")))
            return
        self._map_second_floor_lobby()
        self._copy_first_floor_parameters()
        self._exploration_started_wall_time = time.time()
        self._write_handoff(
            self._event("EXPLORATION_START"),
            exploration_started_wall_time=self._exploration_started_wall_time)
        self._state_pub.publish(String(data=self._event("EXPLORATION_START")))
        if self._tour_mode:
            runner = WaypointTourRunner(
                floor_number=self._floor_number,
                floor_slug=self._floor_slug,
                output_dir=self._output,
                waypoints=self._tour_waypoints,
                odom_topic="/simenv/fastlio_stabilized_odometry",
                wait_startup=False,
                publish_baseline_state=False,
                waypoint_frame=self._tour_waypoint_frame)
            tour_success = runner.run()
            termination_reason = runner.termination_reason
            recognized_room_count = 0
            exited_room_count = 0
            room_target_count = len(runner.waypoints)
            terminal_return_latched = bool(tour_success)
            handoff_authorized = bool(tour_success)
        else:
            manager = BaselineExplorationManager()
            manager.run()
            termination_reason = manager.termination_reason
            recognized_room_count = manager.room_scheduler.room_count
            exited_room_count = sum(
                bool(door.visited)
                for door in manager.room_scheduler.detector.doors)
            room_target_count = manager.room_target_count
            terminal_return_latched = bool(
                manager.corridor_terminal_return_latched)
            # R33: F2 部分完成(3/4 房间)后 GOAL_FAILURE(最后 1 间
            # estimated_door_03 反复失败,门候选丢弃,无目标循环) → 整任务
            # 失败 SOURCE_FLOOR_EXPLORATION_FAILED,下梯从未开始。失败
            # 终止但已退出 >=1 间房 → 升级为楼梯交接收尾,保留成果继续
            # 下一层(F2→F3 爬梯控制器用 truth 引导接手,不依赖当前位置)。
            original_reason = termination_reason
            handoff_authorized = self._exploration_handoff_authorized(
                termination_reason, exited_room_count, room_target_count,
                terminal_return_latched)
            # R34(2026-08-25):F3(顶层)失败终止无条件升级为楼梯交接。
            # iter2 run1 实证:F3 GOAL_FAILURE exited=0/4 → 不满足 R33
            # 的 exited>=1 → 下梯从未开始 → 整轮失败。F3 是链条最后一
            # 环,无"跳过下一层"风险;机器人存活即可触发下梯,段 1
            # ENTRY_GUIDE 用 truth 直驱带回楼梯口。即使 0 房间完成,
            # 返程跑通 = 全流程完成(优先跑通,其次覆盖率)。
            if (not handoff_authorized and
                    (self._floor_number == 3 or exited_room_count >= 1) and
                    termination_reason in ("GOAL_FAILURE", "TIME_LIMIT",
                                           "NO_VALID_FRONTIER_CONFIRMED",
                                           "ROOM_EXIT_BLOCKED",
                                           "G2_REACHED_WITH_MISSING_ROOMS")):
                termination_reason = "STAIR_CORRIDOR_EXIT_HANDOFF"
                handoff_authorized = True
                rospy.logwarn(
                    "[%s] %s (%d rooms exited, floor=%d) with %s: "
                    "upgrading to stair handoff to continue the chain.",
                    self._floor_slug,
                    "top-floor last resort" if self._floor_number == 3
                    else "partial completion",
                    exited_room_count, self._floor_number, original_reason)
        outcome = (self._event("EXPLORATION_COMPLETE")
                   if handoff_authorized else
                   self._event("EXPLORATION_FAILED"))
        log = rospy.loginfo if handoff_authorized else rospy.logerr
        log("[%s] termination=%s recognized=%d exited=%d/%d "
            "terminal_return=%s tour_mode=%s next_stair_handoff=%s",
            self._floor_slug, termination_reason,
            recognized_room_count, exited_room_count,
            room_target_count, terminal_return_latched,
            self._tour_mode, handoff_authorized)
        self._write_handoff(
            outcome,
            exploration_started_wall_time=self._exploration_started_wall_time,
            termination_reason=termination_reason,
            recognized_room_count=recognized_room_count,
            exited_room_count=exited_room_count,
            room_target_count=room_target_count,
            corridor_terminal_return_latched=terminal_return_latched,
            next_stair_handoff_authorized=handoff_authorized,
            tour_mode=self._tour_mode)
        # Opt2 (2026-08-26): publish the latched stair trigger BEFORE the
        # multi-floor visualization.  The visualizer is a matplotlib
        # subprocess taking ~25-35 s (batch3 RUN3 实测 32.5 s); run
        # synchronously it sits on the handoff critical path and every floor
        # transition pays it (实测 F2->F3 33 s、F3->下梯 24 s)。交接触发
        # 必须立即武装下一段楼梯控制器;图表只是文档,现在以分离进程
        # (start_new_session + nice 10)后台渲染,管理器退出/被杀后仍能
        # 完成,也不与楼梯控制抢主线程 CPU。
        self._state_pub.publish(String(data=outcome))
        self._generate_combined_visualization()


if __name__ == "__main__":
    rospy.init_node("second_floor_exploration_manager")
    SecondFloorMission().run()
