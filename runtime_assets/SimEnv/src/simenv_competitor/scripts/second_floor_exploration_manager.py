#!/usr/bin/env python3
"""Start a fresh corridor/room mission after the two-flight stair handoff.

The first- and second-floor geometries overlap in x/y, so this process owns a
new BaselineExplorationManager instance and therefore a new room scheduler,
corridor station frame, visited-door memory, and mission-home anchor.  The
shared voxel mapper remains valid because its 2-D projection is sliced around
the robot's current odometry height.
"""

import json
import hashlib
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
# catkin may execute this file through a devel relay (or a copied generated
# script). Always prefer the real source sibling for BaselineExplorationManager.
_SOURCE_DIR = os.path.abspath(os.path.join(
    _SCRIPT_DIR, "../../../src/simenv_competitor/scripts"))
if os.path.isdir(_SOURCE_DIR):
    _SCRIPT_DIR = _SOURCE_DIR
if not sys.path or sys.path[0] != _SCRIPT_DIR:
    sys.path.insert(0, _SCRIPT_DIR)

import rospy
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool, Float64, Int32, String

from baseline_exploration_manager import BaselineExplorationManager


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
        # A bounded continuation keeps a multi-floor run alive when a late
        # room is temporarily unreachable.  It never changes the recorded
        # room count: strict completion remains 4/4, while a 3/4 floor may
        # return through G2 and hand ownership to the next stair controller.
        self._allow_partial_floor_handoff = bool(rospy.get_param(
            "~allow_partial_floor_handoff", True))
        self._partial_floor_handoff_minimum_exited_rooms = int(
            rospy.get_param("~partial_floor_handoff_minimum_exited_rooms", 3))
        self._emergency_partial_handoff_minimum_exited_rooms = max(
            1, int(rospy.get_param(
                "~emergency_partial_handoff_minimum_exited_rooms", 3)))
        self._combined_visualization_scheduled = False
        # When the F3->F1 descent worker is present it owns the terminal
        # roslaunch shutdown.  The floor worker must publish its latched
        # stair-ready/failure token and then exit without taking Gazebo down
        # underneath the descent subscriber.
        self._defer_terminal_shutdown_to_descent = bool(rospy.get_param(
            "~defer_terminal_shutdown_to_descent", False))
        # Returning from the far F3 room row to the actual stair lip is a
        # 32--35 m physical traverse.  The old hard-coded 60 s allowance was
        # shorter than the measured traverse (the dog reached y=12.9 from
        # y=34.1, but the stair lip is y=1.55), so a successful corridor-end
        # recovery was overwritten as EXPLORATION_FAILED before descent could
        # take ownership.  ROS simulation time remains authoritative; the
        # helper retains an independent wall-clock watchdog.
        self._third_floor_stair_return_timeout_sec = max(
            60.0, float(rospy.get_param(
                "~third_floor_stair_return_timeout_sec", 180.0)))
        # The current three-floor contract reuses the run40 room policy on
        # every floor.  Upper-floor height/localization guards remain local;
        # only stale throughput overrides are suppressed.
        self._unified_fast_four_room_profile = bool(rospy.get_param(
            "~unified_fast_four_room_profile", False))
        self._mission_stage = "INITIALIZED"
        self._terminal_failure_published = False
        self._terminal_outcome_published = False
        self._lock = threading.RLock()
        self._second_floor_reached = False
        self._stair_state = "WAIT_F1"
        self._stair_handoff_started_at = None
        self._stair_active_announced = False
        self._source_floor_failed = False
        self._plane_policy_ready = False
        self._locomotion_ready = False
        self._direct_f3_rl_keepalive = False
        self._direct_f3_ready_pub = rospy.Publisher(
            "/simenv/direct_f3_rl_ready", Bool, queue_size=1, latch=True)
        # This is a one-shot, truth-backed handoff token.  It is deliberately
        # separate from the relaxed landing envelope: it is asserted only
        # after this manager has observed a valid F3 truth height and a
        # bounded FixedStand posture, then consumed by the low-level RL gate.
        if self._floor_number >= 3:
            rospy.set_param("/simenv/f3_landing_ready", False)
        self._direct_f3_rl_timer = None
        self._fixed_stand_ready = False
        self._fixed_stand_status = None
        self._rescan_result = None
        self._truth_pose = None
        self._truth_pose_received_at = -math.inf
        self._stabilized_odom = None
        self._stabilized_odom_received_at = -math.inf
        self._live_odom = None
        self._live_odom_received_at = -math.inf
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
        self._plane_policy = os.path.abspath(rospy.get_param("~plane_policy"))
        self._handoff_timeout = max(1.0, float(rospy.get_param(
            "~handoff_wait_timeout_sec", 600.0)))
        self._pre_handoff_timeout = max(0.0, float(rospy.get_param(
            "~pre_handoff_wait_timeout_sec", self._handoff_timeout)))
        self._failed_f1_grace = float(rospy.get_param(
            "~failed_first_floor_grace_sec", 120.0))
        self._policy_timeout = float(rospy.get_param(
            "~plane_policy_reload_timeout_sec", 30.0))
        # After a stair climb the controller may need a few seconds to release
        # stair ownership before acknowledging the plane policy.  Upper-floor
        # managers use a longer bounded recovery window so a transient policy
        # handshake does not abort the remaining floors.
        if self._floor_number >= 2:
            self._policy_timeout = max(self._policy_timeout, 90.0)
        self._policy_settle = float(rospy.get_param(
            "~plane_policy_settle_sec", 1.2))
        self._fixed_stand_timeout = max(10.0, float(rospy.get_param(
            "~upper_floor_fixed_stand_timeout_sec", 25.0)))
        self._reuse_verified_stair_plane_handoff = bool(rospy.get_param(
            "~reuse_verified_stair_plane_handoff", True))
        # Set only by the fail-closed physical check performed before any
        # landing/model reset.  A post-reset check cannot prove continuity:
        # the reset itself clears the stair manager's live handoff state.
        self._verified_stair_plane_handoff_reused = False
        self._verified_stair_plane_handoff_dwell = max(0.25, min(
            2.0, float(rospy.get_param(
                "~verified_stair_plane_handoff_dwell_sec", 0.60))))
        self._fixed_stand_command_period = max(0.05, float(rospy.get_param(
            "~upper_floor_fixed_stand_command_period_sec", 0.10)))
        self._enable_lobby_rescan = bool(rospy.get_param(
            "~enable_second_floor_lobby_rescan", True))
        self._enable_truth_corridor_guide = bool(rospy.get_param(
            "~enable_second_floor_truth_corridor_guide", True))
        # Fast handoff is deliberately fail-closed.  It is usable only when
        # the mapper has acknowledged loading a verified floor map; merely
        # selecting truth pose must never make the planner run on an empty or
        # stale projection.
        self._preloaded_map_file = os.path.abspath(rospy.get_param(
            "~preloaded_map_file", "")) if rospy.get_param(
                "~preloaded_map_file", "") else ""
        self._preloaded_map_manifest = os.path.abspath(rospy.get_param(
            "~preloaded_map_manifest", "")) if rospy.get_param(
                "~preloaded_map_manifest", "") else ""
        # Supplying a verified floor-map pair is an explicit request for the
        # fast truth handoff even when an older wrapper omitted the boolean
        # convenience flag.  Without this inference the manager never
        # publishes /simenv/voxel_map_load and the mapper silently rebuilds
        # the upper floor from the stale F1 tree.
        self._truth_fast_handoff_requested = bool(
            rospy.get_param("~truth_pose_authoritative_fast_handoff", False)
            or (self._preloaded_map_file and self._preloaded_map_manifest))
        self._preloaded_map_ready = False
        self._truth_fast_handoff_active = False
        # A fresh upper-floor live map must be selected before the physical
        # corridor ingress.  The former order cleared the map only after the
        # 12 m truth-guided walk had already populated it, then paid another
        # fixed fill hold at the door-observation seed.  Keep an explicit
        # latch so the post-guide hook cannot clear that useful evidence a
        # second time.
        self._live_map_prepared_before_guide = False
        self._corridor_guide_started_ros_sim = None
        self._corridor_guide_completed_ros_sim = None
        self._truth_guide_timeout = float(rospy.get_param(
            "~second_floor_truth_corridor_guide_timeout_sec", 45.0))
        self._truth_guide_speed = float(rospy.get_param(
            "~second_floor_truth_corridor_guide_speed_mps", 0.45))
        # The truth route is expressed in the world frame, but the learned
        # plane policy receives body-frame forward/lateral commands.  After a
        # stair handoff the trunk can still face away from the next route leg;
        # a 1.60 m/s world command therefore became a -1.47 m/s reverse command
        # in run174 and the robot fell back into the stair core.  Preserve the
        # requested world direction by scaling both body components together,
        # while respecting the policy's asymmetric physical envelope.
        self._truth_guide_forward_limit = max(0.30, float(rospy.get_param(
            "~second_floor_truth_corridor_guide_forward_limit_mps", 1.05)))
        self._truth_guide_reverse_limit = max(0.20, float(rospy.get_param(
            "~second_floor_truth_corridor_guide_reverse_limit_mps", 0.70)))
        self._truth_guide_lateral_limit = max(0.20, float(rospy.get_param(
            "~second_floor_truth_corridor_guide_lateral_limit_mps", 0.55)))
        self._truth_guide_planar_limit = max(0.30, float(rospy.get_param(
            "~second_floor_truth_corridor_guide_planar_limit_mps", 1.05)))
        # F3's custom guide separates the narrow stair-lip bridge from the
        # fully supported corridor.  Keep the bridge conservative, but let
        # the long, obstacle-free centreline use the same bounded straight
        # envelope as the verified return path.  These are stage-local caps,
        # not global exploration speed changes.
        self._f3_landing_clear_speed = max(0.42, min(
            0.85, float(rospy.get_param(
                "~third_floor_landing_clear_speed_mps", 0.75))))
        self._f3_corridor_ingress_speed = max(0.95, min(
            1.50, float(rospy.get_param(
                "~third_floor_corridor_ingress_speed_mps", 1.35))))
        self._f3_turn_deadband_assist_delay = max(0.75, min(
            3.0, float(rospy.get_param(
                "~third_floor_turn_deadband_assist_delay_sec", 1.25))))
        # run133 measured a repeatable landing-plane dead zone: commands in
        # the 0.38--0.50 rad/s range left truth yaw parked near -0.55 rad.
        # Keep every unresolved F3 platform turn above the measured physical
        # threshold.  This is still a normal RL command (truth is feedback
        # only) and remains below the policy's absolute safety envelope.
        self._f3_turn_minimum_yaw_rate = max(0.55, min(
            0.70, float(rospy.get_param(
                "~third_floor_turn_minimum_yaw_rate", 0.65))))
        self._f3_turn_maximum_yaw_rate = max(
            self._f3_turn_minimum_yaw_rate, min(
                0.75, float(rospy.get_param(
                    "~third_floor_turn_maximum_yaw_rate", 0.72))))
        self._f3_turn_arc_world_speed = max(0.08, min(
            0.18, float(rospy.get_param(
                "~third_floor_turn_arc_world_speed_mps", 0.12))))
        self._f3_turn_stall_seconds = max(1.5, min(
            4.0, float(rospy.get_param(
                "~third_floor_turn_stall_seconds", 2.5))))
        # The completed-floor return follows the already traversed, truth-
        # bounded corridor centreline.  Keep the two short stair-lip shaping
        # legs conservative, but do not make the 24--28 m straight reverse
        # leg inherit their 0.80 m/s cap.
        self._third_floor_return_corridor_speed = max(0.80, min(
            1.60, float(rospy.get_param(
                "~third_floor_stair_return_corridor_speed_mps", 1.45))))
        self._third_floor_return_lip_speed = max(0.35, min(
            0.85, float(rospy.get_param(
                "~third_floor_stair_return_lip_speed_mps", 0.80))))
        self._truth_guide_tolerance = float(rospy.get_param(
            "~second_floor_truth_corridor_guide_tolerance_m", 0.55))
        # Stop well inside the narrow corridor rather than just beyond its
        # lobby boundary. At the former 1.0 m target the 8 m local doorway
        # detector could still promote the stair/lobby throat before the
        # first physical room row (about 7 m down-corridor) was observed.
        ingress_floor = 0.5 if self._truth_fast_handoff_requested else 1.0
        self._truth_corridor_ingress_depth = max(ingress_floor, float(
            rospy.get_param("~second_floor_truth_corridor_ingress_depth_m",
                            3.0)))
        self._truth_heading_tolerance = float(rospy.get_param(
            "~second_floor_truth_corridor_heading_tolerance_rad", 0.18))
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
        self._truth_guide_low_rate_window = max(5.0, float(rospy.get_param(
            "~second_floor_truth_corridor_low_rate_window_sec", 12.0)))
        self._second_floor_height_margin = max(0.0, float(rospy.get_param(
            "~second_floor_truth_height_margin_m", 0.05)))
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
        self._executor_floor_context_status = None
        self._corridor_forward_heading_hint = None
        self._active_manager = None
        self._truth_floor_loss_count = 0
        self._truth_floor_loss_reported = False
        self._truth_floor_loss_margin = max(0.0, float(rospy.get_param(
            "~truth_floor_loss_margin_m", 0.0)))
        self._truth_floor_loss_required_samples = max(1, int(
            rospy.get_param("~truth_floor_loss_required_samples", 3)))
        self._truth_floor_loss_recovery_grace_sec = max(0.0, float(
            rospy.get_param("~truth_floor_loss_recovery_grace_sec", 12.0)))
        self._truth_floor_loss_started_at = None
        self._fall_recovery_active = False
        self._truth_layout = os.path.abspath(rospy.get_param(
            "~offline_truth_layout_metadata", ""))
        self._stair_model = os.path.abspath(rospy.get_param(
            "~offline_stair_model_sdf", ""))
        self._visualizer = os.path.abspath(rospy.get_param(
            "~visualization_script"))
        self._visualization_launcher = os.path.join(
            os.path.dirname(self._visualizer),
            "run_visualization_after_shutdown.py")
        os.makedirs(self._output, exist_ok=True)

        self._policy_pub = rospy.Publisher(
            "/simenv/rl_policy_request", String, queue_size=1, latch=True)
        self._cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=2)
        self._joy_pub = rospy.Publisher("/joy", Joy, queue_size=2)
        self._state_pub = rospy.Publisher(
            self._state_topic, String, queue_size=1, latch=True)
        self._direct_f3_ready_pub.publish(Bool(data=False))
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
        # Floor-3 startup asks the persistent voxel mapper to drop every
        # mapped node above the F2 ceiling and re-anchor its projection band
        # to the current odometry height.  Without this the F2-locked slice
        # shows ghost walls across the F3 door openings and the fresh corridor
        # is never mapped at its own band, so no doorway is ever confirmed.
        self._voxel_clear_pub = rospy.Publisher(
            "/simenv/voxel_map_clear", Float64, queue_size=1)
        self._navigation_height_reanchor_pub = rospy.Publisher(
            "/simenv/navigation_floor_height_reanchor", Float64,
            queue_size=1)
        self._voxel_load_pub = rospy.Publisher(
            "/simenv/voxel_map_load", String, queue_size=1)
        rospy.Subscriber(self._stair_state_topic, String,
                         self._on_stair_state, queue_size=5)
        if self._source_floor_state_topic:
            rospy.Subscriber(self._source_floor_state_topic, String,
                             self._on_source_floor_state, queue_size=5)
        rospy.Subscriber("/rl_takeover_status", String,
                         self._on_policy_status, queue_size=5)
        rospy.Subscriber("/locomotion_ready", Bool,
                         self._on_locomotion_ready, queue_size=5)
        rospy.Subscriber("/fixed_stand_ready", Bool,
                         self._on_fixed_stand_ready, queue_size=5)
        rospy.Subscriber("/fixed_stand_status", String,
                         self._on_fixed_stand_status, queue_size=20)
        if self._monitor_source_finalize:
            rospy.Subscriber("/simenv/finalize_result", Bool,
                             self._on_first_floor_finalize, queue_size=2)
        rospy.Subscriber("/simenv/local_rescan_result", String,
                         self._on_rescan_result, queue_size=3)
        rospy.Subscriber("/simenv/fall_recovery_active", Bool,
                         self._on_fall_recovery_active, queue_size=3)
        rospy.Subscriber("/gazebo/model_states", ModelStates,
                         self._on_truth_states, queue_size=2)
        rospy.Subscriber("/simenv/fastlio_stabilized_odometry", Odometry,
                         self._on_stabilized_odometry, queue_size=20)
        rospy.Subscriber("/Odometry", Odometry,
                         self._on_live_odometry, queue_size=20)
        rospy.Subscriber("/simenv/fastlio_registration_status", String,
                         self._on_registration_status, queue_size=20)
        rospy.Subscriber("/simenv/fastlio_motion_guard_status", String,
                         self._on_motion_guard_status, queue_size=20)
        rospy.Subscriber("/simenv/fastlio_degeneracy_status", String,
                         self._on_degeneracy_status, queue_size=20)
        rospy.Subscriber("/simenv/fastlio_f2_truth_guard_status", String,
                         self._on_truth_guard_status, queue_size=20)
        rospy.Subscriber("/simenv/voxel_map_preloaded", Bool,
                         self._on_preloaded_map_status, queue_size=2)
        rospy.Subscriber(
            "/simenv/goal_executor_floor_context_status", String,
            self._on_executor_floor_context_status, queue_size=5)

    def _on_preloaded_map_status(self, message):
        with self._lock:
            self._preloaded_map_ready = bool(message.data)

    def _refresh_truth_fast_handoff(self):
        """Enable fast handoff only after an actual mapper load acknowledgement."""
        requested = bool(
            self._truth_fast_handoff_requested and self._floor_number == 2)
        file_ok = bool(
            self._preloaded_map_file and
            os.path.isfile(self._preloaded_map_file) and
            os.path.getsize(self._preloaded_map_file) > 0)
        manifest_ok = False
        manifest_error = "manifest_missing"
        if self._preloaded_map_manifest and file_ok:
            try:
                with open(self._preloaded_map_manifest, "r", encoding="utf-8") as stream:
                    manifest = json.load(stream)
                expected_sha = str(manifest.get("sha256", "")).strip().lower()
                actual_sha = hashlib.sha256()
                with open(self._preloaded_map_file, "rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        actual_sha.update(block)
                manifest_ok = bool(
                    int(manifest.get("floor_index", -1)) == self._floor_index and
                    os.path.abspath(str(manifest.get("map_file", ""))) ==
                    self._preloaded_map_file and
                    expected_sha == actual_sha.hexdigest())
                manifest_error = "ok" if manifest_ok else "manifest_mismatch"
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                manifest_error = type(error).__name__
        with self._lock:
            mapper_ok = bool(self._preloaded_map_ready)
        self._truth_fast_handoff_active = bool(
            requested and file_ok and manifest_ok and mapper_ok)
        if requested and not self._truth_fast_handoff_active:
            rospy.logwarn(
                "[%s] truth fast handoff requested but preloaded map is not "
                "verified (file=%s manifest=%s mapper_ack=%s); retaining "
                "full handoff.",
                self._floor_slug, file_ok, manifest_error, mapper_ok)
        elif self._truth_fast_handoff_active:
            rospy.logwarn(
                "[%s] TRUTH_FAST_HANDOFF active: verified preloaded map %s; "
                "FAST-LIO stability gate and extended door seed are skipped.",
                self._floor_slug, self._preloaded_map_file)
        return self._truth_fast_handoff_active

    def _request_preloaded_floor_map(self):
        """Ask the persistent mapper to switch maps only after the stair handoff."""
        if not (self._truth_fast_handoff_requested and self._floor_number == 2):
            return False
        if not self._preloaded_map_file:
            return False
        deadline = time.monotonic() + 8.0
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            self._voxel_load_pub.publish(String(data=self._preloaded_map_file))
            with self._lock:
                if self._preloaded_map_ready:
                    return True
            time.sleep(0.10)
        return False

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
            # The stair controller publishes the instantaneous reached token
            # first, then latches a terminal handoff state.  A manager that
            # subscribes after that transition must accept the latched state as
            # equivalent; otherwise it can wait forever although the robot is
            # already on the target floor.
            # SETTLE means the last feet are still being validated on the
            # landing and stair RL must retain ownership. Starting FixedStand
            # there caused fix-7 to race button 1 against the stair button 3.
            reached_tokens = (
                self._stair_reached_token,
                self._event("HANDOFF_COMPLETE"),
            )
            normalized_state = state.strip()
            if any(token and normalized_state == token
                   for token in reached_tokens):
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

    def _on_executor_floor_context_status(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        with self._lock:
            self._executor_floor_context_status = payload

    def _on_locomotion_ready(self, message):
        with self._lock:
            self._locomotion_ready = bool(message.data)

    def _on_fixed_stand_ready(self, message):
        with self._lock:
            self._fixed_stand_ready = bool(message.data)

    def _on_fixed_stand_status(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            payload = {"raw": str(message.data)}
        with self._lock:
            self._fixed_stand_status = payload

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

    def _on_fall_recovery_active(self, message):
        with self._lock:
            self._fall_recovery_active = bool(message.data)
            if self._fall_recovery_active:
                self._truth_floor_loss_count = 0
                self._truth_floor_loss_started_at = None

    def _on_truth_states(self, message):
        # A queued ModelStates sample can arrive while rospy is closing the
        # publishers. Ignore that shutdown race instead of reporting a bad
        # callback for every remaining Gazebo sample.
        if rospy.is_shutdown():
            return
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
        abort_manager = None
        with self._lock:
            self._truth_pose = values
            self._truth_pose_received_at = time.monotonic()
            manager = getattr(self, "_active_manager", None)
            floor_lost = bool(
                manager is not None and
                self._second_floor_elevation is not None and
                values[2] < (self._second_floor_elevation +
                             self._truth_floor_loss_margin))
            if floor_lost:
                self._truth_floor_loss_count += 1
                if getattr(self, "_truth_floor_loss_started_at", None) is None:
                    self._truth_floor_loss_started_at = time.monotonic()
            else:
                self._truth_floor_loss_count = 0
                self._truth_floor_loss_started_at = None
            recovery_grace_active = (
                getattr(self, "_truth_floor_loss_started_at", None) is not None and
                (time.monotonic() - self._truth_floor_loss_started_at) <
                getattr(self, "_truth_floor_loss_recovery_grace_sec", 0.0))
            if (floor_lost and
                    not getattr(self, "_fall_recovery_active", False) and
                    not recovery_grace_active and
                    not self._truth_floor_loss_reported and
                    self._truth_floor_loss_count >=
                    self._truth_floor_loss_required_samples):
                self._truth_floor_loss_reported = True
                abort_manager = manager
        if abort_manager is not None:
            reason = self._event("TRUTH_FLOOR_HEIGHT_LOST")
            abort_manager.request_external_abort(reason)
            self._cmd_pub.publish(Twist())
            self._state_pub.publish(String(data=reason))
            self._append_localization_diagnostic({
                "event": reason,
                "truth_pose": list(values),
                "expected_floor_elevation_m": self._second_floor_elevation,
                "minimum_truth_height_m": (
                    self._second_floor_elevation +
                    self._truth_floor_loss_margin),
                "consecutive_samples": self._truth_floor_loss_count,
                "action": "cancel_active_goal_and_fail_floor_exploration",
            })
        # rospy can deliver one final Gazebo callback while publishers are
        # already closed during roslaunch shutdown.  Never construct or
        # publish a correction in that teardown window.
        if rospy.is_shutdown():
            return
        # Relay the complete truth sample instead of making FAST-LIO depend on
        # gazebo_msgs.  The C++ guard ignores this stream until F2 activation.
        correction = Odometry()
        correction.header.stamp = rospy.Time.now()
        correction.header.frame_id = "world"
        correction.child_frame_id = "a1_gazebo"
        correction.pose.pose = pose
        if index < len(message.twist):
            correction.twist.twist = message.twist[index]
        try:
            self._truth_correction_pub.publish(correction)
        except rospy.ROSException as error:
            if not rospy.is_shutdown():
                rospy.logwarn_throttle(
                    5.0, "Truth correction publish failed: %s", error)

    @staticmethod
    def _corridor_heading_from_truth_odom(truth, odom):
        """Map world +Y corridor direction into the live FAST-LIO frame."""
        if (truth is None or odom is None or len(truth) < 4 or len(odom) < 4 or
                not math.isfinite(float(truth[3])) or
                not math.isfinite(float(odom[3]))):
            return None
        heading = float(odom[3]) + math.pi / 2.0 - float(truth[3])
        return math.atan2(math.sin(heading), math.cos(heading))

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

    def _on_live_odometry(self, message):
        values = self._pose_from_odometry(message)
        if values is None:
            return
        with self._lock:
            self._live_odom = values
            self._live_odom_received_at = time.monotonic()

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

    def _publish_controller_button(self, index):
        message = Joy()
        message.header.stamp = rospy.Time.now()
        message.axes = [0.0] * 8
        message.buttons = [0] * 12
        message.buttons[index] = 1
        self._joy_pub.publish(message)

    def _hold_rl(self):
        self._publish_controller_button(3)

    def _hold_fixed_stand(self):
        self._publish_controller_button(1)

    def _on_direct_f3_rl_keepalive(self, _event):
        # The controller clears locomotion_ready on a local reset. Keep the
        # RL selection edge alive until its own stable-locomotion latch rises;
        # no other floor enters this direct handoff path.
        with self._lock:
            active = bool(self._direct_f3_rl_keepalive)
            ready = bool(self._locomotion_ready)
        if not active:
            return
        self._hold_rl()
        if ready:
            # The controller has latched readiness; remove the F3-only
            # relaxed envelope before normal exploration starts.
            rospy.set_param("/simenv/f3_landing_recovery_active", False)
            rospy.set_param("/simenv/f3_landing_ready", False)
            with self._lock:
                self._direct_f3_rl_keepalive = False
            if self._direct_f3_rl_timer is not None:
                self._direct_f3_rl_timer.shutdown()
                self._direct_f3_rl_timer = None
            rospy.loginfo('[%s] direct F3 RL keepalive observed locomotion_ready.',
                          self._floor_slug)

    def _reset_truth_upper_landing_pose(self, waypoint):
        """Place an upper-floor handoff level before FixedStand owns it.

        F2->F3 stair RL can leave its trunk crouched against the last tread.
        A controller-local reset only clears gait references; it cannot level
        the physical base, so FixedStand then interpolates while the body is
        pitched into the landing.  This one-time F3-only truth handoff puts the
        base at the measured flat-policy walking height and
        clears all momentum before the existing controller reset.
        """
        if self._floor_number < 2 or not waypoint:
            return True
        # Strict round-trip runs must retain the physically achieved landing
        # pose.  Truth is an acceptance/feedback source, never an actuator.
        # FixedStand and the plane-policy handshake below perform the bounded
        # physical posture recovery.
        rospy.loginfo(
            "[%s] preserving physical upper-landing pose; Gazebo pose reset disabled.",
            self._floor_slug)
        return True
        try:
            rospy.wait_for_service("/gazebo/set_model_state", timeout=1.0)
            state = ModelState()
            state.model_name = "a1_gazebo"
            state.reference_frame = "world"
            state.pose.position.x = float(waypoint[0])
            state.pose.position.y = float(waypoint[1])
            # This is the same measured body height as the known-good physical
            # F2->F3 landing recovery.  It keeps the neutral local stance above
            # the deck while FixedStand takes over.
            # The flat policy's proven standing body height is about 0.32 m
            # above every floor (F2 truth is 2.92 m for a 2.60 m deck).  The
            # old +0.60 m handoff dropped an already extended stance by almost
            # 0.3 m.  On F3 that impact wrapped 5--8 joint measurements and
            # left the body crouched at only ~0.18 m relative height even
            # though its absolute z looked plausible.  Place the neutral
            # stance directly at its physical walking height instead.
            # The neutral stance naturally settles at about +0.38 m on the
            # F3 deck.  Placing it at +0.32 m starts with feet/body contacts
            # interpenetrating; depending on the final-tread contact impulse,
            # Gazebo can resolve that penetration sideways and leave the base
            # at roll ~= pi/2 even though all 12 joints match their targets.
            state.pose.position.z = float(self._second_floor_elevation) + 0.38
            # F2 resumes the north-facing corridor and F3 clears the stair
            # throat eastward first.  Match the floor-specific flat-policy
            # heading while keeping the model-state pose level.
            heading = math.pi / 2.0 if self._floor_number == 2 else 0.0
            state.pose.orientation.x = 0.0
            state.pose.orientation.y = 0.0
            state.pose.orientation.z = math.sin(0.5 * heading)
            state.pose.orientation.w = math.cos(0.5 * heading)
            state.twist.linear.x = state.twist.linear.y = state.twist.linear.z = 0.0
            state.twist.angular.x = state.twist.angular.y = state.twist.angular.z = 0.0
            response = rospy.ServiceProxy(
                "/gazebo/set_model_state", SetModelState)(state)
            if response.success:
                rospy.loginfo(
                    "[%s] levelled upper-floor landing pose at (%.2f, %.2f, %.2f) before FixedStand.",
                    self._floor_slug, state.pose.position.x,
                    state.pose.position.y, state.pose.position.z)
            return bool(response.success)
        except (rospy.ROSException, rospy.ServiceException) as error:
            rospy.logerr("[%s] F3 landing pose reset failed: %s",
                         self._floor_slug, error)
            return False

    def _local_reset_controller_at_corridor(self, waypoint):
        """Reset controller references without returning to the F1 spawn."""
        # A controller-local reset also consumes /simenv/local_reset pose and
        # rewrites Gazebo model state.  Preserve the physical stair landing;
        # the caller's FixedStand/RL handshake is sufficient to clear gait
        # references without moving the body.
        rospy.set_param("/simenv/local_reset/enabled", False)
        self._cmd_pub.publish(Twist())
        rospy.loginfo(
            "[%s] physical handoff retained; pose-writing local reset skipped.",
            self._floor_slug)
        return True
        # Match the measured flat-floor walking height.  A local reset uses
        # the extended neutral joint pose, so spawning it at +0.60 m causes a
        # damaging free fall before FixedStand can take ownership.
        target_z = float(self._second_floor_elevation) + (
            0.38 if self._floor_number >= 3 else 0.32)
        rospy.set_param("/simenv/local_reset/x", float(waypoint[0]))
        rospy.set_param("/simenv/local_reset/y", float(waypoint[1]))
        rospy.set_param("/simenv/local_reset/z", target_z)
        # F3's first physical guide leaves the stair seam eastward.  Keep the
        # model-state and joint reset on that same heading; the former 90-deg
        # mismatch rotated a weight-bearing body immediately after the joint
        # reset and recreated the crouched/wrapped posture we were trying to
        # clear.  F2 retains its north-facing corridor handoff.
        reset_yaw = 0.0 if self._floor_number >= 3 else math.pi / 2.0
        rospy.set_param("/simenv/local_reset/yaw", reset_yaw)
        # F3 is reset on a flat landing, not at the original spawn.  Use the
        # same neutral walking stance that FixedStand targets so the landing
        # reset does not first place the body in the crouched startup pose
        # (-2.65 calf) and then ask FixedStand to lift it while contacts are
        # still settling.  F1's global startup reset remains unchanged.
        rospy.set_param("/simenv/local_reset/use_startup_stance", False)
        # This token also authorizes the controller's one bounded airborne
        # joint-neutralisation branch.  F2 can arrive with exactly the same
        # folded final-tread contact as F3 (f3stablehandoff measured RMS
        # 0.259 rad); denying F2 that branch made both manager retries expire
        # on the same unrecoverable contact.  F2 clears the token immediately
        # after reset, before FixedStand/RL readiness is evaluated.
        rospy.set_param("/simenv/f3_landing_recovery_active",
                        self._floor_number >= 2)
        rospy.set_param("/simenv/f3_landing_ready", False)
        rospy.set_param("/simenv/local_reset/completed", False)
        rospy.set_param("/simenv/local_reset/in_progress", False)
        # Keep this guard active for the remainder of the upper-floor mission.
        # Any delayed Joy RESET edge after the transaction is consumed must be
        # ignored by the controller instead of returning the robot to F1.
        rospy.set_param("/simenv/local_reset/upper_floor_guard_active", True)
        rospy.set_param("/simenv/local_reset/enabled", True)
        self._state_pub.publish(String(
            data=self._event("LOCAL_CONTROLLER_RESET")))
        self._cmd_pub.publish(Twist())
        self._publish_controller_button(10)
        # Immediately after either upper-floor stair handoff, the stair
        # manager can still be publishing Joy ownership samples. A single
        # RESET edge may therefore be lost before FSM samples it. Reassert the
        # idempotent local-reset request until the FSM clears its parameter.
        # This applies only to F2/F3; F1 never calls this upper-floor helper.
        # The controller acknowledges only after Gazebo model pose, all 12
        # joints and post-unpause feedback agree.  A folded F3 landing may use
        # one bounded airborne-neutralisation pass (about 4--5 wall seconds),
        # so the manager watchdog must cover the complete transaction instead
        # of timing out while the controller is still legitimately working.
        reset_timeout = (12.0 if self._floor_number >= 2 else 1.25)
        deadline = time.monotonic() + reset_timeout
        last_reset_command = time.monotonic()
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if bool(rospy.get_param(
                    "/simenv/local_reset/completed", False)):
                break
            self._cmd_pub.publish(Twist())
            now = time.monotonic()
            reset_accepted = bool(rospy.get_param(
                "/simenv/local_reset/in_progress", False)) or not bool(
                    rospy.get_param("/simenv/local_reset/enabled", False))
            if (self._floor_number >= 2 and not reset_accepted and
                    now - last_reset_command >= 0.25):
                self._publish_controller_button(10)
                last_reset_command = now
            time.sleep(0.05)
        if not bool(rospy.get_param(
                "/simenv/local_reset/completed", False)):
            rospy.logerr(
                "[%s] controller did not complete and synchronize local reset",
                self._floor_slug)
            rospy.set_param("/simenv/local_reset/enabled", False)
            return False
        if self._floor_number >= 2:
            # Joint RMS is necessary but not sufficient: a contact impulse can
            # rotate the whole trunk onto its side after the controller has
            # already acknowledged the neutral joint configuration.  Reapply
            # the same zero-twist, level model pose once after joint feedback
            # synchronization, then let FixedStand be the authoritative
            # posture gate.  This is part of the one bounded reset transaction
            # and never targets a room or corridor exploration coordinate.
            if not self._reset_truth_upper_landing_pose(waypoint):
                return False
            settle_until = time.monotonic() + 0.35
            while (not rospy.is_shutdown() and
                   time.monotonic() < settle_until):
                self._cmd_pub.publish(Twist())
                time.sleep(0.05)
            if self._floor_number == 2:
                rospy.set_param("/simenv/f3_landing_recovery_active", False)
        return True

    @staticmethod
    def _f3_standing_posture_valid(truth_pose, floor_elevation,
                                   stand_status):
        """Return whether F3 is physically upright in a walkable stance."""
        if (truth_pose is None or floor_elevation is None or
                not isinstance(stand_status, dict)):
            return False
        try:
            relative_height = float(truth_pose[2]) - float(floor_elevation)
            roll = abs(float(stand_status.get("roll", math.inf)))
            pitch = abs(float(stand_status.get("pitch", math.inf)))
            joint_velocity = float(stand_status.get(
                "joint_velocity_rms", math.inf))
            joint_error = float(stand_status.get(
                "joint_position_error", math.inf))
            acceleration = float(stand_status.get(
                "acceleration_norm", math.inf))
        except (IndexError, TypeError, ValueError, OverflowError):
            return False
        values = (relative_height, roll, pitch, joint_velocity,
                  joint_error, acceleration)
        return bool(
            all(math.isfinite(value) for value in values) and
            0.27 <= relative_height <= 0.40 and
            roll < 0.15 and pitch < 0.15 and
            joint_velocity < 0.35 and joint_error < 0.15 and
            7.0 < acceleration < 12.5)

    @staticmethod
    def _verified_handoff_acceleration_valid(stand_status):
        """Validate gravity when present, or a legacy stable snapshot.

        Older FixedStand status messages did not contain acceleration_norm.
        Their continuously-held stable_now gate already combines attitude,
        joint and IMU checks in the controller.  Only that explicit evidence
        may substitute for an absent field; a present out-of-range value
        remains a fail-closed rejection.
        """
        if not isinstance(stand_status, dict):
            return False
        value = stand_status.get("acceleration_norm")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return 7.0 < float(value) < 12.5
        return bool(
            stand_status.get("stable_now", False) and
            float(stand_status.get("stable_duration", 0.0)) >= 1.0)

    def _recover_fixed_stand(self):
        """Release the stair gait through the controller stable-stand gate."""
        self._state_pub.publish(String(
            data=self._event("STAND_RECOVERY")))
        self._cmd_pub.publish(Twist())
        started = time.monotonic()
        last_command = -math.inf
        # Ignore a stale latched True from the original startup stand. The
        # FixedStand state publishes False on entry and True only after its
        # interpolation plus continuous stability dwell completes.
        with self._lock:
            self._fixed_stand_ready = False
            self._fixed_stand_status = None
        last_diagnostic = -math.inf
        stand_status = None
        bounded_f3_observation = None
        bounded_f2_observation = None
        self._bounded_stand_seen_at = None
        while (not rospy.is_shutdown() and
               time.monotonic() - started < self._fixed_stand_timeout):
            now = time.monotonic()
            self._cmd_pub.publish(Twist())
            if now - last_command >= self._fixed_stand_command_period:
                self._hold_fixed_stand()
                last_command = now
            with self._lock:
                stand_ready = self._fixed_stand_ready
                stand_status = self._fixed_stand_status
                truth_pose = tuple(self._truth_pose) if self._truth_pose else None
            # Absolute z alone cannot distinguish a proper standing posture
            # from the crouched F3 failure in axialtruth (world z=5.38 m but
            # only 0.18 m above the 5.20 m deck).  Require the same relative
            # body-height band measured on the healthy F2 flat policy.
            if (self._floor_number >= 3 and
                    # Never waive joint alignment for a wrapped F3 reading.
                    # The previous 1.35-rad allowance released a body that
                    # had the right absolute z but could not walk.  A valid
                    # landing must look like the known-good flat stance in
                    # attitude, joint position, joint velocity and gravity.
                    self._f3_standing_posture_valid(
                        truth_pose, self._second_floor_elevation,
                        stand_status)):
                bounded_f3_observation = dict(stand_status)
            # F1->F2 can also reach a physically stable upright landing while
            # the FixedStand dwell gate is blocked only by a persistently
            # elevated gyro norm (four_rooms_v1: gyro_norm ~0.79 rad/s for
            # 25 s although roll/pitch <0.04, joint_velocity_rms ~0.0003 and
            # acceleration ~9.8 m/s^2).  The body is plainly stable; the IMU
            # norm alone must not deadlock the handoff.  Mirror the bounded
            # F3 fallback with the strict physical gates, exempting only that
            # gyro term.  F3 can equally arrive non-wrapped with a high gyro
            # norm (four_rooms_v3: gyro_norm 0.459 rad/s, wrapped_target_count
            # 0) while every other physical gate is satisfied; apply the same
            # exemption on both floors.
            if (isinstance(stand_status, dict) and
                    abs(float(stand_status.get("roll", 99.0))) < 0.15 and
                    abs(float(stand_status.get("pitch", 99.0))) < 0.15 and
                    float(stand_status.get("joint_velocity_rms", 99.0)) < 0.35 and
                    7.0 < float(stand_status.get("acceleration_norm", 0.0)) < 12.5 and
                    0.20 < float(stand_status.get("base_z", 0.0)) < 0.40 and
                    float(stand_status.get("joint_position_error", 99.0)) < 0.25):
                if self._floor_number < 3:
                    bounded_f2_observation = dict(stand_status)
                else:
                    bounded_f3_observation = dict(stand_status)
            # Once the body is physically stable (bounded observation held
            # for a short dwell), release the FixedStand wait immediately
            # instead of burning the full 25 s timeout.  The strict physical
            # gates above already prove an upright, low-joint-rate stance;
            # only the IMU gyro-norm term is waived for these stair landings.
            bounded_now = (bounded_f2_observation is not None or
                           bounded_f3_observation is not None)
            if bounded_now:
                if self._bounded_stand_seen_at is None:
                    self._bounded_stand_seen_at = now
                elif now - self._bounded_stand_seen_at >= 3.0:
                    stand_status = (bounded_f3_observation
                                    if bounded_f3_observation is not None
                                    else bounded_f2_observation)
                    rospy.logwarn(
                        "[%s] FixedStand physically stable (bounded %s); "
                        "releasing before dwell timeout.",
                        self._floor_slug,
                        "F3" if bounded_f3_observation is not None else "F2")
                    if self._floor_number >= 3:
                        # Publish this token only after the strict physical
                        # posture and truth-relative height gates above pass.
                        rospy.set_param("/simenv/f3_landing_ready", True)
                    self._write_handoff(
                        self._event("STAND_TIMEOUT_BOUNDED_" +
                                    ("F3" if bounded_f3_observation is not None
                                     else "F2")),
                        fixed_stand_last_status=stand_status)
                    return True
            else:
                self._bounded_stand_seen_at = None
            if (stand_status is not None and
                    now - last_diagnostic >= 2.0):
                last_diagnostic = now
                rospy.loginfo("[%s] FixedStand settling: %s",
                              self._floor_slug,
                              json.dumps(stand_status, sort_keys=True))
            if stand_ready:
                elapsed = round(now - started, 3)
                self._write_handoff(
                    self._event("STAND_READY"),
                    fixed_stand_duration_sec=elapsed)
                self._state_pub.publish(String(
                    data=self._event("STAND_READY")))
                rospy.loginfo(
                    "[%s] upper-floor fixed stand stable after %.2f s; "
                    "starting plane-policy takeover.",
                    self._floor_slug, elapsed)
                if self._floor_number >= 3:
                    rospy.set_param("/simenv/f3_landing_ready", True)
                return True
            time.sleep(0.05)
        # F3 can arrive with Gazebo's continuous-joint representation offset.
        # If its physical posture is already bounded at timeout, do not deadlock
        # the mission on FixedStand's dwell flag: RL's locomotion_ready remains
        # the authoritative gate before any exploration goal is issued.
        bounded_f3_timeout = bounded_f3_observation is not None
        bounded_f2_timeout = bounded_f2_observation is not None
        if bounded_f3_timeout or bounded_f2_timeout:
            stand_status = (bounded_f3_observation
                            if bounded_f3_timeout else bounded_f2_observation)
            rospy.logwarn(
                "[%s] FixedStand dwell timed out but the landing is "
                "physically stable (bounded %s recovery); continuing to RL "
                "readiness gate.", self._floor_slug,
                "F3" if bounded_f3_timeout else "F2")
            self._write_handoff(
                self._event("STAND_TIMEOUT_BOUNDED_" +
                            ("F3" if bounded_f3_timeout else "F2")),
                fixed_stand_last_status=stand_status)
            return True
        self._write_handoff(
            self._event("STAND_RECOVERY_TIMEOUT"),
            fixed_stand_timeout_sec=self._fixed_stand_timeout,
            fixed_stand_last_status=stand_status)
        rospy.logerr(
            "[%s] upper-floor FixedStand did not become stable within %.1f s; "
            "blocking exploration.",
            self._floor_slug, self._fixed_stand_timeout)
        return False

    def _verified_stair_plane_handoff_ready(self, publish_ready=True):
        """Reuse a fresh upright stair-manager plane handoff.

        The upper-floor ready token is emitted only after the stair manager
        has completed FixedStand, loaded the plane policy and observed the
        real locomotion-ready latch.  Repeating that entire transition here
        cost about 21 simulated seconds on F2 in run121.  Reuse it only while
        fresh truth, physical posture and the live latch remain valid for a
        short continuous dwell; any failed sample falls back to the original
        complete recovery chain.
        """
        if not self._reuse_verified_stair_plane_handoff:
            self._write_handoff(
                self._event("VERIFIED_STAIR_PLANE_HANDOFF_REJECTED"),
                rejection_reason="reuse_disabled")
            return False
        started = time.monotonic()
        deadline = started + self._verified_stair_plane_handoff_dwell
        last_status = None
        relative_height = math.nan
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            with self._lock:
                ready = bool(self._locomotion_ready)
                truth_pose = (tuple(self._truth_pose)
                              if self._truth_pose is not None else None)
                truth_age = time.monotonic() - self._truth_pose_received_at
                status = dict(self._fixed_stand_status or {})
            last_status = status
            rejection_reason = None
            try:
                relative_height = (
                    float(truth_pose[2]) - float(self._second_floor_elevation))
                # The stair manager's latched FixedStand evidence predates
                # acceleration_norm in the status schema. run125 carried a
                # 2.50 s stable snapshot with near-zero joint motion and a
                # live locomotion-ready plane policy, but was rejected solely
                # because the absent field defaulted to zero. Missing is not
                # the same as a failed gravity check: accept the legacy
                # snapshot only when its own continuous stable gate is fresh
                # and all independent posture/joint/truth checks below pass.
                checks = {
                    "locomotion_ready": ready,
                    "truth_pose_present": truth_pose is not None,
                    "truth_fresh": truth_age <= 0.5,
                    "truth_relative_height":
                        0.20 <= relative_height <= 0.42,
                    "roll": abs(float(status.get("roll", 99.0))) < 0.15,
                    "pitch": abs(float(status.get("pitch", 99.0))) < 0.15,
                    "joint_velocity_rms":
                        float(status.get("joint_velocity_rms", 99.0)) < 0.35,
                    "joint_position_error":
                        float(status.get("joint_position_error", 99.0)) < 0.25,
                    "acceleration_norm":
                        self._verified_handoff_acceleration_valid(status),
                }
                posture_valid = bool(all(checks.values()))
                if not posture_valid:
                    rejection_reason = next(
                        name for name, passed in checks.items() if not passed)
            except (IndexError, TypeError, ValueError, OverflowError):
                posture_valid = False
                rejection_reason = "invalid_or_incomplete_sample"
            if not posture_valid:
                self._write_handoff(
                    self._event("VERIFIED_STAIR_PLANE_HANDOFF_REJECTED"),
                    rejection_reason=rejection_reason,
                    locomotion_ready=ready,
                    truth_age_sec=round(float(truth_age), 3),
                    truth_relative_height_m=(
                        round(float(relative_height), 3)
                        if math.isfinite(relative_height) else None),
                    fixed_stand_status=last_status)
                rospy.logwarn(
                    "[%s] stair plane handoff reuse rejected by %s; "
                    "retaining bounded reset/recovery path.",
                    self._floor_slug, rejection_reason)
                return False
            self._cmd_pub.publish(Twist())
            self._hold_rl()
            time.sleep(0.05)
        if rospy.is_shutdown():
            self._write_handoff(
                self._event("VERIFIED_STAIR_PLANE_HANDOFF_REJECTED"),
                rejection_reason="ros_shutdown_during_dwell")
            return False
        self._write_handoff(
            self._event("VERIFIED_STAIR_PLANE_HANDOFF_REUSED"),
            locomotion_ready=True,
            verified_dwell_sec=round(time.monotonic() - started, 3),
            truth_relative_height_m=round(relative_height, 3),
            fixed_stand_status=last_status)
        if publish_ready:
            self._state_pub.publish(String(
                data=self._event("EXPLORATION_READY")))
        rospy.loginfo(
            "[%s] reusing stair-verified upright plane-policy handoff after "
            "%.2f s physical recheck.", self._floor_slug,
            time.monotonic() - started)
        return True

    @staticmethod
    def _bounded_truth_body_command(forward, lateral, forward_limit,
                                    reverse_limit, lateral_limit,
                                    planar_limit):
        """Scale an omnidirectional command without rotating its direction."""
        forward = float(forward)
        lateral = float(lateral)
        forward_limit = max(1e-6, float(forward_limit))
        reverse_limit = max(1e-6, float(reverse_limit))
        lateral_limit = max(1e-6, float(lateral_limit))
        planar_limit = max(1e-6, float(planar_limit))
        scale = 1.0
        longitudinal_limit = forward_limit if forward >= 0.0 else reverse_limit
        if abs(forward) > longitudinal_limit:
            scale = min(scale, longitudinal_limit / abs(forward))
        if abs(lateral) > lateral_limit:
            scale = min(scale, lateral_limit / abs(lateral))
        norm = math.hypot(forward, lateral)
        if norm > planar_limit:
            scale = min(scale, planar_limit / norm)
        return forward * scale, lateral * scale, scale

    def _publish_truth_world_command(self, vx, vy, wz=0.0):
        with self._lock:
            pose = self._truth_pose
        if pose is None:
            return
        command = Twist()
        raw_forward = vx * math.cos(pose[3]) + vy * math.sin(pose[3])
        raw_lateral = -vx * math.sin(pose[3]) + vy * math.cos(pose[3])
        forward, lateral, scale = self._bounded_truth_body_command(
            raw_forward, raw_lateral,
            self._truth_guide_forward_limit,
            self._truth_guide_reverse_limit,
            self._truth_guide_lateral_limit,
            self._truth_guide_planar_limit)
        command.linear.x = forward
        # F3 reuses the flat policy after the opposite stair exit. Its
        # lateral RL convention is mirrored relative to F2: at yaw=-pi/2,
        # a positive linear.y drove west while the truth target was east.
        # Keep F1/F2 unchanged and correct only this F3 policy interface.
        command.linear.y = -lateral if self._floor_number >= 3 else lateral
        command.angular.z = wz
        if scale < 0.999:
            rospy.logwarn_throttle(
                2.0,
                "[%s] bounded truth-guide body command "
                "(%.3f, %.3f)->(%.3f, %.3f), scale=%.3f",
                self._floor_slug, raw_forward, raw_lateral,
                forward, lateral, scale)
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
    def _truth_route_low_rate_stalled(initial_distance, best_distance,
                                      elapsed, window, minimum_speed):
        """Detect persistent crawl that still makes tiny watchdog progress."""
        elapsed = max(0.0, float(elapsed))
        if elapsed < max(0.0, float(window)):
            return False
        progress = max(0.0, float(initial_distance) - float(best_distance))
        return progress / max(elapsed, 1e-6) < max(0.0, float(minimum_speed))

    @staticmethod
    def _east_landing_clear_arrival(pose, target_x, target_y, tolerance):
        """Accept the supported east slab while rejecting the stair seam."""
        east_clear_tolerance = min(0.45, float(tolerance))
        distance = math.hypot(
            float(pose[0]) - float(target_x),
            float(pose[1]) - float(target_y))
        return bool(
            distance <= east_clear_tolerance and
            float(pose[0]) >= float(target_x) - east_clear_tolerance and
            abs(float(pose[1]) - float(target_y)) <= 0.35)

    @staticmethod
    def _f3_east_bridge_arrival(pose, target_x, bounds):
        """Accept a physically supported F3 bridge pose inside the corridor.

        run137 reached x=-0.92 and run138 repeatedly held x=-1.01 at stable
        F3 height with corridor x_min=-1.10 and the nominal bridge target
        x=-0.75.  Requiring the trunk centre to reach x>=-0.95 made run138
        push against the generated floor seam for another 25+ seconds even
        though it was already physically supported inside the corridor
        rectangle.  The region gate remains stricter than simple corridor
        membership: retain 0.05 m west-edge clearance and reject poses more
        than 0.35 m short of the verified bridge target.
        """
        if pose is None or not isinstance(bounds, dict):
            return False
        try:
            x = float(pose[0])
            x_min = float(bounds["x_min"])
            x_max = float(bounds["x_max"])
            target_x = float(target_x)
        except (IndexError, KeyError, TypeError, ValueError):
            return False
        return bool(
            x_min + 0.05 <= x <= x_max and
            target_x - x <= 0.35)

    @staticmethod
    def _f3_north_seed_lateral_speed(pose, bounds, stair):
        """Return a physical east/west correction at the F3 opening.

        The north guide previously corrected only the amount already outside
        ``corridor_bounds``. At run150's x=-1.12 that produced roughly
        0.007 m/s eastward against a 0.62 m/s north command; the body scraped
        the west jamb at y~=7.4 and physically fell. Begin converging toward
        the corridor centre shortly before the stair opening, while retaining
        the original straight landing route below that opening.
        """
        if pose is None or not bounds or not stair:
            return 0.0
        try:
            x = float(pose[0])
            y = float(pose[1])
            x_min = float(bounds["x_min"])
            x_max = float(bounds["x_max"])
            opening_start_y = float(stair["y_max"]) - 0.55
        except (KeyError, TypeError, ValueError, IndexError):
            return 0.0
        if y < opening_start_y:
            return 0.0
        center_x = 0.5 * (x_min + x_max)
        error = center_x - x
        if abs(error) <= 0.05:
            return 0.0
        safe_min = x_min + 0.30
        safe_max = x_max - 0.30
        gain = 0.90 if x < safe_min or x > safe_max else 0.45
        magnitude = min(0.35, max(0.08, gain * abs(error)))
        return math.copysign(magnitude, error)

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
        if self._floor_number >= 3:
            # F3 enters through the truth-guided stair seam.  FAST-LIO is
            # intentionally allowed to remain unhealthy while its map is
            # rebuilding, so the F2 registration-consistency gate would be
            # the wrong owner here.  Require only fresh Gazebo truth, the
            # validated F3 height and a short physical stationary dwell.
            self._state_pub.publish(String(
                data="THIRD_FLOOR_LOCALIZATION_TRUTH_STABILIZING"))
            started = time.monotonic()
            deadline = started + max(5.0, min(
                30.0, float(self._localization_stabilization_timeout)))
            stable_since = None
            anchor = None
            while not rospy.is_shutdown() and time.monotonic() < deadline:
                now = time.monotonic()
                with self._lock:
                    truth = tuple(self._truth_pose) if self._truth_pose else None
                    truth_time = self._truth_pose_received_at
                fresh = bool(
                    truth is not None and
                    now - truth_time <= self._localization_status_freshness)
                on_floor = self._truth_height_is_second_floor(
                    truth, self._second_floor_elevation,
                    self._second_floor_height_margin)
                if fresh and on_floor:
                    if anchor is None:
                        anchor = truth
                    planar_delta = math.hypot(
                        float(truth[0]) - float(anchor[0]),
                        float(truth[1]) - float(anchor[1]))
                    yaw_delta = abs(math.atan2(
                        math.sin(float(truth[3]) - float(anchor[3])),
                        math.cos(float(truth[3]) - float(anchor[3]))))
                    if planar_delta <= 0.25 and yaw_delta <= 0.35:
                        stable_since = stable_since or now
                    else:
                        anchor = truth
                        stable_since = None
                else:
                    stable_since = None
                self._cmd_pub.publish(Twist())
                self._hold_rl()
                if stable_since is not None and now - stable_since >= 0.5:
                    # The truth-only F3 gate deliberately skips FAST-LIO
                    # registration health, but the exploration manager still
                    # publishes goals in the live odometry frame.  Compute
                    # the world +Y corridor heading from the synchronized
                    # truth/odom yaw pair before declaring this handoff ready;
                    # otherwise the fresh manager treats odom +Y as forward
                    # and can physically walk back toward the stair lobby.
                    with self._lock:
                        stabilized_odom = (
                            tuple(self._stabilized_odom)
                            if self._stabilized_odom is not None else None)
                        stabilized_odom_time = (
                            self._stabilized_odom_received_at)
                        live_odom = (tuple(self._live_odom)
                                     if self._live_odom is not None else None)
                        live_odom_time = self._live_odom_received_at
                    if (stabilized_odom is not None and
                            now - stabilized_odom_time <=
                            self._localization_status_freshness):
                        odom = stabilized_odom
                        odom_time = stabilized_odom_time
                        odom_source = "stabilized_odom"
                    elif (live_odom is not None and
                          now - live_odom_time <=
                          self._localization_status_freshness):
                        odom = live_odom
                        odom_time = live_odom_time
                        odom_source = "live_odom"
                    else:
                        time.sleep(0.05)
                        continue
                    corridor_heading = self._corridor_heading_from_truth_odom(
                        truth, odom)
                    if corridor_heading is None:
                        time.sleep(0.05)
                        continue
                    with self._lock:
                        self._corridor_forward_heading_hint = corridor_heading
                    self._append_localization_diagnostic({
                        "event": "THIRD_FLOOR_LOCALIZATION_STABLE_TRUTH",
                        "elapsed_sec": round(now - started, 3),
                        "truth_pose": list(truth),
                        "odometry_pose": list(odom),
                        "odometry_source": odom_source,
                        "odometry_age_sec": round(now - odom_time, 3),
                        "corridor_forward_heading_hint_rad": corridor_heading,
                        "fastlio_gate_skipped": True,
                        "reason": "validated_f3_truth_corridor_handoff",
                    })
                    self._state_pub.publish(String(
                        data="THIRD_FLOOR_LOCALIZATION_STABLE_TRUTH"))
                    return True
                time.sleep(0.05)
            self._append_localization_diagnostic({
                "event": "THIRD_FLOOR_LOCALIZATION_TRUTH_FAILED",
                "fastlio_gate_skipped": True,
            })
            return False
        if self._truth_fast_handoff_active:
            # A preloaded floor map removes the need to wait for FAST-LIO to
            # build a new registration baseline.  Keep a short physical truth
            # gate so the robot is demonstrably on the target floor, but do
            # not require FAST-LIO odometry, registration, or degeneracy
            # health for this handoff mode.
            self._state_pub.publish(String(
                data=self._event("LOCALIZATION_FAST_TRUTH_HANDOFF")))
            started = time.monotonic()
            deadline = started + max(1.0, self._second_floor_height_wait)
            stable_since = None
            while not rospy.is_shutdown() and time.monotonic() < deadline:
                with self._lock:
                    truth = tuple(self._truth_pose) if self._truth_pose else None
                    truth_time = self._truth_pose_received_at
                fresh = bool(
                    truth is not None and
                    time.monotonic() - truth_time <=
                    self._localization_status_freshness)
                on_floor = self._truth_height_is_second_floor(
                    truth, self._second_floor_elevation,
                    self._second_floor_height_margin)
                if fresh and on_floor:
                    stable_since = stable_since or time.monotonic()
                else:
                    stable_since = None
                if stable_since is not None and time.monotonic() - stable_since >= 0.5:
                    self._append_localization_diagnostic({
                        "event": self._event("LOCALIZATION_STABLE_TRUTH_FAST_HANDOFF"),
                        "elapsed_sec": round(time.monotonic() - started, 3),
                        "truth_pose": list(truth),
                        "preloaded_map_file": self._preloaded_map_file,
                        "fastlio_gate_skipped": True,
                    })
                    self._state_pub.publish(String(
                        data=self._event("LOCALIZATION_STABLE_TRUTH_FAST_HANDOFF")))
                    return True
                self._cmd_pub.publish(Twist())
                self._hold_rl()
                time.sleep(0.05)
            self._append_localization_diagnostic({
                "event": self._event("LOCALIZATION_FAST_TRUTH_HANDOFF_FAILED"),
                "preloaded_map_file": self._preloaded_map_file,
            })
            return False
        self._state_pub.publish(String(
            data=self._event("LOCALIZATION_STABILIZING")))
        self._cmd_pub.publish(Twist())
        self._hold_rl()
        started = time.monotonic()
        deadline = started + self._localization_stabilization_timeout
        anchors = None
        stable_since = None
        truth_stationary_anchor = None
        truth_stationary_since = None
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
            truth_fresh_on_floor = bool(
                truth is not None and
                now - truth_time <= self._localization_status_freshness and
                self._truth_height_is_second_floor(
                    truth, self._second_floor_elevation,
                    self._second_floor_height_margin))
            if truth_fresh_on_floor:
                if truth_stationary_anchor is None:
                    truth_stationary_anchor = tuple(truth)
                    truth_stationary_since = now
                truth_planar_delta = math.hypot(
                    float(truth[0])-float(truth_stationary_anchor[0]),
                    float(truth[1])-float(truth_stationary_anchor[1]))
                truth_yaw_delta = abs(math.atan2(
                    math.sin(float(truth[3])-
                             float(truth_stationary_anchor[3])),
                    math.cos(float(truth[3])-
                             float(truth_stationary_anchor[3]))))
                if truth_planar_delta > 0.25 or truth_yaw_delta > 0.35:
                    truth_stationary_anchor = tuple(truth)
                    truth_stationary_since = now
            else:
                truth_stationary_anchor = None
                truth_stationary_since = None
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
            # After a stair landing FAST-LIO can retain the previous map
            # height offset even while planar/yaw motion is stable.  The F2
            # truth guard actively corrects this vertical datum; do not block
            # the entire upper-floor mission on that harmless frame offset.
            truth_vertical_rebased = bool(
                truth_guard_healthy and truth_guard.get("f2_context") and
                truth_guard.get("truth_received") and
                int(truth_guard.get("correction_count", 0) or 0) > 0 or
                int(truth_guard.get("height_recovery_count", 0) or 0) > 0)
            vertical_ok = bool(
                disagreement is not None and
                (disagreement["absolute_vertical_disagreement_m"] <=
                 self._localization_maximum_vertical_disagreement or
                 truth_vertical_rebased))
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
                with self._lock:
                    # Transform the known world +Y corridor direction through
                    # the synchronized truth/odom yaw pair. Cardinal snapping
                    # lost 17.7 degrees in fix09 and accumulated 1.9 m lateral
                    # error over a 6 m chord, re-entering the near room.
                    self._corridor_forward_heading_hint = (
                        self._corridor_heading_from_truth_odom(truth, odom))
                result = {
                    "corridor_forward_heading_hint_rad":
                        self._corridor_forward_heading_hint,
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
            # The stabilization topic can legitimately remain absent when
            # the bounded truth guard is the mechanism that keeps FAST-LIO
            # aligned.  Run f2neardoorfix then entered the fallback with
            # ``stabilized_odom=null`` even though fresh /Odometry was live;
            # the corridor heading hint became None and the fresh manager
            # replaced world +Y with a diagonal trajectory PCA axis.  Use the
            # freshest finite live odometry only for this fallback heading.
            odom = (self._stabilized_odom
                    if self._stabilized_odom is not None
                    else self._live_odom)
            odom_source = ("stabilized_odom"
                           if self._stabilized_odom is not None
                           else "live_odom" if self._live_odom is not None
                           else None)
            registration = dict(self._registration_status or {})
            motion_guard = dict(self._motion_guard_status or {})
            truth_guard = dict(self._truth_guard_status or {})
            degeneracy = dict(self._degeneracy_status or {})
        disagreement = (self._localization_disagreement(
            anchors[0], truth, anchors[1], odom)
            if anchors is not None and truth is not None and odom is not None
            else None)
        # A stair landing can leave FAST-LIO frozen or planar-shifted while
        # Gazebo truth, the truth guard, and the motion guard are all healthy.
        # Do not abort the next-floor mission in that bounded stationary case:
        # corridor/stair handoff is truth-guided and can continue safely.
        truth_fallback = bool(
            self._floor_number >= 2 and
            truth is not None and
            truth_guard.get("active") and
            truth_guard.get("truth_received") and
            not truth_guard.get("truth_stale") and
            (motion_guard.get("stationary", False) or
             (truth_stationary_since is not None and
              time.monotonic()-truth_stationary_since >= 0.5)))
        if truth_fallback:
            fallback = dict(
                event=self._event("LOCALIZATION_TRUTH_FALLBACK"),
                elapsed_sec=round(time.monotonic() - started, 3),
                truth_pose=list(truth),
                stabilized_odom=list(odom) if odom is not None else None,
                heading_odometry_source=odom_source,
                final_disagreement=disagreement,
                registration=registration or None,
                motion_guard=motion_guard or None,
                truth_guard=truth_guard or None,
                degeneracy=degeneracy or None,
                rejection_counts=rejection_counts,
                action="continue_truth_guided_exploration")
            with self._lock:
                self._corridor_forward_heading_hint = (
                    self._corridor_heading_from_truth_odom(truth, odom))
            fallback["corridor_forward_heading_hint_rad"] = (
                self._corridor_forward_heading_hint)
            self._append_localization_diagnostic(fallback)
            self._write_handoff(self._event("LOCALIZATION_TRUTH_FALLBACK"), **fallback)
            self._state_pub.publish(String(
                data=self._event("LOCALIZATION_TRUTH_FALLBACK")))
            rospy.logwarn("Using bounded truth fallback after upper-floor "
                          "FAST-LIO instability; continuing exploration.")
            return True
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
        """Switch the executor envelope and wait for a guard-qualified ack."""
        if self._second_floor_elevation is None:
            return False
        payload = self._second_floor_executor_context(
            self._second_floor_elevation, self._floor_index, self._floor_slug)
        request_id = "floor-{}-{}".format(
            self._floor_number, int(time.time() * 1000000))
        payload["request_id"] = request_id
        encoded = json.dumps(payload, sort_keys=True)
        with self._lock:
            self._executor_floor_context_status = None
        deadline = time.monotonic() + 4.0
        acknowledgement = None
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            self._executor_floor_context_pub.publish(String(data=encoded))
            wait_until = min(deadline, time.monotonic() + 0.20)
            while not rospy.is_shutdown() and time.monotonic() < wait_until:
                with self._lock:
                    status = dict(self._executor_floor_context_status or {})
                if (status.get("active") and
                        str(status.get("request_id", "")) == request_id and
                        int(status.get("floor_index", -1)) == self._floor_index and
                        status.get("truth_guard_authorized")):
                    acknowledgement = status
                    break
                time.sleep(0.02)
            if acknowledgement is not None:
                break
        if acknowledgement is None:
            with self._lock:
                last_status = dict(self._executor_floor_context_status or {})
            rospy.logerr(
                "[%s] executor floor context was not acknowledged with a "
                "fresh truth guard: %s", self._floor_slug, last_status)
            self._write_handoff(
                self._event("EXECUTOR_CONTEXT_ACK_TIMEOUT"),
                executor_floor_context=payload,
                executor_floor_context_status=last_status)
            return False
        self._write_handoff(
            self._event("EXECUTOR_CONTEXT_ACTIVE"),
            executor_floor_context=payload,
            executor_floor_context_status=acknowledgement)
        self._state_pub.publish(String(
            data=self._event("EXECUTOR_CONTEXT_ACTIVE")))
        return True

    def _second_floor_truth_route(self):
        """Build a safe upper-landing-to-corridor route from offline truth.

        The first waypoint leaves the stair opening across the physical upper
        landing.  The second lies just inside the floor-2 corridor, where the
        ordinary online corridor/door pipeline can start with the same pose
        and heading assumptions as it does on floor 1.
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
            corridor_x, corridor_y = self._truth_corridor_entry_target(
                corridor, self._truth_corridor_ingress_depth)
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
            # The upper landing is a tight stair-throat. A single long
            # horizontal target is frequently rejected by the local policy
            # at the collision seam, so use short, truth-guided segments.
            if self._floor_number >= 2:
                east_x = float(stair["x_max"]) + 0.75
                north_y = max(float(stair["y_max"]) + 0.55,
                              corridor_y - 0.35)
                route = [
                    {"stage": "upper_stair_exit",
                     "target": (landing_x, landing_y),
                     "stair_bounds": dict(stair),
                     "landing_accept": True},
                    {"stage": "upper_landing_east_step",
                     "target": (float(stair["x_max"]) - 0.55, landing_y),
                     "stair_bounds": dict(stair)},
                    {"stage": "upper_landing_east_clear",
                     "target": (east_x, landing_y),
                     "stair_bounds": dict(stair)}]
                # Both upper landings border the open stair well.  run131
                # accepted the east-clear point 0.56 m early, then sent F2 on
                # one long diagonal.  While yawing, the plane gait drifted
                # west from x=-1.39 across the narrow infill at x=-1.65 and
                # fell back into the stair core.  Use short physical legs on
                # every upper floor.  Hold the safe east slab centre until
                # north of the opening; only then blend toward the corridor.
                opening_clear_y = float(stair["y_max"]) + 0.35
                blend_length = max(0.25, north_y - opening_clear_y)
                # The controller recomputes the world-frame correction from
                # truth on every cycle, so metre-spaced collinear goals do
                # not add a safety observation.  They only force the learned
                # gait to decelerate and reacquire its stride seven times
                # (run152: 33.2 simulation seconds for this straight leg).
                # The east-clear point already proves that the trunk is on
                # the supported slab.  From there to ``opening_clear_y`` the
                # target x is exactly constant and truth feedback recomputes
                # the world-frame correction every control cycle.  A middle
                # collinear goal therefore adds no collision observation; it
                # only forces the learned gait to decelerate and reacquire
                # stride.  Keep the actual opening-clear gate and the final
                # corridor blend, which are the two geometry changes that do
                # carry independent safety meaning.
                # The first target remains at the east slab centre, so a
                # long diagonal across the open stair well is still
                # impossible.
                north_targets = [opening_clear_y, north_y]
                for index, step_y in enumerate(north_targets, 1):
                    blend = max(0.0, min(
                        1.0, (step_y - opening_clear_y) / blend_length))
                    route.append({
                        "stage": "upper_landing_north_step_{:02d}".format(index),
                        "target": (east_x + (corridor_x - east_x) * blend,
                                   step_y),
                        "stair_bounds": dict(stair),
                        "opening_clear_y": opening_clear_y})
            else:
                stair_exit_x = max(float(stair["x_max"]) + 0.70,
                                   landing_x + landing_half_width + 0.55)
                route = [{"stage": "upper_stair_exit",
                          "target": (stair_exit_x, landing_y),
                          "stair_bounds": dict(stair)}]
            # The handoff needs a safely interior corridor pose, not exact
            # convergence to one edge-biased point.  The configured bounded
            # ingress clears the lobby throat while leaving ample distance to
            # the first room row and tolerating plane-policy lateral drift.
            minimum_ingress_y = min(
                corridor_y - self._truth_guide_tolerance,
                float(corridor["y_min"]) + 1.50)
            route.append({"stage": "{}_corridor_entry".format(self._floor_slug),
                 "target": (corridor_x, corridor_y),
                 "corridor_bounds": dict(corridor),
                 "minimum_ingress_y": minimum_ingress_y})
            if ((self._floor_number == 2 and
                 not self._truth_fast_handoff_active) or
                    self._floor_number >= 3):
                # After a stair seam the fresh upper-floor projection is too
                # sparse to plan the first normal corridor chord reliably.
                # Continue the verified truth centreline only to the first
                # door-observation band, then hand control back to the online
                # detector/scheduler.  This is deliberately a corridor pose
                # (not a room or door target): ENTRY, G3/G4, EXIT and hazard
                # confirmation remain online.  F3 needs this same seed; the
                # previous F3 run crossed both truth door rows without any
                # local candidates because it started online at y≈8.8.
                seed_y = min(float(corridor["y_max"]) - 1.0,
                             float(corridor_y) + 4.5)
                route.append({
                    "stage": "{}_door_observation_seed".format(
                        self._floor_slug),
                    "target": (corridor_x, seed_y),
                    "corridor_bounds": dict(corridor),
                    "minimum_ingress_y": min(
                        seed_y - self._truth_guide_tolerance,
                        float(corridor["y_max"]) - 1.0),
                })
            return route
        except (OSError, ValueError, KeyError, StopIteration,
                ET.ParseError, TypeError) as error:
            rospy.logerr("Cannot construct second-floor truth route: %s", error)
            return None

    @staticmethod
    def _truth_corridor_entry_target(corridor, ingress_depth,
                                     terminal_margin=1.0):
        """Return a centreline target safely inside the truth corridor."""
        x_min, x_max = float(corridor["x_min"]), float(corridor["x_max"])
        y_min, y_max = float(corridor["y_min"]), float(corridor["y_max"])
        depth = max(0.5, float(ingress_depth))
        target_y = min(y_min + depth, y_max - max(0.5, terminal_margin))
        return 0.5 * (x_min + x_max), target_y

    @staticmethod
    def _truth_corridor_arrival_valid(pose, waypoint, tolerance):
        """Reject a radial arrival that has not actually cleared the lobby."""
        bounds = waypoint.get("corridor_bounds")
        if not bounds:
            return True
        margin = max(0.0, float(tolerance))
        minimum_y = float(waypoint.get(
            "minimum_ingress_y", float(bounds["y_min"])))
        return bool(
            float(bounds["x_min"]) - margin <= float(pose[0]) <=
            float(bounds["x_max"]) + margin and
            minimum_y <= float(pose[1]) <= float(bounds["y_max"]) + margin)

    def _truth_reposition_to_corridor(self, waypoint, pose):
        """Use the known map pose to cross an upper stair/lobby seam.

        The generated upper-floor stair core can have a collision seam that
        blocks the plane policy after it has already cleared the stair edge.
        This bounded truth handoff applies to both F2 and F3; it is never used
        on the first floor.
        """
        if self._floor_number < 2 or not waypoint or pose is None:
            return False
        rospy.logerr(
            "[%s] physical corridor guide stalled; Gazebo seam reposition is disabled.",
            self._floor_slug)
        return False
        try:
            rospy.wait_for_service("/gazebo/set_model_state", timeout=1.0)
            state = ModelState()
            state.model_name = "a1_gazebo"
            state.reference_frame = "world"
            state.pose.position.x = float(waypoint[0])
            state.pose.position.y = float(waypoint[1])
            state.pose.position.z = max(
                float(pose[2]), float(self._second_floor_elevation) + 0.32)
            yaw = math.pi / 2.0
            state.pose.orientation.z = math.sin(0.5 * yaw)
            state.pose.orientation.w = math.cos(0.5 * yaw)
            response = rospy.ServiceProxy(
                "/gazebo/set_model_state", SetModelState)(state)
            if response.success:
                # The Gazebo service response is synchronous, but the
                # model-states subscriber can deliver the corresponding
                # truth pose one or more timer cycles later.  The guide
                # immediately validates corridor arrival after this bounded
                # seam handoff; leaving _truth_pose at the pre-handoff pose
                # makes a valid F3 ingress look like a failed guide and
                # aborts before the first room target is scheduled.
                with self._lock:
                    self._truth_pose = (
                        float(state.pose.position.x),
                        float(state.pose.position.y),
                        float(state.pose.position.z),
                        float(yaw))
                    self._truth_pose_received_at = time.monotonic()
                rospy.logwarn(
                    "[%s] truth seam handoff: repositioned from "
                    "(%.2f, %.2f) to corridor ingress (%.2f, %.2f).",
                    self._floor_slug, pose[0], pose[1],
                    waypoint[0], waypoint[1])
            return bool(response.success)
        except (rospy.ROSException, rospy.ServiceException) as error:
            rospy.logerr("[%s] truth seam handoff failed: %s",
                         self._floor_slug, error)
            return False

    @staticmethod
    def _wrapped_angle_error(target, current):
        return math.atan2(math.sin(float(target) - float(current)),
                          math.cos(float(target) - float(current)))

    def _publish_truth_body_command(self, forward, yaw_rate):
        """Publish a body-frame command while retaining RL ownership."""
        command = Twist()
        command.linear.x = float(forward)
        command.angular.z = float(yaw_rate)
        self._cmd_pub.publish(command)
        self._hold_rl()

    def _publish_f3_world_command(self, world_x, world_y, yaw_rate, pose):
        """Track an F3 guide leg in world axes despite policy yaw drift."""
        command = Twist()
        yaw = float(pose[3])
        command.linear.x = (float(world_x) * math.cos(yaw) +
                            float(world_y) * math.sin(yaw))
        command.linear.y = (-float(world_x) * math.sin(yaw) +
                            float(world_y) * math.cos(yaw))
        command.angular.z = float(yaw_rate)
        self._cmd_pub.publish(command)
        self._hold_rl()

    def _f3_effective_turn_rate(self, error):
        """Return a physical F3 yaw command outside the measured dead zone."""
        if abs(float(error)) <= self._truth_heading_tolerance:
            return 0.0
        magnitude = min(
            self._f3_turn_maximum_yaw_rate,
            max(self._f3_turn_minimum_yaw_rate, 1.10 * abs(float(error))))
        return math.copysign(magnitude, float(error))

    def _f3_stationary_turn_assist(
            self, heading, recovery_target=None,
            recovery_arrival_tolerance=None):
        """Perform one bounded physical F3 in-place yaw correction.

        Older revisions rewrote Gazebo's model quaternion here.  That hid the
        plane-policy deadband and violated the physical full-flow contract.
        Use truth only as feedback for a real RL yaw command; position and
        velocity are never rewritten.
        """
        if self._floor_number < 3:
            return False
        sim_started = rospy.Time.now()
        wall_deadline = time.monotonic() + 12.0
        stable_since = None
        progress_yaw = None
        progress_started = rospy.Time.now()
        arc_recovery_used = False
        target = None
        if recovery_target is not None:
            try:
                target = (float(recovery_target[0]),
                          float(recovery_target[1]))
            except (TypeError, ValueError, IndexError):
                target = None
        arrival_tolerance = None
        if recovery_arrival_tolerance is not None:
            try:
                arrival_tolerance = max(
                    0.0, float(recovery_arrival_tolerance))
            except (TypeError, ValueError):
                arrival_tolerance = None
        while (not rospy.is_shutdown() and
               (rospy.Time.now() - sim_started).to_sec() < 10.0 and
               time.monotonic() < wall_deadline):
            with self._lock:
                pose = tuple(self._truth_pose) if self._truth_pose else None
            if pose is None:
                return False
            if target is not None and arrival_tolerance is not None:
                target_distance = math.hypot(
                    target[0] - float(pose[0]),
                    target[1] - float(pose[1]))
                if target_distance <= arrival_tolerance:
                    # This assist only pre-aligns a physical route leg.  Once
                    # that leg's actual truth-position gate is already met,
                    # forcing an additional in-place turn can only add drift
                    # or strand the dog at the stair lip.  The next route leg
                    # (and ultimately the descent controller) owns its own
                    # heading and posture gates.
                    self._publish_truth_body_command(0.0, 0.0)
                    rospy.loginfo(
                        "[third_floor] bounded yaw assist reached the "
                        "physical route gate (distance=%.3f m).",
                        target_distance)
                    return True
            error = self._wrapped_angle_error(heading, pose[3])
            if abs(error) <= 0.16:
                stable_since = stable_since or rospy.Time.now()
                self._publish_truth_body_command(0.0, 0.0)
                if (rospy.Time.now() - stable_since).to_sec() >= 0.20:
                    rospy.loginfo(
                        "[third_floor] bounded physical yaw correction complete.")
                    return True
            else:
                stable_since = None
                yaw_rate = self._f3_effective_turn_rate(error)
                if progress_yaw is None:
                    progress_yaw = float(pose[3])
                    progress_started = rospy.Time.now()
                yaw_progress = abs(self._wrapped_angle_error(
                    pose[3], progress_yaw))
                if yaw_progress >= 0.08:
                    progress_yaw = float(pose[3])
                    progress_started = rospy.Time.now()
                stalled = ((rospy.Time.now() - progress_started).to_sec() >=
                           self._f3_turn_stall_seconds)
                if stalled and not arc_recovery_used:
                    # Move a few centimetres while retaining the same
                    # physical yaw command.  This changes the loaded foot
                    # pattern that caused run133's stationary deadlock; no
                    # pose or velocity is rewritten.  Return legs supply
                    # their target direction below, while the original F3
                    # ingress fallback keeps its eastward landing-clear arc.
                    arc_recovery_used = True
                    progress_yaw = float(pose[3])
                    progress_started = rospy.Time.now()
                    rospy.logwarn(
                        "[third_floor] F3 yaw stalled; using one bounded "
                        "eastward physical arc recovery.")
                # A stair-return caller supplies the next collision-safe
                # route waypoint.  Start a small target-directed physical arc
                # immediately instead of waiting for an in-place gait stall:
                # run148 kept making tiny yaw progress, so the old stall gate
                # never fired and the entire 10 s pre-align budget expired.
                # Ingress calls have no target and retain the original
                # position-preserving turn plus one bounded eastward recovery.
                route_arc_active = bool(
                    target is not None and arrival_tolerance is not None and
                    target_distance > arrival_tolerance)
                if arc_recovery_used or route_arc_active:
                    if target is not None:
                        # Break the stationary gait deadlock by moving along
                        # the route that we are already trying to execute.
                        # The old fixed eastward arc was correct for the F3
                        # ingress landing but moved away from the west/south
                        # stair-return targets and caused run147's bounded
                        # pre-align timeout.  This remains a physical RL
                        # command; truth supplies feedback and direction only.
                        dx = target[0] - float(pose[0])
                        dy = target[1] - float(pose[1])
                        norm = math.hypot(dx, dy)
                        if norm > 1.0e-6:
                            arc_speed = min(
                                self._f3_turn_arc_world_speed,
                                max(0.08, 0.35 * target_distance))
                            world_x = arc_speed * dx / norm
                            world_y = arc_speed * dy / norm
                        else:
                            world_x = world_y = 0.0
                    else:
                        world_x = self._f3_turn_arc_world_speed
                        world_y = 0.0
                    self._publish_f3_world_command(
                        world_x, world_y, yaw_rate, pose)
                else:
                    self._publish_truth_body_command(0.0, yaw_rate)
            time.sleep(0.05)
        self._publish_truth_body_command(0.0, 0.0)
        rospy.logerr("[third_floor] bounded physical yaw correction timed out.")
        return False

    def _guide_f3_stair_platform_to_corridor(self, route):
        """Sequential physical F3 platform-to-corridor handoff, no teleport."""
        with self._lock:
            initial = self._truth_pose
        if initial is None:
            return False
        # ``route[-1]`` is the farther door-observation seed (minimum ingress
        # y=12.8 in the generated world), not the corridor-entry contract.
        # The physical platform guide only owns the stair-to-corridor handoff;
        # requiring the later seed here made a valid ingress impossible after
        # the former lip stop.  Select the explicit entry waypoint and leave
        # the room manager to perform the subsequent physical door search.
        corridor_entry_stage = "{}_corridor_entry".format(self._floor_slug)
        corridor_waypoint = next(
            (waypoint for waypoint in route
             if waypoint.get("stage") == corridor_entry_stage),
            None)
        if corridor_waypoint is None:
            return False
        bounds = corridor_waypoint.get("corridor_bounds") or {}
        stair = route[0].get("stair_bounds") or {}
        try:
            # The F3 stair throat ends west of the corridor.  The previous
            # target (-1.35 m) was still outside corridor x_min=-1.10; the
            # generic 0.55 m tolerance accepted it at about -1.82, then the
            # north seed drive hit the corridor wall.  Cross fully into the
            # corridor width before turning north.
            bridge_x = float(bounds["x_min"]) + 0.35
            ingress_y = min(float(bounds["y_min"]) +
                            max(1.5, min(self._truth_corridor_ingress_depth, 3.0)),
                            float(bounds["y_max"]) - 1.0)
        except (KeyError, TypeError, ValueError):
            return False
        trace = []
        guide_started = time.monotonic()
        guide_started_sim = float(rospy.Time.now().to_sec())
        self._state_pub.publish(String(data=self._event("F3_PLATFORM_SETTLING")))
        # FixedStand and the real locomotion-ready gate have already proved a
        # stationary upright body.  Retain a short command-ownership settle,
        # but do not repeat another 1.5 s physical-stability dwell here.
        settle_until = time.monotonic() + 0.50
        while not rospy.is_shutdown() and time.monotonic() < settle_until:
            self._publish_truth_body_command(0.0, 0.0)
            time.sleep(0.05)
        # Both translation controllers already transform a desired world
        # vector into the live body frame and apply a bounded yaw correction.
        # Separate in-place turns therefore duplicate work and repeatedly hit
        # the loaded-foot yaw dead zone (run152: 11.9 + 10.1 simulation
        # seconds).  Converge heading during the same physical east/north
        # motion.  Geometry, platform-height gates and progress watchdogs are
        # unchanged.
        stages = (
            ("stair_exit_drive_out", "east", bridge_x),
            ("corridor_seed_drive", "north", ingress_y),
        )
        for stage, mode, target in stages:
            self._state_pub.publish(String(data=self._event(
                "F3_" + stage.upper())))
            stage_started = time.monotonic()
            with self._lock:
                start_pose = self._truth_pose
            if start_pose is None:
                return False
            axis = 0 if mode == "east" else 1
            start_value = float(start_pose[axis])
            requested_distance = (abs(float(target) - start_value)
                                  if mode != "turn" else 0.0)
            timeout = (24.0 if mode == "turn" else
                       max(22.0, requested_distance / 0.12 + 18.0))
            # F3 platform motion is physically slower than its requested
            # world command.  run222 kept making measurable eastward progress
            # but the one-shot deadline expired only 0.31 m before the bridge
            # target.  Refresh the soft deadline while progress continues,
            # with a bounded hard deadline so a real stall still terminates.
            # Motion budgets are ROS/Gazebo simulation time.  Using a wall
            # deadline here made low RTF change mission semantics: run140's
            # F3 north seed was moving at ~0.8 m/s in Gazebo truth, yet its
            # wall deadline expired after only 8.4 simulation seconds at
            # y=7.62.  Keep a separate generous wall watchdog for a frozen
            # simulator/process, but never charge low RTF to physical motion.
            stage_sim_started = rospy.Time.now()
            stage_sim_deadline_sec = float(timeout)
            stage_sim_hard_deadline_sec = max(
                timeout + 24.0, 1.75 * timeout)
            stage_wall_hard_deadline = stage_started + max(
                180.0, 6.0 * timeout)
            best_progress = 0.0
            # Physical progress and motion timeout share the ROS/Gazebo
            # clock.  A wall-clock ``last_progress`` made the 15 s stall
            # watchdog fire after only a few simulated seconds at low RTF
            # (run142 stopped a still-moving north ingress at y=6.91 and
            # coasted to the valid y=7.63 lip after zero was published).
            # The independent wall hard deadline above remains the frozen-
            # simulator/process watchdog.
            last_progress_sim = stage_sim_started
            last_trace = -math.inf
            completed = False
            turn_assisted = False
            turn_assist_attempted = False
            turn_best_error = math.inf
            turn_initial_error = None
            while (not rospy.is_shutdown() and
                   (rospy.Time.now() - stage_sim_started).to_sec() <
                   stage_sim_deadline_sec and
                   time.monotonic() < stage_wall_hard_deadline):
                with self._lock:
                    pose = self._truth_pose
                if pose is None:
                    time.sleep(0.05)
                    continue
                if pose[2] < (self._second_floor_elevation +
                              self._second_floor_height_margin):
                    self._publish_truth_body_command(0.0, 0.0)
                    self._write_handoff(self._event("F3_GUIDE_HEIGHT_LOST"),
                                        stage=stage, truth_pose=list(pose))
                    return False
                now = time.monotonic()
                if now - last_trace >= 0.25:
                    trace.append({"t": round(now - guide_started, 3),
                                  "stage": stage, "x": pose[0], "y": pose[1],
                                  "z": pose[2], "yaw": pose[3], "target": target})
                    last_trace = now
                if mode == "turn":
                    error = self._wrapped_angle_error(target, pose[3])
                    if turn_initial_error is None:
                        turn_initial_error = abs(error)
                    if abs(error) < turn_best_error:
                        turn_best_error = abs(error)
                    deadband_elapsed = time.monotonic() - stage_started
                    deadband_progress = max(
                        0.0, float(turn_initial_error or 0.0) -
                        float(turn_best_error))
                    deadband_proven = bool(
                        deadband_elapsed >=
                        self._f3_turn_deadband_assist_delay and
                        deadband_progress < 0.12)
                    if (not turn_assist_attempted and deadband_proven and
                            turn_best_error > self._truth_heading_tolerance):
                        turn_assist_attempted = True
                        turn_assisted = bool(
                            self._f3_stationary_turn_assist(target))
                        time.sleep(0.30)
                        if turn_assisted:
                            continue
                    if abs(error) <= self._truth_heading_tolerance:
                        completed = True
                        self._publish_truth_body_command(0.0, 0.0)
                        break
                    yaw_rate = self._f3_effective_turn_rate(error)
                    self._publish_truth_body_command(0.0, yaw_rate)
                else:
                    coordinate = float(pose[axis])
                    remaining = float(target) - coordinate
                    completion_tolerance = (0.12 if mode == "east" else
                                           self._truth_guide_tolerance)
                    # The F3 corridor threshold has a contact lip at y≈7.62
                    # while metadata begins at y=7.85.  Reaching that lip is
                    # useful progress evidence, but it is not completion: in
                    # run143 the robot was still moving north, yet stopping at
                    # y=7.46 was followed by an impossible validation against
                    # the far door-observation seed.  Keep driving physically
                    # to the explicit corridor-entry target.
                    f3_corridor_lip = (
                        self._floor_number >= 3 and mode == "north" and
                        float(bounds.get("x_min", math.inf)) <= pose[0] <=
                        float(bounds.get("x_max", -math.inf)) and
                        pose[1] >= float(bounds.get("y_min", math.inf)) - 0.40)
                    # The generated west lip can hold the body a few
                    # centimetres outside the metadata rectangle even though
                    # it has physically reached the corridor threshold.  Do
                    # not spend a full stall interval pushing into that seam.
                    # A near-lip pose is evidence only for invoking the same
                    # bounded truth handoff used by the watchdog; it is never
                    # accepted directly as corridor arrival.
                    f3_near_lip_outside = (
                        self._floor_number >= 3 and mode == "north" and
                        float(bounds.get("x_min", math.inf)) - 0.18 <= pose[0] <=
                        float(bounds.get("x_max", -math.inf)) + 0.18 and
                        not (float(bounds.get("x_min", math.inf)) <= pose[0] <=
                             float(bounds.get("x_max", -math.inf))) and
                        pose[1] >= float(bounds.get("y_min", math.inf)) - 0.40)
                    if (f3_near_lip_outside and
                            self._truth_reposition_to_corridor(
                                corridor_waypoint["target"], pose)):
                        self._publish_truth_body_command(0.0, 0.0)
                        time.sleep(0.15)
                        with self._lock:
                            seam_pose = self._truth_pose
                        trace.append({
                            "stage": "f3_truth_near_lip_handoff",
                            "x": pose[0], "y": pose[1], "z": pose[2],
                            "target": list(corridor_waypoint["target"]),
                        })
                        if self._truth_corridor_arrival_valid(
                                seam_pose, corridor_waypoint,
                                self._truth_guide_tolerance):
                            self._state_pub.publish(String(data=self._event(
                                "LOCALIZATION_STABILIZING")))
                            self._write_handoff(self._event(
                                "F3_NEAR_LIP_HANDOFF_COMPLETE"),
                                truth_pose=list(seam_pose), route=route,
                                trace=trace)
                            self._write_truth_guide_result(
                                self._event("GT_GUIDE_COMPLETE"),
                                stage="f3_truth_near_lip_handoff",
                                route=route, trace=trace,
                                truth_pose=list(seam_pose),
                                localization_guard_armed=True)
                            self._state_pub.publish(String(data=self._event(
                                "CORRIDOR_ENTRY_REACHED")))
                            return True
                    f3_east_bridge_supported = bool(
                        self._floor_number >= 3 and mode == "east" and
                        self._f3_east_bridge_arrival(pose, target, bounds))
                    if (remaining <= completion_tolerance or
                            f3_east_bridge_supported):
                        completed = True
                        self._publish_truth_body_command(0.0, 0.0)
                        break
                    expected_heading = 0.0 if mode == "east" else math.pi / 2.0
                    heading_error = self._wrapped_angle_error(expected_heading,
                                                              pose[3])
                    yaw_rate = max(-0.28, min(0.28, 0.9 * heading_error))
                    if mode == "east":
                        # The former fixed 0.22 m/s command made the short
                        # bridge itself consume tens of simulated seconds.
                        # Keep the tight landing leg bounded, but use distance
                        # proportional speed and decelerate near the seam.
                        # Stay conservative while weight is still over the
                        # stair throat.  Once the trunk has crossed the known
                        # flat-side anchor, continue the same straight east
                        # chord without the landing crawl cap.
                        flat_side_x = float(stair.get(
                            "x_max", -math.inf)) - 0.55
                        east_cap = (
                            self._f3_landing_clear_speed
                            if pose[0] >= flat_side_x else 0.42)
                        world_x = min(
                            east_cap, max(0.25, 0.70 * abs(remaining)))
                        world_y = 0.0
                    elif mode == "north":
                        # The F3 north seed drive must stay inside the
                        # corridor width.  run ..._227 exited the stair throat
                        # at x=-1.14 while corridor x_min=-1.10; the pure-north
                        # command then hugged the outer stair/corridor junction
                        # wall and the dog fell off its edge (z 5.51 -> 4.17).
                        # Blend a bounded lateral recentre proportional to how
                        # far outside the corridor the body currently is, so
                        # the seed drive approaches the centreline instead of
                        # scraping the west wall.
                        # Once supported inside the corridor, the long north
                        # leg can use the same normal flat-floor envelope as
                        # room transit.  This removes the measured ~95 s F3
                        # seed crawl without changing its geometric route.
                        inside_corridor = bool(
                            bounds and
                            float(bounds.get("x_min", math.inf)) <= pose[0] <=
                            float(bounds.get("x_max", -math.inf)))
                        # Once all four feet are supported inside the broad
                        # corridor, use the proven flat-floor translation
                        # envelope.  The old 0.62 command made run117's
                        # 12-m door-seed leg consume 40.7 simulation seconds.
                        # Outside the corridor keep the conservative seam
                        # speed and lateral recentring below.
                        north_cap = (
                            self._f3_corridor_ingress_speed
                            if inside_corridor else 0.62)
                        world_y = min(
                            north_cap, max(0.28, 0.70 * abs(remaining)))
                        world_x = (
                            self._f3_north_seed_lateral_speed(
                                pose, bounds, stair)
                            if self._floor_number >= 3 else 0.0)
                    else:
                        world_x = 0.0
                        world_y = 0.0
                    self._publish_f3_world_command(world_x, world_y, yaw_rate, pose)
                    progress = max(0.0, coordinate - start_value)
                    if progress >= best_progress + self._truth_guide_progress_epsilon:
                        best_progress = progress
                        last_progress_sim = rospy.Time.now()
                        elapsed_sim = (
                            rospy.Time.now() - stage_sim_started).to_sec()
                        stage_sim_deadline_sec = min(
                            stage_sim_hard_deadline_sec,
                            max(stage_sim_deadline_sec,
                                elapsed_sim +
                                self._truth_guide_stall_timeout +
                                self._truth_guide_timeout_margin))
                    if ((rospy.Time.now() - last_progress_sim).to_sec() >=
                            self._truth_guide_stall_timeout):
                        self._publish_truth_body_command(0.0, 0.0)
                        self._write_handoff(self._event("F3_GUIDE_STALLED"),
                                            stage=stage, target=target,
                                            best_progress=best_progress,
                                            trace=trace)
                        # The east bridge is a generated collision seam, not a
                        # navigable floor: after an evidenced no-progress
                        # watchdog, cross it once into the already validated
                        # F3 corridor ingress.  This mirrors the existing F2
                        # seam recovery and is deliberately unavailable on F1.
                        if self._truth_reposition_to_corridor(
                                corridor_waypoint["target"], pose):
                            trace.append({"stage": "f3_truth_seam_handoff",
                                          "x": pose[0], "y": pose[1],
                                          "z": pose[2],
                                          "target": list(corridor_waypoint["target"])})
                            self._state_pub.publish(String(data=self._event(
                                "F3_SEAM_RL_HOLD")))
                            settle_deadline = time.monotonic() + 0.35
                            while (not rospy.is_shutdown() and
                                   time.monotonic() < settle_deadline):
                                self._publish_truth_body_command(0.0, 0.0)
                                time.sleep(0.05)
                            with self._lock:
                                seam_pose = self._truth_pose
                            if self._truth_corridor_arrival_valid(
                                    seam_pose, corridor_waypoint,
                                    self._truth_guide_tolerance):
                                self._state_pub.publish(String(data=self._event(
                                    "LOCALIZATION_STABILIZING")))
                                self._write_handoff(self._event(
                                    "F3_SEAM_HANDOFF_COMPLETE"),
                                    truth_pose=list(seam_pose), route=route,
                                    trace=trace)
                                self._write_truth_guide_result(
                                    self._event("GT_GUIDE_COMPLETE"),
                                    stage="f3_truth_seam_handoff", route=route,
                                    trace=trace, truth_pose=list(seam_pose),
                                    localization_guard_armed=True)
                                self._state_pub.publish(String(data=self._event(
                                    "CORRIDOR_ENTRY_REACHED")))
                                return True
                        self._write_truth_guide_result(
                            self._event("F3_GUIDE_STALLED"), stage=stage,
                            route=route, trace=trace, target=target,
                            best_progress=best_progress)
                        return False
                time.sleep(0.05)
            if (not completed and mode == "turn" and
                    self._f3_stationary_turn_assist(target)):
                # Contact with the upper landing can pull yaw back after the
                # early deadband assist. Apply one final stationary correction
                # at the bounded turn deadline and immediately continue the
                # physical translation stages; position is never changed.
                time.sleep(0.15)
                with self._lock:
                    corrected_pose = self._truth_pose
                completed = bool(
                    corrected_pose is not None and
                    abs(self._wrapped_angle_error(
                        target, corrected_pose[3])) <=
                    1.5 * self._truth_heading_tolerance)
                if completed:
                    trace.append({"stage": stage + "_deadline_yaw_assist",
                                  "x": corrected_pose[0],
                                  "y": corrected_pose[1],
                                  "z": corrected_pose[2],
                                  "yaw": corrected_pose[3],
                                  "target": target})
            if not completed:
                # The generated F3 stair/corridor seam can produce a small
                # positive eastward motion while the body remains trapped at
                # the collision lip.  That is different from a zero-progress
                # stall, so it never entered the seam-recovery branch above
                # and previously expired the stage deadline.  Apply the same
                # single, truth-bounded seam handoff on this terminal east
                # timeout; it is F3-only and still requires the corridor
                # arrival gate after repositioning.
                with self._lock:
                    timeout_pose = self._truth_pose
                east_timeout_progress = (
                    mode == "east" and timeout_pose is not None and
                    best_progress >= 0.15)
                if (self._floor_number >= 3 and east_timeout_progress and
                        self._truth_reposition_to_corridor(
                            corridor_waypoint["target"], timeout_pose)):
                    trace.append({"stage": "f3_truth_timeout_seam_handoff",
                                  "x": timeout_pose[0], "y": timeout_pose[1],
                                  "z": timeout_pose[2],
                                  "target": list(corridor_waypoint["target"]),
                                  "best_progress": best_progress})
                    with self._lock:
                        seam_pose = self._truth_pose
                    if self._truth_corridor_arrival_valid(
                            seam_pose, corridor_waypoint,
                            self._truth_guide_tolerance):
                        self._state_pub.publish(String(data=self._event(
                            "CORRIDOR_ENTRY_REACHED")))
                        self._write_handoff(
                            self._event("F3_TIMEOUT_SEAM_HANDOFF_COMPLETE"),
                            truth_pose=list(seam_pose), route=route,
                            trace=trace)
                        self._write_truth_guide_result(
                            self._event("GT_GUIDE_COMPLETE"),
                            stage="f3_truth_timeout_seam_handoff", route=route,
                            trace=trace, truth_pose=list(seam_pose),
                            localization_guard_armed=True)
                        return True
                self._publish_truth_body_command(0.0, 0.0)
                self._write_handoff(self._event("F3_GUIDE_TIMEOUT"),
                                    stage=stage, target=target, trace=trace)
                self._write_truth_guide_result(self._event("F3_GUIDE_TIMEOUT"),
                                               stage=stage, route=route,
                                               trace=trace, target=target)
                return False
        with self._lock:
            final_pose = self._truth_pose
        final_at_f3_lip = bool(
            final_pose is not None and
            float(bounds.get("x_min", math.inf)) <= final_pose[0] <=
            float(bounds.get("x_max", -math.inf)) and
            # Keep the terminal deadline check identical to the online
            # ``f3_corridor_lip`` gate above.  entryclearfix reached a valid
            # x=-1.045, y=y_min-0.389 pose between the final loop sample and
            # the deadline; the former stricter -0.30 terminal threshold
            # rejected the same physical entrance that the next online
            # sample would have accepted.
            final_pose[1] >= float(bounds.get("y_min", math.inf)) - 0.40)
        final_in_corridor = self._truth_corridor_arrival_valid(
            final_pose, corridor_waypoint, self._truth_guide_tolerance)
        if final_at_f3_lip and not final_in_corridor:
            # The physical guide has paid the stair/lobby traversal and
            # reached the generated collision lip. Starting the online room
            # manager here leaves only two known-free cells and its first
            # corridor chord can slide off the west edge. Complete the same
            # single bounded seam handoff already used by the stall and
            # near-lip branches, to the verified door-observation seed. Room
            # ENTRY/G3/G4/EXIT remain physical online transactions.
            if self._truth_reposition_to_corridor(
                    corridor_waypoint["target"], final_pose):
                time.sleep(0.15)
                with self._lock:
                    final_pose = self._truth_pose
                final_in_corridor = self._truth_corridor_arrival_valid(
                    final_pose, corridor_waypoint,
                    self._truth_guide_tolerance)
                trace.append({
                    "stage": "f3_truth_terminal_lip_handoff",
                    "x": final_pose[0] if final_pose else None,
                    "y": final_pose[1] if final_pose else None,
                    "z": final_pose[2] if final_pose else None,
                    "target": list(corridor_waypoint["target"]),
                })
                if final_in_corridor:
                    self._write_handoff(self._event(
                        "F3_TERMINAL_LIP_HANDOFF_COMPLETE"),
                        truth_pose=list(final_pose), route=route,
                        trace=trace)
        if not final_in_corridor:
            self._write_handoff(self._event("F3_CORRIDOR_REGION_REJECTED"),
                                truth_pose=list(final_pose) if final_pose else None,
                                corridor_bounds=bounds, trace=trace)
            return False
        self._state_pub.publish(String(data=self._event("LOCALIZATION_STABILIZING")))
        self._write_handoff(self._event("CORRIDOR_ENTRY_REACHED"),
                            truth_pose=list(final_pose), route=route, trace=trace,
                            guide_mode="sequential_physical")
        self._write_truth_guide_result(self._event("GT_GUIDE_COMPLETE"),
                                       stage="sequential_physical", route=route,
                                       trace=trace, truth_pose=list(final_pose),
                                       localization_guard_armed=True)
        self._state_pub.publish(String(data=self._event("CORRIDOR_ENTRY_REACHED")))
        return True

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
        # F3 uses the physical sequential strategy from the reference driver.
        # F2 retains its proven waypoint guide on the wider landing.
        if self._floor_number >= 3:
            return self._guide_f3_stair_platform_to_corridor(route)
        # Allow the stair manager's latched READY callback to release command
        # ownership before the plane-policy guide begins to move.
        release_until = time.monotonic() + 0.5
        while not rospy.is_shutdown() and time.monotonic() < release_until:
            self._cmd_pub.publish(Twist())
            self._hold_rl()
            time.sleep(0.05)
        # F3 may be levelled directly on the flat truth corridor before the
        # controller reset.  It is already beyond the stair seam, so never
        # drive it back through the narrow landing just to replay route legs.
        with self._lock:
            corridor_pose = self._truth_pose
        if (self._floor_number >= 3 and corridor_pose is not None and
                self._truth_corridor_arrival_valid(
                    corridor_pose, route[-1], self._truth_guide_tolerance)):
            trace.append({"stage": "prepositioned_corridor",
                          "x": corridor_pose[0], "y": corridor_pose[1],
                          "z": corridor_pose[2], "yaw": corridor_pose[3],
                          "target": list(route[-1]["target"])})
            self._state_pub.publish(String(
                data=self._event("LOCALIZATION_STABILIZING")))
            settle_deadline = time.monotonic() + 0.35
            while (not rospy.is_shutdown() and
                   time.monotonic() < settle_deadline):
                self._cmd_pub.publish(Twist())
                self._hold_rl()
                time.sleep(0.05)
            self._state_pub.publish(String(
                data=self._event("CORRIDOR_ENTRY_REACHED")))
            self._write_handoff(
                self._event("GT_CORRIDOR_PREPOSITIONED"),
                truth_pose=list(corridor_pose), target=list(route[-1]["target"]))
            self._write_truth_guide_result(
                self._event("GT_GUIDE_COMPLETE"),
                stage="prepositioned_corridor", route=route, trace=trace,
                target=list(route[-1]["target"]), localization_guard_armed=True)
            return True
        for waypoint in route:
            stage = waypoint["stage"]
            target_x, target_y = waypoint["target"]
            self._state_pub.publish(String(data=(
                self._event("GT_STAIR_EXIT_GUIDE") if stage == "upper_stair_exit"
                else self._event("GT_CORRIDOR_ENTRY_GUIDE"))))
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
            stage_started = time.monotonic()
            stage_started_sim = float(rospy.Time.now().to_sec())
            stage_deadline = stage_started + stage_timeout
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
                    now_sim = float(rospy.Time.now().to_sec())
                    trace.append({
                        # Route motion and all user-facing subflow timing use
                        # ROS/Gazebo time.  Wall time remains an explicitly
                        # named watchdog diagnostic only.
                        "t": round(max(0.0, now_sim-guide_started_sim), 3),
                        "ros_sim_t": round(max(
                            0.0, now_sim-guide_started_sim), 3),
                        "wall_t": round(time.monotonic()-guide_started, 3),
                        "stage": stage, "x": pose[0], "y": pose[1],
                        "z": pose[2], "yaw": pose[3],
                        "target": [target_x, target_y],
                        "distance": distance,
                        "stage_timeout_sec": stage_timeout})
                    last_trace = time.monotonic()
                landing_exit_valid = False
                if (stage == "upper_stair_exit" and
                        waypoint.get("landing_accept") and
                        self._floor_number >= 2):
                    # On F3 the robot can remain on the broad upper landing
                    # after the stair controller releases ownership; the
                    # nominal east-edge target may be unreachable because of
                    # the landing collision seam.  Treat any stable pose
                    # inside the landing as a valid handoff, then continue to
                    # the corridor guide.
                    bounds = waypoint.get("stair_bounds", {})
                    # The first F3 waypoint is deliberately reachable from
                    # the actual landing pose. Do not promote an east-edge
                    # crossing by itself: that crossing can occur while the
                    # robot is still at the stair throat and makes the next
                    # diagonal waypoint collide with the seam.
                    landing_exit_valid = bool(
                        float(bounds.get("x_min", -math.inf)) - 0.15 <= pose[0] <=
                        float(bounds.get("x_max", math.inf)) + 0.15 and
                        abs(float(pose[0]) - float(target_x)) <= 1.05 and
                        target_y - 0.65 <= pose[1] <= target_y + 0.65)
                corridor_arrival_valid = bool(
                    waypoint.get("corridor_bounds") and
                    self._truth_corridor_arrival_valid(
                        pose, waypoint, self._truth_guide_tolerance))
                # The north-clear leg can enter the corridor away from its
                # edge-biased nominal x target.  Accept the same region gate
                # used by the final corridor waypoint instead of timing out on
                # Euclidean point distance after a physically valid ingress.
                north_corridor_arrival = bool(
                    stage == "upper_landing_north_clear" and
                    self._truth_corridor_arrival_valid(
                        pose, route[-1], 0.0))
                point_arrival = bool(
                    distance <= self._truth_guide_tolerance and
                     self._truth_corridor_arrival_valid(
                         pose, waypoint, self._truth_guide_tolerance))
                if stage == "upper_landing_east_clear":
                    # The ordinary 0.6 m waypoint tolerance is unsafe at the
                    # stair-well edge.  Require the body centre to be well on
                    # the east slab before beginning any northbound motion.
                    # The nominal target is near the middle of the broad
                    # east slab.  Requiring 0.20 m exact convergence caused
                    # run132 to stall at x=-1.24 despite the body centre
                    # already being east of the support line.  run136 stopped
                    # at x=-1.319 (0.4196 m from the target), only 2 cm outside
                    # that old region.  A 0.45 m region accepts the supported
                    # slab while still rejecting run131's unsafe x=-1.44
                    # release.
                    point_arrival = self._east_landing_clear_arrival(
                        pose, target_x, target_y,
                        self._truth_guide_tolerance)
                if (point_arrival or
                        landing_exit_valid or corridor_arrival_valid or
                        north_corridor_arrival):
                    self._cmd_pub.publish(Twist())
                    if north_corridor_arrival:
                        self._write_handoff(
                            self._event("GT_CORRIDOR_REGION_REACHED"),
                            stage=stage, truth_pose=list(pose),
                            minimum_ingress_y=route[-1].get(
                                "minimum_ingress_y"))
                    break
                if distance <= best_distance - self._truth_guide_progress_epsilon:
                    best_distance = distance
                    last_progress = time.monotonic()
                stage_elapsed_sim = max(
                    0.0, float(rospy.Time.now().to_sec())-stage_started_sim)
                low_rate_stalled = self._truth_route_low_rate_stalled(
                    initial_distance, best_distance,
                    stage_elapsed_sim,
                    self._truth_guide_low_rate_window,
                    # This gate detects a persistent crawl, not merely motion
                    # below the conservative distance-budget speed.  The
                    # independent wall-clock no-progress watchdog above still
                    # stops a genuinely stationary robot.  run178 progressed
                    # steadily at about 0.087 m/s simulation speed but was
                    # falsely aborted because low RTF reduced its wall rate.
                    0.5*self._truth_guide_minimum_progress_speed)
                if ((time.monotonic() - last_progress >=
                     self._truth_guide_stall_timeout) or low_rate_stalled):
                    self._cmd_pub.publish(Twist())
                    self._write_handoff(
                        self._event("GT_GUIDE_STALLED"), stage=stage,
                        target=[target_x, target_y], trace=trace,
                        best_distance=best_distance,
                        stall_timeout_sec=self._truth_guide_stall_timeout,
                        low_rate_elapsed_ros_sim_s=stage_elapsed_sim,
                        low_rate_minimum_speed_mps=(
                            0.5*self._truth_guide_minimum_progress_speed),
                        wall_no_progress_elapsed_sec=(
                            time.monotonic()-last_progress))
                    # The door-observation seed is a corridor-ingress
                    # bootstrap, not a mission objective.  On F2 the plane
                    # controller can settle at the valid corridor boundary
                    # (as happened at y~=11.8) while its yaw deadband prevents
                    # the last 1--2 m to the seed.  Do one bounded truth
                    # handoff to the seed and return control to the normal
                    # online doorway/room state machine.  Without this, the
                    # guide watchdog incorrectly terminates the entire floor
                    # before any door candidate can be scheduled.  This does
                    # not create a room or hazard observation.
                    if (self._floor_number >= 2 and
                            stage.endswith("_door_observation_seed") and
                            self._truth_reposition_to_corridor(
                                (target_x, target_y), pose)):
                        trace.append({"stage": "truth_door_seed_handoff",
                                      "x": pose[0], "y": pose[1],
                                      "z": pose[2], "target": [target_x,
                                                                  target_y]})
                        self._state_pub.publish(String(
                            data=self._event("CORRIDOR_ENTRY_REACHED")))
                        self._write_truth_guide_result(
                            self._event("GT_GUIDE_COMPLETE"),
                            stage="truth_door_seed_handoff", route=route,
                            trace=trace, target=[target_x, target_y],
                            localization_guard_armed=True)
                        return True
                    # Upper-floor stair cores can have a collision seam
                    # between the landing and the corridor. Cross it once with
                    # the offline truth pose, then continue through the same
                    # corridor-entry validation used by normal navigation.
                    if (self._floor_number >= 2 and self._floor_number < 3 and
                            stage in ("upper_stair_exit",
                                      "upper_landing_east_step",
                                      "upper_landing_east_clear",
                                      "upper_landing_north_clear") and
                            self._truth_reposition_to_corridor(
                                route[-1]["target"], pose)):
                        trace.append({"stage": "truth_seam_handoff",
                                      "x": pose[0], "y": pose[1], "z": pose[2],
                                      "target": list(route[-1]["target"])})
                        if self._floor_number >= 3:
                            # F3 arrives here with an acknowledged RL policy.
                            # Do not perform another local controller reset:
                            # RESET re-enters FixedStand and clears exactly the
                            # readiness latch which made the upper-landing
                            # handoff safe. Hold the live RL controller while
                            # the truth/LIO corridor seam is being rebased.
                            self._state_pub.publish(String(data=self._event(
                                "F3_SEAM_RL_HOLD")))
                        # The seam handoff bypasses the normal final heading
                        # loop, so explicitly arm FAST-LIO's upper-floor
                        # truth guard before the executor creates its first
                        # goal.  Without these tokens the LIO/map frame stays
                        # on the stair's pre-climb anchor and the first F3
                        # frontier is issued in stale coordinates.
                        self._state_pub.publish(String(
                            data=self._event("LOCALIZATION_STABILIZING")))
                        settle_deadline = time.monotonic() + 0.35
                        while (not rospy.is_shutdown() and
                               time.monotonic() < settle_deadline):
                            self._cmd_pub.publish(Twist())
                            self._hold_rl()
                            time.sleep(0.05)
                        self._state_pub.publish(String(
                            data=self._event("CORRIDOR_ENTRY_REACHED")))
                        self._write_truth_guide_result(
                            self._event("GT_GUIDE_COMPLETE"),
                            stage="truth_seam_handoff", route=route,
                            trace=trace, target=list(route[-1]["target"]),
                            localization_guard_armed=True)
                        return True
                    self._write_truth_guide_result(
                        self._event("GT_GUIDE_STALLED"), stage=stage,
                        target=[target_x, target_y], route=route, trace=trace,
                        best_distance=best_distance,
                        stall_timeout_sec=self._truth_guide_stall_timeout,
                        low_rate_elapsed_ros_sim_s=stage_elapsed_sim,
                        low_rate_minimum_speed_mps=(
                            0.5*self._truth_guide_minimum_progress_speed),
                        wall_no_progress_elapsed_sec=(
                            time.monotonic()-last_progress))
                    return False
                heading = math.atan2(dy, dx)
                heading_error = math.atan2(
                    math.sin(heading - pose[3]), math.cos(heading - pose[3]))
                speed = min(self._truth_guide_speed,
                            max(0.30, 0.75 * distance))
                # The third-floor plane handoff has the opposite yaw convention; the
                # second-floor controller retains the original convention.
                yaw_sign = 1.0
                yaw_rate=max(-0.40, min(0.40, yaw_sign * 0.9 * heading_error))
                # On F3 the narrow east bridge is traversed sideways from
                # the post-stair heading.  Simultaneous yaw commands make the
                # learned flat policy roll against the landing edge; preserve
                # heading for those two short physical bridge legs and turn
                # only once the supporting lobby slab is reached.
                if (stage in ("upper_landing_east_step",
                              "upper_landing_east_clear") or
                        (stage.startswith("upper_landing_north_step_") and
                         pose[1] <= float(waypoint.get(
                             "opening_clear_y", -math.inf)))):
                    yaw_rate = 0.0
                # Keep translating toward the truth waypoint while yaw is
                # converging.  Requiring a pure-yaw phase here can deadlock at
                # the F3 landing seam: the plane policy turns only a few
                # degrees, the progress watchdog fires, and corridor entry is
                # never reached.  World-frame conversion keeps this bounded
                # diagonal motion collision-safe while the yaw command aligns.
                self._publish_truth_world_command(
                    speed * dx / max(distance, 1e-6),
                    speed * dy / max(distance, 1e-6), yaw_rate)
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
        try:
            ros_sim_time = float(rospy.Time.now().to_sec())
        except (rospy.ROSException, TypeError, ValueError):
            ros_sim_time = None
        payload = {
            "schema": "simenv_{}_handoff_v1".format(self._floor_slug),
            "state": state,
            "wall_time": time.time(),
            "ros_sim_time_sec": ros_sim_time,
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
        # The latest-state JSON remains convenient for launch handoff, while
        # the append-only timeline preserves every subflow boundary for the
        # post-run ROS-time report.
        history_path = os.path.join(
            self._parent_output,
            "{}_handoff_history.jsonl".format(self._floor_slug))
        with open(history_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False,
                                    sort_keys=True) + "\n")

    @staticmethod
    def _handoff_wait_expired(started, handoff_started,
                              pre_handoff_timeout, handoff_timeout, now):
        """Apply the active timeout only after stair ownership transfers."""
        timeout = (float(handoff_timeout) if handoff_started is not None else
                   float(pre_handoff_timeout))
        anchor = handoff_started if handoff_started is not None else started
        return bool(timeout > 0.0 and float(now) - float(anchor) >= timeout)

    @staticmethod
    def _terminal_stair_failure_state(state):
        """Recognize latched terminal failures from a finite stair worker."""
        token = str(state or "").upper()
        return bool(token and token.endswith((
            "_FAILED", "_ALIGNMENT_LOST", "_FALL_DETECTED", "_TIMEOUT",
            "_NOT_REACHED")))

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
            # Stair workers are intentionally non-required because they exit
            # cleanly after a successful handoff.  On failure they publish a
            # latched terminal phase immediately before exiting.  Propagate
            # that explicit failure now instead of leaving the next-floor
            # manager in its 1200 s handoff wait (f3portalstop).
            if self._terminal_stair_failure_state(stair_state):
                self._write_handoff(
                    self._event("STAIR_TRANSITION_FAILED"),
                    stair_state=stair_state,
                    stair_handoff_started=handoff_started is not None)
                return False
            # A source-floor exploration may legitimately outlive the total
            # launch age. Its own failure topic remains authoritative. Once
            # stair ownership transfers, give the physical transition its
            # independent bounded window.
            if self._handoff_wait_expired(
                    started, handoff_started, self._pre_handoff_timeout,
                    self._handoff_timeout, time.monotonic()):
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
        if self._verified_stair_plane_handoff_reused:
            # The stair manager's plane-policy ownership was checked before
            # LANDING_RESET and deliberately preserved.  Do not repeat the
            # dwell here or issue a second controller transition.
            self._state_pub.publish(String(
                data=self._event("EXPLORATION_READY")))
            return True
        if self._verified_stair_plane_handoff_ready():
            return True
        # F1/F2 retain the established FixedStand -> plane-policy sequence.
        # On F3, local RESET has already applied a level platform pose plus
        # the controller's flat-ground joint configuration. FixedStand can
        # remain wedged in its stair-angle interpolation even though that
        # reset is safe, so request the plane policy directly from this unique
        # post-stair state instead of blocking exploration on a dead phase.
        if self._floor_number < 3:
            if not self._recover_fixed_stand():
                return False
        else:
            self._cmd_pub.publish(Twist())
            # Do not bypass FixedStand after a physical F2->F3 climb.  The
            # direct-RL experiment released a still-spinning stair gait onto
            # the one-metre landing (run111: 1.22 m/s, 8.25 rad/s), so it
            # drifted before the sequential corridor guide could begin.
            # Retain the old bypass only as an explicit diagnostic opt-in.
            if (getattr(self, "_f3_physical_corridor_guide", False) and
                    bool(rospy.get_param("~allow_unsafe_f3_direct_rl_handoff",
                                         False))):
                self._state_pub.publish(String(data=self._event("PLANE_POLICY_DIRECT_HANDOFF")))
                self._policy_pub.publish(String(data=self._plane_policy))
                # Do not leave the learned plane policy at a zero external
                # command for seconds: after F2->F3 it can still carry a
                # forward gait phase and drift the body off the small landing.
                # This is only a controller-ownership pulse; the physical
                # truth velocity guide takes over immediately afterwards.
                arm_until = time.monotonic() + 0.75
                while not rospy.is_shutdown() and time.monotonic() < arm_until:
                    self._cmd_pub.publish(Twist())
                    self._hold_rl()
                    time.sleep(0.05)
                with self._lock:
                    self._direct_f3_rl_keepalive = True
                self._direct_f3_ready_pub.publish(Bool(data=True))
                if self._direct_f3_rl_timer is None:
                    self._direct_f3_rl_timer = rospy.Timer(rospy.Duration(0.20), self._on_direct_f3_rl_keepalive)
                return self._wait_for_real_locomotion_ready(
                    timeout_sec=15.0, source="unsafe_direct_f3_handoff")
            # The post-stair reset is physically level but may retain a
            # continuous-joint representation offset. FixedStand handles that
            # bounded F3 case and supplies the required Passive -> RL edge.
            self._state_pub.publish(String(data=self._event("STAND_RECOVERY")))
            if not self._recover_fixed_stand():
                # A stair landing can be truth-levelled yet remain physically
                # wedged on the last tread.  In that case FixedStand cannot
                # converge (the observed roll/pitch and joint error stay
                # large), and letting the required F3 node return would tear
                # down the whole roslaunch.  Use one bounded truth fallback at
                # the already verified flat side of the stair throat.  This
                # is a localization/actuator recovery, not a room shortcut:
                # the normal F3 corridor guide and room state machine still
                # execute after locomotion_ready is latched.
                route = self._second_floor_truth_route() or []
                flat_anchor = None
                for item in route:
                    if item.get("stage") == "upper_landing_east_clear":
                        flat_anchor = tuple(item.get("target") or ())
                        break
                if (flat_anchor is not None and
                        len(flat_anchor) >= 2 and
                        self._recover_f3_flat_truth_anchor(flat_anchor)):
                    rospy.logwarn(
                        "[%s] F3 FixedStand failed; recovered once at flat "
                        "truth anchor (%.2f, %.2f).",
                        self._floor_slug, flat_anchor[0], flat_anchor[1])
                else:
                    rospy.logerr("[%s] F3 FixedStand and bounded flat-anchor "
                                 "recovery both failed after local reset.",
                                 self._floor_slug)
                    return False
            self._state_pub.publish(String(
                data=self._event("PLANE_POLICY_DIRECT_HANDOFF")))
            # The F2->F3 anomaly bridge can already have this exact flat
            # model active, so a duplicate request has no reload ACK. Re-arm
            # its RL latch with a bounded zero-command dwell instead of
            # waiting forever for that optional ACK.
            self._policy_pub.publish(String(data=self._plane_policy))
            # run117 already had a physically stable FixedStand and the same
            # plane policy active, yet this duplicate-policy bridge held zero
            # command for roughly six simulation seconds.  Keep a bounded
            # re-arm dwell long enough for several controller ticks; the
            # following real locomotion-ready gate remains authoritative.
            # A policy request plus the real locomotion-ready dwell below is
            # the authoritative ownership gate.  run119 spent three seconds
            # publishing the same zero/RL pulse before beginning that gate,
            # although several controller ticks are sufficient to consume
            # the request.  Keep 0.75 s (15 ticks at 20 Hz), then rely on the
            # unchanged one-second real-ready proof.
            arm_until = time.monotonic() + 0.75
            while not rospy.is_shutdown() and time.monotonic() < arm_until:
                self._cmd_pub.publish(Twist())
                self._hold_rl()
                time.sleep(0.05)
            with self._lock:
                self._direct_f3_rl_keepalive = True
            self._direct_f3_ready_pub.publish(Bool(data=True))
            if self._direct_f3_rl_timer is None:
                self._direct_f3_rl_timer = rospy.Timer(
                    rospy.Duration(0.20), self._on_direct_f3_rl_keepalive)
            if not self._wait_for_real_locomotion_ready(
                    timeout_sec=15.0, source="f3_plane_policy_restore"):
                self._write_handoff(
                    self._event("LOCOMOTION_REAL_READY_TIMEOUT"),
                    locomotion_ready=False, timeout_sec=15.0)
                return False
            self._write_handoff(
                self._event("EXPLORATION_READY"), policy_reused=True,
                locomotion_ready=bool(self._locomotion_ready),
                direct_rl_arm_seconds=0.75)
            self._state_pub.publish(String(
                data=self._event("EXPLORATION_READY")))
            rospy.loginfo(
                "[%s] F3 direct flat-policy RL latch armed; continuing "
                "without a duplicate reload acknowledgement.", self._floor_slug)
            return True
        self._state_pub.publish(String(
            data=self._event("PLANE_POLICY_LOADING")))
        # Retry a missing acknowledgement, but never reload an acknowledged
        # model while waiting for the controller gait latch.
        attempts = 3 if self._floor_number >= 2 else 1
        acknowledgement_seen = False
        for attempt in range(attempts):
            started = time.monotonic()
            last_request = -1.0
            stable_since = None
            with self._lock:
                self._plane_policy_ready = False
                self._locomotion_ready = False
            # Queue the plane model while FixedStand still owns the joints.
            # State_RL consumes this pending request on its first control tick,
            # before its smooth takeover can produce meaningful actions.
            self._policy_pub.publish(String(data=self._plane_policy))
            last_request = time.monotonic()
            while (not rospy.is_shutdown() and
                   time.monotonic() - started < self._policy_timeout):
                now = time.monotonic()
                self._cmd_pub.publish(Twist())
                self._hold_rl()
                # A request reloads the Torch model even when the path is
                # unchanged. fix-5 issued three reloads in about 1.5 seconds,
                # after which the robot fell and locomotion_ready went false.
                if (not acknowledgement_seen and
                        (last_request < 0.0 or now - last_request >= 5.0)):
                    self._policy_pub.publish(String(data=self._plane_policy))
                    last_request = now
                with self._lock:
                    ack_ready = self._plane_policy_ready
                    locomotion_ready = self._locomotion_ready
                acknowledgement_seen = bool(
                    acknowledgement_seen or ack_ready)
                # An already-active policy can omit a duplicate reload ack,
                # but only locomotion_ready proves the robot can accept goals.
                takeover_grace = 5.0
                policy_confirmed = bool(
                    acknowledgement_seen or
                    (locomotion_ready and
                     now - started >= takeover_grace))
                ready = bool(policy_confirmed and locomotion_ready)
                if ready:
                    stable_since = stable_since or now
                    if now - stable_since >= self._policy_settle:
                        self._write_handoff(
                            self._event("EXPLORATION_READY"),
                            policy_reload_duration_sec=round(
                                now - started, 3),
                            policy_reload_attempt=attempt + 1,
                            locomotion_ready=True)
                        self._state_pub.publish(String(
                            data=self._event("EXPLORATION_READY")))
                        return True
                else:
                    stable_since = None
                time.sleep(0.05)
            # Re-loading an acknowledged model cannot repair a missing gait
            # latch and was the direct cause of the fix-5 fall.
            if acknowledgement_seen:
                break
            if attempt + 1 < attempts:
                rospy.logwarn(
                    "Plane policy handshake window %d/%d expired; "
                    "retrying upper-floor takeover.",
                    attempt + 1, attempts)
                self._state_pub.publish(String(
                    data=self._event("PLANE_POLICY_RETRY")))
        self._write_handoff(
            self._event("PLANE_POLICY_TIMEOUT"),
            policy_reload_attempts=attempts,
            policy_acknowledged=acknowledgement_seen,
            locomotion_ready=False)
        return False

    def _wait_for_real_locomotion_ready(self, timeout_sec=15.0, source=""):
        """Require the low-level walking latch after an F3 handoff.

        ``direct_f3_rl_ready`` is intentionally latched as a bridge token so
        the executor can observe the handoff.  It is not actuator evidence.
        Keep the command at zero while the controller reacquires RL and only
        release this gate after a short continuous real-ready dwell.
        """
        deadline = time.monotonic() + max(1.0, float(timeout_sec))
        stable_since = None
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            self._cmd_pub.publish(Twist())
            self._hold_rl()
            with self._lock:
                ready = bool(self._locomotion_ready)
            now = time.monotonic()
            if ready:
                stable_since = stable_since or now
                if now - stable_since >= 1.0:
                    rospy.loginfo(
                        "[%s] real locomotion_ready confirmed after F3 handoff "
                        "source=%s dwell=1.0s.", self._floor_slug, source)
                    return True
            else:
                stable_since = None
            time.sleep(0.05)
        rospy.logerr(
            "[%s] real locomotion_ready did not return within %.1fs after F3 "
            "handoff source=%s; refusing exploration goals.",
            self._floor_slug, float(timeout_sec), source)
        return False

    def _recover_f3_flat_truth_anchor(self, anchor):
        """One-shot F3 recovery from a wedged stair landing.

        The stair manager has already proved the destination height.  Moving
        the simulated body to the truth-verified flat side anchor clears the
        last-tread contact that prevents FixedStand from converging.  The
        fallback is deliberately finite and requires the RL locomotion latch
        before any navigation goal can be issued.
        """
        if self._floor_number < 3 or getattr(
                self, "_f3_flat_truth_anchor_used", False):
            return False
        self._f3_flat_truth_anchor_used = True
        self._state_pub.publish(String(
            data=self._event("F3_TRUTH_FLAT_ANCHOR_RECOVERY")))
        self._write_handoff(
            self._event("F3_TRUTH_FLAT_ANCHOR_RECOVERY_START"),
            anchor=[float(anchor[0]), float(anchor[1])],
            policy="one_shot_after_fixed_stand_failure")
        if not self._reset_truth_upper_landing_pose(anchor):
            return False
        # The controller reset must be the final writer of both the model
        # pose and joint configuration.  Writing Gazebo truth after it can
        # reintroduce a tilted/colliding body between the reset and FixedStand.
        if not self._local_reset_controller_at_corridor(anchor):
            return False
        # Never infer an upright body from neutral joint feedback or a stale
        # locomotion latch.  The failed axiscontract run had joint RMS 0.073
        # rad while the base roll was 1.59 rad.  Re-enter FixedStand and reuse
        # its strict truth-relative height, roll/pitch, gravity and joint gates;
        # the caller performs the subsequent plane-policy handshake once.
        if self._recover_fixed_stand():
            self._write_handoff(
                self._event("F3_TRUTH_FLAT_ANCHOR_RECOVERED"),
                anchor=[float(anchor[0]), float(anchor[1])],
                fixed_stand_verified=True)
            return True
        self._write_handoff(
            self._event("F3_TRUTH_FLAT_ANCHOR_RECOVERY_FAILED"),
            anchor=[float(anchor[0]), float(anchor[1])],
            fixed_stand_verified=False)
        return False

    def _ensure_exploration_locomotion_ready(self):
        """Revalidate the walking latch immediately before creating F2 goals.

        Corridor guidance and localization stabilization can take longer than
        the initial stair-to-plane handoff. If the controller left RL during
        that interval, the baseline manager would spend all three failure slots
        on goals rejected in milliseconds. Recover through the same bounded
        FixedStand -> plane-policy handshake before exposing exploration goals.
        """
        with self._lock:
            ready = bool(self._locomotion_ready)
            direct_f3 = bool(self._direct_f3_rl_keepalive)
            stand_status = dict(self._fixed_stand_status or {})
        if ready:
            return True
        # The direct-F3 token is only a handoff keepalive.  A FixedStand
        # snapshot cannot substitute for the controller's real
        # /locomotion_ready edge: accepting goals while that edge is false
        # leaves the dog stationary while the manager burns bounded goal
        # retries.  Re-run the bounded policy recovery instead.
        direct_f3_stable = bool(
            self._floor_number >= 3 and direct_f3 and
            stand_status.get("stable_now", False) and
            abs(float(stand_status.get("roll", 99.0))) < 0.45 and
            abs(float(stand_status.get("pitch", 99.0))) < 0.35 and
            float(stand_status.get("joint_velocity_rms", 99.0)) < 0.35)
        if direct_f3_stable:
            rospy.logwarn("[%s] direct F3 token is present but real "
                          "locomotion_ready is false; requiring bounded "
                          "controller recovery before exploration.",
                          self._floor_slug)
        rospy.logwarn(
            "[%s] locomotion readiness was lost before exploration; "
            "re-running bounded FixedStand/plane-policy recovery.",
            self._floor_slug)
        self._write_handoff(self._event("LOCOMOTION_RECOVERY_START"))
        self._state_pub.publish(String(
            data=self._event("LOCOMOTION_RECOVERY")))
        recovered = self._restore_plane_policy()
        if recovered:
            self._write_handoff(self._event("LOCOMOTION_RECOVERED"),
                                locomotion_ready=True)
        return recovered

    @staticmethod
    def _unified_room_profile_defaults():
        """Return the accepted F1 room contract for a fresh floor manager.

        A normal full-flow launch can copy these values from the live F1
        manager.  The F2-stair-to-F3 integration fixture intentionally omits
        that process, however, and used to fall back to the conservative
        constructor defaults (including one physical view).  Keep the shared
        contract self-contained so an isolated upper-floor run exercises the
        same room semantics as the full mission.
        """
        return {
            "room_visit_budget_seconds": 50.0,
            "room_mission_entry_minimum_remaining_seconds": 10.0,
            "room_entry_timeout_seconds": 22.0,
            "room_entry_retry_limit": 2,
            "room_entry_preflight_max_path_m": 5.50,
            "room_entry_preflight_max_detour_ratio": 2.00,
            "room_entry_progress_timeout_seconds": 5.0,
            "room_budget_starts_after_entry": True,
            "room_exit_reserve_seconds": 5.0,
            "room_direct_verified_exit": True,
            "room_adaptive_minimal_viewpoints": True,
            "room_adaptive_maximum_viewpoints": 2,
            "room_adaptive_prefer_central_first": True,
            "room_adaptive_minimum_gain_m2": 0.12,
            "room_adaptive_path_cost_weight": 0.32,
            "room_coverage_minimum_ratio": 0.90,
            "room_coverage_maximum_unknown_m2": 2.00,
            "room_coverage_maximum_shadow_m2": 1.25,
            "room_visual_require_physical_center": False,
            "room_visual_deepening_enabled": True,
            "room_visual_deepening_budget": 4,
            "room_visual_deepening_min_completed_rooms": 0,
            "room_visual_breadth_lateral_m": 4.25,
            "room_visual_breadth_min_baseline_m": 3.60,
            "room_obstacle_second_view_near_door": True,
            "room_obstacle_front_opposite_side_pair": True,
            "room_entry_depth": 1.50,
            "room_exit_opposite_door_revisit_time_s": 40.0,
            "room_doorway_width_min": 0.58,
            "room_prefer_direct_anchored_entry": True,
            "room_post_entry_path_clearance": 0.20,
            "room_side_goal_depth": 2.00,
            "room_maximum_side_lateral": 5.00,
            "room_maximum_center_depth": 2.20,
            "room_maximum_side_depth": 2.80,
            "room_depth_probe_range": 7.50,
            "room_open_preferred_deep_depth_m": 6.70,
            "room_open_minimum_deep_depth_m": 6.25,
            "room_camera_full_spin_per_view": True,
            "room_visual_two_pose_full_sweep_enabled": True,
            "room_minimum_goal_separation": 1.80,
            "room_camera_sweep_angle_rad": 2.967059728,
            "room_visual_full_sweep_angle_rad": 2.967059728,
            "room_visual_secondary_sweep_angle_rad": 2.967059728,
            "room_camera_sweep_angular_speed": 2.10,
            "room_camera_secondary_sweep_angular_speed": 2.10,
            "room_camera_sweep_max_per_room": 2,
            "room_two_pose_waypoint_simplification": True,
            "room_two_pose_scan_lite_enabled": True,
        }

    def _copy_first_floor_parameters(self):
        # The YAML is loaded under /baseline_exploration_manager.  Reuse its
        # validated online parameters without sharing any runtime state.
        base = rospy.get_param("/baseline_exploration_manager", {})
        if isinstance(base, dict):
            for key, value in base.items():
                private = "~" + str(key)
                if not rospy.has_param(private):
                    rospy.set_param(private, value)
        if self._unified_fast_four_room_profile:
            # Shared room-policy keys must win over old YAML/private defaults.
            # Prefer the actual F1 launch value when that manager exists;
            # otherwise use the identical accepted launch default above.
            source = base if isinstance(base, dict) else {}
            inherited = 0
            fallback = 0
            effective = {}
            for key, default in self._unified_room_profile_defaults().items():
                if key in source:
                    value = source[key]
                    inherited += 1
                else:
                    value = default
                    fallback += 1
                rospy.set_param("~" + key, value)
                effective[key] = value
            rospy.logwarn(
                "[%s] unified room contract applied: inherited=%d, "
                "fallback=%d, two_pose=%s, separation=%.2f, "
                "open_depth=%.2f/%.2f, obstacle_pair=%s.",
                self._floor_slug, inherited, fallback,
                effective["room_visual_two_pose_full_sweep_enabled"],
                float(effective["room_minimum_goal_separation"]),
                float(effective["room_open_preferred_deep_depth_m"]),
                float(effective["room_open_minimum_deep_depth_m"]),
                effective["room_obstacle_front_opposite_side_pair"])
        rospy.set_param("~output_dir", self._output)
        # The copied F1 parameter namespace contains floor_number=1. Keep
        # the fresh manager's actual floor so upper-floor startup can request
        # a latched voxel projection refresh after the stair rebase.
        rospy.set_param("~floor_number", self._floor_number)
        if self._second_floor_elevation is not None:
            # Baseline safety gates may receive either a floor-local mapper z
            # or Gazebo world z when truth-degraded localization is active.
            # Publish the authoritative floor elevation so they can normalize
            # the latter before applying flat-body limits.
            rospy.set_param("~current_floor_elevation_m",
                            float(self._second_floor_elevation))
        rospy.set_param("~room_id_prefix", "floor_{}_estimated_room_".format(
            self._floor_number))
        # Use the same deep-then-near open-room contract on every floor.  The
        # former F3-only near-first override left the robot parked at the
        # shallow pose when a stale endpoint rejected the deep G4.  Recovery
        # then had to plan a long route through the least mature part of the
        # fresh F3 map.  Deep-first traverses the verified clear centreline
        # while the ENTRY geometry is still fresh; the short near-door G4 and
        # EXIT remain bounded even if later map updates become conservative.
        rospy.set_param("~room_truth_open_near_first", False)
        rospy.set_param("~stair_handoff_on_corridor_exit", True)
        if self._unified_fast_four_room_profile:
            # The isolated 140 s harness stops at the fourth EXIT, but a
            # multi-floor run must still return to its explicit corridor
            # ingress before the next stair may take ownership.
            rospy.set_param("~stop_immediately_after_all_room_exits", False)
        if self._corridor_forward_heading_hint is not None:
            rospy.set_param(
                "~corridor_forward_heading_hint_rad",
                float(self._corridor_forward_heading_hint))
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
        for key, value in self._second_floor_only_parameter_overrides(self._unified_fast_four_room_profile).items():
            rospy.set_param("~" + key, value)
        if self._floor_number == 2 and self._unified_fast_four_room_profile:
            # Preserve the shared straight-line EXIT speeds.  The former
            # 0.75/0.80 m/s F2 override predated sparse EXIT replay, exact
            # portal alignment and the dense live-3D chord audit, and made
            # every upper-floor egress 16--22 simulated seconds in run87.
            # Keep only the proven F2 combined-load envelope: lateral and yaw
            # remain conservative while an aligned portal chord can use the
            # common speed (and the executor's degraded-speed cap).
            rospy.set_param("~room_exit_lateral_speed_limit", 0.30)
            rospy.set_param("~room_fast_exit_lateral_speed_limit", 0.35)
            rospy.set_param(
                "~room_exit_blended_turn_maximum_yaw_rate", 0.55)
            # After the near doorway pair, landingatomic spent 50.6 simulated
            # seconds on eleven consecutive 1.2 m truth probes before it
            # reached the far pair.  Door callbacks still preempt every probe
            # and each segment is independently collision checked, so use the
            # same bounded 3 m acquisition stride already intended for F3.
            rospy.set_param(
                "~upper_floor_truth_corridor_probe_distance_m", 3.0)
            rospy.set_param(
                "~upper_floor_truth_corridor_probe_prescan", False)
        if self._floor_number >= 3:
            # F3 clears and rebuilds its projection slice at corridor ingress.
            # In the shared fast profile one 3 m, door-preemptible probe
            # reaches the near doorway observation band without paying a
            # stationary 180-degree scan before every probe.  The legacy
            # profile retains its shorter, scan-first acquisition sequence.
            probe_distance = (3.0 if self._unified_fast_four_room_profile else
                              float(rospy.get_param(
                                  "~upper_floor_truth_corridor_probe_distance_m",
                                  1.20)))
            probe_prescan = (False if self._unified_fast_four_room_profile else
                             bool(rospy.get_param(
                                 "~upper_floor_truth_corridor_probe_prescan",
                                 True)))
            rospy.set_param(
                "~upper_floor_truth_corridor_probe_distance_m", probe_distance)
            rospy.set_param(
                "~upper_floor_truth_corridor_probe_prescan", probe_prescan)
            rospy.set_param(
                "~upper_floor_truth_corridor_probe_prescan_angle_rad", math.pi)
            rospy.set_param(
                "~upper_floor_truth_corridor_probe_prescan_every", 1)
            # F3's terminal is about 21 m beyond the near doorway row.  The
            # shared 8 m retrace window can recover only the far pair and can
            # never revisit a near-row portal that failed while the fresh map
            # was immature.  Keep F2 unchanged and allow one F3-only bounded
            # pass spanning both physical doorway rows.
            rospy.set_param(
                "~terminal_missing_room_retrace_max_distance_m", 24.0)
            # F3 run12 reached the first deep G3 with vx ~= 1.90 m/s while
            # turning across a diagonal baseline. The body height then
            # dropped by 0.22 m, the fall guard left RL and all later goals
            # reported locomotion_not_ready. Limit only F3 room manoeuvres;
            # corridor travel and the proven F1/F2 policies stay untouched.
            # These caps are a physical stability envelope, not stale
            # throughput tuning, and therefore remain active in unified mode.
            # run116 completed every F3 contract at the former 0.75 m/s
            # blanket cap, but the same cap also throttled straight chords
            # that had already passed generated-room/furniture and dense
            # live-3D audits.  Keep the proven lateral/yaw envelope below and
            # raise only forward translation to the same executor-capped
            # 1.35 m/s envelope that completed every F1/F2 room in run154.
            # This remains well below the unstable 1.90 m/s run12 experiment;
            # F3 keeps its tighter lateral/yaw limits and every chord still
            # requires generated-furniture plus dense live-3D validation.
            rospy.set_param("~room_entry_speed", 1.35)
            rospy.set_param("~room_open_transition_speed", 1.35)
            rospy.set_param("~room_lateral_speed_limit", 0.25)
            rospy.set_param("~room_heading_gain", 0.75)
            rospy.set_param("~room_maximum_yaw_rate", 0.50)
            rospy.set_param("~room_goal_distance_gain", 1.00)
            rospy.set_param("~room_exit_speed", 1.35)
            rospy.set_param("~room_exit_lateral_speed_limit", 0.25)
            rospy.set_param("~room_fast_exit_approach_speed", 1.35)
            rospy.set_param("~room_fast_exit_lateral_speed_limit", 0.30)
            # The freshly reset F3 rolling slice often exposes about 5.1 m of
            # an otherwise 8.4 m-deep open room on the first scan.  Keep the
            # preferred target at the shared 6.7 m depth, but accept a bounded
            # 5.75 m configured minimum so that the deepest live/A*-verified
            # point is executed instead of being replaced by a stationary map
            # refresh.  The final contract still requires a >=1.8 m physical
            # baseline and a distinct near-door view.
            rospy.set_param("~room_open_minimum_deep_depth_m", 5.75)
        # F2/F3 inherit the same F1 doorway geometry and viewpoint policy (including room_entry_depth=1.50 from the launch file):
        # the deep G1/G2 observations and full two-pose RGB-D sweeps must be
        # identical on every floor; only F3 execution speed is capped above so a red ball cannot be missed merely
        # because the room was entered more shallowly on an upper floor.
        # The former F3 shallow-portal override (0.80/0.60 m) was removed so
        # F3 does not stop just past the doorway.

    @staticmethod
    def _second_floor_only_parameter_overrides(unified_fast_profile=False):
        """Return geometry guards scoped to the fresh F2 manager instance.

        These do not mutate the YAML namespace used by the already completed
        first-floor manager and do not participate in stair control.
        """
        overrides = {
            # The F2 local reset enters a narrow, obliquely aligned corridor.
            # Run36 fell before the first doorway while the generic fast
            # corridor controller sustained a diagonal command. Keep room
            # manoeuvres unchanged, but use a bounded centreline envelope
            # while the corridor map is being reacquired.
            # Run114's local detector confirmed the first doorway pair from
            # t=14 s, but the fresh manager did not establish its corridor
            # frame until t=112 s and consequently rejected those doors as
            # rearward.  The truth handoff has already placed this process at
            # a verified corridor ingress and supplied a live-odometry
            # heading. Use that line only to establish the online station
            # frame after 3.5 m of measured forward travel, then require one
            # measured 1.50 m commit before normal doorway takeover.  The old
            # 0.35 m value was smaller than the navigation arrival tolerance,
            # so each nominal commit moved only ~0.1 m and was immediately
            # reported successful, creating an unbounded same-door loop.
            "planner_timeout": 8.0,
            # The truth handoff leaves F2 in a narrow, obliquely registered
            # corridor. Run7 repeatedly fell while a 5 m guide probe drove
            # at ~1.39 m/s. Use the F3 bounded body-speed envelope here;
            # this is a stability guard, not a perception shortcut.
            "corridor_search_speed": 0.65,
            # Search/door takeover remains conservative, but a corridor
            # centreline transaction that already passed truth, 2-D and live
            # 3-D audits need not inherit the diagonal-search cap. This is
            # selective straight-line acceleration; lateral/yaw limits below
            # remain unchanged.
            "corridor_transit_speed": 1.10,
            "corridor_lateral_speed_limit": 0.10,
            "corridor_maximum_yaw_rate": 0.60,
            "room_entry_speed": 1.00,
            "room_lateral_speed_limit": 0.25,
            "room_heading_gain": 0.75,
            "room_maximum_yaw_rate": 0.50,
            "room_goal_distance_gain": 1.00,
            "room_exit_speed": 0.95,
            "room_exit_lateral_speed_limit": 0.25,
            "room_fast_exit_approach_speed": 1.00,
            "room_fast_exit_lateral_speed_limit": 0.30,
            "corridor_guided_entry_door_context": True,
            # The truth guide hands this manager over at y~=12.8 m, only
            # about 2 m before the near doorway row (y~=14.9 m).  Requiring
            # another 3.5 m of manager-local travel therefore armed doorway
            # takeover at y~=16.6 m, after both near rooms were already
            # behind the robot.  The guide has already proved corridor
            # membership and heading; retain a measured 0.50 m anti-lobby
            # displacement, while the independent 1.50 m forward commit
            # below still prevents a same-door zero-motion loop.
            "corridor_guided_entry_minimum_progress_m": 0.50,
            "corridor_door_forward_commit_distance_m": 1.50,
            "local_door_minimum_semantic_width_m": 0.60,
            # run68 physically returned along the exact ENTRY A* trace, but a
            # post-room map correction shifted the estimated door plane by
            # enough to report corridor_side_not_confirmed.  Accept only this
            # bounded, endpoint-matched reverse traversal on fresh upper-floor
            # managers.  The baseline/F1 default remains false.
            "room_accept_reversed_entry_trace_exit": True,
            "room_reversed_entry_trace_exit_tolerance_m": 0.30,
            # Floor 1 needs an additional corridor-depth margin because its
            # station-band proof tolerates accumulated drift. Fresh upper
            # floors use the exact physically traversed ENTRY endpoint as a
            # bounded exit proof, so do not inherit the F1-only extension.
            "room_reversed_entry_trace_corridor_extension_m": 0.0,
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
            # A truth-axis probe grows only the immediately adjacent unknown
            # map slice.  It must not inherit the ordinary 3.5/7 m corridor
            # transit distance: fix07 turned that long chord into a repeated
            # 33-waypoint A* audit while the robot remained stationary.
            "upper_floor_truth_corridor_probe_distance_m": 1.20,
            # fix_19 proved that the 7 m chord is not preempted quickly enough:
            # the opposite doorway was promoted at t=144 s but navigation only
            # stopped 20 s later, 4.9 m beyond its station.  Replan every 3.5 m
            # on fresh upper floors so paired rooms are handled at their row
            # while cutting planning overhead versus the older 2.5 m step.
            "corridor_forward_distance": 3.5,
            "corridor_far_pair_search_maximum_advance_m": 3.5,
            # At about 27 m online outbound progress the robot is past the
            # far room row (row-1 stations ~8-11 m, row-2 ~21-24 m).  Start
            # the reverse pass only there, or when only 25 s remain, so the
            # outward pass always reaches and enters the far door pair first.
            # 25 s reserve matches the tightened F1 profile so all three
            # floors share the same 650 s / four-room timing.
            "corridor_partial_return_minimum_outbound_progress_m": 27.0,
            "corridor_partial_return_reserve_seconds": 25.0,
            # Leave enough budget for ENTRY, immediate EXIT, and stair handoff.
            "corridor_partial_return_room_minimum_remaining_seconds": 25.0,
            # The 150 s hard phase deadline is an F1 mission contract. Upper
            # floors retain their independent 75 s stair-reserve trigger.
            "enforce_room_phase_target_deadline": False,
            # A truth seam handoff can leave the old FAST-LIO registration
            # invalid for several scans. Allow this fresh upper-floor manager
            # to continue from its verified truth corridor.
            "allow_truth_exploration_when_registration_lost": True,


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
            # The old 14 s guard predated sparse verified traces and bounded
            # truth portal restaging.  run120's F3 deep-room EXIT reached its
            # first anchor, then spent another full 14 s stalled at the jamb
            # before the same restage completed in 5.2 s.  Ten seconds still
            # covers the measured upper-floor alignment/translation legs;
            # no-progress now fails promptly into that physical, bounded
            # restage instead of consuming the room budget.
            "room_exit_progress_timeout_seconds": 10.0,
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
            # The explicit GT corridor ingress places the broad lobby
            # behind the fresh manager. In run108 the frame was not latched
            # until the first physical door row, so the two real doors landed
            # at stations +0.173 m and -0.047 m. Admit a small origin
            # calibration band while still rejecting the rear lobby throat
            # near station -4 m.
            "first_room_minimum_corridor_station_m": -0.25,
            # The second-floor process regenerates the combined root figures
            # only after its own result files have been fully persisted.
            "auto_generate_visualization": False,
        }

        if unified_fast_profile:
            # Preserve upper-floor perception/localization guards, but let the
            # run40 shared profile own throughput and deadline parameters.
            # After the near pair, doorway callbacks can preempt the same
            # continuous far-row corridor goal as on floor 1.  Do not arm the
            # historical 27 m best-effort return: strict completion must keep
            # searching until all configured rooms have a physical EXIT.
            # The upper-floor corridor guide, policy-ready gate and stationary
            # map-fill hold have completed before this fresh manager starts.
            # Do not keep the legacy whole-floor 0.65 m/s profile afterward:
            # F2 inherits the validated shared envelope.  F3's measured
            # post-stair room caps are explicitly re-applied above.
            for key in (
                    "corridor_search_speed",
                    "corridor_transit_speed",
                    "corridor_lateral_speed_limit",
                    "corridor_maximum_yaw_rate",
                    "room_entry_speed",
                    "room_lateral_speed_limit",
                    "room_heading_gain",
                    "room_maximum_yaw_rate",
                    "room_goal_distance_gain",
                    "room_exit_speed",
                    "room_exit_lateral_speed_limit",
                    "room_fast_exit_approach_speed",
                    "room_fast_exit_lateral_speed_limit",
                    "corridor_forward_distance",
                    "corridor_far_pair_search_maximum_advance_m",
                    "corridor_partial_return_minimum_outbound_progress_m",
                    "corridor_partial_return_reserve_seconds",
                    "corridor_partial_return_room_minimum_remaining_seconds",
                    "enforce_room_phase_target_deadline",
                    "terminal_return_blocked_by_known_unvisited_door"):
                overrides.pop(key, None)
        # A repeated no-frontier/footprint-blocked loop must release the
        # continuation pipeline even when local recovery goals keep resetting
        # the ordinary consecutive-failure counter.  Three physical EXITs are
        # the minimum for a partial next-floor handoff; fewer exits terminate
        # explicitly and are never reported as a completed floor.
        overrides["corridor_stall_watchdog_seconds"] = 45.0
        overrides["corridor_stall_partial_return_minimum_exited_rooms"] = 3
        overrides["corridor_stall_abort_minimum_exited_rooms"] = 2
        overrides["corridor_stall_minimum_outbound_progress_m"] = 20.0
        return overrides

    def _map_second_floor_lobby(self):
        # The 360-degree lobby rescan is disabled on upper floors because it
        # can tip the dog over on the narrow stair landing (run34).  But an
        # upper-floor corridor that has just been entered has almost no
        # mapped cells yet: the voxel mapper needs a few seconds of steady
        # LiDAR from the new height before the doorway detector can see side
        # apertures.  Without this pause the corridor starts probing forward
        # immediately (truth probe every 3-5 s), the map never fills in front
        # of the dog, and no doorway is ever confirmed (run ..._230 F3 probed
        # the full 0->32 m corridor with zero confirmed doorways, then latched
        # a partial return).  Hold still -- no rotation -- and let the mapper
        # fill the current corridor slice before exploration begins.
        if self._floor_number >= 2:
            if not self._enable_lobby_rescan:
                if self._floor_number == 2 and self._truth_fast_handoff_active:
                    rospy.logwarn(
                        "[%s] retaining verified preloaded floor map; "
                        "skipping upper-floor map clear/fill hold.",
                        self._floor_slug)
                    return
                if self._live_map_prepared_before_guide:
                    guide_duration = None
                    if (self._corridor_guide_started_ros_sim is not None and
                            self._corridor_guide_completed_ros_sim is not None):
                        guide_duration = max(
                            0.0,
                            float(self._corridor_guide_completed_ros_sim) -
                            float(self._corridor_guide_started_ros_sim))
                    # The physical ingress itself has supplied many fresh
                    # LiDAR callbacks after the clear/re-anchor.  Clearing or
                    # sleeping here would discard exactly that evidence and
                    # serialize map preparation behind corridor motion.
                    self._write_handoff(
                        self._event("CORRIDOR_MAP_READY_FROM_INGRESS"),
                        corridor_guide_ros_sim_sec=(
                            round(guide_duration, 3)
                            if guide_duration is not None else None),
                        map_fill_mode="physical_ingress_callbacks")
                    rospy.logwarn(
                        "[%s] fresh live map was populated during physical "
                        "corridor ingress; skipping duplicate clear/fill hold.",
                        self._floor_slug)
                    return
                if self._floor_number == 2:
                    # F1 voxels are not needed for F2 navigation and make each
                    # SCAN-lite body audit traverse a tree roughly three times
                    # larger than the standalone F2 map. Preserve the complete
                    # pre-clear tree on disk in the mapper, then rebuild the
                    # live tree from the verified F2 corridor ingress.
                    rospy.logwarn(
                        "[%s] clearing accumulated F1 voxels before fresh "
                        "F2 corridor mapping.", self._floor_slug)
                    try:
                        self._voxel_clear_pub.publish(Float64(data=-1.0e9))
                    except rospy.ROSException:
                        pass
                    time.sleep(0.25)
                if self._floor_number >= 3:
                    # The F1/F2 octree and the F2-locked projection band
                    # pollute the F3 corridor slice and make every 3-D body
                    # audit traverse old floors.  Preserve the complete tree
                    # offline, but rebuild an empty *live* F3 tree just as F2
                    # already does.  Keeping nodes below 4.30 m was not a
                    # per-floor lifecycle; axialtruth entered F3 with more
                    # than 300k stale lower-floor nodes.
                    rospy.logwarn(
                        "[%s] clearing accumulated upper map and re-anchoring "
                        "the projection band for the fresh corridor.",
                        self._floor_slug)
                    try:
                        # -2e9 is the mapper's explicit F3 clear-all marker;
                        # it differs from F2's -1e9 only for archive naming.
                        self._voxel_clear_pub.publish(Float64(data=-2.0e9))
                        # The F2 navigation-height latch is intentionally
                        # absolute. Do not depend on raw FAST-LIO z recovering
                        # at exactly the right callback after the second
                        # stair: this handoff already has fresh, validated F3
                        # Gazebo truth, so atomically rebase the stabilized
                        # navigation stream to that measured floor height.
                        with self._lock:
                            truth = (tuple(self._truth_pose)
                                     if self._truth_pose is not None else None)
                            truth_age = (time.monotonic() -
                                         self._truth_pose_received_at)
                        if (truth is not None and truth_age <= 1.0 and
                                math.isfinite(float(truth[2]))):
                            self._navigation_height_reanchor_pub.publish(
                                Float64(data=float(truth[2])))
                        else:
                            rospy.logwarn(
                                "[%s] fresh F3 truth height unavailable for "
                                "explicit navigation re-anchor; retaining "
                                "bounded raw-odometry fallback.",
                                self._floor_slug)
                    except rospy.ROSException:
                        pass
                    # F3 has just been truth-reanchored and the first doorway
                    # transaction performs its own live prescan and 3-D
                    # preflight.  A ten-second stationary map-fill duplicated
                    # that evidence in run117.  Retain two full seconds for a
                    # fresh local slice, then let the ordinary gates decide.
                    hold_seconds = max(2.0, float(rospy.get_param(
                        "~third_floor_corridor_map_fill_hold_seconds", 2.0)))
                else:
                    hold_seconds = max(2.0, float(rospy.get_param(
                        "~second_floor_corridor_map_fill_hold_seconds", 2.0)))
                rospy.logwarn(
                    "[%s] corridor map-fill hold: waiting for the voxel mapper "
                    "to fill the entrance slice before exploration.",
                    self._floor_slug)
                fill_deadline = time.monotonic() + hold_seconds
                # Publish zero command so the RL holder keeps command
                # ownership while we wait.
                while not rospy.is_shutdown() and time.monotonic() < fill_deadline:
                    self._cmd_pub.publish(Twist())
                    self._hold_rl()
                    time.sleep(0.1)
                rospy.logwarn(
                    "[%s] corridor map-fill hold complete.", self._floor_slug)
            return
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

    def _prepare_upper_floor_live_map_for_ingress(self):
        """Clear/re-anchor once, before physical ingress populates the map.

        This operation changes only the mapper lifecycle.  It does not move
        the robot and it deliberately remains disabled for a verified F2
        preloaded map, whose manifest/mapper acknowledgement is the stronger
        source of floor-local evidence.
        """
        if self._floor_number < 2 or self._enable_lobby_rescan:
            return True
        if self._floor_number == 2 and self._truth_fast_handoff_active:
            return True
        if self._live_map_prepared_before_guide:
            return True
        marker = -1.0e9 if self._floor_number == 2 else -2.0e9
        reanchor_height = None
        try:
            self._voxel_clear_pub.publish(Float64(data=marker))
            if self._floor_number >= 3:
                with self._lock:
                    truth = (tuple(self._truth_pose)
                             if self._truth_pose is not None else None)
                    truth_age = time.monotonic() - self._truth_pose_received_at
                if (truth is not None and truth_age <= 1.0 and
                        math.isfinite(float(truth[2]))):
                    reanchor_height = float(truth[2])
                    self._navigation_height_reanchor_pub.publish(
                        Float64(data=reanchor_height))
                else:
                    rospy.logerr(
                        "[%s] cannot prepare fresh live map without a fresh "
                        "truth height.", self._floor_slug)
                    return False
        except rospy.ROSException as error:
            rospy.logerr("[%s] upper-floor map preparation failed: %s",
                         self._floor_slug, error)
            return False
        # Keep the former bounded publisher/mapper release window, but pay it
        # before motion so every subsequent LiDAR callback belongs to the new
        # floor rather than clearing a completed ingress afterward.
        release_until = time.monotonic() + 0.25
        while not rospy.is_shutdown() and time.monotonic() < release_until:
            self._cmd_pub.publish(Twist())
            self._hold_rl()
            time.sleep(0.05)
        if rospy.is_shutdown():
            return False
        self._live_map_prepared_before_guide = True
        self._write_handoff(
            self._event("LIVE_MAP_PREPARED_FOR_INGRESS"),
            voxel_clear_marker=marker,
            navigation_height_reanchor_m=reanchor_height,
            map_fill_mode="physical_ingress_callbacks")
        return True

    def _generate_combined_visualization(self, initial_delay=4.0):
        """Schedule one combined render outside the roslaunch process group."""
        if (not self._auto_generate_visualization or
                self._combined_visualization_scheduled):
            return
        if (not os.path.isfile(self._visualizer) or
                not os.path.isfile(self._visualization_launcher)):
            rospy.logerr(
                "Combined visualization scripts missing: %s %s",
                self._visualization_launcher, self._visualizer)
            return
        command = [
            sys.executable, self._visualization_launcher,
            "--run-dir", self._parent_output,
            "--visualizer", self._visualizer,
            "--initial-delay", str(max(0.0, float(initial_delay))),
        ]
        try:
            subprocess.Popen(
                command, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                close_fds=True, start_new_session=True)
            self._combined_visualization_scheduled = True
            rospy.loginfo(
                "Detached combined visualization scheduled for %s",
                self._parent_output)
        except OSError as error:
            rospy.logerr(
                "Could not schedule combined multi-floor visualization: %s",
                error)

    def _publish_exploration_failure(self, reason, state_suffix=None,
                                     **fields):
        """Publish one latched failure token without losing legacy states."""
        generic_state = self._event("EXPLORATION_FAILED")
        legacy_state = (self._event(state_suffix)
                        if state_suffix else generic_state)
        with self._lock:
            if self._terminal_outcome_published:
                return False
            if self._terminal_failure_published:
                return False
            self._terminal_failure_published = True
            self._terminal_outcome_published = True
        payload = dict(fields)
        payload.update({
            "failure_reason": str(reason),
            "failure_state": legacy_state,
            "mission_stage": self._mission_stage,
            "next_stair_handoff_authorized": False,
        })
        try:
            self._cmd_pub.publish(Twist())
        except rospy.ROSException as error:
            rospy.logerr("[%s] failed to publish stop on mission failure: %s",
                         self._floor_slug, error)
        try:
            self._write_handoff(generic_state, **payload)
        except (OSError, TypeError, ValueError) as error:
            rospy.logerr("[%s] failed to persist mission failure: %s",
                         self._floor_slug, error)
        try:
            # Preserve the existing detailed state for observers, then leave
            # the generic failure token latched so a waiting next-floor
            # process is released without ever seeing a completion trigger.
            if legacy_state != generic_state:
                self._state_pub.publish(String(data=legacy_state))
            self._state_pub.publish(String(data=generic_state))
        except rospy.ROSException as error:
            rospy.logerr("[%s] failed to publish mission failure: %s",
                         self._floor_slug, error)
        # Publish the terminal token before rendering.  The renderer may need
        # tens of wall-clock seconds; delaying the token left the required
        # descent supervisor waiting while Gazebo appeared hung in F3_FAILED.
        # Rendering is diagnostic and must never own mission termination.
        self._generate_combined_visualization()
        rospy.logerr("[%s] exploration failed at %s: %s",
                     self._floor_slug, self._mission_stage, reason)
        return False

    def _seed_baseline_corridor_entry_anchor(self, manager,
                                             timeout_sec=2.0):
        """Use Baseline's own odometry frame to fix the upper-floor ingress."""
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        anchor = None
        while not rospy.is_shutdown() and time.monotonic() <= deadline:
            with manager.lock:
                candidate = (manager.corridor_entry_anchor or
                             manager.mission_start_pose or manager.pose)
            if candidate is not None and len(candidate) >= 3:
                values = tuple(float(value) for value in candidate[:3])
                if all(math.isfinite(value) for value in values):
                    anchor = values
                    break
            time.sleep(0.02)
        if anchor is None:
            rospy.logwarn(
                "[%s] Baseline odometry was unavailable; retaining its "
                "online stair-wait anchor capture.",
                self._floor_slug)
            return None
        with manager.lock:
            if manager.corridor_entry_anchor is None:
                manager.corridor_entry_anchor = anchor
            # _stair_wait_target() prioritizes this field.  Seeding it before
            # manager.run() prevents the first long corridor goal endpoint
            # from replacing the verified ingress as the return target.
            manager.stair_wait_anchor = anchor
            history = getattr(manager, "corridor_sweep_history", None)
            if isinstance(history, list):
                history.append({
                    "event": "UPPER_FLOOR_CORRIDOR_ENTRY_ANCHOR_SEEDED",
                    "elapsed_sec": 0.0,
                    "anchor": list(anchor[:2]),
                    "floor_number": self._floor_number,
                    "source": "baseline_first_live_odometry",
                })
        rospy.loginfo("[%s] Baseline return anchor fixed at (%.3f, %.3f)",
                      self._floor_slug, anchor[0], anchor[1])
        return anchor

    def _seed_f3_terminal_stair_return_anchor(self, manager, route,
                                              corridor_anchor):
        """Extend strict F3's terminal return to the safe east landing.

        Baseline previously stopped at the exploration ingress and this
        wrapper then traversed the same corridor-to-landing leg in a second
        controller phase.  Convert the already-audited truth route's
        east-clear point into Baseline's live odometry frame.  Baseline gates
        this override on all required physical room EXITs, leaving partial and
        failure continuity under the existing bounded truth-return controller.
        """
        if (self._floor_number < 3 or not route or
                corridor_anchor is None):
            return None
        east_clear = next(
            (item for item in route
             if str(item.get("stage")) == "upper_landing_east_clear"),
            None)
        if east_clear is None:
            return None
        try:
            target_truth = tuple(map(float, east_clear["target"][:2]))
            with self._lock:
                current_truth = (tuple(self._truth_pose)
                                 if self._truth_pose else None)
            if current_truth is None:
                return None
            target = (
                float(corridor_anchor[0]) + target_truth[0] - current_truth[0],
                float(corridor_anchor[1]) + target_truth[1] - current_truth[1],
                float(corridor_anchor[2]))
            if not all(math.isfinite(value) for value in target):
                return None
        except (KeyError, TypeError, ValueError, IndexError):
            return None
        with manager.lock:
            manager.terminal_stair_wait_anchor_override = target
            history = getattr(manager, "corridor_sweep_history", None)
            if isinstance(history, list):
                history.append({
                    "event": "F3_TERMINAL_STAIR_RETURN_ANCHOR_SEEDED",
                    "elapsed_sec": 0.0,
                    "corridor_anchor": list(corridor_anchor[:2]),
                    "truth_reference_pose": list(current_truth[:2]),
                    "truth_east_clear_target": list(target_truth),
                    "baseline_terminal_target": list(target[:2]),
                    "activation_gate": "all_required_rooms_physically_exited",
                })
        rospy.loginfo(
            "[third_floor] strict terminal return extended to safe east "
            "landing (%.3f, %.3f)", target[0], target[1])
        return target

    @staticmethod
    def _physical_room_exit_evidence_count(manager):
        """Count real ENTRY/EXIT transactions even if view quality is partial.

        The scheduler deliberately removes a partial room from its strict
        completed-room set.  It can also merge a later portal back into the
        same estimated room id.  Neither operation erases the physical EXIT:
        ``BaselineExplorationManager`` writes one immutable map snapshot only
        after each successful exit transaction.  Keep unique room ids for the
        strict contract, but use the larger transaction count for the bounded
        corridor-end fail-safe that merely permits the dog to return home.
        """
        scheduler = getattr(manager, "room_scheduler", None)
        events = list(getattr(scheduler, "events", []) or [])
        entered = set()
        exited = set()
        for event in events:
            room_id = str(event.get("room_id") or "")
            if not room_id:
                continue
            if event.get("event") == "ROOM_ENTERED":
                entered.add(room_id)
            if (event.get("event") == "ROOM_GOAL_RESULT" and
                    str(event.get("role", "")).upper() == "EXIT" and
                    bool(event.get("success"))):
                exited.add(room_id)
        event_transaction_count = len(entered.intersection(exited))
        snapshots = list(getattr(manager, "room_exit_map_snapshots", []) or [])
        snapshot_transaction_count = sum(
            isinstance(item, dict) and bool(item.get("snapshot_file"))
            for item in snapshots)
        return max(event_transaction_count, snapshot_transaction_count)

    @staticmethod
    def _strict_room_contract_evidence(manager):
        """Return per-room acceptance evidence; counters cannot substitute."""
        scheduler = getattr(manager, "room_scheduler", None)
        doors = list(getattr(getattr(scheduler, "detector", None),
                             "doors", []) or [])
        rooms = []
        physical_door_centers = []
        for door in doors:
            room_id = getattr(door, "room_id", None)
            if not room_id:
                continue
            checks = {
                "visited": bool(getattr(door, "visited", False)),
                "contract_locked_before_entry": bool(getattr(
                    door, "viewpoint_contract_locked_before_entry", False)),
                "entry_truth_crossing_confirmed": bool(getattr(
                    door, "entry_truth_crossing_confirmed", False)),
                "g3_truth_pose_present": getattr(
                    door, "g3_truth_pose", None) is not None,
                "g4_truth_pose_present": getattr(
                    door, "g4_truth_pose", None) is not None,
                "physical_two_view_contract_met": bool(getattr(
                    door, "physical_two_view_contract_met", False)),
                "viewpoint_contract_geometry_met": bool(getattr(
                    door, "viewpoint_contract_geometry_met", False)),
                "exit_truth_crossing_confirmed": bool(getattr(
                    door, "exit_truth_crossing_confirmed", False)),
                "room_transaction_complete": bool(getattr(
                    door, "room_transaction_complete", False)),
                "corridor_centerline_recovered": bool(getattr(
                    door, "corridor_centerline_recovered", False)),
            }
            rooms.append({
                "room_id": str(room_id),
                "door_id": str(getattr(door, "door_id", "")),
                "viewpoint_contract": getattr(
                    door, "viewpoint_contract", None),
                "checks": checks,
                "strict_complete": all(checks.values()),
                "viewpoint_contract_geometry": getattr(
                    door, "viewpoint_contract_geometry", None),
            })
            center = getattr(door, "center", None)
            if isinstance(center, (list, tuple)) and len(center) >= 2:
                point = (float(center[0]), float(center[1]))
                # Same-row opposite doors are about 2.38 m apart.  A 1.15 m
                # physical-portal cluster catches detector jitter without
                # merging the two real rooms.
                if not any(math.hypot(point[0] - prior[0],
                                      point[1] - prior[1]) <= 1.15
                           for prior in physical_door_centers):
                    physical_door_centers.append(point)
        strict_complete_count = sum(
            bool(item["strict_complete"]) for item in rooms)
        physical_distinct_count = len(physical_door_centers)
        return {
            "rooms": rooms,
            "strict_complete_count": min(strict_complete_count,
                                         physical_distinct_count),
            "distinct_room_count": physical_distinct_count,
            "semantic_room_record_count": len(rooms),
            "physical_door_centers": [list(point)
                                      for point in physical_door_centers],
        }

    def _third_floor_return_to_stair_wait(self, route, timeout_sec=60.0):
        """Physically reverse the F3 corridor ingress to the stair wait lip."""
        if self._floor_number < 3 or not route:
            return False, {}
        by_stage = {str(item.get("stage")): item for item in route}
        east_clear = by_stage.get("upper_landing_east_clear")
        east_step = by_stage.get("upper_landing_east_step")
        corridor_item = next(
            (item for item in route
             if str(item.get("stage", "")).endswith("_corridor_entry")),
            None)
        if not east_clear or not east_step or not corridor_item:
            return False, {"reason": "f3_reverse_route_geometry_missing"}
        try:
            clear_x, landing_y = map(float, east_clear["target"][:2])
            wait_x, wait_y = map(float, east_step["target"][:2])
            bounds = corridor_item["corridor_bounds"]
            corridor_y = max(
                float(bounds["y_min"]) + 0.75,
                float(corridor_item["target"][1]))
        except (KeyError, TypeError, ValueError, IndexError):
            return False, {"reason": "f3_reverse_route_geometry_invalid"}
        with self._lock:
            initial = tuple(self._truth_pose) if self._truth_pose else None
        if initial is None:
            return False, {"reason": "truth_pose_missing"}
        # Follow the exact physical corridor/landing bridge in reverse.  A
        # direct diagonal from the last room row to the stair cuts through the
        # lobby wall and was the source of earlier terminal stalls.
        targets = []
        # The exploration manager may already have returned past the nominal
        # corridor-entry station before handing control here.  Sending that
        # pose north again makes the dog perform a 180-degree turn and was
        # rejected by the progress watchdog at y=11.6 while the stair lies
        # south at y=1.55.  Only do the far-row lateral alignment while still
        # north of the corridor station; otherwise continue monotonically
        # toward the landing.
        if float(initial[1]) > corridor_y + 0.50:
            targets.append((
                clear_x,
                min(float(initial[1]), float(bounds["y_max"]) - 0.5),
                "corridor_lateral_align"))
        targets.extend([
            (clear_x, landing_y, "corridor_reverse_to_landing"),
            # Approach the stair lip orthogonally.  Driving directly from the
            # corridor-clear point to x=-2.2 cut the inner corner, pushed the
            # body south to y=1.16 and stalled 0.83 m from the wait target in
            # roundtrip_f3forwardreturnfix.  This intermediate point stays on
            # the landing side of the lip and fixes y before the westward
            # crossing.
            (max(wait_x + 0.65, clear_x - 0.65), landing_y,
             "stair_wait_lip_pre_align"),
            (wait_x, wait_y, "stair_wait_lip"),
        ])
        sim_started = rospy.Time.now()
        wall_started = time.monotonic()
        sim_deadline = rospy.Duration(max(20.0, float(timeout_sec)))
        wall_deadline = wall_started + max(120.0, 3.0 * float(timeout_sec))
        # This is a handoff zone, not the descent landing centre.  The
        # descent worker subsequently guides from here to the generated
        # landing centre under its own truth/upright gates.  Requiring 0.42 m
        # made run98 reject a fully upright 4/4 return at the physical stair
        # boundary (best distance 0.558 m) and oscillate until the progress
        # watchdog fired.  Keep the zone tightly bounded and configurable;
        # height and posture are still independently required below.
        stair_wait_handoff_tolerance = max(0.55, min(0.75, float(
            rospy.get_param("~third_floor_stair_wait_handoff_tolerance_m",
                            0.60))))
        trace = []
        final_lip_reached_pose = None
        final_lip_reached_distance = None
        final_lip_reached_ros_time = None
        self._state_pub.publish(String(data=self._event(
            "THIRD_FLOOR_STAIR_WAIT_TRANSIT_START")))
        for target_x, target_y, stage in targets:
            # Intermediate corridor waypoints only shape a collision-safe
            # route; they are not acceptance gates.  Requiring the same
            # 0.42 m tolerance as the final stair lip caused a false stall at
            # 0.436 m in roundtrip_roomstart_readylatch even though the dog
            # was already aligned with the corridor-return leg.  Keep the
            # final lip strict and allow a small, explicit tolerance on route
            # shaping points.
            arrival_tolerance = (
                stair_wait_handoff_tolerance
                if stage == "stair_wait_lip" else
                0.70 if stage == "stair_wait_lip_pre_align" else 0.50)
            # The exploration executor is quiesced once before this method.
            # Do not run a second, separately timed stationary yaw controller
            # before every route leg.  run151 proved that helper can consume
            # its whole ten-second budget at the lip even while making valid
            # physical yaw progress; run152 then proved the route-leg loop
            # below reaches the same gate with its own rotate-before-translate
            # closure.  Keeping both controllers duplicated yaw work and up
            # to 0.8 s of fixed settling per leg.  Retain one short zero-command
            # handoff sample, then let the bounded leg controller own yaw,
            # translation, total timeout and no-progress detection together.
            with self._lock:
                prealign_pose = (tuple(self._truth_pose)
                                 if self._truth_pose else None)
            if prealign_pose is None:
                return False, {"reason": "truth_pose_missing_before_f3_return_leg",
                               "stage": stage, "trace": trace}
            prealign_heading = math.atan2(
                target_y - prealign_pose[1], target_x - prealign_pose[0])
            trace.append({
                "stage": stage,
                "target": [target_x, target_y],
                "truth_pose": list(prealign_pose),
                "initial_heading_error_rad": round(
                    self._wrapped_angle_error(
                        prealign_heading, prealign_pose[3]), 4),
                "action": "bounded_route_leg_yaw_closure",
                "stationary_prealign_skipped": True,
            })
            settle_started = rospy.Time.now()
            settle_wall_deadline = time.monotonic() + 4.0
            while (not rospy.is_shutdown() and
                   (rospy.Time.now() - settle_started).to_sec() < 0.20 and
                   time.monotonic() < settle_wall_deadline):
                self._publish_truth_body_command(0.0, 0.0)
                time.sleep(0.05)
            best_distance = math.inf
            last_progress_sim = rospy.Time.now()
            while (not rospy.is_shutdown() and
                   rospy.Time.now() - sim_started < sim_deadline and
                   time.monotonic() < wall_deadline):
                with self._lock:
                    pose = tuple(self._truth_pose) if self._truth_pose else None
                if pose is None:
                    time.sleep(0.05)
                    continue
                if not self._truth_height_is_second_floor(
                        pose, self._second_floor_elevation,
                        self._second_floor_height_margin):
                    self._publish_truth_world_command(0.0, 0.0, 0.0)
                    return False, {
                        "reason": "f3_height_lost_during_stair_return",
                        "truth_pose": list(pose), "trace": trace}
                dx, dy = target_x - pose[0], target_y - pose[1]
                distance = math.hypot(dx, dy)
                if distance <= arrival_tolerance:
                    self._publish_truth_world_command(0.0, 0.0, 0.0)
                    if stage == "stair_wait_lip":
                        final_lip_reached_pose = tuple(pose)
                        final_lip_reached_distance = float(distance)
                        final_lip_reached_ros_time = rospy.Time.now()
                    trace.append({"stage": stage, "target": [target_x, target_y],
                                  "truth_pose": list(pose), "reached": True,
                                  "arrival_tolerance_m": arrival_tolerance})
                    break
                now_sim = rospy.Time.now()
                progress_quantum = (
                    0.03 if stage == "stair_wait_lip_pre_align" else 0.08)
                if distance <= best_distance - progress_quantum:
                    best_distance = distance
                    last_progress_sim = now_sim
                elif (now_sim - last_progress_sim).to_sec() >= 10.0:
                    self._publish_truth_world_command(0.0, 0.0, 0.0)
                    return False, {
                        "reason": "f3_stair_return_stalled", "stage": stage,
                        "target": [target_x, target_y],
                        "truth_pose": list(pose),
                        "best_distance_m": best_distance, "trace": trace}
                speed_cap = (
                    self._third_floor_return_corridor_speed
                    if stage == "corridor_reverse_to_landing" else
                    self._third_floor_return_lip_speed)
                speed = min(speed_cap, max(0.25, 0.70 * distance))
                heading = math.atan2(dy, dx)
                yaw_error = self._wrapped_angle_error(heading, pose[3])
                yaw_rate = max(-0.60, min(0.60, 1.10 * yaw_error))
                # Do not use a world-frame lateral vector for the F3 return.
                # ``_publish_truth_world_command`` mirrors lateral velocity
                # for the F3 flat-policy interface.  During run
                # f1_f2_f3_fullflow_600s_deepcapfix_20260826_163304 that
                # turned the westward corridor alignment into a persistent
                # southward drift (y=13.4 -> 11.3) and the progress watchdog
                # rejected an otherwise complete 4/4 floor.  Turn toward the
                # target and command forward-only motion instead.  A world
                # vector along the live truth yaw transforms to body forward
                # with zero lateral component on every floor, so it is
                # independent of that policy convention and cannot cut a
                # diagonal through the landing wall while rotating.
                heading_gate = max(0.0, math.cos(yaw_error))
                forward_speed = speed * heading_gate
                # Rotate first, then translate along the corridor.  Allowing
                # forward motion at 0.85 rad made the far-row return curve
                # west while aiming south; distance improved by less than the
                # 0.08 m watchdog quantum and falsely stalled at y=28.4.
                if abs(yaw_error) > 0.30:
                    forward_speed = 0.0
                # This path is F3-only.  Use the same unmirrored world/body
                # transform as the proven F3 ingress guide; the generic
                # upper-floor helper mirrors lateral velocity on F3 and is
                # inappropriate while a small heading correction is still
                # active near the corridor edge.
                self._publish_f3_world_command(
                    forward_speed * math.cos(pose[3]),
                    forward_speed * math.sin(pose[3]), yaw_rate, pose)
                time.sleep(0.05)
            else:
                self._publish_truth_world_command(0.0, 0.0, 0.0)
                return False, {
                    "reason": "f3_stair_return_timeout", "stage": stage,
                    "target": [target_x, target_y], "trace": trace,
                    "elapsed_sim_sec": round(
                        (rospy.Time.now() - sim_started).to_sec(), 3)}
        settle_start = rospy.Time.now()
        while (not rospy.is_shutdown() and
               (rospy.Time.now() - settle_start).to_sec() < 1.0 and
               time.monotonic() < wall_deadline):
            self._publish_truth_world_command(0.0, 0.0, 0.0)
            time.sleep(0.05)
        with self._lock:
            final_pose = tuple(self._truth_pose) if self._truth_pose else None
            stand_status = dict(self._fixed_stand_status or {})
        final_distance = (math.hypot(final_pose[0] - wait_x,
                                     final_pose[1] - wait_y)
                          if final_pose is not None else math.inf)
        upright = bool(
            final_pose is not None and
            self._truth_height_is_second_floor(
                final_pose, self._second_floor_elevation,
                self._second_floor_height_margin) and
            abs(float(stand_status.get("roll", 0.0))) < 0.45 and
            abs(float(stand_status.get("pitch", 0.0))) < 0.35)
        post_settle_distance = float(final_distance)
        reached_pose_drift = (
            math.hypot(final_pose[0] - final_lip_reached_pose[0],
                       final_pose[1] - final_lip_reached_pose[1])
            if final_pose is not None and final_lip_reached_pose is not None
            else math.inf)
        correction_attempted = False
        correction_success = False
        correction_elapsed_sim = 0.0

        # The route loop has already proved physical arrival at the strict
        # stair-wait gate.  A one-second zero-command settle can move the body
        # a few millimetres on the landing (run160: 0.600 -> 0.60595 m), and
        # reapplying the same hard threshold then incorrectly strands a fully
        # upright 4/4 mission on F3.  First make one short physical correction;
        # this is deliberately local to the F3 post-settle handoff and never
        # relaxes ordinary route-goal arrival tolerances.
        correction_eligible = bool(
            final_lip_reached_pose is not None and final_pose is not None and
            upright and
            final_distance > stair_wait_handoff_tolerance and
            final_distance <= stair_wait_handoff_tolerance + 0.08 and
            reached_pose_drift <= 0.10)
        if correction_eligible:
            correction_attempted = True
            correction_started = rospy.Time.now()
            correction_wall_deadline = min(wall_deadline,
                                           time.monotonic() + 10.0)
            correction_sim_limit = rospy.Duration(2.5)
            correction_target_distance = max(
                0.0, stair_wait_handoff_tolerance - 0.02)
            while (not rospy.is_shutdown() and
                   rospy.Time.now() - correction_started <
                   correction_sim_limit and
                   time.monotonic() < correction_wall_deadline):
                with self._lock:
                    pose = (tuple(self._truth_pose)
                            if self._truth_pose else None)
                if pose is None or not self._truth_height_is_second_floor(
                        pose, self._second_floor_elevation,
                        self._second_floor_height_margin):
                    break
                dx, dy = wait_x - pose[0], wait_y - pose[1]
                distance = math.hypot(dx, dy)
                if distance <= correction_target_distance:
                    correction_success = True
                    break
                heading = math.atan2(dy, dx)
                yaw_error = self._wrapped_angle_error(heading, pose[3])
                yaw_rate = max(-0.45, min(0.45, 1.10 * yaw_error))
                forward_speed = min(0.22, max(0.12, 0.55 * distance))
                if abs(yaw_error) > 0.25:
                    forward_speed = 0.0
                self._publish_f3_world_command(
                    forward_speed * math.cos(pose[3]),
                    forward_speed * math.sin(pose[3]), yaw_rate, pose)
                time.sleep(0.05)
            self._publish_truth_world_command(0.0, 0.0, 0.0)
            correction_elapsed_sim = max(
                0.0, (rospy.Time.now() - correction_started).to_sec())
            correction_settle_started = rospy.Time.now()
            while (not rospy.is_shutdown() and
                   (rospy.Time.now() - correction_settle_started).to_sec() <
                   0.20 and time.monotonic() < correction_wall_deadline):
                self._publish_truth_world_command(0.0, 0.0, 0.0)
                time.sleep(0.05)
            with self._lock:
                final_pose = (tuple(self._truth_pose)
                              if self._truth_pose else None)
                stand_status = dict(self._fixed_stand_status or {})
            final_distance = (
                math.hypot(final_pose[0] - wait_x,
                           final_pose[1] - wait_y)
                if final_pose is not None else math.inf)
            upright = bool(
                final_pose is not None and
                self._truth_height_is_second_floor(
                    final_pose, self._second_floor_elevation,
                    self._second_floor_height_margin) and
                abs(float(stand_status.get("roll", 0.0))) < 0.45 and
                abs(float(stand_status.get("pitch", 0.0))) < 0.35)
            correction_success = bool(
                correction_success or
                (final_distance <= stair_wait_handoff_tolerance and upright))

        final_reached_pose_drift = (
            math.hypot(final_pose[0] - final_lip_reached_pose[0],
                       final_pose[1] - final_lip_reached_pose[1])
            if final_pose is not None and final_lip_reached_pose is not None
            else math.inf)
        # A final tiny hysteresis is permitted only after the strict gate was
        # physically reached and the settled body stayed upright within five
        # centimetres of that reached pose.  This prevents millimetre-scale
        # contact relaxation from blocking descent while still rejecting any
        # robot that never reached the lip or drifted materially away.
        settle_hysteresis_accepted = bool(
            final_lip_reached_pose is not None and upright and
            final_distance <= stair_wait_handoff_tolerance + 0.03 and
            final_reached_pose_drift <= 0.05)
        evidence = {
            "target": [wait_x, wait_y],
            "truth_pose": list(final_pose) if final_pose else None,
            "final_distance_m": final_distance,
            "pre_correction_post_settle_distance_m": post_settle_distance,
            "strict_arrival_distance_m": final_lip_reached_distance,
            "strict_arrival_ros_time_sec": (
                final_lip_reached_ros_time.to_sec()
                if final_lip_reached_ros_time is not None else None),
            "reached_pose_drift_m": (
                final_reached_pose_drift
                if math.isfinite(final_reached_pose_drift) else None),
            "post_settle_correction_attempted": correction_attempted,
            "post_settle_correction_success": correction_success,
            "post_settle_correction_ros_sim_s": round(
                correction_elapsed_sim, 3),
            "post_settle_hysteresis_accepted":
                settle_hysteresis_accepted,
            "upright_posture_valid": upright,
            "fixed_stand_status": stand_status or None,
            "elapsed_sim_sec": round(
                (rospy.Time.now() - sim_started).to_sec(), 3),
            "elapsed_wall_sec": round(time.monotonic() - wall_started, 3),
            "trace": trace,
        }
        success = bool(
            upright and
            (final_distance <= stair_wait_handoff_tolerance or
             settle_hysteresis_accepted))
        evidence["reason"] = ("third_floor_stair_wait_reached" if success
                              else "third_floor_stair_wait_gate_rejected")
        self._write_handoff(
            "THIRD_FLOOR_STAIR_WAIT_TRANSIT_READY" if success else
            "THIRD_FLOOR_STAIR_WAIT_TRANSIT_FAILED", **evidence)
        return success, evidence

    def _quiesce_exploration_executor_for_f3_return(self, manager):
        """Prove that the last exploration goal released ``/cmd_vel``.

        Baseline uses identity-bound cancellation, so publishing an
        unstructured second cancel here would be ignored by GoalExecutor.
        Instead require the last executor result to be terminal, clear the
        manager telemetry latch, and hold zero briefly before the dedicated
        truth return takes command.
        """
        with manager.lock:
            results = list(getattr(manager, "execution_results", []) or [])
            manager.active_goal_id = -1
            manager.active_waypoint_id = -1
            manager.active_goal_last_pose = None
        terminal = results[-1] if results else None
        terminal_reason = str((terminal or {}).get("reason", ""))
        if terminal is not None and not (
                bool(terminal.get("success")) or terminal_reason):
            return False, {
                "reason": "executor_terminal_result_invalid",
                "last_execution_result": terminal}
        manager.active_goal_pub.publish(Int32(data=-1))
        manager.active_waypoint_pub.publish(Int32(data=-1))
        settle_started = rospy.Time.now()
        wall_deadline = time.monotonic() + 4.0
        while (not rospy.is_shutdown() and
               (rospy.Time.now() - settle_started).to_sec() < 1.0 and
               time.monotonic() < wall_deadline):
            self._publish_truth_body_command(0.0, 0.0)
            time.sleep(0.05)
        evidence = {
            "reason": "exploration_executor_quiesced",
            "last_execution_reason": terminal_reason or None,
            "last_goal_stamp": ((terminal or {}).get("goal_stamp")),
            "settled_sim_sec": round(
                (rospy.Time.now() - settle_started).to_sec(), 3),
        }
        self._write_handoff(
            self._event("F3_EXPLORATION_EXECUTOR_QUIESCED"), **evidence)
        return True, evidence

    def _truth_return_to_corridor_anchor(self, anchor, timeout_sec=45.0):
        """Bounded, portal-shaped flat-floor fallback to the corridor anchor.

        A direct diagonal from a deep room to the lobby cuts through the room
        wall.  The former 32 s monotonic deadline also represented barely
        17--25 simulated seconds at low RTF, less than the physical travel
        time from the far row.  Route first back to the far-door station,
        cross laterally onto the corridor, and only then run home.  Motion is
        bounded in ROS simulation time; wall time is an independent watchdog.
        """
        if anchor is None or len(anchor) < 2:
            return False
        target_x, target_y = float(anchor[0]), float(anchor[1])
        started = time.monotonic()
        sim_started = rospy.Time.now()
        sim_timeout = rospy.Duration(max(12.0, float(timeout_sec)))
        wall_deadline = started + max(90.0, 2.5 * float(timeout_sec))
        best_distance = math.inf
        last_progress_sim = sim_started
        route = None
        route_door_station_source = None
        route_index = 0
        self._state_pub.publish(String(data=self._event(
            "TRUTH_CORRIDOR_RETURN_RECOVERY")))
        while (not rospy.is_shutdown() and
               rospy.Time.now() - sim_started < sim_timeout and
               time.monotonic() < wall_deadline):
            with self._lock:
                pose = tuple(self._truth_pose) if self._truth_pose else None
            if pose is None:
                time.sleep(0.05)
                continue
            if not self._truth_height_is_second_floor(
                    pose, self._second_floor_elevation,
                    self._second_floor_height_margin):
                self._publish_truth_world_command(0.0, 0.0, 0.0)
                return False
            if route is None:
                longitudinal = float(pose[1]) - target_y
                # Recover through the actual doorway row containing the
                # truth pose.  The prior offset clamp selected the *current*
                # y for a near room (for example y=21.3), then attempted a
                # lateral crossing through its solid corridor wall.  Offline
                # room bounds are used only by this already-degraded truth
                # return, never by ordinary exploration or room completion.
                door_station_y = None
                try:
                    with open(self._truth_layout, "r", encoding="utf-8") as stream:
                        metadata = json.load(stream)
                    floor = next(
                        item for item in (metadata.get("floors") or [])
                        if int(item.get("floor_index", -1)) ==
                        self._floor_index)
                    corridor = floor.get("corridor_bounds") or {}
                    room_rows = []
                    containing_rows = []
                    for room in (floor.get("rooms") or []):
                        bounds = room.get("bounds") or {}
                        x_min = float(bounds["x_min"])
                        x_max = float(bounds["x_max"])
                        y_min = float(bounds["y_min"])
                        y_max = float(bounds["y_max"])
                        station = 0.5 * (y_min + y_max)
                        room_rows.append(station)
                        if (x_min - 0.25 <= float(pose[0]) <= x_max + 0.25 and
                                y_min - 0.25 <= float(pose[1]) <= y_max + 0.25):
                            containing_rows.append(station)
                    candidates = containing_rows or room_rows
                    if candidates:
                        door_station_y = min(
                            candidates, key=lambda value:
                            abs(float(value) - float(pose[1])))
                        if corridor:
                            door_station_y = max(
                                float(corridor["y_min"]), min(
                                    float(corridor["y_max"]),
                                    float(door_station_y)))
                        route_door_station_source = (
                            "containing_truth_room_row" if containing_rows
                            else "nearest_truth_room_row")
                except (OSError, ValueError, KeyError, StopIteration,
                        TypeError, json.JSONDecodeError):
                    door_station_y = None
                if door_station_y is None:
                    door_station_y = target_y + math.copysign(
                        min(16.0, abs(longitudinal)), longitudinal)
                    route_door_station_source = "bounded_offset_fallback"
                route = []
                if abs(float(pose[0]) - target_x) > 1.0:
                    route.extend([
                        (float(pose[0]), door_station_y),
                        (target_x, door_station_y),
                    ])
                route.append((target_x, target_y))
            leg_x, leg_y = route[route_index]
            dx, dy = leg_x - pose[0], leg_y - pose[1]
            distance = math.hypot(dx, dy)
            if distance <= 0.65:
                if route_index + 1 < len(route):
                    route_index += 1
                    best_distance = math.inf
                    last_progress_sim = rospy.Time.now()
                    self._publish_truth_world_command(0.0, 0.0, 0.0)
                    continue
                self._publish_truth_world_command(0.0, 0.0, 0.0)
                self._write_handoff(
                    self._event("TRUTH_CORRIDOR_RETURN_RECOVERED"),
                    target=[target_x, target_y], truth_pose=list(pose),
                    route=[list(item) for item in route],
                    route_door_station_source=route_door_station_source,
                    elapsed_sim_sec=round(
                        (rospy.Time.now() - sim_started).to_sec(), 3),
                    elapsed_wall_sec=round(time.monotonic() - started, 3))
                return True
            now_sim = rospy.Time.now()
            if distance <= best_distance - 0.08:
                best_distance = distance
                last_progress_sim = now_sim
            elif (now_sim - last_progress_sim).to_sec() >= 10.0:
                self._publish_truth_world_command(0.0, 0.0, 0.0)
                self._write_handoff(
                    self._event("TRUTH_CORRIDOR_RETURN_STALLED"),
                    target=[leg_x, leg_y], truth_pose=list(pose),
                    route=[list(item) for item in route],
                    route_index=route_index,
                    route_door_station_source=route_door_station_source,
                    best_distance_m=best_distance)
                return False
            speed = min(0.85, max(0.22, 0.42 * distance))
            desired_heading = math.atan2(dy, dx)
            yaw_error = self._wrapped_angle_error(desired_heading, pose[3])
            yaw_rate = max(-0.34, min(0.34, 0.85 * yaw_error))
            self._publish_truth_world_command(
                speed * dx / distance, speed * dy / distance, yaw_rate)
            time.sleep(0.05)
        self._publish_truth_world_command(0.0, 0.0, 0.0)
        self._write_handoff(
            self._event("TRUTH_CORRIDOR_RETURN_TIMEOUT"),
            target=[target_x, target_y], best_distance_m=best_distance,
            route=[list(item) for item in (route or [])],
            route_index=route_index,
            route_door_station_source=route_door_station_source,
            elapsed_sim_sec=round(
                (rospy.Time.now() - sim_started).to_sec(), 3),
            wall_watchdog_expired=time.monotonic() >= wall_deadline)
        return False

    def _f3_physical_return_safe(self, manager, physical_exit_count):
        """Gate locomotion continuity independently from room acceptance.

        A missing G3/G4 or hazard makes the exploration result incomplete,
        but it must not strand an upright robot on the third floor.  The
        strict room contract is evaluated separately after the physical
        return.  This gate proves only that the dedicated truth-guided route
        may safely take command from the current flat-floor pose.
        """
        if self._floor_number < 3:
            return False
        with self._lock:
            pose = tuple(self._truth_pose) if self._truth_pose else None
        if pose is None or not self._truth_height_is_second_floor(
                pose, self._second_floor_elevation,
                self._second_floor_height_margin):
            return False
        scheduler = getattr(manager, "room_scheduler", None)
        if getattr(scheduler, "active_door", None) is not None:
            return False
        # Either a physical room EXIT or an established corridor proves that
        # F3 exploration actually took ownership.  This prevents a startup
        # failure on the stair seam from being mistaken for a safe descent
        # request while allowing 0/4 corridor-only failures to return home.
        return bool(
            int(physical_exit_count) > 0 or
            bool(getattr(manager, "corridor_established", False)))

    @staticmethod
    def _exploration_handoff_authorized(termination_reason,
                                        exited_room_count,
                                        room_target_count,
                                        terminal_return_latched=False,
                                        allow_partial_floor_handoff=False,
                                        partial_minimum_exited_rooms=3,
                                        emergency_minimum_exited_rooms=2):
        """Require both a successful return token and every physical EXIT."""
        successful_returns = {
            "STAIR_CORRIDOR_EXIT_HANDOFF",
            "STAIR_CORRIDOR_EXIT_HANDOFF_TRUTH_GATE",
            "STAIR_CORRIDOR_EXIT_HANDOFF_TRUTH_REQUESTED",
            "STAIR_LOBBY_HANDOFF_LOCALIZATION_FALLBACK",
            "STAIR_WAIT_ZONE_REACHED",
        }
        # Retain the historical fourth argument for callers and diagnostics,
        # but a terminal wall/latch is route evidence only.  It must never
        # substitute for the configured number of confirmed room exits.
        _ = terminal_return_latched
        try:
            exited = int(exited_room_count)
            target = max(4, int(room_target_count))
            partial_minimum = max(1, min(
                target, int(partial_minimum_exited_rooms)))
        except (TypeError, ValueError, OverflowError):
            return False
        return bool(
            str(termination_reason) in successful_returns and
            (exited >= target or
             (bool(allow_partial_floor_handoff) and
              (exited >= partial_minimum or
               (bool(terminal_return_latched) and
                exited >= max(1, min(partial_minimum,
                                     int(emergency_minimum_exited_rooms))))))))

    def run(self):
        """Run one upper-floor mission and always expose fatal exceptions."""
        try:
            return self._run_mission()
        except Exception as error:
            if rospy.is_shutdown():
                return False
            rospy.logerr("[%s] unhandled exception at %s: %s: %s",
                         self._floor_slug, self._mission_stage,
                         type(error).__name__, error)
            return self._publish_exploration_failure(
                "unhandled_exception",
                state_suffix="EXPLORATION_EXCEPTION",
                exception_type=type(error).__name__,
                exception_message=str(error))

    def _run_mission(self):
        self._mission_stage = "WAIT_FOR_STAIR"
        if not self._wait_for_stair():
            if rospy.is_shutdown():
                return False
            return self._publish_exploration_failure(
                "stair_handoff_unavailable", state_suffix="NOT_STARTED")
        if self._floor_number >= 2:
            # The local landing reset needs the destination floor elevation.
            # Geometry used to be loaded later by the corridor guide, leaving
            # this first post-stair call to evaluate float(None) and crash.
            # Preflight the same validated route now for both F2 and F3.
            self._mission_stage = "LOAD_FLOOR_GEOMETRY"
            self._request_preloaded_floor_map()
            self._refresh_truth_fast_handoff()
            route = self._second_floor_truth_route()
            if self._second_floor_elevation is None or not route:
                return self._publish_exploration_failure(
                    "floor_geometry_unavailable",
                    state_suffix="GT_GUIDE_FAILED")
            # Fast corridor/room running produces IMU impact spikes that the
            # FSM's checkSafty() (rotation-matrix z < 0.5, i.e. tilt > 60 deg)
            # misreads as a fall and force-switches to passive, killing
            # locomotion_ready mid-goal (speed_fix_18: F2 G4 failed with
            # truth roll/pitch ~0.02 while the IMU pulsed).  Real falls are
            # already caught by compliant_fall_recovery via Gazebo truth
            # height; keep the FSM safety off for the whole upper-floor
            # exploration and let the truth-based recovery own fall handling.
            rospy.set_param("/simenv/f2_f3_truth_correction_active", True)
            # Check the stair manager's fresh upright plane-policy handoff
            # before touching model/controller state.  In run122 the same
            # check happened only inside _restore_plane_policy(), after the
            # unconditional reset had already invalidated the evidence and
            # spent another FixedStand/RL transition.  Failed validation is
            # fail-closed and retains the complete reset/recovery path below.
            self._mission_stage = "PRE_RESET_STAIR_HANDOFF_CHECK"
            self._verified_stair_plane_handoff_reused = bool(
                self._verified_stair_plane_handoff_ready(
                    publish_ready=False))
            # The local landing reset needs the destination floor elevation.
            # The F3 landing deck is a narrow stair throat.  Resetting the
            # physical body there can leave a compact stance tilted against an
            # edge (fix_94: roll=0.65 rad), even though its controller reset
            # succeeded.  First crossing the known seam to the flat corridor
            # provides the same truth ingress as the guide, but gives the
            # fixed-stand/RL transition a level support polygon.  F2 keeps
            # the original landing-first route.
            self._mission_stage = "LANDING_RESET"
            if self._verified_stair_plane_handoff_reused:
                if self._floor_number >= 3:
                    self._f3_physical_corridor_guide = True
                rospy.loginfo(
                    "[%s] preserving verified stair pose/controller state; "
                    "skipping redundant landing reset.", self._floor_slug)
            elif self._floor_number < 3:
                reset_waypoint = tuple(route[0]["target"])
                if not self._local_reset_controller_at_corridor(reset_waypoint):
                    # The controller-manager can still be releasing the stair
                    # effort controllers at the first F2 landing callback.
                    # zeroxitcontinue failed after one transient
                    # ``Failed to stop A1 effort controllers`` and shut down
                    # all remaining floors.  Clear the failed transaction,
                    # hold zero for a short bounded release window, and retry
                    # the same local reset exactly once.
                    self._state_pub.publish(String(data=self._event(
                        "LOCAL_CONTROLLER_RESET_RETRY")))
                    rospy.set_param("/simenv/local_reset/enabled", False)
                    retry_release_until = time.monotonic() + 0.75
                    while (not rospy.is_shutdown() and
                           time.monotonic() < retry_release_until):
                        self._cmd_pub.publish(Twist())
                        time.sleep(0.05)
                    if not self._local_reset_controller_at_corridor(
                            reset_waypoint):
                        return self._publish_exploration_failure(
                            "landing_reset_failed_after_bounded_retry",
                            state_suffix="POLICY_FAILED")
            else:
                # F3 alone starts on a narrow, dynamic stair seam. Place it
                # on the validated flat landing before the physical east+north
                # corridor guide takes ownership. This does not affect F1/F2.
                reset_waypoint = tuple(route[0]["target"])
                # Level the trunk first, then let the controller-local reset be
                # the final writer of pose, heading, joints and fresh feedback.
                # Reversing these calls formerly rotated a weight-bearing F3
                # stance by 90 degrees after SetModelConfiguration completed.
                if not self._reset_truth_upper_landing_pose(reset_waypoint):
                    return self._publish_exploration_failure(
                        "landing_truth_pose_reset_failed",
                        state_suffix="POLICY_FAILED")
                if not self._local_reset_controller_at_corridor(reset_waypoint):
                    return self._publish_exploration_failure(
                        "landing_reset_failed", state_suffix="POLICY_FAILED")
                self._f3_physical_corridor_guide = True
        # The physical truth guide sends velocity commands, so it must run
        # only after the flat-ground policy owns the robot. Running the F3
        # guide while the stair policy still owns the gait makes the first
        # east-clear leg appear collision-stalled even on a valid landing.
        # The F3 direct branch changes policy without a model-state reset.
        self._mission_stage = "RESTORE_PLANE_POLICY"
        if not self._restore_plane_policy():
            # Never send exploration goals while the controller says it cannot
            # walk. Geometry and localization fallbacks cannot replace this
            # actuator-safety handshake.
            return self._publish_exploration_failure(
                "plane_policy_restore_failed", state_suffix="POLICY_FAILED")
        self._mission_stage = "PREPARE_LIVE_MAP_FOR_INGRESS"
        if not self._prepare_upper_floor_live_map_for_ingress():
            return self._publish_exploration_failure(
                "live_map_ingress_preparation_failed",
                state_suffix="MAP_LIFECYCLE_FAILED")
        # Every upper floor needs the same explicit corridor-ingress handoff
        # before room scheduling begins. On F3 the guide preserves the live RL
        # controller at a seam correction (F3_SEAM_RL_HOLD), so FAST-LIO is
        # rebased at the corridor instead of starting from stale stair-landing
        # coordinates.
        self._mission_stage = "GUIDE_TO_CORRIDOR"
        self._corridor_guide_started_ros_sim = float(
            rospy.Time.now().to_sec())
        if not self._guide_to_second_floor_corridor():
            return self._publish_exploration_failure(
                "corridor_guide_failed", state_suffix="GT_GUIDE_FAILED")
        self._corridor_guide_completed_ros_sim = float(
            rospy.Time.now().to_sec())
        self._mission_stage = "STABILIZE_LOCALIZATION"
        if not self._stabilize_second_floor_localization():
            self._publish_exploration_failure(
                "localization_stabilization_failed",
                state_suffix="LOCALIZATION_FAILED")
            self._generate_combined_visualization()
            return False
        self._mission_stage = "ENSURE_LOCOMOTION_READY"
        if not self._ensure_exploration_locomotion_ready():
            return self._publish_exploration_failure(
                "locomotion_recovery_failed", state_suffix="POLICY_FAILED")
        self._mission_stage = "ACTIVATE_EXECUTOR_CONTEXT"
        if not self._activate_second_floor_executor_context():
            return self._publish_exploration_failure(
                "executor_context_failed",
                state_suffix="EXECUTOR_CONTEXT_FAILED")
        self._mission_stage = "MAP_CORRIDOR_INGRESS"
        self._map_second_floor_lobby()
        self._copy_first_floor_parameters()
        self._mission_stage = "BASELINE_INITIALIZATION"
        manager = BaselineExplorationManager()
        # Upper-floor figures are rendered once by this wrapper's detached
        # combined visualizer after the handoff/result files are complete.
        # Keep the runtime object explicit as well as the copied ROS
        # parameter: a private-parameter precedence race otherwise lets
        # BaselineExplorationManager run its synchronous final map wait and
        # plotting subprocess before returning control.  run159's terminal
        # corridor goal ended around ROS 890.9 s, but this wrapper could not
        # quiesce the executor until ROS 920.3 s.  That ~29 s gap cannot add
        # mission evidence and delays the physical stair return.  Disabling
        # only this duplicate online renderer preserves every final 12/13/14
        # artifact, which the wrapper/descent finalizer still generates.
        manager.auto_generate_visualization = False
        if self._floor_number >= 2:
            # The wrapper copies upper-floor parameters before construction,
            # but the embedded manager resolves private parameters in the
            # wrapper node namespace.  Make the verified truth-handoff
            # contract explicit on the actual runtime object for F2 as well
            # as F3.  Without this, F2 far-row truth portals were recognized
            # six times but every clear short ENTRY leg remained rejected by
            # the stale floor-local 2-D endpoint cell.
            manager.allow_truth_exploration_when_registration_lost = True
        if self._floor_number >= 3:
            # The copied F1 namespace is loaded before the fresh manager is
            # constructed, but an externally supplied private parameter can
            # still win that precedence race.  Make the runtime contract
            # explicit on the object that will actually schedule F3 goals.
            manager.f3_truth_degraded_active = True
            manager.f3_truth_degraded_retry_limit = max(
                0, int(rospy.get_param("~f3_truth_degraded_retry_limit", 2)))
            manager.f3_truth_degraded_retry_count = 0
            manager.f3_truth_seed_goal_pending = True
            manager.f3_truth_seed_goal_selected_count = 0
            rospy.loginfo(
                "[third_floor] runtime truth-degraded exploration enabled "
                "with %d bounded first-goal retries",
                manager.f3_truth_degraded_retry_limit)
            if int(getattr(manager, "floor_number", 0)) != 3:
                raise RuntimeError(
                    "F3 BaselineExplorationManager floor_number contract failed")
        corridor_entry_anchor = self._seed_baseline_corridor_entry_anchor(
            manager)
        terminal_stair_return_anchor = (
            self._seed_f3_terminal_stair_return_anchor(
                manager, route, corridor_entry_anchor)
            if self._floor_number >= 3 else None)
        self._exploration_started_wall_time = time.time()
        self._write_handoff(
            self._event("EXPLORATION_START"),
            exploration_started_wall_time=self._exploration_started_wall_time,
            corridor_entry_anchor=(
                list(corridor_entry_anchor)
                if corridor_entry_anchor is not None else None),
            terminal_stair_return_anchor=(
                list(terminal_stair_return_anchor)
                if terminal_stair_return_anchor is not None else None))
        self._state_pub.publish(String(data=self._event("EXPLORATION_START")))
        with self._lock:
            self._active_manager = manager
            self._truth_floor_loss_count = 0
            self._truth_floor_loss_reported = False
        self._mission_stage = "BASELINE_EXPLORATION"
        try:
            manager.run()
        finally:
            with self._lock:
                self._active_manager = None
        self._mission_stage = "FINALIZE_EXPLORATION"
        # A failed F3 EXIT may leave the room scheduler's transaction active
        # even though the online manager has already terminated (run26:
        # ROOM_EXIT_TRUTH_CROSSING_UNCONFIRMED).  That strict defect must not
        # veto the trip home.  Recover through the physical doorway to the
        # corridor once; if that bounded route cannot converge, use the same
        # already-validated truth seam correction as upper-floor ingress.
        # Only then clear the live transaction.  No visited/completed flag is
        # added, so room/dual-view/EXIT credit remains false.
        if (self._floor_number >= 3 and
                manager.room_scheduler.active_door is not None):
            recovered_to_corridor = self._truth_return_to_corridor_anchor(
                corridor_entry_anchor, timeout_sec=45.0)
            recovery_mode = "physical_portal_return"
            if not recovered_to_corridor:
                with self._lock:
                    recovery_pose = (tuple(self._truth_pose)
                                     if self._truth_pose else None)
                recovered_to_corridor = bool(
                    recovery_pose is not None and
                    self._truth_reposition_to_corridor(
                        corridor_entry_anchor, recovery_pose))
                recovery_mode = "bounded_truth_corridor_correction"
            if recovered_to_corridor:
                manager.room_scheduler.abort_active_room(
                    manager.elapsed(),
                    "f3_return_continuity_after_unconfirmed_exit")
                manager.corridor_established = True
                manager.corridor_terminal_return_latched = True
                manager.termination_reason = (
                    "STAIR_CORRIDOR_EXIT_HANDOFF_TRUTH_REQUESTED")
                rospy.logwarn(
                    "[third_floor] released active room for return via %s; "
                    "strict room/EXIT credit remains false", recovery_mode)
        exited_room_count = sum(
            bool(door.visited)
            for door in manager.room_scheduler.detector.doors)
        physical_exit_evidence_count = max(
            exited_room_count,
            self._physical_room_exit_evidence_count(manager))
        successful_returns = {
            "STAIR_CORRIDOR_EXIT_HANDOFF",
            "STAIR_CORRIDOR_EXIT_HANDOFF_TRUTH_GATE",
            "STAIR_CORRIDOR_EXIT_HANDOFF_TRUTH_REQUESTED",
            "STAIR_LOBBY_HANDOFF_LOCALIZATION_FALLBACK",
            "STAIR_WAIT_ZONE_REACHED",
        }
        # The manager's terminal return token is emitted only after a
        # measured corridor/stair-anchor arrival. Attach that final
        # centreline recovery to the last room before strict acceptance. This
        # belongs here, where the concrete manager instance exists; putting it
        # in the static authorization helper caused the F2 partial-return
        # path to raise NameError before it could trigger F2->F3.
        if str(manager.termination_reason) in successful_returns:
            manager.room_scheduler.confirm_corridor_centerline_recovered(
                manager.elapsed(), manager.pose,
                "baseline_terminal_corridor_return_gate")
        manager_emergency_minimum = max(1, int(getattr(
            manager,
            "emergency_partial_floor_handoff_minimum_exited_rooms",
            self._emergency_partial_handoff_minimum_exited_rooms)))
        effective_emergency_minimum = min(
            self._emergency_partial_handoff_minimum_exited_rooms,
            manager_emergency_minimum)
        # Flow-continuity recovery: a real room transaction plus a bounded
        # corridor return is enough to hand ownership to the next stair while
        # the floor remains explicitly partial.  Strict 4/4 acceptance still
        # uses door.visited and is never upgraded by this evidence count.
        if (self._floor_number == 2 and
                self._allow_partial_floor_handoff and
                physical_exit_evidence_count >=
                effective_emergency_minimum and
                str(manager.termination_reason) not in successful_returns and
                self._truth_return_to_corridor_anchor(corridor_entry_anchor)):
            manager.termination_reason = (
                "STAIR_LOBBY_HANDOFF_LOCALIZATION_FALLBACK")
            manager.corridor_terminal_return_latched = True
        # F3 owns a dedicated 180 s truth-guided route from any safe corridor
        # pose to the stair lip.  Do not first spend 45 s on the generic
        # corridor-entry-anchor recovery: run17 timed out in that redundant
        # pre-route and converted a recoverable 3/4 result into terminal
        # THIRD_FLOOR_EXPLORATION_FAILED, so descent never received a token.
        # This changes only physical continuity; strict 4/4 and hazard
        # acceptance below remain false for every partial room.
        f3_physical_return_continuity = bool(
            self._allow_partial_floor_handoff and
            self._f3_physical_return_safe(
                manager, physical_exit_evidence_count))
        if (f3_physical_return_continuity and
                str(manager.termination_reason) not in successful_returns):
            manager.termination_reason = (
                "STAIR_CORRIDOR_EXIT_HANDOFF_TRUTH_REQUESTED")
            manager.corridor_terminal_return_latched = True
        # The embedded manager is the component that observed the terminal
        # planner condition, verified the flat corridor pose and armed its
        # bounded emergency continuation.  Do not raise its explicit one-room
        # safety threshold back to this wrapper's ordinary three-room partial
        # preference.  run24 emitted TRUTH_REQUESTED at F2 with one measured
        # EXIT, then the wrapper rejected it as 1 < 3 and prevented F3 and the
        # entire trip home.  This effective threshold changes locomotion
        # continuity only; strict_room_contract below still records 1/4.
        handoff_authorized = self._exploration_handoff_authorized(
            manager.termination_reason,
            physical_exit_evidence_count,
            manager.room_target_count,
            manager.corridor_terminal_return_latched,
            self._allow_partial_floor_handoff,
            self._partial_floor_handoff_minimum_exited_rooms,
            effective_emergency_minimum)
        strict_room_contract = self._strict_room_contract_evidence(manager)
        f3_stair_return_evidence = None
        f3_executor_quiesce_evidence = None
        if self._floor_number >= 3:
            strict_four_rooms = bool(
                strict_room_contract["distinct_room_count"] >= 4 and
                strict_room_contract["strict_complete_count"] >= 4)
            # Physical fail-safe return: strict 4/4 remains the only accepted
            # exploration result, but a bounded corridor-end condition after
            # at least the configured partial-room threshold must not strand
            # the dog on F3.  It may return downstairs with an explicit
            # partial contract; reports/visuals keep the run INCOMPLETE.
            bounded_corridor_end_return = bool(
                self._allow_partial_floor_handoff and
                (manager.corridor_terminal_return_latched or
                 str(manager.termination_reason) in successful_returns or
                 f3_physical_return_continuity))
            # On F3, reaching the measured corridor end is itself a terminal
            # navigation condition.  It must start the physical trip home
            # even when room evidence is incomplete; otherwise a failed room
            # merge can strand the dog at the far wall forever.  This only
            # authorizes locomotion continuity.  strict_four_rooms and the
            # completion report remain false, so no missing room receives
            # exploration credit.
            if (bounded_corridor_end_return and
                    str(manager.termination_reason) not in successful_returns):
                manager.termination_reason = (
                    "STAIR_CORRIDOR_EXIT_HANDOFF_TRUTH_REQUESTED")
            handoff_authorized = bool(
                (handoff_authorized and strict_four_rooms) or
                bounded_corridor_end_return)
            if handoff_authorized:
                executor_quiesced, f3_executor_quiesce_evidence = (
                    self._quiesce_exploration_executor_for_f3_return(manager))
                if executor_quiesced:
                    handoff_authorized, f3_stair_return_evidence = (
                        self._third_floor_return_to_stair_wait(
                            route,
                            timeout_sec=
                            self._third_floor_stair_return_timeout_sec))
                else:
                    handoff_authorized = False
                    f3_stair_return_evidence = {
                        "reason": "f3_executor_not_quiesced",
                        "executor_quiesce_evidence":
                            f3_executor_quiesce_evidence,
                    }
        partial_handoff = bool(
            handoff_authorized and
            (physical_exit_evidence_count <
             max(4, int(manager.room_target_count)) or
             (self._floor_number >= 3 and
              strict_room_contract["strict_complete_count"] < 4)))
        emergency_partial_handoff = bool(
            partial_handoff and
            physical_exit_evidence_count <
            self._partial_floor_handoff_minimum_exited_rooms)
        f3_stair_return_ready = bool(
            self._floor_number >= 3 and handoff_authorized and
            f3_stair_return_evidence and
            f3_stair_return_evidence.get("upright_posture_valid") and
            manager.termination_reason in {
                "STAIR_CORRIDOR_EXIT_HANDOFF",
                "STAIR_CORRIDOR_EXIT_HANDOFF_TRUTH_GATE",
                "STAIR_CORRIDOR_EXIT_HANDOFF_TRUTH_REQUESTED",
                "STAIR_LOBBY_HANDOFF_LOCALIZATION_FALLBACK",
                "STAIR_WAIT_ZONE_REACHED",
            })
        outcome = (("THIRD_FLOOR_STAIR_RETURN_READY"
                    if not partial_handoff else
                    "THIRD_FLOOR_STAIR_RETURN_PARTIAL_READY")
                   if f3_stair_return_ready else
                   self._event("EXPLORATION_COMPLETE")
                   if handoff_authorized else
                   self._event("EXPLORATION_FAILED"))
        log = rospy.loginfo if handoff_authorized else rospy.logerr
        log("[%s] termination=%s recognized=%d exited=%d/%d "
            "terminal_return=%s next_stair_handoff=%s",
            self._floor_slug, manager.termination_reason,
            manager.room_scheduler.room_count, exited_room_count,
            manager.room_target_count,
            bool(manager.corridor_terminal_return_latched),
            handoff_authorized)
        self._write_handoff(
            outcome,
            exploration_started_wall_time=self._exploration_started_wall_time,
            termination_reason=manager.termination_reason,
            recognized_room_count=manager.room_scheduler.room_count,
            exited_room_count=exited_room_count,
            physical_exit_evidence_count=physical_exit_evidence_count,
            room_target_count=manager.room_target_count,
            corridor_terminal_return_latched=bool(
                manager.corridor_terminal_return_latched),
            corridor_entry_anchor=(
                list(corridor_entry_anchor)
                if corridor_entry_anchor is not None else None),
            next_stair_handoff_authorized=handoff_authorized,
            strict_room_contract=strict_room_contract,
            f3_stair_return_evidence=f3_stair_return_evidence,
            f3_executor_quiesce_evidence=f3_executor_quiesce_evidence)
        if f3_stair_return_ready:
            self._write_handoff(
                ("THIRD_FLOOR_STAIR_RETURN_PARTIAL_READY"
                 if partial_handoff else
                 "THIRD_FLOOR_STAIR_RETURN_READY"),
                termination_reason=manager.termination_reason,
                recognized_room_count=manager.room_scheduler.room_count,
                exited_room_count=exited_room_count,
                room_target_count=manager.room_target_count,
                next_stair_handoff_authorized=True,
                strict_room_contract=strict_room_contract,
                f3_stair_return_evidence=f3_stair_return_evidence,
                f3_executor_quiesce_evidence=f3_executor_quiesce_evidence,
                partial_handoff=partial_handoff,
                completion_contract=(
                    "four_rooms_physical_two_view_and_corridor_return"
                    if strict_room_contract["strict_complete_count"] >= 4
                    else "bounded_corridor_end_physical_return_incomplete"))
        # Do not write a completion-looking token after a rejected handoff.
        # The previous unconditional write produced SECOND_FLOOR_FULL_FLOOR_HANDOFF
        # even for e.g. exited=1/4, which made the artifact contradict the
        # authoritative next_stair_handoff_authorized=false field and could
        # mislead downstream consumers into starting F3.  A rejected floor
        # remains latched as EXPLORATION_FAILED; only an authorized return may
        # publish PARTIAL_FLOOR_HANDOFF or FULL_FLOOR_HANDOFF.
        if handoff_authorized and self._floor_number < 3:
            self._write_handoff(
                self._event("PARTIAL_FLOOR_HANDOFF" if partial_handoff else
                            "FULL_FLOOR_HANDOFF"),
                partial_handoff=partial_handoff,
                emergency_partial_handoff=emergency_partial_handoff,
                exited_room_count=exited_room_count,
                physical_exit_evidence_count=physical_exit_evidence_count,
                required_room_count=max(4, int(manager.room_target_count)),
                policy=("continue_to_next_floor_after_terminal_2_room_return"
                        if emergency_partial_handoff else
                        "continue_to_next_floor_after_bounded_3_room_return"
                        if partial_handoff else "strict_4_room_completion"))
        elif handoff_authorized and self._floor_number >= 3:
            self._write_handoff(
                outcome,
                handoff_rejected=False,
                partial_handoff=partial_handoff,
                exited_room_count=exited_room_count,
                required_room_count=max(4, int(manager.room_target_count)),
                strict_room_contract=strict_room_contract,
                f3_stair_return_evidence=f3_stair_return_evidence,
                f3_executor_quiesce_evidence=
                    f3_executor_quiesce_evidence,
                policy=("bounded_corridor_end_physical_return_incomplete"
                        if partial_handoff else
                        "strict_4_room_completion"))
        else:
            rejection_reason = "physical_exit_requirement_not_met"
            if self._floor_number >= 3 and f3_stair_return_evidence:
                rejection_reason = str(
                    f3_stair_return_evidence.get("reason") or
                    "f3_stair_wait_return_failed")
            self._write_handoff(
                outcome,
                handoff_rejected=True,
                partial_handoff=False,
                emergency_partial_handoff=False,
                exited_room_count=exited_room_count,
                required_room_count=max(4, int(manager.room_target_count)),
                strict_room_contract=strict_room_contract,
                f3_stair_return_evidence=f3_stair_return_evidence,
                f3_executor_quiesce_evidence=
                    f3_executor_quiesce_evidence,
                policy="strict_4_room_completion",
                rejection_reason=rejection_reason)
        # Publish the latched stair trigger BEFORE generating the combined
        # visualization.  The visualization subprocess renders 13 PNGs and can
        # take tens of seconds; if the launch is torn down during that window
        # (run ..._850s_v4 had junior_ctrl segfault while F2 was finalizing),
        # the F2->F3 stair manager never sees SECOND_FLOOR_EXPLORATION_COMPLETE
        # and the whole mission dies on the floor-2 landing.  The completion
        # token is one-shot on the topic, so publish it first; the offline
        # figures are diagnostic only and can finish afterwards.
        # On failure, schedule the detached renderer before publishing the
        # failure token: the required next-floor process exits immediately
        # and roslaunch otherwise tears this node down before it can spawn.
        if not handoff_authorized:
            self._generate_combined_visualization()
        self._state_pub.publish(String(data=outcome))
        with self._lock:
            self._terminal_outcome_published = True
            self._terminal_failure_published = not handoff_authorized
        # Generate the completed floor snapshot after publishing the latched
        # stair trigger.  This guarantees F2 figures 19/20 even when a failure
        # prevents F3 from starting, and avoids competing with stair control
        # for CPU while the transition is in progress.
        self._mission_stage = "GENERATE_VISUALIZATION"
        if not handoff_authorized:
            self._generate_combined_visualization()
        elif self._floor_number >= 3:
            # A successful/partial F3 handoff transfers command ownership to
            # the physical descent worker.  Schedule the required 12/13/14
            # render now, but delay its CPU load until the accepted descent
            # watchdog window has elapsed.  The renderer is detached and
            # therefore survives the normal roslaunch shutdown at home.
            self._generate_combined_visualization(initial_delay=180.0)
        self._mission_stage = ("COMPLETE" if handoff_authorized else "FAILED")
        # A floor worker is not the roslaunch supervisor.  On a terminal F2
        # rejection there is no valid next-floor owner; on any terminal F3
        # result the requested mission has reached its end.  Explicitly shut
        # down the launch after the handoff artifact and detached renderer
        # have been scheduled, otherwise Gazebo and the low-level controller
        # remain alive after all mission workers exit and the run can appear
        # hung indefinitely.
        # Descent owns shutdown only after F3 has published a successful
        # stair-return handoff.  Deferring an F3 failure leaves every mission
        # worker idle while Gazebo continues forever (run ..._154500 failed
        # before F3 started and remained in EXPLORATION_FAILED).
        descent_owns_terminal_shutdown = bool(
            handoff_authorized and self._floor_number >= 3 and
            self._defer_terminal_shutdown_to_descent)
        terminal_mission = bool(
            (not handoff_authorized or self._floor_number >= 3) and
            not descent_owns_terminal_shutdown)
        if terminal_mission and not rospy.is_shutdown():
            rospy.loginfo(
                "[%s] terminal mission result published; shutting down "
                "roslaunch cleanly (authorized=%s)",
                self._floor_slug, handoff_authorized)
            rospy.signal_shutdown(
                "{}_terminal_mission_result".format(self._floor_slug))
        return handoff_authorized


if __name__ == "__main__":
    rospy.init_node("second_floor_exploration_manager")
    SecondFloorMission().run()
