#!/usr/bin/env python3
"""Headless, GT-guided Room0 loop used only for SLAM diagnostics."""

import json
import math
import os
import sys
import threading
import time

import rospy
import sensor_msgs.point_cloud2 as point_cloud2
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState, Joy, PointCloud, PointCloud2
from std_msgs.msg import Bool, String
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# Catkin's devel-space wrapper executes this source file without adding the
# source script directory to sys.path. Keep the pure helper beside the node.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from room0_mapping_core import (  # noqa: E402
    control_for_target,
    generate_room0_route,
    generate_room0_coverage_loop_route,
    generate_room0_small_loop_route,
    is_fallen,
    rl_control_for_target,
    rl_holonomic_control_for_target,
    route_progress_error,
)


BUTTON_STAND = 1
BUTTON_RL_CMD_VEL = 3
BUTTON_TROTTING = 4
BUTTON_RESET = 10


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class Room0MappingController:
    def __init__(self):
        self.output_dir = os.path.abspath(rospy.get_param("~output_dir"))
        self.layout_file = os.path.abspath(rospy.get_param("~layout_file"))
        self.route_variant = str(rospy.get_param("~route_variant", "full_loop"))
        if self.route_variant == "small_loop":
            route_generator = generate_room0_small_loop_route
        elif self.route_variant == "coverage_loop":
            route_generator = generate_room0_coverage_loop_route
        else:
            route_generator = generate_room0_route
        self.route, self.room, self.floor = route_generator(self.layout_file)
        self.route_waypoint_limit = int(
            rospy.get_param("~route_waypoint_limit", 0)
        )
        if self.route_waypoint_limit > 0:
            self.route = self.route[:self.route_waypoint_limit]
        self.motion_mode = str(rospy.get_param("~motion_mode", "classic_trot"))
        self.require_exit = bool(rospy.get_param("~require_exit", True))
        self.mission_timeout = float(rospy.get_param("~mission_timeout", 240.0))
        self.waypoint_timeout = float(rospy.get_param("~waypoint_timeout", 55.0))
        self.settle_seconds = float(rospy.get_param("~settle_seconds", 5.0))
        self.stand_seconds = float(rospy.get_param("~stand_seconds", 5.0))
        self.trotting_warmup_seconds = float(
            rospy.get_param("~trotting_warmup_seconds", 4.5)
        )
        self.pre_motion_dwell_seconds = float(
            rospy.get_param("~pre_motion_dwell_seconds", 4.0)
        )
        self.stand_timeout_seconds = float(rospy.get_param("~stand_timeout_seconds", 12.0))
        self.stable_pose_seconds = float(rospy.get_param("~stable_pose_seconds", 1.5))
        # A1's Gazebo base_w origin settles near z=0.257 m in FixedStand.
        # This is the base-link origin, not the nominal leg/body height.
        self.stand_min_z = float(rospy.get_param("~stand_min_z", 0.23))
        self.maximum_cloud_points = int(rospy.get_param("~maximum_cloud_points", 250000))
        self.stagnation_seconds = float(rospy.get_param("~stagnation_seconds", 6.0))
        self.max_stall_recoveries = int(rospy.get_param("~max_stall_recoveries", 2))
        self.scan_pause_seconds = float(rospy.get_param("~scan_pause_seconds", 1.2))
        self.route_deviation_limit = float(rospy.get_param("~route_deviation_limit", 1.5))
        self.scan_pause_waypoints = set(
            rospy.get_param(
                "~scan_pause_waypoints",
                ["door_align", "door_crossing_inside", "loop_start", "loop_closed"],
            )
        )
        self.forward_limits = {
            "corridor": float(rospy.get_param("~corridor_joy_limit", 0.52)),
            "door": float(rospy.get_param("~door_joy_limit", 0.40)),
            "room": float(rospy.get_param("~room_joy_limit", 0.55)),
        }
        self.kinematic_speeds = {
            "corridor": float(rospy.get_param("~kinematic_corridor_speed", 0.65)),
            "door": float(rospy.get_param("~kinematic_door_speed", 0.35)),
            "room": float(rospy.get_param("~kinematic_room_speed", 0.50)),
        }
        self.rl_speeds = {
            "corridor": float(rospy.get_param("~rl_corridor_speed", 0.35)),
            "door": float(rospy.get_param("~rl_door_speed", 0.20)),
            "room": float(rospy.get_param("~rl_room_speed", 0.25)),
        }
        self.rl_maximum_yaw = float(rospy.get_param("~rl_maximum_yaw", 0.35))
        self.rl_arc_turning = bool(rospy.get_param("~rl_arc_turning", False))
        self.rl_holonomic = bool(rospy.get_param("~rl_holonomic", False))
        self.rl_zero_warmup_seconds = float(
            rospy.get_param("~rl_zero_warmup_seconds", 3.0)
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self.lock = threading.Lock()
        self.gt = None
        self.est = None
        self.cloud_message = None
        self.gt_samples = []
        self.est_samples = []
        self.last_gt_record = None
        self.last_est_record = None
        self.imu_seen = False
        self.scan_seen = False
        self.registered_seen = False
        self.joints_seen = False
        self.classic_controller_ready = False
        self.topic_counts = {"scan": 0, "imu": 0, "registered": 0, "map": 0, "odom": 0}
        self.phase = "INIT"
        self.active_index = 0
        self.arrivals = []
        self.failure_reason = ""
        self.fallen = False
        self.entered_room = False
        self.loop_closed = False
        self.exited_room = False
        self.started_stamp = None
        self.route_finished_stamp = None
        self.finished_stamp = None
        self.motion_started_stamp = None
        self.rl_abort_reason = ""
        self.settle_estimates = []
        self.last_reception = {"gt": 0.0, "imu": 0.0, "scan": 0.0,
                               "registered": 0.0, "map": 0.0, "odom": 0.0}
        self.last_state_log = 0.0
        self.last_logged_phase = ""
        self.parameter_snapshot = {
            "motion_mode": self.motion_mode,
            "route_variant": self.route_variant,
            "route_waypoint_limit": self.route_waypoint_limit,
            "require_exit": self.require_exit,
            "mission_timeout": self.mission_timeout,
            "waypoint_timeout": self.waypoint_timeout,
            "settle_seconds": self.settle_seconds,
            "stand_seconds": self.stand_seconds,
            "trotting_warmup_seconds": self.trotting_warmup_seconds,
            "pre_motion_dwell_seconds": self.pre_motion_dwell_seconds,
            "stand_timeout_seconds": self.stand_timeout_seconds,
            "stable_pose_seconds": self.stable_pose_seconds,
            "stand_min_z": self.stand_min_z,
            "route_deviation_limit": self.route_deviation_limit,
            "forward_limits": self.forward_limits,
            "rl_speeds": self.rl_speeds,
            "rl_maximum_yaw": self.rl_maximum_yaw,
            "rl_arc_turning": self.rl_arc_turning,
            "rl_holonomic": self.rl_holonomic,
            "rl_zero_warmup_seconds": self.rl_zero_warmup_seconds,
        }

        self.joy_pub = rospy.Publisher("/joy", Joy, queue_size=4)
        self.desired_cmd_pub = rospy.Publisher(
            "/simenv/desired_cmd_vel", Twist, queue_size=5
        )
        self.rl_armed_pub = rospy.Publisher(
            "/simenv/rl_motion_armed", Bool, queue_size=1, latch=True
        )
        self.state_pub = rospy.Publisher(
            "/simenv/quick_validation_state", String, queue_size=1, latch=True
        )
        self.done_pub = rospy.Publisher(
            "/simenv/mission_complete", Bool, queue_size=1, latch=True
        )
        self.set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        rospy.Subscriber("/ground_truth/base_w", Odometry, self._on_gt, queue_size=50)
        rospy.Subscriber("/Odometry", Odometry, self._on_est, queue_size=50)
        rospy.Subscriber("/livox/imu", Imu, self._on_imu, queue_size=100)
        rospy.Subscriber("/scan", PointCloud, self._on_scan, queue_size=2)
        rospy.Subscriber("/cloud_registered", PointCloud2, self._on_registered, queue_size=2)
        rospy.Subscriber("/cloud_map", PointCloud2, self._on_map, queue_size=1)
        rospy.Subscriber("/a1_gazebo/joint_states", JointState, self._on_joints, queue_size=5)
        rospy.Subscriber(
            "/simenv/classic_controller_ready", Bool,
            self._on_classic_controller_ready, queue_size=1,
        )
        rospy.Subscriber(
            "/simenv/rl_motion_abort", String, self._on_rl_motion_abort, queue_size=1
        )
        self.rl_armed_pub.publish(Bool(data=False))
        rospy.on_shutdown(self._stop)

    def _on_classic_controller_ready(self, message):
        with self.lock:
            self.classic_controller_ready = bool(message.data)

    def _on_rl_motion_abort(self, message):
        try:
            payload = json.loads(message.data)
            reason = str(payload.get("reason", "rl_motion_abort"))
        except (TypeError, ValueError):
            reason = "rl_motion_abort"
        with self.lock:
            self.rl_abort_reason = reason

    @staticmethod
    def _sample(message):
        pose = message.pose.pose
        stamp = message.header.stamp.to_sec() or rospy.Time.now().to_sec()
        q = pose.orientation
        roll, pitch, yaw = euler_from_quaternion((q.x, q.y, q.z, q.w))
        return {
            "t_abs": float(stamp),
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "z": float(pose.position.z),
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
        }

    def _append_sample(self, bucket, sample, last_name):
        previous_time = getattr(self, last_name)
        if previous_time is None or sample["t_abs"] - previous_time >= 0.08:
            bucket.append(sample)
            setattr(self, last_name, sample["t_abs"])

    def _on_gt(self, message):
        sample = self._sample(message)
        with self.lock:
            self.gt = sample
            self.last_reception["gt"] = time.monotonic()
            self._append_sample(self.gt_samples, sample, "last_gt_record")

    def _on_est(self, message):
        sample = self._sample(message)
        with self.lock:
            self.est = sample
            self.last_reception["odom"] = time.monotonic()
            self.topic_counts["odom"] += 1
            self._append_sample(self.est_samples, sample, "last_est_record")
            if self.phase == "SETTLE":
                self.settle_estimates.append(sample)

    def _on_imu(self, _message):
        with self.lock:
            self.imu_seen = True
            self.last_reception["imu"] = time.monotonic()
            self.topic_counts["imu"] += 1

    def _on_scan(self, _message):
        with self.lock:
            self.scan_seen = True
            self.last_reception["scan"] = time.monotonic()
            self.topic_counts["scan"] += 1

    def _on_registered(self, _message):
        with self.lock:
            self.registered_seen = True
            self.last_reception["registered"] = time.monotonic()
            self.topic_counts["registered"] += 1

    def _on_map(self, message):
        with self.lock:
            self.cloud_message = message
            self.last_reception["map"] = time.monotonic()
            self.topic_counts["map"] += 1

    def _on_joints(self, message):
        with self.lock:
            self.joints_seen = len(message.name) >= 12

    def _joy(self, forward=0.0, yaw_axis=0.0, button=None):
        message = Joy()
        message.header.stamp = rospy.Time.now()
        message.axes = [0.0, float(forward), 0.0, float(yaw_axis), 0.0, 0.0]
        message.buttons = [0] * 11
        if button is not None:
            message.buttons[button] = 1
        self.joy_pub.publish(message)

    def _desired_cmd(self, linear=0.0, yaw=0.0, lateral=0.0):
        message = Twist()
        message.linear.x = float(linear)
        message.linear.y = float(lateral)
        message.angular.z = float(yaw)
        self.desired_cmd_pub.publish(message)

    def _mark_motion_started(self):
        if self.motion_started_stamp is not None:
            return
        self.motion_started_stamp = rospy.Time.now().to_sec()
        path = os.path.join(self.output_dir, "motion_started.json")
        temporary = path + ".tmp"
        self._write_json(temporary, {
            "schema": "simenv_room0_effective_run_v1",
            "motion_mode": self.motion_mode,
            "stamp": self.motion_started_stamp,
        })
        os.replace(temporary, path)

    def _stop(self):
        if not hasattr(self, "joy_pub"):
            return
        try:
            self.rl_armed_pub.publish(Bool(data=False))
            for _ in range(3):
                self._desired_cmd()
                self._joy(button=BUTTON_STAND)
                time.sleep(0.03)
        except Exception:
            pass

    def _state(self, phase, **extra):
        self.phase = phase
        payload = {
            "schema": "simenv_room0_mapping_state_v1",
            "phase": phase,
            "active_waypoint": self.active_index,
            "waypoint_count": len(self.route),
            "diagnostic_gt_control": True,
            "motion_mode": self.motion_mode,
            "route_variant": self.route_variant,
        }
        payload.update(extra)
        self.state_pub.publish(String(data=json.dumps(payload, sort_keys=True)))
        now = time.monotonic()
        if phase != self.last_logged_phase or now - self.last_state_log >= 2.0:
            rospy.loginfo("Room0 mapping phase=%s %s", phase, extra)
            self.last_logged_phase = phase
            self.last_state_log = now

    def _ready(self):
        with self.lock:
            ready = (
                self.gt is not None
                and self.est is not None
                and self.cloud_message is not None
                and self.imu_seen
                and self.scan_seen
                and self.registered_seen
                and self.joints_seen
                and self.classic_controller_ready
                and self.joy_pub.get_num_connections() > 0
            )
        if self.motion_mode == "official_rl":
            ready = ready and self.desired_cmd_pub.get_num_connections() > 0
        return ready

    def _wait_ready(self):
        self._state("WAIT_READY")
        deadline = time.monotonic() + 75.0
        rate = rospy.Rate(10)
        reset_sent = False
        reset_release_cycles = 0
        reset_completed_at = None
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            with self.lock:
                reset_prerequisites = (
                    self.gt is not None and self.joints_seen and
                    self.classic_controller_ready and
                    self.joy_pub.get_num_connections() > 0
                )
            if self.motion_mode == "official_rl":
                # The official RL baseline uses the model's spawn pose and
                # never injects a Gazebo RESET discontinuity into the IMU.
                reset_sent = bool(reset_prerequisites)
                reset_completed_at = reset_completed_at or (
                    time.monotonic() if reset_sent else None
                )
                self._desired_cmd()
                self._joy(button=BUTTON_STAND)
            elif not reset_sent and reset_prerequisites:
                # Reset before the delayed FAST-LIO sensor feed begins.  This
                # makes the spawn/joint initial condition deterministic without
                # injecting a pose discontinuity into the SLAM estimator.
                self._joy(button=BUTTON_RESET)
                reset_release_cycles += 1
                if reset_release_cycles == 1:
                    rospy.loginfo("Room0 pre-SLAM RESET pulse sequence started")
                if reset_release_cycles >= 1:
                    reset_sent = True
                    reset_completed_at = time.monotonic()
                    rospy.loginfo("Room0 pre-SLAM RESET pulse sequence completed")
            else:
                self._joy(button=BUTTON_STAND)
            reset_settled = (
                reset_sent and reset_completed_at is not None and
                time.monotonic() - reset_completed_at >= (
                    0.5 if self.motion_mode == "official_rl" else 4.0
                )
            )
            if reset_settled and self._ready():
                return True
            rate.sleep()
        self.failure_reason = "readiness_timeout"
        return False

    def _reset_and_stand(self):
        # The model has just been spawned at the configured start pose.
        # Resetting it *after* FAST-LIO initialization creates a discontinuous
        # IMU impulse and corrupts the estimator, so only command fixed stand.
        self._state("STAND")
        start = time.monotonic()
        stable_since = None
        gt = None
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and time.monotonic() - start < self.stand_timeout_seconds:
            self._joy(button=BUTTON_STAND)
            with self.lock:
                gt = dict(self.gt) if self.gt else None
            stable = (
                gt is not None
                and gt["z"] >= self.stand_min_z
                and abs(gt["roll"]) <= 0.18
                and abs(gt["pitch"]) <= 0.18
            )
            if stable:
                stable_since = stable_since or time.monotonic()
                if (
                    time.monotonic() - stable_since >= self.stable_pose_seconds
                    and time.monotonic() - start >= self.stand_seconds
                ):
                    return True
            else:
                stable_since = None
            rate.sleep()
        self.failure_reason = "failed_to_reach_stable_stand"
        self.fallen = bool(gt and is_fallen(gt["z"], gt["roll"], gt["pitch"]))
        return False

    def _run_route(self):
        # State_Trotting lifts the body and starts its gait generator on entry.
        # Give it a zero-velocity transition before applying translation.
        self._state("TROTTING_WARMUP")
        warmup_start = rospy.Time.now()
        warmup_rate = rospy.Rate(20)
        warmup_cycles = 0
        while (not rospy.is_shutdown() and
               (rospy.Time.now() - warmup_start).to_sec() < self.trotting_warmup_seconds):
            # START is edge-like in the legacy FSM and also doubles as its
            # data-record button. Pulse it only long enough to enter Trotting;
            # holding it causes repeated file opens inside a 20 Hz side thread.
            self._joy(button=BUTTON_TROTTING if warmup_cycles < 3 else None)
            warmup_cycles += 1
            warmup_rate.sleep()
        # Extra zero-velocity dwell so the first swing pair finishes before
        # longitudinal Joy is applied (reduces tip-over on heavily loaded hosts).
        settle_start = rospy.Time.now()
        while (not rospy.is_shutdown() and
               (rospy.Time.now() - settle_start).to_sec() < self.pre_motion_dwell_seconds):
            self._joy()
            warmup_rate.sleep()
        self._state("ROUTE")
        self.started_stamp = rospy.Time.now().to_sec()
        mission_start = rospy.Time.now()
        waypoint_start = rospy.Time.now()
        last_progress_time = rospy.Time.now()
        last_progress_xy = None
        translation_attempted = False
        turning_for_heading = False
        stall_recoveries = 0
        stall_recoveries = 0
        commanded_forward = 0.0
        commanded_yaw = 0.0
        stall_recoveries = 0
        rate = rospy.Rate(20)

        while not rospy.is_shutdown() and self.active_index < len(self.route):
            mission_elapsed = (rospy.Time.now() - mission_start).to_sec()
            if mission_elapsed > self.mission_timeout:
                self.failure_reason = "mission_timeout"
                return False
            with self.lock:
                gt = dict(self.gt) if self.gt else None
                gt_age = time.monotonic() - self.last_reception["gt"]
            if gt is None:
                self.failure_reason = "ground_truth_stale"
                return False
            if gt_age > 0.5:
                self.failure_reason = "ground_truth_stale"
                return False
            if is_fallen(gt["z"], gt["roll"], gt["pitch"]):
                self.failure_reason = "robot_fallen"
                self.fallen = True
                return False
            waypoint = self.route[self.active_index]
            forward, yaw_axis, distance, heading_error = control_for_target(
                gt["x"], gt["y"], gt["yaw"],
                (waypoint["x"], waypoint["y"]), waypoint["zone"], self.forward_limits,
            )
            # State_Trotting already limits velocity changes in calcCmd().
            # An additional very slow Joy ramp kept the gait just above its
            # stepping threshold while making almost no forward progress,
            # producing unstable in-place trot cycles.  The direct set-point
            # below matches the configuration that completed 56.58 m in run
            # 20260716T032859Z.
            commanded_forward = forward
            commanded_yaw = yaw_axis
            tolerance = 0.30 if waypoint["zone"] == "door" else 0.45
            if distance <= tolerance:
                self.arrivals.append({
                    "index": self.active_index,
                    "name": waypoint["name"],
                    "error_m": round(distance, 4),
                    "t": round(rospy.Time.now().to_sec() - self.started_stamp, 3),
                })
                if waypoint["name"] == "door_crossing_inside":
                    self.entered_room = True
                elif waypoint["name"] == "loop_closed":
                    self.loop_closed = True
                elif waypoint["name"] == "exit_outside":
                    self.exited_room = True
                self.active_index += 1
                waypoint_start = rospy.Time.now()
                last_progress_time = rospy.Time.now()
                last_progress_xy = None
                translation_attempted = False
                turning_for_heading = False
                stall_recoveries = 0
                stall_recoveries = 0
                self._joy()
                commanded_forward = 0.0
                commanded_yaw = 0.0
                self._state("ROUTE", reached=waypoint["name"])
                # Brief stationary dwell at geometric bottlenecks so FAST-LIO
                # can lock door/wall constraints before the next translation.
                if (
                    waypoint["name"] in self.scan_pause_waypoints
                    and self.scan_pause_seconds > 0.0
                ):
                    pause_until = rospy.Time.now() + rospy.Duration(self.scan_pause_seconds)
                    while not rospy.is_shutdown() and rospy.Time.now() < pause_until:
                        with self.lock:
                            gt = dict(self.gt) if self.gt else gt
                        if is_fallen(gt["z"], gt["roll"], gt["pitch"]):
                            self.failure_reason = "robot_fallen"
                            self.fallen = True
                            return False
                        self._joy()
                        rate.sleep()
                rate.sleep()
                continue

            elapsed_waypoint = (rospy.Time.now() - waypoint_start).to_sec()
            if elapsed_waypoint > self.waypoint_timeout:
                self.failure_reason = "waypoint_timeout:{}".format(waypoint["name"])
                return False
            deviation = route_progress_error(gt["x"], gt["y"], self.route, self.active_index)
            if self.active_index > 0 and deviation > self.route_deviation_limit:
                self.failure_reason = "route_deviation:{}:{:.2f}".format(
                    waypoint["name"], deviation
                )
                return False

            if forward > 0.12:
                if last_progress_xy is None:
                    last_progress_xy = (gt["x"], gt["y"])
                    last_progress_time = rospy.Time.now()
                elif math.hypot(
                    gt["x"] - last_progress_xy[0], gt["y"] - last_progress_xy[1]
                ) >= 0.08:
                    last_progress_xy = (gt["x"], gt["y"])
                    last_progress_time = rospy.Time.now()
                elif (
                    rospy.Time.now() - last_progress_time
                ).to_sec() > self.stagnation_seconds:
                    if stall_recoveries >= self.max_stall_recoveries:
                        self.failure_reason = "robot_stagnant:{}".format(waypoint["name"])
                        return False
                    stall_recoveries += 1
                    rospy.logwarn(
                        "Room0 stall recovery %s at %s",
                        stall_recoveries,
                        waypoint["name"],
                    )
                    recover_until = rospy.Time.now() + rospy.Duration(0.8)
                    while not rospy.is_shutdown() and rospy.Time.now() < recover_until:
                        with self.lock:
                            gt = dict(self.gt) if self.gt else gt
                        if is_fallen(gt["z"], gt["roll"], gt["pitch"]):
                            self.failure_reason = "robot_fallen"
                            self.fallen = True
                            return False
                        self._joy(forward=-0.14, yaw_axis=0.0)
                        rate.sleep()
                    pause_until = rospy.Time.now() + rospy.Duration(0.5)
                    while not rospy.is_shutdown() and rospy.Time.now() < pause_until:
                        self._joy()
                        rate.sleep()
                    commanded_forward = 0.0
                    commanded_yaw = 0.0
                    last_progress_xy = None
                    last_progress_time = rospy.Time.now()
                    continue
            else:
                last_progress_xy = None
                last_progress_time = rospy.Time.now()

            self._joy(forward=commanded_forward, yaw_axis=commanded_yaw)
            self._state(
                "ROUTE",
                target=waypoint["name"],
                distance=round(distance, 3),
                heading_error=round(heading_error, 3),
            )
            rate.sleep()
        return self.active_index == len(self.route)

    def _run_official_rl_route(self):
        """Track the diagnostic route through the official plane RL policy."""
        self._state("RL_ZERO_WARMUP")
        rate = rospy.Rate(20)
        warmup_start = rospy.Time.now()
        cycles = 0
        unstable_since = None
        while (
            not rospy.is_shutdown()
            and (rospy.Time.now() - warmup_start).to_sec() < self.rl_zero_warmup_seconds
        ):
            self._desired_cmd()
            self._joy(button=BUTTON_RL_CMD_VEL if cycles < 3 else None)
            cycles += 1
            with self.lock:
                gt = dict(self.gt) if self.gt else None
            unstable = (
                gt is None or gt["z"] < 0.27
                or abs(gt["roll"]) > 0.18 or abs(gt["pitch"]) > 0.18
            )
            if unstable:
                unstable_since = unstable_since or time.monotonic()
                if time.monotonic() - unstable_since > 0.5:
                    self.failure_reason = "rl_warmup_pose_unstable"
                    return False
            else:
                unstable_since = None
            rate.sleep()

        with self.lock:
            gt = dict(self.gt) if self.gt else None
        if (
            gt is None or gt["z"] < 0.27
            or abs(gt["roll"]) > 0.18 or abs(gt["pitch"]) > 0.18
        ):
            self.failure_reason = "rl_warmup_pose_unstable"
            return False

        self.rl_armed_pub.publish(Bool(data=True))
        self._state("RL_ROUTE")
        self.started_stamp = rospy.Time.now().to_sec()
        mission_start = rospy.Time.now()
        waypoint_start = rospy.Time.now()
        last_progress_time = rospy.Time.now()
        last_progress_xy = None
        translation_attempted = False
        turning_for_heading = False
        stall_recoveries = 0

        while not rospy.is_shutdown() and self.active_index < len(self.route):
            if (rospy.Time.now() - mission_start).to_sec() > self.mission_timeout:
                self.failure_reason = "mission_timeout"
                return False
            with self.lock:
                gt = dict(self.gt) if self.gt else None
                gt_age = time.monotonic() - self.last_reception["gt"]
                abort_reason = self.rl_abort_reason
            if abort_reason:
                self.failure_reason = "rl_safety_abort:" + abort_reason
                return False
            if gt is None or gt_age > 0.30:
                self.failure_reason = "ground_truth_stale"
                return False
            if is_fallen(gt["z"], gt["roll"], gt["pitch"]):
                self.failure_reason = "robot_fallen"
                self.fallen = True
                return False

            waypoint = self.route[self.active_index]
            lateral = 0.0
            if self.rl_holonomic:
                linear, lateral, distance, heading_error = (
                    rl_holonomic_control_for_target(
                        gt["x"], gt["y"], gt["yaw"],
                        (waypoint["x"], waypoint["y"]), waypoint["zone"],
                        self.rl_speeds,
                    )
                )
                yaw_rate = 0.0
            else:
                linear, yaw_rate, distance, heading_error = rl_control_for_target(
                    gt["x"], gt["y"], gt["yaw"],
                    (waypoint["x"], waypoint["y"]), waypoint["zone"],
                    self.rl_speeds, self.rl_maximum_yaw, turning_for_heading,
                    self.rl_arc_turning,
                )
            turning_for_heading = abs(linear) < 1e-6 and abs(yaw_rate) > 1e-6
            tolerance = 0.30 if waypoint["zone"] == "door" else 0.45
            if distance <= tolerance:
                self.arrivals.append({
                    "index": self.active_index,
                    "name": waypoint["name"],
                    "error_m": round(distance, 4),
                    "t": round(rospy.Time.now().to_sec() - self.started_stamp, 3),
                })
                if waypoint["name"] == "door_crossing_inside":
                    self.entered_room = True
                elif waypoint["name"] == "loop_closed":
                    self.loop_closed = True
                elif waypoint["name"] == "exit_outside":
                    self.exited_room = True
                self.active_index += 1
                waypoint_start = rospy.Time.now()
                last_progress_time = rospy.Time.now()
                last_progress_xy = None
                translation_attempted = False
                turning_for_heading = False
                stall_recoveries = 0
                self._desired_cmd()
                self._state("RL_ROUTE", reached=waypoint["name"])
                if (
                    waypoint["name"] in self.scan_pause_waypoints
                    and self.scan_pause_seconds > 0.0
                ):
                    pause_until = rospy.Time.now() + rospy.Duration(self.scan_pause_seconds)
                    while not rospy.is_shutdown() and rospy.Time.now() < pause_until:
                        self._desired_cmd()
                        with self.lock:
                            abort_reason = self.rl_abort_reason
                        if abort_reason:
                            self.failure_reason = "rl_safety_abort:" + abort_reason
                            return False
                        rate.sleep()
                continue

            if (rospy.Time.now() - waypoint_start).to_sec() > self.waypoint_timeout:
                self.failure_reason = "waypoint_timeout:{}".format(waypoint["name"])
                return False
            deviation = route_progress_error(
                gt["x"], gt["y"], self.route, self.active_index
            )
            if self.active_index > 0 and deviation > self.route_deviation_limit:
                self.failure_reason = "route_deviation:{}:{:.2f}".format(
                    waypoint["name"], deviation
                )
                return False

            stalled = False
            if math.hypot(linear, lateral) > 0.08:
                translation_attempted = True
                if last_progress_xy is None:
                    last_progress_xy = (gt["x"], gt["y"])
                    last_progress_time = rospy.Time.now()
                elif math.hypot(
                    gt["x"] - last_progress_xy[0], gt["y"] - last_progress_xy[1]
                ) >= 0.08:
                    last_progress_xy = (gt["x"], gt["y"])
                    last_progress_time = rospy.Time.now()
                elif (rospy.Time.now() - last_progress_time).to_sec() > self.stagnation_seconds:
                    stalled = True
            elif translation_attempted and (
                rospy.Time.now() - last_progress_time
            ).to_sec() > self.stagnation_seconds:
                stalled = True

            if stalled:
                if stall_recoveries >= self.max_stall_recoveries:
                    self.failure_reason = "robot_stagnant:{}".format(waypoint["name"])
                    return False
                stall_recoveries += 1
                rospy.logwarn(
                    "Official RL zero-command stall recovery %d/%d at %s",
                    stall_recoveries, self.max_stall_recoveries, waypoint["name"],
                )
                self._state(
                    "RL_STALL_RECOVERY", target=waypoint["name"],
                    recovery=stall_recoveries,
                )
                recover_until = rospy.Time.now() + rospy.Duration(1.2)
                while not rospy.is_shutdown() and rospy.Time.now() < recover_until:
                    self._desired_cmd()
                    with self.lock:
                        abort_reason = self.rl_abort_reason
                    if abort_reason:
                        self.failure_reason = "rl_safety_abort:" + abort_reason
                        return False
                    rate.sleep()
                last_progress_xy = None
                last_progress_time = rospy.Time.now()
                translation_attempted = False
                turning_for_heading = False
                continue

            if abs(linear) > 1e-6 or abs(lateral) > 1e-6 or abs(yaw_rate) > 1e-6:
                self._mark_motion_started()
            self._desired_cmd(linear, yaw_rate, lateral)
            self._state(
                "RL_ROUTE", target=waypoint["name"],
                distance=round(distance, 3),
                heading_error=round(heading_error, 3),
                desired_linear=round(linear, 3),
                desired_lateral=round(lateral, 3),
                desired_yaw=round(yaw_rate, 3),
            )
            rate.sleep()

        self._desired_cmd()
        return self.active_index == len(self.route)

    def _set_kinematic_pose(self, x, y, z, yaw, vx=0.0, vy=0.0, wz=0.0):
        state = ModelState()
        state.model_name = "a1_gazebo"
        state.reference_frame = "world"
        state.pose.position.x = float(x)
        state.pose.position.y = float(y)
        state.pose.position.z = float(z)
        quaternion = quaternion_from_euler(0.0, 0.0, float(yaw))
        state.pose.orientation.x = quaternion[0]
        state.pose.orientation.y = quaternion[1]
        state.pose.orientation.z = quaternion[2]
        state.pose.orientation.w = quaternion[3]
        state.twist.linear.x = float(vx)
        state.twist.linear.y = float(vy)
        state.twist.angular.z = float(wz)
        response = self.set_model_state(state)
        if not response.success:
            raise RuntimeError("set_model_state failed: " + response.status_message)

    def _run_kinematic_route(self):
        """Smooth GT route driver used only to isolate SLAM from gait failures."""
        rospy.wait_for_service("/gazebo/set_model_state", timeout=30.0)
        with self.lock:
            gt = dict(self.gt) if self.gt else None
        if gt is None:
            self.failure_reason = "ground_truth_stale"
            return False

        self._state("KINEMATIC_ROUTE")
        self.started_stamp = rospy.Time.now().to_sec()
        start_time = time.monotonic()
        x, y, yaw = gt["x"], gt["y"], gt["yaw"]
        base_z = max(0.23, min(0.32, gt["z"]))
        rate_hz = 20.0
        dt = 1.0 / rate_hz
        yaw_rate_limit = 0.55
        rate = rospy.Rate(rate_hz)

        while not rospy.is_shutdown() and self.active_index < len(self.route):
            if time.monotonic() - start_time > self.mission_timeout:
                self.failure_reason = "mission_timeout"
                return False
            waypoint = self.route[self.active_index]
            dx, dy = waypoint["x"] - x, waypoint["y"] - y
            distance = math.hypot(dx, dy)
            target_yaw = yaw if distance < 1e-6 else math.atan2(dy, dx)
            heading_error = (target_yaw - yaw + math.pi) % (2.0 * math.pi) - math.pi

            if distance <= 0.08:
                with self.lock:
                    measured = dict(self.gt) if self.gt else None
                error = distance if measured is None else math.hypot(
                    measured["x"] - waypoint["x"], measured["y"] - waypoint["y"]
                )
                self.arrivals.append({
                    "index": self.active_index,
                    "name": waypoint["name"],
                    "error_m": round(error, 4),
                    "t": round(rospy.Time.now().to_sec() - self.started_stamp, 3),
                })
                if waypoint["name"] == "door_crossing_inside":
                    self.entered_room = True
                elif waypoint["name"] == "loop_closed":
                    self.loop_closed = True
                elif waypoint["name"] == "exit_outside":
                    self.exited_room = True
                self.active_index += 1
                self._state("KINEMATIC_ROUTE", reached=waypoint["name"])
                continue

            if abs(heading_error) > 0.03:
                wz = max(-yaw_rate_limit, min(yaw_rate_limit, heading_error / dt))
                yaw += wz * dt
                vx = vy = 0.0
            else:
                yaw = target_yaw
                speed = self.kinematic_speeds.get(waypoint["zone"], 0.4)
                step = min(distance, speed * dt)
                vx, vy = speed * math.cos(yaw), speed * math.sin(yaw)
                x += step * math.cos(yaw)
                y += step * math.sin(yaw)
                wz = 0.0
            self._set_kinematic_pose(x, y, base_z, yaw, vx, vy, wz)
            self._joy(button=BUTTON_STAND)
            self._state(
                "KINEMATIC_ROUTE", target=waypoint["name"],
                distance=round(distance, 3), heading_error=round(heading_error, 3),
            )
            rate.sleep()

        self._set_kinematic_pose(x, y, base_z, yaw)
        return self.active_index == len(self.route)

    def _settle(self):
        self._state("SETTLE")
        start = rospy.Time.now()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and (rospy.Time.now() - start).to_sec() < self.settle_seconds:
            if self.motion_mode == "official_rl":
                self._desired_cmd()
            else:
                self._joy()
            rate.sleep()
        self.finished_stamp = rospy.Time.now().to_sec()

    @staticmethod
    def _trajectory_document(samples):
        if not samples:
            return {"schema": "simenv_exploration_trajectory_v1", "poses": []}
        start = samples[0]["t_abs"]
        poses = [
            {"t": round(s["t_abs"] - start, 4), "x": round(s["x"], 5),
             "y": round(s["y"], 5), "z": round(s["z"], 5),
             "roll": round(s["roll"], 5), "pitch": round(s["pitch"], 5),
             "yaw": round(s["yaw"], 5)}
            for s in samples
        ]
        length = sum(
            math.hypot(b["x"] - a["x"], b["y"] - a["y"])
            for a, b in zip(poses, poses[1:])
        )
        return {
            "schema": "simenv_exploration_trajectory_v1",
            "frame_id": "world",
            "start_stamp": start,
            "duration_sec": round(poses[-1]["t"], 4),
            "path_length_m": round(length, 4),
            "start": poses[0], "end": poses[-1], "poses": poses,
        }

    @staticmethod
    def _write_json(path, payload):
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

    def _cloud_points(self):
        with self.lock:
            message = self.cloud_message
        if message is None:
            return []
        voxels = {}
        for x, y, z in point_cloud2.read_points(
            message, field_names=("x", "y", "z"), skip_nans=True
        ):
            if not all(math.isfinite(v) for v in (x, y, z)):
                continue
            key = (int(math.floor(x / 0.08)), int(math.floor(y / 0.08)), int(math.floor(z / 0.08)))
            voxels[key] = (float(x), float(y), float(z))
            if len(voxels) >= self.maximum_cloud_points:
                break
        return list(voxels.values())

    def _write_pcd(self, points):
        path = os.path.join(self.output_dir, "cloud_map.pcd")
        with open(path, "w", encoding="ascii") as stream:
            stream.write("# .PCD v0.7 - Point Cloud Data file format\n")
            stream.write("VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n")
            stream.write("WIDTH {}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n".format(len(points)))
            stream.write("POINTS {}\nDATA ascii\n".format(len(points)))
            for x, y, z in points:
                stream.write("{:.5f} {:.5f} {:.5f}\n".format(x, y, z))

    def _render(self, points, estimate, truth):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        if points:
            stride = max(1, len(points) // 100000)
            cloud = points[::stride]
            xs, ys, zs = zip(*cloud)
        else:
            xs, ys, zs = [], [], []

        fig, ax = plt.subplots(figsize=(9, 12))
        if xs:
            ax.scatter(xs, ys, c=zs, s=0.25, cmap="viridis", alpha=0.75, rasterized=True)
        if estimate:
            ax.plot([p["x"] for p in estimate], [p["y"] for p in estimate], color="#f28e2b", lw=1.5)
        ax.set_aspect("equal")
        ax.set_title("Room0 online FAST-LIO map (no layout/GT overlay)")
        ax.set_xlabel("map x [m]")
        ax.set_ylabel("map y [m]")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "online_map.png"), dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 12))
        bounds = self.room["bounds"]
        ax.add_patch(Rectangle(
            (bounds["x_min"], bounds["y_min"]),
            bounds["x_max"] - bounds["x_min"], bounds["y_max"] - bounds["y_min"],
            fill=False, lw=2.0, color="black", label="Room0 bounds",
        ))
        for item in self.room.get("furniture", []):
            x, y = item["pose"][:2]
            sx, sy = item["size"][:2]
            ax.add_patch(Rectangle((x - sx / 2, y - sy / 2), sx, sy, color="#888888", alpha=0.6))
        ax.plot([p["x"] for p in self.route], [p["y"] for p in self.route], "--", color="#4e79a7", lw=1.2, label="planned route")
        if truth:
            ax.plot([p["x"] for p in truth], [p["y"] for p in truth], color="black", lw=1.4, label="Gazebo GT")
        aligned_estimate = estimate
        if estimate and truth:
            rotation = truth[0]["yaw"] - estimate[0]["yaw"]
            cosine, sine = math.cos(rotation), math.sin(rotation)
            aligned_estimate = []
            for pose in estimate:
                dx, dy = pose["x"] - estimate[0]["x"], pose["y"] - estimate[0]["y"]
                aligned_estimate.append({
                    "x": truth[0]["x"] + cosine * dx - sine * dy,
                    "y": truth[0]["y"] + sine * dx + cosine * dy,
                })
        if aligned_estimate:
            ax.plot([p["x"] for p in aligned_estimate], [p["y"] for p in aligned_estimate], color="#f28e2b", lw=1.2, label="FAST-LIO (initial SE2 aligned)")
        door = self.room["door_pose"]
        ax.scatter([door[0]], [door[1]], marker="s", color="red", s=45, label="Room0 door")
        ax.set_aspect("equal")
        ax.set_xlim(bounds["x_min"] - 1.5, 1.0)
        ax.set_ylim(-4.0, bounds["y_max"] + 1.0)
        ax.set_title("Room0 diagnostic route and localization")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "room0_mapping_result.png"), dpi=180)
        plt.close(fig)

    def _finalize(self, route_success):
        self._stop()
        self.done_pub.publish(Bool(data=True))
        with self.lock:
            gt_samples = list(self.gt_samples)
            est_samples = list(self.est_samples)
            counts = dict(self.topic_counts)
            settle = list(self.settle_estimates)
        points = self._cloud_points()
        gt_doc = self._trajectory_document(gt_samples)
        est_doc = self._trajectory_document(est_samples)
        self._write_json(os.path.join(self.output_dir, "trajectory_ground_truth.json"), gt_doc)
        self._write_json(os.path.join(self.output_dir, "trajectory_estimate.json"), est_doc)
        self._write_json(os.path.join(self.output_dir, "planned_route.json"), {
            "schema": "simenv_room0_route_v1", "diagnostic_only": True,
            "room_id": self.room["id"], "waypoints": self.route,
        })
        self._write_json(
            os.path.join(self.output_dir, "run_parameters.json"),
            {"schema": "simenv_room0_baseline_parameters_v1", **self.parameter_snapshot},
        )
        self._write_pcd(points)
        self._render(points, est_samples, gt_samples)

        settle_drift = 0.0
        if len(settle) >= 2:
            first = settle[0]
            settle_drift = max(math.hypot(s["x"] - first["x"], s["y"] - first["y"]) for s in settle)
        duration = 0.0
        if self.started_stamp is not None:
            duration = (self.finished_stamp or rospy.Time.now().to_sec()) - self.started_stamp
        max_arrival = max((item["error_m"] for item in self.arrivals), default=float("inf"))
        gates = {
            "route_complete": bool(route_success and len(self.arrivals) == len(self.route)),
            "entered_room0": self.entered_room,
            "loop_closed": self.loop_closed,
            "exited_room0": self.exited_room or not self.require_exit,
            "waypoint_error_le_0_5m": max_arrival <= 0.5,
            "duration_le_240s": duration <= 240.0,
            "not_fallen": not self.fallen,
            "cloud_finite_and_nonempty": len(points) >= 1000,
            "settle_drift_le_0_15m": settle_drift <= 0.15,
        }
        payload = {
            "schema": "simenv_room0_mapping_summary_v1",
            "diagnostic_only": True,
            "motion_mode": self.motion_mode,
            "route_variant": self.route_variant,
            "require_exit": self.require_exit,
            "passed": all(gates.values()) and not self.failure_reason,
            "localization_pending": True,
            "gates": gates,
            "failure_reason": self.failure_reason,
            "duration_sec": round(duration, 3),
            "arrivals": self.arrivals,
            "max_arrival_error_m": None if not self.arrivals else round(max_arrival, 4),
            "settle_drift_m": round(settle_drift, 4),
            "cloud_points": len(points),
            "topic_counts": counts,
            "timing": {
                "route_start_stamp": self.started_stamp,
                "motion_started_stamp": self.motion_started_stamp,
                "route_finished_stamp": self.route_finished_stamp,
                "finished_stamp": self.finished_stamp,
            },
            "artifacts": {
                "online_map": "online_map.png",
                "result": "room0_mapping_result.png",
                "cloud": "cloud_map.pcd",
                "estimate": "trajectory_estimate.json",
                "ground_truth": "trajectory_ground_truth.json",
            },
        }
        self._write_json(os.path.join(self.output_dir, "summary.json"), payload)
        self._state("COMPLETE" if payload["passed"] else "FAILED", reason=self.failure_reason)

    def run(self):
        success = False
        try:
            if not self._wait_ready():
                return
            if not self._reset_and_stand():
                return
            if self.motion_mode == "kinematic_gt_diagnostic":
                success = self._run_kinematic_route()
            elif self.motion_mode == "official_rl":
                success = self._run_official_rl_route()
            else:
                success = self._run_route()
            self.route_finished_stamp = rospy.Time.now().to_sec()
            self._settle()
        except Exception as error:
            self.failure_reason = "exception:{}".format(error)
            rospy.logerr("Room0 mapping diagnostic failed: %s", error)
        finally:
            self._finalize(success)


if __name__ == "__main__":
    rospy.init_node("room0_mapping_controller")
    Room0MappingController().run()
