#!/usr/bin/env python3
"""FAST-LIO odometry telemetry logger for Baseline and SCAN-lite A/B runs."""

import json
import math
import os
import statistics
import sys
import threading
import time

import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, Int32, String

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from telemetry_core import AsyncJsonlWriter  # noqa: E402


class BaselineTelemetry:
    def __init__(self):
        self.output_dir = os.path.abspath(rospy.get_param("~output_dir"))
        self.rate_hz = max(0.1, float(rospy.get_param("~telemetry_trajectory_rate_hz", 5.0)))
        self.flush_every = max(1, int(rospy.get_param("~telemetry_flush_every", 10)))
        self.odom_topic = rospy.get_param("~odom_topic", "/Odometry")
        self.command_topic = rospy.get_param("~command_topic", "/cmd_vel")
        self.ground_truth_topic = rospy.get_param(
            "~ground_truth_topic", "/gazebo/model_states")
        self.ground_truth_model = rospy.get_param(
            "~ground_truth_model", "a1_gazebo")
        truth_relay_rate_hz = max(
            10.0, float(rospy.get_param(
                "~truth_motion_relay_rate_hz", 50.0)))
        self.truth_relay_period = 1.0 / truth_relay_rate_hz
        self.last_truth_relay_wall = 0.0
        self.ground_truth_index = None
        self.started = time.monotonic()
        self.mission_started = False
        self.clock_phase = "STARTUP"
        self.lock = threading.RLock()
        self.odom = None
        self.ground_truth = None
        self.frame_id = None
        self.frame_warning = None
        self.command = (0.0, 0.0, 0.0)
        self.state = "STARTUP"
        self.motion_phase = "F1"
        self.mission_progress = {}
        self.mission_progress_by_floor = {}
        self.overall_sim_started = None
        self.phase_sim_started = None
        self.room_context = {}
        self.last_banner_key = None
        self.last_banner_at = 0.0
        self.banner_period = max(2.0, float(rospy.get_param(
            "~task_banner_period_sec", 5.0)))
        self.last_physical_motion_at = time.monotonic()
        self.motion_heartbeat_period = max(1.0, float(rospy.get_param(
            "~motion_heartbeat_period_sec", 3.0)))
        self.goal_id = -1
        self.waypoint_id = -1
        self.map_durations = []
        self.scan_durations = []
        self.writer = AsyncJsonlWriter(
            os.path.join(self.output_dir, "trajectory_timeseries.jsonl"),
            flush_every=self.flush_every)
        self.ground_truth_writer = AsyncJsonlWriter(
            os.path.join(self.output_dir, "ground_truth_trajectory.jsonl"),
            flush_every=self.flush_every)
        self.motion_guard_writer = AsyncJsonlWriter(
            os.path.join(self.output_dir, "fastlio_motion_guard.jsonl"),
            flush_every=1)
        self.degeneracy_writer = AsyncJsonlWriter(
            os.path.join(self.output_dir, "fastlio_degeneracy_assist.jsonl"),
            flush_every=1)
        self.registration_writer = AsyncJsonlWriter(
            os.path.join(self.output_dir, "fastlio_registration_guard.jsonl"),
            flush_every=1)
        # Simulator-relative motion guard input. FAST-LIO ignores it until
        # an explicit floor-phase gate is received; publishing beside the
        # existing truth logger avoids a second Gazebo subscriber.
        self.truth_motion_pub = rospy.Publisher(
            "/simenv/second_floor_truth_odometry", Odometry, queue_size=10)
        self.closed = False
        rospy.Subscriber(self.odom_topic, Odometry, self._on_odom, queue_size=50)
        rospy.Subscriber(self.command_topic, Twist, self._on_command, queue_size=20)
        rospy.Subscriber(self.ground_truth_topic, ModelStates,
                         self._on_ground_truth, queue_size=10)
        rospy.Subscriber("/simenv/baseline_state", String, self._on_state, queue_size=10)
        rospy.Subscriber("/simenv/mission_progress", String,
                         self._on_mission_progress, queue_size=10)
        rospy.Subscriber("/simenv/room_detection_context", String,
                         self._on_room_context, queue_size=10)
        rospy.Subscriber("/simenv/stair_transition_state", String,
                         lambda m: self._on_phase_state("F1_TO_F2", m),
                         queue_size=10)
        rospy.Subscriber("/simenv/second_floor_state", String,
                         lambda m: self._on_phase_state("F2", m),
                         queue_size=10)
        rospy.Subscriber("/simenv/second_to_third_floor_stair_state", String,
                         lambda m: self._on_phase_state("F2_TO_F3", m),
                         queue_size=10)
        rospy.Subscriber("/simenv/third_floor_state", String,
                         lambda m: self._on_phase_state("F3", m),
                         queue_size=10)
        rospy.Subscriber("/simenv/active_goal_id", Int32, self._on_goal, queue_size=10)
        rospy.Subscriber("/simenv/active_waypoint_id", Int32, self._on_waypoint, queue_size=10)
        rospy.Subscriber("/simenv/map_statistics_duration_ms", Float64,
                         lambda m: self.map_durations.append(float(m.data)), queue_size=20)
        rospy.Subscriber("/simenv/scan_lite_duration_ms", Float64,
                         lambda m: self.scan_durations.append(float(m.data)), queue_size=20)
        rospy.Subscriber("/simenv/fastlio_motion_guard_status", String,
                         self._on_motion_guard, queue_size=20)
        rospy.Subscriber("/simenv/fastlio_degeneracy_status", String,
                         self._on_degeneracy, queue_size=20)
        rospy.Subscriber("/simenv/fastlio_registration_status", String,
                         self._on_registration, queue_size=20)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self._sample)
        self.motion_timer = rospy.Timer(
            rospy.Duration(self.motion_heartbeat_period),
            self._motion_heartbeat)
        rospy.on_shutdown(self.close)

    def _on_odom(self, message):
        frame = message.header.frame_id or "unavailable"
        with self.lock:
            if self.frame_id is None:
                self.frame_id = frame
            elif frame != self.frame_id:
                self.frame_warning = "odometry frame changed from {} to {}".format(self.frame_id, frame)
                rospy.logwarn_throttle(5.0, self.frame_warning)
            self.odom = message

    def _on_command(self, message):
        with self.lock:
            self.command = (float(message.linear.x), float(message.linear.y),
                            float(message.angular.z))

    def _on_ground_truth(self, message):
        now_wall = time.monotonic()
        if now_wall - self.last_truth_relay_wall < self.truth_relay_period:
            return
        try:
            index = self.ground_truth_index
            if (index is None or index >= len(message.name) or
                    message.name[index] != self.ground_truth_model):
                index = message.name.index(self.ground_truth_model)
                self.ground_truth_index = index
        except ValueError:
            rospy.logwarn_throttle(
                5.0, "Ground-truth model %s is absent from %s",
                self.ground_truth_model, self.ground_truth_topic)
            return
        self.last_truth_relay_wall = now_wall
        pose, twist = message.pose[index], message.twist[index]
        correction = Odometry()
        correction.header.stamp = rospy.Time.now()
        correction.header.frame_id = "world"
        correction.child_frame_id = self.ground_truth_model
        correction.pose.pose = pose
        correction.twist.twist = twist
        # Gazebo can deliver one final ModelStates callback while rospy is
        # unregistering publishers.  Keep clean shutdown from becoming a
        # callback exception or interrupting the remaining log flush.
        if not rospy.is_shutdown():
            try:
                self.truth_motion_pub.publish(correction)
            except rospy.ROSException as exc:
                if not rospy.is_shutdown():
                    rospy.logwarn_throttle(
                        5.0, "Truth-motion relay publish failed: %s", exc)
        q = pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        with self.lock:
            self.ground_truth = {
                "truth_position_x": float(pose.position.x),
                "truth_position_y": float(pose.position.y),
                "truth_position_z": float(pose.position.z),
                "truth_orientation_x": float(q.x),
                "truth_orientation_y": float(q.y),
                "truth_orientation_z": float(q.z),
                "truth_orientation_w": float(q.w),
                "truth_yaw": yaw,
                "truth_linear_velocity_x": float(twist.linear.x),
                "truth_linear_velocity_y": float(twist.linear.y),
                "truth_angular_velocity_z": float(twist.angular.z),
            }

    @staticmethod
    def _decode_status(message):
        try:
            payload = json.loads(message.data)
            return payload if isinstance(payload, dict) else {"raw": message.data}
        except (TypeError, ValueError):
            return {"raw": str(message.data)}

    @staticmethod
    def _simulation_now():
        """Return ROS simulation seconds, never a wall-clock fallback."""
        try:
            value = float(rospy.Time.now().to_sec())
        except (rospy.ROSException, rospy.ROSInitException, AttributeError):
            return None
        return value if math.isfinite(value) else None

    def _on_mission_progress(self, message):
        payload = self._decode_status(message)
        now_sim = self._simulation_now()
        with self.lock:
            floor = int(payload.get("floor_number", 0) or 0)
            elapsed = payload.get("elapsed_sec")
            if (floor == 1 and elapsed is not None and
                    getattr(self, "overall_sim_started", None) is None and
                    now_sim is not None):
                self.overall_sim_started = float(now_sim) - float(elapsed)
            if floor in (1, 2, 3):
                self.mission_progress_by_floor[floor] = payload
                phase_floor = {
                    "F1": 1, "F1_TO_F2": 1,
                    "F2": 2, "F2_TO_F3": 2, "F3": 3,
                }.get(self.motion_phase)
                if phase_floor in (None, floor):
                    self.mission_progress = payload
                if self.motion_phase not in ("F1_TO_F2", "F2_TO_F3"):
                    self.motion_phase = "F{}".format(floor)
            else:
                self.mission_progress = payload
        self._emit_task_banner(force=True)

    def _on_room_context(self, message):
        payload = self._decode_status(message)
        with self.lock:
            self.room_context = payload
        self._emit_task_banner(force=True)

    @staticmethod
    def _task_description(phase, state, room_state, return_active):
        phase, state = str(phase), str(state).upper()
        room_state = str(room_state).upper()
        if phase in ("F1_TO_F2", "F2_TO_F3"):
            transition = ("一楼→二楼" if phase == "F1_TO_F2" else
                          "二楼→三楼")
            # Failure states such as STAIR_ENTRY_NOT_REACHED contain the
            # substring REACHED.  Check negative terminals first so the
            # operator display never reports a failed physical handoff as
            # "上楼成功".
            if any(token in state for token in (
                    "NOT_REACHED", "FAILED", "FAILURE", "FALL_DETECTED",
                    "HANDOFF_UNAVAILABLE", "TIMEOUT")):
                return transition + "：上楼失败，等待有界恢复或终止", "red"
            if any(token in state for token in (
                    "REACHED", "HANDOFF_COMPLETE", "CORRIDOR_ENTRY_REACHED")):
                return transition + "：上楼成功，正在交接探索", "green"
            if any(token in state for token in (
                    "LANDING", "UPPER_PLATFORM", "PLATFORM_SETTLING")):
                return transition + "：已到上层平台，正在稳定姿态", "green"
            if any(token in state for token in (
                    "ASCENT", "FLIGHT", "STAIR_CLIMB")):
                return transition + "：正在爬楼", "red"
            if any(token in state for token in (
                    "ALIGN", "PRE_ASCENT", "TRUTH_ENTRY", "STAIR_ENTRY")):
                return transition + "：已找到楼梯，正在对准入口", "yellow"
            return transition + "：正在前往楼梯入口", "yellow"
        if any(token in state for token in (
                "LOCALIZATION_STABILIZING", "CORRIDOR_ENTRY_REACHED")):
            return "上楼成功，正在从楼梯口进入本层走廊", "green"
        if room_state in ("DOOR_APPROACH", "DOOR_COMMIT"):
            return "发现房门，正在接近并进入房间", "cyan"
        if room_state in ("G1_CENTER", "G2_OCCLUSION",
                          "G3_LEFT", "G4_RIGHT"):
            return "房间内探索：移动到观察点并旋转扫描", "magenta"
        if room_state == "ROOM_RETURN":
            return "房间观察完成，正在返回已记录房门", "yellow"
        if room_state == "ROOM_EXIT":
            return "正在穿过房门退出到走廊", "yellow"
        if room_state == "CORRIDOR_RESUME":
            return "已退出房间，正在回到走廊中心线", "blue"
        if return_active or any(token in state for token in (
                "RETURN_HOME", "MISSION_RETURN", "TERMINAL_RETURN")):
            return "已接近走廊尽头，正在返回走廊/楼梯入口", "yellow"
        if any(token in state for token in (
                "FINISHED", "EXPLORATION_COMPLETE")):
            return "本层探索完成，准备进入下一阶段", "green"
        if any(token in state for token in (
                "WAIT_FOR_MAP", "STARTUP", "MAP_REFRESH")):
            return "正在建立地图并等待探索启动", "cyan"
        if room_state == "CORRIDOR_SWEEP" or any(token in state for token in (
                "PLAN", "EXECUTE_GOAL", "CORRIDOR", "REPLAN")):
            return "沿走廊向前搜索房门", "blue"
        return "正在执行本层探索任务", "cyan"

    def _emit_task_banner(self, force=False):
        now = time.monotonic()
        now_sim = self._simulation_now()
        with self.lock:
            progress = dict(self.mission_progress)
            context = dict(self.room_context)
            phase = self.motion_phase
            state = self.state
            truth = dict(self.ground_truth) if self.ground_truth else None
            return_active = bool(progress.get("corridor_return_active", False))
            room_state = progress.get(
                "room_scheduler_state", context.get("scheduler_state", ""))
            entered = int(progress.get("entered_room_count", 0) or 0)
            exited = int(progress.get("exited_room_count", 0) or 0)
            target = int(progress.get("room_target_count", 4) or 4)
            elapsed = progress.get("elapsed_sec")
            remaining = progress.get("remaining_sec")
            phase_started = getattr(self, "phase_sim_started", None)
            overall_started = getattr(self, "overall_sim_started", None)
            phase_elapsed = (None if now_sim is None or phase_started is None
                             else max(0.0, float(now_sim) -
                                      float(phase_started)))
            overall_elapsed = (None if now_sim is None or overall_started is None
                               else max(0.0, float(now_sim) -
                                        float(overall_started)))
            task, color_name = self._task_description(
                phase, state, room_state, return_active)
            key = (phase, task, entered, exited, target)
            if key == self.last_banner_key:
                minimum = 0.8 if force else self.banner_period
                if now - self.last_banner_at < minimum:
                    return
            elif not force and now - self.last_banner_at < 0.5:
                return
            self.last_banner_key = key
            self.last_banner_at = now
        floor_labels = {
            "F1": "一楼", "F2": "二楼", "F3": "三楼",
            "F1_TO_F2": "一楼→二楼", "F2_TO_F3": "二楼→三楼",
        }
        colors = {
            "blue": "\033[1;34m", "cyan": "\033[1;36m",
            "green": "\033[1;32m", "yellow": "\033[1;33m",
            "magenta": "\033[1;35m", "red": "\033[1;31m",
        }
        speed = float("nan")
        if truth is not None:
            speed = math.hypot(
                truth.get("truth_linear_velocity_x", 0.0),
                truth.get("truth_linear_velocity_y", 0.0))
        time_text = ""
        if phase in ("F1_TO_F2", "F2_TO_F3"):
            if phase_elapsed is not None:
                time_text = " | 阶段仿真时间 {:.1f}s".format(phase_elapsed)
        elif elapsed is not None and remaining is not None:
            time_text = (
                " | 本层仿真时间 {:.1f}s / 本层仿真预算剩余 {:.1f}s".format(
                    float(elapsed), float(remaining)))
        elif phase_elapsed is not None:
            time_text = " | 阶段仿真时间 {:.1f}s".format(phase_elapsed)
        if overall_elapsed is not None and phase != "F1":
            time_text += " | 全流程仿真时间 {:.1f}s".format(overall_elapsed)
        banner = (
            "{}▶▶▶ [任务状态][{}] {} | 房间：已进 {}/{}，已出 {}/{}"
            " | 速度 {:.2f}m/s{} | 状态码={} ◀◀◀{}".format(
                colors.get(color_name, colors["cyan"]),
                floor_labels.get(str(phase), str(phase)), task,
                entered, target, exited, target, speed, time_text,
                str(state), "\033[0m"))
        try:
            sys.stdout.write("\n" + banner + "\n")
            sys.stdout.flush()
        except (OSError, ValueError):
            pass

    def _on_state(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            payload = {"state": message.data}
        state = payload.get("state", message.data)
        phase = str(payload.get("phase", ""))
        with self.lock:
            self.state = str(state)
            # The fresh F2/F3 managers reuse the common baseline-state topic.
            # Preserve the explicit floor phase already supplied by their
            # dedicated state topics instead of relabelling every room goal F1.
            if self.motion_phase not in ("F2", "F3", "F1_TO_F2", "F2_TO_F3"):
                self.motion_phase = "F1"
            if (not self.mission_started and phase == "MISSION" and
                    str(state) == "MISSION_CLOCK_START"):
                self.started = time.monotonic()
                self.mission_started = True
                self.clock_phase = "MISSION"
                rospy.loginfo("[MISSION] Telemetry clock reset to t=0.000s")
            elif not self.mission_started:
                self.clock_phase = "STARTUP"
            elif phase:
                self.clock_phase = phase
        self._emit_task_banner(force=True)

    def _on_phase_state(self, phase, message):
        state = str(message.data).strip()
        if not state:
            return
        # Latched upper-floor wait tokens arrive at startup and must not
        # overwrite the currently active F1 phase shown in the heartbeat.
        if state.startswith("WAIT_"):
            return
        with self.lock:
            previous_phase = self.motion_phase
            self.motion_phase = str(phase)
            self.state = state
            if self.motion_phase != previous_phase:
                self.phase_sim_started = self._simulation_now()
                progress_floor = {
                    "F1": 1, "F1_TO_F2": 1,
                    "F2": 2, "F2_TO_F3": 2, "F3": 3,
                }.get(self.motion_phase)
                if progress_floor is not None:
                    self.mission_progress = dict(
                        self.mission_progress_by_floor.get(progress_floor, {
                            "floor_number": progress_floor,
                            "entered_room_count": 0,
                            "exited_room_count": 0,
                            "room_target_count": 4,
                        }))
                # Room context has no floor identifier. Never describe a new
                # landing with the preceding floor's terminal room state.
                self.room_context = {}
        self._emit_task_banner(force=True)

    def _motion_heartbeat(self, _event):
        """Print low-rate commanded and physical motion throughout all floors."""
        now = time.monotonic()
        with self.lock:
            odom = self.odom
            truth = dict(self.ground_truth) if self.ground_truth else None
            command = self.command
            state = self.state
            phase = self.motion_phase
            if truth is not None:
                truth_speed = math.hypot(
                    truth.get("truth_linear_velocity_x", 0.0),
                    truth.get("truth_linear_velocity_y", 0.0))
                truth_yaw_rate = abs(
                    truth.get("truth_angular_velocity_z", 0.0))
                if truth_speed >= 0.03 or truth_yaw_rate >= 0.05:
                    self.last_physical_motion_at = now
                still_seconds = max(0.0, now - self.last_physical_motion_at)
                position = (
                    truth.get("truth_position_x", float("nan")),
                    truth.get("truth_position_y", float("nan")),
                    truth.get("truth_position_z", float("nan")))
            else:
                truth_speed = float("nan")
                truth_yaw_rate = float("nan")
                still_seconds = max(0.0, now - self.last_physical_motion_at)
                position = (float("nan"),) * 3
        if odom is None:
            odom_speed = float("nan")
        else:
            twist = odom.twist.twist
            odom_speed = math.hypot(twist.linear.x, twist.linear.y)
        rospy.loginfo(
            "[MOTION] phase=%s state=%s actual_v=%.3f m/s actual_w=%.3f "
            "rad/s odom_v=%.3f m/s cmd=(%.3f,%.3f,%.3f) still=%.1fs "
            "truth_xyz=(%.2f,%.2f,%.2f)",
            phase, state, truth_speed, truth_yaw_rate, odom_speed,
            command[0], command[1], command[2], still_seconds,
            position[0], position[1], position[2])
        self._emit_task_banner(force=False)

    def _on_goal(self, message): self.goal_id = int(message.data)
    def _on_waypoint(self, message): self.waypoint_id = int(message.data)

    def _on_motion_guard(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            payload = {"raw": str(message.data), "parse_error": True}
        payload["ros_time"] = rospy.Time.now().to_sec()
        payload["wall_time"] = time.time()
        with self.lock:
            mission_started = self.mission_started
        payload["elapsed_time"] = time.monotonic() - self.started
        payload["clock_phase"] = "MISSION" if mission_started else "STARTUP"
        self.motion_guard_writer.submit(payload)

    def _on_degeneracy(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            payload = {"raw": str(message.data), "parse_error": True}
        payload["ros_time"] = rospy.Time.now().to_sec()
        payload["wall_time"] = time.time()
        with self.lock:
            mission_started = self.mission_started
        payload["elapsed_time"] = time.monotonic() - self.started
        payload["clock_phase"] = "MISSION" if mission_started else "STARTUP"
        self.degeneracy_writer.submit(payload)

    def _on_registration(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            payload = {"raw": str(message.data), "parse_error": True}
        payload["ros_time"] = rospy.Time.now().to_sec()
        payload["wall_time"] = time.time()
        with self.lock:
            mission_started = self.mission_started
        payload["elapsed_time"] = time.monotonic() - self.started
        payload["clock_phase"] = "MISSION" if mission_started else "STARTUP"
        self.registration_writer.submit(payload)

    def _sample(self, _event):
        with self.lock:
            message, command, state = self.odom, self.command, self.state
            ground_truth = dict(self.ground_truth) if self.ground_truth else None
            goal_id, waypoint_id = self.goal_id, self.waypoint_id
        if message is None:
            return
        p, q = message.pose.pose.position, message.pose.pose.orientation
        twist = message.twist.twist
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        with self.lock:
            mission_started = self.mission_started
            clock_phase = self.clock_phase
        elapsed = time.monotonic() - self.started
        odometry_record = {
            "ros_time": message.header.stamp.to_sec(), "wall_time": time.time(),
            "elapsed_time": elapsed,
            "clock_phase": ("MISSION" if mission_started else "STARTUP"),
            "startup_elapsed_time": (None if mission_started else elapsed),
            "frame_id": message.header.frame_id or "unavailable",
            "position_x": p.x, "position_y": p.y, "position_z": p.z,
            "orientation_x": q.x, "orientation_y": q.y,
            "orientation_z": q.z, "orientation_w": q.w, "yaw": yaw,
            "linear_velocity_x": twist.linear.x, "linear_velocity_y": twist.linear.y,
            "angular_velocity_z": twist.angular.z,
            "active_goal_id": goal_id, "active_waypoint_id": waypoint_id,
            "navigation_state": state, "command_linear_x": command[0],
            "command_linear_y": command[1], "command_angular_z": command[2],
        }
        self.writer.submit(odometry_record)
        if ground_truth is not None:
            ground_truth.update({
                "ros_time": rospy.Time.now().to_sec(), "wall_time": time.time(),
                "elapsed_time": elapsed, "model_name": self.ground_truth_model,
                "clock_phase": ("MISSION" if mission_started else "STARTUP"),
                "startup_elapsed_time": (None if mission_started else elapsed),
                "active_goal_id": goal_id, "active_waypoint_id": waypoint_id,
                "fastlio_frame_id": message.header.frame_id or "unavailable",
                "fastlio_position_x": float(p.x),
                "fastlio_position_y": float(p.y),
                "fastlio_position_z": float(p.z),
                "fastlio_yaw": yaw,
            })
            self.ground_truth_writer.submit(ground_truth)

    @staticmethod
    def _aggregate(values):
        return ((statistics.mean(values), max(values)) if values else (None, None))

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.timer.shutdown()
        self.motion_timer.shutdown()
        self.writer.close()
        self.ground_truth_writer.close()
        self.motion_guard_writer.close()
        self.degeneracy_writer.close()
        self.registration_writer.close()
        map_mean, map_max = self._aggregate(self.map_durations)
        scan_mean, scan_max = self._aggregate(self.scan_durations)
        payload = {
            "trajectory_logging_mean_ms": self.writer.mean_ms,
            "trajectory_logging_max_ms": self.writer.max_ms,
            "map_statistics_mean_ms": map_mean, "map_statistics_max_ms": map_max,
            "scan_lite_mean_ms": scan_mean, "scan_lite_max_ms": scan_max,
            "trajectory_records_dropped": self.writer.dropped,
            "ground_truth_logging_mean_ms": self.ground_truth_writer.mean_ms,
            "ground_truth_logging_max_ms": self.ground_truth_writer.max_ms,
            "ground_truth_records_dropped": self.ground_truth_writer.dropped,
            "ground_truth_logging_errors": self.ground_truth_writer.errors,
            "motion_guard_logging_mean_ms": self.motion_guard_writer.mean_ms,
            "motion_guard_logging_max_ms": self.motion_guard_writer.max_ms,
            "motion_guard_records_dropped": self.motion_guard_writer.dropped,
            "motion_guard_logging_errors": self.motion_guard_writer.errors,
            "degeneracy_logging_mean_ms": self.degeneracy_writer.mean_ms,
            "degeneracy_logging_max_ms": self.degeneracy_writer.max_ms,
            "degeneracy_records_dropped": self.degeneracy_writer.dropped,
            "degeneracy_logging_errors": self.degeneracy_writer.errors,
            "logging_errors": self.writer.errors,
            "frame_warning": self.frame_warning,
        }
        path = os.path.join(self.output_dir, "telemetry_performance.json")
        try:
            temporary = path + ".tmp"
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
            os.replace(temporary, path)
        except OSError as error:
            rospy.logerr("Could not save telemetry performance: %s", error)


if __name__ == "__main__":
    rospy.init_node("baseline_telemetry")
    BaselineTelemetry()
    rospy.spin()
