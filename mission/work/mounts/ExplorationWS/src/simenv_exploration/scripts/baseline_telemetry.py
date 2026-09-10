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
        self.closed = False
        rospy.Subscriber(self.odom_topic, Odometry, self._on_odom, queue_size=50)
        rospy.Subscriber(self.command_topic, Twist, self._on_command, queue_size=20)
        rospy.Subscriber(self.ground_truth_topic, ModelStates,
                         self._on_ground_truth, queue_size=10)
        rospy.Subscriber("/simenv/baseline_state", String, self._on_state, queue_size=10)
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
        try:
            index = message.name.index(self.ground_truth_model)
        except ValueError:
            rospy.logwarn_throttle(
                5.0, "Ground-truth model %s is absent from %s",
                self.ground_truth_model, self.ground_truth_topic)
            return
        pose, twist = message.pose[index], message.twist[index]
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

    def _on_state(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            payload = {"state": message.data}
        state = payload.get("state", message.data)
        phase = str(payload.get("phase", ""))
        with self.lock:
            self.state = str(state)
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
