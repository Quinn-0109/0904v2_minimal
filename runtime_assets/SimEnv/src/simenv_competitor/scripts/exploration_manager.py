#!/usr/bin/env python3
"""Cycle Voxel Map -> FUEL-lite -> Goal Executor until exploration terminates."""

import csv
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Bool, Float32, String

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exploration_manager_core import (  # noqa: E402
    confirmed_frontier_exhaustion, coverage_from_statistics, is_duplicate_goal,
    mark_visited_disk, visited_area_from_statistics,
)
from goal_executor_core import quaternion_yaw  # noqa: E402
from structured_topology_core import OccupancyMap  # noqa: E402


class ExplorationManager:
    def __init__(self):
        self._output_dir = os.path.abspath(rospy.get_param("~output_dir"))
        self._map_file = os.path.abspath(rospy.get_param("~map_file"))
        self._statistics_file = os.path.abspath(rospy.get_param("~statistics_file"))
        self._odom_topic = rospy.get_param("~odom_topic", "/Odometry")
        self._goal_topic = rospy.get_param("~goal_topic", "/exploration_goal")
        self._planner_timeout = float(rospy.get_param("~planner_timeout", 35.0))
        self._goal_timeout = float(rospy.get_param("~goal_timeout", 75.0))
        self._startup_timeout = float(rospy.get_param("~startup_timeout", 100.0))
        self._maximum_duration = float(rospy.get_param("~maximum_duration", 300.0))
        self._coverage_threshold = float(rospy.get_param("~coverage_threshold", 0.95))
        self._gain_threshold = float(rospy.get_param("~information_gain_threshold", 20.0))
        self._stagnation_threshold = float(rospy.get_param("~stagnation_seconds", 30.0))
        self._duplicate_radius = float(rospy.get_param("~duplicate_goal_radius", 0.35))
        self._maximum_goals = int(rospy.get_param("~maximum_goals", 40))
        self._maximum_goal_distance = float(rospy.get_param("~maximum_goal_distance", 8.0))
        self._maximum_consecutive_failures = int(
            rospy.get_param("~maximum_consecutive_failures", 3))
        self._minimum_map_updates = int(rospy.get_param("~minimum_map_updates", 1))
        self._exploration_ready_free_voxel_threshold = int(
            rospy.get_param("~exploration_ready_free_voxel_threshold", 5000))
        self._exploration_ready_frontier_cluster_min_size = int(
            rospy.get_param("~exploration_ready_frontier_cluster_min_size", 8))
        self._exploration_ready_timeout = float(
            rospy.get_param("~exploration_ready_timeout", 60.0))
        self._post_ready_dwell = float(rospy.get_param("~post_ready_dwell", 8.0))
        self._exploration_speed = float(rospy.get_param("~exploration_speed", 0.28))
        self._finalization_reserve = float(rospy.get_param("~finalization_reserve", 5.0))
        self._frontier_empty_confirmations = int(
            rospy.get_param("~frontier_empty_confirmations", 3))
        self._frontier_empty_seconds = float(
            rospy.get_param("~frontier_empty_seconds", 30.0))
        self._visited_radius = float(rospy.get_param("~visited_radius", 1.0))
        self._visited_resolution = float(rospy.get_param("~visited_resolution", 0.25))
        os.makedirs(self._output_dir, exist_ok=True)
        self._cycle_root = os.path.join(self._output_dir, "planner_cycles")
        os.makedirs(self._cycle_root, exist_ok=True)

        self._lock = threading.RLock()
        self._pose = None
        self._pose_frame = "camera_init"
        self._last_pose_wall = 0.0
        self._trajectory = []
        self._coverage_history = []
        self._frontier_history = []
        self._goal_history = []
        self._events = []
        self._execution_results = []
        self._score_history = []
        self._visited_history = []
        self._visited_cells = set()
        self._blacklist = []
        self._empty_cycles = 0
        self._empty_since = None
        self._termination_checks = []
        self._entry_constraint = None
        self._latest_statistics = None
        self._grid = None
        self._grid_stamp = None
        self._previous_free = None
        self._last_free_growth = time.monotonic()
        self._last_visited_growth = time.monotonic()
        self._start_wall = None
        self._start_ros = None
        self._termination = None
        self._last_trajectory_stamp = 0.0
        self._last_coverage_record = 0.0
        self._node_started_wall = time.monotonic()
        self._startup_timing = {
            "fast_lio_ready": None,
            "voxel_ready": None,
            "frontier_ready": None,
            "first_fuel_goal": None,
        }
        self._exploration_ready = False
        self._exploration_ready_frontier_count = 0
        self._fuel_plan_count = 0

        self._goal_pub = rospy.Publisher(self._goal_topic, PoseStamped, queue_size=1)
        self._complete_pub = rospy.Publisher(
            "/simenv/mission_complete", Bool, queue_size=1, latch=True
        )
        self._state_pub = rospy.Publisher(
            "/simenv/exploration_manager_state", String, queue_size=2, latch=True
        )
        self._speed_limit_pub = rospy.Publisher(
            "/simenv/goal_speed_limit", Float32, queue_size=1, latch=True
        )
        rospy.Subscriber(self._odom_topic, Odometry, self._on_odom, queue_size=30)
        rospy.Subscriber(
            "/simenv/voxel_floor_projection", OccupancyGrid,
            self._on_grid, queue_size=2)
        rospy.Subscriber(
            "/simenv/goal_execution_result", String,
            self._on_execution_result, queue_size=5,
        )
        rospy.Timer(rospy.Duration(1.0), self._sample_map)
        rospy.on_shutdown(self._shutdown)

    @staticmethod
    def _load_json(path):
        try:
            with open(path, encoding="utf-8") as stream:
                return json.load(stream)
        except (OSError, ValueError):
            return None

    @staticmethod
    def _write_json(path, payload):
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)

    def _elapsed(self):
        return 0.0 if self._start_wall is None else time.monotonic() - self._start_wall

    def _event(self, kind, **extra):
        item = {"t": round(self._elapsed(), 3), "event": kind}
        item.update(extra)
        with self._lock:
            self._events.append(item)
        self._state_pub.publish(String(data=json.dumps(item, sort_keys=True)))

    def _mark_startup_time(self, name):
        with self._lock:
            if self._startup_timing.get(name) is not None:
                return
            self._startup_timing[name] = {
                "wall_elapsed_sec": round(
                    time.monotonic() - self._node_started_wall, 3),
                "ros_time_sec": round(rospy.Time.now().to_sec(), 6),
            }

    def _on_odom(self, message):
        p, q = message.pose.pose.position, message.pose.pose.orientation
        pose = {
            "x": float(p.x), "y": float(p.y), "z": float(p.z),
            "yaw": quaternion_yaw(q.x, q.y, q.z, q.w),
        }
        if not all(math.isfinite(value) for value in pose.values()):
            return
        now = time.monotonic()
        with self._lock:
            first_valid_odom = self._pose is None
            self._pose = pose
            self._pose_frame = message.header.frame_id or self._pose_frame
            self._last_pose_wall = now
            if now - self._last_trajectory_stamp >= 0.10:
                self._last_trajectory_stamp = now
                self._trajectory.append({"t": round(self._elapsed(), 3), **pose})
                previous_visited_count = len(self._visited_cells)
                mark_visited_disk(self._visited_cells, pose["x"], pose["y"],
                                  self._visited_radius, self._visited_resolution)
                if len(self._visited_cells) > previous_visited_count:
                    self._last_visited_growth = now
        if first_valid_odom:
            self._mark_startup_time("fast_lio_ready")

    def _on_execution_result(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        payload["received_wall"] = time.monotonic()
        with self._lock:
            self._execution_results.append(payload)

    def _on_grid(self, message):
        try:
            data = np.asarray(message.data, dtype=np.int16).reshape(
                message.info.height, message.info.width)
            grid = OccupancyMap(
                data, message.info.resolution,
                message.info.origin.position.x,
                message.info.origin.position.y)
        except (TypeError, ValueError):
            return
        with self._lock:
            self._grid = grid
            self._grid_stamp = message.header.stamp.to_sec()

    def _sample_map(self, _event=None):
        statistics = self._load_json(self._statistics_file)
        if not statistics:
            return
        now = time.monotonic()
        free = int(statistics.get("free_voxel_count", 0))
        with self._lock:
            if self._previous_free is None or free > self._previous_free:
                self._last_free_growth = now
            self._previous_free = free
            self._latest_statistics = statistics
            if now - self._last_coverage_record >= 1.0:
                self._last_coverage_record = now
                self._coverage_history.append({
                    "t": round(self._elapsed(), 3),
                    "coverage_ratio": coverage_from_statistics(statistics),
                    "free_voxels": free,
                    "occupied_voxels": int(statistics.get("occupied_voxel_count", 0)),
                    "unknown_voxels": int(statistics.get("unknown_voxel_count", 0)),
                    "update_count": int(statistics.get("update_count", 0)),
                })
                self._visited_history.append({
                    "t": round(self._elapsed(), 3),
                    "visited_area_m2": round(
                        len(self._visited_cells) * self._visited_resolution ** 2, 4),
                    "observed_free_area_proxy_m2": round(
                        visited_area_from_statistics(statistics), 4),
                })

    def _wait_ready(self):
        deadline = time.monotonic() + self._startup_timeout
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            self._sample_map()
            with self._lock:
                pose = self._pose
                age = time.monotonic() - self._last_pose_wall
                stats = self._latest_statistics
                grid = self._grid
            if (pose is not None and age < 0.5 and stats is not None
                    and grid is not None
                    and int(stats.get("update_count", 0)) >= self._minimum_map_updates
                    and os.path.isfile(self._map_file)
                    and os.path.getsize(self._map_file) > 0):
                return True
            time.sleep(0.25)
        return False

    def _projection_frontier_count(self, grid):
        data = grid.data
        unknown = data < 0
        free = data == 0
        adjacent_unknown = np.zeros_like(free, dtype=bool)
        adjacent_unknown[1:, :] |= unknown[:-1, :]
        adjacent_unknown[:-1, :] |= unknown[1:, :]
        adjacent_unknown[:, 1:] |= unknown[:, :-1]
        adjacent_unknown[:, :-1] |= unknown[:, 1:]
        mask = free & adjacent_unknown
        visited = np.zeros_like(mask, dtype=bool)
        clusters = 0
        for y, x in np.argwhere(mask):
            if visited[y, x]:
                continue
            stack = [(int(x), int(y))]
            visited[y, x] = True
            size = 0
            while stack:
                cx, cy = stack.pop()
                size += 1
                for nx, ny in ((cx - 1, cy), (cx + 1, cy),
                               (cx, cy - 1), (cx, cy + 1)):
                    if (0 <= nx < grid.width and 0 <= ny < grid.height and
                            mask[ny, nx] and not visited[ny, nx]):
                        visited[ny, nx] = True
                        stack.append((nx, ny))
            clusters += int(
                size >= self._exploration_ready_frontier_cluster_min_size)
        return clusters

    def _wait_exploration_ready(self):
        """Hold zero motion until the static map can seed FUEL directly."""
        deadline = time.monotonic() + self._exploration_ready_timeout
        last_report = None
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            self._sample_map()
            with self._lock:
                stats = dict(self._latest_statistics or {})
                grid = self._grid
                pose_age = time.monotonic() - self._last_pose_wall
            free_voxels = int(stats.get("free_voxel_count", 0))
            frontier_count = self._projection_frontier_count(grid) if grid else 0
            voxel_ready = (
                free_voxels >= self._exploration_ready_free_voxel_threshold)
            frontier_ready = frontier_count > 0
            if voxel_ready:
                self._mark_startup_time("voxel_ready")
            if frontier_ready:
                self._mark_startup_time("frontier_ready")
            self._exploration_ready_frontier_count = frontier_count
            report = (voxel_ready, frontier_ready, free_voxels, frontier_count)
            if report != last_report:
                self._event(
                    "exploration_ready_check", voxel_ready=voxel_ready,
                    frontier_ready=frontier_ready, free_voxels=free_voxels,
                    frontier_count=frontier_count)
                last_report = report
            if voxel_ready and frontier_ready and pose_age < 0.75:
                self._exploration_ready = True
                self._event(
                    "exploration_ready", free_voxels=free_voxels,
                    frontier_count=frontier_count)
                return True
            time.sleep(0.25)
        return False

    def _write_planner_history(self):
        path = os.path.join(self._output_dir, "planner_history.csv")
        with self._lock:
            trajectory = list(self._trajectory)
            goals = list(self._goal_history)
        with open(path, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["# x", "y", "source"])
            for item in trajectory[::10]:
                writer.writerow([item["x"], item["y"], "trajectory"])
            for item in goals:
                position = item.get("position") or []
                if len(position) >= 2:
                    # A failed goal is governed by the expiring blacklist.  It
                    # must not become a permanent execution-point exclusion.
                    source = "goal" if item.get("success") else "failed_goal"
                    writer.writerow([position[0], position[1], source])
        return path

    def _run_planner(self, cycle_index, pose, frame, relaxed=False):
        cycle_dir = os.path.join(self._cycle_root, "goal_{:03d}".format(cycle_index))
        if relaxed:
            cycle_dir += "_recovery"
        os.makedirs(cycle_dir, exist_ok=True)
        statistics_snapshot = self._load_json(self._statistics_file)
        if statistics_snapshot:
            self._write_json(os.path.join(cycle_dir, "map_statistics_snapshot.json"),
                             statistics_snapshot)
        try:
            shutil.copy2(self._map_file, os.path.join(cycle_dir, "voxel_map_snapshot.bt"))
        except OSError:
            pass
        history_file = self._write_planner_history()
        command = [
            "rosrun", "simenv_competitor", "fuel_lite_planner",
            "__name:=fuel_lite_cycle_{:03d}".format(cycle_index),
            "_map_file:=" + self._map_file,
            "_output_dir:=" + cycle_dir,
            "_frame_id:=" + frame,
            "_use_pose_parameters:=true", "_exit_after_plan:=true",
            "_robot_x:={:.9f}".format(pose["x"]),
            "_robot_y:={:.9f}".format(pose["y"]),
            "_robot_z:={:.9f}".format(pose["z"]),
            "_frontier_slice_half_height:=0.30", "_minimum_cluster_size:=8",
            "_maximum_clusters:=80", "_candidates_per_cluster:=6",
            "_candidate_min_distance:={}".format(0.25 if relaxed else 0.30),
            "_candidate_max_distance:={}".format(2.3 if relaxed else 2.0),
            "_safety_clearance:=0.45",
            "_adaptive_clearance:={}".format(0.25 if relaxed else 0.30),
            "_minimum_information_gain:={}".format(2.0 if relaxed else 5.0),
            "_history_file:=" + history_file, "_sensor_range:=6.0",
            "_alpha:=0.25", "_beta:=0.3", "_gamma:=1.0",
            "_unknown_weight:=5.0", "_cluster_weight:=1.0",
            "_novelty_weight:=600.0", "_revisit_weight:=500.0",
            "/exploration_goal:=/simenv/planner_raw_goal",
        ]
        if self._entry_constraint:
            command.extend([
                "_use_entry_halfspace:=true",
                "_entry_boundary_x:={:.9f}".format(self._entry_constraint["x"]),
                "_entry_boundary_y:={:.9f}".format(self._entry_constraint["y"]),
                "_entry_forward_x:={:.9f}".format(self._entry_constraint["forward_x"]),
                "_entry_forward_y:={:.9f}".format(self._entry_constraint["forward_y"]),
            ])
        log_path = os.path.join(cycle_dir, "planner.log")
        try:
            with open(log_path, "w", encoding="utf-8") as stream:
                completed = subprocess.run(
                    command, stdout=stream, stderr=subprocess.STDOUT,
                    timeout=self._planner_timeout, check=False,
                )
            return cycle_dir, completed.returncode
        except subprocess.TimeoutExpired:
            return cycle_dir, 124

    def _planner_outputs(self, cycle_dir):
        frontiers = self._load_json(os.path.join(cycle_dir, "frontiers.json")) or []
        scores = self._load_json(os.path.join(cycle_dir, "exploration_score.json")) or {}
        goal = self._load_json(os.path.join(cycle_dir, "current_exploration_goal.json"))
        raw = int((scores.get("counts") or {}).get("raw_frontier_voxels", 0))
        score_items = scores.get("scores") or []
        max_gain = max((float(item.get("information_gain", 0.0)) for item in score_items),
                       default=0.0)
        return frontiers, raw, max_gain, goal

    def _select_executable_goal(self, cycle_dir, planner_goal, history):
        candidates = self._load_json(
            os.path.join(cycle_dir, "candidate_viewpoints.json")) or []
        score_payload = self._load_json(
            os.path.join(cycle_dir, "exploration_score.json")) or {}
        by_id = {int(item.get("id", -1)): item for item in candidates}
        ranked = sorted(score_payload.get("scores") or [],
                        key=lambda item: float(item.get("score", -math.inf)),
                        reverse=True)
        rejected = 0
        rejected_frontiers = set()
        for score in ranked:
            candidate = by_id.get(int(score.get("candidate_id", -1)))
            if not candidate:
                continue
            position = candidate.get("position") or []
            distance = float(candidate.get("distance_to_robot", math.inf))
            blacklisted = self._is_blacklisted(position) if len(position) >= 2 else True
            duplicate = (is_duplicate_goal(position, history, self._duplicate_radius)
                         if len(position) >= 2 else True)
            transit_revisit = bool(candidate.get("transit_revisit_fallback"))
            if (len(position) < 3 or distance > self._maximum_goal_distance or
                    blacklisted or (duplicate and not (
                        transit_revisit and distance >= 0.40))):
                rejected += 1
                if candidate.get("frontier_id") is not None:
                    rejected_frontiers.add(int(candidate["frontier_id"]))
                continue
            yaw = float(candidate.get("yaw", 0.0))
            self._annotate_manager_filter(cycle_dir, rejected, rejected_frontiers, 1)
            return {
                "frame_id": (planner_goal or {}).get("frame_id") or self._pose_frame,
                "candidate_id": candidate.get("id"),
                "frontier_id": candidate.get("frontier_id"),
                "position": position,
                "orientation": [0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)],
                "yaw": yaw,
                "information_gain": score.get("information_gain", 0.0),
                "score": score.get("score"),
                "score_decomposition": dict(score),
                "source": ("transit_revisit_fallback" if transit_revisit
                           else "frontier_candidate"),
            }
        self._annotate_manager_filter(cycle_dir, rejected, rejected_frontiers, 0)
        return None

    def _annotate_manager_filter(self, cycle_dir, rejected, rejected_frontiers,
                                 final_goal_count):
        path = os.path.join(cycle_dir, "frontier_filter_diagnostic.json")
        payload = self._load_json(path)
        if not payload:
            return
        counts = payload.setdefault("counts", {})
        counts["duplicate_or_blacklisted"] = int(rejected)
        counts["final_valid_goal_count"] = int(final_goal_count)
        payload["manager_filter"] = {
            "duplicate_or_blacklisted_candidates": int(rejected),
            "affected_frontier_ids": sorted(rejected_frontiers),
            "final_valid_goal_count": int(final_goal_count),
        }
        if not final_goal_count:
            for cluster in payload.get("clusters") or []:
                if int(cluster.get("id", -1)) in rejected_frontiers and \
                        cluster.get("reason") == "valid":
                    cluster["reason"] = "duplicate_or_blacklisted"
        self._write_json(path, payload)

    def _is_blacklisted(self, position):
        with self._lock:
            current_update = int((self._latest_statistics or {}).get("update_count", 0))
        current_cycle = len(self._frontier_history)
        return any(
            math.hypot(float(position[0]) - item["x"],
                       float(position[1]) - item["y"]) < 0.75 and
            item["expires_cycle"] >= current_cycle and
            current_update <= item.get("failed_map_update", current_update)
            for item in self._blacklist)

    def _historical_recovery_goal(self, history):
        """Reuse a still-safe, non-visited historical candidate for one recovery."""
        validator = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "..", "devel", "lib", "simenv_competitor", "octomap_goal_validator")
        validator = os.path.abspath(validator)
        for score_cycle in reversed(self._score_history[-8:]):
            cycle_dir = os.path.join(self._output_dir, score_cycle["planner_cycle"])
            candidates = self._load_json(os.path.join(
                cycle_dir, "candidate_viewpoints.json")) or []
            by_id = {int(item.get("id", -1)): item for item in candidates}
            ranked = sorted(score_cycle.get("scores") or [],
                            key=lambda item: float(item.get("score", -math.inf)),
                            reverse=True)
            for score in ranked:
                candidate = by_id.get(int(score.get("candidate_id", -1)))
                position = (candidate or {}).get("position") or []
                if len(position) < 3 or is_duplicate_goal(
                        position, history, max(self._duplicate_radius, 0.60)):
                    continue
                if self._is_blacklisted(position):
                    continue
                try:
                    checked = subprocess.run(
                        [validator, self._map_file, str(position[0]), str(position[1]),
                         str(position[2]), "0.30"], stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL, timeout=5.0, check=False)
                except (OSError, subprocess.TimeoutExpired):
                    continue
                if checked.returncode != 0:
                    continue
                yaw = float(candidate.get("yaw", 0.0))
                return {
                    "frame_id": self._pose_frame,
                    "candidate_id": candidate.get("id"),
                    "frontier_id": candidate.get("frontier_id"),
                    "position": position, "yaw": yaw,
                    "orientation": [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)],
                    "information_gain": score.get("information_gain", 0.0),
                    "score": score.get("score"), "score_decomposition": dict(score),
                    "source": "historical_recovery",
                }
        return None

    def _publish_goal(self, goal):
        position = goal["position"]
        orientation = goal.get("orientation") or [0.0, 0.0, 0.0, 1.0]
        message = PoseStamped()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = goal.get("frame_id") or self._pose_frame
        message.pose.position.x, message.pose.position.y = position[0], position[1]
        message.pose.position.z = position[2]
        message.pose.orientation.x, message.pose.orientation.y = orientation[0], orientation[1]
        message.pose.orientation.z, message.pose.orientation.w = orientation[2], orientation[3]
        with self._lock:
            baseline = len(self._execution_results)
        self._goal_pub.publish(message)
        return baseline

    def _wait_goal_result(self, baseline):
        remaining_mission = max(
            0.0, self._maximum_duration - self._finalization_reserve - self._elapsed())
        deadline = time.monotonic() + min(self._goal_timeout, remaining_mission)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            with self._lock:
                if len(self._execution_results) > baseline:
                    return dict(self._execution_results[-1])
                pose_age = time.monotonic() - self._last_pose_wall
            if pose_age > 0.75:
                return {"success": False, "reason": "localization_stale"}
            time.sleep(0.10)
        return {"success": False, "reason": "manager_goal_timeout"}

    def _write_artifacts(self):
        with self._lock:
            trajectory = list(self._trajectory)
            coverage = list(self._coverage_history)
            frontiers = list(self._frontier_history)
            goals = list(self._goal_history)
            events = list(self._events)
            stats = dict(self._latest_statistics or {})
            visited = list(self._visited_history)
            scores = list(self._score_history)
            startup_timing = dict(self._startup_timing)
        with open(os.path.join(self._output_dir, "trajectory.csv"), "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["t", "x", "y", "z", "yaw"])
            writer.writeheader(); writer.writerows(trajectory)
        with open(os.path.join(self._output_dir, "coverage_history.csv"), "w", newline="", encoding="utf-8") as stream:
            fields = ["t", "coverage_ratio", "free_voxels", "occupied_voxels", "unknown_voxels", "update_count"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(coverage)
        with open(os.path.join(self._output_dir, "frontier_history.csv"), "w", newline="", encoding="utf-8") as stream:
            fields = ["t", "frontier_count", "raw_frontier_voxels", "max_information_gain"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(frontiers)
        with open(os.path.join(self._output_dir, "visited_area_history.csv"), "w", newline="", encoding="utf-8") as stream:
            fields = ["t", "visited_area_m2", "observed_free_area_proxy_m2"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(visited)
        self._write_json(os.path.join(self._output_dir, "goal_history.json"), goals)
        self._write_json(os.path.join(self._output_dir, "goal_score_history.json"), scores)
        self._write_json(os.path.join(
            self._output_dir, "exploration_startup_status.json"), {
                "schema": "simenv_direct_fuel_startup_status_v1",
                "startup_mode": "sensor_warmup_direct_fuel",
                "fast_lio_initialized": bool(self._pose),
                "voxel_map_updates": int(stats.get("update_count", 0)),
                "free_voxels": int(stats.get("free_voxel_count", 0)),
                "free_voxel_threshold":
                    self._exploration_ready_free_voxel_threshold,
                "exploration_ready": bool(self._exploration_ready),
                "fuel_plan_count": int(self._fuel_plan_count),
                "frontier_count_at_ready":
                    int(self._exploration_ready_frontier_count),
                "timing": startup_timing,
                "termination_reason": self._termination or "ros_shutdown",
            })
        self._write_json(os.path.join(self._output_dir, "termination_diagnostic.json"), {
            "schema": "simenv_termination_diagnostic_v1",
            "decision": self._termination or "ros_shutdown",
            "required_empty_cycles": self._frontier_empty_confirmations,
            "required_empty_seconds": self._frontier_empty_seconds,
            "checks": self._termination_checks,
            "blacklist": self._blacklist,
        })
        cycle_diagnostics = []
        for path in sorted(glob.glob(os.path.join(
                self._cycle_root, "goal_*", "frontier_filter_diagnostic.json"))):
            payload = self._load_json(path)
            if payload:
                cycle_diagnostics.append({
                    "planner_cycle": os.path.relpath(os.path.dirname(path), self._output_dir),
                    **payload,
                })
        self._write_json(os.path.join(self._output_dir, "frontier_filter_diagnostic.json"), {
            "schema": "simenv_frontier_filter_diagnostic_history_v1",
            "cycles": cycle_diagnostics,
            "last_cycle": cycle_diagnostics[-1] if cycle_diagnostics else None,
        })
        cycle_images = sorted(glob.glob(os.path.join(
            self._cycle_root, "goal_*", "frontier_filter_diagnostic.png")))
        if cycle_images:
            shutil.copy2(cycle_images[-1], os.path.join(
                self._output_dir, "frontier_filter_diagnostic.png"))
        self._write_json(os.path.join(self._output_dir, "exploration_log.json"), {
            "schema": "simenv_first_floor_exploration_log_v1", "events": events,
        })
        wall_elapsed = self._elapsed()
        ros_now = rospy.Time.now().to_sec()
        ros_elapsed = (max(0.0, ros_now - self._start_ros)
                       if self._start_ros is not None else 0.0)
        total_wall_elapsed = time.monotonic() - self._node_started_wall
        summary = {
            "schema": "simenv_first_floor_exploration_summary_v1",
            "passed": bool(self._exploration_ready and self._fuel_plan_count > 0),
            "closed_loop_passed": bool(
                goals and any(item.get("success") for item in goals) and
                not str(self._termination or "").startswith("exception")),
            "exploration_complete": False,
            "rooms_entered": 0,
            # Kept for compatibility; this is exploration wall-clock time.
            "total_time": round(wall_elapsed, 3),
            "timing": {
                "exploration_wall_elapsed_sec": round(wall_elapsed, 3),
                "exploration_sim_elapsed_sec": round(ros_elapsed, 3),
                "total_wall_elapsed_including_startup_sec": round(
                    total_wall_elapsed, 3),
                "final_sim_time_sec": round(ros_now, 3),
                "real_time_factor": round(
                    ros_elapsed / wall_elapsed, 4) if wall_elapsed > 0.0 else None,
            },
            "configured_limits": {
                "maximum_duration_sec": self._maximum_duration,
                "maximum_goals": self._maximum_goals,
                "maximum_goals_unlimited": self._maximum_goals <= 0,
            },
            "number_of_goals": len(goals),
            "reached_goals": sum(bool(item.get("success")) for item in goals),
            "coverage_ratio": coverage_from_statistics(stats),
            "visited_area": visited[-1]["visited_area_m2"] if visited else 0.0,
            "observed_free_area_proxy": visited_area_from_statistics(stats),
            "visited_area_definition": "trajectory swept disk, radius {:.2f} m".format(
                self._visited_radius),
            "termination_reason": self._termination or "ros_shutdown",
            "startup_mode": "sensor_warmup_direct_fuel",
            "exploration_ready": bool(self._exploration_ready),
            "fuel_plan_count": int(self._fuel_plan_count),
            "online_entry_constraint": self._entry_constraint,
            "final_map_statistics": stats,
            "initial_map_statistics": coverage[0] if coverage else {},
            "map_growth": {
                "free_voxels": (int(stats.get("free_voxel_count", 0)) -
                                int(coverage[0].get("free_voxels", 0))) if coverage else 0,
                "updates": (int(stats.get("update_count", 0)) -
                            int(coverage[0].get("update_count", 0))) if coverage else 0,
            },
            "artifacts": {
                "trajectory": "trajectory.csv", "coverage": "coverage_history.csv",
                "frontiers": "frontier_history.csv", "goals": "goal_history.json",
                "visited": "visited_area_history.csv",
                "scores": "goal_score_history.json",
                "termination": "termination_diagnostic.json",
                "startup": "exploration_startup_status.json",
                "log": "exploration_log.json",
            },
        }
        self._write_json(os.path.join(self._output_dir, "exploration_summary.json"), summary)
        return summary

    def run(self):
        self._complete_pub.publish(Bool(data=False))
        if not self._wait_ready():
            self._termination = "startup_timeout"
            self._write_artifacts()
            return
        self._start_wall = time.monotonic()
        self._start_ros = rospy.Time.now().to_sec()
        self._event("exploration_started")
        if self._post_ready_dwell > 0.0:
            self._event("localization_settle_started", seconds=self._post_ready_dwell)
            deadline = time.monotonic() + self._post_ready_dwell
            while not rospy.is_shutdown() and time.monotonic() < deadline:
                self._sample_map()
                time.sleep(0.1)
            # Static settling must not consume the free-voxel stagnation budget.
            self._last_free_growth = time.monotonic()
            self._last_visited_growth = time.monotonic()
            self._event("localization_settle_finished")
        # Direct-FUEL startup: the robot remains stationary while LiDAR builds
        # enough known free space and at least one frontier cluster.
        self._speed_limit_pub.publish(Float32(data=self._exploration_speed))
        if not self._wait_exploration_ready():
            self._termination = "exploration_ready_timeout"
            self._event("exploration_ready_timeout")
        cycle = 0
        consecutive_failures = 0
        while not rospy.is_shutdown() and self._termination is None:
            self._sample_map()
            with self._lock:
                pose = dict(self._pose) if self._pose else None
                frame = self._pose_frame
                stats = dict(self._latest_statistics or {})
                stagnant = time.monotonic() - self._last_free_growth
                visited_stagnant = time.monotonic() - self._last_visited_growth
            if self._elapsed() >= self._maximum_duration - self._finalization_reserve:
                self._termination = "timeout"
                break
            with self._lock:
                executed_goals = sum(item.get("index", 0) > 0 for item in self._goal_history)
            # A non-positive goal limit means time/frontier termination only.
            if self._maximum_goals > 0 and executed_goals >= self._maximum_goals:
                self._termination = "maximum_goals"
                break
            cycle += 1
            cycle_dir, return_code = self._run_planner(cycle, pose, frame)
            self._fuel_plan_count += 1
            frontiers, raw_frontiers, max_gain, goal = self._planner_outputs(cycle_dir)
            if goal:
                self._mark_startup_time("first_fuel_goal")
            score_payload = self._load_json(
                os.path.join(cycle_dir, "exploration_score.json")) or {}
            filter_payload = self._load_json(
                os.path.join(cycle_dir, "frontier_filter_diagnostic.json")) or {}
            self._score_history.append({
                "cycle": cycle, "t": round(self._elapsed(), 3),
                "planner_cycle": os.path.relpath(cycle_dir, self._output_dir),
                "selected_candidate_id": score_payload.get("selected_candidate_id"),
                "scores": score_payload.get("scores") or [],
            })
            frontier_sample = {
                "t": round(self._elapsed(), 3), "frontier_count": len(frontiers),
                "raw_frontier_voxels": raw_frontiers,
                "max_information_gain": max_gain,
            }
            with self._lock:
                self._frontier_history.append(frontier_sample)
            valid_candidates = int((filter_payload.get("counts") or {}).get(
                "valid_candidates", 0))
            clusters = filter_payload.get("clusters") or []
            large_unknown = any(
                int(item.get("size", 0)) >= 24 and
                int(item.get("candidate_count_after_connectivity", 0)) > 0
                for item in clusters)
            planner_empty = return_code != 0 or not goal or valid_candidates == 0
            if planner_empty:
                if self._empty_since is None:
                    self._empty_since = time.monotonic()
                self._empty_cycles += 1
            empty_seconds = (0.0 if self._empty_since is None else
                             time.monotonic() - self._empty_since)
            check = {
                "cycle": cycle, "t": round(self._elapsed(), 3),
                "planner_empty": planner_empty, "empty_cycles": self._empty_cycles,
                "empty_seconds": round(empty_seconds, 3),
                "visited_growth_stagnant_seconds": round(visited_stagnant, 3),
                "observed_free_growth_stagnant_seconds": round(stagnant, 3),
                "raw_frontier_voxels": raw_frontiers,
                "large_connected_unknown_remains": large_unknown,
                "termination_allowed": confirmed_frontier_exhaustion(
                    empty_cycles=self._empty_cycles, empty_seconds=empty_seconds,
                    visited_stagnant_seconds=visited_stagnant,
                    large_connected_unknown_remains=large_unknown,
                    required_cycles=self._frontier_empty_confirmations,
                    required_seconds=self._frontier_empty_seconds,
                    stagnation_seconds=self._stagnation_threshold),
            }
            self._termination_checks.append(check)
            if check["termination_allowed"]:
                self._termination = "frontier_empty_confirmed"
                break
            if planner_empty:
                self._event("planner_recovery_started", cycle=cycle,
                            empty_cycles=self._empty_cycles)
                cycle_dir, return_code = self._run_planner(cycle, pose, frame, relaxed=True)
                self._fuel_plan_count += 1
                frontiers, raw_frontiers, max_gain, goal = self._planner_outputs(cycle_dir)
                if goal:
                    self._mark_startup_time("first_fuel_goal")
                recovery_scores = self._load_json(
                    os.path.join(cycle_dir, "exploration_score.json")) or {}
                self._score_history.append({
                    "cycle": cycle, "recovery": True, "t": round(self._elapsed(), 3),
                    "planner_cycle": os.path.relpath(cycle_dir, self._output_dir),
                    "selected_candidate_id": recovery_scores.get("selected_candidate_id"),
                    "scores": recovery_scores.get("scores") or [],
                })
                if return_code != 0 or not goal:
                    with self._lock:
                        historical_positions = [
                            item["position"] for item in self._goal_history
                            if item.get("success")]
                    goal = self._historical_recovery_goal(historical_positions)
                    if not goal:
                        self._event("planner_recovery_failed", cycle=cycle)
                        time.sleep(1.0)
                        continue
                    self._event("historical_frontier_recovery_selected", cycle=cycle,
                                position=goal["position"])
                else:
                    self._event("planner_recovery_succeeded", cycle=cycle)
            with self._lock:
                previous_positions = [
                    item["position"] for item in self._goal_history
                    if item.get("success")]
            if goal.get("source") != "historical_recovery":
                goal = self._select_executable_goal(cycle_dir, goal, previous_positions)
            if not goal:
                goal = self._historical_recovery_goal(previous_positions)
                if goal:
                    self._event("historical_frontier_recovery_selected", cycle=cycle,
                                position=goal["position"], trigger="manager_filter_empty")
            if not goal:
                self._event("no_new_executable_goal", cycle=cycle)
                if self._empty_since is None:
                    self._empty_since = time.monotonic()
                self._empty_cycles += 1
                empty_seconds = time.monotonic() - self._empty_since
                allowed = confirmed_frontier_exhaustion(
                    empty_cycles=self._empty_cycles, empty_seconds=empty_seconds,
                    visited_stagnant_seconds=visited_stagnant,
                    large_connected_unknown_remains=large_unknown,
                    required_cycles=self._frontier_empty_confirmations,
                    required_seconds=self._frontier_empty_seconds,
                    stagnation_seconds=self._stagnation_threshold)
                self._termination_checks.append({
                    "cycle": cycle, "t": round(self._elapsed(), 3),
                    "stage": "manager_duplicate_or_blacklist_filter",
                    "planner_empty": False, "effective_goal_empty": True,
                    "empty_cycles": self._empty_cycles,
                    "empty_seconds": round(empty_seconds, 3),
                    "visited_growth_stagnant_seconds": round(visited_stagnant, 3),
                    "large_connected_unknown_remains": large_unknown,
                    "termination_allowed": allowed,
                })
                if allowed:
                    self._termination = "frontier_empty_confirmed"
                    break
                time.sleep(1.0)
                continue
            position = goal["position"]
            self._empty_since = None
            self._empty_cycles = 0
            self._event("goal_selected", index=cycle, position=position,
                        frontier_count=len(frontiers), information_gain=max_gain)
            baseline = self._publish_goal(goal)
            result = self._wait_goal_result(baseline)
            history = {
                "index": cycle, "position": position,
                "source": goal.get("source", "frontier_candidate"),
                "orientation": goal.get("orientation"),
                "candidate_id": goal.get("candidate_id"),
                "frontier_id": goal.get("frontier_id"),
                "information_gain": goal.get("information_gain", max_gain),
                "score_decomposition": goal.get("score_decomposition"),
                "planner_cycle": os.path.relpath(cycle_dir, self._output_dir),
                "success": bool(result.get("success")),
                "result": result,
            }
            with self._lock:
                self._goal_history.append(history)
                for score_cycle in reversed(self._score_history):
                    if score_cycle.get("cycle") == cycle:
                        score_cycle["executed_candidate_id"] = goal.get("candidate_id")
                        score_cycle["executed_goal_position"] = list(position)
                        score_cycle["executed_score_decomposition"] = goal.get(
                            "score_decomposition")
                        break
            self._event("goal_finished", index=cycle, success=history["success"],
                        reason=result.get("reason"))
            if not history["success"]:
                self._blacklist.append({
                    "x": float(position[0]), "y": float(position[1]),
                    "failed_cycle": cycle, "expires_cycle": cycle + 5,
                    "failed_map_update": int(stats.get("update_count", 0)),
                    "reason": result.get("reason", "unknown"),
                })
                consecutive_failures += 1
                if consecutive_failures >= self._maximum_consecutive_failures:
                    self._termination = "consecutive_goal_failures"
                    break
                self._event("goal_skipped_after_failure", index=cycle,
                            consecutive_failures=consecutive_failures)
                time.sleep(1.0)
                continue
            consecutive_failures = 0
            # Let the mapper atomically publish at least one post-motion snapshot.
            before_update = int(stats.get("update_count", 0))
            refresh_deadline = time.monotonic() + 8.0
            while time.monotonic() < refresh_deadline and not rospy.is_shutdown():
                self._sample_map()
                with self._lock:
                    current_update = int((self._latest_statistics or {}).get("update_count", 0))
                if current_update > before_update:
                    break
                time.sleep(0.20)

        self._event("exploration_terminated", reason=self._termination)
        self._complete_pub.publish(Bool(data=True))
        time.sleep(0.8)
        summary = self._write_artifacts()
        rospy.loginfo("Exploration manager finished: %s goals=%d coverage=%.3f",
                      summary["termination_reason"], summary["number_of_goals"],
                      summary["coverage_ratio"])

    def _shutdown(self):
        if not os.path.exists(os.path.join(self._output_dir, "exploration_summary.json")):
            self._write_artifacts()


if __name__ == "__main__":
    rospy.init_node("exploration_manager")
    ExplorationManager().run()
