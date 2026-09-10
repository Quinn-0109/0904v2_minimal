#!/usr/bin/env python3
"""Normalize native UFOExplorer output and own ordered backend fallback."""

import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from std_msgs.msg import String

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_SCRIPT_DIR = os.path.dirname(SCRIPT_DIR)
for module_dir in (SCRIPT_DIR, PARENT_SCRIPT_DIR):
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

from hierarchical_ros_utils import occupancy_from_message
from planner_interface import AStarPlanner
from corridor_semantic_core import score_path_for_corridor


class UfoExplorerInterface:
    def __init__(self):
        self._lock = threading.RLock()
        self._output_dir = os.path.abspath(rospy.get_param("~output_dir", "/tmp/ufo"))
        self._log_path = os.path.join(
            self._output_dir, "logs", "ufoexplorer_goal_history.jsonl")
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        self._frame = rospy.get_param("~world_frame", "map")
        self._timeout = float(rospy.get_param("~ufo_output_timeout", 5.0))
        self._failure_limit = int(rospy.get_param("~ufo_failure_limit", 3))
        self._recover_after = float(rospy.get_param(
            "~fallback_recover_after_seconds", 30.0))
        self._fallback_order = rospy.get_param(
            "~explorer_fallback_order",
            ["ufoexplorer", "fuel_lite", "astar_frontier"])
        self._map_file = os.path.abspath(rospy.get_param(
            "~map_file", os.path.join(self._output_dir, "voxel_map.bt")))
        self._pose = None
        self._grid = None
        self._grid_observed = False
        self._last_ufo = time.monotonic()
        self._last_failure = -1e9
        self._failure_count = 0
        self._fallback_until = -1e9
        self._fallback_running = False
        self._corridor = None
        self._semantic_candidate = None
        self._semantic_candidate_score = -math.inf
        self._semantic_window_started = None
        self._semantic_window = float(rospy.get_param(
            "~corridor_path_selection_window", 1.0))
        self._semantic_startup_wait = float(rospy.get_param(
            "~corridor_semantic_startup_wait", 6.0))
        self._input_ready_at = None
        self._held_startup_path = None
        self._first_raw_path_at = None
        self._corridor_bootstrap_sent = False
        self._room_mission_active = False
        self._held_during_room = None
        self._room_release_at = -1e9
        self._path_pub = rospy.Publisher(
            "/simenv/ufo_exploration_path", Path, queue_size=2, latch=True)
        self._goal_pub = rospy.Publisher(
            "/simenv/ufo_exploration_goal", PoseStamped, queue_size=2, latch=True)
        self._status_pub = rospy.Publisher(
            "/simenv/ufoexplorer_status", String, queue_size=10, latch=True)
        rospy.Subscriber(rospy.get_param(
            "~raw_path_topic", "/ufoexplorer_graph_node/path"),
            Path, self._on_path, queue_size=2)
        rospy.Subscriber(rospy.get_param(
            "~raw_goal_topic", "/ufoexplorer/goal"),
            PoseStamped, self._on_goal, queue_size=2)
        rospy.Subscriber(rospy.get_param("~odom_topic", "/simenv/ufo/odometry"),
                         Odometry, self._on_odom, queue_size=20)
        rospy.Subscriber(rospy.get_param(
            "~map_topic", "/simenv/voxel_floor_projection"),
            OccupancyGrid, self._on_map, queue_size=2)
        rospy.Subscriber("/simenv/ufoexplorer_replan_request",
                         String, self._on_replan, queue_size=10)
        rospy.Subscriber("/simenv/ufo_corridor_semantics",
                         String, self._on_corridor, queue_size=5)
        rospy.Subscriber("/simenv/ufo_room_mission_status",
                         String, self._on_room_status, queue_size=10)
        rospy.Subscriber("/simenv/ufo_room_mission_path",
                         Path, self._on_room_path, queue_size=2)
        self._astar = AStarPlanner(
            clearance=float(rospy.get_param("~astar_clearance", 0.30)),
            reached_tolerance=0.15, maximum_expansions=100000,
            allow_blocked_start=True)
        rospy.Timer(rospy.Duration(1.0), self._on_timer)
        self._publish_status("WAITING_FOR_UFO", reason="startup")

    def _record(self, **payload):
        payload.update(timestamp=time.time(),
                       robot_pose=list(self._pose) if self._pose else None)
        with open(self._log_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")

    def _publish_status(self, state, **extra):
        payload = dict(state=state, failure_count=self._failure_count,
                       fallback_active=time.monotonic() < self._fallback_until)
        payload.update(extra)
        self._status_pub.publish(String(data=json.dumps(payload, sort_keys=True)))

    def _on_odom(self, message):
        with self._lock:
            self._pose = (message.pose.pose.position.x,
                          message.pose.pose.position.y,
                          message.pose.pose.position.z)
            if self._grid_observed and self._input_ready_at is None:
                self._input_ready_at = time.monotonic()

    def _on_map(self, message):
        try:
            grid = occupancy_from_message(message)
        except (IndexError, TypeError, ValueError):
            return
        with self._lock:
            self._grid = grid
            self._grid_observed = int(np.count_nonzero(grid.data >= 0)) >= 100
            if (self._grid_observed and self._pose is not None and
                    self._input_ready_at is None):
                self._input_ready_at = time.monotonic()

    @staticmethod
    def _path_length(message):
        return sum(math.hypot(
            b.pose.position.x - a.pose.position.x,
            b.pose.position.y - a.pose.position.y)
            for a, b in zip(message.poses[:-1], message.poses[1:]))

    def _accept_path(self, message, source, semantic_score=None):
        if not message.poses:
            self._fail("UFO_PATH_EMPTY")
            return
        length = self._path_length(message)
        if length < 0.15:
            self._fail("UFO_PATH_TOO_SHORT")
            return
        output = message
        output.header.frame_id = output.header.frame_id or self._frame
        self._path_pub.publish(output)
        self._goal_pub.publish(output.poses[-1])
        self._failure_count = 0
        self._last_ufo = time.monotonic()
        self._publish_status("PATH_READY", source=source,
                             raw_path_length=length,
                             waypoint_count=len(output.poses),
                             corridor_semantic_score=semantic_score,
                             corridor_semantics=self._corridor)
        self._record(ufo_raw_output_type="path", ufo_raw_goal=None,
                     ufo_raw_path_length=length,
                     trimmed_path_length=None, selected_waypoints=[],
                     scan_lite_status="PENDING", executor_status="PENDING",
                     fallback_used=source in ("fuel_lite", "astar_frontier"),
                     reason=source,
                     corridor_semantic_score=semantic_score,
                     corridor_semantics=self._corridor)

    def _on_corridor(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        with self._lock:
            self._corridor = payload if payload.get("confirmed") else None
            if self._corridor is None or self._pose is None or self._grid is None:
                return
            if self._corridor.get("robot_inside"):
                self._corridor_bootstrap_sent = True
                return
            if self._corridor_bootstrap_sent:
                return
            approach = self._corridor.get("approach_point")
            if not approach or len(approach) < 2:
                return
            result = self._astar.plan(self._pose[:2], approach[:2], self._grid)
            if result.success and result.travel_cost >= 0.6:
                path = self._astar.as_nav_path(result, self._frame)
                score = score_path_for_corridor(
                    [(item.pose.position.x, item.pose.position.y)
                     for item in path.poses], self._pose[:2], self._corridor)
                self._held_startup_path = None
                self._corridor_bootstrap_sent = True
                self._accept_path(path, "ufo_corridor_semantic_astar",
                                  semantic_score=score)

    def _queue_semantic_path(self, message):
        points = [(pose.pose.position.x, pose.pose.position.y)
                  for pose in message.poses]
        score = score_path_for_corridor(points, self._pose[:2], self._corridor)
        now = time.monotonic()
        if self._semantic_window_started is None:
            self._semantic_window_started = now
        if score > self._semantic_candidate_score:
            self._semantic_candidate = message
            self._semantic_candidate_score = score
        return now - self._semantic_window_started >= self._semantic_window

    def _flush_semantic_path(self):
        if self._semantic_candidate is None:
            return False
        message = self._semantic_candidate
        score = self._semantic_candidate_score
        self._semantic_candidate = None
        self._semantic_candidate_score = -math.inf
        self._semantic_window_started = None
        self._accept_path(message, "ufoexplorer", semantic_score=score)
        return True

    def _on_path(self, message):
        with self._lock:
            if self._room_mission_active:
                # Keep only diagnostics/a freshness hint.  A native path
                # observed before the room exit is stale afterwards and is
                # deliberately not replayed.
                self._held_during_room = message
                self._last_ufo = time.monotonic()
                return
            if time.monotonic() < self._fallback_until:
                return
            if self._first_raw_path_at is None:
                self._first_raw_path_at = time.monotonic()
            if (self._corridor is None and
                    time.monotonic() - self._first_raw_path_at <
                    self._semantic_startup_wait):
                # Preserve the newest native candidate, but do not let it move
                # the robot before the initial LiDAR geometry vote completes.
                self._held_startup_path = message
                return
            if self._corridor is not None and self._pose is not None:
                if self._queue_semantic_path(message):
                    self._flush_semantic_path()
                return
            self._accept_path(message, "ufoexplorer")

    def _on_room_status(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        with self._lock:
            was_active = self._room_mission_active
            self._room_mission_active = bool(payload.get("active"))
            if was_active and not self._room_mission_active:
                # Resume from a fresh UFO graph update after the saved door
                # exit anchor, never from a pre-room path prefix.
                self._held_during_room = None
                self._last_ufo = time.monotonic()
                self._room_release_at = time.monotonic()
                self._publish_status("ROOM_MISSION_RELEASED",
                                     reason=payload.get("reason"),
                                     room_state=payload.get("state"))

    def _on_room_path(self, message):
        with self._lock:
            self._room_mission_active = True
            self._fallback_until = -1e9
            self._accept_path(message, "ufo_room_semantic")

    def _on_goal(self, message):
        with self._lock:
            if self._room_mission_active:
                self._last_ufo = time.monotonic()
                return
            if self._pose is None or self._grid is None:
                self._fail("UFO_GOAL_WITHOUT_MAP")
                return
            goal = (message.pose.position.x, message.pose.position.y)
            result = self._astar.plan(self._pose[:2], goal, self._grid)
            if not result.success:
                self._fail("UFO_GOAL_UNREACHABLE")
                return
            self._accept_path(self._astar.as_nav_path(
                result, message.header.frame_id or self._frame), "ufo_goal_astar")

    def _on_replan(self, message):
        with self._lock:
            if self._room_mission_active:
                self._publish_status("ROOM_MISSION_REPLAN",
                                     reason=message.data or "REPLAN_REQUESTED")
                return
            if (message.data == "LOCAL_PATH_COMPLETE" and
                    time.monotonic() - self._room_release_at < 2.0):
                # The adapter reports completion just after the room node has
                # released ownership.  This is a successful exit, not a UFO
                # backend failure deserving fallback escalation.
                self._publish_status("WAITING_FRESH_UFO_AFTER_ROOM",
                                     reason="room_exit_path_complete")
                return
            self._fail(message.data or "REPLAN_REQUESTED")

    def _fail(self, reason):
        now = time.monotonic()
        # Count one backend failure per complete output-timeout window.  A
        # high-rate timer must not consume the failure budget in a few ticks.
        if now - self._last_failure < self._timeout:
            return
        self._last_failure = now
        self._failure_count += 1
        self._publish_status(reason, reason=reason)
        self._record(ufo_raw_output_type="empty", ufo_raw_goal=None,
                     ufo_raw_path_length=0.0, trimmed_path_length=0.0,
                     selected_waypoints=[], scan_lite_status="NOT_RUN",
                     executor_status="WAITING", fallback_used=False,
                     reason=reason)
        if self._failure_count >= self._failure_limit and not self._fallback_running:
            self._fallback_running = True
            threading.Thread(target=self._run_fallback, daemon=True).start()

    def _run_fuel(self):
        if not os.path.isfile(self._map_file) or self._pose is None:
            return None
        temporary = tempfile.mkdtemp(prefix="ufo_fuel_fallback_")
        command = [
            "rosrun", "simenv_competitor", "fuel_lite_planner",
            "__name:=ufo_fuel_fallback", "_map_file:=" + self._map_file,
            "_output_dir:=" + temporary, "_frame_id:=" + self._frame,
            "_use_pose_parameters:=true", "_exit_after_plan:=true",
            "_robot_x:=" + str(self._pose[0]), "_robot_y:=" + str(self._pose[1]),
            "_robot_z:=" + str(self._pose[2]), "_minimum_cluster_size:=6",
            "_maximum_clusters:=30", "_candidate_min_distance:=0.6",
            "_candidate_max_distance:=3.2"]
        try:
            subprocess.run(command, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=12.0, check=False)
            candidate_file = os.path.join(temporary, "candidate_viewpoints.json")
            with open(candidate_file, encoding="utf-8") as stream:
                candidates = json.load(stream)
            for candidate in candidates:
                point = candidate.get("position") or candidate.get("goal")
                if point and len(point) >= 2:
                    result = self._astar.plan(self._pose[:2], point[:2], self._grid)
                    if result.success:
                        return self._astar.as_nav_path(result, self._frame)
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
        return None

    def _astar_frontier(self):
        if self._grid is None or self._pose is None:
            return None
        data = self._grid.data
        candidates = []
        for y in range(1, self._grid.height - 1):
            for x in range(1, self._grid.width - 1):
                if data[y, x] != 0:
                    continue
                neighborhood = data[y - 1:y + 2, x - 1:x + 2]
                unknown = int(np.count_nonzero(neighborhood < 0))
                if unknown == 0:
                    continue
                point = self._grid.cell_to_world((x, y))
                dist = math.hypot(point[0] - self._pose[0],
                                  point[1] - self._pose[1])
                if 0.6 <= dist <= 6.0:
                    candidates.append((unknown * 10.0 + dist, point))
        for _, point in sorted(candidates, reverse=True)[:80]:
            result = self._astar.plan(self._pose[:2], point, self._grid)
            if result.success and result.travel_cost >= 0.6:
                return self._astar.as_nav_path(result, self._frame)
        return None

    def _run_fallback(self):
        selected = None
        source = None
        with self._lock:
            order = list(self._fallback_order)
        for backend in order:
            if backend == "ufoexplorer":
                continue
            # FUEL may take several seconds.  Do not hold the callback lock:
            # odometry and map snapshots must continue updating meanwhile.
            selected = (self._run_fuel() if backend == "fuel_lite"
                        else self._astar_frontier()
                        if backend == "astar_frontier" else None)
            if selected is not None:
                source = backend
                break
        with self._lock:
            if selected is not None:
                self._fallback_until = time.monotonic() + self._recover_after
                self._accept_path(selected, source)
            else:
                self._publish_status("FALLBACK_FAILED", reason="all_backends_failed")
            self._fallback_running = False

    def _on_timer(self, _event):
        with self._lock:
            if self._room_mission_active:
                self._publish_status("ROOM_MISSION_ACTIVE",
                                     reason="semantic_room_path_owns_execution")
                return
            if (self._semantic_window_started is not None and
                    time.monotonic() - self._semantic_window_started >=
                    self._semantic_window):
                self._flush_semantic_path()
            if self._pose is None or self._grid is None:
                self._publish_status("WAITING_INPUT", reason="pose_or_map_missing")
                return
            if (self._held_startup_path is not None and
                    self._first_raw_path_at is not None and
                    time.monotonic() - self._first_raw_path_at >=
                    self._semantic_startup_wait):
                held, self._held_startup_path = self._held_startup_path, None
                if self._corridor is None:
                    self._accept_path(held, "ufoexplorer_no_corridor")
            if (self._first_raw_path_at is not None and self._corridor is None and
                    time.monotonic() - self._first_raw_path_at <
                    self._semantic_startup_wait):
                self._publish_status("WAITING_CORRIDOR_SEMANTICS",
                                     reason="initial_parallel_wall_vote")
                return
            if (time.monotonic() >= self._fallback_until and
                    time.monotonic() - self._last_ufo > self._timeout):
                self._fail("UFO_NO_PATH")


if __name__ == "__main__":
    rospy.init_node("ufoexplorer_interface")
    UfoExplorerInterface()
    rospy.spin()
