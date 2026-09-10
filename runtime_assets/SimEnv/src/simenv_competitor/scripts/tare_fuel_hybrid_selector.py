#!/usr/bin/env python3
"""Persistent, topology-aware arbitration between Official TARE and FUEL."""

import collections
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import rospy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Bool, String

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from hierarchical_ros_utils import occupancy_from_message
from planner_interface import AStarPlanner
from structured_topology_core import (
    CorridorDetector, DoorDetector, GeometryParameters, OccupancyMap,
    RoomExpansionDetector, TemporalTopologyTracker)
from tare_fuel_hybrid_core import (
    FrontierIntent, PersistentCoverageMemory, bilateral_corridor_evidence,
    candidate_intent, select_safe_forward_candidate,
    clustered_pose_stall, corridor_entry_ready, intent_key, planar_distance,
    pop_next_eligible,
    rank_fuel_candidates, repeated_goal_region)


class TareFuelHybridSelector:
    FUEL_BOOTSTRAP = "FUEL_BOOTSTRAP"
    TARE_CRUISE = "TARE_CRUISE"
    FUEL_ROOM = "FUEL_ROOM"
    RETURN_MAINLINE = "RETURN_MAINLINE"

    def __init__(self):
        self._output_dir = os.path.abspath(rospy.get_param("~output_dir"))
        self._map_file = os.path.abspath(rospy.get_param(
            "~map_file", os.path.join(self._output_dir, "voxel_map.bt")))
        self._frame = rospy.get_param("~world_frame", "map")
        self._stall_window = float(rospy.get_param("~stall_window", 28.0))
        self._stall_radius = float(rospy.get_param("~stall_region_radius", 0.85))
        self._repeat_radius = float(rospy.get_param("~repeat_region_radius", 1.45))
        self._repeat_count = int(rospy.get_param("~repeat_goal_count", 4))
        self._repeat_span = float(rospy.get_param("~repeat_minimum_span", 12.0))
        self._fuel_goal_tolerance = float(rospy.get_param("~fuel_goal_tolerance", 0.55))
        self._fuel_min_distance = float(rospy.get_param("~fuel_minimum_distance", 0.45))
        self._fuel_max_distance = float(rospy.get_param("~fuel_maximum_distance", 3.2))
        self._fuel_timeout = float(rospy.get_param("~fuel_planner_timeout", 12.0))
        self._fuel_duplicate_radius = float(rospy.get_param("~fuel_duplicate_radius", 1.0))
        self._room_goal_limit = int(rospy.get_param("~room_goal_limit", 3))
        self._blacklist_duration = float(rospy.get_param("~blacklist_duration", 20.0))
        self._fuel_failure_limit = int(rospy.get_param("~fuel_failure_limit", 3))
        self._tare_debounce_distance = float(rospy.get_param(
            "~tare_debounce_distance", 0.45))
        self._tare_debounce_period = float(rospy.get_param(
            "~tare_debounce_period", 2.0))
        self._bootstrap_retry_period = float(rospy.get_param(
            "~bootstrap_retry_period", 8.0))
        self._default_forward_delay = float(rospy.get_param(
            "~default_forward_delay", 3.0))
        self._tare_invalidation_match_distance = float(rospy.get_param(
            "~tare_invalidation_match_distance", 1.0))
        self._corridor_entry_min_progress = float(rospy.get_param(
            "~corridor_entry_min_progress", 2.0))

        os.makedirs(os.path.join(self._output_dir, "logs"), exist_ok=True)
        self._log_path = os.path.join(
            self._output_dir, "logs", "tare_fuel_hybrid.json")
        self._lock = threading.RLock()
        self._latest_tare = None
        self._pose = None
        self._odom_yaw = 0.0
        self._locomotion_ready = False
        self._bootstrap_origin = None
        self._pose_frame = self._frame
        self._recent_pose_samples = collections.deque(maxlen=300)
        self._completed_tare = collections.deque(maxlen=30)
        self._memory = PersistentCoverageMemory(
            endpoint_radius=float(rospy.get_param("~covered_endpoint_radius", 1.0)),
            frontier_radius=float(rospy.get_param("~frontier_repeat_radius", 1.5)),
            direction_tolerance=math.radians(float(rospy.get_param(
                "~frontier_direction_tolerance_deg", 35.0))),
            trajectory_spacing=float(rospy.get_param(
                "~coverage_trajectory_spacing", 0.45)),
            observed_cell_size=float(rospy.get_param(
                "~coverage_cell_size", 0.45)))
        self._events = []
        self._mode = self.FUEL_BOOTSTRAP
        self._fuel_planning = False
        self._fuel_active = None
        self._fuel_queue = []
        self._fuel_goals_completed = 0
        self._fuel_failures = 0
        self._fuel_retry_after = 0.0
        self._blacklist = {}
        self._last_result_sequence = None
        self._have_navigation_activity = False
        self._last_published = None
        self._last_published_stamp = -1e9
        self._default_forward_active = False

        self._grid = None
        self._grid_ready_stamp = None
        self._topology_grid = None
        self._grid_sequence = 0
        self._processed_grid_sequence = -1
        geometry = GeometryParameters(
            corridor_width_min=1.0, corridor_width_max=4.0,
            corridor_forward_depth_min=4.0, corridor_confirmation_frames=3,
            door_confirmation_frames=3)
        self._corridor_detector = CorridorDetector(geometry)
        self._door_detector = DoorDetector(geometry)
        self._room_detector = RoomExpansionDetector(geometry)
        self._topology = TemporalTopologyTracker(geometry)
        self._current_corridor = None
        self._corridor_entry_confirmed = False
        self._bilateral_corridor_confirmed = False
        self._room_expansions = []
        self._room_progress = {}
        self._last_room_trigger = -1e9
        self._mainline_anchor = None
        self._mainline_corridor_id = None
        self._active_room_door_id = None
        self._return_active = False

        self._astar = AStarPlanner(
            clearance=float(rospy.get_param("~astar_clearance", 0.30)),
            maximum_expansions=int(rospy.get_param(
                "~astar_maximum_expansions", 100000)),
            allow_blocked_start=True)

        self._publisher = rospy.Publisher(
            rospy.get_param("~output_goal_topic", "/hybrid_exploration_goal"),
            PointStamped, queue_size=3)
        rospy.Subscriber(
            rospy.get_param("~tare_goal_topic", "/refined_exploration_goal"),
            PointStamped, self._on_tare_goal, queue_size=10)
        rospy.Subscriber(
            rospy.get_param("~map_topic", "/simenv/voxel_floor_projection"),
            OccupancyGrid, self._on_grid, queue_size=2)
        rospy.Subscriber(
            rospy.get_param("~odom_topic", "/tare/state_estimation_at_scan"),
            Odometry, self._on_odom, queue_size=30)
        rospy.Subscriber(
            rospy.get_param("~far_status_topic", "/simenv/tare_far_status"),
            String, self._on_far_status, queue_size=20)
        rospy.Subscriber(
            rospy.get_param("~execution_result_topic",
                            "/simenv/goal_execution_result"),
            String, self._on_execution_result, queue_size=20)
        rospy.Subscriber(
            rospy.get_param("~arbitration_status_topic",
                            "/tare/arbitration_status"),
            String, self._on_arbitration_status, queue_size=20)
        rospy.Subscriber("/locomotion_ready", Bool,
                         self._on_locomotion_ready, queue_size=2)
        rospy.Timer(rospy.Duration(1.0), self._on_timer)
        rospy.Timer(rospy.Duration(10.0), lambda _: self._write_log())
        rospy.on_shutdown(self._write_log)
        self._record("started", map_file=self._map_file,
                     policy="persistent_coverage_online_topology")
        rospy.loginfo(
            "TARE/FUEL hybrid ready in FUEL_BOOTSTRAP; no layout coordinates are used")

    @staticmethod
    def _decode(message):
        try:
            value = json.loads(message.data)
            return value if isinstance(value, dict) else {}
        except (TypeError, ValueError):
            return {}

    @staticmethod
    def _now():
        return rospy.Time.now().to_sec()

    @staticmethod
    def _intent_key(intent):
        return intent_key(intent)

    def _record(self, event, **extra):
        item = {"event": event, "stamp": self._now(), "mode": self._mode}
        item.update(extra)
        self._events.append(item)
        if len(self._events) > 4000:
            self._events = self._events[-3000:]

    def _publish(self, point, source, reason, force=False):
        now = self._now()
        if (not force and self._last_published is not None and
                planar_distance(point, self._last_published) <
                self._tare_debounce_distance and
                now - self._last_published_stamp < self._tare_debounce_period):
            self._record("goal_debounced", source=source, reason=reason,
                         goal=list(point))
            return False
        message = PointStamped()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = self._pose_frame or self._frame
        message.point.x, message.point.y, message.point.z = point
        self._publisher.publish(message)
        self._last_published = tuple(point)
        self._last_published_stamp = now
        if source != "safe_forward":
            self._default_forward_active = False
        self._record("goal_published", source=source, reason=reason,
                     goal=list(point))
        return True

    def _on_grid(self, message):
        try:
            regular = occupancy_from_message(message)
            data = np.asarray(message.data, dtype=np.int16).reshape(
                (message.info.height, message.info.width))
            topology = OccupancyMap(
                data=data, resolution=float(message.info.resolution),
                origin_x=float(message.info.origin.position.x),
                origin_y=float(message.info.origin.position.y))
            stride = max(1, int(round(
                self._memory.observed_cell_size /
                max(float(message.info.resolution), 1e-3))))
            sampled = data[::stride, ::stride]
            free_rows, free_cols = np.nonzero((sampled >= 0) & (sampled < 50))
            observed_free = zip(
                float(message.info.origin.position.x) +
                (free_cols * stride + 0.5) * float(message.info.resolution),
                float(message.info.origin.position.y) +
                (free_rows * stride + 0.5) * float(message.info.resolution))
        except (TypeError, ValueError, IndexError):
            return
        with self._lock:
            self._grid = regular
            self._topology_grid = topology
            if self._grid_ready_stamp is None:
                self._grid_ready_stamp = self._now()
            self._memory.observe_free_cells(observed_free)
            self._grid_sequence += 1

    def _on_odom(self, message):
        point = message.pose.pose.position
        pose = (point.x, point.y, point.z)
        now = self._now()
        with self._lock:
            self._pose = pose
            orientation = message.pose.pose.orientation
            siny = 2.0 * (orientation.w * orientation.z +
                          orientation.x * orientation.y)
            cosy = 1.0 - 2.0 * (orientation.y * orientation.y +
                                orientation.z * orientation.z)
            self._odom_yaw = math.atan2(siny, cosy)
            if self._bootstrap_origin is None:
                self._bootstrap_origin = pose
            self._pose_frame = message.header.frame_id or self._pose_frame
            self._memory.add_pose(pose)
            if (not self._recent_pose_samples or
                    now - self._recent_pose_samples[-1][0] >= 0.5):
                self._recent_pose_samples.append((now, pose))
            self._update_room_crossings_locked()
            if (self._return_active and self._mainline_anchor is not None and
                    planar_distance(pose, self._mainline_anchor) <=
                    self._fuel_goal_tolerance):
                self._finish_mainline_return_locked("anchor_pose_reached")
                return
            if (self._fuel_active is not None and
                    planar_distance(pose, self._fuel_active["intent"].endpoint) <=
                    self._fuel_goal_tolerance):
                self._complete_fuel_goal_locked("pose_reached")

    def _on_locomotion_ready(self, message):
        with self._lock:
            self._locomotion_ready = bool(message.data)
            if not self._locomotion_ready:
                self._default_forward_active = False

    def _on_tare_goal(self, message):
        point = (message.point.x, message.point.y, message.point.z)
        if not all(math.isfinite(value) for value in point):
            return
        trigger = False
        with self._lock:
            self._latest_tare = point
            if self._mode != self.TARE_CRUISE or self._fuel_planning:
                return
            intent = FrontierIntent(point, source="tare", stamp=self._now())
            maximum_gain = max([item.gain for item in self._memory.intents] + [1.0])
            repeat = self._memory.repeat_reason(intent, maximum_gain)
            coverage = self._memory.observed_coverage(point)
            if repeat is None and coverage >= 0.65:
                repeat = "historical_lidar_coverage"
            if repeat:
                self._record("tare_goal_rejected", reason=repeat,
                             goal=list(point), alternate=None,
                             policy="no_unarbitrated_tare_path_bypass")
                trigger = True
            else:
                self._publish(point, "tare", "valid_novel_tare")
        if trigger:
            self._request_fuel("tare_intent_repeated", self.FUEL_ROOM)

    def _on_far_status(self, message):
        data = self._decode(message)
        state = str(data.get("state", ""))
        request = None
        with self._lock:
            if state in ("waiting_for_far", "executing"):
                self._have_navigation_activity = True
            if state in ("planning_failed", "scan_rejected_path"):
                if self._return_active:
                    self._record("mainline_return_failed", reason=state)
                    self._resume_tare_locked("mainline_return_failed")
                    return
                if self._fuel_active is not None:
                    self._reject_active_fuel_locked(state)
                    return
                if self._default_forward_active:
                    self._default_forward_active = False
                    self._record("safe_forward_failed", reason=state)
                request = "{}_tare".format(state)
        if request:
            mode = (self.FUEL_ROOM if self._corridor_entry_confirmed
                    else self.FUEL_BOOTSTRAP)
            self._request_fuel(request, mode)

    def _on_arbitration_status(self, message):
        data = self._decode(message)
        if data.get("event") != "current_goal_invalidated":
            return
        goal = data.get("goal")
        if not isinstance(goal, (list, tuple)) or len(goal) < 2:
            return
        invalidated = (float(goal[0]), float(goal[1]),
                       float(goal[2]) if len(goal) > 2 else 0.0)
        with self._lock:
            if (self._latest_tare is not None and
                    planar_distance(self._latest_tare, invalidated) <=
                    self._tare_invalidation_match_distance):
                stale = self._latest_tare
                self._latest_tare = None
                self._record(
                    "tare_goal_invalidated", goal=list(stale),
                    arbitration_goal=list(invalidated),
                    reason=str(data.get("reason", "failure_limit")),
                    policy="do_not_resume_failed_cached_tare")

    def _on_execution_result(self, message):
        data = self._decode(message)
        sequence = data.get("goal_sequence")
        request = None
        with self._lock:
            if sequence is not None and sequence == self._last_result_sequence:
                return
            self._last_result_sequence = sequence
            self._have_navigation_activity = True
            success = bool(data.get("success", False))
            if self._return_active:
                if (success and self._pose is not None and
                        self._mainline_anchor is not None and
                        planar_distance(self._pose, self._mainline_anchor) <=
                        self._fuel_goal_tolerance):
                    self._finish_mainline_return_locked("anchor_execution_reached")
                else:
                    self._record("stale_execution_result_while_returning",
                                 success=success,
                                 reason=str(data.get("reason", "unknown")))
                return
            if self._default_forward_active:
                self._default_forward_active = False
                self._record("safe_forward_completed" if success else
                             "safe_forward_failed",
                             reason=str(data.get("reason", "execution_result")))
                if not success:
                    request = "safe_forward_execution_failed"
                else:
                    self._fuel_retry_after = self._now()
                if request is None:
                    return
            if self._fuel_active is not None:
                if not success:
                    self._reject_active_fuel_locked(
                        str(data.get("reason", "execution_failed")))
                elif (self._pose is not None and
                      planar_distance(self._pose, self._fuel_active["intent"].endpoint) <=
                      self._fuel_goal_tolerance):
                    self._complete_fuel_goal_locked("execution_reached")
                return
            if self._latest_tare is not None:
                intent = FrontierIntent(
                    tuple(self._latest_tare), source="tare", stamp=self._now(),
                    completed=success)
                self._memory.add_intent(intent)
                self._completed_tare.append((self._now(), self._latest_tare))
            if not success:
                request = "tare_execution_failed"
            elif repeated_goal_region(
                    list(self._completed_tare), self._repeat_radius,
                    self._repeat_count, self._repeat_span):
                request = "tare_goal_region_repeated"
        if request:
            mode = (self.FUEL_ROOM if self._corridor_entry_confirmed
                    else self.FUEL_BOOTSTRAP)
            self._request_fuel(request, mode)

    def _on_timer(self, _event):
        self._update_topology()
        request = None
        mode = None
        safe_forward = None
        with self._lock:
            now = self._now()
            if self._safe_forward_due_locked(now):
                safe_forward = self._select_safe_forward_goal_locked()
                if safe_forward is not None:
                    self._default_forward_active = True
                    self._publish(safe_forward, "safe_forward",
                                  "no_exploration_goal_available", force=True)
                    self._record("safe_forward_started",
                                 goal=list(safe_forward), yaw=self._odom_yaw)
            if (safe_forward is None and
                    self._mode == self.FUEL_BOOTSTRAP and not self._fuel_planning and
                    self._fuel_active is None and
                    not self._default_forward_active and
                    now >= self._fuel_retry_after):
                request, mode = "bootstrap", self.FUEL_BOOTSTRAP
            elif (self._mode == self.TARE_CRUISE and self._latest_tare is not None and
                  self._have_navigation_activity and clustered_pose_stall(
                      list(self._recent_pose_samples), now, self._stall_window,
                      self._stall_radius)):
                request, mode = "pose_region_stall", self.FUEL_ROOM
            elif (self._mode == self.TARE_CRUISE and
                  self._bilateral_corridor_confirmed and
                  self._unvisited_confirmed_doors_locked() and
                  now - self._last_room_trigger >= 10.0):
                request, mode = "unvisited_door_available", self.FUEL_ROOM
                self._last_room_trigger = now
        if request:
            self._request_fuel(request, mode)

    def _safe_forward_due_locked(self, now):
        if (not self._locomotion_ready or self._pose is None or
                self._grid is None or self._grid_ready_stamp is None or
                self._default_forward_active or
                self._fuel_active is not None or self._fuel_planning or
                self._return_active):
            return False
        if now - self._grid_ready_stamp < self._default_forward_delay:
            return False
        if self._mode == self.FUEL_BOOTSTRAP:
            waiting_for_retry = now < self._fuel_retry_after
            no_goal_yet = self._last_published is None
            return ((waiting_for_retry or no_goal_yet) and
                    now - self._last_published_stamp >= self._default_forward_delay)
        return (self._mode == self.TARE_CRUISE and self._latest_tare is None and
                now - self._last_published_stamp >= self._default_forward_delay)

    def _select_safe_forward_goal_locked(self):
        """Choose a short observed-free body-forward goal; never command velocity."""
        point = select_safe_forward_candidate(
            self._pose, self._odom_yaw, self._astar, self._grid,
            coverage_query=lambda candidate: self._memory.observed_coverage(
                candidate, minimum_age_points=10))
        if point is not None:
            return point
        self._record("safe_forward_blocked", pose=list(self._pose),
                     yaw=self._odom_yaw,
                     reason="no_observed_free_forward_candidate")
        return None

    def _update_topology(self):
        with self._lock:
            if (self._topology_grid is None or self._pose is None or
                    self._processed_grid_sequence == self._grid_sequence):
                return
            grid, pose = self._topology_grid, self._pose
            sequence = self._grid_sequence
            previous = (None if self._current_corridor is None else
                        self._current_corridor.principal_direction)
        try:
            observed = self._corridor_detector.detect(grid, pose, previous)
            with self._lock:
                tracked = self._topology.update_corridor(observed)
                if tracked is not None:
                    self._current_corridor = tracked
                self._bilateral_corridor_confirmed = \
                    bilateral_corridor_evidence(tracked)
            door_observations = [] if tracked is None else \
                self._door_detector.detect(grid, tracked)
            room_expansions = []
            if tracked is not None and tracked.confirmed:
                room_expansions = self._room_detector.detect(grid, tracked)
            with self._lock:
                doors = self._topology.update_doors(door_observations)
                if tracked is not None:
                    for door in doors:
                        door.corridor_id = tracked.corridor_id
                self._room_expansions = room_expansions
                for door in doors:
                    if not door.confirmed:
                        continue
                    progress = self._room_progress.setdefault(
                        door.door_id,
                        {"entered": False, "depth": False, "breadth": [],
                         "goals": 0, "expanded": False, "exhausted": False})
                    for expansion in room_expansions:
                        target = expansion.portal_center or expansion.centroid
                        if (planar_distance(target, door.center) <= 3.0 and
                                door.signed_depth(target) >= 0.8):
                            progress["expanded"] = True
                            break
                self._processed_grid_sequence = sequence
                self._record(
                    "topology_updated",
                    corridor=None if tracked is None else tracked.to_dict(),
                    bilateral_corridor=self._bilateral_corridor_confirmed,
                    doors=[item.to_dict() for item in doors],
                    room_expansions=[item.to_dict() for item in room_expansions])
                if corridor_entry_ready(
                        pose, tracked, self._bootstrap_origin,
                        self._corridor_entry_min_progress):
                    self._corridor_entry_confirmed = True
                    if (self._mode == self.FUEL_BOOTSTRAP and
                            self._fuel_active is None and not self._fuel_planning):
                        self._resume_tare_locked("confirmed_corridor_entered")
        except (RuntimeError, ValueError) as error:
            with self._lock:
                self._processed_grid_sequence = sequence
                self._record("topology_update_failed", error=str(error))

    def _unvisited_confirmed_doors_locked(self):
        return [item for item in self._topology.doors
                if (item.confirmed and not item.visited and
                    self._room_progress.get(item.door_id, {}).get("goals", 0) <
                    self._room_goal_limit)]

    def _update_room_crossings_locked(self):
        if self._pose is None:
            return
        for door in self._topology.doors:
            progress = self._room_progress.setdefault(
                door.door_id, {"entered": False, "depth": False,
                               "breadth": [], "goals": 0,
                               "expanded": False, "exhausted": False})
            if (door.signed_depth(self._pose) >= 1.2 and
                    abs(door.lateral_offset(self._pose)) <= 2.0):
                if not progress["entered"]:
                    self._record("room_entered", door_id=door.door_id,
                                 pose=list(self._pose))
                progress["entered"] = True
                progress["depth"] = True
            if progress["goals"] >= self._room_goal_limit:
                progress["exhausted"] = True
            if (progress["entered"] and progress["depth"] and
                    progress["expanded"] and
                    progress["goals"] >= self._room_goal_limit):
                door.visited = True

    def _request_fuel(self, reason, desired_mode):
        with self._lock:
            if (self._fuel_planning or self._fuel_active is not None or
                    self._default_forward_active):
                return
            if self._pose is None or self._grid is None:
                return
            self._mode = desired_mode
            if (desired_mode == self.FUEL_ROOM and
                    self._bilateral_corridor_confirmed and
                    self._unvisited_confirmed_doors_locked() and
                    self._mainline_anchor is None):
                self._mainline_anchor = tuple(self._pose)
                self._mainline_corridor_id = (
                    None if self._current_corridor is None else
                    self._current_corridor.corridor_id)
                self._record(
                    "mainline_anchor_saved", anchor=list(self._mainline_anchor),
                    corridor_id=self._mainline_corridor_id,
                    topology="bilateral_long_corridor")
            self._fuel_planning = True
            pose, frame = self._pose, self._pose_frame
            self._record("fuel_triggered", reason=reason,
                         tare_goal=None if self._latest_tare is None else
                         list(self._latest_tare), pose=list(pose))
        threading.Thread(target=self._run_fuel,
                         args=(reason, pose, frame), daemon=True).start()

    def _history_file(self, directory):
        path = os.path.join(directory, "planner_history.csv")
        with open(path, "w") as output:
            output.write("# x,y,source\n")
            for point in self._memory.trajectory:
                output.write("{:.6f},{:.6f},trajectory\n".format(point[0], point[1]))
            for intent in self._memory.intents:
                source = "goal" if intent.completed else "failed_goal"
                output.write("{:.6f},{:.6f},{}\n".format(
                    intent.endpoint[0], intent.endpoint[1], source))
        return path

    def _run_fuel(self, reason, pose, frame):
        temporary = tempfile.mkdtemp(prefix="tare_fuel_hybrid_")
        ranked, error = [], None
        try:
            if not os.path.isfile(self._map_file):
                raise RuntimeError("voxel map is not available yet")
            with self._lock:
                history = self._history_file(temporary)
            command = [
                "rosrun", "simenv_competitor", "fuel_lite_planner",
                "__name:=tare_fuel_cycle_{}".format(int(time.time() * 1000)),
                "_map_file:=" + self._map_file, "_output_dir:=" + temporary,
                "_frame_id:=" + (frame or self._frame),
                "_use_pose_parameters:=true", "_exit_after_plan:=true",
                "_robot_x:={:.9f}".format(pose[0]),
                "_robot_y:={:.9f}".format(pose[1]),
                "_robot_z:={:.9f}".format(pose[2]),
                "_history_file:=" + history,
                "_execution_duplicate_radius:={:.6f}".format(
                    self._fuel_duplicate_radius),
                "_minimum_cluster_size:=6", "_maximum_clusters:=30",
                "_candidates_per_cluster:=8", "_candidate_min_distance:=0.25",
                "_candidate_max_distance:=2.5", "_safety_clearance:=0.40",
                "_adaptive_clearance:=0.25", "_rolling_goal_lookahead:=2.8",
                "_sensor_range:=4.2", "_enable_room_information_gain:=false",
                "_exploration_mode:=GENERIC",
                "/exploration_goal:=/simenv/fuel_recovery_raw_goal"]
            completed = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, timeout=self._fuel_timeout, check=False)
            candidate_path = os.path.join(temporary, "candidate_viewpoints.json")
            score_path = os.path.join(temporary, "exploration_score.json")
            if not os.path.isfile(candidate_path) or not os.path.isfile(score_path):
                raise RuntimeError("FUEL artifacts missing (exit {}): {}".format(
                    completed.returncode, (completed.stdout or "")[-800:]))
            with open(candidate_path) as source:
                candidates = json.load(source)
            with open(score_path) as source:
                scores = json.load(source).get("scores", [])
            self._annotate_topology(candidates, scores, pose)
            with self._lock:
                evaluated = rank_fuel_candidates(
                    candidates, scores, pose, self._memory,
                    self._fuel_min_distance, self._fuel_max_distance,
                    strict_visited_endpoint=self._mode == self.FUEL_BOOTSTRAP,
                    now=self._now(), include_rejected=True)
                rejection_reasons = collections.Counter(
                    item["repeat_reason"] for item in evaluated
                    if item["repeat_reason"] is not None)
                ranked = [item for item in evaluated
                          if item["repeat_reason"] is None and
                          self._blacklist.get(self._intent_key(item["intent"]), -1e9)
                          <= self._now()]
                if self._mainline_anchor is not None:
                    door_ranked = [item for item in ranked
                                   if item["intent"].door_id is not None]
                    if self._active_room_door_id is not None:
                        matching = [item for item in door_ranked
                                    if item["intent"].door_id ==
                                    self._active_room_door_id]
                        door_ranked = matching
                    ranked = door_ranked
                self._record(
                    "fuel_candidate_batch_ranked", reason=reason,
                    eligible_count=len(ranked),
                    rejected_count=sum(rejection_reasons.values()),
                    rejection_reasons=dict(rejection_reasons),
                    candidates=[{
                        "candidate_id": item["candidate"].get("id"),
                        "intent": item["intent"].to_dict(),
                        "final_score": item["final_score"],
                        "components": item["components"],
                        "repeat_reason": item["repeat_reason"],
                    } for item in ranked])
            diagnostic_dir = os.path.join(
                self._output_dir, "fuel_recovery_cycles",
                "{:013d}".format(int(time.time() * 1000)))
            shutil.copytree(temporary, diagnostic_dir)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            error = str(exc)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

        with self._lock:
            self._fuel_planning = False
            self._fuel_queue = ranked
            if not ranked:
                self._record("fuel_plan_failed", reason=reason,
                             error=error or "no eligible candidates")
                self._fuel_retry_after = self._now() + self._bootstrap_retry_period
                if self._mode == self.FUEL_BOOTSTRAP:
                    self._record("bootstrap_waiting_for_fuel_map",
                                 retry_after=self._fuel_retry_after)
                elif self._mainline_anchor is not None:
                    self._start_mainline_return_locked("fuel_plan_failed")
                else:
                    self._resume_tare_locked("fuel_plan_failed")
                return
            self._publish_next_fuel_locked(reason)

    def _annotate_topology(self, candidates, scores, pose):
        score_by_id = {int(item.get("candidate_id", -1)): item for item in scores}
        ordered = sorted(candidates, key=lambda item: score_by_id.get(
            int(item.get("id", -1)), {}).get("score", 0.0), reverse=True)
        unique_corridors = {}
        with self._lock:
            grid = self._topology_grid
            doors = list(self._topology.doors)
            expansions = list(self._room_expansions)
            progress = dict(self._room_progress)
            mode = self._mode
            bilateral = self._bilateral_corridor_confirmed
        if grid is None:
            return
        corridor_checks = 0
        for candidate in ordered[:40]:
            position = candidate.get("position", [])
            frontier = candidate.get("frontier_center", [])
            if len(position) < 2:
                continue
            topology_bonus, door_id = 0.0, None
            key = (round(float(position[0]), 2), round(float(position[1]), 2))
            corridor = unique_corridors.get(key)
            if key not in unique_corridors and corridor_checks < 12:
                corridor = self._corridor_detector.detect(grid, position)
                unique_corridors[key] = corridor
                corridor_checks += 1
            if corridor is not None and corridor.confidence >= 0.60:
                topology_bonus = max(topology_bonus, 1.0 if mode == self.FUEL_BOOTSTRAP else 0.65)
            for door in doors:
                if not bilateral:
                    break
                if door.visited or len(frontier) < 2:
                    continue
                state = progress.get(door.door_id, {})
                if state.get("goals", 0) >= self._room_goal_limit:
                    continue
                toward_room = door.signed_depth(frontier) > 0.5
                near_portal = planar_distance(position, door.center) <= 4.0
                if not toward_room or not near_portal:
                    continue
                door_id = door.door_id
                depth = door.signed_depth(frontier)
                lateral = door.lateral_offset(frontier)
                if not state.get("depth"):
                    desired = 1.0 if depth >= 1.2 else 0.20
                    view_type = "depth" if depth >= 1.2 else "approach"
                else:
                    lateral = door.lateral_offset(frontier)
                    desired = 1.0 if all(abs(lateral - old) >= 1.0
                                         for old in state["breadth"]) else 0.55
                    view_type = "breadth" if desired >= 1.0 else "breadth_overlap"
                topology_bonus = max(topology_bonus, desired)
                candidate["room_view_type"] = view_type
            for expansion in expansions:
                target = expansion.portal_center or expansion.centroid
                if planar_distance(position, target) <= 4.0:
                    topology_bonus = max(topology_bonus, 0.85)
            candidate["topology_bonus"] = topology_bonus
            candidate["door_id"] = door_id

    def _publish_next_fuel_locked(self, reason):
        now = self._now()
        selected = pop_next_eligible(self._fuel_queue, self._blacklist, now)
        if selected is not None:
            intent = selected["intent"]
            if (self._mainline_anchor is not None and
                    self._active_room_door_id is None and intent.door_id):
                self._active_room_door_id = intent.door_id
                self._record("room_branch_bound", door_id=intent.door_id,
                             anchor=list(self._mainline_anchor))
            self._fuel_active = selected
            self._publish(intent.endpoint, "fuel", reason, force=True)
            self._record(
                "fuel_candidate_selected", intent=intent.to_dict(),
                final_score=selected["final_score"],
                components=selected["components"],
                repeat_reason=selected["repeat_reason"],
                remaining_candidates=len(self._fuel_queue))
            rospy.logwarn(
                "Hybrid %s selected FUEL point (%.2f, %.2f), topology=%.2f",
                self._mode, intent.endpoint[0], intent.endpoint[1],
                selected["components"]["topology"])
            return True
        return False

    def _reject_active_fuel_locked(self, reason):
        if self._fuel_active is None:
            return
        intent = self._fuel_active["intent"]
        self._blacklist[self._intent_key(intent)] = self._now() + self._blacklist_duration
        self._memory.add_intent(FrontierIntent(
            endpoint=intent.endpoint, frontier=intent.frontier, gain=intent.gain,
            source="fuel_failed", stamp=self._now(), completed=False,
            door_id=intent.door_id))
        self._fuel_active = None
        self._fuel_failures += 1
        self._record("fuel_candidate_blacklisted", reason=reason,
                     intent=intent.to_dict(), failures=self._fuel_failures,
                     until=self._blacklist[self._intent_key(intent)])
        if (self._fuel_failures < self._fuel_failure_limit and
                self._publish_next_fuel_locked("candidate_retry_after_" + reason)):
            return
        self._fuel_queue = []
        if self._mode == self.FUEL_BOOTSTRAP:
            self._fuel_retry_after = self._now() + self._bootstrap_retry_period
            self._record("bootstrap_candidate_batch_exhausted",
                         retry_after=self._fuel_retry_after)
        elif self._mainline_anchor is not None:
            self._start_mainline_return_locked("fuel_candidates_rejected")
        else:
            self._resume_tare_locked("fuel_candidates_rejected")

    def _complete_fuel_goal_locked(self, reason):
        if self._fuel_active is None:
            return
        selected = self._fuel_active
        intent = selected["intent"]
        intent.completed = True
        self._memory.add_intent(intent)
        self._fuel_active = None
        self._fuel_goals_completed += 1
        if intent.door_id:
            progress = self._room_progress.setdefault(
                intent.door_id, {"entered": False, "depth": False,
                                 "breadth": [], "goals": 0,
                                 "expanded": False, "exhausted": False})
            progress["goals"] += 1
            door = next((item for item in self._topology.doors
                         if item.door_id == intent.door_id), None)
            if door is not None and intent.frontier is not None:
                depth = door.signed_depth(intent.frontier)
                lateral = door.lateral_offset(intent.frontier)
                view_type = selected["candidate"].get("room_view_type")
                if view_type == "depth" and depth >= 1.2:
                    progress["depth"] = True
                if (view_type == "breadth" and
                        all(abs(lateral - old) >= 1.0
                            for old in progress["breadth"])):
                    progress["breadth"].append(lateral)
                if progress["goals"] >= self._room_goal_limit:
                    progress["exhausted"] = True
                if (progress["entered"] and progress["depth"] and
                        progress["expanded"] and
                        progress["goals"] >= self._room_goal_limit):
                    door.visited = True
        self._record("fuel_goal_completed", reason=reason,
                     intent=intent.to_dict(), burst_count=self._fuel_goals_completed,
                     room_progress=self._room_progress.get(intent.door_id))
        self._fuel_failures = 0
        if self._mode == self.FUEL_BOOTSTRAP:
            if self._corridor_entry_confirmed:
                self._resume_tare_locked("confirmed_corridor_goal_completed")
            else:
                self._fuel_retry_after = self._now()
        elif intent.door_id is None or self._mainline_anchor is None:
            # A generic recovery candidate is not a room viewpoint batch.
            self._resume_tare_locked("generic_fuel_recovery_complete")
        elif self._fuel_goals_completed >= self._room_goal_limit:
            self._start_mainline_return_locked("room_fuel_budget_complete")
        else:
            self._fuel_retry_after = self._now()
            threading.Thread(target=self._request_fuel,
                             args=("room_next_viewpoint", self.FUEL_ROOM),
                             daemon=True).start()

    def _start_mainline_return_locked(self, reason):
        if self._mainline_anchor is None or self._pose is None:
            self._resume_tare_locked(reason)
            return
        if planar_distance(self._pose, self._mainline_anchor) <= self._fuel_goal_tolerance:
            self._finish_mainline_return_locked("already_at_mainline_anchor")
            return
        self._mode = self.RETURN_MAINLINE
        self._fuel_active = None
        self._fuel_queue = []
        self._return_active = True
        self._record("mainline_return_started", reason=reason,
                     anchor=list(self._mainline_anchor),
                     door_id=self._active_room_door_id)
        # This is a transit waypoint through known free space, not a new
        # exploration endpoint.  FAR/A*/SCAN-lite still validate the path.
        self._publish(self._mainline_anchor, "mainline_return", reason, force=True)

    def _finish_mainline_return_locked(self, reason):
        anchor = self._mainline_anchor
        door_id = self._active_room_door_id
        self._return_active = False
        self._record("mainline_anchor_reached", reason=reason,
                     anchor=None if anchor is None else list(anchor),
                     door_id=door_id)
        self._resume_tare_locked("mainline_return_complete")

    def _resume_tare_locked(self, reason):
        self._mode = self.TARE_CRUISE
        self._fuel_active = None
        self._fuel_queue = []
        self._fuel_goals_completed = 0
        self._fuel_failures = 0
        self._return_active = False
        self._mainline_anchor = None
        self._mainline_corridor_id = None
        self._active_room_door_id = None
        self._completed_tare.clear()
        # Persistent coverage memory and trajectory deliberately survive.
        self._recent_pose_samples.clear()
        self._record("tare_resumed", reason=reason,
                     persistent_trajectory_points=len(self._memory.trajectory),
                     persistent_intents=len(self._memory.intents))
        if self._latest_tare is not None:
            intent = FrontierIntent(
                tuple(self._latest_tare), source="tare", stamp=self._now())
            if (self._memory.repeat_reason(intent, 1.0) is None and
                    self._memory.observed_coverage(self._latest_tare) < 0.65):
                self._publish(self._latest_tare, "tare", reason)

    def _write_log(self):
        with self._lock:
            payload = {
                "schema": "simenv_tare_fuel_hybrid_v2",
                "mode": self._mode, "events": list(self._events),
                "coverage_memory": self._memory.to_dict(),
                "blacklist": [
                    {"intent_key": list(key), "until": value}
                    for key, value in self._blacklist.items()],
                "topology": {
                    "corridors": [item.to_dict() for item in self._topology.corridors],
                    "doors": [item.to_dict() for item in self._topology.doors],
                    "room_progress": self._room_progress,
                    "bilateral_corridor_confirmed":
                    self._bilateral_corridor_confirmed,
                    "mainline_anchor": self._mainline_anchor,
                    "active_room_door_id": self._active_room_door_id,
                },
            }
        temporary = self._log_path + ".tmp"
        try:
            with open(temporary, "w") as output:
                json.dump(payload, output, indent=2, sort_keys=True)
            os.replace(temporary, self._log_path)
        except OSError as error:
            rospy.logwarn("Could not write hybrid selector log: %s", error)


def main():
    rospy.init_node("tare_fuel_hybrid_selector")
    TareFuelHybridSelector()
    rospy.spin()


if __name__ == "__main__":
    main()
