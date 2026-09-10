#!/usr/bin/env python3
"""Adaptive, structure-enhanced first-floor exploration manager.

FUEL-lite supplies local candidates.  This node owns the high-level topology,
room queue and completion decision, and rejects every path that is not fully
observed free or crosses a room boundary outside a confirmed doorway.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, String

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from goal_executor_core import quaternion_yaw  # noqa: E402
from structured_topology_core import (  # noqa: E402
    CompletionEvidence,
    CorridorRegion,
    CorridorDetector,
    DoorDetector,
    Doorway,
    GeometryParameters,
    MissionPhase,
    OccupancyMap,
    RoomExpansionDetector,
    RoomExpansionObservation,
    RoomRegion,
    RoomState,
    TemporalTopologyTracker,
    angle_distance,
    align_candidate_door_to_corridor,
    astar_free_path,
    build_room_queue,
    completion_decision,
    corridor_end_geometry_evidence,
    corridor_junction_target,
    depth_geometry_observation,
    distance,
    dot,
    door_entry_evidence,
    doorway_from_room_candidate,
    interpolate_polyline,
    plan_door_route,
    room_complete,
    room_free_region,
    select_room_coverage_viewpoint,
    select_forward_free_bootstrap,
    select_room_viewpoint,
    unit,
    update_room_geometry,
    unified_candidate_score,
    validate_path,
    wrap_angle,
)


Point = Tuple[float, float]


class StructuredExplorationManager:
    def __init__(self) -> None:
        self.output_dir = os.path.abspath(rospy.get_param("~output_dir"))
        self.map_file = os.path.abspath(rospy.get_param("~map_file"))
        self.statistics_file = os.path.abspath(rospy.get_param("~statistics_file"))
        self.odom_topic = rospy.get_param("~odom_topic", "/Odometry")
        self.grid_topic = rospy.get_param(
            "~grid_topic", "/simenv/voxel_floor_projection")
        self.goal_topic = rospy.get_param("~goal_topic", "/exploration_goal")
        self.cancel_goal_topic = rospy.get_param(
            "~cancel_goal_topic", "/simenv/cancel_exploration_goal")
        self.depth_topic = rospy.get_param("~depth_topic", "/real_sense/depth/image_raw")
        self.use_depth = bool(rospy.get_param("~use_depth_door_confirmation", True))
        self.use_rgb = bool(rospy.get_param("~use_rgb_semantic_confirmation", False))
        self.maximum_duration = float(rospy.get_param("~maximum_duration", 1800.0))
        self.startup_timeout = float(rospy.get_param("~startup_timeout", 150.0))
        self.goal_timeout = float(rospy.get_param("~goal_timeout", 180.0))
        self.require_goal_result_identity = bool(rospy.get_param(
            "~require_goal_result_identity", True))
        self.cancel_ack_timeout = float(rospy.get_param(
            "~cancel_ack_timeout", 1.0))
        self.planner_timeout = float(rospy.get_param("~planner_timeout", 40.0))
        self.corridor_speed = float(rospy.get_param("~corridor_speed", 0.50))
        self.room_speed = float(rospy.get_param("~room_speed", 0.30))
        self.door_speed = float(rospy.get_param("~door_speed", 0.25))
        self.generic_goal_tolerance = float(rospy.get_param(
            "~generic_goal_tolerance", 0.30))
        self.door_crossing_goal_tolerance = float(rospy.get_param(
            "~door_crossing_goal_tolerance", 0.12))
        # Corridor-network routes may include ninety-degree junction turns.
        # A point-sized A* path can clip the inside corner even though every
        # centre cell is free, so use a wider planning stencil before the
        # authoritative oriented-footprint validation.  Keep this separate
        # from room-local planning, where a circular approximation would
        # unnecessarily discard useful viewpoints.
        self.corridor_network_clearance = float(rospy.get_param(
            "~corridor_network_clearance", 0.40))
        self.initial_unknown_footprint_grace = float(rospy.get_param(
            "~initial_unknown_footprint_grace", 0.45))
        self.enable_room_information_gain = bool(rospy.get_param(
            "~enable_room_information_gain", True))
        self.room_unknown_decay_lambda = float(rospy.get_param(
            "~room_unknown_decay_lambda", 0.35))
        self.door_detection_evaluation_only = bool(rospy.get_param(
            "~door_detection_evaluation_only", False))
        self.p = GeometryParameters(
            corridor_width_min=float(rospy.get_param("~corridor_width_min", 1.0)),
            corridor_width_max=float(rospy.get_param("~corridor_width_max", 4.0)),
            corridor_forward_depth_min=float(rospy.get_param(
                "~corridor_forward_depth_min", 4.0)),
            corridor_width_variation=float(rospy.get_param(
                "~corridor_width_variation", 0.45)),
            parallel_wall_angle_tolerance_deg=float(rospy.get_param(
                "~parallel_wall_angle_tolerance_deg", 15.0)),
            corridor_confirmation_frames=int(rospy.get_param(
                "~temporal_confirmation_frames", 3)),
            corridor_end_confirmation_frames=int(rospy.get_param(
                "~corridor_end_confirmation_frames", 3)),
            door_width_min=float(rospy.get_param("~door_width_min", 0.8)),
            door_width_max=float(rospy.get_param("~door_width_max", 2.0)),
            door_confirmation_frames=int(rospy.get_param("~door_confirmation_frames", 3)),
            door_position_tolerance=float(rospy.get_param("~door_position_tolerance", 0.35)),
            door_angle_tolerance_deg=float(rospy.get_param("~door_angle_tolerance_deg", 15.0)),
            door_outside_distance=float(rospy.get_param(
                "~door_outside_distance", 0.85)),
            door_inside_distance=float(rospy.get_param(
                "~door_inside_distance", 1.50)),
            door_inside_confirmation_samples=int(rospy.get_param(
                "~door_inside_confirmation_samples", 8)),
            door_entry_max_attempts=int(rospy.get_param(
                "~door_entry_max_attempts", 3)),
            inside_depth_min=float(rospy.get_param("~inside_depth_min", 1.0)),
            room_goal_baseline_min=float(rospy.get_param(
                "~room_goal_baseline_min", 0.8)),
            path_sample_spacing=float(rospy.get_param("~path_sample_spacing", 0.075)),
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self.cycle_dir = os.path.join(self.output_dir, "fuel_cycles")
        os.makedirs(self.cycle_dir, exist_ok=True)
        self.voxel_check_dir = os.path.join(self.output_dir, "voxel_path_checks")
        os.makedirs(self.voxel_check_dir, exist_ok=True)

        self.lock = threading.RLock()
        self.grid: Optional[OccupancyMap] = None
        self.grid_frame = "camera_init"
        self.grid_stamp = 0.0
        self.grid_sequence = 0
        self.last_topology_grid_sequence = -1
        self.pose: Optional[dict] = None
        self.pose_stamp = 0.0
        self.pose_frame = "camera_init"
        self.start_wall = 0.0
        self.phase = MissionPhase.EXPLORE_GENERIC
        self.previous_phase = self.phase
        self.corridor_detector = CorridorDetector(self.p)
        self.door_detector = DoorDetector(self.p)
        self.room_expansion_detector = RoomExpansionDetector(self.p)
        self.topology = TemporalTopologyTracker(self.p)
        self.current_corridor = None
        self.completed_corridors = set()
        self.corridor_edges = set()
        self.rooms: Dict[str, RoomRegion] = {}
        self.room_candidates: Dict[str, RoomExpansionObservation] = {}
        self.room_candidate_sequence = 0
        self.room_queue: List[str] = []
        self.current_door_id: Optional[str] = None
        self.current_room_id: Optional[str] = None
        self.corridor_end_cycles = 0
        self.last_corridor_end_grid_sequence = -1
        self.ending_corridor_id: Optional[str] = None
        self.branch_scan_cycles = 0
        self.last_branch_scan_grid_sequence = -1
        self.no_candidate_cycles = 0
        self.global_no_candidate_cycles = 0
        self.last_topology_signature = (0, 0, 0)
        self.large_reachable_unknown = True
        self.last_visited_growth = time.monotonic()
        self.visited_cells = set()
        self.last_pose_for_crossing = None
        self.door_stable_sides = {}
        self.pending_depth_doors = set()
        self.last_depth_stamp = 0.0
        self.latest_depth = {}
        self.planner_cycle = 0
        self.voxel_check_sequence = 0
        self.bootstrap_attempted = False
        self.result_sequence = 0
        self.goal_command_sequence = 0
        self.last_goal_identity_nanoseconds = 0
        self.command_active = False
        self.room_exit_failures = 0
        self.return_corridor_failures = 0
        self.execution_results: List[dict] = []
        self.termination_reason: Optional[str] = None
        self.exploration_complete = False
        self.finalized = False
        self.last_artifact_write = 0.0
        self.voxel_saved_sequence = 0
        self.final_voxel_snapshot = {
            "requested": False, "acknowledged": False, "reason": "not_finalized"}
        self.actual_trajectory_validation = {
            "performed": False, "safe": False, "reason": "not_finalized"}

        self.corridor_history: List[dict] = []
        self.doorway_history: List[dict] = []
        self.room_history: List[dict] = []
        self.room_candidate_history: List[dict] = []
        self.door_traversal_history: List[dict] = []
        self.path_history: List[dict] = []
        self.semantic_history: List[dict] = []
        self.trajectory: List[dict] = []
        self.goal_history: List[dict] = []
        self.events: List[dict] = []

        self.goal_pub = rospy.Publisher(self.goal_topic, PoseStamped, queue_size=1)
        self.cancel_goal_pub = rospy.Publisher(
            self.cancel_goal_topic, String, queue_size=2)
        self.speed_pub = rospy.Publisher("/simenv/goal_speed_limit", Float32,
                                         queue_size=1, latch=True)
        self.goal_tolerance_pub = rospy.Publisher(
            "/simenv/goal_tolerance", Float32, queue_size=1, latch=True)
        self.state_pub = rospy.Publisher("/simenv/structured_exploration_state", String,
                                         queue_size=2, latch=True)
        self.complete_pub = rospy.Publisher("/simenv/mission_complete", Bool,
                                            queue_size=1, latch=True)
        self.finalize_voxel_pub = rospy.Publisher(
            "/simenv/finalize_voxel_map", Bool, queue_size=1)
        rospy.Subscriber(self.grid_topic, OccupancyGrid, self._on_grid, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self._on_odom, queue_size=30)
        rospy.Subscriber(self.depth_topic, Image, self._on_depth, queue_size=1)
        rospy.Subscriber("/simenv/goal_execution_result", String,
                         self._on_execution_result, queue_size=10)
        rospy.Subscriber("/simenv/voxel_map_saved", Bool,
                         self._on_voxel_map_saved, queue_size=2)
        rospy.on_shutdown(self._shutdown)

    def elapsed(self) -> float:
        return 0.0 if not self.start_wall else time.monotonic() - self.start_wall

    @staticmethod
    def _atomic_json(path: str, payload) -> None:
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)

    @staticmethod
    def _load_json(path: str):
        try:
            with open(path, encoding="utf-8") as stream:
                return json.load(stream)
        except (OSError, ValueError):
            return None

    def _event(self, name: str, **extra) -> None:
        payload = {"t": round(self.elapsed(), 3), "event": name,
                   "phase": self.phase.value}
        payload.update(extra)
        self.events.append(payload)
        self.state_pub.publish(String(data=json.dumps(payload, sort_keys=True)))
        rospy.loginfo("Structured exploration: %s phase=%s", name, self.phase.value)

    def _transition(self, phase: MissionPhase, reason: str) -> None:
        # A safety failure is terminal.  Nested path callers may still return
        # False and attempt their normal recovery transition; never allow that
        # to revive a mission after _fail_mission().
        if self.phase == MissionPhase.FINISHED and phase != MissionPhase.FINISHED:
            return
        if self.phase == phase:
            return
        old = self.phase
        self.previous_phase = old
        self.phase = phase
        if phase == MissionPhase.GLOBAL_RECOVERY:
            self.global_no_candidate_cycles = 0
        self._event("state_transition", previous=old.value, reason=reason)

    def _fail_mission(self, reason: str) -> None:
        self.exploration_complete = False
        self.termination_reason = reason
        self._transition(MissionPhase.FINISHED, reason)

    def _on_grid(self, message: OccupancyGrid) -> None:
        expected = int(message.info.width) * int(message.info.height)
        if expected <= 0 or len(message.data) != expected:
            return
        array = np.asarray(message.data, dtype=np.int16).reshape(
            (int(message.info.height), int(message.info.width)))
        grid = OccupancyMap(array, float(message.info.resolution),
                            float(message.info.origin.position.x),
                            float(message.info.origin.position.y))
        with self.lock:
            self.grid = grid
            self.grid_frame = message.header.frame_id or self.grid_frame
            self.grid_stamp = time.monotonic()
            self.grid_sequence += 1

    def _on_odom(self, message: Odometry) -> None:
        p, q = message.pose.pose.position, message.pose.pose.orientation
        pose = {"x": float(p.x), "y": float(p.y), "z": float(p.z),
                "yaw": quaternion_yaw(q.x, q.y, q.z, q.w)}
        if not all(math.isfinite(value) for value in pose.values()):
            return
        now = time.monotonic()
        with self.lock:
            previous = self.pose
            self.pose = pose
            self.pose_stamp = now
            self.pose_frame = message.header.frame_id or self.pose_frame
            if not self.trajectory or now - self.trajectory[-1]["wall"] >= 0.10:
                sample = {"t": round(self.elapsed(), 3), **pose, "wall": now}
                self.trajectory.append(sample)
                cell = (int(round(pose["x"] / 0.25)), int(round(pose["y"] / 0.25)))
                if cell not in self.visited_cells:
                    self.visited_cells.add(cell)
                    self.last_visited_growth = now
            if previous is not None:
                self._record_door_crossings((previous["x"], previous["y"]),
                                            (pose["x"], pose["y"]))
                if self.command_active and self.grid is not None and distance(
                        (previous["x"], previous["y"]), (pose["x"], pose["y"])) > 0.02:
                    actual = validate_path(
                        self.grid,
                        [(previous["x"], previous["y"]), (pose["x"], pose["y"])],
                        spacing=self.p.path_sample_spacing,
                    )
                    if not actual.valid:
                        self.path_history.append({
                            "t": round(self.elapsed(), 3),
                            "phase": self.phase.value,
                            "source": "actual_trajectory_segment",
                            "path": [(previous["x"], previous["y"]),
                                     (pose["x"], pose["y"])],
                            "execution_authorized": True,
                            "actual_execution": True,
                            **actual.to_dict(),
                        })

    def _record_door_crossings(self, previous: Point, current: Point) -> None:
        candidates = []
        for door in self.topology.doors:
            if not (door.confirmed and door.traversable and
                    door.source_room_candidate_id):
                continue
            current_depth = door.signed_depth(current)
            current_side = 1 if current_depth >= 0.08 else (
                -1 if current_depth <= -0.08 else 0)
            if current_side == 0:
                continue
            stable = self.door_stable_sides.get(door.door_id)
            if stable is None:
                self.door_stable_sides[door.door_id] = (current_side, current)
                continue
            previous_side, stable_point = stable
            self.door_stable_sides[door.door_id] = (current_side, current)
            if current_side == previous_side:
                continue
            a, b = door.signed_depth(stable_point), current_depth
            denominator = abs(a) + abs(b)
            alpha = 0.5 if denominator < 1e-8 else abs(a) / denominator
            crossing = (
                stable_point[0] + alpha * (current[0] - stable_point[0]),
                stable_point[1] + alpha * (current[1] - stable_point[1]))
            lateral = abs(door.lateral_offset(crossing))
            if lateral > 0.5 * door.width + 0.45:
                continue
            legal = lateral <= max(0.0, 0.5 * door.width - 0.05)
            candidates.append((lateral, door, crossing,
                               legal, "enter" if current_side > previous_side else "exit"))
        if not candidates:
            return
        legal_candidates = [item for item in candidates if item[3]]
        _, door, crossing, legal, direction = min(
            legal_candidates or candidates, key=lambda item: item[0])
        item = {
            "t": round(self.elapsed(), 3), "door_id": door.door_id,
            "point": crossing, "legal": legal, "direction": direction,
            "source": "actual_trajectory",
        }
        self.door_traversal_history.append(item)

    @staticmethod
    def _decode_depth(message: Image) -> Optional[np.ndarray]:
        try:
            if message.encoding in ("32FC1", "32FC"):
                return np.frombuffer(message.data, dtype=np.float32).reshape(
                    message.height, message.step // 4)[:, :message.width].copy()
            if message.encoding in ("16UC1", "mono16"):
                raw = np.frombuffer(message.data, dtype=np.uint16).reshape(
                    message.height, message.step // 2)[:, :message.width]
                return raw.astype(np.float32) * 0.001
        except (ValueError, TypeError):
            return None
        return None

    def _on_depth(self, message: Image) -> None:
        depth = self._decode_depth(message)
        if depth is None:
            return
        observation = depth_geometry_observation(depth)
        now = message.header.stamp.to_sec() if message.header.stamp else rospy.Time.now().to_sec()
        with self.lock:
            pose = dict(self.pose) if self.pose else None
            corridor_confidence = (self.current_corridor.confidence
                                   if self.current_corridor else 0.0)
            visible_doors = []
            projected_depth = []
            projected_rooms = []
            if pose is not None:
                for door in self.topology.doors:
                    bearing = math.atan2(door.center[1] - pose["y"],
                                         door.center[0] - pose["x"])
                    bearing_offset = wrap_angle(bearing - pose["yaw"])
                    range_m = distance(door.center, (pose["x"], pose["y"]))
                    if range_m <= 8.0 and abs(bearing_offset) <= math.radians(28.0):
                        visible_doors.append(door)
                        center_fraction = 0.5 + bearing_offset / math.radians(60.0)
                        angular_width = 2.0 * math.atan2(0.5 * door.width, max(range_m, 0.2))
                        roi_fraction = float(np.clip(
                            angular_width / math.radians(60.0) * 1.4, 0.12, 0.38))
                        door_depth = depth_geometry_observation(
                            depth, center_fraction=center_fraction,
                            roi_fraction=roi_fraction)
                        door_depth["door_id"] = door.door_id
                        door_depth["bearing_offset_rad"] = bearing_offset
                        door_depth["range_to_candidate_m"] = range_m
                        projected_depth.append(door_depth)
                        if door_depth.get("door_confidence_depth", 0.0) >= 0.65:
                            self.pending_depth_doors.add(door.door_id)
                for candidate in self.room_candidates.values():
                    target = candidate.portal_center or candidate.centroid
                    bearing = math.atan2(target[1] - pose["y"],
                                         target[0] - pose["x"])
                    bearing_offset = wrap_angle(bearing - pose["yaw"])
                    range_m = distance(target, (pose["x"], pose["y"]))
                    if range_m > 8.0 or abs(bearing_offset) > math.radians(28.0):
                        continue
                    center_fraction = 0.5 + bearing_offset / math.radians(60.0)
                    angular_width = 2.0 * math.atan2(
                        0.5 * max(candidate.portal_width, 1.0), max(range_m, 0.2))
                    roi_fraction = float(np.clip(
                        angular_width / math.radians(60.0) * 1.5, 0.14, 0.42))
                    room_depth = depth_geometry_observation(
                        depth, center_fraction=center_fraction,
                        roi_fraction=roi_fraction)
                    visible_depth = float(room_depth.get("visible_free_depth") or 0.0)
                    range_support = min(1.0, visible_depth / max(range_m, 0.5))
                    opening_support = float(room_depth.get(
                        "door_confidence_depth", 0.0))
                    depth_confidence = min(
                        1.0, 0.55 * opening_support + 0.45 * range_support)
                    candidate.depth_confidence = max(
                        candidate.depth_confidence, depth_confidence)
                    if depth_confidence >= 0.58:
                        candidate.depth_confirmations += 1
                    self._fuse_room_candidate(candidate)
                    room_depth.update({
                        "room_candidate_id": candidate.candidate_id,
                        "bearing_offset_rad": bearing_offset,
                        "range_to_candidate_m": range_m,
                        "room_confidence_depth": depth_confidence,
                        "fused_room_confidence": candidate.fused_confidence,
                    })
                    projected_rooms.append(room_depth)
            if projected_depth:
                observation = max(projected_depth,
                                  key=lambda item: item.get("door_confidence_depth", 0.0))
            payload = {
                "timestamp": now,
                "corridor_confidence_geometry": corridor_confidence,
                "door_confidence_geometry": max(
                    (door.confidence for door in visible_doors), default=0.0),
                "room_expansion_confidence": max(
                    [room.expansion_confidence for room in self.rooms.values()]
                    + [candidate.fused_confidence
                       for candidate in self.room_candidates.values()], default=0.0),
                "visible_door_ids": [door.door_id for door in visible_doors],
                "door_depth_observations": projected_depth,
                "room_depth_observations": projected_rooms,
                "confirmed_room_candidate_ids": [
                    candidate.candidate_id
                    for candidate in self.room_candidates.values()
                    if candidate.confirmed],
                "use_depth_door_confirmation": self.use_depth,
                "use_rgb_semantic_confirmation": self.use_rgb,
                **observation,
            }
            self.latest_depth = payload
            self.semantic_history.append(payload)
            self.last_depth_stamp = time.monotonic()

    def _on_execution_result(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (ValueError, TypeError):
            return
        with self.lock:
            self.result_sequence += 1
            payload["sequence"] = self.result_sequence
            self.execution_results.append(payload)
            if len(self.execution_results) > 1000:
                self.execution_results = self.execution_results[-1000:]

    @staticmethod
    def _goal_result_matches(candidate: dict, expected_sequence: int,
                             expected_stamp: float, point: Point,
                             require_identity: bool = True) -> bool:
        """Correlate an asynchronous executor result to exactly one command."""
        try:
            returned_goal = candidate.get("goal") or {}
            coordinates_match = (
                abs(float(returned_goal.get("x", math.inf)) - point[0]) <= 0.05 and
                abs(float(returned_goal.get("y", math.inf)) - point[1]) <= 0.05)
            # rospy overwrites Header.seq when publishing.  A unique
            # manager-generated wall-clock stamp plus the already-required
            # target coordinates is therefore the transport-stable identity;
            # goal_sequence remains useful only as audit metadata.
            has_identity = "goal_stamp" in candidate
            identity_match = (
                has_identity and
                abs(float(candidate.get("goal_stamp")) -
                    float(expected_stamp)) <= 1e-6)
        except (TypeError, ValueError, OverflowError):
            return False
        if not coordinates_match:
            return False
        if require_identity:
            return identity_match
        return identity_match or not has_identity

    def _cancel_goal_and_wait(self, expected_sequence: int,
                              expected_stamp: float, point: Point,
                              reason: str, result_baseline: int) -> dict:
        """Stop one exact executor goal and require a bounded result ack."""
        payload = {
            "goal_sequence": int(expected_sequence),
            "goal_stamp": float(expected_stamp),
            "goal_x": float(point[0]),
            "goal_y": float(point[1]),
            "reason": str(reason),
        }
        deadline = time.monotonic() + max(0.1, self.cancel_ack_timeout)
        next_publish = 0.0
        acknowledgement = None
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_publish:
                self.cancel_goal_pub.publish(
                    String(data=json.dumps(payload, sort_keys=True)))
                next_publish = now + 0.25
            with self.lock:
                candidates = [dict(item) for item in self.execution_results
                              if int(item.get("sequence", 0)) > result_baseline]
            acknowledgement = next((item for item in candidates
                                    if self._goal_result_matches(
                                        item, expected_sequence, expected_stamp,
                                        point, True)), None)
            if acknowledgement is not None:
                break
            time.sleep(0.02)
        result = {
            "requested": True,
            "acknowledged": acknowledgement is not None,
            "goal_sequence": int(expected_sequence),
            "goal_stamp": float(expected_stamp),
            "reason": str(reason),
        }
        if acknowledgement is not None:
            result["executor_result"] = acknowledgement
        self._event("goal_cancel_ack" if result["acknowledged"] else
                    "goal_cancel_unacknowledged", **result)
        return result

    def _on_voxel_map_saved(self, message: Bool) -> None:
        if not message.data:
            return
        with self.lock:
            self.voxel_saved_sequence += 1

    def _request_final_voxel_snapshot(self, timeout: float = 15.0) -> bool:
        """Request an atomic .bt/statistics snapshot before final safety audit."""
        with self.lock:
            baseline = self.voxel_saved_sequence
            baseline_grid_stamp = self.grid_stamp
        requested_at = time.monotonic()
        self.final_voxel_snapshot = {
            "requested": True, "acknowledged": False,
            "request_t": round(self.elapsed(), 3), "timeout_seconds": timeout,
        }
        self.finalize_voxel_pub.publish(Bool(data=True))
        deadline = requested_at + max(0.1, timeout)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            with self.lock:
                acknowledged = self.voxel_saved_sequence > baseline
                projection_refreshed = self.grid_stamp > baseline_grid_stamp
            if acknowledged and projection_refreshed:
                map_exists = os.path.isfile(self.map_file)
                statistics_exists = os.path.isfile(self.statistics_file)
                self.final_voxel_snapshot.update({
                    "acknowledged": True,
                    "projection_refreshed": True,
                    "ack_latency_seconds": round(time.monotonic() - requested_at, 3),
                    "map_file_exists": map_exists,
                    "statistics_file_exists": statistics_exists,
                    "complete": map_exists and statistics_exists,
                })
                return map_exists and statistics_exists
            time.sleep(0.05)
        self.final_voxel_snapshot.update({
            "reason": "voxel_snapshot_ack_timeout",
            "projection_refreshed": self.grid_stamp > baseline_grid_stamp,
            "complete": False,
            "map_file_exists": os.path.isfile(self.map_file),
            "statistics_file_exists": os.path.isfile(self.statistics_file),
        })
        return False

    def _wait_ready(self) -> bool:
        deadline = time.monotonic() + self.startup_timeout
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            with self.lock:
                ready = (self.grid is not None and self.pose is not None and
                         time.monotonic() - self.grid_stamp < 2.0 and
                         time.monotonic() - self.pose_stamp < 0.6 and
                         (not self.use_depth or
                          time.monotonic() - self.last_depth_stamp < 2.0))
                frames_match = self.grid_frame == self.pose_frame
            if ready and frames_match and os.path.isfile(self.map_file):
                return True
            if ready and not frames_match:
                rospy.logwarn_throttle(5.0, "Grid/odom frame mismatch: %s != %s",
                                       self.grid_frame, self.pose_frame)
            time.sleep(0.25)
        return False

    @staticmethod
    def _room_candidate_matches(
        tracked: RoomExpansionObservation,
        observed: RoomExpansionObservation,
    ) -> bool:
        if tracked.corridor_id != observed.corridor_id or tracked.side != observed.side:
            return False
        overlap = min(tracked.longitudinal_max, observed.longitudinal_max) - max(
            tracked.longitudinal_min, observed.longitudinal_min)
        gap = max(0.0, max(tracked.longitudinal_min, observed.longitudinal_min)
                  - min(tracked.longitudinal_max, observed.longitudinal_max))
        return overlap >= -0.5 or gap <= 2.5

    @staticmethod
    def _fuse_room_candidate(candidate: RoomExpansionObservation) -> None:
        temporal = min(1.0, candidate.observations / 3.0)
        depth = min(1.0, candidate.depth_confirmations / 3.0)
        candidate.fused_confidence = min(
            1.0,
            0.58 * candidate.geometry_confidence
            + 0.14 * candidate.portal_confidence
            + 0.14 * temporal
            + 0.14 * max(candidate.depth_confidence, depth),
        )
        # Depth is positive corroboration when the candidate is visible, not a
        # mandatory gate: a side-facing room can remain outside the camera FOV
        # while LiDAR/OctoMap supplies strong repeated geometric evidence.
        candidate.confirmed = (
            candidate.observations >= 2
            and (candidate.geometry_confidence >= 0.68 or
                 (candidate.geometry_confidence >= 0.60
                  and candidate.depth_confirmations >= 2))
            and (candidate.depth_confirmations >= 2 or candidate.observations >= 3)
            and candidate.fused_confidence >= 0.68
        )

    def _update_room_candidates(
        self, observations: Sequence[RoomExpansionObservation]
    ) -> List[RoomExpansionObservation]:
        updated = []
        for observed in observations:
            match = next((candidate for candidate in self.room_candidates.values()
                          if self._room_candidate_matches(candidate, observed)), None)
            if match is None:
                self.room_candidate_sequence += 1
                observed.candidate_id = "room_candidate_{:02d}".format(
                    self.room_candidate_sequence)
                match = observed
                self.room_candidates[match.candidate_id] = match
            else:
                weight = 1.0 / float(match.observations + 1)
                match.centroid = (
                    (1.0 - weight) * match.centroid[0] + weight * observed.centroid[0],
                    (1.0 - weight) * match.centroid[1] + weight * observed.centroid[1],
                )
                match.core_area = max(match.core_area, observed.core_area)
                match.longitudinal_min = min(
                    match.longitudinal_min, observed.longitudinal_min)
                match.longitudinal_max = max(
                    match.longitudinal_max, observed.longitudinal_max)
                match.lateral_expansion_depth = max(
                    match.lateral_expansion_depth, observed.lateral_expansion_depth)
                match.maximum_clearance = max(
                    match.maximum_clearance, observed.maximum_clearance)
                match.expansion_ratio = max(match.expansion_ratio,
                                            observed.expansion_ratio)
                match.geometry_confidence = max(
                    match.geometry_confidence, observed.geometry_confidence)
                if observed.portal_confidence >= match.portal_confidence:
                    match.portal_center = observed.portal_center
                    match.portal_width = observed.portal_width
                    match.portal_normal_direction = observed.portal_normal_direction
                    match.portal_confidence = observed.portal_confidence
                match.observations += 1
            self._fuse_room_candidate(match)
            updated.append(match)
        return updated

    def _promote_room_candidates_to_doors(
        self, grid: OccupancyMap, grid_sequence: int
    ) -> List[Doorway]:
        """Validate confirmed room portals and expose only safe door channels."""
        promoted = []
        for candidate in self.room_candidates.values():
            if candidate.door_id:
                tracked = self._door(candidate.door_id)
                if tracked is not None and tracked.confirmed and tracked.traversable:
                    promoted.append(tracked)
                    continue
            previous = candidate.channel_validation or {}
            last_sequence = int(previous.get("grid_sequence", -1000000))
            # A rejected channel may become observable as mapping advances, but
            # invoking the external 3-D validator on every projection update is
            # unnecessary.  Retry after three distinct maps.
            if grid_sequence - last_sequence < 3:
                continue
            door = doorway_from_room_candidate(candidate, self.p)
            if door is None:
                continue
            raw_portal_center = door.center
            corridor = self._corridor(candidate.corridor_id)
            if corridor is not None:
                door = align_candidate_door_to_corridor(
                    door, candidate, corridor, self.p)
            corridor_aligned_center = door.center
            door, two_d, fit = self._fit_executable_room_channel(grid, door, candidate)
            channel = [door.corridor_side, door.center, door.interior_side]
            voxel = ({"safe": False, "reason": "two_dimensional_channel_invalid",
                      "skipped": True}
                     if not two_d.valid else
                     self._voxel_validate_path(
                         channel, "room_candidate_channel_" + candidate.candidate_id,
                         required_clearance=0.27))
            safe = bool(two_d.valid and voxel.get("safe"))
            candidate.channel_validation = {
                "grid_sequence": int(grid_sequence),
                "channel": channel,
                "raw_portal_center": raw_portal_center,
                "corridor_aligned_center": corridor_aligned_center,
                "channel_fit": fit,
                "required_clearance_m": 0.27,
                "two_dimensional": two_d.to_dict(),
                "voxel": voxel,
                "safe": safe,
            }
            self._event(
                "room_candidate_channel_validated" if safe else
                "room_candidate_channel_rejected",
                candidate_id=candidate.candidate_id,
                two_dimensional_reason=two_d.reason,
                voxel_reason=voxel.get("reason"), safe=safe)
            if not safe:
                continue
            tracked = self.topology.update_doors([door])[0]
            self._apply_door_geometry(tracked, door)
            tracked.confirmed = True
            tracked.traversable = True
            tracked.corridor_id = candidate.corridor_id
            tracked.adjacent_region_id = "room_" + candidate.candidate_id
            tracked.source_room_candidate_id = candidate.candidate_id
            tracked.last_validated_grid_sequence = int(grid_sequence)
            tracked.channel_validation = dict(candidate.channel_validation)
            tracked.estimated_unknown_size = max(
                tracked.estimated_unknown_size, candidate.core_area)
            candidate.door_id = tracked.door_id
            if tracked.door_id not in self.room_queue and not tracked.visited:
                self.room_queue.append(tracked.door_id)
            promoted.append(tracked)
            self._event("room_candidate_promoted_to_door",
                        candidate_id=candidate.candidate_id,
                        door_id=tracked.door_id,
                        room_id=tracked.adjacent_region_id,
                        channel=channel)
        return promoted

    def _fit_executable_room_channel(
        self, grid: OccupancyMap, seed: Doorway,
        candidate: RoomExpansionObservation,
    ) -> Tuple[Doorway, object, dict]:
        """Snap noisy wall-gap geometry to an observed free, framed channel."""
        seed_channel = [seed.corridor_side, seed.center, seed.interior_side]
        seed_validity = validate_path(
            grid, seed_channel, [seed], required_door_id=seed.door_id,
            spacing=self.p.path_sample_spacing, check_footprint=True)
        if seed_validity.valid:
            return seed, seed_validity, {
                "adjusted": False, "tangent_offset_m": 0.0,
                "normal_offset_m": 0.0, "frame_evidence": True,
            }

        nx, ny = seed.normal
        tx, ty = seed.tangent
        step = max(grid.resolution, 0.15)
        tangent_limit = max(3.0, min(5.0, 0.5 * (
            candidate.longitudinal_max - candidate.longitudinal_min) + 1.0))
        normal_limit = max(0.9, min(1.8, 0.6 * seed.width + 0.6))
        tangent_offsets = np.arange(-tangent_limit, tangent_limit + 0.5 * step, step)
        normal_offsets = np.arange(-normal_limit, normal_limit + 0.5 * step, step)
        offsets = sorted(
            ((float(t), float(n)) for t in tangent_offsets for n in normal_offsets),
            key=lambda item: (abs(item[0]) + 1.5 * abs(item[1]), abs(item[0])))
        checked = 0
        framed = 0
        for tangent_offset, normal_offset in offsets:
            if abs(tangent_offset) < 1e-9 and abs(normal_offset) < 1e-9:
                continue
            dx = tangent_offset * tx + normal_offset * nx
            dy = tangent_offset * ty + normal_offset * ny
            center = (seed.center[0] + dx, seed.center[1] + dy)
            # The expansion centroid must remain on the proposed room side;
            # this prevents snapping to an unrelated opposite-facing gap.
            if dot((candidate.centroid[0] - center[0],
                    candidate.centroid[1] - center[1]), (nx, ny)) <= 0.5:
                continue
            shifted = Doorway(
                **{**seed.to_dict(),
                   "center": center,
                   "left_frame_point": (seed.left_frame_point[0] + dx,
                                        seed.left_frame_point[1] + dy),
                   "right_frame_point": (seed.right_frame_point[0] + dx,
                                         seed.right_frame_point[1] + dy),
                   "corridor_side": (center[0] - self.p.door_outside_distance * nx,
                                     center[1] - self.p.door_outside_distance * ny),
                   "interior_side": (center[0] + self.p.door_inside_distance * nx,
                                     center[1] + self.p.door_inside_distance * ny)})
            # A free strip in a hall is not a doorway.  Both ends of the
            # measured aperture must have occupied wall/frame support.
            frame_radius = max(0.22, 1.25 * grid.resolution)
            if not (grid.has_occupied_near(shifted.left_frame_point, frame_radius)
                    and grid.has_occupied_near(shifted.right_frame_point, frame_radius)):
                continue
            framed += 1
            channel = [shifted.corridor_side, shifted.center, shifted.interior_side]
            checked += 1
            validity = validate_path(
                grid, channel, [shifted], required_door_id=shifted.door_id,
                spacing=self.p.path_sample_spacing, check_footprint=True)
            if validity.valid:
                return shifted, validity, {
                    "adjusted": True,
                    "tangent_offset_m": tangent_offset,
                    "normal_offset_m": normal_offset,
                    "frame_evidence": True,
                    "framed_hypotheses": framed,
                    "validated_hypotheses": checked,
                }
        return seed, seed_validity, {
            "adjusted": False, "reason": "no_framed_free_channel_found",
            "framed_hypotheses": framed, "validated_hypotheses": checked,
        }

    @staticmethod
    def _apply_door_geometry(target: Doorway, source: Doorway) -> None:
        """Copy one validated door plane without replacing tracker state."""
        for attribute in (
                "center", "normal_direction", "width", "left_frame_point",
                "right_frame_point", "corridor_side", "interior_side"):
            setattr(target, attribute, getattr(source, attribute))

    def _candidate_for_door(
        self, door: Doorway
    ) -> Optional[RoomExpansionObservation]:
        candidate_id = door.source_room_candidate_id
        if candidate_id:
            return self.room_candidates.get(candidate_id)
        return next((candidate for candidate in self.room_candidates.values()
                     if candidate.door_id == door.door_id), None)

    def _refresh_candidate_door_channel(self, door: Doorway, stage: str) -> dict:
        """Refit and revalidate a candidate door against the newest live map."""
        if door.geometry_locked:
            return {"safe": True, "locked": True, "geometry_changed": False,
                    "grid_sequence": door.last_validated_grid_sequence}
        candidate = self._candidate_for_door(door)
        if candidate is None:
            return {"safe": False,
                    "reason": "raw_door_candidate_is_not_executable",
                    "geometry_changed": False}
        with self.lock:
            grid = self.grid
            grid_sequence = self.grid_sequence
        if grid is None:
            return {"safe": False, "reason": "latest_grid_missing",
                    "geometry_changed": False}
        if (door.last_validated_grid_sequence == grid_sequence and
                door.channel_validation.get("safe")):
            return {"safe": True, "locked": False, "cached": True,
                    "geometry_changed": False,
                    "grid_sequence": grid_sequence}
        old_center = door.center
        seed = Doorway(**door.to_dict())
        corridor = self._corridor(candidate.corridor_id)
        if corridor is not None:
            seed = align_candidate_door_to_corridor(
                seed, candidate, corridor, self.p)
        fitted, two_d, fit = self._fit_executable_room_channel(
            grid, seed, candidate)
        channel = [fitted.corridor_side, fitted.center, fitted.interior_side]
        voxel = ({"safe": False, "reason": "two_dimensional_channel_invalid",
                  "skipped": True}
                 if not two_d.valid else
                 self._voxel_validate_path(
                     channel, "{}_latest_{}".format(stage, door.door_id),
                     required_clearance=0.27))
        safe = bool(two_d.valid and voxel.get("safe"))
        validation = {
            "stage": stage,
            "grid_sequence": int(grid_sequence),
            "channel": channel,
            "channel_fit": fit,
            "required_clearance_m": 0.27,
            "two_dimensional": two_d.to_dict(),
            "voxel": voxel,
            "safe": safe,
        }
        candidate.channel_validation = dict(validation)
        door.channel_validation = dict(validation)
        door.last_validated_grid_sequence = int(grid_sequence)
        geometry_changed = distance(old_center, fitted.center) > 0.10
        if safe:
            self._apply_door_geometry(door, fitted)
            door.traversable = True
        else:
            # Do not keep a stale authorization alive.  Topology updates may
            # refit and promote this same remembered candidate on a later map.
            door.traversable = False
        self._event(
            "door_channel_latest_map_validated" if safe else
            "door_channel_latest_map_rejected",
            door_id=door.door_id, candidate_id=candidate.candidate_id,
            stage=stage, grid_sequence=grid_sequence,
            geometry_changed=geometry_changed,
            old_center=old_center, fitted_center=fitted.center,
            two_dimensional_reason=two_d.reason,
            voxel_reason=voxel.get("reason"))
        return {"safe": safe, "locked": False,
                "geometry_changed": geometry_changed,
                "grid_sequence": grid_sequence,
                "two_dimensional_reason": two_d.reason,
                "voxel_reason": voxel.get("reason")}

    def _room_reference_corridor(self, pose: dict) -> Optional[CorridorRegion]:
        """Select and odometry-stabilize a corridor frame for room geometry.

        Room detection must not disappear merely because the local wall
        detector misses one update.  The selected axis is still map-derived;
        odometry only scores alignment with the actually traversed motion and
        extends the longitudinal support to the current robot station.
        """
        choices = list(self.topology.corridors)
        if not choices:
            return None
        if len(self.trajectory) >= 20:
            points = np.asarray([[item["x"], item["y"]]
                                 for item in self.trajectory[-2000:]], dtype=float)
            centered = points - np.mean(points, axis=0)
            covariance = np.cov(centered.T)
            values, vectors = np.linalg.eigh(covariance)
            principal = vectors[:, int(np.argmax(values))]
            motion_angle = math.atan2(principal[1], principal[0])
            if float(np.max(values)) < 0.25:
                motion_angle = None
        else:
            motion_angle = None

        def score(corridor):
            length = distance(corridor.centerline[0], corridor.centerline[-1])
            alignment = (0.0 if motion_angle is None else
                         angle_distance(corridor.principal_direction, motion_angle))
            return (1.0 if corridor.confirmed else 0.0) + corridor.confidence + min(
                length / 10.0, 1.0) - 2.5 * alignment

        source = max(choices, key=score)
        axis = unit(source.principal_direction)
        origin = source.centerline[0]
        endpoints = list(source.centerline)
        projections = [dot((point[0] - origin[0], point[1] - origin[1]), axis)
                       for point in endpoints]
        robot_station = dot((pose["x"] - origin[0], pose["y"] - origin[1]), axis)
        # The robot's traversed station is authoritative free-space evidence.
        # A short look-ahead lets already observed side rooms form a complete
        # core without inventing occupancy or a doorway.
        maximum = max(max(projections), robot_station + 2.0 * source.estimated_width)
        minimum = min(min(projections), robot_station)
        return CorridorRegion(
            corridor_id=source.corridor_id,
            centerline=[(origin[0] + minimum * axis[0], origin[1] + minimum * axis[1]),
                        (origin[0] + maximum * axis[0], origin[1] + maximum * axis[1])],
            principal_direction=source.principal_direction,
            estimated_width=source.estimated_width,
            forward_extent=source.forward_extent,
            left_wall=list(source.left_wall), right_wall=list(source.right_wall),
            confidence=source.confidence, confirmed=source.confirmed,
            observations=source.observations,
        )

    def _update_topology(self) -> None:
        with self.lock:
            grid = self.grid
            pose = dict(self.pose) if self.pose else None
            grid_sequence = self.grid_sequence
            # Temporal confirmation means distinct Voxel projection updates,
            # not repeated manager iterations over one immutable grid.
            if grid_sequence == self.last_topology_grid_sequence:
                return
            self.last_topology_grid_sequence = grid_sequence
            depth_ids = set(self.pending_depth_doors)
            self.pending_depth_doors.clear()
        if grid is None or pose is None:
            return
        topology_phases = {
            MissionPhase.EXPLORE_GENERIC, MissionPhase.EXPLORE_CORRIDOR_AWARE,
            MissionPhase.RECOVERY,
            MissionPhase.CORRIDOR_INITIALIZE, MissionPhase.CORRIDOR_DISCOVERY,
            MissionPhase.CORRIDOR_END_CONFIRM, MissionPhase.BUILD_ROOM_QUEUE,
            MissionPhase.APPROACH_DOOR, MissionPhase.RETURN_TO_CORRIDOR,
            MissionPhase.SELECT_NEXT_ROOM, MissionPhase.GLOBAL_RECOVERY,
        }
        corridor = None
        tracked_corridors = []
        room_expansion_observations = []
        discover_branches = False
        if self.phase in topology_phases:
            discover_branches = (
                self.current_corridor is None or
                self.phase in (MissionPhase.EXPLORE_GENERIC, MissionPhase.RECOVERY,
                               MissionPhase.CORRIDOR_INITIALIZE,
                               MissionPhase.CORRIDOR_END_CONFIRM,
                               MissionPhase.GLOBAL_RECOVERY) or
                self.current_corridor.corridor_id in self.completed_corridors)
            if discover_branches:
                observations = self.corridor_detector.detect_candidates(
                    grid, (pose["x"], pose["y"]))
            else:
                observed = self.corridor_detector.detect(
                    grid, (pose["x"], pose["y"]),
                    self.current_corridor.principal_direction)
                observations = [observed] if observed is not None else []
            for observed in observations:
                tracked = self.topology.update_corridor(observed)
                if tracked is None:
                    continue
                tracked_corridors.append(tracked)
            for tracked in tracked_corridors:
                door_observations = self.door_detector.detect(grid, tracked)
                room_door_observations = []
                for door_observation in door_observations:
                    junction = corridor_junction_target(
                        door_observation, tracked.corridor_id,
                        self.topology.corridors)
                    if junction:
                        edge = tuple(sorted((tracked.corridor_id, junction)))
                        if edge not in self.corridor_edges:
                            self.corridor_edges.add(edge)
                            self._event("corridor_branch_registered",
                                        corridor_ids=edge)
                    else:
                        room_door_observations.append(door_observation)
                updated_doors = self.topology.update_doors(
                    room_door_observations, depth_ids if self.use_depth else ())
                for door in updated_doors:
                    door.corridor_id = tracked.corridor_id
            retained_doors = []
            candidate_door_ids = {
                item.door_id for item in self.room_candidates.values()
                if item.door_id
            }
            for door in self.topology.doors:
                if door.door_id in candidate_door_ids:
                    retained_doors.append(door)
                    continue
                junction = corridor_junction_target(
                    door, door.corridor_id, self.topology.corridors)
                if (junction and door.corridor_id and
                        door.door_id != self.current_door_id):
                    edge = tuple(sorted((door.corridor_id, junction)))
                    self.corridor_edges.add(edge)
                    continue
                retained_doors.append(door)
            self.topology.doors = retained_doors
            unfinished = [item for item in tracked_corridors
                          if item.corridor_id not in self.completed_corridors]
            confirmed_unfinished = [item for item in unfinished if item.confirmed]
            choices = confirmed_unfinished or unfinished
            if choices:
                corridor = max(choices, key=lambda item: (
                    item.confidence, item.forward_extent))
                self.current_corridor = corridor
            elif tracked_corridors:
                corridor = max(tracked_corridors,
                               key=lambda item: item.confidence)
                if self.current_corridor is None:
                    self.current_corridor = corridor
        room_reference = self._room_reference_corridor(pose)
        if room_reference is not None:
            room_expansion_observations.extend(
                self.room_expansion_detector.detect(grid, room_reference))
        updated_room_candidates = self._update_room_candidates(
            room_expansion_observations)
        promoted_room_doors = self._promote_room_candidates_to_doors(
            grid, grid_sequence)
        self.corridor_history.append({
            "t": round(self.elapsed(), 3),
            "observation": None if corridor is None else corridor.to_dict(),
            "observations": [item.to_dict() for item in tracked_corridors],
            "branch_discovery": discover_branches,
        })
        self.doorway_history.append({
            "t": round(self.elapsed(), 3),
            "doors": [door.to_dict() for door in self.topology.doors],
        })
        self.room_candidate_history.append({
            "t": round(self.elapsed(), 3),
            "observations": [item.to_dict() for item in updated_room_candidates],
            "tracked_candidates": [item.to_dict()
                                   for item in self.room_candidates.values()],
            "promoted_door_ids": [item.door_id for item in promoted_room_doors],
            "sensor_inputs": {
                "lidar_octomap_projection": True,
                "odometry_corridor_pose": True,
                "depth_camera_recent": (
                    self.use_depth and time.monotonic() - self.last_depth_stamp < 2.0),
            },
        })
        if self.current_room_id and self.current_door_id:
            room = self.rooms[self.current_room_id]
            door = self._door(self.current_door_id)
            if door:
                self._refresh_room_geometry(room, door, (pose["x"], pose["y"]))
                self.room_history.append({"t": round(self.elapsed(), 3),
                                          "room": room.to_dict()})
        signature = (len(self.topology.corridors), len(self.topology.doors),
                     len(self.rooms))
        if signature != self.last_topology_signature:
            self.last_topology_signature = signature
            self.global_no_candidate_cycles = 0
            self._event("new_topology_candidate", signature=signature)

    def _door(self, door_id: Optional[str]) -> Optional[Doorway]:
        return next((door for door in self.topology.doors if door.door_id == door_id), None)

    @staticmethod
    def _room_resolved(room: RoomRegion) -> bool:
        """Only completed rooms or explicitly proven unreachable rooms resolve."""
        return (room.state == RoomState.COMPLETED or
                (room.state == RoomState.TEMPORARILY_UNREACHABLE and
                 room.unreachable_confirmed and
                 bool(room.unreachable_reason)))

    def _corridor(self, corridor_id: Optional[str]):
        return next((item for item in self.topology.corridors
                     if item.corridor_id == corridor_id), None)

    def _path_stays_in_corridor(self, path: Sequence[Point], margin: float = 0.16) -> bool:
        corridor = self.current_corridor
        if not path or corridor is None or not corridor.centerline:
            return False
        axis = unit(corridor.principal_direction)
        normal = (-axis[1], axis[0])
        origin = corridor.centerline[0]
        limit = max(0.20, 0.5 * corridor.estimated_width - float(margin))
        return all(abs((point[0] - origin[0]) * normal[0] +
                       (point[1] - origin[1]) * normal[1]) <= limit
                   for point in path)

    @staticmethod
    def _point_in_corridor_region(
        point: Point, corridor, margin: float = 0.10,
    ) -> bool:
        if corridor is None or len(corridor.centerline) < 2:
            return False
        start, end = corridor.centerline[0], corridor.centerline[-1]
        axis = unit(corridor.principal_direction)
        normal = (-axis[1], axis[0])
        longitudinal = ((point[0] - start[0]) * axis[0] +
                        (point[1] - start[1]) * axis[1])
        length = distance(start, end)
        lateral = abs((point[0] - start[0]) * normal[0] +
                      (point[1] - start[1]) * normal[1])
        return (-0.5 <= longitudinal <= length + 0.5 and
                lateral <= max(0.20, 0.5 * corridor.estimated_width - margin))

    def _path_stays_in_corridor_union(self, path: Sequence[Point]) -> bool:
        corridors = [item for item in self.topology.corridors if item.confirmed]
        samples = interpolate_polyline(path, self.p.path_sample_spacing)
        return bool(samples and corridors and all(
            any(self._point_in_corridor_region(point, corridor)
                for corridor in corridors) for point in samples))

    @staticmethod
    def _nearest_corridor_point(point: Point, corridor) -> Point:
        start, end = corridor.centerline[0], corridor.centerline[-1]
        segment = (end[0] - start[0], end[1] - start[1])
        length_squared = segment[0] ** 2 + segment[1] ** 2
        alpha = 0.0 if length_squared < 1e-9 else float(np.clip(
            ((point[0] - start[0]) * segment[0] +
             (point[1] - start[1]) * segment[1]) / length_squared,
            0.0, 1.0))
        return (start[0] + alpha * segment[0],
                start[1] + alpha * segment[1])

    def _refresh_room_geometry(self, room: RoomRegion, door: Doorway,
                               robot_pose: Point) -> None:
        new_map_sample = room.last_map_sequence != self.grid_sequence
        update_room_geometry(room, self.grid, door, robot_pose,
                             map_sequence=self.grid_sequence)
        seed = (door.center[0] + max(1.05, room.maximum_inside_depth) * door.normal[0],
                door.center[1] + max(1.05, room.maximum_inside_depth) * door.normal[1])
        region = set(room_free_region(self.grid, door, seed))
        visited_in_room = 0
        for cell_x, cell_y in self.visited_cells:
            world = (cell_x * 0.25, cell_y * 0.25)
            grid_cell = self.grid.world_to_cell(world)
            visited_in_room += int(grid_cell in region)
        room.visited_free_area = visited_in_room * 0.25 ** 2
        if new_map_sample:
            room.visited_growth_history.append(room.visited_free_area)
            if len(room.visited_growth_history) > 10:
                room.visited_growth_history = room.visited_growth_history[-10:]

    def _write_fuel_history(self) -> str:
        path = os.path.join(self.output_dir, "fuel_execution_history.csv")
        with open(path, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["# x", "y", "source"])
            for item in self.trajectory[::10]:
                writer.writerow([item["x"], item["y"], "trajectory"])
            for item in self.goal_history:
                position = item.get("position") or []
                if len(position) >= 2 and item.get("success"):
                    writer.writerow([position[0], position[1], "goal"])
        return path

    def _run_fuel(self, mode: str) -> Tuple[List[dict], dict, str]:
        with self.lock:
            pose = dict(self.pose)
            frame = self.pose_frame
        self.planner_cycle += 1
        output = os.path.join(self.cycle_dir, "cycle_{:04d}_{}".format(
            self.planner_cycle, mode.lower()))
        os.makedirs(output, exist_ok=True)
        history = self._write_fuel_history()
        exploration_mode = ("ROOM_EXPLORATION" if mode == "room"
                            else mode.upper())
        command = [
            "rosrun", "simenv_competitor", "fuel_lite_planner",
            "__name:=structured_fuel_{:04d}".format(self.planner_cycle),
            "_map_file:=" + self.map_file, "_output_dir:=" + output,
            "_frame_id:=" + frame, "_use_pose_parameters:=true",
            "_exit_after_plan:=true", "_robot_x:={:.9f}".format(pose["x"]),
            "_robot_y:={:.9f}".format(pose["y"]),
            "_robot_z:={:.9f}".format(pose["z"]),
            "_frontier_slice_half_height:=0.30", "_minimum_cluster_size:=6",
            "_maximum_clusters:=100", "_candidates_per_cluster:=8",
            "_candidate_min_distance:=0.25", "_candidate_max_distance:=2.5",
            "_safety_clearance:=0.40", "_adaptive_clearance:=0.25",
            "_minimum_information_gain:=" + (
                "2.0" if mode == "corridor" else
                "0.0" if mode == "room_recovery" else "0.5"),
            "_exploration_mode:=" + exploration_mode,
            "_enable_room_information_gain:=" +
            str(self.enable_room_information_gain).lower(),
            "_room_unknown_decay_lambda:={:.9f}".format(
                self.room_unknown_decay_lambda),
            "_history_file:=" + history, "_sensor_range:=6.0",
            "_alpha:=0.25", "_beta:=0.3", "_gamma:=1.0",
            "_unknown_weight:=5.0", "_cluster_weight:=1.0",
            "_novelty_weight:=500.0", "_revisit_weight:=400.0",
            "/exploration_goal:=/simenv/structured_planner_raw_goal",
        ]
        try:
            with open(os.path.join(output, "planner.log"), "w", encoding="utf-8") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=self.planner_timeout, check=False)
            if completed.returncode != 0:
                return [], {}, output
        except (OSError, subprocess.TimeoutExpired):
            return [], {}, output
        candidates = self._load_json(os.path.join(output, "candidate_viewpoints.json")) or []
        scores = self._load_json(os.path.join(output, "exploration_score.json")) or {}
        by_score = {int(item.get("candidate_id", -1)): item
                    for item in scores.get("scores") or []}
        for candidate in candidates:
            decomposition = by_score.get(int(candidate.get("id", -1)), {})
            candidate["score_decomposition"] = decomposition
            candidate["manager_score"] = float(decomposition.get("score", -1e18))
        candidates.sort(key=lambda item: item["manager_score"], reverse=True)
        diagnostic = self._load_json(
            os.path.join(output, "frontier_filter_diagnostic.json")) or {}
        clusters = diagnostic.get("clusters") or []
        if clusters:
            self.large_reachable_unknown = any(
                int(item.get("size", 0)) >= 24 and
                int(item.get("candidate_count_after_connectivity", 0)) > 0
                for item in clusters)
        else:
            # Older planner artifacts do not carry connectivity counts. Be
            # conservative in that compatibility case.
            self.large_reachable_unknown = any(
                int(item.get("size", 0)) >= 24 for item in (
                    self._load_json(os.path.join(output, "frontiers.json")) or []))
        return candidates, scores, output

    def _select_fuel_goal(self, mode: str) -> Optional[dict]:
        candidates, _, output = self._run_fuel(mode)
        with self.lock:
            grid, pose = self.grid, dict(self.pose)
        if grid is None:
            return None
        current = (pose["x"], pose["y"])
        room = self.rooms.get(self.current_room_id) if self.current_room_id else None
        door = self._door(self.current_door_id)
        corridor = self.current_corridor
        room_cells = set()
        # All non-room candidates share one pool.  Confirmed structure only
        # contributes bonuses; missing corridor evidence leaves generic FUEL
        # terms intact and can never reject a candidate.
        if not mode.startswith("room"):
            for candidate in candidates:
                score = candidate.get("score_decomposition") or {}
                position = candidate.get("position") or [math.inf, math.inf]
                corridor_bonus = 0.0
                doorway_bonus = 0.0
                room_entry_bonus = 0.0
                if corridor is not None and corridor.confirmed and len(position) >= 2:
                    axis = unit(corridor.principal_direction)
                    progress = ((position[0] - current[0]) * axis[0] +
                                (position[1] - current[1]) * axis[1])
                    corridor_bonus = max(0.0, min(3.0, progress))
                for known_door in self.topology.doors:
                    if (known_door.confirmed and known_door.traversable and
                            not known_door.visited and len(position) >= 2):
                        proximity = max(0.0, 1.0 - distance(position, known_door.center) / 2.0)
                        doorway_bonus = max(doorway_bonus, proximity)
                        if known_door.estimated_unknown_size > 0.0:
                            room_entry_bonus = max(room_entry_bonus, proximity)
                candidate["corridor_progress_bonus"] = corridor_bonus
                candidate["doorway_bonus"] = doorway_bonus
                candidate["room_entry_bonus"] = room_entry_bonus
                candidate["local_score"] = unified_candidate_score(
                    float(score.get("information_gain", 0.0)),
                    float(score.get("local_unknown_volume", 0.0)),
                    float(score.get("novelty_bonus", 0.0)),
                    float(score.get("distance_cost", candidate.get("distance_to_robot", 0.0))),
                    float(score.get("collision_cost", 0.0)),
                    float(score.get("revisit_penalty", 0.0)),
                    corridor_bonus, doorway_bonus, room_entry_bonus)
            candidates.sort(key=lambda item: item.get("local_score", -math.inf), reverse=True)
        if mode.startswith("room") and room and door:
            seed = (door.center[0] + max(1.2, room.maximum_inside_depth) * door.normal[0],
                    door.center[1] + max(1.2, room.maximum_inside_depth) * door.normal[1])
            room_cells = set(room_free_region(grid, door, seed))
            for candidate in candidates:
                score = candidate.get("score_decomposition") or {}
                position = candidate.get("position") or [math.inf, math.inf]
                separation = min(
                    (distance(position, old) for old in room.successful_goals),
                    default=3.0)
                candidate["local_score"] = (
                    float(score.get("information_gain", 0.0))
                    + 5.0 * float(score.get("local_unknown_volume", 0.0))
                    + 2.0 * float(score.get("novelty_bonus", 0.0))
                    + min(3.0, separation)
                    - 0.30 * float(score.get("distance_cost", 0.0))
                    - float(score.get("collision_cost", 0.0))
                    - 2.0 * float(score.get("revisit_penalty", 0.0)))
            candidates.sort(key=lambda item: item.get("local_score", -math.inf),
                            reverse=True)
        rejected = []
        # Generic FUEL exploration already enforces 3-D connectivity and a
        # 0.40 m candidate clearance.  Its 2-D projection can contain narrow
        # unknown slivers even where the 3-D map is clear, so use centreline
        # occupancy plus the authoritative OctoMap path check here.  Keep the
        # full oriented footprint for room/corridor/door topology manoeuvres.
        strict_footprint = mode.startswith("room") or mode == "corridor"
        for candidate in candidates:
            position = candidate.get("position") or []
            if len(position) < 2:
                continue
            target = (float(position[0]), float(position[1]))
            if mode.startswith("corridor") and mode != "corridor_aware":
                if corridor is None or len(corridor.centerline) < 2:
                    continue
                axis = unit(corridor.principal_direction)
                normal = (-axis[1], axis[0])
                origin = corridor.centerline[0]
                rel = (target[0] - origin[0], target[1] - origin[1])
                lateral = abs(rel[0] * normal[0] + rel[1] * normal[1])
                forward = ((target[0] - current[0]) * axis[0] +
                           (target[1] - current[1]) * axis[1])
                if lateral > 0.5 * corridor.estimated_width - 0.28 or forward < 0.35:
                    rejected.append({"id": candidate.get("id"), "reason": "outside_corridor"})
                    continue
                path = astar_free_path(
                    grid, current, target, allow_unknown_at_start=True)
                corridor_limit = 0.5 * corridor.estimated_width - 0.24
                if path and any(abs(
                        (point[0] - origin[0]) * normal[0] +
                        (point[1] - origin[1]) * normal[1]) > corridor_limit
                                for point in path):
                    rejected.append({"id": candidate.get("id"),
                                     "reason": "path_leaves_corridor_region"})
                    self._record_rejected_candidate_path(
                        path, mode, candidate.get("id"),
                        "path_leaves_corridor_region")
                    continue
            elif mode.startswith("room"):
                cell = grid.world_to_cell(target)
                if not room or not door or cell not in room_cells or door.signed_depth(target) < 1.0:
                    rejected.append({"id": candidate.get("id"), "reason": "outside_room"})
                    continue
                if room.successful_goals and min(distance(target, old)
                                                 for old in room.successful_goals) < self.p.room_goal_baseline_min:
                    rejected.append({"id": candidate.get("id"), "reason": "viewpoint_baseline"})
                    continue
                path = astar_free_path(grid, current, target,
                                       allowed_side=(door, 1, 0.08),
                                       allow_unknown_at_start=True)
                if path and any(grid.world_to_cell(point) not in room_cells
                                for point in path[1:]):
                    rejected.append({"id": candidate.get("id"),
                                     "reason": "path_leaves_current_room_region"})
                    self._record_rejected_candidate_path(
                        path, mode, candidate.get("id"),
                        "path_leaves_current_room_region")
                    continue
            else:
                path = astar_free_path(
                    grid, current, target, allow_unknown_at_start=True)
                if (mode == "recovery" and
                        any(item.confirmed for item in self.topology.corridors) and
                        not self._path_stays_in_corridor_union(path)):
                    rejected.append({
                        "id": candidate.get("id"),
                        "reason": "global_recovery_leaves_confirmed_corridor_network",
                    })
                    self._record_rejected_candidate_path(
                        path, mode, candidate.get("id"),
                        "global_recovery_leaves_confirmed_corridor_network")
                    continue
            validity = validate_path(
                grid, path, spacing=self.p.path_sample_spacing,
                initial_unknown_footprint_grace=
                self.initial_unknown_footprint_grace,
                check_footprint=strict_footprint)
            if path and validity.valid:
                self._atomic_json(os.path.join(output, "structured_region_filter.json"), {
                    "mode": mode, "selected_candidate_id": candidate.get("id"),
                    "rejected": rejected,
                })
                return {"position": target, "yaw": float(candidate.get("yaw", 0.0)),
                        "path": path, "source": "fuel_" + mode,
                        "candidate_id": candidate.get("id"), "planner_cycle": output,
                        "local_score": candidate.get("local_score"),
                        "corridor_progress_bonus": candidate.get("corridor_progress_bonus", 0.0),
                        "doorway_bonus": candidate.get("doorway_bonus", 0.0),
                        "room_entry_bonus": candidate.get("room_entry_bonus", 0.0),
                        "score_decomposition": candidate.get("score_decomposition")}
            rejected.append({"id": candidate.get("id"), "reason": validity.reason})
            self._record_rejected_candidate_path(
                path, mode, candidate.get("id"), validity.reason, validity)
        self._atomic_json(os.path.join(output, "structured_region_filter.json"), {
            "mode": mode, "selected_candidate_id": None, "rejected": rejected,
        })
        return None

    def _adaptive_exploration_step(self) -> None:
        """Always try generic FUEL; topology is a parallel score enhancer."""
        candidate_door_ids = {
            candidate.door_id for candidate in self.room_candidates.values()
            if candidate.door_id
        }
        reachable_promoted = [
            door_id for door_id in self._build_reachable_room_queue()
            if door_id in candidate_door_ids
        ]
        if reachable_promoted and not self.door_detection_evaluation_only:
            self.room_queue = reachable_promoted
            self.current_door_id = reachable_promoted[0]
            self._transition(MissionPhase.APPROACH_DOOR,
                             "validated_room_candidate_door_reachable")
            return
        corridor_ready = bool(self.current_corridor and self.current_corridor.confirmed)
        desired = (MissionPhase.EXPLORE_CORRIDOR_AWARE if corridor_ready
                   else MissionPhase.EXPLORE_GENERIC)
        if self.phase != desired:
            self._transition(desired, "corridor_scoring_enabled" if corridor_ready
                             else "generic_exploration_available")
        goal = self._select_fuel_goal("corridor_aware" if corridor_ready else "generic")
        if goal:
            self.no_candidate_cycles = 0
            self.global_no_candidate_cycles = 0
            if (not self.door_detection_evaluation_only and
                    goal.get("room_entry_bonus", 0.0) > 0.5):
                eligible = [door for door in self.topology.doors
                            if door.confirmed and door.traversable and not door.visited]
                if eligible:
                    self.current_door_id = min(
                        eligible, key=lambda door: distance(goal["position"], door.center)).door_id
                    self._transition(MissionPhase.APPROACH_DOOR,
                                     "unified_pool_selected_room_entry")
                    return
            self.speed_pub.publish(Float32(data=self.corridor_speed))
            if not self._execute_path(goal["path"], goal["source"], metadata={
                    key: goal.get(key) for key in (
                        "candidate_id", "planner_cycle", "local_score",
                        "score_decomposition")}, strict_footprint=False):
                self._transition(MissionPhase.RECOVERY, "generic_goal_execution_failed")
            return
        self.no_candidate_cycles += 1
        self.global_no_candidate_cycles += 1
        self._transition(MissionPhase.RECOVERY, "no_safe_generic_candidate")

    def _record_rejected_candidate_path(
        self,
        path: Sequence[Point],
        mode: str,
        candidate_id,
        reason: str,
        validity=None,
    ) -> None:
        """Persist a proposed-but-never-authorized path for audit/rendering."""
        if not path:
            return
        if validity is None:
            validity = validate_path(
                self.grid, path, spacing=self.p.path_sample_spacing,
                initial_unknown_footprint_grace=
                self.initial_unknown_footprint_grace)
        self.path_history.append({
            "t": round(self.elapsed(), 3),
            "phase": self.phase.value,
            "source": "candidate_filter_" + mode,
            "candidate_id": candidate_id,
            "path": list(path),
            "execution_authorized": False,
            **validity.to_dict(),
            "valid": False,
            "reason": reason,
            "voxel_validation": {
                "safe": False,
                "skipped": "candidate_rejected_before_execution",
            },
        })

    def _publish_goal_and_wait(self, point: Point, yaw: float, source: str,
                               metadata: Optional[dict] = None) -> bool:
        message = PoseStamped()
        message.header.frame_id = self.pose_frame
        message.pose.position.x, message.pose.position.y = point
        message.pose.position.z = self.pose["z"]
        message.pose.orientation.z = math.sin(0.5 * yaw)
        message.pose.orientation.w = math.cos(0.5 * yaw)
        with self.lock:
            self.goal_command_sequence += 1
            expected_goal_sequence = self.goal_command_sequence
            # Goal headers are not sensor timestamps.  Use an always-unique
            # wall-clock identity even when Gazebo /use_sim_time is paused.
            identity_nanoseconds = max(
                time.time_ns(), self.last_goal_identity_nanoseconds + 1000)
            self.last_goal_identity_nanoseconds = identity_nanoseconds
            message.header.stamp = rospy.Time(
                identity_nanoseconds // 1000000000,
                identity_nanoseconds % 1000000000)
            message.header.seq = expected_goal_sequence
            expected_goal_stamp = message.header.stamp.to_sec()
            baseline = self.result_sequence
            self.command_active = True
        self.goal_pub.publish(message)
        deadline = time.monotonic() + min(self.goal_timeout,
                                          max(0.0, self.maximum_duration - self.elapsed()))
        result = None
        local_abort_reason = None
        inspected_results = set()
        last_motion_topology_wall = 0.0
        last_execution_safety_wall = 0.0
        consecutive_safety_risks = 0
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            now = time.monotonic()
            with self.lock:
                new_results = [dict(item) for item in self.execution_results
                               if int(item.get("sequence", 0)) > baseline]
                pose_age = time.monotonic() - self.pose_stamp
            for candidate in new_results:
                result_sequence = int(candidate.get("sequence", 0))
                if result_sequence in inspected_results:
                    continue
                inspected_results.add(result_sequence)
                if self._goal_result_matches(
                        candidate, expected_goal_sequence,
                        expected_goal_stamp, point,
                        self.require_goal_result_identity):
                    result = candidate
                    break
                self._event(
                    "stale_goal_result_ignored",
                    expected_goal_sequence=expected_goal_sequence,
                    returned_goal_sequence=candidate.get("goal_sequence"),
                    result_sequence=result_sequence)
            if result is not None:
                break
            if pose_age > 0.8:
                result = {"success": False, "reason": "localization_stale"}
                local_abort_reason = "localization_stale"
                break
            time.sleep(0.10)
        if result is None:
            result = {"success": False, "reason": "structured_manager_goal_timeout"}
            local_abort_reason = "structured_manager_goal_timeout"
        if local_abort_reason is not None:
            cancel = self._cancel_goal_and_wait(
                expected_goal_sequence, expected_goal_stamp, point,
                local_abort_reason, baseline)
            result["cancel"] = cancel
            if not cancel["acknowledged"]:
                self._fail_mission("goal_cancel_not_acknowledged")
        with self.lock:
            self.command_active = False
        item = {"index": len(self.goal_history), "t": round(self.elapsed(), 3),
                "phase": self.phase.value, "source": source,
                "goal_sequence": expected_goal_sequence,
                "goal_stamp": expected_goal_stamp,
                "position": [point[0], point[1], self.pose["z"]],
                "success": bool(result.get("success")), "result": result}
        if metadata:
            item["planning_metadata"] = metadata
        self.goal_history.append(item)
        return item["success"]

    def _compress_path(self, grid: OccupancyMap, path: Sequence[Point],
                       door: Optional[Doorway] = None,
                       strict_footprint: bool = True) -> List[Point]:
        if len(path) < 2:
            return []
        output = [path[0]]
        index = 0
        allowed = [door] if door else []
        while index < len(path) - 1:
            chosen = index + 1
            for candidate in range(index + 1, len(path)):
                if distance(path[index], path[candidate]) > 1.0:
                    break
                check = validate_path(grid, [path[index], path[candidate]], allowed,
                                      spacing=self.p.path_sample_spacing,
                                      check_footprint=strict_footprint)
                if check.valid:
                    chosen = candidate
            output.append(path[chosen])
            index = chosen
        return output

    def _voxel_validate_path(self, path: Sequence[Point], source: str,
                             required_clearance: float = 0.27) -> dict:
        samples = interpolate_polyline(path, self.p.path_sample_spacing)
        self.voxel_check_sequence += 1
        stem = "check_{:05d}_{}".format(
            self.voxel_check_sequence,
            "".join(character if character.isalnum() else "_" for character in source)[:48],
        )
        path_file = os.path.join(self.voxel_check_dir, stem + ".csv")
        with open(path_file, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["# x", "y", "yaw"])
            for index, point in enumerate(samples):
                if index + 1 < len(samples):
                    yaw = math.atan2(samples[index + 1][1] - point[1],
                                     samples[index + 1][0] - point[0])
                elif index > 0:
                    yaw = math.atan2(point[1] - samples[index - 1][1],
                                     point[0] - samples[index - 1][0])
                else:
                    yaw = 0.0
                writer.writerow([point[0], point[1], yaw])
        command = [
            "rosrun", "simenv_competitor", "octomap_path_validator",
            self.map_file, path_file, "{:.9f}".format(self.pose["z"]),
            "{:.9f}".format(required_clearance),
            "{:.9f}".format(self.p.path_sample_spacing + 1e-4),
            "{:.9f}".format(self.initial_unknown_footprint_grace),
        ]
        try:
            completed = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True, timeout=20.0, check=False)
            lines = [line for line in completed.stdout.splitlines() if line.strip()]
            payload = json.loads(lines[-1]) if lines else {}
            payload["return_code"] = completed.returncode
            payload["path_file"] = os.path.relpath(path_file, self.output_dir)
            payload["safe"] = bool(payload.get("safe")) and completed.returncode == 0
        except (OSError, subprocess.TimeoutExpired, ValueError) as error:
            payload = {"safe": False, "reason": "voxel_validator_error",
                       "error": repr(error),
                       "path_file": os.path.relpath(path_file, self.output_dir)}
        self._atomic_json(os.path.join(self.voxel_check_dir, stem + ".json"), payload)
        return payload

    def _audit_actual_trajectory(self) -> dict:
        """Validate the executed trajectory against the final 2-D and .bt maps."""
        with self.lock:
            grid = self.grid
            points = [(item["x"], item["y"]) for item in self.trajectory]
            doors = [door for door in self.topology.doors
                     if door.confirmed and door.traversable and
                     door.source_room_candidate_id]
        if grid is None or len(points) < 2:
            return {
                "performed": False, "safe": False,
                "reason": "insufficient_final_map_or_trajectory",
                "trajectory_points": len(points),
            }
        two_d = validate_path(
            grid, points, doors, spacing=self.p.path_sample_spacing)
        voxel = (self._voxel_validate_path(points, "final_actual_trajectory")
                 if os.path.isfile(self.map_file)
                 else {"safe": False, "reason": "final_voxel_map_missing"})
        illegal_events = sum(
            not bool(item.get("legal"))
            for item in self.door_traversal_history
            if isinstance(item, dict))
        safe = bool(two_d.valid and voxel.get("safe") and illegal_events == 0)
        return {
            "schema": "simenv_actual_trajectory_validation_v1",
            "performed": True,
            "safe": safe,
            "trajectory_points": len(points),
            "sample_spacing_m": self.p.path_sample_spacing,
            "final_grid_frame": self.grid_frame,
            "two_dimensional_validation": two_d.to_dict(),
            "voxel_validation": voxel,
            "illegal_actual_door_events": illegal_events,
            "reason": "safe" if safe else "final_actual_trajectory_unsafe",
        }

    def _execute_path(self, path: Sequence[Point], source: str,
                      door: Optional[Doorway] = None,
                      required_door: bool = False,
                      metadata: Optional[dict] = None,
                      strict_footprint: bool = True) -> bool:
        # A generic 0.30 m stopping radius is appropriate for frontier goals,
        # but can stop the body before a narrow door plane and make the next
        # segment cut diagonally into a frame.  Door entry/exit uses a tighter
        # tolerance so the alignment and plane-crossing waypoints are real.
        tolerance = (self.door_crossing_goal_tolerance
                     if door is not None and required_door
                     else self.generic_goal_tolerance)
        self.goal_tolerance_pub.publish(Float32(data=tolerance))
        time.sleep(0.05)
        with self.lock:
            grid = self.grid
        validity = validate_path(grid, path, [door] if door else [],
                                 door.door_id if door and required_door else None,
                                 self.p.path_sample_spacing,
                                 self.initial_unknown_footprint_grace,
                                 strict_footprint)
        # Generic FUEL motion follows already-observed free center cells.  At
        # the 0.15 m live OctoMap resolution an isolated quantized return can
        # report 0.225 m clearance and permanently deadlock recovery against
        # the former 0.27 m threshold.  Keep the stricter threshold for every
        # topology manoeuvre (corridor/door/room) and the final audit.
        voxel_clearance = 0.27 if strict_footprint else 0.20
        voxel = (self._voxel_validate_path(
            path, source, required_clearance=voxel_clearance)
                 if validity.valid else {"safe": False, "skipped": "2d_path_invalid"})
        record = {"t": round(self.elapsed(), 3), "phase": self.phase.value,
                  "source": source, "path": list(path), "door_id": door.door_id if door else None,
                  **validity.to_dict(), "voxel_validation": voxel}
        record["valid"] = bool(validity.valid and voxel.get("safe"))
        record["execution_authorized"] = record["valid"]
        if validity.valid and not voxel.get("safe"):
            record["reason"] = "voxel_path_invalid"
        self.path_history.append(record)
        if not record["valid"]:
            self._event("path_rejected", reason=record["reason"], source=source)
            return False
        waypoints = self._compress_path(
            grid, path, door, strict_footprint=strict_footprint)
        for _planned_start, target in zip(waypoints[:-1], waypoints[1:]):
            with self.lock:
                current_grid = self.grid
                actual_start = (self.pose["x"], self.pose["y"])
            segment = validate_path(current_grid, [actual_start, target],
                                    [door] if door else [],
                                    spacing=self.p.path_sample_spacing,
                                    initial_unknown_footprint_grace=
                                    self.initial_unknown_footprint_grace,
                                    check_footprint=strict_footprint)
            segment_voxel = (self._voxel_validate_path(
                [actual_start, target], source + "_online_recheck",
                required_clearance=voxel_clearance)
                if segment.valid else {"safe": False, "skipped": "2d_path_invalid"})
            if not segment.valid or not segment_voxel.get("safe"):
                self.path_history.append({
                    "t": round(self.elapsed(), 3), "phase": self.phase.value,
                    "source": source + "_online_recheck",
                    "path": [actual_start, target],
                    "execution_authorized": False,
                    **segment.to_dict(),
                    "valid": False,
                    "reason": (segment.reason if not segment.valid
                               else "voxel_path_invalid"),
                    "voxel_validation": segment_voxel,
                })
                return False
            yaw = math.atan2(target[1] - actual_start[1],
                             target[0] - actual_start[0])
            if not self._publish_goal_and_wait(target, yaw, source, metadata):
                return False
        return True

    def _corridor_step(self) -> None:
        goal = self._select_fuel_goal("corridor")
        if goal:
            self.corridor_end_cycles = 0
            self.no_candidate_cycles = 0
            self.speed_pub.publish(Float32(data=self.corridor_speed))
            if not self._execute_path(goal["path"], goal["source"], metadata={
                    key: goal.get(key) for key in (
                        "candidate_id", "planner_cycle", "local_score",
                        "score_decomposition")}):
                self._transition(MissionPhase.GLOBAL_RECOVERY, "corridor_path_execution_failed")
            return
        end_geometry = bool(
            self.current_corridor and
            self.current_corridor.forward_extent < self.p.corridor_forward_depth_min and
            corridor_end_geometry_evidence(
                self.grid, self.current_corridor,
                (self.pose["x"], self.pose["y"])))
        if end_geometry:
            if self.grid_sequence != self.last_corridor_end_grid_sequence:
                self.last_corridor_end_grid_sequence = self.grid_sequence
                self.corridor_end_cycles += 1
                self._event("corridor_end_evidence", cycles=self.corridor_end_cycles,
                            grid_sequence=self.grid_sequence,
                            evidence="known_occupied_ahead")
        else:
            self.corridor_end_cycles = 0
            self.last_corridor_end_grid_sequence = -1
            self.no_candidate_cycles += 1
            if self.no_candidate_cycles >= 3:
                self._transition(MissionPhase.GLOBAL_RECOVERY,
                                 "forward_depth_exists_but_fuel_goal_missing")
                return
        if self.corridor_end_cycles >= self.p.corridor_end_confirmation_frames:
            self.ending_corridor_id = self.current_corridor.corridor_id
            self.branch_scan_cycles = 0
            self.last_branch_scan_grid_sequence = self.grid_sequence
            self._transition(MissionPhase.CORRIDOR_END_CONFIRM,
                             "no_forward_corridor_goal_confirmed")
        else:
            time.sleep(1.0)

    def _entrance_bootstrap(self) -> bool:
        """Cross the public entrance using only body pose and observed free cells."""
        with self.lock:
            grid = self.grid
            pose = dict(self.pose)
        target, path = select_forward_free_bootstrap(
            grid, (pose["x"], pose["y"]), pose["yaw"],
            minimum_distance=1.0, maximum_distance=4.0,
            spacing=self.p.path_sample_spacing)
        self.bootstrap_attempted = True
        if target is None:
            self._event("pose_relative_bootstrap_unavailable")
            return False
        self._event("pose_relative_bootstrap_selected", target=target,
                    definition="body-forward observed-free; no world/layout coordinate")
        self.speed_pub.publish(Float32(data=min(self.corridor_speed, 0.40)))
        return self._execute_path(path, "pose_relative_entrance_bootstrap")

    def _approach_door(self) -> None:
        door = self._door(self.current_door_id)
        if not door:
            self._transition(MissionPhase.SELECT_NEXT_ROOM, "door_disappeared")
            return
        refresh = self._refresh_candidate_door_channel(door, "approach_door")
        if not refresh.get("safe"):
            self._transition(MissionPhase.SELECT_NEXT_ROOM,
                             "latest_map_door_channel_unavailable")
            return
        nx, ny = door.normal
        outside = (
            door.center[0] - self.p.door_outside_distance * nx,
            door.center[1] - self.p.door_outside_distance * ny)
        current = (self.pose["x"], self.pose["y"])
        # First approach the ordinary outside staging point along short A*
        # segments.  Applying tolerance compensation from the distant
        # observation pose made the previous run demand one large, nearly
        # right-angle diagonal manoeuvre and the locomotion controller reported
        # no progress three times.  Precision compensation belongs only to the
        # final local alignment step below.
        path = astar_free_path(
            self.grid, current, outside,
            clearance_radius=self.corridor_network_clearance)
        already_near_staging = bool(
            not path and
            distance(current, outside) <= self.door_crossing_goal_tolerance)
        if not path and not already_near_staging:
            # A* can collapse nearby poses into one 0.15 m map cell.  Preserve
            # the metric approach command; _execute_path still applies normal
            # 2-D and voxel checks before publishing it.
            path = [current, outside]
        source_corridor = self._corridor(door.corridor_id)
        if source_corridor is not None:
            self.current_corridor = source_corridor
        self.speed_pub.publish(Float32(data=self.door_speed))
        approach_succeeded = already_near_staging or self._execute_path(
            path, "approach_door",
            strict_footprint=False,
            maximum_segment_distance=0.30,
            goal_tolerance=self.door_crossing_goal_tolerance)
        if approach_succeeded:
            # Goal success means only that the body entered a tolerance circle.
            # Perform a separate short correction so the subsequent segment is
            # parallel to the validated door normal instead of clipping a frame.
            aligned_pose = (self.pose["x"], self.pose["y"])
            alignment_error = distance(aligned_pose, outside)
            lateral_error = abs(door.lateral_offset(aligned_pose))
            precise_tolerance = min(0.08, self.door_crossing_goal_tolerance)
            precise_needed = bool(
                alignment_error > 0.05 or lateral_error > 0.04)
            precise_target = outside
            if precise_needed:
                precise_target = tolerance_compensated_target(
                    aligned_pose, outside, precise_tolerance)
                self._event(
                    "door_precise_alignment_started",
                    door_id=door.door_id,
                    position=aligned_pose,
                    desired_target=precise_anchor,
                    commanded_target=precise_target,
                    alignment_error=alignment_error,
                    lateral_error=lateral_error)
                approach_succeeded = self._execute_path(
                    [aligned_pose, precise_target],
                    "door_precise_alignment",
                    strict_footprint=False,
                    maximum_segment_distance=0.25,
                    goal_tolerance=precise_tolerance)
            if approach_succeeded:
                final_pose = (self.pose["x"], self.pose["y"])
                final_alignment_error = distance(final_pose, outside)
                final_lateral_error = abs(door.lateral_offset(final_pose))
                # If the controller reports success on an unexpected side of
                # its tolerance circle, keep the confirmed door local and align
                # again instead of attempting a diagonal crossing.
                approach_succeeded = bool(
                    final_alignment_error <= 0.11 and
                    final_lateral_error <= 0.07)
                self._event(
                    "door_waiting_point_alignment_completed",
                    door_id=door.door_id,
                    desired_target=precise_anchor,
                    commanded_target=precise_target,
                    final_position=final_pose,
                    final_alignment_error=final_alignment_error,
                    final_lateral_error=final_lateral_error,
                    accepted=approach_succeeded)
        if approach_succeeded:
            self.door_approach_failure_cycles.pop(door.door_id, None)
            self._transition(MissionPhase.CROSS_DOOR, "door_waiting_point_reached")
        else:
            approach_attempt = self.door_approach_failure_cycles.get(
                door.door_id, 0) + 1
            self.door_approach_failure_cycles[door.door_id] = approach_attempt
            if approach_attempt < self.p.door_entry_max_attempts:
                self._event("door_entry_retry_locked", door_id=door.door_id,
                            attempt=approach_attempt,
                            maximum_attempts=self.p.door_entry_max_attempts,
                            reason="door_approach_unreachable")
                self._transition(MissionPhase.APPROACH_DOOR,
                                 "retry_same_door_approach")
            else:
                door.blacklist_until = self.elapsed() + min(
                    120.0, 20.0 * door.attempt_count)
                self._transition(MissionPhase.SELECT_NEXT_ROOM,
                                 "door_approach_attempts_exhausted")

    def _cross_door(self) -> None:
        door = self._door(self.current_door_id)
        if not door:
            self._transition(MissionPhase.SELECT_NEXT_ROOM, "door_disappeared")
            return
        refresh = self._refresh_candidate_door_channel(door, "cross_door")
        if not refresh.get("safe"):
            self._transition(MissionPhase.SELECT_NEXT_ROOM,
                             "latest_map_door_channel_unavailable")
            return
        if refresh.get("geometry_changed"):
            # The robot is waiting at the old outside point.  Re-approach the
            # newly fitted entrance instead of cutting diagonally across its
            # frames from a stale pose.
            self._transition(MissionPhase.APPROACH_DOOR,
                             "door_geometry_refitted_reapproach_required")
            return
        heading_error = wrap_angle(
            door.normal_direction - float(self.pose["yaw"]))
        if abs(heading_error) > self.door_heading_tolerance:
            door.geometry_locked = False
            self._event(
                "door_crossing_heading_guard_rejected",
                door_id=door.door_id, current_yaw=self.pose["yaw"],
                desired_yaw=door.normal_direction,
                yaw_error=heading_error,
                tolerance=self.door_heading_tolerance)
            self._transition(
                MissionPhase.APPROACH_DOOR,
                "door_heading_alignment_required_before_crossing")
            return
        nx, ny = door.normal
        inside = (
            door.center[0] + self.p.door_inside_distance * nx,
            door.center[1] + self.p.door_inside_distance * ny)
        current = (self.pose["x"], self.pose["y"])
        route, validity = plan_door_route(self.grid, current, inside, door, entering=True,
                                          sample_spacing=self.p.path_sample_spacing,
                                          outside_distance=self.p.door_outside_distance,
                                          inside_distance=self.p.door_inside_distance)
        # When already at the outside wait point, the generic route ends at
        # the same inside point and still proves the required door crossing.
        if not validity.valid:
            door.geometry_locked = False
            self.path_history.append({"t": round(self.elapsed(), 3),
                                      "source": "cross_door", "path": route,
                                      "execution_authorized": False,
                                      **validity.to_dict()})
            candidate = self._candidate_for_door(door)
            if candidate is not None:
                door.traversable = False
                invalidation = dict(door.channel_validation)
                invalidation.update({
                    "safe": False,
                    "stage": "cross_door_planned_route",
                    "grid_sequence": int(self.grid_sequence),
                    "planned_route_validation": validity.to_dict(),
                })
                door.channel_validation = invalidation
                candidate.channel_validation = dict(invalidation)
                self._event("candidate_door_route_invalidated",
                            door_id=door.door_id,
                            candidate_id=candidate.candidate_id,
                            reason=validity.reason,
                            grid_sequence=self.grid_sequence)
                self._transition(MissionPhase.SELECT_NEXT_ROOM,
                                 "candidate_door_requires_new_map_refit")
                return
            door.attempt_count += 1
            if door.attempt_count < self.p.door_entry_max_attempts:
                self._event("door_entry_retry_locked", door_id=door.door_id,
                            attempt=door.attempt_count,
                            maximum_attempts=self.p.door_entry_max_attempts,
                            reason="door_crossing_path_invalid")
                self._transition(MissionPhase.APPROACH_DOOR,
                                 "retry_same_door_after_path_refresh")
            else:
                door.blacklist_until = self.elapsed() + min(
                    120.0, 20.0 * door.attempt_count)
                self._transition(MissionPhase.SELECT_NEXT_ROOM,
                                 "door_crossing_path_attempts_exhausted")
            return
        self.speed_pub.publish(Float32(data=self.door_speed))
        traversal_baseline = len(self.door_traversal_history)
        trajectory_baseline = max(0, len(self.trajectory) - 1)
        if not self._execute_path(route, "cross_door", door, required_door=True):
            door.attempt_count += 1
            if door.signed_depth((self.pose["x"], self.pose["y"])) > 0.05:
                if door.attempt_count >= self.p.door_entry_max_attempts:
                    door.blacklist_until = self.elapsed() + min(
                        120.0, 20.0 * door.attempt_count)
                self._transition(MissionPhase.EXIT_ROOM,
                                 "door_crossing_failed_after_plane")
            else:
                if door.attempt_count < self.p.door_entry_max_attempts:
                    self._event("door_entry_retry_locked", door_id=door.door_id,
                                attempt=door.attempt_count,
                                maximum_attempts=self.p.door_entry_max_attempts,
                                reason="door_crossing_failed_before_plane")
                    self._transition(MissionPhase.APPROACH_DOOR,
                                     "retry_same_door_after_execution_failure")
                else:
                    door.blacklist_until = self.elapsed() + min(
                        120.0, 20.0 * door.attempt_count)
                    self._transition(MissionPhase.SELECT_NEXT_ROOM,
                                     "door_crossing_attempts_exhausted")
            return
        actual_enter = any(
            item.get("door_id") == door.door_id and item.get("legal") and
            item.get("direction") == "enter"
            for item in self.door_traversal_history[traversal_baseline:])
        entry_evidence = door_entry_evidence(
            door, self.trajectory[trajectory_baseline:],
            inside_depth_min=self.p.inside_depth_min,
            minimum_inside_samples=self.p.door_inside_confirmation_samples)
        self._event("door_entry_evidence", door_id=door.door_id,
                    evidence=entry_evidence)
        if not actual_enter or not entry_evidence["confirmed"]:
            door.attempt_count += 1
            if door.attempt_count >= self.p.door_entry_max_attempts:
                door.blacklist_until = self.elapsed() + min(
                    120.0, 20.0 * door.attempt_count)
            self._transition(MissionPhase.EXIT_ROOM,
                             ("planned_crossing_lacks_actual_door_event"
                              if not actual_enter else
                              "door_entry_trajectory_not_confirmed"))
            return
        door.attempt_count = 0
        # This exact validated plane is now the only legal return portal.
        # Future wall/room observations may grow the room, but cannot move its
        # remembered entrance while the robot is inside.
        door.geometry_locked = True
        self._event("room_entry_door_geometry_locked", door_id=door.door_id,
                    center=door.center, normal_direction=door.normal_direction,
                    corridor_side=door.corridor_side,
                    interior_side=door.interior_side,
                    validated_grid_sequence=door.last_validated_grid_sequence)
        room_id = door.adjacent_region_id or "room_" + door.door_id
        door.adjacent_region_id = room_id
        room = self.rooms.setdefault(room_id, RoomRegion(room_id, door.door_id))
        self.current_room_id = room_id
        self.room_exit_failures = 0
        self.return_corridor_failures = 0
        room.state = RoomState.ENTERING
        room.no_goal_cycles = 0
        room.no_goal_map_sequences = []
        self._refresh_room_geometry(
            room, door, (self.pose["x"], self.pose["y"]))
        if (door.signed_depth((self.pose["x"], self.pose["y"])) >= self.p.inside_depth_min
                and room.expansion_confidence > 0.0):
            room.state = RoomState.EXPLORING
            self._transition(MissionPhase.EXPLORE_ROOM_LOCAL,
                             "door_crossed_and_room_expansion_confirmed")
        else:
            room.unreachable_attempt_count += 1
            room.unreachable_confirmed = room.unreachable_attempt_count >= 3
            room.unreachable_reason = "room_geometry_not_confirmed_after_door_crossing"
            door.blacklist_until = self.elapsed() + min(
                120.0, 20.0 * room.unreachable_attempt_count)
            room.state = RoomState.TEMPORARILY_UNREACHABLE
            self._transition(MissionPhase.EXIT_ROOM, "room_confirmation_failed")

    def _room_step(self) -> None:
        room = self.rooms[self.current_room_id]
        door = self._door(self.current_door_id)
        self._refresh_room_geometry(
            room, door, (self.pose["x"], self.pose["y"]))
        current = (self.pose["x"], self.pose["y"])
        goal = None
        # Establish room scale before asking local frontiers for refinements.
        # The first full path goes into the room depth; the following two fan
        # out laterally.  Keeping the full macro path prevents every rolling
        # FUEL waypoint from becoming a new high-level room target.
        if room.local_goal_count < 3:
            role = "primary" if room.local_goal_count == 0 else "breadth"
            coverage = select_room_coverage_viewpoint(
                room, self.grid, door, current, role,
                minimum_baseline=self.p.room_goal_baseline_min)
            if coverage is not None:
                target = coverage["position"]
                candidate = self._candidate_for_door(door)
                yaw = (math.atan2(candidate.centroid[1] - target[1],
                                  candidate.centroid[0] - target[0])
                       if candidate is not None else door.normal_direction)
                metrics = {
                    key: value for key, value in coverage.items()
                    if key not in ("position", "path")
                }
                goal = {
                    "position": target,
                    "yaw": yaw,
                    "path": coverage["path"],
                    "source": ("room_depth_primary" if role == "primary" else
                               "room_breadth_expand"),
                    "candidate_id": None,
                    "planner_cycle": None,
                    "local_score": None,
                    "score_decomposition": metrics,
                }
        if not goal:
            goal = self._select_fuel_goal("room")
        if not goal and room.local_goal_count < 2:
            goal = self._select_fuel_goal("room_recovery")
        if not goal and room.local_goal_count < 2:
            # Frontier extraction can legitimately become empty after one
            # wide LiDAR observation while the required second, separated
            # camera viewpoint is still missing.  Choose a geometry-only
            # fallback inside the same door-bounded room component; FUEL
            # remains the primary scorer and this path still passes every
            # 2-D/Voxel/footprint execution gate.
            target = select_room_viewpoint(
                room, self.grid, door, self.p.room_goal_baseline_min)
            if target is not None:
                seed = (door.center[0] +
                        max(1.2, room.maximum_inside_depth) * door.normal[0],
                        door.center[1] +
                        max(1.2, room.maximum_inside_depth) * door.normal[1])
                room_cells = set(room_free_region(self.grid, door, seed))
                path = astar_free_path(
                    self.grid, (self.pose["x"], self.pose["y"]), target,
                    allowed_side=(door, 1, 0.08))
                validity = validate_path(
                    self.grid, path, spacing=self.p.path_sample_spacing)
                if (path and validity.valid and
                        all(self.grid.world_to_cell(point) in room_cells
                            for point in path[1:])):
                    goal = {
                        "position": target,
                        "yaw": door.normal_direction,
                        "path": path,
                        "source": "room_geometry_recovery",
                        "candidate_id": None,
                        "planner_cycle": None,
                        "local_score": None,
                        "score_decomposition": {
                            "reason": "second_separated_viewpoint_after_empty_frontier",
                        },
                    }
        if goal:
            room.no_goal_cycles = 0
            room.no_goal_map_sequences = []
            self.speed_pub.publish(Float32(data=self.room_speed))
            success = self._execute_path(goal["path"], goal["source"], door,
                                         metadata={key: goal.get(key) for key in (
                                             "candidate_id", "planner_cycle", "local_score",
                                             "score_decomposition")})
            if success:
                target = goal["position"]
                room.successful_goals.append(target)
                room.local_goal_count += 1
            else:
                room.no_goal_cycles += 1
            return
        # Count only fresh map cycles. Re-running the planner against one
        # unchanged grid must not manufacture the three-cycle stability
        # evidence required by the room completion rule.
        if (not room.no_goal_map_sequences or
                room.no_goal_map_sequences[-1] != self.grid_sequence):
            room.no_goal_map_sequences.append(self.grid_sequence)
            room.no_goal_map_sequences = room.no_goal_map_sequences[-3:]
            room.no_goal_cycles = len(room.no_goal_map_sequences)
        else:
            time.sleep(0.5)
            return
        if room_complete(room):
            room.state = RoomState.COMPLETED
            room.unreachable_confirmed = False
            room.unreachable_reason = None
            self._transition(MissionPhase.EXIT_ROOM, "room_completion_evidence_satisfied")
        elif room.no_goal_cycles >= 3:
            room.unreachable_attempt_count += 1
            room.unreachable_confirmed = room.unreachable_attempt_count >= 3
            room.unreachable_reason = (
                "three_independent_room_visits_without_safe_local_viewpoint"
                if room.unreachable_confirmed else
                "room_revisit_required_after_no_safe_local_viewpoint")
            door.blacklist_until = self.elapsed() + min(
                120.0, 20.0 * room.unreachable_attempt_count)
            room.state = RoomState.TEMPORARILY_UNREACHABLE
            self._transition(MissionPhase.EXIT_ROOM, "room_local_goals_exhausted")
        else:
            time.sleep(1.0)

    def _exit_room(self) -> None:
        door = self._door(self.current_door_id)
        if not door:
            self._fail_mission("lost_entry_door_inside_room")
            return
        target = door.corridor_side
        route, validity = plan_door_route(
            self.grid, (self.pose["x"], self.pose["y"]), target, door, entering=False,
            sample_spacing=self.p.path_sample_spacing,
            outside_distance=self.p.door_outside_distance,
            inside_distance=self.p.door_inside_distance)
        room = self.rooms.get(self.current_room_id)
        if room:
            seed = (door.center[0] + max(1.2, room.maximum_inside_depth) * door.normal[0],
                    door.center[1] + max(1.2, room.maximum_inside_depth) * door.normal[1])
            region = set(room_free_region(self.grid, door, seed))
            if any(door.signed_depth(point) > 0.05 and
                   self.grid.world_to_cell(point) not in region for point in route):
                validity.valid = False
                validity.reason = "exit_path_leaves_current_room_region"
        if not validity.valid:
            self.path_history.append({
                "t": round(self.elapsed(), 3), "phase": self.phase.value,
                "source": "exit_room", "path": route, "door_id": door.door_id,
                "execution_authorized": False,
                **validity.to_dict(),
                "voxel_validation": {"safe": False, "skipped": "2d_path_invalid"},
            })
            self.room_exit_failures += 1
            if self.room_exit_failures >= 3:
                self._fail_mission("room_exit_path_failed")
            else:
                time.sleep(1.0)
            return
        traversal_baseline = len(self.door_traversal_history)
        if not self._execute_path(route, "exit_room", door, required_door=True):
            self.room_exit_failures += 1
            if self.room_exit_failures >= 3:
                self._fail_mission("room_exit_path_failed")
            else:
                time.sleep(1.0)
            return
        actual_exit = any(
            item.get("door_id") == door.door_id and item.get("legal") and
            item.get("direction") == "exit"
            for item in self.door_traversal_history[traversal_baseline:])
        if not actual_exit:
            self._fail_mission("planned_exit_lacks_actual_door_event")
            return
        self.room_exit_failures = 0
        if room and self._room_resolved(room):
            door.visited = True
        self._transition(MissionPhase.RETURN_TO_CORRIDOR, "room_exit_door_crossed")

    def _return_corridor(self) -> None:
        door = self._door(self.current_door_id)
        target = door.corridor_side
        source_corridor = self._corridor(door.corridor_id)
        if source_corridor:
            self.current_corridor = source_corridor
            axis = unit(source_corridor.principal_direction)
            origin = source_corridor.centerline[0]
            station = ((target[0] - origin[0]) * axis[0] +
                       (target[1] - origin[1]) * axis[1])
            target = (origin[0] + station * axis[0], origin[1] + station * axis[1])
        path = astar_free_path(
            self.grid, (self.pose["x"], self.pose["y"]), target,
            clearance_radius=self.corridor_network_clearance)
        if not self._path_stays_in_corridor_union(path):
            path = []
        if self._execute_path(path, "return_to_corridor"):
            self.return_corridor_failures = 0
            self.current_room_id = None
            self.current_door_id = None
            self._transition(MissionPhase.SELECT_NEXT_ROOM, "corridor_centerline_rejoined")
        else:
            self.return_corridor_failures += 1
            if self.return_corridor_failures >= 3:
                self._fail_mission("return_to_corridor_failed")
            else:
                time.sleep(1.0)

    def _global_recovery(self) -> None:
        corridor_incomplete = bool(
            self.current_corridor and self.current_corridor.confirmed and
            self.current_corridor.corridor_id not in self.completed_corridors)
        queue = self._build_reachable_room_queue()
        self.room_queue = queue
        if (corridor_incomplete and self.current_corridor and
                not self._point_in_corridor_region(
                    (self.pose["x"], self.pose["y"]), self.current_corridor)):
            current = (self.pose["x"], self.pose["y"])
            target = self._nearest_corridor_point(current, self.current_corridor)
            transfer = astar_free_path(
                self.grid, current, target,
                clearance_radius=self.corridor_network_clearance)
            if self._path_stays_in_corridor_union(transfer):
                self.speed_pub.publish(Float32(data=self.corridor_speed))
                if self._execute_path(transfer, "corridor_branch_transfer"):
                    self._transition(MissionPhase.CORRIDOR_DISCOVERY,
                                     "corridor_branch_centerline_reached")
                    return
        if (queue and not corridor_incomplete and
                not self.door_detection_evaluation_only):
            self._transition(MissionPhase.SELECT_NEXT_ROOM, "unvisited_door_available")
            return
        # A recovery candidate is always drawn from the unrestricted generic
        # pool.  Corridor evidence may improve its score but cannot constrain
        # its region or make recovery loop waiting for recognition.
        goal = self._select_fuel_goal("generic_recovery")
        if goal:
            self.no_candidate_cycles = 0
            self.global_no_candidate_cycles = 0
            self.speed_pub.publish(Float32(data=self.corridor_speed))
            if self._execute_path(goal["path"], goal["source"], metadata={
                    key: goal.get(key) for key in (
                        "candidate_id", "planner_cycle", "local_score",
                        "score_decomposition")}, strict_footprint=False):
                self._transition(MissionPhase.EXPLORE_GENERIC,
                                 "recovery_goal_reached_resume_generic")
            else:
                self.no_candidate_cycles += 1
            return
        self.no_candidate_cycles += 1
        self.global_no_candidate_cycles += 1
        rooms_resolved = all(self._room_resolved(room)
                             for room in self.rooms.values())
        unresolved_confirmed_doors = any(
            door.confirmed and door.traversable and
            bool(door.source_room_candidate_id) and not door.visited
            for door in self.topology.doors)
        complete, reason = completion_decision(CompletionEvidence(
            all_corridor_branches_complete=(
                bool(self.completed_corridors) and all(
                    (not corridor.confirmed or
                     corridor.corridor_id in self.completed_corridors)
                    for corridor in self.topology.corridors)),
            door_queue_empty=(not queue and not unresolved_confirmed_doors),
            rooms_resolved=rooms_resolved,
            large_reachable_unknown=self.large_reachable_unknown,
            no_candidate_cycles=self.global_no_candidate_cycles,
            visited_stagnant_seconds=time.monotonic() - self.last_visited_growth,
            elapsed_seconds=self.elapsed(),
        ))
        if complete:
            self.exploration_complete = True
            self.termination_reason = reason
            self._transition(MissionPhase.FINISHED, reason)
        else:
            time.sleep(2.0)

    def _dispatch(self) -> None:
        if self.phase in (MissionPhase.EXPLORE_GENERIC,
                          MissionPhase.EXPLORE_CORRIDOR_AWARE):
            self._adaptive_exploration_step()
        elif self.phase == MissionPhase.EXPLORE_ROOM_LOCAL:
            self._room_step()
        elif self.phase == MissionPhase.RECOVERY:
            # Recovery never requires corridor/door/room evidence.  Replan
            # directly from the generic pool and return to the default state.
            goal = self._select_fuel_goal("generic_recovery")
            if goal:
                self.speed_pub.publish(Float32(data=self.corridor_speed))
                if self._execute_path(goal["path"], goal["source"], metadata={
                        key: goal.get(key) for key in (
                            "candidate_id", "planner_cycle", "local_score",
                            "score_decomposition")}):
                    self.no_candidate_cycles = 0
                    self.global_no_candidate_cycles = 0
                    self._transition(MissionPhase.EXPLORE_GENERIC,
                                     "generic_recovery_goal_reached")
            else:
                self._global_recovery()
        elif self.phase == MissionPhase.CORRIDOR_INITIALIZE:
            if self.current_corridor and self.current_corridor.confirmed:
                self._transition(MissionPhase.CORRIDOR_DISCOVERY,
                                 "multi_frame_corridor_confirmed")
            else:
                self.no_candidate_cycles += 1
                if self.no_candidate_cycles >= self.p.corridor_confirmation_frames:
                    if not self.bootstrap_attempted and self._entrance_bootstrap():
                        self.no_candidate_cycles = 0
                        self._event("pose_relative_bootstrap_completed")
                    else:
                        self._transition(MissionPhase.GLOBAL_RECOVERY,
                                         "corridor_not_yet_observable")
                else:
                    time.sleep(0.8)
        elif self.phase == MissionPhase.CORRIDOR_DISCOVERY:
            self._corridor_step()
        elif self.phase == MissionPhase.CORRIDOR_END_CONFIRM:
            if self.grid_sequence != self.last_branch_scan_grid_sequence:
                self.last_branch_scan_grid_sequence = self.grid_sequence
                self.branch_scan_cycles += 1
            if self.branch_scan_cycles < self.p.corridor_confirmation_frames:
                time.sleep(0.5)
                return
            if self.ending_corridor_id:
                self.completed_corridors.add(self.ending_corridor_id)
            unfinished = [item for item in self.topology.corridors
                          if item.confirmed and
                          item.corridor_id not in self.completed_corridors]
            if unfinished:
                self.current_corridor = max(
                    unfinished, key=lambda item: (item.confidence,
                                                  item.forward_extent))
                self._transition(MissionPhase.GLOBAL_RECOVERY,
                                 "confirmed_corridor_branch_pending")
            else:
                self._transition(MissionPhase.BUILD_ROOM_QUEUE,
                                 "corridor_end_and_branch_scan_stable")
        elif self.phase == MissionPhase.BUILD_ROOM_QUEUE:
            self.room_queue = self._build_reachable_room_queue()
            self._event("room_queue_built", door_ids=self.room_queue)
            self._transition(
                (MissionPhase.GLOBAL_RECOVERY
                 if self.door_detection_evaluation_only
                 else MissionPhase.SELECT_NEXT_ROOM),
                ("door_detection_evaluation_suppresses_entry"
                 if self.door_detection_evaluation_only
                 else "door_queue_ready"))
        elif self.phase == MissionPhase.SELECT_NEXT_ROOM:
            if self.door_detection_evaluation_only:
                self._transition(
                    MissionPhase.GLOBAL_RECOVERY,
                    "door_detection_evaluation_suppresses_entry")
                return
            self.room_queue = self._build_reachable_room_queue()
            if self.room_queue:
                self.current_door_id = self.room_queue[0]
                self._transition(MissionPhase.APPROACH_DOOR, "next_unvisited_door_selected")
            else:
                self._transition(MissionPhase.GLOBAL_RECOVERY, "room_queue_empty")
        elif self.phase == MissionPhase.APPROACH_DOOR:
            self._approach_door()
        elif self.phase == MissionPhase.CROSS_DOOR:
            self._cross_door()
        elif self.phase == MissionPhase.ROOM_LOCAL_EXPLORE:
            self._room_step()
        elif self.phase == MissionPhase.EXIT_ROOM:
            self._exit_room()
        elif self.phase == MissionPhase.RETURN_TO_CORRIDOR:
            self._return_corridor()
        elif self.phase == MissionPhase.GLOBAL_RECOVERY:
            self._global_recovery()

    def _summary(self) -> dict:
        illegal_actual = sum(not item.get("legal", False)
                             for item in self.door_traversal_history)
        authorized_paths = [item for item in self.path_history
                            if item.get("execution_authorized")]
        rejected_paths = [item for item in self.path_history
                          if not item.get("execution_authorized")]
        path_walls = sum(int(item.get("occupied_intersections", 0))
                         for item in authorized_paths)
        path_unknown = sum(int(item.get("unknown_intersections", 0))
                           for item in authorized_paths)
        footprint_collisions = sum(int(item.get("footprint_collisions", 0))
                                   for item in authorized_paths)
        voxel_unknown = sum(int((item.get("voxel_validation") or {}).get(
            "unknown_samples", 0)) for item in authorized_paths)
        voxel_occupied = sum(int((item.get("voxel_validation") or {}).get(
            "occupied_samples", 0)) for item in authorized_paths)
        voxel_clearance = sum(int((item.get("voxel_validation") or {}).get(
            "clearance_failures", 0)) for item in authorized_paths)
        voxel_validator_errors = sum(
            bool(item.get("voxel_validation")) and
            not bool((item.get("voxel_validation") or {}).get("safe")) and
            not bool((item.get("voxel_validation") or {}).get("skipped")) and
            not any(int((item.get("voxel_validation") or {}).get(key, 0)) > 0
                    for key in ("unknown_samples", "occupied_samples",
                                "clearance_failures", "spacing_failures"))
            for item in authorized_paths)
        non_door = sum(int(item.get("non_door_crossings", 0))
                       for item in authorized_paths) + illegal_actual
        actual = self.actual_trajectory_validation or {}
        actual_2d = actual.get("two_dimensional_validation") or {}
        actual_voxel = actual.get("voxel_validation") or {}
        actual_safe = bool(actual.get("performed") and actual.get("safe"))
        snapshot_safe = bool(self.final_voxel_snapshot.get("complete"))
        confirmed_corridors = [corridor for corridor in self.topology.corridors
                               if corridor.confirmed]
        confirmed_doors = [door for door in self.topology.doors if door.confirmed]
        executable_doors = [
            door for door in confirmed_doors
            if door.traversable and door.source_room_candidate_id]
        rooms_resolved = all(self._room_resolved(room)
                             for room in self.rooms.values())
        all_corridors_complete = bool(self.completed_corridors) and all(
            corridor.corridor_id in self.completed_corridors
            for corridor in confirmed_corridors)
        unresolved_doors = [door for door in executable_doors if not door.visited]
        visited_stagnant_seconds = max(
            0.0, time.monotonic() - self.last_visited_growth)
        completion_evidence = {
            "all_corridor_branches_complete": all_corridors_complete,
            "door_queue_empty": not unresolved_doors and not self.room_queue,
            "rooms_resolved": rooms_resolved,
            "large_reachable_unknown": self.large_reachable_unknown,
            "global_no_candidate_cycles": self.global_no_candidate_cycles,
            "visited_stagnant_seconds": round(visited_stagnant_seconds, 3),
        }
        complete = bool(self.exploration_complete and path_walls == 0 and
                        path_unknown == 0 and footprint_collisions == 0 and non_door == 0)
        complete = bool(complete and voxel_unknown == 0 and voxel_occupied == 0 and
                        voxel_clearance == 0 and voxel_validator_errors == 0)
        complete = bool(complete and snapshot_safe and actual_safe)
        return {
            "schema": "simenv_structured_exploration_summary_v1",
            "corridors_detected": sum(c.confirmed for c in self.topology.corridors),
            "corridors_completed": len(self.completed_corridors),
            "corridor_branch_edges": len(self.corridor_edges),
            "doors_confirmed": sum(d.confirmed for d in self.topology.doors),
            "doors_executable": sum(
                d.confirmed and d.traversable and bool(d.source_room_candidate_id)
                for d in self.topology.doors),
            "room_candidates_detected": len(self.room_candidates),
            "room_candidates_confirmed": sum(
                candidate.confirmed for candidate in self.room_candidates.values()),
            "room_candidates_with_depth_confirmation": sum(
                candidate.depth_confirmations > 0
                for candidate in self.room_candidates.values()),
            "rooms_discovered": len(self.rooms),
            "rooms_entered": sum(r.maximum_inside_depth >= self.p.inside_depth_min
                                 for r in self.rooms.values()),
            "rooms_completed": sum(r.state == RoomState.COMPLETED
                                   for r in self.rooms.values()),
            "path_wall_intersections": path_walls,
            "path_unknown_intersections": path_unknown,
            "footprint_collisions": footprint_collisions,
            "voxel_unknown_samples": voxel_unknown,
            "voxel_occupied_samples": voxel_occupied,
            "voxel_clearance_failures": voxel_clearance,
            "voxel_validator_errors": voxel_validator_errors,
            "non_door_crossings": non_door,
            "rejected_path_count": len(rejected_paths),
            "rejected_path_wall_intersections": sum(
                int(item.get("occupied_intersections", 0))
                for item in rejected_paths),
            "rejected_path_unknown_intersections": sum(
                int(item.get("unknown_intersections", 0))
                for item in rejected_paths),
            "rejected_path_footprint_collisions": sum(
                int(item.get("footprint_collisions", 0))
                for item in rejected_paths),
            "final_voxel_snapshot_acknowledged": bool(
                self.final_voxel_snapshot.get("acknowledged")),
            "final_voxel_snapshot_complete": snapshot_safe,
            "actual_trajectory_safe": actual_safe,
            "actual_trajectory_occupied_intersections": int(
                actual_2d.get("occupied_intersections", 0)),
            "actual_trajectory_unknown_intersections": int(
                actual_2d.get("unknown_intersections", 0)),
            "actual_trajectory_footprint_collisions": int(
                actual_2d.get("footprint_collisions", 0)),
            "actual_trajectory_non_door_crossings": int(
                actual_2d.get("non_door_crossings", 0)) + int(
                    actual.get("illegal_actual_door_events", 0)),
            "actual_trajectory_voxel_unknown_samples": int(
                actual_voxel.get("unknown_samples", 0)),
            "actual_trajectory_voxel_occupied_samples": int(
                actual_voxel.get("occupied_samples", 0)),
            "actual_trajectory_voxel_clearance_failures": int(
                actual_voxel.get("clearance_failures", 0)),
            "global_no_candidate_cycles": self.global_no_candidate_cycles,
            "completion_evidence": completion_evidence,
            "door_traversal_successes": sum(item.get("legal", False)
                                            for item in self.door_traversal_history),
            "exploration_complete": complete,
            "termination_reason": self.termination_reason or (
                "ros_shutdown" if rospy.is_shutdown() else "running"),
            "elapsed_seconds": round(self.elapsed(), 3),
            "final_phase": self.phase.value,
            "topology_map_source": "fastlio_octomap_projection",
            "topology_grid_topic": self.grid_topic,
            "voxel_map_file": self.map_file,
            "use_depth_door_confirmation": self.use_depth,
            "use_rgb_semantic_confirmation": self.use_rgb,
            "door_detection_evaluation_only": self.door_detection_evaluation_only,
        }

    def _room_queue_records(self) -> List[dict]:
        records = []
        axis = unit(self.current_corridor.principal_direction) if self.current_corridor else (1.0, 0.0)
        left_normal = (-axis[1], axis[0])
        current = ((self.pose["x"], self.pose["y"])
                   if self.pose else (0.0, 0.0))
        for door_id in self.room_queue:
            door = self._door(door_id)
            if not door:
                continue
            side_dot = door.normal[0] * left_normal[0] + door.normal[1] * left_normal[1]
            records.append({
                "door_id": door.door_id,
                "door_pose": {"center": door.center,
                              "normal_direction": door.normal_direction},
                "left_or_right_side": "left" if side_dot >= 0.0 else "right",
                "adjacent_region_id": door.adjacent_region_id,
                "estimated_unknown_size": door.estimated_unknown_size,
                "distance_from_current": distance(current, door.center),
                "confidence": door.confidence,
                "visited": door.visited,
                "attempt_count": door.attempt_count,
                "blacklist_until": door.blacklist_until,
            })
        return records

    def _build_reachable_room_queue(self) -> List[str]:
        if self.pose is None or self.grid is None:
            return []
        current = (self.pose["x"], self.pose["y"])
        ordered = build_room_queue(self.topology.doors, current, self.elapsed())
        reachable = []
        for door_id in ordered:
            door = self._door(door_id)
            nx, ny = door.normal
            outside = (door.center[0] - 0.75 * nx,
                       door.center[1] - 0.75 * ny)
            path = astar_free_path(
                self.grid, current, outside,
                clearance_radius=self.corridor_network_clearance)
            validity = validate_path(
                self.grid, path, spacing=self.p.path_sample_spacing)
            # Corridor topology is a scheduling hint, not a collision layer.
            # A confirmed door can sit just outside a temporarily drifted or
            # truncated corridor polygon.  Known-free A*, full-footprint 2-D
            # validation and the execution-time OctoMap check remain the
            # authoritative safety gates for reaching its waiting point.
            if path and validity.valid:
                reachable.append(door_id)
        return reachable

    def _write_artifacts(self) -> None:
        edges = []
        for first, second in sorted(self.corridor_edges):
            edges.append({"from": first, "to": second,
                          "type": "corridor_to_corridor"})
        for door in self.topology.doors:
            corridor_id = door.corridor_id
            if corridor_id:
                edges.append({"from": corridor_id, "to": door.door_id,
                              "type": "corridor_to_door"})
            if door.adjacent_region_id:
                edges.append({"from": door.door_id, "to": door.adjacent_region_id,
                              "type": "door_to_room"})
        topology = {
            "schema": "simenv_indoor_topology_v1",
            "map_provenance": {
                "source": "fastlio_octomap_projection",
                "grid_topic": self.grid_topic,
                "voxel_map_file": self.map_file,
                "frame_id": self.grid_frame,
            },
            "corridors": [item.to_dict() for item in self.topology.corridors],
            "doors": [item.to_dict() for item in self.topology.doors],
            "room_candidates": [item.to_dict()
                                for item in self.room_candidates.values()],
            "rooms": [item.to_dict() for item in self.rooms.values()],
            "edges": edges,
            "room_queue": self._room_queue_records(),
            "completed_corridor_ids": sorted(self.completed_corridors),
            "current_phase": self.phase.value,
        }
        self._atomic_json(os.path.join(self.output_dir, "indoor_topology.json"), topology)
        self._atomic_json(os.path.join(self.output_dir, "topology_evidence.json"), topology)
        self._atomic_json(os.path.join(self.output_dir, "corridor_history.json"),
                          self.corridor_history)
        self._atomic_json(os.path.join(self.output_dir, "doorway_history.json"),
                          self.doorway_history)
        self._atomic_json(os.path.join(self.output_dir, "room_region_history.json"),
                          self.room_history)
        self._atomic_json(os.path.join(self.output_dir, "room_candidate_history.json"),
                          self.room_candidate_history)
        self._atomic_json(os.path.join(self.output_dir, "door_traversal_history.json"),
                          self.door_traversal_history)
        self._atomic_json(os.path.join(self.output_dir, "path_validity_history.json"),
                          self.path_history)
        self._atomic_json(os.path.join(self.output_dir, "semantic_geometry_observation.json"), {
            "schema": "simenv_semantic_geometry_observation_v1",
            "latest": self.latest_depth, "observations": self.semantic_history,
        })
        self._atomic_json(os.path.join(self.output_dir, "goal_history.json"), self.goal_history)
        self._atomic_json(os.path.join(self.output_dir, "structured_events.json"), self.events)
        self._atomic_json(os.path.join(self.output_dir, "state_history.json"), self.events)
        self._atomic_json(os.path.join(self.output_dir, "final_voxel_snapshot.json"),
                          self.final_voxel_snapshot)
        self._atomic_json(os.path.join(self.output_dir,
                                       "actual_trajectory_validation.json"),
                          self.actual_trajectory_validation)
        self._atomic_json(os.path.join(self.output_dir, "exploration_summary.json"),
                          self._summary())
        with open(os.path.join(self.output_dir, "trajectory.csv"), "w", newline="",
                  encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["t", "x", "y", "z", "yaw"])
            writer.writeheader()
            for item in self.trajectory:
                writer.writerow({key: item[key] for key in writer.fieldnames})
        self._render_images()

    def _draw_base(self, axis) -> None:
        if self.grid is None:
            return
        display = np.ones((*self.grid.data.shape, 3), dtype=np.float32)
        display[self.grid.data < 0] = (0.55, 0.55, 0.55)
        display[self.grid.data >= 50] = (0.05, 0.05, 0.05)
        axis.imshow(display, origin="lower", extent=(
            self.grid.origin_x, self.grid.origin_x + self.grid.width * self.grid.resolution,
            self.grid.origin_y, self.grid.origin_y + self.grid.height * self.grid.resolution),
                    interpolation="nearest")

    def _render_images(self) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch, Polygon
        except Exception as error:
            rospy.logwarn("Structured rendering unavailable: %s", error)
            return
        # 1: topology graph.
        fig, axis = plt.subplots(figsize=(10, 9), dpi=130)
        self._draw_base(axis)
        for corridor in self.topology.corridors:
            if corridor.centerline:
                axis.plot(*zip(*corridor.centerline), color="#ff8c22", linewidth=2.2)
            if corridor.left_wall:
                axis.plot(*zip(*corridor.left_wall), color="#4a90e2", linewidth=1.2)
            if corridor.right_wall:
                axis.plot(*zip(*corridor.right_wall), color="#4a90e2", linewidth=1.2)
        for door in self.topology.doors:
            axis.plot([door.left_frame_point[0], door.right_frame_point[0]],
                      [door.left_frame_point[1], door.right_frame_point[1]],
                      color="#35c46a" if door.confirmed else "#e6c229", linewidth=4)
            axis.text(door.center[0], door.center[1], door.door_id, fontsize=7)
        for candidate in self.room_candidates.values():
            color = "#8e24aa" if candidate.confirmed else "#ab7ac1"
            axis.scatter(candidate.centroid[0], candidate.centroid[1],
                         color=color, marker="s", s=38, zorder=5)
            axis.text(candidate.centroid[0], candidate.centroid[1],
                      "{} R={:.2f}".format(
                          candidate.candidate_id, candidate.fused_confidence),
                      fontsize=7, color=color)
            if candidate.portal_center:
                axis.plot([candidate.portal_center[0], candidate.centroid[0]],
                          [candidate.portal_center[1], candidate.centroid[1]],
                          color=color, linestyle=":", linewidth=1.0)
                axis.scatter(candidate.portal_center[0], candidate.portal_center[1],
                             color=color, marker="x", s=30, zorder=5)
        for room in self.rooms.values():
            if room.estimated_boundary:
                axis.add_patch(Polygon(room.estimated_boundary, closed=True, fill=False,
                                       edgecolor="#b053d1", linewidth=1.4))
                centroid = (
                    sum(point[0] for point in room.estimated_boundary) /
                    len(room.estimated_boundary),
                    sum(point[1] for point in room.estimated_boundary) /
                    len(room.estimated_boundary),
                )
                entry = self._door(room.entry_door_id)
                if entry:
                    axis.plot([entry.center[0], centroid[0]],
                              [entry.center[1], centroid[1]],
                              color="#b053d1", linestyle="--", linewidth=0.9)
        for door in self.topology.doors:
            corridor = next((item for item in self.topology.corridors
                             if item.corridor_id == door.corridor_id), None)
            if corridor and corridor.centerline:
                start, end = corridor.centerline[0], corridor.centerline[-1]
                segment = (end[0] - start[0], end[1] - start[1])
                length_squared = segment[0] ** 2 + segment[1] ** 2
                alpha = 0.0 if length_squared < 1e-9 else float(np.clip(
                    ((door.center[0] - start[0]) * segment[0] +
                     (door.center[1] - start[1]) * segment[1]) /
                    length_squared, 0.0, 1.0))
                nearest = (start[0] + alpha * segment[0],
                           start[1] + alpha * segment[1])
                axis.plot([nearest[0], door.center[0]],
                          [nearest[1], door.center[1]],
                          color="#35c46a", linestyle="--", linewidth=0.8)
        axis.set_aspect("equal"); axis.set_title("Online corridor-door-room topology")
        fig.tight_layout(); fig.savefig(os.path.join(self.output_dir, "indoor_topology.png"))
        plt.close(fig)
        # 2: strict path/door validation.
        fig, axis = plt.subplots(figsize=(10, 9), dpi=130)
        self._draw_base(axis)
        if self.trajectory:
            axis.plot([p["x"] for p in self.trajectory], [p["y"] for p in self.trajectory],
                      color="#ff8c22", linewidth=1.6, label="actual trajectory")
            actual_unsafe = []
            if self.grid is not None:
                actual_samples = interpolate_polyline(
                    [(p["x"], p["y"]) for p in self.trajectory],
                    self.p.path_sample_spacing)
                for sample_index, point in enumerate(actual_samples):
                    if sample_index + 1 < len(actual_samples):
                        following = actual_samples[sample_index + 1]
                        yaw = math.atan2(following[1] - point[1],
                                         following[0] - point[0])
                    elif sample_index > 0:
                        previous = actual_samples[sample_index - 1]
                        yaw = math.atan2(point[1] - previous[1],
                                         point[0] - previous[0])
                    else:
                        yaw = 0.0
                    if (self.grid.state(point) != 0 or
                            not self.grid.footprint_safe(point, yaw)):
                        actual_unsafe.append(point)
            if actual_unsafe:
                axis.scatter([point[0] for point in actual_unsafe],
                             [point[1] for point in actual_unsafe],
                             s=18, c="red", marker="x", alpha=0.9)
        for item in self.path_history:
            path = item.get("path") or []
            if len(path) >= 2:
                axis.plot(*zip(*path), color="#42d4f4" if item.get("valid") else "red",
                          alpha=0.65, linewidth=0.9)
                unsafe = []
                if self.grid is not None:
                    samples = interpolate_polyline(path, self.p.path_sample_spacing)
                    for sample_index, point in enumerate(samples):
                        if sample_index + 1 < len(samples):
                            following = samples[sample_index + 1]
                            yaw = math.atan2(following[1] - point[1],
                                             following[0] - point[0])
                        elif sample_index > 0:
                            previous = samples[sample_index - 1]
                            yaw = math.atan2(point[1] - previous[1],
                                             point[0] - previous[0])
                        else:
                            yaw = 0.0
                        if (self.grid.state(point) != 0 or
                                not self.grid.footprint_safe(point, yaw)):
                            unsafe.append(point)
                if unsafe:
                    axis.scatter([point[0] for point in unsafe],
                                 [point[1] for point in unsafe],
                                 s=12, c="red", marker="x", alpha=0.8)
        for door in self.topology.doors:
            nx, ny = door.normal; tx, ty = door.tangent
            half = 0.5 * door.width
            polygon = [
                (door.center[0] - half * tx - 0.18 * nx,
                 door.center[1] - half * ty - 0.18 * ny),
                (door.center[0] + half * tx - 0.18 * nx,
                 door.center[1] + half * ty - 0.18 * ny),
                (door.center[0] + half * tx + 0.18 * nx,
                 door.center[1] + half * ty + 0.18 * ny),
                (door.center[0] - half * tx + 0.18 * nx,
                 door.center[1] - half * ty + 0.18 * ny),
            ]
            axis.add_patch(Polygon(polygon, closed=True, fill=False,
                                   edgecolor="#35c46a", linewidth=1.6))
        for crossing in self.door_traversal_history:
            axis.scatter(crossing["point"][0], crossing["point"][1], s=25,
                         c="#35c46a" if crossing["legal"] else "red", marker="x")
        axis.legend(handles=[
            Line2D([0], [0], color="#ff8c22", linewidth=1.6,
                   label="actual trajectory"),
            Line2D([0], [0], color="#42d4f4", linewidth=1.2,
                   label="execution-authorized path"),
            Line2D([0], [0], color="red", linewidth=1.2,
                   label="rejected / never executed"),
            Line2D([0], [0], color="red", marker="x", linestyle="None",
                   label="unsafe sample"),
            Patch(facecolor="none", edgecolor="#35c46a",
                  label="confirmed door channel"),
        ], loc="best", fontsize=8)
        axis.set_aspect("equal"); axis.set_title("Path and confirmed-door validation")
        fig.tight_layout(); fig.savefig(os.path.join(
            self.output_dir, "path_and_door_validation.png")); plt.close(fig)
        # 3: queue, local goals and observed/unknown map.
        fig, axis = plt.subplots(figsize=(10, 9), dpi=130)
        self._draw_base(axis)
        if self.visited_cells:
            visited_x = [cell[0] * 0.25 for cell in self.visited_cells]
            visited_y = [cell[1] * 0.25 for cell in self.visited_cells]
            axis.scatter(visited_x, visited_y, s=28, c="#77c8ff", marker="s",
                         alpha=0.45, linewidths=0, label="visited area")
        if self.trajectory:
            axis.plot([p["x"] for p in self.trajectory], [p["y"] for p in self.trajectory],
                      color="#ff8c22", linewidth=1.7)
        for room in self.rooms.values():
            goals = room.successful_goals
            if goals:
                axis.scatter([p[0] for p in goals], [p[1] for p in goals],
                             s=35, c="#b053d1", marker="*")
        for index, door_id in enumerate(self.room_queue):
            door = self._door(door_id)
            if door:
                axis.text(door.center[0], door.center[1], "Q{}".format(index + 1),
                          color="#ffdd33", fontsize=8, fontweight="bold")
        legend = [
            Patch(facecolor="0.55", label="unknown"),
            Patch(facecolor="white", edgecolor="0.7",
                  label="observed free / unvisited"),
            Patch(facecolor="#77c8ff", alpha=0.45, label="visited area"),
        ]
        axis.legend(handles=legend, loc="best", fontsize=8)
        axis.set_aspect("equal"); axis.set_title(
            "Structured exploration | phase={}".format(self.phase.value))
        fig.tight_layout(); fig.savefig(os.path.join(
            self.output_dir, "structured_exploration_map.png")); plt.close(fig)
        shutil.copyfile(os.path.join(self.output_dir, "structured_exploration_map.png"),
                        os.path.join(self.output_dir,
                                     "adaptive_structured_exploration.png"))
    def run(self) -> None:
        self.complete_pub.publish(Bool(data=False))
        if not self._wait_ready():
            self.termination_reason = "startup_timeout_or_frame_mismatch"
            self._write_artifacts()
            return
        self.start_wall = time.monotonic()
        self.last_visited_growth = self.start_wall
        self._event("structured_exploration_started")
        try:
            while not rospy.is_shutdown() and self.phase != MissionPhase.FINISHED:
                if self.elapsed() >= self.maximum_duration:
                    self.termination_reason = "safety_timeout"
                    self.exploration_complete = False
                    self._transition(MissionPhase.FINISHED, "1800_second_safety_limit")
                    break
                self._update_topology()
                self._dispatch()
                if time.monotonic() - self.last_artifact_write >= 10.0:
                    self.last_artifact_write = time.monotonic()
                    self._write_artifacts()
        except Exception as error:  # preserve complete diagnostics on runtime faults
            self.termination_reason = "exception:" + type(error).__name__
            self.exploration_complete = False
            rospy.logerr("Structured exploration exception: %r", error)
            self._event("runtime_exception", error=repr(error))
            self.phase = MissionPhase.FINISHED
        finally:
            snapshot_ok = self._request_final_voxel_snapshot()
            if snapshot_ok:
                self.actual_trajectory_validation = self._audit_actual_trajectory()
            else:
                self.actual_trajectory_validation = {
                    "performed": False, "safe": False,
                    "reason": "final_voxel_snapshot_unavailable",
                }
            if self.exploration_complete and not snapshot_ok:
                self.exploration_complete = False
                self.termination_reason = "final_voxel_snapshot_failed"
            elif self.exploration_complete and not self.actual_trajectory_validation.get(
                    "safe", False):
                self.exploration_complete = False
                self.termination_reason = "final_actual_trajectory_validation_failed"
            self._write_artifacts()
            self.finalized = True
            self.complete_pub.publish(Bool(
                data=bool(self._summary()["exploration_complete"])))

    def _shutdown(self) -> None:
        if not self.finalized:
            if self.termination_reason is None:
                self.termination_reason = "ros_shutdown"
            self._write_artifacts()


if __name__ == "__main__":
    rospy.init_node("structured_exploration_manager")
    StructuredExplorationManager().run()
