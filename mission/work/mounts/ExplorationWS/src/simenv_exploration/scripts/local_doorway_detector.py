#!/usr/bin/env python3
"""Layout-free doorway detection in a robot-centred occupancy window.

The detector deliberately does not know a corridor station, a global y range,
or Gazebo/layout metadata.  Global coordinates only label the locally observed
free-space candidate so downstream A*/SCAN-lite can execute it.
"""

import json
import math
import os
import sys
import threading
import time
from collections import defaultdict

import numpy as np
import rospy
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from simenv_exploration.srv import CheckTwinCylinder, CheckTwinCylinderRequest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_planning_core import OccupancyGrid2D, astar_safe_path  # noqa: E402
from goal_executor_core import quaternion_yaw  # noqa: E402


class LocalDoorwayDetector:
    def __init__(self):
        self.output_dir = os.path.abspath(rospy.get_param("~output_dir"))
        os.makedirs(os.path.join(self.output_dir, "logs"), exist_ok=True)
        self.log_path = os.path.join(
            self.output_dir, "logs", "local_door_candidates.jsonl")
        self.window_size = float(rospy.get_param("~local_window_size", 8.0))
        self.wall_search = float(rospy.get_param("~wall_search_range", 2.6))
        self.behind_depth = float(rospy.get_param("~opening_probe_depth", 3.5))
        self.width_min = float(rospy.get_param("~doorway_width_min", 0.75))
        self.width_max = float(rospy.get_param("~doorway_width_max", 2.40))
        self.entry_depth = float(rospy.get_param("~entry_depth", 1.55))
        self.clearance = float(rospy.get_param("~clearance", 0.28))
        self.minimum_open_area = float(rospy.get_param(
            "~minimum_open_area_m2", 2.5))
        self.confirmations = int(rospy.get_param("~confirmation_updates", 2))
        self.scan_enabled = bool(rospy.get_param("~enable_scan_lite", True))
        self.candidate_astar_enabled = bool(rospy.get_param(
            "~enable_candidate_astar", False))
        self.rate_hz = float(rospy.get_param("~rate", 2.0))
        self.started = time.monotonic()
        self.lock = threading.RLock()
        self.pose = None
        self.grid = None
        self.grid_seq = 0
        self.cloud_stamp = 0.0
        self.trajectory = []
        self.observations = defaultdict(int)
        # Quantised map coordinates are not stable enough while FAST-LIO is
        # still settling in the first corridor.  Keep short-lived geometric
        # tracks so the same physical aperture retains its confirmation count
        # across a sub-metre map-frame correction.
        self.tracks = {}
        self.next_track_id = 1
        self.track_match_radius = float(rospy.get_param(
            "~track_match_radius_m", 1.05))
        self.track_max_age = float(rospy.get_param(
            "~track_max_age_seconds", 5.0))
        self.last_processed_grid = -1
        self.door_pub = rospy.Publisher(
            "/simenv/local_door_candidates", String, queue_size=2, latch=True)
        self.entry_pub = rospy.Publisher(
            "/simenv/room_entry_candidates", String, queue_size=2, latch=True)
        self.scan = rospy.ServiceProxy(
            "/baseline_voxel_mapper/check_twin_cylinder", CheckTwinCylinder)
        rospy.Subscriber("/Odometry", Odometry, self._on_odom, queue_size=20)
        rospy.Subscriber("/cloud_registered", PointCloud2,
                         self._on_cloud, queue_size=2)
        rospy.Subscriber("/simenv/voxel_floor_projection", OccupancyGrid,
                         self._on_grid, queue_size=2)
        rospy.Timer(rospy.Duration(1.0 / max(0.2, self.rate_hz)), self._tick)

    def elapsed(self):
        return time.monotonic() - self.started

    def _on_odom(self, message):
        p, q = message.pose.pose.position, message.pose.pose.orientation
        pose = (float(p.x), float(p.y), float(p.z),
                quaternion_yaw(q.x, q.y, q.z, q.w))
        if not all(math.isfinite(v) for v in pose):
            return
        with self.lock:
            self.pose = pose
            if (not self.trajectory or
                    math.hypot(pose[0] - self.trajectory[-1][1],
                               pose[1] - self.trajectory[-1][2]) >= 0.10):
                self.trajectory.append((self.elapsed(), pose[0], pose[1]))
                self.trajectory = self.trajectory[-300:]

    def _on_cloud(self, message):
        with self.lock:
            self.cloud_stamp = time.monotonic()

    def _on_grid(self, message):
        try:
            data = np.asarray(message.data, dtype=np.int16).reshape(
                message.info.height, message.info.width)
            grid = OccupancyGrid2D(
                data, float(message.info.resolution),
                float(message.info.origin.position.x),
                float(message.info.origin.position.y))
        except (TypeError, ValueError):
            return
        with self.lock:
            self.grid = grid
            self.grid_seq += 1

    @staticmethod
    def _value(grid, point):
        cell = grid.world_to_cell(point)
        return None if cell is None else int(grid.data[cell[1], cell[0]])

    def _free_with_clearance(self, grid, point):
        cell = grid.world_to_cell(point)
        if cell is None or int(grid.data[cell[1], cell[0]]) != 0:
            return False
        radius = int(math.ceil(self.clearance / grid.resolution))
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if math.hypot(dx, dy) * grid.resolution > self.clearance:
                    continue
                x, y = cell[0] + dx, cell[1] + dy
                if not (0 <= x < grid.width and 0 <= y < grid.height):
                    return False
                if int(grid.data[y, x]) >= 50:
                    return False
        return True

    def _local_unknown_area(self, grid, pose):
        half = self.window_size * 0.5
        x0 = max(0, int((pose[0] - half - grid.origin_x) / grid.resolution))
        x1 = min(grid.width, int(math.ceil(
            (pose[0] + half - grid.origin_x) / grid.resolution)))
        y0 = max(0, int((pose[1] - half - grid.origin_y) / grid.resolution))
        y1 = min(grid.height, int(math.ceil(
            (pose[1] + half - grid.origin_y) / grid.resolution)))
        if x1 <= x0 or y1 <= y0:
            return 0.0
        return float(np.count_nonzero(grid.data[y0:y1, x0:x1] < 0) *
                     grid.resolution * grid.resolution)

    def _scan_safe(self, point, z, yaw):
        if not self.scan_enabled:
            return True, "disabled"
        request = CheckTwinCylinderRequest()
        request.pose_x, request.pose_y, request.pose_z = point[0], point[1], z
        request.yaw = yaw
        request.front_offset, request.rear_offset = 0.06675, -0.06675
        request.radius = 0.11772
        request.min_height, request.max_height = -0.057, 0.057
        request.clearance_search_radius = 0.05
        try:
            response = self.scan(request)
            safe = bool(response.map_available and
                        not response.occupied_collision)
            return safe, str(response.status)
        except (rospy.ServiceException, rospy.ROSException) as error:
            return False, "scan_service_error:" + str(error)

    def _side_profiles(self, grid, pose, axis, side):
        normal = np.asarray([-axis[1], axis[0]], dtype=float) * float(side)
        origin = np.asarray(pose[:2], dtype=float)
        step = max(0.15, grid.resolution)
        profiles = []
        for station in np.arange(-2.8, 2.81, step):
            base = origin + station * axis
            wall = None
            free_depth = 0.0
            for depth in np.arange(0.25, self.wall_search + step, step):
                value = self._value(grid, base + depth * normal)
                if value is None or value < 0:
                    break
                if value >= 50:
                    wall = float(depth)
                    break
                free_depth = float(depth)
            # A side aperture is a break in the ordinary near wall with known
            # free space extending behind it. Unknown itself is never a goal.
            open_ray = wall is None and free_depth >= min(1.20, self.entry_depth)
            profiles.append((float(station), open_ray, free_depth, base, normal))
        return profiles

    def _track_key(self, side, door_center, desired):
        now = time.monotonic()
        # Same side plus proximity is required, so opposite room doors at one
        # corridor station can never merge into a confirmation.
        choices = [(math.hypot(track["center"][0] - door_center[0],
                               track["center"][1] - door_center[1]), key)
                   for key, track in self.tracks.items()
                   if track["side"] == side and now - track["seen"] <= self.track_max_age]
        if choices and min(choices)[0] <= self.track_match_radius:
            key = min(choices)[1]
        else:
            key = (int(side), self.next_track_id)
            self.next_track_id += 1
        self.tracks[key] = {"side": int(side),
                            "center": (float(door_center[0]), float(door_center[1])),
                            "desired": (float(desired[0]), float(desired[1])),
                            "seen": now}
        # Bound memory without deleting a currently visible aperture.
        self.tracks = {item_key: track for item_key, track in self.tracks.items()
                       if now - track["seen"] <= self.track_max_age}
        return key

    def _opening_candidates(self, grid, pose):
        # A1/RL can translate laterally while its body yaw barely changes.
        # Door geometry must therefore follow recent travel, not body yaw.
        with self.lock:
            trajectory = list(self.trajectory)
        axis = None
        current = np.asarray(pose[:2], dtype=float)
        for sample in reversed(trajectory[:-1]):
            delta = current - np.asarray(sample[1:3], dtype=float)
            distance = float(np.linalg.norm(delta))
            if distance >= 0.80:
                axis = delta / distance
                break
        if axis is None:
            axis = np.asarray(
                [math.cos(pose[3]), math.sin(pose[3])], dtype=float)
        candidates = []
        for side in (-1, 1):
            profiles = self._side_profiles(grid, pose, axis, side)
            groups, group = [], []
            for profile in profiles:
                if profile[1]:
                    group.append(profile)
                elif group:
                    groups.append(group); group = []
            if group:
                groups.append(group)
            for samples in groups:
                width = samples[-1][0] - samples[0][0] + grid.resolution
                if not (self.width_min <= width <= self.width_max):
                    continue
                middle = samples[len(samples) // 2]
                station, _, _, base, normal = middle
                wall_samples = [item[2] for item in profiles
                                if not item[1] and item[2] >= 0.30]
                if not wall_samples:
                    continue
                wall_depth = float(np.median(wall_samples))
                if not (0.35 <= wall_depth <= self.wall_search):
                    continue
                door_center = base + wall_depth * normal
                # Quantify mapped expansion behind the aperture. Both free and
                # unknown support room evidence; only known-free cells support
                # executable candidates.
                open_cells = unknown_cells = 0
                for longitudinal in np.arange(-1.6, 1.61, grid.resolution):
                    for depth in np.arange(0.8, self.behind_depth,
                                           grid.resolution):
                        value = self._value(
                            grid, door_center + longitudinal * axis + depth * normal)
                        if value is None:
                            continue
                        if value < 0:
                            unknown_cells += 1
                        elif value == 0:
                            open_cells += 1
                area = (open_cells + unknown_cells) * grid.resolution ** 2
                if area < self.minimum_open_area:
                    continue
                desired = None
                for depth in np.arange(
                        min(self.entry_depth, self.behind_depth - 0.2),
                        0.75, -max(0.10, grid.resolution)):
                    point = door_center + float(depth) * normal
                    if self._free_with_clearance(grid, point):
                        desired = (float(point[0]), float(point[1]))
                        break
                if desired is None:
                    continue
                if self.candidate_astar_enabled:
                    path = astar_safe_path(
                        grid, (pose[0], pose[1]), desired,
                        clearance_radius=self.clearance,
                        reached_tolerance=0.15, maximum_expansions=120000,
                        allow_blocked_start=True)
                    if not path.get("success"):
                        continue
                    path_length = sum(math.hypot(b[0] - a[0], b[1] - a[1])
                                      for a, b in zip(path["path"][:-1], path["path"][1:]))
                else:
                    # This is only a candidate.  The room scheduler performs
                    # portal A* and the manager runs final SCAN-lite before
                    # publishing motion, so repeating a full A* for every
                    # transient opening is redundant.
                    path_length = math.hypot(desired[0] - pose[0], desired[1] - pose[1])
                target_yaw = math.atan2(normal[1], normal[0])
                scan_safe, scan_status = self._scan_safe(
                    desired, float(pose[2]), target_yaw)
                key = self._track_key(side, door_center, desired)
                self.observations[key] += 1
                candidates.append({
                    "candidate_id": "local-door-%d-track-%d" % key,
                    "side": int(side), "width_m": round(width, 3),
                    "corridor_relative_station_m": round(station, 3),
                    "door_center": [round(float(door_center[0]), 4),
                                    round(float(door_center[1]), 4)],
                    "corridor_side": [
                        round(float(door_center[0] - 0.60 * normal[0]), 4),
                        round(float(door_center[1] - 0.60 * normal[1]), 4)],
                    "entry_goal": [round(desired[0], 4), round(desired[1], 4)],
                    "yaw": target_yaw, "open_area_m2": round(area, 3),
                    "unknown_behind_m2": round(
                        unknown_cells * grid.resolution ** 2, 3),
                    "free_behind_m2": round(
                        open_cells * grid.resolution ** 2, 3),
                    "astar_reachable": True,
                    "path_length_m": round(path_length, 3),
                    "scan_lite_safe": scan_safe,
                    "scan_lite_status": scan_status,
                    "confirmation_count": self.observations[key],
                    "confirmed": bool(
                        scan_safe and self.observations[key] >= self.confirmations),
                })
        return candidates

    def _tick(self, _event):
        with self.lock:
            pose, grid, sequence = self.pose, self.grid, self.grid_seq
            cloud_age = (time.monotonic() - self.cloud_stamp
                         if self.cloud_stamp else math.inf)
        if pose is None or grid is None or sequence == self.last_processed_grid:
            return
        self.last_processed_grid = sequence
        candidates = self._opening_candidates(grid, pose)
        confirmed = [item for item in candidates if item["confirmed"]]
        payload = {
            "timestamp": time.time(), "elapsed_sec": round(self.elapsed(), 3),
            "robot_pose": {"x": pose[0], "y": pose[1], "z": pose[2],
                           "yaw": pose[3]},
            "frame": "robot_local_geometry_in_fastlio_map",
            "window_size_m": self.window_size,
            "cloud_fresh": cloud_age <= 1.0,
            "local_unknown_area_m2": round(
                self._local_unknown_area(grid, pose), 3),
            "side_opening_count": len(candidates),
            "confirmed_count": len(confirmed),
            "candidates": candidates,
        }
        message = String(data=json.dumps(payload, sort_keys=True))
        self.door_pub.publish(message)
        self.entry_pub.publish(String(data=json.dumps({
            **payload, "candidates": confirmed,
            "room_entry_candidate_count": len(confirmed),
        }, sort_keys=True)))
        try:
            with open(self.log_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, sort_keys=True) + "\n")
        except OSError as error:
            rospy.logwarn_throttle(10.0, "local door log failed: %s", error)


if __name__ == "__main__":
    rospy.init_node("local_doorway_detector")
    LocalDoorwayDetector()
    rospy.spin()
