#!/usr/bin/env python3
"""Online first-floor skeleton patrol with one move_base/DWA control chain."""

import json
import math
import os
import sys
import threading
from enum import Enum

import actionlib
import numpy as np
import rospy
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import PointCloud
from std_msgs.msg import Bool, String
from std_srvs.srv import Empty

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from first_floor_core import (
    DoorClusterer,
    DoorObservation,
    astar_grid_length,
    bilateral_corridor_axis_delta,
    corridor_lookahead_s,
    extract_occupied_wall_gaps,
    must_return,
    occupancy_patch_traversable,
    pair_door_groups,
    return_time_budget,
    select_room_nbvs,
    select_projected_door_gap,
    should_update_rolling_goal,
    wall_center_error,
    wrap_angle,
    vertically_supported_obstacles,
)


def yaw_from_quaternion(quaternion):
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


class MissionState(Enum):
    INIT = "INIT"
    LOBBY_EGRESS = "LOBBY_EGRESS"
    CORRIDOR_DISCOVERY = "CORRIDOR_DISCOVERY"
    ROOM_FIRST_PASS = "ROOM_FIRST_PASS"
    ROOM_REVISIT = "ROOM_REVISIT"
    RETURN_HOME = "RETURN_HOME"
    FINALIZE = "FINALIZE"


class FirstFloorMissionManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._mission_limit = float(rospy.get_param("~mission_limit", 150.0))
        self._room_limit = float(rospy.get_param("~room_first_pass_limit", 14.0))
        self._maximum_rooms = int(rospy.get_param("~maximum_rooms", 4))
        self._egress_distance = float(rospy.get_param("~egress_probe_distance", 5.5))
        self._corridor_lookahead = float(rospy.get_param("~corridor_lookahead", 6.0))
        self._minimum_outbound = float(rospy.get_param("~minimum_outbound", 8.0))
        self._door_open_range = float(rospy.get_param("~door_open_range", 2.1))
        self._door_confirmations = int(rospy.get_param("~door_confirmations", 3))
        self._home_tolerance = float(rospy.get_param("~home_tolerance", 0.7))
        self._unpitch = float(rospy.get_param("~lidar_unpitch_rad", 0.785))
        self._result_file = rospy.get_param(
            "~room_status_file", os.path.join(os.getcwd(), "results", "room_status.json")
        )
        self._route_file = rospy.get_param(
            "~route_file", os.path.join(os.path.dirname(self._result_file), "topological_route.json")
        )

        self._started = rospy.Time.now()
        self._state = MissionState.INIT
        self._state_since = self._started
        self._route_phase = "INIT"
        self._speed_mode = "STOP"
        self._pose = None
        self._home = None
        self._corridor_yaw = None
        self._corridor_reference_yaw = None
        self._localization_healthy = False
        self._map = None
        self._map_info = None
        self._coverage = {}

        self._door_clusterer = DoorClusterer(0.8, 2.0, 1.2)
        self._corridor_hits = 0
        self._corridor_acquired = False
        self._corridor_start_s = None
        self._front_clearance = 20.0
        self._wall_center_error = 0.0
        self._wall_center_valid = False
        # Short-term online wall geometry.  Paired doors remove both current
        # side-wall returns, so doorway extraction must not require the wall
        # opposite the opening to remain visible in that same scan.
        self._corridor_half_width = {1: None, -1: None}
        self._furthest_s = 0.0
        self._best_frontier_s = 0.0
        self._frontier_advanced = self._started
        self._last_map_door_extract = rospy.Time(0)
        self._map_ready_since = rospy.Time(0)
        self._outbound_started = self._started
        self._turnaround_started = rospy.Time(0)
        self._door_groups = []
        self._group_index = 0

        self._visited = set()
        self._revisited = set()
        self._rooms = []
        self._current_door = None
        self._current_group = None
        self._room_phase = ""
        self._room_views = []
        self._room_started = rospy.Time(0)
        self._room_crossed = rospy.Time(0)
        self._room_deadline = rospy.Time(0)
        self._room_entry_deadline = rospy.Time(0)
        self._room_max_depth = 0.0
        self._scan_hold_until = rospy.Time(0)
        self._pending_room_advance = False
        self._unconfirmed_candidates = []

        self._goal = None
        self._goal_sent = rospy.Time(0)
        self._last_plan_query = rospy.Time(0)
        self._cached_home_path = 0.0
        self._last_state_publish = rospy.Time(0)
        self._last_forward_desired = self._started
        self._last_clear = rospy.Time(0)
        self._stall_retries = 0
        self._finalized = False
        self._finish_reason = ""
        self._route_points = []
        self._route_events = []

        self._client = actionlib.SimpleActionClient("/move_base", MoveBaseAction)
        self._clear_costmaps = rospy.ServiceProxy("/move_base/clear_costmaps", Empty)
        self._state_pub = rospy.Publisher(
            "/simenv/mission_state", String, queue_size=1, latch=True
        )
        self._route_pub = rospy.Publisher(
            "/simenv/topological_route", Path, queue_size=1, latch=True
        )
        self._finalize_pub = rospy.Publisher(
            "/simenv/finalize_result", Bool, queue_size=1, latch=True
        )
        rospy.Subscriber("/state_estimation", Odometry, self._on_pose, queue_size=20)
        rospy.Subscriber("/scan", PointCloud, self._on_scan, queue_size=2)
        rospy.Subscriber(
            "/simenv/first_floor_map", OccupancyGrid, self._on_map, queue_size=2
        )
        rospy.Subscriber(
            "/simenv/localization_healthy", Bool, self._on_health, queue_size=2
        )
        rospy.Subscriber(
            "/simenv/coverage_status", String, self._on_coverage, queue_size=2
        )
        rospy.Subscriber(
            "/simenv/danger_candidate", PointStamped, self._on_candidate, queue_size=10
        )
        rospy.Subscriber(
            "/simenv/desired_cmd_vel", Twist, self._on_desired, queue_size=20
        )
        rospy.Timer(rospy.Duration(0.20), self._on_timer)
        rospy.on_shutdown(self._write_status)
        self._publish_state(force=True)

    def _elapsed(self):
        return max(0.0, (rospy.Time.now() - self._started).to_sec())

    def _transition(self, state, reason=""):
        if self._state != state:
            self._state = state
            self._state_since = rospy.Time.now()
            rospy.loginfo("Mission state -> %s (%s)", state.value, reason)
        self._publish_state(force=True)

    def _set_route_phase(self, phase, speed_mode=None):
        if phase != self._route_phase:
            rospy.loginfo("Skeleton route phase -> %s", phase)
            self._route_phase = phase
        if speed_mode is not None:
            self._speed_mode = speed_mode
        self._publish_state(force=True)

    def _on_health(self, message):
        with self._lock:
            self._localization_healthy = bool(message.data)

    def _on_coverage(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        with self._lock:
            self._coverage = payload

    def _on_map(self, message):
        data = np.asarray(message.data, dtype=np.int8)
        expected = int(message.info.width) * int(message.info.height)
        if len(data) != expected:
            return
        with self._lock:
            self._map = data.reshape((message.info.height, message.info.width))
            self._map_info = message.info

    def _on_pose(self, message):
        position = message.pose.pose.position
        pose = (
            float(position.x),
            float(position.y),
            yaw_from_quaternion(message.pose.pose.orientation),
        )
        if not all(math.isfinite(value) for value in pose):
            return
        with self._lock:
            self._pose = pose
            if self._home is None:
                self._home = pose
                self._corridor_yaw = pose[2]
                self._corridor_reference_yaw = pose[2]
                self._record_route_point(*pose, "start")
            if self._current_door is not None and self._state in (
                MissionState.ROOM_FIRST_PASS,
                MissionState.ROOM_REVISIT,
            ):
                door = self._current_door
                depth = (pose[0] - door.center_x) * door.normal_x + (
                    pose[1] - door.center_y
                ) * door.normal_y
                self._room_max_depth = max(self._room_max_depth, depth)
                if depth >= 0.15 and self._room_crossed == rospy.Time(0):
                    self._room_crossed = rospy.Time.now()
                    self._room_deadline = self._room_crossed + rospy.Duration(self._room_limit)
                    self._route_events.append(
                        {"kind": "door_crossing", "x": pose[0], "y": pose[1], "t": self._elapsed()}
                    )

    def _on_candidate(self, message):
        with self._lock:
            point = (float(message.point.x), float(message.point.y))
            if not any(math.hypot(point[0] - old[0], point[1] - old[1]) < 0.6 for old in self._unconfirmed_candidates):
                self._unconfirmed_candidates.append(point)

    def _on_desired(self, message):
        if float(message.linear.x) > 0.02:
            with self._lock:
                self._last_forward_desired = rospy.Time.now()
                self._stall_retries = 0

    def _level_points(self, message):
        raw = np.asarray(
            [(point.x, point.y, point.z) for point in message.points], dtype=np.float64
        )
        if raw.size == 0:
            return raw.reshape((-1, 3))
        cosine, sine = math.cos(self._unpitch), math.sin(self._unpitch)
        levelled = raw.copy()
        levelled[:, 0] = cosine * raw[:, 0] + sine * raw[:, 2]
        levelled[:, 2] = -sine * raw[:, 0] + cosine * raw[:, 2]
        return levelled

    @staticmethod
    def _side_wall_distance(points, side):
        angle = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
        wanted = np.logical_and(side * angle >= 65.0, side * angle <= 115.0)
        samples = np.abs(points[wanted, 1])
        samples = samples[np.logical_and(samples >= 0.45, samples <= 2.20)]
        return float(np.median(samples)) if len(samples) >= 4 else None

    def _fit_corridor_axis(self, points, pose):
        local_angle = bilateral_corridor_axis_delta(points[:, :2])
        if local_angle is None:
            return
        candidate = wrap_angle(pose[2] + local_angle)
        # The public start points down the lobby/corridor.  Online wall fits
        # may refine that heading, but a lobby cross-wall must not create the
        # positive-feedback bend observed in the baseline.
        reference_error = wrap_angle(candidate - self._corridor_reference_yaw)
        candidate = wrap_angle(
            self._corridor_reference_yaw + max(-0.18, min(0.18, reference_error))
        )
        error = wrap_angle(candidate - self._corridor_yaw)
        self._corridor_yaw = wrap_angle(self._corridor_yaw + 0.12 * error)

    def _door_observation(self, points, side, pose):
        expected_wall = self._corridor_half_width.get(side)
        if expected_wall is not None and self._wall_center_valid:
            expected_wall = expected_wall - side * self._wall_center_error
        if expected_wall is None:
            # Only bootstrap from the opposite wall before this side has a
            # confirmed baseline; thereafter paired openings use memory.
            expected_wall = self._side_wall_distance(points, -side)
        if expected_wall is None:
            return None
        angle = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
        ranges = np.linalg.norm(points[:, :2], axis=1)
        sector = np.logical_and(side * angle >= 52.0, side * angle <= 128.0)
        candidates = points[np.logical_and(sector, ranges >= self._door_open_range)]
        if len(candidates) < 6:
            return None
        scale = (side * expected_wall) / candidates[:, 1]
        projected = candidates[:, 0] * scale
        projected = projected[np.logical_and(np.isfinite(projected), scale > 0.0)]
        gap = select_projected_door_gap(projected, 0.8, 2.0)
        if gap is None:
            return None
        local_x, width = gap
        local_y = float(side * expected_wall)
        x, y, yaw = pose
        cosine, sine = math.cos(yaw), math.sin(yaw)
        centre_x = x + cosine * local_x - sine * local_y
        centre_y = y + sine * local_x + cosine * local_y
        normal_yaw = self._corridor_yaw + side * math.pi / 2.0
        axis_x, axis_y = math.cos(self._corridor_yaw), math.sin(self._corridor_yaw)
        longitudinal = (centre_x - self._home[0]) * axis_x + (centre_y - self._home[1]) * axis_y
        return DoorObservation(
            longitudinal=longitudinal,
            side=side,
            width=width,
            center_x=centre_x,
            center_y=centre_y,
            normal_x=math.cos(normal_yaw),
            normal_y=math.sin(normal_yaw),
            corridor_half_width=expected_wall,
        )

    def _on_scan(self, message):
        if not message.points:
            return
        with self._lock:
            if self._state not in (MissionState.LOBBY_EGRESS, MissionState.CORRIDOR_DISCOVERY):
                return
            if self._pose is None or self._home is None or self._corridor_yaw is None:
                return
            pose = self._pose
        points = self._level_points(message)
        planar_range = np.linalg.norm(points[:, :2], axis=1)
        valid = (
            np.all(np.isfinite(points), axis=1)
            & (points[:, 2] >= -0.05)
            & (points[:, 2] <= 1.20)
            & (planar_range >= 0.82)
        )
        points = vertically_supported_obstacles(points[valid])
        if len(points) < 20:
            return
        forward = points[np.logical_and(points[:, 0] > 0.0, np.abs(points[:, 1]) < 0.42)]
        front_clearance = float(np.percentile(forward[:, 0], 5.0)) if len(forward) >= 4 else 20.0
        with self._lock:
            self._front_clearance = front_clearance
            left_wall = self._side_wall_distance(points, 1)
            right_wall = self._side_wall_distance(points, -1)
            corridor_like = (
                left_wall is not None
                and right_wall is not None
                and 0.45 <= left_wall <= 2.20
                and 0.45 <= right_wall <= 2.20
                and left_wall + right_wall <= 3.60
            )
            if corridor_like:
                self._corridor_hits += 1
                self._fit_corridor_axis(points, pose)
                # The map corridor axis is its centreline.  Store the common
                # half-width; individual robot-to-wall ranges include lateral
                # tracking error and would shift wall-gap sampling off-wall.
                measured_half_width = 0.5 * (left_wall + right_wall)
                for side in (1, -1):
                    previous = self._corridor_half_width[side]
                    self._corridor_half_width[side] = (
                        measured_half_width
                        if previous is None
                        else 0.92 * previous + 0.08 * measured_half_width
                    )
                measured_center = wall_center_error(left_wall, right_wall)
                if self._wall_center_valid:
                    self._wall_center_error = 0.75 * self._wall_center_error + 0.25 * measured_center
                else:
                    self._wall_center_error = measured_center
                    self._wall_center_valid = True
            else:
                self._corridor_hits = max(0, self._corridor_hits - 1)
            if not self._corridor_acquired and self._corridor_hits >= 3:
                self._corridor_acquired = True
                self._corridor_start_s = self._longitudinal(pose[0], pose[1])
                rospy.loginfo("Corridor axis acquired online at s=%.2f yaw=%.3f", self._corridor_start_s, self._corridor_yaw)
            if not self._corridor_acquired or self._route_phase != "OUTBOUND":
                return
            lateral = self._lateral(pose[0], pose[1])
            if abs(lateral) > 0.80 or abs(wrap_angle(pose[2] - self._corridor_yaw)) > 0.30:
                return
            for side in (1, -1):
                observation = self._door_observation(points, side, pose)
                if observation is None:
                    continue
                if self._corridor_start_s is not None and observation.longitudinal < self._corridor_start_s + 1.2:
                    continue
                door = self._door_clusterer.add(observation)
                if door is not None and door.observations == self._door_confirmations:
                    rospy.loginfo(
                        "Door confirmed online: s=%.2f side=%+d width=%.2f",
                        door.longitudinal,
                        door.side,
                        door.width,
                    )
                    self._record_route_point(door.center_x, door.center_y, math.atan2(door.normal_y, door.normal_x), "door")

    def _longitudinal(self, x, y):
        return (x - self._home[0]) * math.cos(self._corridor_yaw) + (y - self._home[1]) * math.sin(self._corridor_yaw)

    def _lateral(self, x, y):
        return -(x - self._home[0]) * math.sin(self._corridor_yaw) + (y - self._home[1]) * math.cos(self._corridor_yaw)

    def _corridor_xy(self, longitudinal):
        return (
            self._home[0] + longitudinal * math.cos(self._corridor_yaw),
            self._home[1] + longitudinal * math.sin(self._corridor_yaw),
        )

    def _corridor_goal_xy(self, longitudinal):
        x, y = self._corridor_xy(longitudinal)
        if not self._wall_center_valid:
            return x, y
        # Positive measured error means the robot is closer to the left wall.
        # Offset the goal to its right in the estimator frame until both wall
        # distances are symmetric; this remains fully onboard and corrects
        # lateral LIO dropout without moving the map frame.
        correction = max(-0.65, min(0.65, -self._wall_center_error))
        lateral_x, lateral_y = -math.sin(self._corridor_yaw), math.cos(self._corridor_yaw)
        return x + correction * lateral_x, y + correction * lateral_y

    def _map_value(self, x, y):
        if self._map is None or self._map_info is None:
            return -1
        origin = self._map_info.origin.position
        resolution = self._map_info.resolution
        ix = int(math.floor((x - origin.x) / resolution))
        iy = int(math.floor((y - origin.y) / resolution))
        if 0 <= ix < self._map.shape[1] and 0 <= iy < self._map.shape[0]:
            return int(self._map[iy, ix])
        return -1

    def _extract_map_doors(self, now):
        if (
            self._map is None
            or self._map_info is None
            or self._corridor_start_s is None
            or (now - self._last_map_door_extract).to_sec() < 0.8
        ):
            return
        self._last_map_door_extract = now
        # Re-scan the full corridor behind the robot.  The shallow door band
        # may be traversed before bilateral walls have accumulated the three
        # frames required to declare the corridor acquired.
        start = 1.0
        stop = self._furthest_s + 0.45
        step = max(0.10, float(self._map_info.resolution))
        if stop - start < 1.0:
            return
        stations = np.arange(start, stop + 0.5 * step, step)
        lateral_x, lateral_y = -math.sin(self._corridor_yaw), math.cos(self._corridor_yaw)
        for side in (1, -1):
            half_width = self._corridor_half_width.get(side)
            if half_width is None:
                continue
            wall_values, beyond_values = [], []
            for station in stations:
                centre_x, centre_y = self._corridor_xy(float(station))
                wall_band = [
                    self._map_value(
                        centre_x + side * (half_width + radial) * lateral_x,
                        centre_y + side * (half_width + radial) * lateral_y,
                    )
                    for radial in (-0.18, 0.0, 0.18)
                ]
                wall_values.append(
                    100 if 100 in wall_band else (0 if 0 in wall_band else -1)
                )
                beyond_values.append(
                    self._map_value(
                        centre_x + side * (half_width + 0.60) * lateral_x,
                        centre_y + side * (half_width + 0.60) * lateral_y,
                    )
                )
            for station, width in extract_occupied_wall_gaps(
                stations, wall_values, beyond_values, 0.8, 2.0
            ):
                centre_x, centre_y = self._corridor_xy(station)
                centre_x += side * half_width * lateral_x
                centre_y += side * half_width * lateral_y
                normal_yaw = self._corridor_yaw + side * math.pi / 2.0
                door = self._door_clusterer.add(
                    DoorObservation(
                        longitudinal=station,
                        side=side,
                        width=width,
                        center_x=centre_x,
                        center_y=centre_y,
                        normal_x=math.cos(normal_yaw),
                        normal_y=math.sin(normal_yaw),
                        corridor_half_width=half_width,
                    )
                )
                if door is not None and door.observations == self._door_confirmations:
                    rospy.loginfo(
                        "Door confirmed from online wall gap: s=%.2f side=%+d width=%.2f",
                        door.longitudinal,
                        door.side,
                        door.width,
                    )
                    self._record_route_point(
                        door.center_x,
                        door.center_y,
                        math.atan2(door.normal_y, door.normal_x),
                        "door",
                    )

    def _cell_patch_free(self, x, y, radius=0.34):
        if self._map is None or self._map_info is None:
            return False
        resolution = self._map_info.resolution
        origin = self._map_info.origin.position
        ix = int(math.floor((x - origin.x) / resolution))
        iy = int(math.floor((y - origin.y) / resolution))
        cells = max(1, int(math.ceil(radius / resolution)))
        if ix - cells < 0 or iy - cells < 0 or ix + cells >= self._map.shape[1] or iy + cells >= self._map.shape[0]:
            return False
        patch = self._map[iy - cells : iy + cells + 1, ix - cells : ix + cells + 1]
        centre = int(self._map[iy, ix])
        # Ray maps naturally retain a thin unknown fringe between beams.  A
        # traversable footprint needs a free centre, no persistent obstacle,
        # and substantial free evidence; requiring every raster cell to be
        # observed prevented any rolling goal from being emitted.
        return occupancy_patch_traversable(patch, centre, 0.65)

    def _corridor_free_frontier(self, direction=1):
        current_s = self._longitudinal(self._pose[0], self._pose[1])
        step = max(0.15, float(self._map_info.resolution)) if self._map_info is not None else 0.15
        last = current_s
        blocked = 0
        for offset in np.arange(step, 20.0 + step, step):
            candidate_s = current_s + direction * float(offset)
            x, y = self._corridor_goal_xy(candidate_s)
            if self._cell_patch_free(x, y, 0.30):
                last = candidate_s
                blocked = 0
            else:
                blocked += 1
                if blocked >= 2:
                    break
        return last

    @staticmethod
    def _pose_stamped(x, y, yaw):
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "map"
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.orientation.z = math.sin(float(yaw) * 0.5)
        pose.pose.orientation.w = math.cos(float(yaw) * 0.5)
        return pose

    def _send_goal(self, x, y, yaw, purpose, timeout, xy_tolerance=0.35, yaw_tolerance=0.25, rolling=False):
        now = rospy.Time.now()
        if rolling and self._goal is not None and self._goal.get("purpose") == purpose:
            moved = math.hypot(float(x) - self._goal["x"], float(y) - self._goal["y"])
            yaw_change = abs(wrap_angle(float(yaw) - self._goal["yaw"]))
            remaining = math.hypot(self._goal["x"] - self._pose[0], self._goal["y"] - self._pose[1])
            # Keep one DWA plan active while its target remains well ahead.
            # Replanning every 0.5 s reset the acceleration ramp and prevented
            # the robot from ever reaching the selected 2.0 m/s tier.
            if not should_update_rolling_goal(remaining, moved, yaw_change):
                return False
        goal = MoveBaseGoal()
        goal.target_pose = self._pose_stamped(x, y, yaw)
        self._client.send_goal(goal)
        self._goal = {
            "x": float(x), "y": float(y), "yaw": float(yaw), "purpose": purpose,
            "deadline": now + rospy.Duration(max(1.0, float(timeout))),
            "xy_tolerance": float(xy_tolerance), "yaw_tolerance": float(yaw_tolerance),
            "rolling": bool(rolling),
        }
        self._goal_sent = now
        self._record_route_point(x, y, yaw, purpose.lower())
        rospy.loginfo("Goal %s -> (%.2f, %.2f, %.2f)", purpose, x, y, yaw)
        return True

    def _goal_result(self, now):
        if self._goal is None or self._pose is None:
            return None
        distance = math.hypot(self._pose[0] - self._goal["x"], self._pose[1] - self._goal["y"])
        yaw_error = abs(wrap_angle(self._goal["yaw"] - self._pose[2]))
        if distance <= self._goal["xy_tolerance"] and yaw_error <= self._goal["yaw_tolerance"]:
            return True
        if now >= self._goal["deadline"]:
            return False
        status = self._client.get_state()
        if status == GoalStatus.SUCCEEDED:
            return True
        if status in (GoalStatus.ABORTED, GoalStatus.REJECTED, GoalStatus.LOST) and (now - self._goal_sent).to_sec() > 1.0:
            return False
        return None

    def _consume_goal(self):
        goal = self._goal
        self._goal = None
        return goal

    def _is_safe(self, x, y):
        return self._cell_patch_free(x, y, 0.45)

    def _confirmed_doors(self):
        return [door for door in self._door_clusterer.doors if door.observations >= self._door_confirmations]

    def _send_outbound_goal(self):
        current_s = self._longitudinal(self._pose[0], self._pose[1])
        frontier_s = self._corridor_free_frontier(1)
        if frontier_s > self._best_frontier_s + 0.30:
            self._best_frontier_s = frontier_s
            self._frontier_advanced = rospy.Time.now()
        target_s = corridor_lookahead_s(current_s, frontier_s, self._corridor_lookahead)
        if target_s is None:
            # A directly observed forward ray is also confirmed free space.
            # Use a short braking-margin probe to traverse a local raster
            # speckle; never send past the current sensor free frontier.
            if self._front_clearance < 2.0:
                return False
            target_s = current_s + min(2.5, self._front_clearance - 0.8)
        x, y = self._corridor_goal_xy(target_s)
        return self._send_goal(x, y, self._corridor_yaw, "CORRIDOR_OUTBOUND", 8.0, 0.45, 0.25, rolling=True)

    def _outbound_complete(self, now):
        confirmed = self._confirmed_doors()
        groups = pair_door_groups(confirmed, minimum_observations=self._door_confirmations)
        progress = self._furthest_s - (self._corridor_start_s or 0.0)
        all_rooms_seen = len(confirmed) >= self._maximum_rooms and len(groups) >= 2
        ended = self._front_clearance < 1.25
        return all_rooms_seen or (
            progress >= self._minimum_outbound and len(confirmed) >= 2 and ended
        )

    def _start_turnaround(self, reason):
        self._client.cancel_all_goals()
        self._goal = None
        self._turnaround_started = rospy.Time.now()
        self._set_route_phase("TURNAROUND", "TURN_IN_PLACE")
        self._route_events.append({"kind": "turnaround", "x": self._pose[0], "y": self._pose[1], "t": self._elapsed()})
        self._send_goal(
            self._pose[0], self._pose[1], wrap_angle(self._corridor_yaw + math.pi),
            "CORRIDOR_TURNAROUND", 5.0, 0.25, 0.12,
        )
        rospy.loginfo("Outbound complete: %s doors=%d", reason, len(self._confirmed_doors()))

    def _start_return_sweep(self):
        self._door_groups = pair_door_groups(
            self._confirmed_doors(), minimum_observations=self._door_confirmations
        )
        self._group_index = 0
        self._goal = None
        self._set_route_phase("RETURN_SWEEP", "CORRIDOR_CRUISE")
        self._send_return_sweep_goal()

    def _next_sweep_door(self):
        while self._group_index < len(self._door_groups):
            group = self._door_groups[self._group_index]
            for door in group.doors:
                if id(door) not in self._visited:
                    return group, door
            self._group_index += 1
        return None, None

    def _send_return_sweep_goal(self):
        group, door = self._next_sweep_door()
        if door is None:
            self._after_first_pass()
            return
        self._current_group = group
        current_s = self._longitudinal(self._pose[0], self._pose[1])
        if abs(current_s - group.longitudinal) <= 1.15 and abs(self._lateral(self._pose[0], self._pose[1])) <= 0.75:
            self._begin_room(door)
            return
        direction = -1.0 if current_s > group.longitudinal else 1.0
        target_s = current_s + direction * min(5.5, abs(current_s - group.longitudinal))
        x, y = self._corridor_goal_xy(target_s)
        yaw = wrap_angle(self._corridor_yaw + (math.pi if direction < 0.0 else 0.0))
        self._speed_mode = "CORRIDOR_CRUISE"
        self._send_goal(x, y, yaw, "CORRIDOR_RETURN_SWEEP", 8.0, 0.45, 0.28, rolling=True)

    def _begin_room(self, door, revisit=False):
        self._client.cancel_all_goals()
        self._goal = None
        views = select_room_nbvs(door, self._is_safe, entry_depth=2.7, baseline=1.25)
        if len(views) < 2:
            rospy.logwarn("No two safe sweep views for door s=%.2f side=%+d", door.longitudinal, door.side)
            self._visited.add(id(door))
            self._rooms.append({
                "door_longitudinal_m": round(door.longitudinal, 3), "side": int(door.side),
                "entered": False, "viewpoint_count": 0, "viewpoint_baseline_m": 0.0,
                "max_penetration_m": 0.0, "duration_sec": 0.0, "revisit": revisit,
                "exit_reason": "no_safe_sweep_views",
            })
            self._resume_return_sweep()
            return
        self._current_door = door
        self._room_views = views
        self._room_started = rospy.Time.now()
        self._room_crossed = rospy.Time(0)
        self._room_deadline = rospy.Time(0)
        self._room_entry_deadline = self._room_started + rospy.Duration(9.0)
        self._room_max_depth = 0.0
        self._room_phase = "VIEW0"
        self._set_route_phase("ROOM_REVISIT" if revisit else "ROOM_FIRST_PASS", "ROOM")
        self._transition(MissionState.ROOM_REVISIT if revisit else MissionState.ROOM_FIRST_PASS, "paired_return_sweep")
        inward = math.atan2(door.normal_y, door.normal_x)
        x, y = views[0]
        self._send_goal(x, y, inward - math.pi / 3.0, "ROOM_VIEW0", 6.0, 0.38, 0.18)

    def _advance_room(self, _succeeded=True):
        door = self._current_door
        if door is None:
            self._resume_return_sweep()
            return
        inward = math.atan2(door.normal_y, door.normal_x)
        phase = self._room_phase
        if phase == "VIEW0":
            self._room_phase = "SCAN0"
            self._speed_mode = "SCAN"
            x, y = self._room_views[0]
            self._record_scan_event(x, y, 0)
            self._send_goal(x, y, inward + math.pi / 3.0, "ROOM_SCAN0", 4.0, 0.20, 0.10)
        elif phase == "SCAN0":
            self._room_phase = "VIEW1"
            self._speed_mode = "ROOM"
            x, y = self._room_views[1]
            self._send_goal(x, y, inward + math.pi / 3.0, "ROOM_VIEW1", 4.0, 0.38, 0.18)
        elif phase == "VIEW1":
            self._room_phase = "SCAN1"
            self._speed_mode = "SCAN"
            x, y = self._room_views[1]
            self._record_scan_event(x, y, 1)
            self._send_goal(x, y, inward - math.pi / 3.0, "ROOM_SCAN1", 4.0, 0.20, 0.10)
        elif phase == "SCAN1":
            self._begin_room_exit("views_complete")
        else:
            self._begin_room_exit("phase_complete")

    def _record_scan_event(self, x, y, index):
        self._route_events.append({
            "kind": "scan", "view_index": int(index), "x": float(x), "y": float(y),
            "sweep_deg": 180.0, "t": self._elapsed(),
        })
        self._record_route_point(x, y, self._pose[2], "scan")

    def _begin_room_exit(self, reason):
        if self._current_door is None:
            self._resume_return_sweep()
            return
        self._client.cancel_all_goals()
        self._goal = None
        self._room_phase = "EXIT_TURN"
        self._speed_mode = "TURN_IN_PLACE"
        outward = wrap_angle(math.atan2(self._current_door.normal_y, self._current_door.normal_x) + math.pi)
        self._send_goal(self._pose[0], self._pose[1], outward, "ROOM_EXIT_TURN", 3.5, 0.22, 0.12)
        rospy.loginfo("Room exit: %s depth=%.2f", reason, self._room_max_depth)

    def _drive_room_exit(self):
        door = self._current_door
        self._room_phase = "EXIT_DRIVE"
        self._speed_mode = "ROOM"
        exit_x, exit_y = self._corridor_goal_xy(door.longitudinal)
        outward = wrap_angle(math.atan2(door.normal_y, door.normal_x) + math.pi)
        self._send_goal(exit_x, exit_y, outward, "ROOM_EXIT_DRIVE", 8.0, 0.48, 0.30)

    def _finish_room(self, reason):
        door = self._current_door
        revisit = self._state == MissionState.ROOM_REVISIT
        report = {
            "door_longitudinal_m": round(door.longitudinal, 3),
            "side": int(door.side),
            "door_width_m": round(door.width, 3),
            "max_penetration_m": round(self._room_max_depth, 3),
            "entered": self._room_max_depth >= 2.5,
            "viewpoint_count": len(self._room_views),
            "viewpoint_baseline_m": round(math.dist(self._room_views[0], self._room_views[1]), 3),
            "first_pass_duration_sec": round((rospy.Time.now() - (self._room_crossed if self._room_crossed != rospy.Time(0) else self._room_started)).to_sec(), 3),
            "duration_sec": round((rospy.Time.now() - self._room_started).to_sec(), 3),
            "revisit": revisit,
            "exit_reason": reason,
        }
        self._rooms.append(report)
        (self._revisited if revisit else self._visited).add(id(door))
        self._current_door = None
        self._room_phase = ""
        self._room_views = []
        self._pending_room_advance = False
        self._write_status()
        self._resume_return_sweep()

    def _resume_return_sweep(self):
        if len(self._visited) >= self._maximum_rooms:
            self._after_first_pass()
            return
        self._transition(MissionState.CORRIDOR_DISCOVERY, "room_exit")
        self._set_route_phase("RETURN_SWEEP", "CORRIDOR_CRUISE")
        self._send_return_sweep_goal()

    def _candidate_revisit_door(self):
        best = None
        best_score = float("inf")
        for door in self._confirmed_doors():
            if id(door) not in self._visited or id(door) in self._revisited:
                continue
            for x, y in self._unconfirmed_candidates:
                depth = (x - door.center_x) * door.normal_x + (y - door.center_y) * door.normal_y
                lateral = abs((x - door.center_x) * (-door.normal_y) + (y - door.center_y) * door.normal_x)
                if depth > 0.0 and lateral < 7.5 and lateral < best_score:
                    best, best_score = door, lateral
        return best

    def _after_first_pass(self):
        door = self._candidate_revisit_door()
        path = self._home_path_length()
        enough = self._mission_limit - self._elapsed() > return_time_budget(path) + 13.0
        coverage = float(self._coverage.get("camera_coverage_pct", 0.0))
        if door is not None and enough and coverage < 65.0:
            self._begin_room(door, revisit=True)
            return
        self._start_return("first_pass_complete" if coverage >= 65.0 else "first_pass_complete_no_high_gain_revisit")

    def _home_path_length(self):
        if self._pose is None or self._home is None:
            return 0.0
        now = rospy.Time.now()
        if (now - self._last_plan_query).to_sec() < 1.0 and self._cached_home_path > 0.0:
            return self._cached_home_path
        self._last_plan_query = now
        euclidean = math.hypot(self._pose[0] - self._home[0], self._pose[1] - self._home[1])
        if self._map is None or self._map_info is None:
            return euclidean
        origin = self._map_info.origin.position
        resolution = self._map_info.resolution
        start = (int((self._pose[0] - origin.x) // resolution), int((self._pose[1] - origin.y) // resolution))
        goal = (int((self._home[0] - origin.x) // resolution), int((self._home[1] - origin.y) // resolution))
        length = astar_grid_length(self._map, start, goal, resolution)
        self._cached_home_path = max(euclidean, length) if math.isfinite(length) else euclidean
        return self._cached_home_path

    def _start_return(self, reason):
        if self._state in (MissionState.RETURN_HOME, MissionState.FINALIZE):
            return
        self._client.cancel_all_goals()
        self._goal = None
        self._finish_reason = reason
        self._set_route_phase("RETURN_HOME", "CORRIDOR_CRUISE")
        self._transition(MissionState.RETURN_HOME, reason)
        remaining = max(2.0, self._mission_limit - self._elapsed() - 1.0)
        self._send_goal(*self._home, "RETURN_HOME", remaining, self._home_tolerance, 0.40)

    def _finalize(self, reason):
        if self._finalized:
            return
        self._client.cancel_all_goals()
        self._goal = None
        self._finish_reason = reason
        self._set_route_phase("FINALIZE", "STOP")
        self._transition(MissionState.FINALIZE, reason)
        self._finalized = True
        self._write_status()
        self._finalize_pub.publish(Bool(data=True))

    def _recover_stall(self, now):
        if self._goal is None or self._speed_mode in ("SCAN", "TURN_IN_PLACE", "STOP"):
            return False
        distance = math.hypot(self._goal["x"] - self._pose[0], self._goal["y"] - self._pose[1])
        if distance < 0.9 or (now - self._last_forward_desired).to_sec() <= 1.5:
            return False
        self._stall_retries += 1
        self._corridor_lookahead = max(2.0, self._corridor_lookahead - 1.0)
        self._client.cancel_all_goals()
        self._goal = None
        if (now - self._last_clear).to_sec() > 2.5:
            try:
                self._clear_costmaps()
            except rospy.ServiceException:
                pass
            self._last_clear = now
        self._last_forward_desired = now
        rospy.logwarn("DWA forward stall: retry=%d lookahead=%.1f", self._stall_retries, self._corridor_lookahead)
        if self._stall_retries >= 3 and self._state in (MissionState.ROOM_FIRST_PASS, MissionState.ROOM_REVISIT):
            self._begin_room_exit("dwa_stall")
        return True

    def _record_route_point(self, x, y, yaw, kind):
        point = {"x": float(x), "y": float(y), "yaw": float(yaw), "kind": str(kind), "t": self._elapsed()}
        if self._route_points:
            last = self._route_points[-1]
            if kind not in ("scan", "door", "start") and math.hypot(point["x"] - last["x"], point["y"] - last["y"]) < 0.35:
                return
        self._route_points.append(point)
        self._publish_route()

    def _publish_route(self):
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = "map"
        path.poses = [self._pose_stamped(item["x"], item["y"], item["yaw"]) for item in self._route_points]
        self._route_pub.publish(path)

    def _write_route(self):
        payload = {
            "schema": "simenv_topological_route_v1", "online_only": True,
            "route_phase": self._route_phase, "points": list(self._route_points),
            "events": list(self._route_events),
            "doors": [
                {"s": round(door.longitudinal, 3), "side": int(door.side), "x": round(door.center_x, 3),
                 "y": round(door.center_y, 3), "observations": int(door.observations)}
                for door in self._confirmed_doors()
            ],
        }
        os.makedirs(os.path.dirname(os.path.abspath(self._route_file)), exist_ok=True)
        temporary = self._route_file + ".tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, self._route_file)

    def _publish_state(self, force=False):
        now = rospy.Time.now()
        if not force and (now - self._last_state_publish).to_sec() < 0.2:
            return
        self._last_state_publish = now
        goal = self._goal or {}
        yaw_error = wrap_angle(goal.get("yaw", self._pose[2] if self._pose else 0.0) - (self._pose[2] if self._pose else 0.0))
        goal_distance = math.hypot(goal.get("x", self._pose[0] if self._pose else 0.0) - (self._pose[0] if self._pose else 0.0), goal.get("y", self._pose[1] if self._pose else 0.0) - (self._pose[1] if self._pose else 0.0))
        if self._wall_center_valid:
            center_error = self._wall_center_error
        else:
            center_error = self._lateral(self._pose[0], self._pose[1]) if self._pose and self._home and self._corridor_yaw is not None else 0.0
        payload = {
            "schema": "simenv_first_floor_mission_v3", "state": self._state.value,
            "route_phase": self._route_phase, "speed_mode": self._speed_mode,
            "room_phase": self._room_phase, "elapsed_sec": round(self._elapsed(), 3),
            "rooms_confirmed": len(self._confirmed_doors()), "door_groups": len(self._door_groups),
            "rooms_first_passed": len(self._visited),
            "rooms_entered": sum(1 for room in self._rooms if room.get("entered") and not room.get("revisit")),
            "in_room": self._state in (MissionState.ROOM_FIRST_PASS, MissionState.ROOM_REVISIT),
            "near_door": self._room_phase in ("VIEW0", "EXIT_DRIVE"),
            "sharp_turn": self._speed_mode in ("SCAN", "TURN_IN_PLACE") or abs(yaw_error) >= 0.32,
            "yaw_error_rad": round(yaw_error, 4), "center_error_m": round(center_error, 4),
            "goal_distance_m": round(goal_distance, 3), "corridor_acquired": self._corridor_acquired,
            "wall_center_error_m": round(self._wall_center_error, 4) if self._wall_center_valid else None,
            "route_progress": round(self._furthest_s, 3),
            "scan_heading": round(goal.get("yaw", 0.0), 4) if self._speed_mode == "SCAN" else None,
            "current_door_group": round(self._current_group.longitudinal, 3) if self._current_group else None,
            "localization_healthy": self._localization_healthy, "finish_reason": self._finish_reason,
        }
        self._state_pub.publish(String(data=json.dumps(payload, sort_keys=True)))

    def _write_status(self):
        home_distance = None
        if self._pose is not None and self._home is not None:
            home_distance = math.hypot(self._pose[0] - self._home[0], self._pose[1] - self._home[1])
        payload = {
            "schema": "simenv_first_floor_room_status_v3", "online_only": True,
            "mission_state": self._state.value, "route_phase": self._route_phase,
            "elapsed_sec": round(self._elapsed(), 3), "finish_reason": self._finish_reason,
            "door_count": len(self._confirmed_doors()), "door_group_count": len(self._door_groups),
            "first_pass_count": len(self._visited),
            "entered_count": sum(1 for room in self._rooms if room.get("entered") and not room.get("revisit")),
            "home_distance_m": round(home_distance, 3) if home_distance is not None else None,
            "camera_coverage_pct": self._coverage.get("camera_coverage_pct"), "rooms": list(self._rooms),
        }
        os.makedirs(os.path.dirname(os.path.abspath(self._result_file)), exist_ok=True)
        temporary = self._result_file + ".tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, self._result_file)
        self._write_route()

    def _on_timer(self, _event):
        with self._lock:
            now = rospy.Time.now()
            self._publish_state()
            if self._finalized or self._pose is None or self._home is None:
                return
            elapsed = self._elapsed()
            if elapsed >= self._mission_limit - 0.5:
                distance = math.hypot(self._pose[0] - self._home[0], self._pose[1] - self._home[1])
                self._finalize("deadline_home" if distance <= self._home_tolerance else "deadline_away_from_home")
                return
            if self._state not in (MissionState.INIT, MissionState.RETURN_HOME, MissionState.FINALIZE):
                if must_return(elapsed, self._mission_limit, self._home_path_length()):
                    self._start_return("dynamic_return_budget")
                    return

            if self._state == MissionState.INIT:
                if self._map is None or not self._localization_healthy or not self._client.wait_for_server(rospy.Duration(0.02)):
                    return
                current_s = self._longitudinal(self._pose[0], self._pose[1])
                initial_frontier = self._corridor_free_frontier(1)
                if initial_frontier - current_s < 2.0:
                    self._map_ready_since = rospy.Time(0)
                    return
                if self._map_ready_since == rospy.Time(0):
                    self._map_ready_since = now
                    return
                if (now - self._map_ready_since).to_sec() < 1.0:
                    return
                self._transition(MissionState.LOBBY_EGRESS, "localization_and_map_ready")
                self._set_route_phase("LOBBY_EGRESS", "CORRIDOR_CRUISE")
                self._outbound_started = now
                self._send_outbound_goal()
                return

            if self._recover_stall(now):
                return

            if self._state == MissionState.LOBBY_EGRESS:
                progress = self._longitudinal(self._pose[0], self._pose[1])
                if self._corridor_acquired or progress >= self._egress_distance or (now - self._outbound_started).to_sec() >= 12.0:
                    self._transition(MissionState.CORRIDOR_DISCOVERY, "corridor_reached")
                    self._set_route_phase("OUTBOUND", "CORRIDOR_CRUISE")
                    self._corridor_start_s = progress if self._corridor_start_s is None else self._corridor_start_s
                    self._frontier_advanced = now
                self._send_outbound_goal()
                return

            result = self._goal_result(now)
            if self._state == MissionState.CORRIDOR_DISCOVERY:
                if self._route_phase == "OUTBOUND":
                    current_s = self._longitudinal(self._pose[0], self._pose[1])
                    self._furthest_s = max(self._furthest_s, current_s)
                    self._extract_map_doors(now)
                    if self._outbound_complete(now):
                        self._start_turnaround("doors_and_end_confirmed")
                        return
                    if result is not None:
                        self._consume_goal()
                    self._send_outbound_goal()
                    return
                if self._route_phase == "TURNAROUND":
                    if result is not None:
                        self._consume_goal()
                        self._start_return_sweep()
                    return
                if self._route_phase == "RETURN_SWEEP":
                    group, door = self._next_sweep_door()
                    if door is None:
                        self._after_first_pass()
                        return
                    current_s = self._longitudinal(self._pose[0], self._pose[1])
                    if abs(current_s - group.longitudinal) <= 1.15 and abs(self._lateral(self._pose[0], self._pose[1])) <= 0.75:
                        self._begin_room(door)
                        return
                    if result is not None:
                        self._consume_goal()
                    self._send_return_sweep_goal()
                    return

            if self._state in (MissionState.ROOM_FIRST_PASS, MissionState.ROOM_REVISIT):
                if (
                    not self._room_phase.startswith("EXIT")
                    and self._room_crossed == rospy.Time(0)
                    and now >= self._room_entry_deadline
                ):
                    self._begin_room_exit("entry_timeout")
                    return
                if self._room_deadline != rospy.Time(0) and now >= self._room_deadline and not self._room_phase.startswith("EXIT"):
                    self._begin_room_exit("room_time_budget")
                    return
                if self._pending_room_advance and now >= self._scan_hold_until:
                    self._pending_room_advance = False
                    self._advance_room(True)
                    return
                if result is not None:
                    self._consume_goal()
                    if self._room_phase in ("SCAN0", "SCAN1"):
                        self._pending_room_advance = True
                        self._scan_hold_until = now + rospy.Duration(0.20)
                    elif self._room_phase == "EXIT_TURN":
                        self._drive_room_exit()
                    elif self._room_phase == "EXIT_DRIVE":
                        self._finish_room("exit_reached" if result else "exit_timeout")
                    else:
                        self._advance_room(result)
                return

            if self._state == MissionState.RETURN_HOME:
                distance = math.hypot(self._pose[0] - self._home[0], self._pose[1] - self._home[1])
                if distance <= self._home_tolerance:
                    self._finalize("home_reached")
                elif result is False:
                    self._consume_goal()
                    remaining = max(1.0, self._mission_limit - elapsed - 0.5)
                    self._send_goal(*self._home, "RETURN_HOME_RETRY", remaining, self._home_tolerance, 0.40)


if __name__ == "__main__":
    rospy.init_node("first_floor_mission_manager")
    FirstFloorMissionManager()
    rospy.spin()
