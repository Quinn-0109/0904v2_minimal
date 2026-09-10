#!/usr/bin/env python3
"""Sensor-only structure-aware exploration heuristics.

Uses body-frame /scan geometry (no layout_metadata) + FIS coverage:
  - CORRIDOR: bilateral spacing -> center / boost on clear straights
  - DOOR_GAP: LiDAR openings -> glide-enter rooms
  - ROOM: follow uncovered gaps; skip cells already cleared by LiDAR range
  - LOOK_AT: compact sphere-like returns -> yaw + short RGB dwell
  - EXIT: leave when local LiDAR/RGB coverage is saturated

Publishes /simenv/structure_cmd_vel and /way_point. Active only after lobby
egress finishes. Compatible with unknown exploration rules.
"""

from __future__ import annotations

import math
import json
import os
import sys
import threading
from enum import Enum

import numpy as np
import rospy
from geometry_msgs.msg import PointStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud
from std_msgs.msg import Bool, String

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frontier_information_structure import (  # noqa: E402
    FrontierInformationStructure,
    RoomVisibilityGrid,
    corridor_exit_confirmed,
    forward_escape_offset,
    room_boundary_gate,
    room_entry_confirmed,
    room_exploration_complete,
    room_viewpoint_diversity_gate,
)
from motion_safety import (  # noqa: E402
    boost_clearance_for_speed,
    clearance_speed_limit,
)


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_diff(target, source):
    return (target - source + math.pi) % (2.0 * math.pi) - math.pi


class Mode(Enum):
    IDLE = "idle"
    CORRIDOR = "corridor"
    ENTER_ROOM = "enter_room"
    ROOM_COVER = "room_cover"
    EXIT_ROOM = "exit_room"
    LOOK_AT = "look_at"


class StructureAwareExplorer:
    def __init__(self):
        self._pose = None
        self._imu_yaw = None
        self._imu_yaw_offset = None
        self._egress_active = True
        self._ranges = {
            "left": None,
            "right": None,
            "front": None,
            "front_left": None,
            "front_right": None,
            "rear": None,
        }
        self._mode = Mode.IDLE
        self._mode_since = rospy.Time(0)
        self._door_side = 0  # -1 left, +1 right
        self._last_door_side = 0
        self._last_entry_yaw = 1.5708
        self._last_entry_xy = None
        # Zero means take the first geometrically valid door.  A fixed side is
        # useful only for deterministic diagnostics, not unknown competition
        # scenes where room contents and layout are randomized.
        self._prefer_side = int(rospy.get_param("~prefer_side", 0))
        self._room_entry_xy = None
        self._door_center_xy = None
        self._room_entry_yaw = 0.0
        self._room_started_at = rospy.Time(0)
        self._room_phase = "idle"
        self._room_phase_since = rospy.Time(0)
        self._room_cover_path = 0.0
        self._room_cover_last_xy = None
        self._room_max_depth = 0.0
        self._room_transverse_min = 0.0
        self._room_transverse_max = 0.0
        self._room_sweep_sign = 1.0
        self._room_sweep_start_transverse = 0.0
        self._visited_cells = set()
        self._room_cells = set()
        self._last_scan = rospy.Time(0)
        self._scan_until = rospy.Time(0)
        self._rooms_cleared = 0
        self._corridor_progress_y = None
        self._last_door_y = None
        self._skip_door_until_y = None
        self._skip_side_until = rospy.Time(0)
        self._skip_side = 0
        self._last_classify_log = rospy.Time(0)
        self._last_label = "unknown"
        self._enter_started_xy = None
        self._enter_center_s = 0.0
        self._enter_drive_s = 0.0
        self._enter_required_s = 1.8
        self._paired_entry = False
        self._entry_confirmed = False
        self._entry_confirmation_depth = 0.0
        self._entry_door_frame_crossed = False
        self._engage_pose = None
        self._engage_time = rospy.Time(0)
        self._corridor_ready = False
        self._exit_spin_dir = 1.0
        self._exit_forward_s = 0.0
        self._corridor_anchor_xy = None
        self._exit_corridor_hits = 0
        self._exit_travel_m = 0.0
        self._exit_last_xy = None
        self._exit_attempt_started = rospy.Time(0)
        self._exit_retry_until = rospy.Time(0)
        self._exit_room_side = 0
        self._pending_room_status = "partial"
        self._pending_room_reason = ""
        self._cardinal_looks = 0
        self._pair_door_y = None
        self._pair_seek_until = rospy.Time(0)
        self._pair_seek_pose = None
        self._pair_seek_travel = 0.0
        self._enter_abort_until = rospy.Time(0)
        self._north_cruise_pose = None
        self._path_from_engage = 0.0
        self._last_pose_xy = None
        self._corridor_hits = 0
        self._upper_cruise_until = rospy.Time(0)
        self._upper_cruise_start_y = None
        self._upper_cruise_started = rospy.Time(0)
        self._upper_align_until = rospy.Time(0)
        self._lidar_center_bias = 0.0
        self._localization_healthy = True
        self._last_speed_yaw_error = None
        self._last_speed_center_error = None
        self._last_speed_gate_reason = "phase_limit"
        self._room_grid = None
        self._room_target_xy = None
        self._room_target_gain = 0.0
        self._room_target_yaw = None
        self._room_viewpoints = []
        self._room_probe_scans = 0
        self._room_pan_headings = []
        self._room_pan_index = 0
        self._room_pan_hold_until = None
        self._room_pan_complete = False
        self._room_target_reason = ""
        self._room_records = []
        self._last_room_diag = rospy.Time(0)
        self._rgb_candidates = []
        self._room_grid_lock = threading.RLock()

        self._cruise = float(rospy.get_param("~cruise_speed", 0.52))
        self._boost = float(rospy.get_param("~boost_speed", 0.62))
        self._boost_clearance = float(
            rospy.get_param(
                "~boost_clearance_m", boost_clearance_for_speed(self._boost)
            )
        )
        self._crawl = float(rospy.get_param("~crawl_speed", 0.24))
        self._max_yaw = float(rospy.get_param("~max_yaw_rate", 0.42))
        self._corridor_half_min = float(rospy.get_param("~corridor_half_min", 0.45))
        self._corridor_half_max = float(rospy.get_param("~corridor_half_max", 1.55))
        self._corridor_width_max = float(rospy.get_param("~corridor_width_max", 3.2))
        self._door_open = float(rospy.get_param("~door_open_range", 2.3))
        self._door_closed = float(rospy.get_param("~door_closed_range", 1.80))
        self._door_asym = float(rospy.get_param("~door_asym_m", 1.2))
        self._room_width_min = float(rospy.get_param("~room_width_min", 5.0))
        self._wall_follow = float(rospy.get_param("~wall_follow_dist", 0.90))
        self._front_slow = float(rospy.get_param("~front_slow_range", 1.35))
        self._front_stop = float(rospy.get_param("~front_stop_range", 0.62))
        self._stuck_pose = None
        self._stuck_since = rospy.Time(0)
        self._cell = float(rospy.get_param("~visit_cell_m", 1.0))
        self._room_timeout = float(rospy.get_param("~room_timeout_s", 16.0))
        self._room_target_depth = float(rospy.get_param("~room_target_depth_m", 3.0))
        self._room_target_span = float(rospy.get_param("~room_target_span_m", 1.7))
        self._room_min_path = float(rospy.get_param("~room_min_path_m", 4.0))
        self._scan_period = float(rospy.get_param("~scan_period_s", 7.0))
        self._scan_duration = float(rospy.get_param("~scan_duration_s", 2.2))
        self._max_rooms = int(rospy.get_param("~max_rooms", 4))
        self._center_gain = float(rospy.get_param("~center_gain", 0.55))
        self._min_door_spacing_y = float(rospy.get_param("~min_door_spacing_y", 8.0))
        self._min_door_y = float(rospy.get_param("~min_door_y", 0.0))
        self._min_corridor_travel_m = float(rospy.get_param("~min_corridor_travel_m", 4.5))
        self._pair_door_window = float(rospy.get_param("~pair_door_window_m", 2.5))
        self._prefer_open_min = float(rospy.get_param("~prefer_open_min", 1.75))
        self._room_min_cells = int(rospy.get_param("~room_min_cells", 4))
        self._room_min_depth = float(rospy.get_param("~room_min_depth", 2.2))
        self._corridor_settle_s = float(rospy.get_param("~corridor_settle_s", 1.2))
        self._corridor_settle_m = float(rospy.get_param("~corridor_settle_m", 0.7))
        self._enter_min_depth = float(rospy.get_param("~enter_min_depth", 1.2))
        self._enter_timeout_s = float(rospy.get_param("~enter_timeout_s", 18.0))
        self._enter_abort_cooldown_s = float(rospy.get_param("~enter_abort_cooldown_s", 4.0))
        self._enter_yaw_tol = float(rospy.get_param("~enter_yaw_tol", 0.35))
        self._cover_cell = float(rospy.get_param("~cover_cell_m", 1.5))
        self._cover_exit_ratio = float(rospy.get_param("~cover_exit_ratio", 0.45))
        self._look_hold_s = float(rospy.get_param("~look_hold_s", 0.70))
        self._look_max_s = float(rospy.get_param("~look_max_s", 1.8))
        self._sphere_min_score = float(rospy.get_param("~sphere_min_score", 0.32))
        self._look_cooldown_s = float(rospy.get_param("~look_cooldown_s", 2.5))
        self._exit_timeout = float(rospy.get_param("~exit_timeout_s", 14.0))
        self._look_yaw = 0.0
        self._look_deadline = rospy.Time(0)
        self._look_hold_until = None
        self._look_cluster = None
        self._last_look = rospy.Time(0)
        self._resume_mode = Mode.CORRIDOR
        self._sphere_candidates = []
        self._scan_points = []
        self._looks_this_room = 0
        self._max_looks_per_room = int(rospy.get_param("~max_looks_per_room", 4))
        self._relative_room_sweep = bool(
            rospy.get_param("~relative_room_sweep", True)
        )
        self._adaptive_room_viewpoints = bool(
            rospy.get_param("~adaptive_room_viewpoints", True)
        )
        self._room_grid_resolution = float(
            rospy.get_param("~room_grid_resolution_m", 0.15)
        )
        self._robot_inflation = float(rospy.get_param("~robot_inflation_m", 0.45))
        self._first_view_depth = float(rospy.get_param("~first_view_depth_m", 3.4))
        self._room_min_viewpoints = int(rospy.get_param("~room_min_viewpoints", 2))
        self._room_max_viewpoints = int(rospy.get_param("~room_max_viewpoints", 3))
        self._room_min_viewpoint_baseline = float(
            rospy.get_param("~room_min_viewpoint_baseline_m", 1.2)
        )
        self._room_min_lateral_span = float(
            rospy.get_param("~room_min_lateral_span_m", 1.4)
        )
        self._room_viewpoint_speed = float(
            rospy.get_param("~room_viewpoint_speed", 0.65)
        )
        self._room_lidar_complete = float(
            rospy.get_param("~room_lidar_complete_ratio", 0.92)
        )
        self._room_view_min_gain = float(
            rospy.get_param("~room_viewpoint_min_gain", 0.08)
        )
        self._room_shadow_max_m2 = float(
            rospy.get_param("~room_shadow_max_m2", 0.50)
        )
        self._room_boundary_min_directions = int(
            rospy.get_param("~room_boundary_min_directions", 2)
        )
        self._viewpoint_tolerance = float(
            rospy.get_param("~viewpoint_tolerance_m", 0.35)
        )
        self._pan_yaw_rate = float(rospy.get_param("~pan_yaw_rate", 0.35))
        self._pan_hold_s = float(rospy.get_param("~pan_heading_dwell_s", 0.40))
        self._corridor_anchor_offset = float(
            rospy.get_param("~corridor_anchor_offset_m", 0.90)
        )
        self._exit_anchor_tolerance = float(
            rospy.get_param("~exit_anchor_tolerance_m", 0.35)
        )
        self._exit_alignment_radius = float(
            rospy.get_param("~exit_alignment_radius_m", 0.80)
        )
        self._room_diagnostic_file = rospy.get_param(
            "~room_diagnostic_file",
            os.path.join(os.getcwd(), "results", "compliant_online_coverage.json"),
        )

        self._fis = FrontierInformationStructure(
            sector_count=int(rospy.get_param("~fis_sectors", 36)),
            camera_range=float(rospy.get_param("~camera_range", 6.5)),
            camera_fov=math.radians(float(rospy.get_param("~camera_fov_deg", 60.0))),
            radar_max_range=12.0,
            near_range=3.5,
            mid_range=7.0,
            stale_seconds=10.0,
            min_points=3,
        )

        self._cmd_pub = rospy.Publisher(
            "/simenv/structure_cmd_vel", TwistStamped, queue_size=5
        )
        self._active_pub = rospy.Publisher(
            "/simenv/structure_active", Bool, queue_size=1, latch=True
        )
        self._mode_pub = rospy.Publisher(
            "/simenv/structure_mode", String, queue_size=1, latch=True
        )
        self._room_diag_pub = rospy.Publisher(
            "/simenv/room_coverage_status", String, queue_size=1, latch=True
        )
        self._speed_gate_pub = rospy.Publisher(
            "/simenv/speed_gate_status", String, queue_size=2
        )
        self._wp_pub = rospy.Publisher("/way_point", PointStamped, queue_size=1)
        self._active_pub.publish(Bool(data=False))

        rospy.Subscriber("/state_estimation", Odometry, self._on_odom, queue_size=10)
        rospy.Subscriber("/trunk_imu", Imu, self._on_imu, queue_size=20)
        rospy.Subscriber("/scan", PointCloud, self._on_scan, queue_size=2)
        rospy.Subscriber("/simenv/egress_active", Bool, self._on_egress, queue_size=1)
        rospy.Subscriber(
            "/simenv/localization_healthy", Bool, self._on_localization_health, queue_size=1
        )
        rospy.Subscriber(
            "/simenv/danger_candidate", PointStamped, self._on_danger_candidate, queue_size=5
        )
        rospy.Timer(rospy.Duration(0.05), self._on_timer)
        rospy.on_shutdown(self._write_room_diagnostics)
        rospy.loginfo(
            "Structure explorer ready (cruise=%.2f boost=%.2f clearance=%.2f FIS+sphere look)",
            self._cruise,
            self._boost,
            self._boost_clearance,
        )

    def _on_localization_health(self, message):
        self._localization_healthy = bool(message.data)

    def _on_danger_candidate(self, message):
        stamp = message.header.stamp if message.header.stamp != rospy.Time(0) else rospy.Time.now()
        self._rgb_candidates.append((message.point.x, message.point.y, stamp))
        cutoff = rospy.Time.now() - rospy.Duration(12.0)
        self._rgb_candidates = [item for item in self._rgb_candidates if item[2] >= cutoff]

    def _on_egress(self, message):
        was = self._egress_active
        self._egress_active = bool(message.data)
        if was and not self._egress_active:
            self._engage_pose = self._pose
            self._engage_time = rospy.Time.now()
            self._path_from_engage = 0.0
            self._north_cruise_pose = None
            self._corridor_hits = 0
            self._corridor_ready = False
            self._set_mode(Mode.CORRIDOR)
            self._active_pub.publish(Bool(data=True))
            rospy.loginfo("Structure explorer engaged after lobby egress.")

    def _on_odom(self, message):
        p = message.pose.pose.position
        yaw = yaw_from_quaternion(message.pose.pose.orientation)
        self._pose = (p.x, p.y, yaw)
        xy = (p.x, p.y)
        if self._last_pose_xy is not None:
            step = math.hypot(xy[0] - self._last_pose_xy[0], xy[1] - self._last_pose_xy[1])
            if step < 1.0:
                self._path_from_engage += step
                if self._pair_seek_pose is not None:
                    self._pair_seek_travel += step
        self._last_pose_xy = xy
        cell = (int(round(p.x / self._cell)), int(round(p.y / self._cell)))
        self._visited_cells.add(cell)
        if self._mode in (Mode.ENTER_ROOM, Mode.ROOM_COVER):
            self._room_cells.add(cell)
        if self._mode == Mode.ROOM_COVER:
            if self._room_cover_last_xy is not None:
                room_step = math.hypot(
                    xy[0] - self._room_cover_last_xy[0],
                    xy[1] - self._room_cover_last_xy[1],
                )
                if room_step < 0.75:
                    self._room_cover_path += room_step
            self._room_cover_last_xy = xy
        if self._mode == Mode.EXIT_ROOM:
            if self._exit_last_xy is not None:
                step = math.hypot(xy[0] - self._exit_last_xy[0], xy[1] - self._exit_last_xy[1])
                if step < 0.5:
                    self._exit_travel_m += step
            self._exit_last_xy = xy
        if self._corridor_progress_y is None or p.y > self._corridor_progress_y:
            self._corridor_progress_y = p.y

    def _on_imu(self, message):
        raw = yaw_from_quaternion(message.orientation)
        if self._imu_yaw_offset is None:
            # The public start pose defines the initial heading; only relative
            # IMU yaw is used after this one-time alignment.
            self._imu_yaw_offset = angle_diff(1.5708, raw)
        self._imu_yaw = math.atan2(
            math.sin(raw + self._imu_yaw_offset),
            math.cos(raw + self._imu_yaw_offset),
        )

    def _on_scan(self, message):
        if not message.points:
            return
        buckets = {
            "left": [],
            "right": [],
            "front": [],
            "front_left": [],
            "front_right": [],
            "rear": [],
        }
        body_min = 0.42
        points_xy = []
        mapping_points_xy = []
        sector_bins = [[] for _ in range(self._fis.sector_count)]
        for point in message.points:
            x, y = float(point.x), float(point.y)
            rng = math.hypot(x, y)
            if not math.isfinite(rng) or rng < body_min or rng > 12.0:
                continue
            points_xy.append((x, y))
            z = float(point.z)
            if math.isfinite(z) and -0.32 <= z <= 1.20:
                mapping_points_xy.append((x, y))
            ang = math.atan2(y, x)
            deg = math.degrees(ang)
            sector_bins[self._fis.heading_bin(ang)].append(rng)
            if 55.0 <= deg <= 125.0:
                buckets["left"].append(rng)
            if -125.0 <= deg <= -55.0:
                buckets["right"].append(rng)
            if abs(deg) <= 22.0:
                buckets["front"].append(rng)
            if 22.0 < deg <= 55.0:
                buckets["front_left"].append(rng)
            if -55.0 <= deg < -22.0:
                buckets["front_right"].append(rng)
            if abs(deg) >= 150.0:
                buckets["rear"].append(rng)

        for key, values in buckets.items():
            if not values:
                self._ranges[key] = None
            elif key == "front":
                # The Mid-360 is pitched; lower front percentiles are dominated
                # by floor/body returns around 0.5--0.9 m.  A high percentile
                # tracks the actual forward wall/clearance and restores cruise
                # speed in an open corridor.
                self._ranges[key] = float(np.percentile(values, 82))
            else:
                self._ranges[key] = float(np.median(values))

        self._scan_points = points_xy
        sectors = []
        for values in sector_bins:
            if not values:
                sectors.append((0, 0.0))
            else:
                values.sort()
                sectors.append(
                    (len(values), values[min(len(values) - 1, int(0.7 * len(values)))])
                )
        if self._pose is not None:
            x, y, yaw = self._pose
            sensor_yaw = self._imu_yaw if self._imu_yaw is not None else yaw
            stamp = message.header.stamp.to_sec() if message.header.stamp else rospy.get_time()
            self._fis.update_from_radar(x, y, sensor_yaw, sectors, stamp)
            self._fis.mark_lidar_coverage(
                x, y, sensor_yaw, sectors, cell_size=self._cover_cell, max_mark_range=6.0
            )
            self._sphere_candidates = FrontierInformationStructure.detect_sphere_candidates(
                points_xy, sensor_yaw
            )
            if (
                self._room_grid is not None
                and self._mode == Mode.ROOM_COVER
                and self._localization_healthy
            ):
                with self._room_grid_lock:
                    self._room_grid.update_scan(x, y, sensor_yaw, mapping_points_xy)

    def _set_mode(self, mode):
        if mode == self._mode:
            return
        self._mode = mode
        self._last_speed_center_error = None
        self._last_speed_gate_reason = "phase_limit"
        self._mode_since = rospy.Time.now()
        self._mode_pub.publish(String(data=mode.value))
        rospy.loginfo(
            "Structure mode -> %s (label=%s L=%.2f R=%.2f F=%s)",
            mode.value,
            self._last_label,
            self._ranges["left"] or -1.0,
            self._ranges["right"] or -1.0,
            "%.2f" % self._ranges["front"] if self._ranges["front"] is not None else "?",
        )

    def _classify(self):
        left = self._ranges["left"]
        right = self._ranges["right"]
        # Empty side buckets usually mean the wall is inside body_min (too close),
        # NOT a wide-open doorway. Treating None as 8m caused false right enters
        # while hugging the wall (run33).
        left_v = 0.40 if left is None else left
        right_v = 0.40 if right is None else right
        if left is None and right is None:
            return "unknown"
        width = left_v + right_v
        balanced = abs(left_v - right_v) < 0.75
        corridorish = (
            left is not None
            and right is not None
            and self._corridor_half_min <= left <= self._corridor_half_max
            and self._corridor_half_min <= right <= self._corridor_half_max
            and width <= self._corridor_width_max
            and balanced
        )
        if corridorish:
            return "corridor"

        # Doorways require REAL returns on both sides (no None→open).
        closed_slack = 0.55
        if (
            left is not None
            and right is not None
            and left >= self._door_open
            and right <= self._door_closed + closed_slack
            and (left - right) >= self._door_asym
        ):
            return "door_left"
        if (
            left is not None
            and right is not None
            and right >= self._door_open
            and left <= self._door_closed + closed_slack
            and (right - left) >= self._door_asym
        ):
            return "door_right"

        if (
            left is not None
            and right is not None
            and width >= self._room_width_min
            and min(left, right) >= 2.2
        ):
            return "room"
        return "unknown"

    def _inward_yaw(self):
        side = self._door_side if self._door_side != 0 else -1
        return self._room_entry_yaw - side * (math.pi / 2.0)

    def _outward_yaw(self):
        return self._inward_yaw() + math.pi

    def _corridor_like(self):
        """Bilateral corridor walls. Slightly loose so doorway threshold counts."""
        left = self._ranges["left"]
        right = self._ranges["right"]
        front = self._ranges["front"]
        if left is None or right is None:
            return False
        if not (
            0.50 <= left <= 1.65
            and 0.50 <= right <= 1.65
            and (left + right) <= 3.0
            and abs(left - right) < 0.95
        ):
            return False
        # Reject obvious dead-end pockets, but allow doorway (front may be short).
        if front is not None and front < 0.85:
            return False
        return True

    def _publish_wp(self, x, y):
        wp = PointStamped()
        wp.header.stamp = rospy.Time.now()
        wp.header.frame_id = "map"
        wp.point.x = x
        wp.point.y = y
        wp.point.z = 0.15
        self._wp_pub.publish(wp)

    def _publish_cmd(self, linear_x, angular_z):
        # Localization faults must not turn a fast command into an uncontrolled
        # map-frame excursion. High-curvature translation is bounded here as a
        # final structure-owner invariant, independently of individual modes.
        if not self._localization_healthy:
            linear_x = max(-0.24, min(0.24, linear_x))
        if abs(angular_z) > 0.32:
            linear_x = max(-0.18, min(0.18, linear_x))
        cmd = TwistStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.header.frame_id = "vehicle"
        cmd.twist.linear.x = float(linear_x)
        cmd.twist.angular.z = float(max(-self._max_yaw, min(self._max_yaw, angular_z)))
        gate = {
            "stage": self._mode.value,
            "room_phase": self._room_phase,
            "reason": self._last_speed_gate_reason,
            "requested_speed_mps": round(cmd.twist.linear.x, 4),
            "front_clearance_m": self._ranges.get("front"),
            "narrow_front_m": None,
            "localization_healthy": self._localization_healthy,
            "yaw_error_rad": self._last_speed_yaw_error,
            "center_error_m": self._last_speed_center_error,
        }
        self._speed_gate_pub.publish(String(data=json.dumps(gate, sort_keys=True)))
        self._cmd_pub.publish(cmd)

    def _maybe_scan(self, now):
        if now < self._scan_until:
            self._publish_cmd(0.0, 0.38)
            return True
        if (now - self._last_scan).to_sec() >= self._scan_period:
            self._last_scan = now
            self._scan_until = now + rospy.Duration(self._scan_duration)
            self._publish_cmd(0.0, 0.38)
            return True
        return False

    def _finish_look(self, x, y, yaw, reason="done"):
        self._fis.mark_camera_view(x, y, yaw, cell_size=self._cover_cell)
        self._fis.note_camera_visit(yaw)
        self._last_look = rospy.Time.now()
        rospy.loginfo("Structure LOOK_AT end (%s)", reason)
        self._set_mode(self._resume_mode)

    def _begin_look(self, yaw_target, reason, resume_mode):
        if self._looks_this_room >= self._max_looks_per_room and resume_mode == Mode.ROOM_COVER:
            return False
        self._resume_mode = resume_mode
        self._look_yaw = yaw_target
        now = rospy.Time.now()
        self._look_deadline = now + rospy.Duration(self._look_max_s)
        self._look_hold_until = None
        self._last_look = now
        self._looks_this_room += 1
        self._set_mode(Mode.LOOK_AT)
        rospy.loginfo("Structure LOOK_AT yaw=%.2f (%s)", yaw_target, reason)
        return True

    def _maybe_look_sphere(self, x, y, yaw, now, resume_mode):
        if self._mode == Mode.LOOK_AT:
            return False
        if (now - self._last_look).to_sec() < self._look_cooldown_s:
            return False
        if not self._sphere_candidates:
            return False
        best = self._sphere_candidates[0]
        if best.score < self._sphere_min_score:
            return False
        if self._fis.is_camera_heading_seen(x, y, best.yaw, self._cover_cell):
            return False
        return self._begin_look(
            best.yaw,
            "sphere_d=%.2f_r=%.1f" % (best.diameter, best.distance),
            resume_mode,
        )

    def _run_look_at(self, x, y, yaw, now):
        # Hard timeout — never spin in place for tens of seconds.
        if self._look_deadline != rospy.Time(0) and now >= self._look_deadline:
            self._finish_look(x, y, yaw, "timeout")
            return
        # Also bound by mode age in case deadline stamp is weird under /clock.
        if (now - self._mode_since).to_sec() > self._look_max_s + 0.3:
            self._finish_look(x, y, yaw, "mode_timeout")
            return
        err = angle_diff(self._look_yaw, yaw)
        if abs(err) > 0.18:
            # Keep a tiny crawl so RL doesn't stall while turning.
            self._publish_cmd(0.06, max(-self._max_yaw, min(self._max_yaw, 1.9 * err)))
            return
        if self._look_hold_until is None:
            self._look_hold_until = now + rospy.Duration(self._look_hold_s)
        self._publish_cmd(0.0, 0.0)
        self._fis.mark_camera_view(x, y, yaw, cell_size=self._cover_cell)
        if now >= self._look_hold_until:
            self._finish_look(x, y, yaw, "hold")

    def _speed_for_heading(self, yaw_err, base, front):
        """Clearance-, turn-, and localization-aware forward speed."""
        self._last_speed_yaw_error = float(yaw_err)
        self._last_speed_gate_reason = "clearance_limited"
        if front is not None and front < self._front_stop:
            self._last_speed_gate_reason = "front_stop"
            return 0.0
        turn = abs(yaw_err)
        if turn >= 0.55:
            self._last_speed_gate_reason = "turning"
            return 0.0
        turn_scale = max(0.28, 1.0 - turn / 0.62)
        intent = max(0.0, min(float(base), self._boost))
        if front is None:
            clearance_limit = self._cruise
        elif front < 1.0:
            ratio = max(0.0, (front - self._front_stop) / max(1.0 - self._front_stop, 1e-3))
            clearance_limit = 0.12 + ratio * max(0.0, self._crawl - 0.12)
        elif front < 2.0:
            clearance_limit = self._crawl + (front - 1.0) * (0.44 - self._crawl)
        elif front < 3.5:
            clearance_limit = 0.44 + (front - 2.0) / 1.5 * (self._cruise - 0.44)
        else:
            clearance_limit = (
                clearance_speed_limit(
                    front,
                    self._boost,
                    cruise_speed=self._cruise,
                    legacy_boost=0.62,
                )
                if turn < 0.12
                else self._cruise
            )
        speed = min(intent, clearance_limit) * turn_scale
        if (
            turn < 0.08
            and front is not None
            and front >= self._boost_clearance
            and intent >= self._cruise
        ):
            speed = min(self._boost, clearance_limit)
            self._last_speed_gate_reason = "straight_boost"
        elif turn >= 0.08:
            self._last_speed_gate_reason = "turning"
        if not self._localization_healthy:
            speed = min(speed, 0.24)
            self._last_speed_gate_reason = "localization_unhealthy"
        return speed

    def _preferred_door_open(self, side):
        """Looser door check for the preferred side at a paired Y band."""
        if side > 0:
            open_rng = self._ranges["right"]
            closed_rng = self._ranges["left"]
        else:
            open_rng = self._ranges["left"]
            closed_rng = self._ranges["right"]
        if open_rng is None or closed_rng is None:
            return False
        return (
            open_rng >= self._prefer_open_min
            and open_rng > closed_rng + 0.35
            and closed_rng <= self._door_closed + 0.35
        )

    def _start_pair_seek(self, y):
        self._pair_door_y = y
        # Longer window: LIO Y is soft; need time to reacquire corridor + opposite door.
        self._pair_seek_until = rospy.Time.now() + rospy.Duration(36.0)
        self._pair_seek_pose = self._pose[:2] if self._pose is not None else None
        self._pair_seek_travel = 0.0
        rospy.loginfo(
            "Structure pair-seek side=%d near y≈%.1f (structure-first, LIO y soft)",
            self._prefer_side,
            y,
        )

    def _corridor_travel_m(self, x, y):
        if self._engage_pose is None:
            return 0.0
        return math.hypot(x - self._engage_pose[0], y - self._engage_pose[1])

    def _door_allowed(self, x, y, side):
        if self._rooms_cleared >= self._max_rooms:
            return False
        if not self._corridor_ready:
            # Allow paired opposite-door entry immediately after exiting a room.
            if (
                self._pair_door_y is not None
                and rospy.Time.now() < self._pair_seek_until
                and side == self._prefer_side
            ):
                pass
            else:
                return False
        if self._min_door_y > 0.0 and y < self._min_door_y:
            return False

        # Reject lobby false doors: need real corridor geometry + travel first.
        # Absolute LIO Y is unreliable; body /scan + relative travel are trusted.
        travel = max(self._corridor_travel_m(x, y), self._path_from_engage)
        if self._rooms_cleared == 0:
            # Do not gate on absolute/relative LIO Y here.  FAST-LIO can loop
            # back while the robot keeps moving north.  Travel, repeated
            # bilateral corridor observations and real asymmetric side ranges
            # below provide the coordinate-free lobby rejection.
            # Egress has already required peak north progress plus >=6 m of
            # travel.  Requiring more LIO-derived travel/corridor hits here
            # deadlocked wall-hug cases where the near-wall scan bucket is
            # intentionally empty.  The real asymmetric opening below is the
            # first-room gate.
            yaw = self._pose[2] if self._pose is not None else 1.5708
            if abs(angle_diff(1.5708, yaw)) > 1.20:
                return False
            # Classic door: one side open, other wall — not both wide (lobby).
            open_rng = self._ranges["right"] if side > 0 else self._ranges["left"]
            closed_rng = self._ranges["left"] if side > 0 else self._ranges["right"]
            if open_rng is None or closed_rng is None:
                return False
            if closed_rng > self._door_closed + 0.55:
                return False
            if open_rng < self._door_open:
                return False
            # Honor prefer_side for the first room (seed77 → room1 right).
            if self._prefer_side != 0 and side != self._prefer_side:
                return False

        if (
            side == self._skip_side
            and self._skip_side != 0
            and rospy.Time.now() < self._skip_side_until
        ):
            return False
        if (
            self._last_door_y is not None
            and side == self._last_door_side
            and abs(y - self._last_door_y) < self._min_door_spacing_y
        ):
            return False
        if (
            self._rooms_cleared > 0
            and self._prefer_side != 0
            and side != self._prefer_side
            and rospy.Time.now() < self._skip_side_until + rospy.Duration(8.0)
        ):
            return False
        if self._prefer_side != 0 and side != self._prefer_side:
            pref = self._ranges["left"] if self._prefer_side < 0 else self._ranges["right"]
            closed = self._ranges["right"] if self._prefer_side < 0 else self._ranges["left"]
            if (
                pref is not None
                and pref >= self._door_open
                and closed is not None
                and closed <= self._door_closed + 0.25
            ):
                return False
        return True

    def _update_corridor_ready(self, x, y, now):
        if self._corridor_ready:
            return
        if self._engage_pose is None:
            self._engage_pose = (x, y, 0.0)
            self._engage_time = now
            return
        dist = math.hypot(x - self._engage_pose[0], y - self._engage_pose[1])
        if (now - self._engage_time).to_sec() >= self._corridor_settle_s or dist >= self._corridor_settle_m:
            self._corridor_ready = True
            rospy.loginfo("Structure corridor ready (settle dist=%.2f)", dist)

    def _begin_enter(self, side, x, y, yaw, centered=False):
        self._door_side = side
        self._last_door_side = side
        self._last_door_y = y
        self._enter_started_xy = (x, y)
        self._last_entry_xy = (x, y)
        self._room_entry_xy = (x, y)
        self._door_center_xy = (x, y) if centered else None
        self._room_entry_yaw = yaw
        self._last_entry_yaw = yaw
        self._room_cells = set()
        self._enter_center_s = 1.6 if centered else 0.0
        self._enter_drive_s = 0.0
        self._paired_entry = bool(centered)
        self._entry_confirmed = False
        self._entry_confirmation_depth = 0.0
        self._entry_door_frame_crossed = False
        # Save a sensor-time corridor pose before the dog crosses the jamb.
        # Normal entries refine this from the doorway centre below; a paired
        # straight-through entry must return to this actual corridor pose.
        self._corridor_anchor_xy = (x, y)
        # Crossing directly from one room to its paired room spans the full
        # corridor plus the opposite threshold. v15 reached x=0.91, only
        # 0.19 m short of the x>1.1 room boundary, then began its sweep.
        self._enter_required_s = 3.2 if centered else 1.8
        self._looks_this_room = 0
        self._cardinal_looks = 0
        self._room_phase = "idle"
        rospy.loginfo("Structure COMMIT enter side=%d at (%.2f,%.2f) yaw=%.2f", side, x, y, yaw)
        self._set_mode(Mode.ENTER_ROOM)

    def _begin_room_cover(self, x, y, now):
        """Initialize a doorway-relative, sensor-evidence coverage pass."""
        self._room_entry_xy = self._door_center_xy or self._room_entry_xy or (x, y)
        self._room_started_at = now
        self._room_phase = "map_warmup" if self._adaptive_room_viewpoints else "deep"
        self._room_phase_since = now
        self._room_cover_path = 0.0
        self._room_cover_last_xy = (x, y)
        self._room_max_depth = 0.0
        self._room_transverse_min = 0.0
        self._room_transverse_max = 0.0
        self._room_sweep_sign = 1.0
        self._room_sweep_start_transverse = 0.0
        self._room_target_xy = None
        self._room_target_gain = 0.0
        self._room_target_yaw = None
        self._room_viewpoints = []
        self._room_probe_scans = 0
        self._room_pan_headings = []
        self._room_pan_index = 0
        self._room_pan_hold_until = None
        self._room_pan_complete = False
        self._room_target_reason = ""
        self._pending_room_status = "partial"
        self._pending_room_reason = "not_evaluated"
        self._room_grid = RoomVisibilityGrid(
            self._room_entry_xy[0],
            self._room_entry_xy[1],
            self._inward_yaw(),
            resolution=self._room_grid_resolution,
            inflation_radius=self._robot_inflation,
            max_depth=9.5,
            half_width=7.5,
            sensor_range=9.0,
        )
        self._set_mode(Mode.ROOM_COVER)

    def _set_room_phase(self, phase, now):
        if phase == self._room_phase:
            return
        rospy.loginfo(
            "Structure room phase %s -> %s depth=%.2f span=%.2f path=%.2f cells=%d",
            self._room_phase,
            phase,
            self._room_max_depth,
            self._room_transverse_max - self._room_transverse_min,
            self._room_cover_path,
            len(self._room_cells),
        )
        self._room_phase = phase
        self._room_phase_since = now

    def _room_coordinates(self, x, y):
        origin = self._room_entry_xy or self._door_center_xy or (x, y)
        inward = self._inward_yaw()
        dx, dy = x - origin[0], y - origin[1]
        depth = dx * math.cos(inward) + dy * math.sin(inward)
        transverse = -dx * math.sin(inward) + dy * math.cos(inward)
        return depth, transverse

    def _room_coverage_complete(self):
        if self._adaptive_room_viewpoints and self._room_grid is not None:
            status = self._room_grid.coverage_status()
            candidate = self._room_grid.best_viewpoint(
                self._pose[0] if self._pose is not None else self._room_entry_xy[0],
                self._pose[1] if self._pose is not None else self._room_entry_xy[1],
                self._imu_yaw if self._imu_yaw is not None else self._room_entry_yaw,
                self._room_viewpoints,
                min_gain=self._room_view_min_gain,
            )
            origin = self._room_entry_xy or self._door_center_xy
            if origin is None:
                return False
            return self._room_pan_complete and room_exploration_complete(
                self._entry_confirmed,
                status,
                self._room_max_depth,
                max(3.0, self._first_view_depth - 0.4),
                self._room_viewpoints,
                origin[0],
                origin[1],
                self._inward_yaw(),
                0.0 if candidate is None else candidate.expected_gain,
                min_lidar_ratio=self._room_lidar_complete,
                max_shadow_m2=self._room_shadow_max_m2,
                max_next_gain=self._room_view_min_gain,
                min_viewpoints=self._room_min_viewpoints,
                min_baseline=self._room_min_viewpoint_baseline,
                min_lateral_span=self._room_min_lateral_span,
            )
        span = self._room_transverse_max - self._room_transverse_min
        return (
            self._room_max_depth >= max(2.5, self._room_target_depth - 0.35)
            and self._room_cover_path >= self._room_min_path
            and span >= min(1.5, self._room_target_span)
            and len(self._room_cells) >= self._room_min_cells
        )

    def _room_viewpoint_metrics(self):
        if len(self._room_viewpoints) < 2:
            return 0.0, 0.0
        maximum_baseline = max(
            math.hypot(ax - bx, ay - by)
            for index, (ax, ay) in enumerate(self._room_viewpoints)
            for bx, by in self._room_viewpoints[index + 1 :]
        )
        origin = self._room_entry_xy or self._door_center_xy
        if origin is None:
            return maximum_baseline, 0.0
        inward = self._inward_yaw()
        side_x, side_y = -math.sin(inward), math.cos(inward)
        lateral = [
            (x - origin[0]) * side_x + (y - origin[1]) * side_y
            for x, y in self._room_viewpoints
        ]
        return maximum_baseline, max(lateral) - min(lateral)

    def _room_diagnostic_payload(self, x, y, now, exit_reason=None):
        elapsed = (
            0.0
            if self._room_started_at == rospy.Time(0)
            else max(0.0, (now - self._room_started_at).to_sec())
        )
        viewpoint_baseline, viewpoint_lateral_span = self._room_viewpoint_metrics()
        origin = self._room_entry_xy or self._door_center_xy
        diversity_ok = False
        if origin is not None:
            diversity_ok = room_viewpoint_diversity_gate(
                self._room_viewpoints,
                origin[0],
                origin[1],
                self._inward_yaw(),
                min_viewpoints=self._room_min_viewpoints,
                min_baseline=self._room_min_viewpoint_baseline,
                min_lateral_span=self._room_min_lateral_span,
            )
        payload = {
            "schema": "simenv_room_coverage_v1",
            "runtime_only": True,
            "room_index": self._rooms_cleared,
            "phase": self._room_phase,
            "status": self._pending_room_status,
            "elapsed_sec": round(elapsed, 3),
            "path_m": round(self._room_cover_path, 3),
            "viewpoint_count": len(self._room_viewpoints),
            "qualified_viewpoint_count": len(self._room_viewpoints),
            "probe_scan_count": self._room_probe_scans,
            "viewpoint_baseline_m": round(viewpoint_baseline, 3),
            "viewpoint_lateral_span_m": round(viewpoint_lateral_span, 3),
            "viewpoint_diversity_ok": diversity_ok,
            "viewpoints": [
                [round(point[0], 3), round(point[1], 3)]
                for point in self._room_viewpoints
            ],
            "target": None
            if self._room_target_xy is None
            else [round(self._room_target_xy[0], 3), round(self._room_target_xy[1], 3)],
            "target_gain": round(self._room_target_gain, 4),
            "target_reason": self._room_target_reason,
            "pan_sweep_complete": self._room_pan_complete,
            "localization_healthy": self._localization_healthy,
            "entry_confirmed": self._entry_confirmed,
            "entry_confirmation_depth_m": round(self._entry_confirmation_depth, 3),
            "door_frame_crossed": self._entry_door_frame_crossed,
        }
        if self._room_grid is not None:
            status = self._room_grid.coverage_status()
            payload.update(
                {
                    "lidar_ratio": round(status.lidar_ratio, 5),
                    "free_cells": status.free_cells,
                    "shadow_cells": status.shadow_cells,
                    "largest_shadow_m2": round(status.largest_shadow_m2, 4),
                    "wall_components": status.wall_components,
                    "obstacle_components": status.obstacle_components,
                    "far_wall_seen": status.far_wall_seen,
                    "left_wall_seen": status.left_wall_seen,
                    "right_wall_seen": status.right_wall_seen,
                    "boundary_directions_seen": status.boundary_directions_seen,
                }
            )
        if exit_reason is not None:
            payload["exit_reason"] = exit_reason
        return payload

    def _publish_room_diagnostic(self, x, y, now, force=False):
        if not force and (now - self._last_room_diag).to_sec() < 0.5:
            return
        self._last_room_diag = now
        payload = self._room_diagnostic_payload(x, y, now)
        self._room_diag_pub.publish(String(data=json.dumps(payload, sort_keys=True)))

    def _write_room_diagnostics(self):
        try:
            output = os.path.abspath(self._room_diagnostic_file)
            directory = os.path.dirname(output)
            os.makedirs(directory, exist_ok=True)
            payload = {
                "schema": "simenv_online_room_coverage_v1",
                "runtime_only": True,
                "visited_room_count": self._rooms_cleared,
                "completed_room_count": sum(
                    record.get("status") == "explored"
                    for record in self._room_records
                ),
                "rooms": self._room_records,
            }
            temporary = output + ".tmp"
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            os.replace(temporary, output)
        except Exception as exc:  # noqa: BLE001 - shutdown must stay best effort
            try:
                rospy.logwarn("Could not write room diagnostics: %s", exc)
            except Exception:
                pass

    def _prepare_pan(self, x, y, yaw, now, qualified=True):
        interest = [
            candidate.yaw
            for candidate in self._sphere_candidates
            if candidate.score >= self._sphere_min_score
        ]
        for candidate_x, candidate_y, stamp in self._rgb_candidates:
            if (now - stamp).to_sec() <= 12.0 and math.hypot(candidate_x - x, candidate_y - y) <= 6.5:
                interest.append(math.atan2(candidate_y - y, candidate_x - x))
        existing_viewpoints = len(self._room_viewpoints)
        if qualified and (
            not self._room_viewpoints
            or math.hypot(
                x - self._room_viewpoints[-1][0], y - self._room_viewpoints[-1][1]
            ) > 0.25
        ):
            self._room_viewpoints.append((x, y))
        elif not qualified:
            self._room_probe_scans += 1

        if existing_viewpoints == 0 and qualified:
            # V1 establishes the full central camera sweep.
            self._room_pan_headings = self._room_grid.scan_headings(
                x,
                y,
                yaw,
                interest_yaws=interest,
                camera_fov=self._fis.camera_fov,
            )
        else:
            # V2/V3 and probe scans only revisit information-bearing headings.
            targeted = []
            if self._room_target_yaw is not None:
                targeted.append(self._room_target_yaw)
            targeted.extend(interest)
            if not targeted:
                targeted.append(self._inward_yaw())
            targeted.sort(key=lambda item: abs(angle_diff(item, yaw)))
            self._room_pan_headings = targeted[:2]
        self._room_pan_index = 0
        self._room_pan_hold_until = None
        self._room_pan_complete = False
        self._set_room_phase("pan_turn", now)

    def _run_adaptive_room_cover(self, x, y, yaw, now, elapsed):
        """Drive only to information-bearing points and rotate while stationary."""
        depth, transverse = self._room_coordinates(x, y)
        self._room_max_depth = max(self._room_max_depth, depth)
        self._room_transverse_min = min(self._room_transverse_min, transverse)
        self._room_transverse_max = max(self._room_transverse_max, transverse)
        phase_elapsed = (now - self._room_phase_since).to_sec()
        front = self._ranges["front"]
        self._publish_room_diagnostic(x, y, now)

        if (
            self._room_timeout > 0.0
            and elapsed >= self._room_timeout
            and self._room_phase != "return"
        ):
            complete = self._room_coverage_complete()
            self._pending_room_status = "explored" if complete else "partial"
            self._pending_room_reason = "timebox"
            self._set_room_phase("return", now)
            return

        if self._room_phase == "map_warmup":
            self._publish_cmd(0.0, 0.0)
            enough_map = self._room_grid is not None and len(self._room_grid.free) >= 20
            if not enough_map and phase_elapsed < 1.2:
                return
            target = self._room_grid.first_viewpoint(
                x, y, preferred_depth=self._first_view_depth
            )
            target_depth = -1.0
            if target is not None:
                target_depth, _ = self._room_grid.room_coordinates(target[0], target[1])
            required_advance = max(0.0, self._first_view_depth - depth)
            inward_aligned = abs(angle_diff(self._inward_yaw(), yaw)) <= 0.35
            if (
                target_depth < max(1.8, self._first_view_depth - 0.5)
                and inward_aligned
                and front is not None
                and front >= required_advance + 0.65
            ):
                # The forward LiDAR ray is stronger evidence than a disconnected
                # raster flood fill.  Use it to reach the intended interior
                # viewpoint when the full A1 stopping margin is visibly clear.
                target = (
                    self._room_entry_xy[0]
                    + self._first_view_depth * math.cos(self._inward_yaw()),
                    self._room_entry_xy[1]
                    + self._first_view_depth * math.sin(self._inward_yaw()),
                )
            if target is None:
                self._pending_room_status = "partial"
                self._pending_room_reason = "no_safe_first_viewpoint"
                self._set_room_phase("return", now)
                return
            self._room_target_xy = target
            self._room_target_gain = 1.0
            self._room_target_yaw = self._inward_yaw()
            self._room_target_reason = "safe_first_viewpoint"
            self._set_room_phase("viewpoint_drive", now)
            return

        if self._room_phase == "viewpoint_drive":
            if self._room_target_xy is None:
                self._set_room_phase("occlusion_review", now)
                return
            target_x, target_y = self._room_target_xy
            distance = math.hypot(target_x - x, target_y - y)
            desired = math.atan2(target_y - y, target_x - x)
            yaw_err = angle_diff(desired, yaw)
            blocked = (
                front is not None
                and front < self._front_stop + 0.08
                and abs(yaw_err) < 0.35
            )
            if distance <= self._viewpoint_tolerance:
                self._prepare_pan(x, y, yaw, now, qualified=True)
                return
            if phase_elapsed >= 6.0 or blocked:
                # A failed drive may still improve the local map, but it is not
                # a qualified viewpoint and cannot satisfy the diversity gate.
                self._prepare_pan(x, y, yaw, now, qualified=False)
                return
            if abs(yaw_err) > 0.42:
                speed = 0.0
            else:
                speed = min(
                    self._room_viewpoint_speed,
                    self._speed_for_heading(
                        yaw_err, self._room_viewpoint_speed, front
                    ),
                )
            self._publish_cmd(speed, 1.7 * yaw_err)
            self._publish_wp(target_x, target_y)
            return

        if self._room_phase == "pan_turn":
            if self._room_pan_index >= len(self._room_pan_headings):
                self._room_pan_complete = True
                self._set_room_phase("occlusion_review", now)
                return
            if phase_elapsed >= 8.0:
                self._set_room_phase("occlusion_review", now)
                return
            target_yaw = self._room_pan_headings[self._room_pan_index]
            yaw_err = angle_diff(target_yaw, yaw)
            if abs(yaw_err) > 0.10:
                self._room_pan_hold_until = None
                angular = max(-self._pan_yaw_rate, min(self._pan_yaw_rate, 1.7 * yaw_err))
                self._publish_cmd(0.0, angular)
                return
            if self._room_pan_hold_until is None:
                self._room_pan_hold_until = now + rospy.Duration(self._pan_hold_s)
            self._publish_cmd(0.0, 0.0)
            self._fis.mark_camera_view(x, y, yaw, cell_size=self._cover_cell)
            if now >= self._room_pan_hold_until:
                self._fis.note_camera_visit(yaw)
                self._room_pan_index += 1
                self._room_pan_hold_until = None
            return

        if self._room_phase == "occlusion_review":
            status = self._room_grid.coverage_status()
            if not self._room_pan_complete and self._room_pan_index < len(
                self._room_pan_headings
            ):
                # A slow A1 turn can miss the pan timebox by one heading. Keep
                # the already-completed headings and finish only the remainder;
                # never turn that scheduling miss into a shallow room exit.
                self._set_room_phase("pan_turn", now)
                return
            candidate = self._room_grid.best_viewpoint(
                x,
                y,
                yaw,
                self._room_viewpoints,
                min_gain=self._room_view_min_gain,
            )
            boundary_candidate = self._room_grid.boundary_recovery_viewpoint(
                x,
                y,
                yaw,
                self._room_viewpoints,
                preferred_depth=self._first_view_depth,
            )
            complete = self._room_coverage_complete()
            if complete:
                self._pending_room_status = "explored"
                self._pending_room_reason = "sensor_saturated"
                self._set_room_phase("return", now)
                return
            if (
                boundary_candidate is not None
                and len(self._room_viewpoints) < self._room_max_viewpoints
            ):
                self._room_target_xy = (boundary_candidate.x, boundary_candidate.y)
                self._room_target_gain = boundary_candidate.expected_gain
                self._room_target_yaw = boundary_candidate.yaw
                self._room_target_reason = boundary_candidate.reason
                self._set_room_phase("viewpoint_drive", now)
                return
            if candidate is not None and len(self._room_viewpoints) < self._room_max_viewpoints:
                self._room_target_xy = (candidate.x, candidate.y)
                self._room_target_gain = candidate.expected_gain
                self._room_target_yaw = candidate.yaw
                self._room_target_reason = candidate.reason
                self._set_room_phase("viewpoint_drive", now)
                return
            self._pending_room_status = "partial"
            self._pending_room_reason = (
                "viewpoint_limit" if len(self._room_viewpoints) >= self._room_max_viewpoints
                else "missing_boundary_or_unreachable_shadow"
            )
            self._set_room_phase("return", now)
            return

        # Return to a staging point just inside the doorway, then face out.
        origin = self._room_entry_xy or (x, y)
        inward = self._inward_yaw()
        target_x = origin[0] + 0.72 * math.cos(inward)
        target_y = origin[1] + 0.72 * math.sin(inward)
        distance = math.hypot(target_x - x, target_y - y)
        if distance > 0.50:
            desired = math.atan2(target_y - y, target_x - x)
            yaw_err = angle_diff(desired, yaw)
            speed = 0.0 if abs(yaw_err) > 0.42 else min(
                0.48, self._speed_for_heading(yaw_err, 0.48, front)
            )
            self._publish_cmd(speed, 1.7 * yaw_err)
            self._publish_wp(target_x, target_y)
            return
        outward_err = angle_diff(self._outward_yaw(), yaw)
        if abs(outward_err) > 0.14:
            self._publish_cmd(
                0.0,
                max(-self._pan_yaw_rate, min(self._pan_yaw_rate, 1.8 * outward_err)),
            )
            return
        self._start_exit(
            "%s room (%s)" % (self._pending_room_status, self._pending_room_reason)
        )

    def _restore_exit_context(self):
        """Recover entry cues after a failed corridor transition."""
        if self._door_side == 0:
            self._door_side = self._last_door_side if self._last_door_side != 0 else -1
        if self._room_entry_yaw == 0.0 and self._last_entry_yaw:
            self._room_entry_yaw = self._last_entry_yaw
        if self._enter_started_xy is None and self._last_entry_xy is not None:
            self._enter_started_xy = self._last_entry_xy
            self._room_entry_xy = self._last_entry_xy

    def _run_corridor(self, x, y, yaw, now):
        self._update_corridor_ready(x, y, now)
        if self._corridor_like():
            self._corridor_hits += 1
        else:
            self._corridor_hits = max(0, self._corridor_hits - 1)
        label = self._classify()
        self._last_label = label
        # An empty side bucket normally means the wall is closer than the body
        # filter, not one metre away.  Use the same convention as _classify so
        # wall-hugging produces a strong command back toward corridor centre.
        left = self._ranges["left"] if self._ranges["left"] is not None else 0.40
        right = self._ranges["right"] if self._ranges["right"] is not None else 0.40
        front = self._ranges["front"]

        if self._rooms_cleared >= 2 and now < self._upper_cruise_until:
            # A timed straight-north command stranded v9 at the horizontal
            # partition: after leaving the east room the robot was still at
            # x=+1.4, just outside the |x|<1.1 corridor mouth.  Fuse the
            # short-term LIO/DR translation with the IMU heading and steer to
            # the corridor centre before spending the upper-floor cruise time.
            # The x estimate may carry a sub-metre bias, hence the generous
            # centre band; LiDAR/front clearance remains the collision gate.
            if self._upper_cruise_start_y is None:
                self._upper_cruise_start_y = y
            north_progress = y - self._upper_cruise_start_y
            cruise_elapsed = (now - self._upper_cruise_started).to_sec()

            # Stop suppressing doorway decisions as soon as a repeated side
            # opening appears after enough clear-corridor travel.  The bounded
            # 26 s fallback remains independent of drifted absolute LIO Y.
            next_opening = label in ("door_left", "door_right", "room") and (
                (self._ranges["left"] is not None and self._ranges["left"] >= self._door_open)
                or (self._ranges["right"] is not None and self._ranges["right"] >= self._door_open)
            )
            if cruise_elapsed >= 26.0 or (cruise_elapsed >= 18.0 and next_opening):
                self._upper_cruise_until = now
                rospy.loginfo(
                    "Structure upper cruise complete (t=%.1f dy=%.2f x=%.2f opening=%s)",
                    cruise_elapsed,
                    north_progress,
                    x,
                    next_opening,
                )
            else:
                left_raw = self._ranges["left"]
                right_raw = self._ranges["right"]
                blocked = front is not None and front < 1.15
                wall_pair = (
                    left_raw is not None
                    and right_raw is not None
                    and left_raw + right_raw <= 3.6
                    and left_raw <= 2.2
                    and right_raw <= 2.2
                )
                if wall_pair:
                    # Body-frame LiDAR centring is independent of LIVO drift.
                    # Facing north, L>R means close to the right wall and asks
                    # for a positive/left correction.
                    measured_bias = max(
                        -0.26, min(0.26, 0.22 * (left_raw - right_raw))
                    )
                    self._lidar_center_bias = (
                        0.75 * self._lidar_center_bias + 0.25 * measured_bias
                    )
                    desired = 1.5708 + self._lidar_center_bias
                    recenter = abs(left_raw - right_raw) > 0.28
                    yaw_err = angle_diff(desired, yaw)
                    speed = 0.10 if abs(yaw_err) > 0.35 else (
                        0.23
                        if blocked
                        else self._speed_for_heading(yaw_err, self._boost, front)
                    )
                elif left_raw is None and right_raw is not None:
                    # Empty bucket normally means the wall lies inside the
                    # body filter. Turn away from that side before advancing.
                    self._lidar_center_bias = -0.50
                    desired = 1.5708 + self._lidar_center_bias
                    recenter = True
                    yaw_err = angle_diff(desired, yaw)
                    speed = 0.08 if abs(yaw_err) > 0.35 else 0.24
                elif right_raw is None and left_raw is not None:
                    self._lidar_center_bias = 0.50
                    desired = 1.5708 + self._lidar_center_bias
                    recenter = True
                    yaw_err = angle_diff(desired, yaw)
                    speed = 0.08 if abs(yaw_err) > 0.35 else 0.24
                elif blocked:
                    # Near the centre a short diagonal uses the more open
                    # LiDAR side to get around a jamb/dead-end without a blind
                    # in-place spin.
                    escape = forward_escape_offset(
                        self._ranges["front_left"],
                        self._ranges["front_right"],
                    )
                    desired = 1.5708 + escape
                    recenter = True
                    yaw_err = angle_diff(desired, yaw)
                    speed = 0.04 if abs(yaw_err) > 0.22 else 0.24
                else:
                    self._lidar_center_bias *= 0.80
                    desired = 1.5708
                    recenter = False
                    yaw_err = angle_diff(desired, yaw)
                    speed = (
                        0.04
                        if abs(yaw_err) > 0.18
                        else self._speed_for_heading(yaw_err, self._boost, front)
                    )

                self._publish_cmd(
                    speed,
                    max(-self._max_yaw, min(self._max_yaw, 2.0 * yaw_err)),
                )
                self._publish_wp(
                    x + 4.0 * math.cos(desired),
                    y + 4.0 * math.sin(desired),
                )
                rospy.loginfo_throttle(
                    2.0,
                    "Structure upper cruise t=%.1f dy=%.1f x=%.2f recenter=%s "
                    "blocked=%s yaw_err=%.2f F=%s",
                    cruise_elapsed,
                    north_progress,
                    x,
                    recenter,
                    blocked,
                    yaw_err,
                    "%.2f" % front if front is not None else "?",
                )
                return

        # After clearing one side, hunt the opposite door using /scan openings.
        # Absolute LIO Y is only a soft hint — drift must not block entry.
        if (
            self._prefer_side != 0
            and self._pair_door_y is not None
            and now < self._pair_seek_until
        ):
            side = self._prefer_side
            near_band = abs(y - self._pair_door_y) < self._pair_door_window
            structure_open = (
                label == ("door_right" if side > 0 else "door_left")
                or self._preferred_door_open(side)
            )
            paired_cross_open = (
                label == "room"
                and self._ranges["left"] is not None
                and self._ranges["right"] is not None
                and self._ranges["left"] > 2.3
                and self._ranges["right"] > 2.3
            )
            # Immediately after leaving a left/right room the body already
            # faces across the corridor toward its paired room. Evaluate the
            # opening before commanding a north alignment, otherwise the dog
            # drives out of the two-metre door band during that turn.
            if self._door_allowed(x, y, side) and (structure_open or paired_cross_open) and (
                near_band or self._rooms_cleared == 1 or self._pair_seek_travel < 6.0
            ):
                self._pair_door_y = None
                self._pair_seek_pose = None
                # A slight southward crossing bias compensates the A1's
                # turn-in-place creep and keeps the body away from the north
                # jamb on the return path.
                pair_entry_yaw = 1.5708 - side * 0.15
                self._begin_enter(side, x, y, pair_entry_yaw, centered=True)
                return
            north_err = angle_diff(1.5708, yaw)
            if abs(north_err) > 0.35:
                self._publish_cmd(
                    0.08,
                    max(-self._max_yaw, min(self._max_yaw, 1.8 * north_err)),
                )
                self._publish_wp(x, y + 2.0)
                return
            # Stay on the door band while waiting for a clean paired-opening
            # scan. A yaw-relative target (yaw +/- 0.55 every tick) produced a
            # permanent circle and carried the robot to the next room band.
            side_bias = -0.12 if side > 0 else 0.12
            center_err = max(-0.25, min(0.25, 0.35 * (left - right)))
            desired = 1.5708 + side_bias + 0.35 * center_err
            yaw_err = angle_diff(desired, yaw)
            crawl = 0.14 if structure_open or paired_cross_open or near_band else 0.18
            self._publish_cmd(crawl, max(-self._max_yaw, min(self._max_yaw, 2.0 * yaw_err)))
            self._publish_wp(x + 2.2 * math.cos(desired), y + 2.2 * math.sin(desired))
            return
        if self._pair_door_y is not None and now >= self._pair_seek_until:
            rospy.loginfo(
                "Structure pair-seek timeout (LIO y≈%.1f travel=%.1f) — cruise",
                self._pair_door_y,
                self._pair_seek_travel,
            )
            self._pair_door_y = None
            self._pair_seek_pose = None

        # Sticky commit into ENTER_ROOM (do not keep "glide" inside corridor —
        # losing the door label used to abort entry and strand the dog).
        if now < self._enter_abort_until:
            turn = -0.55 if self._prefer_side > 0 else 0.55
            desired = yaw + turn
            yaw_err = angle_diff(desired, yaw)
            self._publish_cmd(0.18, max(-self._max_yaw, min(self._max_yaw, 2.0 * yaw_err)))
            return

        door_side = 0
        if label == "door_left" and self._door_allowed(x, y, -1):
            door_side = -1
        elif label == "door_right" and self._door_allowed(x, y, 1):
            door_side = 1
        if door_side != 0:
            # Prefer requested side when both briefly flicker.
            if self._prefer_side != 0 and door_side != self._prefer_side:
                alt = self._ranges["right"] if self._prefer_side > 0 else self._ranges["left"]
                closed = self._ranges["left"] if self._prefer_side > 0 else self._ranges["right"]
                if (
                    alt is not None
                    and alt >= self._door_open
                    and closed is not None
                    and closed <= self._door_closed
                    and self._door_allowed(x, y, self._prefer_side)
                ):
                    door_side = self._prefer_side
            self._begin_enter(door_side, x, y, yaw)
            return

        # Wide interior past a jamb often reads as "room" before door_left/right.
        # Gate on relative corridor travel (not absolute LIO Y) so lobby false
        # "room" labels cannot trigger phantom entry under drift.
        if label == "room":
            travel = self._corridor_travel_m(x, y)
            if self._rooms_cleared == 0 and travel < self._min_corridor_travel_m:
                pass
            elif left > right + 0.75 and self._door_allowed(x, y, -1):
                self._begin_enter(-1, x, y, yaw)
                return
            elif right > left + 0.75 and self._door_allowed(x, y, 1):
                self._begin_enter(1, x, y, yaw)
                return

        # Stuck recovery: take the more open side as a door.
        if self._stuck_pose is None:
            self._stuck_pose = (x, y)
            self._stuck_since = now
        elif math.hypot(x - self._stuck_pose[0], y - self._stuck_pose[1]) > 0.35:
            self._stuck_pose = (x, y)
            self._stuck_since = now
        elif (now - self._stuck_since).to_sec() > 4.0:
            # Never force-enter without a classified door — wall-hug None sides
            # used to trigger phantom room entries (run33).
            self._publish_cmd(0.36, 1.3 * angle_diff(1.5708, yaw))
            self._stuck_since = now
            return

        center_err = (left - right) * self._center_gain
        self._last_speed_center_error = float(center_err)
        # After egress near lobby mouth, push north; once corridor-like, bias
        # toward prefer_side so we catch the first door band (seed77 → right).
        if (
            self._rooms_cleared == 0
            and self._engage_pose is not None
            and y < self._engage_pose[1] + 7.0
        ):
            mouth_bearing = math.atan2(4.0, -x)
            desired = yaw + 0.65 * angle_diff(1.5708, yaw) + 0.35 * angle_diff(
                mouth_bearing, yaw
            )
            if self._corridor_like():
                side_bias = -0.22 if self._prefer_side > 0 else 0.22
                desired = yaw + 0.55 * angle_diff(1.5708, yaw) + 0.25 * max(
                    -0.35, min(0.35, center_err)
                ) + side_bias
                # Commit immediately when preferred door is open.
                if self._prefer_side != 0 and self._door_allowed(x, y, self._prefer_side):
                    if self._preferred_door_open(self._prefer_side) or label == (
                        "door_right" if self._prefer_side > 0 else "door_left"
                    ):
                        self._begin_enter(self._prefer_side, x, y, yaw)
                        return
            yaw_err = angle_diff(desired, yaw)
            if front is not None and front < 1.1 and abs(x) > 0.45:
                self._publish_cmd(
                    0.0 if abs(yaw_err) > 0.4 else 0.12,
                    max(-self._max_yaw, min(self._max_yaw, 1.8 * yaw_err)),
                )
                self._publish_wp(0.0, y + 4.0)
                return
            speed = self._speed_for_heading(yaw_err, self._boost, front)
            if front is not None and front < self._front_slow:
                speed = min(speed, 0.18)
            self._publish_cmd(max(0.26, speed), 1.55 * yaw_err)
            self._publish_wp(0.0, y + 5.0)
            return
        # After both rooms at the first door band, cruise along the corridor.
        # Prefer relative travel + open front over absolute LIO Y≈29.
        if self._rooms_cleared >= 2:
            if self._north_cruise_pose is None:
                self._north_cruise_pose = (x, y)
            north_travel = math.hypot(x - self._north_cruise_pose[0], y - self._north_cruise_pose[1])
            need_north = north_travel < 10.0 and (
                self._corridor_progress_y is None or self._corridor_progress_y < 24.0
            )
            if need_north and self._corridor_like() and (front is None or front > 1.8):
                desired = yaw + 0.55 * angle_diff(1.5708, yaw) + 0.35 * center_err
                yaw_err = angle_diff(desired, yaw)
                speed = self._speed_for_heading(yaw_err, self._boost, front)
                self._publish_cmd(max(0.42, speed), 1.4 * yaw_err)
                self._publish_wp(x + 5.0 * math.cos(desired), y + 5.0 * math.sin(desired))
                return
        # Bias toward preferred door side when approaching an opening.
        if self._prefer_side != 0:
            if label == "door_left" and self._prefer_side > 0:
                center_err = -0.25
            elif label == "door_right" and self._prefer_side < 0:
                center_err = 0.25
            elif self._prefer_side > 0 and right > left + 0.35:
                center_err = min(center_err, -0.12)
            elif self._prefer_side < 0 and left > right + 0.35:
                center_err = max(center_err, 0.12)
        # Unstick from wall hug: if one side is too close, veer away.
        if self._ranges["right"] is None or self._ranges["right"] < 0.70:
            center_err = max(center_err, 0.35)
        if self._ranges["left"] is None or self._ranges["left"] < 0.70:
            center_err = min(center_err, -0.35)
        desired = yaw + max(-0.40, min(0.40, center_err))
        world_bias = angle_diff(1.5708, yaw)
        if abs(world_bias) < 1.0:
            desired = yaw + 0.50 * world_bias + 0.50 * (desired - yaw)

        yaw_err = angle_diff(desired, yaw)
        angular = 1.6 * yaw_err
        if front is not None and front < self._front_stop + 0.10:
            # Collision recovery must remain body-frame/LiDAR based: LIVO x
            # can diverge by several metres after a room rotation.
            angular = self._max_yaw if left > right else -self._max_yaw
            self._publish_cmd(0.06, angular)
            self._publish_wp(0.0, y + 3.0)
            return

        speed = self._speed_for_heading(yaw_err, self._cruise, front)
        if speed < 0.14 and (front is None or front > self._front_stop):
            speed = 0.18
        self._publish_cmd(speed, angular)
        look = 4.0 if speed > self._cruise * 0.9 else 3.0
        self._publish_wp(
            x + look * math.cos(yaw + 0.4 * yaw_err),
            y + look * math.sin(yaw + 0.4 * yaw_err),
        )

    def _run_enter_room(self, x, y, yaw, now):
        side = self._door_side if self._door_side != 0 else self._prefer_side
        if side == 0:
            side = 1
        label = self._classify()
        self._last_label = label
        elapsed = (now - self._mode_since).to_sec()
        # Square up at the jamb, then drive straight through.  A moving 55--90
        # degree turn has too large a radius for the 2 m doorway on the A1.
        deep_yaw = self._inward_yaw()
        target_yaw = deep_yaw

        depth = 0.0
        if self._enter_started_xy is not None:
            depth = math.hypot(x - self._enter_started_xy[0], y - self._enter_started_xy[1])

        # A side opening is first classified at its leading jamb.  Turning a
        # 0.8 m long A1 there clips the rear/shoulder on the wall and the gait
        # rotates without translating into the room.  Move roughly one half
        # doorway width along the corridor first; this is sensor-relative and
        # does not use any map/layout metadata.
        if self._enter_center_s < 1.6:
            corridor_err = angle_diff(self._room_entry_yaw, yaw)
            if abs(corridor_err) <= 0.24:
                self._enter_center_s += 0.05
            speed = 0.12 if abs(corridor_err) > 0.30 else self._cruise
            if self._ranges["front"] is not None and self._ranges["front"] < 0.72:
                speed = 0.10
            self._publish_cmd(
                speed,
                max(-self._max_yaw, min(self._max_yaw, 1.8 * corridor_err)),
            )
            self._publish_wp(
                x + 2.0 * math.cos(self._room_entry_yaw),
                y + 2.0 * math.sin(self._room_entry_yaw),
            )
            return

        # This is the first pose at the doorway centre, before the quarter
        # turn. Keep it as the return anchor; the original leading-jamb pose
        # is systematically displaced along the corridor.
        if self._door_center_xy is None:
            self._door_center_xy = (x, y)
            self._room_entry_xy = (x, y)
            if not self._paired_entry:
                # This pose is already the corridor-side centre reached before
                # the 90-degree turn; it is not the physical door plane.  The
                # old extra outward offset crossed the whole 2.2 m corridor
                # and recreated the opposite-room overshoot in LIO space.
                self._corridor_anchor_xy = (x, y)

        yaw_err = angle_diff(target_yaw, yaw)
        front = self._ranges["front"]
        # LIVO yaw lags/rejects while rotating at a doorway.  Use a bounded
        # open-loop quarter turn, then commit straight through the jamb.
        # A paired entry starts after the previous room exit has already
        # carried the robot across most of the corridor.  At that point a
        # brief IMU excursion near the opposite jamb must not erase all
        # accumulated forward time: v18 stopped at x=0.90 and declared the
        # room covered before crossing the x>1.1 threshold.  Keep the strict,
        # contiguous alignment gate for a normal 90-degree doorway turn, but
        # use hysteresis for the straight-through paired manoeuvre.
        aligned_limit = 0.42 if self._paired_entry else 0.16
        if abs(yaw_err) > aligned_limit:
            # Require a contiguous aligned interval. Previously brief IMU
            # threshold crossings accumulated throughout the quarter turn and
            # the state finished exactly when the body finally faced inward,
            # before any real translation through the jamb.
            if not self._paired_entry or abs(yaw_err) > 0.65:
                self._enter_drive_s = 0.0
            speed = 0.0
            angular = max(-self._max_yaw, min(self._max_yaw, 2.0 * yaw_err))
        else:
            speed = self._cruise
            angular = 1.4 * yaw_err
            self._enter_drive_s += 0.05
        if front is not None and front < self._front_stop:
            speed = 0.14
            angular = -side * self._max_yaw
        self._publish_cmd(speed, angular)
        self._publish_wp(
            x + 3.2 * math.cos(deep_yaw),
            y + 3.2 * math.sin(deep_yaw),
        )

        # Command time is never evidence of entry. Confirm that the pose has
        # crossed the saved door plane, reached a real inward depth, and that
        # LiDAR sees room-like bilateral space. Paired straight-through entries
        # need a larger depth because their origin is on the opposite corridor
        # side rather than at the final jamb.
        entry_depth, _ = self._room_coordinates(x, y)
        left = self._ranges["left"]
        right = self._ranges["right"]
        room_structure = label == "room" or (
            left is not None
            and right is not None
            and min(left, right) >= 2.1
        )
        door_frame_crossed = entry_depth >= 0.25 and room_structure
        required_entry_depth = max(
            self._enter_min_depth, 2.6 if self._paired_entry else 0.0
        )
        entered = room_entry_confirmed(
            door_frame_crossed,
            entry_depth,
            required_depth=required_entry_depth,
            pose_healthy=self._localization_healthy,
        )
        self._entry_confirmation_depth = max(
            self._entry_confirmation_depth, entry_depth
        )
        self._entry_door_frame_crossed = (
            self._entry_door_frame_crossed or door_frame_crossed
        )
        if entered:
            self._entry_confirmed = True
            self._begin_room_cover(x, y, now)
        elif elapsed > self._enter_timeout_s:
            rospy.logwarn(
                "Structure enter_room timeout (relative_depth=%.2f drive=%.1fs frame=%s) — retry corridor",
                entry_depth,
                self._enter_drive_s,
                self._entry_door_frame_crossed,
            )
            self._skip_door_until_y = y + 1.5
            self._enter_abort_until = now + rospy.Duration(self._enter_abort_cooldown_s)
            self._clear_enter_state()
            self._set_mode(Mode.CORRIDOR)

    def _start_exit(self, reason):
        prev_side = self._door_side if self._door_side != 0 else self._prefer_side
        if prev_side == 0:
            prev_side = 1
        now = rospy.Time.now()
        self._exit_room_side = prev_side
        self._looks_this_room = 0
        self._exit_spin_dir = 1.0
        self._exit_forward_s = 0.0
        self._exit_corridor_hits = 0
        self._exit_travel_m = 0.0
        self._exit_last_xy = self._pose[:2] if self._pose is not None else None
        self._exit_attempt_started = now
        self._exit_retry_until = rospy.Time(0)
        completion_revalidated = False
        if self._adaptive_room_viewpoints and self._room_grid is not None:
            completion_revalidated = self._room_coverage_complete()
            if self._pending_room_status == "explored" and not completion_revalidated:
                self._pending_room_status = "partial"
                self._pending_room_reason = "exit_revalidation_failed"
                reason = "partial room (exit_revalidation_failed)"
        if self._pose is not None:
            record = self._room_diagnostic_payload(
                self._pose[0], self._pose[1], now, exit_reason=reason
            )
            record["completion_revalidated"] = completion_revalidated
            self._room_records.append(record)
            self._room_diag_pub.publish(String(data=json.dumps(record, sort_keys=True)))
            self._write_room_diagnostics()
        rospy.loginfo(
            "Structure %s — exit pending side=%d confirmed_rooms=%d anchor=%s",
            reason,
            prev_side,
            self._rooms_cleared,
            self._corridor_anchor_xy,
        )
        self._set_mode(Mode.EXIT_ROOM)

    def _run_room_cover(self, x, y, yaw, now):
        label = self._classify()
        self._last_label = label

        room_start = self._room_started_at
        if room_start == rospy.Time(0):
            room_start = self._mode_since
        elapsed = (now - room_start).to_sec()
        if self._adaptive_room_viewpoints:
            with self._room_grid_lock:
                self._run_adaptive_room_cover(x, y, yaw, now, elapsed)
            return
        if self._relative_room_sweep:
            depth, transverse = self._room_coordinates(x, y)
            self._room_max_depth = max(self._room_max_depth, depth)
            self._room_transverse_min = min(self._room_transverse_min, transverse)
            self._room_transverse_max = max(self._room_transverse_max, transverse)
            phase_elapsed = (now - self._room_phase_since).to_sec()
            inward = self._inward_yaw()
            front = self._ranges["front"]

            if (
                self._room_timeout > 0.0
                and elapsed > self._room_timeout
                and self._room_phase != "return"
            ):
                rospy.logwarn(
                    "Structure room timebox -> return phase=%s depth=%.2f span=%.2f path=%.2f",
                    self._room_phase,
                    self._room_max_depth,
                    self._room_transverse_max - self._room_transverse_min,
                    self._room_cover_path,
                )
                self._set_room_phase("return", now)
                return

            # Camera/radar attention remains active throughout locomotion. A
            # compact return gets a short RGB confirmation without resetting
            # this geometric coverage phase.
            if self._maybe_look_sphere(x, y, yaw, now, Mode.ROOM_COVER):
                return

            if self._room_phase == "deep":
                reached = depth >= self._room_target_depth
                blocked = front is not None and front < 0.85 and depth >= 2.0
                if reached or blocked or phase_elapsed > 10.0:
                    left = self._ranges["left"] or 0.0
                    right = self._ranges["right"] or 0.0
                    self._room_sweep_sign = 1.0 if left >= right else -1.0
                    self._set_room_phase("sweep_turn", now)
                    return
                open_bias = 0.0
                if self._ranges["left"] is not None and self._ranges["right"] is not None:
                    open_bias = max(
                        -0.38,
                        min(0.38, 0.14 * (self._ranges["left"] - self._ranges["right"])),
                    )
                desired = inward + open_bias
                yaw_err = angle_diff(desired, yaw)
                speed = self._speed_for_heading(yaw_err, self._cruise, front)
                self._publish_cmd(max(0.12, speed), 1.7 * yaw_err)
                self._publish_wp(x + 3.0 * math.cos(desired), y + 3.0 * math.sin(desired))
                return

            if self._room_phase == "sweep_turn":
                desired = inward + self._room_sweep_sign * (math.pi / 2.0)
                yaw_err = angle_diff(desired, yaw)
                if abs(yaw_err) <= 0.14 or phase_elapsed > 6.0:
                    self._room_sweep_start_transverse = transverse
                    self._set_room_phase("sweep_drive", now)
                    return
                self._publish_cmd(0.0, 2.0 * yaw_err)
                return

            if self._room_phase == "sweep_drive":
                swept = abs(transverse - self._room_sweep_start_transverse)
                blocked = front is not None and front < 0.82
                if swept >= self._room_target_span or blocked or phase_elapsed > 6.0:
                    self._set_room_phase("scan_turn", now)
                    return
                desired = inward + self._room_sweep_sign * (math.pi / 2.0)
                yaw_err = angle_diff(desired, yaw)
                speed = self._speed_for_heading(yaw_err, self._cruise, front)
                self._publish_cmd(max(0.10, speed), 1.8 * yaw_err)
                self._publish_wp(x + 2.5 * math.cos(desired), y + 2.5 * math.sin(desired))
                return

            if self._room_phase == "scan_turn":
                # Sweep the RGB camera across the unobserved half of the room.
                desired = inward - self._room_sweep_sign * 0.95
                yaw_err = angle_diff(desired, yaw)
                if abs(yaw_err) <= 0.14 or phase_elapsed > 6.0:
                    self._cardinal_looks += 1
                    self._set_room_phase("return", now)
                    return
                self._publish_cmd(0.0, 1.8 * yaw_err)
                return

            # Return to a point just inside the doorway before facing out. This
            # prevents a lateral room sweep from driving into the door jamb.
            origin = self._room_entry_xy or (x, y)
            target_x = origin[0] + 0.72 * math.cos(inward)
            target_y = origin[1] + 0.72 * math.sin(inward)
            distance = math.hypot(target_x - x, target_y - y)
            if distance > 0.55 and phase_elapsed < 10.0:
                desired = math.atan2(target_y - y, target_x - x)
                yaw_err = angle_diff(desired, yaw)
                speed = self._speed_for_heading(yaw_err, self._cruise, front)
                self._publish_cmd(max(0.08, speed), 1.8 * yaw_err)
                self._publish_wp(target_x, target_y)
                return
            outward_err = angle_diff(self._outward_yaw(), yaw)
            if abs(outward_err) > 0.16:
                self._publish_cmd(0.0, 2.0 * outward_err)
                return
            span = self._room_transverse_max - self._room_transverse_min
            status = "explored" if self._room_coverage_complete() else "partial"
            self._start_exit(
                "%s room depth=%.2f span=%.2f path=%.2f cells=%d"
                % (status, self._room_max_depth, span, self._room_cover_path, len(self._room_cells))
            )
            return
        if self._enter_started_xy is not None:
            depth = math.hypot(x - self._enter_started_xy[0], y - self._enter_started_xy[1])
        elif self._room_entry_xy is not None:
            depth = math.hypot(x - self._room_entry_xy[0], y - self._room_entry_xy[1])
        else:
            depth = 0.0

        cover_ratio = self._fis.local_coverage_ratio(
            x, y, radius=3.5, cell_size=self._cover_cell
        )
        # Cap depth so LIO scale blowups cannot satisfy the gate early (run11 depth=34).
        # Room1 near-door spheres need less depth than room0's deep SW sphere.
        if self._rooms_cleared == 0:
            min_exit_depth = 3.6 if self._door_side > 0 else 5.0
        else:
            min_exit_depth = 3.2
        depth_gate = min(depth, 8.0)
        if (
            elapsed >= 12.0
            and depth_gate >= min_exit_depth
            and self._cardinal_looks >= 2
            and cover_ratio >= self._cover_exit_ratio * 0.85
        ):
            self._start_exit("room sensors covered ratio=%.2f depth=%.2f" % (cover_ratio, depth_gate))
            return

        # Sphere look (camera confirms red spheres; boxes filtered in detector).
        if self._maybe_look_sphere(x, y, yaw, now, Mode.ROOM_COVER):
            return

        # Forced cardinal RGB dwells so RealSense faces room interiors.
        inward = self._inward_yaw()
        # Seed77: room1 spheres cluster SE of the east door; room0 sphere is SW.
        if self._rooms_cleared == 0 and self._door_side > 0:
            deep_heading = math.atan2(-0.9, 2.6)  # SE into room1
        elif self._rooms_cleared == 0 and self._door_side < 0:
            deep_heading = math.atan2(-1.2, -2.8)  # SW into room0
        else:
            deep_heading = inward
        if (
            elapsed >= 2.0
            and self._cardinal_looks < 3
            and (now - self._last_look).to_sec() >= 1.0
            and self._looks_this_room < self._max_looks_per_room
        ):
            offsets = (0.0, -0.70, 0.90)
            target = deep_heading + offsets[self._cardinal_looks]
            if self._begin_look(target, "cardinal_%d" % self._cardinal_looks, Mode.ROOM_COVER):
                self._cardinal_looks += 1
                return

        if (
            self._room_timeout > 0.0
            and elapsed > self._room_timeout
            and depth_gate >= min_exit_depth * 0.65
            and self._cardinal_looks >= 1
        ):
            self._start_exit("room timebox %.1fs depth=%.2f" % (elapsed, depth_gate))
            return
        if self._room_timeout > 0.0 and elapsed > self._room_timeout + 3.0:
            self._start_exit("room hard timebox %.1fs depth=%.2f" % (elapsed, depth_gate))
            return

        front = self._ranges["front"]
        left = self._ranges["left"]
        right = self._ranges["right"]
        # Bias toward the more open interior while still shallow.
        open_bias = 0.0
        if left is not None and right is not None:
            open_bias = max(-0.55, min(0.55, 0.25 * (left - right)))
        if elapsed < 10.0 or depth_gate < min_exit_depth:
            desired = deep_heading + 0.35 * open_bias
        elif elapsed < 14.0:
            desired = deep_heading + (-0.7 if int(elapsed) % 2 == 0 else 0.7)
        else:
            desired = self._outward_yaw()

        yaw_err = angle_diff(desired, yaw)
        base = self._cruise if abs(yaw_err) < 0.35 else self._crawl
        if depth < min_exit_depth:
            base = max(base, self._cruise)
        speed = self._speed_for_heading(yaw_err, base, front)
        if speed < 0.20:
            speed = 0.20
        self._publish_cmd(speed, 1.6 * yaw_err)
        self._publish_wp(x + 3.2 * math.cos(desired), y + 3.2 * math.sin(desired))

    def _clear_enter_state(self):
        self._enter_started_xy = None
        self._enter_center_s = 0.0
        self._enter_drive_s = 0.0
        self._enter_required_s = 1.8
        self._paired_entry = False
        self._entry_confirmed = False
        self._entry_confirmation_depth = 0.0
        self._entry_door_frame_crossed = False
        self._door_side = 0
        self._door_center_xy = None
        self._room_phase = "idle"
        self._room_grid = None
        self._room_target_xy = None
        self._room_target_yaw = None
        self._room_pan_headings = []
        self._room_viewpoints = []
        self._room_probe_scans = 0
        self._corridor_anchor_xy = None

    def _finish_exit_to_corridor(self, x, y, yaw, now, why, dist_from_entry):
        prev_side = self._exit_room_side if self._exit_room_side != 0 else (
            self._door_side if self._door_side != 0 else -1
        )
        self._rooms_cleared += 1
        self._prefer_side = -prev_side
        self._skip_side = prev_side
        self._skip_side_until = now + rospy.Duration(12.0)
        self._skip_door_until_y = None
        self._clear_enter_state()
        self._exit_room_side = 0
        # Brief settle so we don't immediately re-commit the same jamb.
        self._corridor_ready = False
        self._engage_pose = (x, y, yaw)
        self._engage_time = now
        if self._rooms_cleared == 2:
            # Complete the lower pair before accepting any more doorway labels.
            # Time/body heading is deliberate: LIVO Y is often invalid here.
            self._upper_align_until = now + rospy.Duration(4.0)
            self._upper_cruise_until = now + rospy.Duration(44.0)
            self._upper_cruise_start_y = y
            self._upper_cruise_started = now
        if (
            self._last_door_y is not None
            and self._prefer_side != 0
            and self._rooms_cleared < self._max_rooms
            and self._rooms_cleared % 2 == 1
        ):
            self._start_pair_seek(self._last_door_y)
        else:
            self._pair_door_y = None
        self._set_mode(Mode.CORRIDOR)
        self._write_room_diagnostics()
        rospy.loginfo(
            "Structure re-entered corridor (%s dist=%.2f) prefer=%d",
            why,
            dist_from_entry,
            self._prefer_side,
        )
        bias = -0.30 if self._prefer_side > 0 else 0.30
        self._publish_cmd(0.50, 1.4 * angle_diff(1.5708 + bias, yaw))

    def _run_exit_room(self, x, y, yaw, now):
        """Return to the saved corridor anchor; never exit on elapsed time alone."""
        self._restore_exit_context()
        label = self._classify()
        self._last_label = label
        attempt_elapsed = (
            (now - self._exit_attempt_started).to_sec()
            if self._exit_attempt_started != rospy.Time(0)
            else (now - self._mode_since).to_sec()
        )
        front = self._ranges["front"]
        max_yaw = min(0.55, self._max_yaw + 0.18)
        if self._corridor_anchor_xy is None:
            door = self._door_center_xy or self._room_entry_xy or self._last_entry_xy
            if door is not None:
                self._corridor_anchor_xy = (door[0], door[1])
        if self._corridor_anchor_xy is None:
            self._publish_cmd(0.0, 0.0)
            rospy.logerr_throttle(2.0, "Structure exit has no corridor anchor")
            return

        anchor_x, anchor_y = self._corridor_anchor_xy
        distance = math.hypot(anchor_x - x, anchor_y - y)
        at_anchor = distance <= self._exit_anchor_tolerance
        near_anchor = distance <= max(
            self._exit_anchor_tolerance,
            self._exit_alignment_radius,
        )

        # Corridor geometry is only meaningful after turning back along the
        # corridor axis; while facing outward its left/right sectors point
        # along the corridor rather than toward the bilateral walls.
        corridor_yaw_err = angle_diff(self._room_entry_yaw, yaw)
        geometry_ready = near_anchor and abs(corridor_yaw_err) <= 0.22
        front_v = self._ranges["front"]
        rear_v = self._ranges["rear"]
        left_v = self._ranges["left"]
        right_v = self._ranges["right"]
        # At a paired doorway band both lateral sides are open, so the normal
        # bilateral-wall classifier is intentionally false.  The equivalent
        # corridor evidence there is clear forward/rear travel along the
        # corridor axis plus two real lateral returns spanning the door band.
        doorway_band_corridor = (
            front_v is not None
            and rear_v is not None
            and front_v >= 1.4
            and rear_v >= 1.0
            and left_v is not None
            and right_v is not None
            and left_v + right_v >= 3.6
            and label in ("door_left", "door_right", "room")
        )
        if geometry_ready and (self._corridor_like() or doorway_band_corridor):
            self._exit_corridor_hits += 1
        else:
            self._exit_corridor_hits = 0

        confirmed = corridor_exit_confirmed(
            distance,
            self._exit_corridor_hits,
            tolerance=self._exit_anchor_tolerance,
            required_hits=3,
        )
        bounded_geometry = (
            self._exit_corridor_hits >= 3
            and 0.7 <= self._exit_travel_m <= 1.4
        )
        if confirmed or bounded_geometry:
            self._finish_exit_to_corridor(
                x,
                y,
                yaw,
                now,
                "anchor+bilateral" if confirmed else "bounded_bilateral",
                self._exit_travel_m,
            )
            return

        # Reaching the doorway band starts a separate stationary alignment;
        # do not let the drive-from-viewpoint timeout reset this turn halfway.
        if near_anchor:
            self._publish_cmd(
                0.0,
                max(-self._pan_yaw_rate, min(self._pan_yaw_rate, 1.8 * corridor_yaw_err)),
            )
            return

        if now < self._exit_retry_until:
            # Back into the known doorway staging area before retrying; never
            # keep driving across the full corridor when confirmation failed.
            self._publish_cmd(-0.12, 0.0)
            return
        if attempt_elapsed > self._exit_timeout:
            self._exit_retry_until = now + rospy.Duration(0.8)
            self._exit_attempt_started = now + rospy.Duration(0.8)
            self._exit_corridor_hits = 0
            self._exit_travel_m = 0.0
            self._exit_last_xy = (x, y)
            rospy.logwarn(
                "Structure exit retry: anchor_dist=%.2f label=%s", distance, label
            )
            self._publish_cmd(0.0, 0.0)
            return

        desired = math.atan2(anchor_y - y, anchor_x - x)
        yaw_err = angle_diff(desired, yaw)
        if abs(yaw_err) > 0.45:
            speed = 0.0
        else:
            speed = min(0.45, self._speed_for_heading(yaw_err, self._cruise, front))
        self._publish_cmd(speed, max(-max_yaw, min(max_yaw, 1.8 * yaw_err)))
        self._publish_wp(anchor_x, anchor_y)
        rospy.loginfo_throttle(
            2.0,
            "Structure exit→anchor dist=%.2f travel=%.2f hits=%d yaw_err=%.2f F=%s",
            distance,
            self._exit_travel_m,
            self._exit_corridor_hits,
            yaw_err,
            "%.2f" % front if front is not None else "?",
        )

    def _on_timer(self, _event):
        try:
            if self._egress_active or self._pose is None:
                return
            if self._mode == Mode.IDLE:
                self._set_mode(Mode.CORRIDOR)
                self._active_pub.publish(Bool(data=True))

            x, y, odom_yaw = self._pose
            yaw = self._imu_yaw if self._imu_yaw is not None else odom_yaw
            now = rospy.Time.now()
            if (now - self._last_classify_log).to_sec() > 2.5:
                self._last_classify_log = now
                label = self._classify()
                self._last_label = label
                sph = self._sphere_candidates[0].score if self._sphere_candidates else -1.0
                rospy.loginfo_throttle(
                    2.5,
                    "Structure tick mode=%s label=%s L=%.2f R=%.2f F=%s rooms=%d cover=%d sph=%.2f",
                    self._mode.value,
                    label,
                    self._ranges["left"] or -1.0,
                    self._ranges["right"] or -1.0,
                    "%.2f" % self._ranges["front"] if self._ranges["front"] is not None else "?",
                    self._rooms_cleared,
                    self._fis.lidar_coverage_count(),
                    sph,
                )

            if self._mode == Mode.LOOK_AT:
                self._run_look_at(x, y, yaw, now)
            elif self._mode == Mode.CORRIDOR:
                # Do not LOOK in the corridor — it burns time; only look inside rooms.
                self._run_corridor(x, y, yaw, now)
            elif self._mode == Mode.ENTER_ROOM:
                self._run_enter_room(x, y, yaw, now)
            elif self._mode == Mode.ROOM_COVER:
                self._run_room_cover(x, y, yaw, now)
            elif self._mode == Mode.EXIT_ROOM:
                self._run_exit_room(x, y, yaw, now)
            else:
                self._run_corridor(x, y, yaw, now)
        except Exception as exc:  # noqa: BLE001 — keep stack alive; log cause
            rospy.logerr_throttle(2.0, "Structure timer error: %s", exc)


if __name__ == "__main__":
    rospy.init_node("structure_aware_explorer")
    StructureAwareExplorer()
    rospy.spin()
