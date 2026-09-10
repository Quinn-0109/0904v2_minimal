#!/usr/bin/env python3
"""Return, stop at the black-wall observation distance, turn right, then detect stairs.

This top-level sideband reuses the preserved StairEntryServo motion interface.
It never publishes /cmd_vel or /joy and never consumes Gazebo/layout truth.
The stair detector is deliberately gated off until the relative right turn has
completed and has been verified from the robot IMU.
"""
import importlib.util
import json
import math
import os
import time
from collections import deque

import numpy as np
import rospy
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Float32, String


_spec = importlib.util.spec_from_file_location(
    "simenv_stair_entry_visual_servo_base",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "stair_entry_visual_servo.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


class ReturnTurnEntry(_base.StairEntryServo):
    def __init__(self):
        self.detector_enabled = False
        self.turn_start_imu = None
        self.turn_peak_delta = 0.0
        super().__init__()
        self.wall_min = float(rospy.get_param("~wall_distance_min_m", 0.85))
        self.wall_max = float(rospy.get_param("~wall_distance_max_m", 1.65))
        # The old return phase drove to one fixed map coordinate.  Its final
        # diagonal correction could push the robot into the wall when the
        # realised yaw varied between runs.  Keep ``home`` only as a corridor
        # bearing reference and stop from fresh RGB-D range observations.
        self.return_wall_stop_distance = float(rospy.get_param(
            "~return_wall_stop_distance_m", 1.55))
        self.return_wall_confirmations = int(rospy.get_param(
            "~return_wall_confirmations", 3))
        self.return_goal_overshoot = float(rospy.get_param(
            "~return_goal_overshoot_m", 3.0))
        self.return_monitor_start_distance = float(rospy.get_param(
            "~return_monitor_start_distance_m", 0.50))
        self.return_monitor_heading_tolerance = float(rospy.get_param(
            "~return_monitor_heading_tolerance_rad", 0.15))
        self.return_past_anchor_guard = float(rospy.get_param(
            "~return_past_anchor_guard_m", 0.60))
        # Optional top-level shortcut for experiments that spawn directly in
        # the corridor.  It bypasses the old map-coordinate return bearing and
        # drives along the measured startup yaw until the same RGB-D wall stop
        # triggers.  The preserved return mode remains the default.
        self.direct_forward_wall_mode = bool(rospy.get_param(
            "~direct_forward_wall_mode", False))
        self.direct_forward_goal_distance = float(rospy.get_param(
            "~direct_forward_goal_distance_m", 14.0))
        self.direct_forward_guard_distance = float(rospy.get_param(
            "~direct_forward_guard_distance_m", 12.0))
        # Optional terminal arc length for the directly planned corridor leg.
        # This is measured from the live FAST-LIO start pose along the chosen
        # route heading; it is not a Gazebo/world coordinate.  RGB-D remains
        # an earlier obstacle stop, while this endpoint prevents missing or
        # max-range depth returns from driving beyond the planned path.
        self.direct_forward_stop_progress = float(rospy.get_param(
            "~direct_forward_stop_progress_m", 0.0))
        # Keep the corridor transit on one measured map-frame line instead of
        # aiming once at a far point.  A far target makes cross-track error
        # weak near the start, so small gait/yaw variation can shift the later
        # stair approach by several tenths of a metre.  Rolling lookahead goals
        # preserve continuous motion while repeatedly correcting that error.
        self.direct_forward_line_tracking = bool(rospy.get_param(
            "~direct_forward_line_tracking_enabled", False))
        self.direct_forward_lookahead = float(rospy.get_param(
            "~direct_forward_lookahead_m", 2.0))
        self.direct_forward_update_period = float(rospy.get_param(
            "~direct_forward_update_period_sec", 0.10))
        self.direct_forward_reference_heading = rospy.get_param(
            "~direct_forward_reference_heading_rad", None)
        # Independent side-view stair landmark. RGB-D supplies a metric point,
        # FAST-LIO places it in map, and the mission projects it onto the
        # measured corridor line. This replaces black-door/fixed-distance
        # positioning without changing the public motion interface.
        self.side_projection_stop_enabled = bool(rospy.get_param(
            "~side_stair_projection_stop_enabled", False))
        self.side_projection_confirmations = int(rospy.get_param(
            "~side_stair_projection_confirmations", 3))
        self.side_projection_max_spread = float(rospy.get_param(
            "~side_stair_projection_max_spread_m", 0.50))
        self.side_projection_min_forward = float(rospy.get_param(
            "~side_stair_projection_min_forward_m", 0.80))
        self.side_projection_max_forward = float(rospy.get_param(
            "~side_stair_projection_max_forward_m", 6.00))
        self.side_projection_min_right = float(rospy.get_param(
            "~side_stair_projection_min_right_m", 0.45))
        self.side_projection_max_right = float(rospy.get_param(
            "~side_stair_projection_max_right_m", 5.00))
        self.side_projection_stop_offset = float(rospy.get_param(
            "~side_stair_projection_stop_offset_m", 0.0))
        self.side_projection_min_lead = float(rospy.get_param(
            "~side_stair_projection_min_lead_m", 0.35))
        self.side_projection_samples = deque(maxlen=max(
            3, self.side_projection_confirmations + 2))
        self.side_projection_lock = None
        self.side_projection_sequence = 0
        self.corridor_side_detection_enabled = False
        # The longitudinal stop is measured from the front RGB-D wall patch,
        # but it is armed only after the independent right-side stair profile
        # has been observed.  This prevents an ordinary corridor wall from
        # becoming a parking landmark and works on upper floors where the wall
        # is not the dark movable door used on floor one.
        self.require_stair_context_for_wall_stop = bool(rospy.get_param(
            "~require_stair_context_for_wall_stop", False))
        self.stair_context_confirmations = int(rospy.get_param(
            "~stair_context_confirmations", 2))
        self.stair_context_count = 0
        self.stair_context_latched = False
        self.stair_context_first_stamp = None
        self.stair_context_last = None
        # The former centre-depth stop is not repeatable once the dark door
        # opens: the centre pixels then become no-return depth.  The opened
        # doorway is, however, a strong RGB landmark (dark upper field over a
        # bright floor).  This optional sideband trigger stops on consecutive
        # fresh camera frames; the odometry endpoint remains only an overrun
        # guard.  It consumes no simulator/layout truth.
        self.door_visual_stop_enabled = bool(rospy.get_param(
            "~door_visual_stop_enabled", False))
        self.door_black_threshold = float(rospy.get_param(
            "~door_black_intensity_threshold", 35.0))
        self.door_black_fraction_threshold = float(rospy.get_param(
            "~door_black_fraction_threshold", 0.80))
        self.door_visual_confirmations = int(rospy.get_param(
            "~door_visual_confirmations", 2))
        self.door_black_fraction = None
        self.door_visual_sample_sequence = 0
        self.right_turn = float(rospy.get_param("~right_turn_rad", math.pi / 2.0))
        self.turn_verify_min = float(rospy.get_param("~turn_verify_min_rad", 1.20))
        self.turn_verify_final = float(rospy.get_param("~turn_verify_final_rad", 1.00))
        self.turn_max_attempts = int(rospy.get_param("~turn_max_attempts", 3))
        # Optional fast 90-degree turn implemented entirely through the
        # existing public goal interface.  A virtual goal is continuously kept
        # at the robot's instantaneous right side, so the executor supplies
        # yaw while translation stays close to zero.  No /cmd_vel or /joy is
        # published by this mission.
        self.fast_goal_turn_enabled = bool(rospy.get_param(
            "~fast_goal_turn_enabled", False))
        self.fast_turn_virtual_goal = float(rospy.get_param(
            "~fast_turn_virtual_goal_m", 4.0))
        self.fast_turn_lateral_limit = float(rospy.get_param(
            "~fast_turn_lateral_limit_mps", 0.02))
        self.fast_turn_restore_lateral_limit = float(rospy.get_param(
            "~fast_turn_restore_lateral_limit_mps", 0.30))
        self.fast_turn_tolerance = float(rospy.get_param(
            "~fast_turn_tolerance_rad", 0.10))
        self.fast_turn_target_offset = float(rospy.get_param(
            "~fast_turn_target_offset_rad", 0.0))
        self.fast_turn_update_period = float(rospy.get_param(
            "~fast_turn_update_period_sec", 0.08))
        self.fast_turn_timeout = float(rospy.get_param(
            "~fast_turn_timeout_sec", 10.0))
        self.resume_turn_reference_imu = rospy.get_param(
            "~resume_turn_reference_imu", None)
        self.entry_standoff = float(rospy.get_param("~entry_standoff_m", 0.35))
        self.entry_wall_side_offset = float(rospy.get_param(
            "~entry_wall_side_offset_m", 0.20))
        self.require_wall_step_midpoint = bool(rospy.get_param(
            "~require_wall_step_midpoint", False))
        self.entry_arrival_tolerance = float(rospy.get_param(
            "~entry_arrival_tolerance_m", 0.35))
        # The old approach split one straight line into short legs.  That did
        # not change its geometry and the line still crossed the near pillar.
        # This optional top-level path first reaches the visually detected
        # aisle lateral coordinate close to the robot, then follows that aisle
        # forward to the locked entrance point.
        self.entry_aisle_path_enabled = bool(rospy.get_param(
            "~entry_aisle_path_enabled", False))
        self.entry_stage_forward = float(rospy.get_param(
            "~entry_stage_forward_m", 0.60))
        self.entry_stage_lateral_fraction = float(rospy.get_param(
            "~entry_stage_lateral_fraction", 1.0))
        # Optional continuous polyline execution.  The stage point remains the
        # collision-avoiding corner, but the final goal replaces it before the
        # executor brakes to a stop.  This keeps path geometry unchanged while
        # removing the artificial stop-and-go delay at the internal waypoint.
        self.entry_pass_through_enabled = bool(rospy.get_param(
            "~entry_pass_through_enabled", False))
        self.entry_pass_through_radius = float(rospy.get_param(
            "~entry_pass_through_radius_m", 0.24))
        self.entry_pass_through_timeout = float(rospy.get_param(
            "~entry_pass_through_timeout_sec", 6.0))
        self.entry_stage_goal_extension = float(rospy.get_param(
            "~entry_stage_goal_extension_m", 0.0))
        self.entry_stage_min_progress_fraction = float(rospy.get_param(
            "~entry_stage_min_progress_fraction", 0.0))
        # Keep the post-turn aisle path in the mission's planned map-frame
        # direction.  Projecting camera points with the instantaneous body yaw
        # makes a small turn overshoot become a lateral path shift near the
        # stair.  ``None`` preserves the legacy behaviour for other callers.
        self.entry_path_reference_heading = rospy.get_param(
            "~entry_path_reference_heading_rad", None)
        self.post_turn_settle = float(rospy.get_param(
            "~post_turn_settle_sec", 0.35))
        self.profiled_final_enabled = bool(rospy.get_param(
            "~profiled_final_enabled", False))
        self.profiled_cruise_speed = float(rospy.get_param(
            "~profiled_cruise_speed_mps", 1.8))
        self.profiled_approach_speed = float(rospy.get_param(
            "~profiled_approach_speed_mps", 1.0))
        self.profiled_brake_distance = float(rospy.get_param(
            "~profiled_brake_distance_m", 1.0))
        # Stable longitudinal parking reference for the stair entrance.  The
        # staircase side profile first provides semantics; the colour-agnostic
        # RGB-D plane of the physical wall ahead then provides metric range.
        # Combining route progress with that range estimates one fixed wall
        # station in map coordinates.  The robot parks at a configurable
        # stand-off before that station, so no black-door appearance or world
        # coordinate is required (a grey upper-floor wall is equivalent).
        self.entry_wall_station_stop_enabled = bool(rospy.get_param(
            "~entry_wall_station_stop_enabled", False))
        self.entry_wall_station_standoff = float(rospy.get_param(
            "~entry_wall_station_standoff_m", 1.20))
        self.entry_wall_station_confirmations = int(rospy.get_param(
            "~entry_wall_station_confirmations", 5))
        self.entry_wall_station_max_spread = float(rospy.get_param(
            "~entry_wall_station_max_spread_m", 0.28))
        self.entry_wall_station_min_progress_span = float(rospy.get_param(
            "~entry_wall_station_min_progress_span_m", 0.65))
        self.entry_wall_station_min_support = float(rospy.get_param(
            "~entry_wall_station_min_support", 0.55))
        self.entry_wall_station_min_lead = float(rospy.get_param(
            "~entry_wall_station_min_lead_m", 0.65))
        self.entry_wall_station_min_observation_lead = float(rospy.get_param(
            "~entry_wall_station_min_observation_lead_m", 3.00))
        self.entry_wall_station_required_clusters = int(rospy.get_param(
            "~entry_wall_station_required_clusters", 2))
        self.entry_wall_station_cluster_separation = float(rospy.get_param(
            "~entry_wall_station_cluster_separation_m", 1.50))
        self.entry_wall_plane_min_depth = float(rospy.get_param(
            "~entry_wall_plane_min_depth_m", 0.70))
        self.entry_wall_plane_max_depth = float(rospy.get_param(
            "~entry_wall_plane_max_depth_m", 4.80))
        self.entry_wall_plane_bin_width = float(rospy.get_param(
            "~entry_wall_plane_bin_width_m", 0.10))
        self.entry_wall_plane_band = float(rospy.get_param(
            "~entry_wall_plane_band_m", 0.16))
        self.wall_range_safety_only = bool(rospy.get_param(
            "~wall_range_safety_only", False))
        self.mission_started_monotonic = None
        self.wall_depths = deque(maxlen=20)
        self.wall_sample_sequence = 0
        self.entry_wall_plane_depth = None
        self.entry_wall_plane_support = None
        self.entry_wall_plane_sequence = 0
        self.entry_wall_station_samples = deque(maxlen=max(
            8, self.entry_wall_station_confirmations + 4))
        self.entry_wall_station_lock = None
        self.entry_wall_station_clusters = []
        self.lateral_limit_pub = rospy.Publisher(
            "/simenv/goal_lateral_speed_limit", Float32, queue_size=1)
        self.speed_limit_pub = rospy.Publisher(
            "/simenv/goal_speed_limit", Float32, queue_size=1)
        rospy.Subscriber("/real_sense/depth/image_raw", Image,
                         self.on_depth, queue_size=2)
        rospy.Subscriber("/real_sense/rgb/image_raw", Image,
                         self.on_corridor_rgb, queue_size=2)

    def on_det(self, message):
        try:
            detection = json.loads(message.data)
        except (TypeError, ValueError):
            return
        # Corridor images still cannot become entry goals.  A structural
        # side-view stair hypothesis is used only to arm the independent
        # forward-wall range stop.
        side = detection.get("side_stair_candidate")
        if self.corridor_side_detection_enabled and isinstance(side, dict):
            structure_ok = bool(side.get("structure_ok"))
            if structure_ok:
                self.stair_context_count += 1
                self.stair_context_last = {
                    "stamp": detection.get("stamp"),
                    "tread_count": side.get("tread_count"),
                    "side_count": side.get("side_count"),
                    "anchor_pixel": side.get("anchor_pixel"),
                    "reason": side.get("reason"),
                }
                if self.stair_context_first_stamp is None:
                    self.stair_context_first_stamp = detection.get("stamp")
                if self.stair_context_count >= max(
                        1, self.stair_context_confirmations):
                    self.stair_context_latched = True
            elif not self.stair_context_latched:
                self.stair_context_count = 0

        # Corridor images still cannot become entry goals. Only the detector's
        # separate side-view output is consumed as a geometric landmark.
        self.consume_side_stair_landmark(detection)
        # Hard phase gate: corridor/turn images cannot become goals.
        if not self.detector_enabled:
            return
        # ``detected_stable`` is intentionally hysteretic in the visual
        # detector and may stay true for a few frames after the current stair
        # hypothesis has failed.  A navigation goal must come from a fresh,
        # currently valid post-turn frame, not that stale stable latch.
        if not detection.get("detected_raw"):
            return
        if not detection.get("structure", {}).get("ok"):
            return
        if not detection.get("depth", {}).get("ok"):
            return
        if (self.require_wall_step_midpoint and
                not isinstance(detection.get("aisle_midpoint_camera_point"), list)):
            return
        self.det = detection

    def consume_side_stair_landmark(self, detection):
        if (not self.corridor_side_detection_enabled or self.odom is None or
                not isinstance(detection, dict)):
            return
        side = detection.get("side_stair_candidate")
        if (not isinstance(side, dict) or not side.get("detected_raw") or
                side.get("reason") != "ok"):
            return
        point = side.get("camera_point")
        if not isinstance(point, list) or len(point) != 3:
            return
        try:
            right, down, forward = [float(v) for v in point]
        except (TypeError, ValueError):
            return
        if not (math.isfinite(right) and math.isfinite(forward) and
                self.side_projection_min_forward <= forward <=
                self.side_projection_max_forward and
                self.side_projection_min_right <= right <=
                self.side_projection_max_right):
            return
        ox, oy, _oz, yaw = self.odom
        landmark = (
            ox + forward*math.cos(yaw) + right*math.sin(yaw),
            oy + forward*math.sin(yaw) - right*math.cos(yaw),
        )
        sample = {
            "map": list(landmark),
            "camera_point": [right, down, forward],
            "odom": list(self.odom),
            "stamp": detection.get("stamp"),
            "confidence": side.get("confidence"),
            "anchor_pixel": side.get("anchor_pixel"),
        }
        self.side_projection_samples.append(sample)
        required = max(3, self.side_projection_confirmations)
        if len(self.side_projection_samples) < required:
            return
        recent = list(self.side_projection_samples)[-required:]
        mx = float(np.median([s["map"][0] for s in recent]))
        my = float(np.median([s["map"][1] for s in recent]))
        spread = max(math.hypot(s["map"][0]-mx, s["map"][1]-my)
                     for s in recent)
        if spread > self.side_projection_max_spread:
            return
        self.side_projection_lock = {
            "map": [mx, my],
            "spread_m": spread,
            "sample_count": required,
            "samples": recent,
        }
        self.side_projection_sequence += 1

    def on_imu(self, message):
        super().on_imu(message)
        if self.turn_start_imu is not None and self.imu is not None:
            delta = abs(_base.wrap(self.imu - self.turn_start_imu))
            self.turn_peak_delta = max(self.turn_peak_delta, delta)

    def on_depth(self, message):
        try:
            if not message.data or message.height <= 0 or message.width <= 0:
                return
            if message.encoding in ("32FC1", "32FC"):
                cols = message.step // 4
                arr = np.frombuffer(message.data, dtype=np.float32).reshape(
                    message.height, cols)[:, :message.width]
            elif message.encoding in ("16UC1", "mono16"):
                cols = message.step // 2
                arr = np.frombuffer(message.data, dtype=np.uint16).reshape(
                    message.height, cols)[:, :message.width].astype(np.float32) * 0.001
            else:
                return
            h, w = arr.shape
            roi = arr[int(.42*h):int(.68*h), int(.38*w):int(.62*w)]
            valid = roi[np.isfinite(roi) & (roi > .15) & (roi < 10.0)]
            if valid.size >= 30:
                self.wall_depths.append(float(np.median(valid)))
                self.wall_sample_sequence += 1

            # Central entrance-wall ROI.  Colour is irrelevant; only a stable
            # metric depth plane is used.  Select the densest depth mode rather
            # than a raw median, which can be pulled between wall and floor.
            wall_roi = arr[int(.36*h):int(.68*h),
                           int(.34*w):int(.66*w)]
            wall_valid = wall_roi[
                np.isfinite(wall_roi) &
                (wall_roi >= self.entry_wall_plane_min_depth) &
                (wall_roi <= self.entry_wall_plane_max_depth)]
            if wall_valid.size >= 120:
                width = max(0.04, self.entry_wall_plane_bin_width)
                bins = np.arange(self.entry_wall_plane_min_depth,
                                 self.entry_wall_plane_max_depth + 2.0*width,
                                 width)
                counts, edges = np.histogram(wall_valid, bins=bins)
                if counts.size:
                    index = int(np.argmax(counts))
                    centre = 0.5*(edges[index] + edges[index+1])
                    band = wall_valid[np.abs(wall_valid-centre) <=
                                      max(width, self.entry_wall_plane_band)]
                    minimum_support = max(80, int(.08*wall_valid.size))
                    if band.size >= minimum_support:
                        self.entry_wall_plane_depth = float(np.median(band))
                        self.entry_wall_plane_support = float(
                            band.size / max(1, wall_valid.size))
                        self.entry_wall_plane_sequence += 1
        except (ValueError, TypeError):
            return

    def on_corridor_rgb(self, message):
        """Measure the opened dark doorway in a fixed central image ROI."""
        if not self.door_visual_stop_enabled:
            return
        try:
            if not message.data or message.height <= 0 or message.width <= 0:
                return
            channels = max(1, message.step // message.width)
            if channels < 3:
                return
            image = np.frombuffer(message.data, dtype=np.uint8).reshape(
                message.height, message.step)[:, :message.width*channels]
            image = image.reshape(message.height, message.width, channels)
            gray = image[:, :, :3].astype(np.float32).mean(axis=2)
            h, w = gray.shape
            roi = gray[int(.05*h):int(.75*h), int(.15*w):int(.85*w)]
            if roi.size < 100:
                return
            self.door_black_fraction = float(
                np.mean(roi < self.door_black_threshold))
            self.door_visual_sample_sequence += 1
        except (ValueError, TypeError):
            return

    def wall_distance(self, timeout=4.0):
        # Do not reuse samples captured before the latest motion leg.
        self.wall_depths.clear()
        end = time.time() + timeout
        while not rospy.is_shutdown() and time.time() < end:
            if len(self.wall_depths) >= 5:
                return float(np.median(list(self.wall_depths)[-10:]))
            rospy.sleep(.1)
        return None

    def return_until_black_wall(self):
        """Drive the corridor path until its endpoint or an earlier RGB-D stop.

        A single goal beyond the former observation point keeps the controller
        moving continuously.  A fresh wall-range threshold atomically preempts
        it with a current-pose goal through the same public goal interface.
        """
        confirmations = 0
        start = self.odom
        if self.direct_forward_wall_mode:
            heading = (start[3] if self.direct_forward_reference_heading is None
                       else float(self.direct_forward_reference_heading))
            ux, uy = math.cos(heading), math.sin(heading)
            anchor_distance = self.direct_forward_guard_distance
            travel = max(self.direct_forward_goal_distance,
                         anchor_distance + 0.50)
            bearing_anchor = None
            guard_progress_limit = anchor_distance
            motion_mode = "startup_heading_forward"
        else:
            dx = self.home[0]-start[0]
            dy = self.home[1]-start[1]
            anchor_distance = math.hypot(dx, dy)
            if anchor_distance < 0.50:
                self.records.append({"name": "return_wall_range_guard",
                                     "reason": "return_bearing_too_short",
                                     "odom": self.odom})
                return False
            ux, uy = dx/anchor_distance, dy/anchor_distance
            heading = math.atan2(uy, ux)
            travel = anchor_distance+self.return_goal_overshoot
            bearing_anchor = list(self.home)
            guard_progress_limit = (anchor_distance+
                                    self.return_past_anchor_guard)
            motion_mode = "home_bearing"
        gx, gy = start[0]+ux*travel, start[1]+uy*travel
        self.side_projection_samples.clear()
        self.side_projection_lock = None
        self.side_projection_sequence = 0
        self.corridor_side_detection_enabled = True
        self.stair_context_count = 0
        self.stair_context_latched = False
        self.stair_context_first_stamp = None
        self.stair_context_last = None
        projection_target = None
        entry_wall_target = None
        last_side_sequence = 0
        last_entry_wall_sequence = self.entry_wall_plane_sequence
        self.entry_wall_station_samples.clear()
        self.entry_wall_station_lock = None
        self.entry_wall_station_clusters = []

        # Start one continuous transit.  Optional rolling lookahead replaces
        # only the active target; it never inserts an intermediate stop.
        self.goal_result = None
        goal = _base.PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = self.frame
        goal.pose.position.x = gx
        goal.pose.position.y = gy
        goal.pose.orientation = _base.quat(heading)
        self.wall_depths.clear()
        last_sample_sequence = self.wall_sample_sequence
        last_visual_sequence = self.door_visual_sample_sequence
        visual_confirmations = 0
        self.goal_pub.publish(goal)
        last_line_goal_update = time.monotonic()
        line_goal_updates = 1
        self.records.append({
            "name": "continuous_return_goal",
            "goal": [gx, gy, heading],
            "bearing_anchor": bearing_anchor,
            "start_odom": start,
            "motion_mode": motion_mode,
            "stop_rule": (
                "stair_context_entry_wall_station"
                if self.entry_wall_station_stop_enabled else
                ("side_stair_projection_or_safety_fallback"
                 if self.side_projection_stop_enabled else
                 "fresh_depth_threshold")),
        })

        end = time.time()+self.goal_timeout
        while not rospy.is_shutdown() and time.time() < end:
            ox, oy, _oz, robot_yaw = self.odom
            travelled = math.hypot(ox-start[0], oy-start[1])
            progress_along_return = ((ox-start[0])*ux +
                                     (oy-start[1])*uy)
            heading_ok = abs(_base.wrap(robot_yaw-heading)) <= \
                self.return_monitor_heading_tolerance

            # Convert one temporally stable side-view stair landmark into an
            # along-corridor station.  Orthogonal projection removes lateral
            # stair distance, leaving the point where the robot should stop
            # before turning toward the stair.  During diagnostic runs the
            # target is recorded but does not preempt motion.
            if (projection_target is None and
                    self.side_projection_sequence != last_side_sequence and
                    isinstance(self.side_projection_lock, dict)):
                last_side_sequence = self.side_projection_sequence
                lock = self.side_projection_lock
                lx, ly = lock["map"]
                landmark_progress = ((lx-start[0])*ux +
                                     (ly-start[1])*uy)
                candidate_target = (landmark_progress +
                                    self.side_projection_stop_offset)
                lead = candidate_target-progress_along_return
                cross_track = ((lx-start[0])*(-uy) +
                               (ly-start[1])*ux)
                accepted = (lead >= self.side_projection_min_lead and
                            candidate_target <= guard_progress_limit)
                self.records.append({
                    "name": "side_stair_projection_candidate",
                    "accepted": accepted,
                    "landmark": list(lock["map"]),
                    "landmark_progress_m": landmark_progress,
                    "stop_offset_m": self.side_projection_stop_offset,
                    "projected_stop_progress_m": candidate_target,
                    "current_progress_m": progress_along_return,
                    "lead_m": lead,
                    "cross_track_m": cross_track,
                    "lock": lock,
                    "start_odom": start,
                    "route_heading_rad": heading,
                })
                if accepted:
                    projection_target = candidate_target

            # Estimate the longitudinal station of the wall beside the stair
            # entrance.  Each candidate is expressed as route progress plus
            # the forward component of the fresh RGB-D plane range.  A static
            # wall therefore produces the same station while the robot moves;
            # transient floor/stair modes do not and fail the spread gate.
            if (self.entry_wall_station_stop_enabled and
                    self.entry_wall_plane_sequence !=
                    last_entry_wall_sequence):
                last_entry_wall_sequence = self.entry_wall_plane_sequence
                depth = self.entry_wall_plane_depth
                heading_error = _base.wrap(robot_yaw-heading)
                context_ok = (not self.require_stair_context_for_wall_stop or
                              self.stair_context_latched)
                candidate_ok = (
                    context_ok and heading_ok and depth is not None and
                    math.isfinite(depth) and
                    self.entry_wall_plane_min_depth <= depth <=
                    self.entry_wall_plane_max_depth)
                station = None
                if candidate_ok:
                    station = (progress_along_return +
                               depth*math.cos(heading_error))
                    self.entry_wall_station_samples.append({
                        "station_m": float(station),
                        "progress_m": float(progress_along_return),
                        "depth_m": float(depth),
                        "heading_error_rad": float(heading_error),
                        "support": self.entry_wall_plane_support,
                    })
                required = max(3, self.entry_wall_station_confirmations)
                recent = list(self.entry_wall_station_samples)[-required:]
                if len(recent) >= required:
                    median_station = float(np.median(
                        [s["station_m"] for s in recent]))
                    spread = max(abs(s["station_m"]-median_station)
                                 for s in recent)
                    progress_span = (max(s["progress_m"] for s in recent) -
                                     min(s["progress_m"] for s in recent))
                    median_support = float(np.median([
                        s["support"] if s["support"] is not None else 0.0
                        for s in recent]))
                    candidate_target = (median_station -
                                        self.entry_wall_station_standoff)
                    lead = candidate_target-progress_along_return
                    wall_observation_lead = (median_station-
                                             progress_along_return)
                    accepted = (
                        spread <= self.entry_wall_station_max_spread and
                        progress_span >=
                        self.entry_wall_station_min_progress_span and
                        median_support >=
                        self.entry_wall_station_min_support and
                        wall_observation_lead >=
                        self.entry_wall_station_min_observation_lead and
                        lead >= self.entry_wall_station_min_lead and
                        candidate_target <= guard_progress_limit)
                    if accepted:
                        observation = {
                            "wall_station_m": median_station,
                            "parking_progress_m": candidate_target,
                            "standoff_m": self.entry_wall_station_standoff,
                            "spread_m": spread,
                            "progress_span_m": progress_span,
                            "median_support": median_support,
                            "samples": recent,
                        }
                        nearest = None
                        if self.entry_wall_station_clusters:
                            nearest = min(
                                self.entry_wall_station_clusters,
                                key=lambda cluster: abs(
                                    cluster["wall_station_m"]-
                                    median_station))
                        if (nearest is None or
                                abs(nearest["wall_station_m"]-
                                    median_station) >=
                                self.entry_wall_station_cluster_separation):
                            self.entry_wall_station_clusters.append(
                                observation)
                            self.entry_wall_station_clusters.sort(
                                key=lambda cluster:
                                cluster["wall_station_m"])
                        if len(self.entry_wall_station_clusters) >= max(
                                1, self.entry_wall_station_required_clusters):
                            self.entry_wall_station_lock = dict(
                                self.entry_wall_station_clusters[-1])
                            self.entry_wall_station_lock["clusters"] = list(
                                self.entry_wall_station_clusters)
                            entry_wall_target = self.entry_wall_station_lock[
                                "parking_progress_m"]
                self.records.append({
                    "name": "entry_wall_station_sample",
                    "accepted_sample": candidate_ok,
                    "plane_depth_m": depth,
                    "plane_support": self.entry_wall_plane_support,
                    "candidate_station_m": station,
                    "required_progress_span_m": (
                        self.entry_wall_station_min_progress_span),
                    "required_support": self.entry_wall_station_min_support,
                    "sample_count": len(self.entry_wall_station_samples),
                    "cluster_count": len(self.entry_wall_station_clusters),
                    "clusters": list(self.entry_wall_station_clusters),
                    "locked_target_m": entry_wall_target,
                    "lock": self.entry_wall_station_lock,
                    "progress_m": progress_along_return,
                    "stair_context_latched": self.stair_context_latched,
                })

            if (self.direct_forward_wall_mode and
                    self.direct_forward_line_tracking and
                    time.monotonic()-last_line_goal_update >=
                    max(0.03, self.direct_forward_update_period)):
                # The lookahead point lies on the original measured route
                # line, not on the robot's drifted instantaneous heading.
                target_progress = min(
                    travel,
                    (entry_wall_target
                     if (self.entry_wall_station_stop_enabled and
                         entry_wall_target is not None) else travel),
                    (projection_target
                     if (self.side_projection_stop_enabled and
                         projection_target is not None) else travel),
                    max(0.0, progress_along_return) +
                    max(0.60, self.direct_forward_lookahead))
                gx = start[0] + ux*target_progress
                gy = start[1] + uy*target_progress
                goal = _base.PoseStamped()
                goal.header.stamp = rospy.Time.now()
                goal.header.frame_id = self.frame
                goal.pose.position.x = gx
                goal.pose.position.y = gy
                goal.pose.orientation = _base.quat(heading)
                self.goal_pub.publish(goal)
                last_line_goal_update = time.monotonic()
                line_goal_updates += 1

            if (self.entry_wall_station_stop_enabled and
                    entry_wall_target is not None and
                    progress_along_return >= entry_wall_target-0.10):
                stop_pose = self.odom
                stopped = self.send_goal(
                    "stair_entry_wall_station_stop", stop_pose[0],
                    stop_pose[1], stop_pose[3])
                self.records.append({
                    "name": "stair_entry_wall_station_stop_verified",
                    "success": stopped,
                    "planned_progress_m": entry_wall_target,
                    "actual_progress_m": progress_along_return,
                    "start_odom": start,
                    "stop_odom": self.odom,
                    "lock": self.entry_wall_station_lock,
                    "line_goal_updates": line_goal_updates,
                    "source": "stair_context_entrance_rgbd_wall_plane",
                })
                return stopped

            if (self.side_projection_stop_enabled and
                    projection_target is not None and
                    progress_along_return >= projection_target):
                stop_pose = self.odom
                stopped = self.send_goal(
                    "side_stair_projection_stop", stop_pose[0],
                    stop_pose[1], stop_pose[3])
                self.records.append({
                    "name": "side_stair_projection_stop_verified",
                    "success": stopped,
                    "planned_progress_m": projection_target,
                    "actual_progress_m": progress_along_return,
                    "start_odom": start,
                    "stop_odom": self.odom,
                    "line_goal_updates": line_goal_updates,
                    "source": "rgbd_stair_landmark_corridor_projection",
                })
                return stopped

            # A direct route has a finite endpoint.  The depth stream is still
            # allowed to stop earlier for an unexpected close obstacle, but a
            # missing/no-return wall pixel cannot extend motion past the route.
            if (self.direct_forward_wall_mode and
                    self.direct_forward_stop_progress > 0.0 and
                    progress_along_return >=
                    self.direct_forward_stop_progress):
                stop_pose = self.odom
                stopped = self.send_goal(
                    "direct_corridor_endpoint_stop", stop_pose[0],
                    stop_pose[1], stop_pose[3])
                self.records.append({
                    "name": "direct_corridor_endpoint_stop_verified",
                    "success": stopped,
                    "planned_progress_m": self.direct_forward_stop_progress,
                    "actual_progress_m": progress_along_return,
                    "start_odom": start,
                    "stop_odom": self.odom,
                    "line_goal_updates": line_goal_updates,
                    "depth_role": "early_obstacle_stop_only",
                })
                return stopped

            # Prefer the repeatable appearance of the opened doorway over its
            # invalid centre depth.  The ROI threshold was separated by more
            # than 0.94 between the saved start frame and the verified stop
            # frame; consecutive-frame confirmation rejects transient shadows.
            if (self.door_visual_stop_enabled and
                    self.door_visual_sample_sequence != last_visual_sequence):
                last_visual_sequence = self.door_visual_sample_sequence
                fraction = self.door_black_fraction
                stair_context_ok = (
                    not self.require_stair_context_for_wall_stop or
                    self.stair_context_latched)
                armed = (travelled >= self.return_monitor_start_distance and
                         heading_ok and stair_context_ok)
                matched = (armed and fraction is not None and
                           fraction >= self.door_black_fraction_threshold)
                visual_confirmations = (visual_confirmations+1
                                        if matched else 0)
                self.records.append({
                    "name": "door_visual_stream_sample",
                    "black_fraction": fraction,
                    "black_fraction_threshold": (
                        self.door_black_fraction_threshold),
                    "monitor_armed": armed,
                    "stair_context_required": (
                        self.require_stair_context_for_wall_stop),
                    "stair_context_latched": self.stair_context_latched,
                    "stair_context_count": self.stair_context_count,
                    "stair_context_last": self.stair_context_last,
                    "confirmation_count": visual_confirmations,
                    "travelled_m": travelled,
                    "odom": self.odom,
                })
                if visual_confirmations >= self.door_visual_confirmations:
                    stop_pose = self.odom
                    stopped = self.send_goal(
                        "door_visual_stop", stop_pose[0], stop_pose[1],
                        stop_pose[3])
                    self.records.append({
                        "name": "door_visual_stop_verified",
                        "success": stopped,
                        "black_fraction": fraction,
                        "black_fraction_threshold": (
                            self.door_black_fraction_threshold),
                        "planned_progress_guard_m": (
                            self.direct_forward_stop_progress),
                        "actual_progress_m": progress_along_return,
                        "start_odom": start,
                        "stop_odom": self.odom,
                        "line_goal_updates": line_goal_updates,
                    })
                    return stopped

            # Consume each camera sample at most once.  This is a true
            # consecutive-frame confirmation, not repeated polling of one frame.
            if self.wall_sample_sequence != last_sample_sequence:
                last_sample_sequence = self.wall_sample_sequence
                distance = (self.wall_depths[-1]
                            if self.wall_depths else None)
                stair_context_ok = (
                    not self.require_stair_context_for_wall_stop or
                    self.stair_context_latched)
                armed = (travelled >= self.return_monitor_start_distance and
                         heading_ok and stair_context_ok)
                below = (armed and distance is not None and
                         math.isfinite(distance) and
                         distance <= self.return_wall_stop_distance)
                confirmations = confirmations+1 if below else 0
                self.records.append({
                    "name": "return_wall_stream_sample",
                    "distance_m": distance,
                    "stop_threshold_m": self.return_wall_stop_distance,
                    "monitor_armed": armed,
                    "stair_context_required": (
                        self.require_stair_context_for_wall_stop),
                    "stair_context_latched": self.stair_context_latched,
                    "stair_context_count": self.stair_context_count,
                    "stair_context_last": self.stair_context_last,
                    "travelled_m": travelled,
                    "heading_error_rad": _base.wrap(robot_yaw-heading),
                    "confirmation_count": confirmations,
                    "odom": self.odom,
                })
                if confirmations >= self.return_wall_confirmations:
                    stop_pose = self.odom
                    stopped = self.send_goal("stair_front_wall_range_stop",
                                             stop_pose[0], stop_pose[1],
                                             stop_pose[3])
                    self.records.append({
                        "name": "stair_front_wall_range_stop_verified",
                        "success": stopped,
                        "distance_m": distance,
                        "stop_threshold_m": self.return_wall_stop_distance,
                        "start_odom": start,
                        "stop_odom": self.odom,
                        "source": (
                            "front_wall_depth_with_side_stair_context"),
                        "stair_context_latched": (
                            self.stair_context_latched),
                        "stair_context_first_stamp": (
                            self.stair_context_first_stamp),
                        "stair_context_last": self.stair_context_last,
                        "line_tracking": self.direct_forward_line_tracking,
                        "line_reference_heading_rad": heading,
                        "line_goal_updates": line_goal_updates,
                    })
                    return (False if self.wall_range_safety_only else stopped)

            # This bound is reached only when the depth stream cannot produce
            # a trustworthy stop. Preempt safely rather than driving into wall.
            if progress_along_return >= guard_progress_limit:
                stop_pose = self.odom
                self.send_goal("return_wall_guard_stop",
                               stop_pose[0], stop_pose[1], stop_pose[3])
                self.records.append({
                    "name": "return_wall_range_guard",
                    "reason": "no_confirmed_wall_range_after_anchor_guard",
                    "progress_along_return_m": progress_along_return,
                    "anchor_distance_m": anchor_distance,
                    "guard_progress_limit_m": guard_progress_limit,
                    "motion_mode": motion_mode,
                    "odom": self.odom,
                })
                return False

            result = self.goal_result
            result_goal = (result.get("goal", {})
                           if isinstance(result, dict) else {})
            if (isinstance(result, dict) and
                    abs(result_goal.get("x", 1e9)-gx) < .06 and
                    abs(result_goal.get("y", 1e9)-gy) < .06):
                if (self.entry_wall_station_stop_enabled and
                        entry_wall_target is not None and
                        entry_wall_target-progress_along_return <= 0.30 and
                        bool(result.get("success"))):
                    self.records.append({
                        "name": "stair_entry_wall_station_stop_verified",
                        "success": True,
                        "planned_progress_m": entry_wall_target,
                        "actual_progress_m": progress_along_return,
                        "start_odom": start,
                        "stop_odom": self.odom,
                        "lock": self.entry_wall_station_lock,
                        "line_goal_updates": line_goal_updates,
                        "source": (
                            "stair_context_entrance_rgbd_wall_plane_goal_reached"),
                    })
                    return True
                self.records.append({
                    "name": "continuous_return_ended_before_wall_stop",
                    "result": result, "odom": self.odom})
                return False
            rospy.sleep(.03)

        stop_pose = self.odom
        self.send_goal("return_wall_timeout_stop",
                       stop_pose[0], stop_pose[1], stop_pose[3])
        self.records.append({"name": "return_wall_range_guard",
                             "reason": "continuous_return_timeout",
                             "odom": self.odom})
        return False

    def turn_right_and_verify(self, reference_imu=None):
        if self.fast_goal_turn_enabled:
            return self.fast_goal_turn_right(reference_imu)
        if self.imu is None:
            return False
        self.turn_start_imu = (self.imu if reference_imu is None
                               else float(reference_imu))
        self.turn_peak_delta = abs(_base.wrap(self.imu-self.turn_start_imu))
        start_odom = self.odom
        attempts = []
        last_result = {"success": True}
        for attempt in range(self.turn_max_attempts):
            before_delta = abs(_base.wrap(self.imu-self.turn_start_imu))
            current_delta = abs(_base.wrap(self.imu-self.turn_start_imu))
            if (self.turn_peak_delta >= self.turn_verify_min and
                    current_delta >= self.turn_verify_final):
                break
            remaining = max(0.0, self.right_turn-current_delta)
            # The RL gait realizes roughly half of an open-loop yaw request in
            # this scene. Recompute from feedback after every same-direction
            # attempt; never issue a reverse correction.
            command_angle = min(self.right_turn, max(0.28, 2.0*remaining))
            request_id = "stair_explicit_right_90_%02d" % attempt
            self.scan_result = None
            self.scan_pub.publish(String(data=json.dumps({
                "request_id": request_id,
                "angle_rad": command_angle,
                "angular_speed": self.turn_speed,
                "direction": -1.0,
                "timeout_sec": command_angle / self.turn_speed + 8.0,
            })))
            end = time.time() + command_angle / self.turn_speed + 12.0
            result = None
            while not rospy.is_shutdown() and time.time() < end:
                if (isinstance(self.scan_result, dict) and
                        self.scan_result.get("request_id") == request_id):
                    result = self.scan_result
                    break
                rospy.sleep(.05)
            attempts.append({"request_id": request_id,
                             "command_angle_rad": command_angle,
                             "result": result,
                             "end_imu_yaw": self.imu,
                             "cumulative_delta_rad": abs(_base.wrap(
                                 self.imu-self.turn_start_imu))})
            last_result = result
            after_delta = abs(_base.wrap(self.imu-self.turn_start_imu))
            # A scan timeout is not itself a reason to accept an incomplete
            # turn, nor to abandon a turn that made measurable progress.  Use
            # the independent IMU delta and continue only in the same direction.
            if ((not result or not result.get("success")) and
                    after_delta < before_delta + 0.08):
                break
        final_delta = (abs(_base.wrap(self.imu - self.turn_start_imu))
                       if self.imu is not None else 0.0)
        self.records.append({
            "name": "explicit_right_turn_90",
            "requested_rad": self.right_turn,
            "start_imu_yaw": self.turn_start_imu,
            "end_imu_yaw": self.imu,
            "peak_delta_rad": self.turn_peak_delta,
            "final_delta_rad": final_delta,
            "start_odom": start_odom,
            "end_odom": self.odom,
            "attempts": attempts,
        })
        self.turn_start_imu = None
        # The scan executor may report ``local_rescan_timeout`` after the
        # commanded body turn has already completed.  The mission transition
        # is therefore decided by the independent IMU feedback above; the
        # scan result remains recorded for diagnosis but cannot veto a
        # physically verified turn.
        return bool(self.turn_peak_delta >= self.turn_verify_min and
                    final_delta >= self.turn_verify_final)

    def fast_goal_turn_right(self, reference_imu=None):
        """Turn right using only the existing exploration-goal interface.

        Re-publishing a virtual goal at the instantaneous body-right bearing
        keeps heading error near -pi/2.  The executor therefore applies its
        normal bounded yaw command; a very small public lateral-speed limit
        prevents the virtual goal from becoming a translation manoeuvre.
        IMU feedback, not elapsed open-loop time, terminates the action.
        """
        if self.imu is None or self.odom is None:
            return False
        self.turn_start_imu = (self.imu if reference_imu is None
                               else float(reference_imu))
        self.turn_peak_delta = abs(_base.wrap(self.imu-self.turn_start_imu))
        start_odom = self.odom
        started = time.monotonic()
        target_delta = max(
            0.10, self.right_turn+self.fast_turn_target_offset-
            self.fast_turn_tolerance)
        self.lateral_limit_pub.publish(Float32(
            data=max(0.001, self.fast_turn_lateral_limit)))
        rospy.sleep(0.10)

        signed_delta = 0.0
        updates = 0
        while not rospy.is_shutdown():
            signed_delta = _base.wrap(self.turn_start_imu-self.imu)
            self.turn_peak_delta = max(self.turn_peak_delta, abs(signed_delta))
            if signed_delta >= target_delta:
                break
            if time.monotonic()-started >= self.fast_turn_timeout:
                break
            ox, oy, _oz, robot_yaw = self.odom
            distance = max(1.0, self.fast_turn_virtual_goal)
            # Global coordinates of the robot's instantaneous body-right ray.
            gx = ox + distance*math.sin(robot_yaw)
            gy = oy - distance*math.cos(robot_yaw)
            goal = _base.PoseStamped()
            goal.header.stamp = rospy.Time.now()
            goal.header.frame_id = self.frame
            goal.pose.position.x = gx
            goal.pose.position.y = gy
            goal.pose.orientation = _base.quat(robot_yaw-self.right_turn)
            self.goal_pub.publish(goal)
            updates += 1
            rospy.sleep(max(0.02, self.fast_turn_update_period))

        # Replace the virtual goal atomically with the measured current pose.
        stop_pose = self.odom
        stopped = self.send_goal("fast_right_turn_stop", stop_pose[0],
                                 stop_pose[1], stop_pose[3])
        self.lateral_limit_pub.publish(Float32(
            data=max(0.001, self.fast_turn_restore_lateral_limit)))
        final_delta = _base.wrap(self.turn_start_imu-self.imu)
        elapsed = time.monotonic()-started
        success = bool(stopped and final_delta >= target_delta and
                       self.turn_peak_delta <= self.right_turn+0.35)
        self.records.append({
            "name": "fast_goal_right_turn_90",
            "success": success,
            "requested_rad": self.right_turn,
            "target_offset_rad": self.fast_turn_target_offset,
            "target_delta_rad": target_delta,
            "start_imu_yaw": self.turn_start_imu,
            "end_imu_yaw": self.imu,
            "final_right_delta_rad": final_delta,
            "peak_delta_rad": self.turn_peak_delta,
            "duration_sec": elapsed,
            "virtual_goal_updates": updates,
            "lateral_speed_limit_mps": self.fast_turn_lateral_limit,
            "start_odom": start_odom,
            "end_odom": self.odom,
        })
        self.turn_start_imu = None
        return success

    def entry_ground_map(self, detection):
        midpoint = detection.get("aisle_midpoint_camera_point")
        if isinstance(midpoint, list) and len(midpoint) >= 3:
            entry_right, _down, midpoint_forward = [
                float(v) for v in midpoint[:3]]
        else:
            entry_right, _down, midpoint_forward = [
                float(v) for v in detection["entry_camera_point"]]
        step_right, _step_down, step_forward = [
            float(v) for v in detection["first_step_camera_point"]]
        # Keep the lateral coordinate anchored to the *observed free-floor*
        # pixel below the first riser, then add only a small wall-side bias.
        # Offsetting from the step centre itself is unstable: when the riser
        # appears near the image edge a fixed offset can push the goal through
        # the wall.  Forward distance still comes from the first riser, so the
        # goal remains physically in front of the step rather than under it.
        if isinstance(midpoint, list) and len(midpoint) >= 3:
            right = entry_right
            forward = max(0.20, midpoint_forward)
            target_policy = "locked_wall_step_free_space_midpoint"
        else:
            right = entry_right - self.entry_wall_side_offset
            forward = max(0.20, step_forward - self.entry_standoff)
            target_policy = "legacy_entry_offset"
        ox, oy, _oz, robot_yaw = self.odom
        path_yaw = (robot_yaw if self.entry_path_reference_heading is None
                    else float(self.entry_path_reference_heading))
        target = (ox + forward*math.cos(path_yaw) + right*math.sin(path_yaw),
                  oy + forward*math.sin(path_yaw) - right*math.cos(path_yaw))
        self.records.append({"name": "entry_ground_policy",
                             "policy": target_policy,
                             "camera_forward_m": forward,
                             "camera_right_m": right,
                             "target": target})
        return target

    def stair_boundary_map(self, detection):
        """Transform the wall-side boundary of the first stair into map.

        The free-floor midpoint is the translation target, while this separate
        physical boundary is used only for the final facing direction.  Fall
        back to the preserved first-step landmark for legacy detector output.
        """
        point = detection.get("stair_boundary_camera_point")
        if not isinstance(point, list) or len(point) < 3:
            return self.first_step_map(detection)
        right, _down, forward = [float(v) for v in point[:3]]
        ox, oy, _oz, robot_yaw = self.odom
        path_yaw = (robot_yaw if self.entry_path_reference_heading is None
                    else float(self.entry_path_reference_heading))
        return (ox + forward*math.cos(path_yaw) + right*math.sin(path_yaw),
                oy + forward*math.sin(path_yaw) - right*math.cos(path_yaw))

    def camera_ground_to_map(self, origin, forward, right):
        ox, oy, _oz, robot_yaw = origin
        path_yaw = (robot_yaw if self.entry_path_reference_heading is None
                    else float(self.entry_path_reference_heading))
        return (ox + forward*math.cos(path_yaw) + right*math.sin(path_yaw),
                oy + forward*math.sin(path_yaw) - right*math.cos(path_yaw))

    def send_profiled_goal(self, name, x, y, heading):
        """Send one goal with a continuous cruise-to-approach speed profile."""
        self.goal_result = None
        self.speed_limit_pub.publish(Float32(
            data=max(0.05, self.profiled_cruise_speed)))
        goal = _base.PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = self.frame
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation = _base.quat(heading)
        self.goal_pub.publish(goal)
        started = time.monotonic()
        brake_pose = None
        while not rospy.is_shutdown() and \
                time.monotonic()-started < self.goal_timeout:
            remaining = math.hypot(x-self.odom[0], y-self.odom[1])
            if (brake_pose is None and
                    remaining <= max(0.20, self.profiled_brake_distance)):
                brake_pose = self.odom
                self.speed_limit_pub.publish(Float32(
                    data=max(0.05, self.profiled_approach_speed)))
            result = self.goal_result
            result_goal = (result.get("goal", {})
                           if isinstance(result, dict) else {})
            if (isinstance(result, dict) and
                    result.get("success") is not None and
                    abs(result_goal.get("x", 1e9)-x) < .06 and
                    abs(result_goal.get("y", 1e9)-y) < .06):
                self.records.append({
                    "name": name, "goal": [x, y, heading],
                    "odom": self.odom, "result": result,
                    "speed_profile": {
                        "cruise_mps": self.profiled_cruise_speed,
                        "approach_mps": self.profiled_approach_speed,
                        "brake_distance_m": self.profiled_brake_distance,
                        "brake_pose": brake_pose,
                    },
                })
                return bool(result["success"])
            rospy.sleep(.02)
        self.records.append({
            "name": name, "goal": [x, y, heading], "odom": self.odom,
            "result": {"success": False, "reason": "timeout"},
            "speed_profile": {"brake_pose": brake_pose},
        })
        return False

    def execute_entry_aisle_path(self, detection, entry_ground, first_step):
        """Execute a two-segment visual path that clears the near pillar."""
        midpoint = detection.get("aisle_midpoint_camera_point")
        if not isinstance(midpoint, list) or len(midpoint) < 3:
            return False
        right = float(midpoint[0])
        forward = float(midpoint[2])
        origin = self.odom
        stage_forward = min(max(0.30, self.entry_stage_forward),
                            max(0.30, forward-0.45))
        stage_right = right*self.entry_stage_lateral_fraction
        stage = self.camera_ground_to_map(origin, stage_forward, stage_right)
        path = [stage, entry_ground]
        self.records.append({
            "name": "entry_aisle_path_plan",
            "policy": "early_lateral_alignment_then_aisle_forward",
            "origin": origin,
            "camera_midpoint": [right, float(midpoint[1]), forward],
            "stage_camera": [stage_right, 0.0, stage_forward],
            "waypoints": [list(point) for point in path],
            "first_step": first_step,
        })
        started = time.monotonic()
        if self.entry_pass_through_enabled:
            ox, oy, _oz, _yaw = self.odom
            stage_dx, stage_dy = stage[0]-ox, stage[1]-oy
            stage_length = max(1e-6, math.hypot(stage_dx, stage_dy))
            stage_heading = math.atan2(stage_dy, stage_dx)
            extension = max(0.0, self.entry_stage_goal_extension)
            control_stage = (
                stage[0] + extension*stage_dx/stage_length,
                stage[1] + extension*stage_dy/stage_length,
            )
            self.goal_result = None
            if self.profiled_final_enabled:
                self.speed_limit_pub.publish(Float32(
                    data=max(0.05, self.profiled_cruise_speed)))
            goal = _base.PoseStamped()
            goal.header.stamp = rospy.Time.now()
            goal.header.frame_id = self.frame
            goal.pose.position.x = control_stage[0]
            goal.pose.position.y = control_stage[1]
            goal.pose.orientation = _base.quat(stage_heading)
            self.goal_pub.publish(goal)
            switch_started = time.monotonic()
            switch_pose = None
            while not rospy.is_shutdown():
                remaining = math.hypot(stage[0]-self.odom[0],
                                       stage[1]-self.odom[1])
                progress_fraction = (
                    (self.odom[0]-ox)*stage_dx +
                    (self.odom[1]-oy)*stage_dy) / (stage_length*stage_length)
                if (remaining <= max(0.16, self.entry_pass_through_radius)
                        and progress_fraction >=
                        max(0.0, self.entry_stage_min_progress_fraction)):
                    switch_pose = self.odom
                    break
                if time.monotonic()-switch_started >= \
                        self.entry_pass_through_timeout:
                    self.records.append({
                        "name": "entry_aisle_path_result", "success": False,
                        "reason": "stage_pass_through_timeout",
                        "stage_remaining_m": remaining,
                        "duration_sec": time.monotonic()-started,
                        "odom": self.odom,
                    })
                    return False
                rospy.sleep(.02)
            self.records.append({
                "name": "entry_aisle_pass_through_switch",
                "stage": list(stage),
                "control_stage": list(control_stage),
                "stage_goal_extension_m": extension,
                "stage_progress_fraction": progress_fraction,
                "minimum_stage_progress_fraction": (
                    self.entry_stage_min_progress_fraction),
                "switch_radius_m": self.entry_pass_through_radius,
                "switch_pose": switch_pose,
                "duration_sec": time.monotonic()-switch_started,
            })
            ox, oy, _oz, _yaw = self.odom
            heading = math.atan2(entry_ground[1]-oy,
                                 entry_ground[0]-ox)
            final_ok = (self.send_profiled_goal(
                "entry_aisle_final", entry_ground[0], entry_ground[1], heading)
                if self.profiled_final_enabled else self.send_goal(
                    "entry_aisle_final", entry_ground[0], entry_ground[1], heading))
            if not final_ok:
                self.records.append({
                    "name": "entry_aisle_path_result", "success": False,
                    "reason": "final_goal_failed",
                    "duration_sec": time.monotonic()-started,
                    "odom": self.odom,
                })
                return False
            remaining = math.hypot(entry_ground[0]-self.odom[0],
                                   entry_ground[1]-self.odom[1])
            success = remaining <= max(self.executor_tolerance,
                                       self.entry_arrival_tolerance)
            self.records.append({
                "name": "entry_aisle_path_result", "success": success,
                "policy": "continuous_stage_pass_through",
                "duration_sec": time.monotonic()-started,
                "remaining_m": remaining, "odom": self.odom,
            })
            return success

        for index, point in enumerate(path):
            ox, oy, _oz, _yaw = self.odom
            distance = math.hypot(point[0]-ox, point[1]-oy)
            if distance <= max(self.executor_tolerance,
                               self.entry_arrival_tolerance):
                continue
            heading = math.atan2(point[1]-oy, point[0]-ox)
            if not self.send_goal("entry_aisle_waypoint_%02d" % index,
                                  point[0], point[1], heading):
                self.records.append({
                    "name": "entry_aisle_path_result", "success": False,
                    "failed_waypoint": index,
                    "duration_sec": time.monotonic()-started,
                    "odom": self.odom,
                })
                return False
        remaining = math.hypot(entry_ground[0]-self.odom[0],
                               entry_ground[1]-self.odom[1])
        success = remaining <= max(self.executor_tolerance,
                                   self.entry_arrival_tolerance)
        self.records.append({
            "name": "entry_aisle_path_result", "success": success,
            "duration_sec": time.monotonic()-started,
            "remaining_m": remaining, "odom": self.odom,
        })
        return success

    def save(self, ok, reason):
        os.makedirs(self.output, exist_ok=True)
        with open(os.path.join(self.output, "stair_return_turn_entry.json"), "w") as stream:
            json.dump({"success": ok, "reason": reason,
                       "records": self.records, "odom": self.odom,
                       "imu_yaw": self.imu,
                       "mission_elapsed_sec": (None if self.mission_started_monotonic is None
                                               else time.monotonic()-self.mission_started_monotonic)},
                      stream, indent=2)

    def run(self):
        if not self.ready():
            return self.save(False, "not_ready")
        self.mission_started_monotonic = time.monotonic()

        if self.resume_turn_reference_imu is None:
            # Phase 1: either follow the preserved return bearing or, when the
            # robot is spawned in the corridor, move along its startup heading.
            # Both modes stop from measured black-wall range and keep stair
            # detections hard-gated throughout this phase.
            corridor_ok = self.return_until_black_wall()
            self.corridor_side_detection_enabled = False
            if not corridor_ok:
                return self.save(False, "return_wall_threshold_not_reached")
        else:
            self.records.append({"name": "resume_partial_right_turn",
                                 "reference_imu_yaw": float(
                                     self.resume_turn_reference_imu),
                                 "current_imu_yaw": self.imu,
                                 "odom": self.odom})

        # Phase 2: one explicit relative right turn. No visual target is read.
        self.det = None
        if not self.turn_right_and_verify(self.resume_turn_reference_imu):
            return self.save(False, "right_turn_90_not_verified")

        # Phase 3: only now may a fresh stair observation enter the mission.
        self.det = None
        self.detector_enabled = True
        rospy.sleep(max(0.0, self.post_turn_settle))
        detection = self.detection()
        if detection is None:
            return self.save(False, "no_stair_after_verified_turn")
        first_step = self.stair_boundary_map(detection)
        entry_ground = self.entry_ground_map(detection)
        self.records.append({
            "name": "post_turn_stair_lock",
            "detection": detection,
            "first_step_landmark": first_step,
            "entry_ground_landmark": entry_ground,
            "odom": self.odom,
        })

        if self.entry_aisle_path_enabled:
            if self.execute_entry_aisle_path(detection, entry_ground,
                                             first_step):
                return self.save(True, "stair_entry_ground_reached")
            return self.save(False, "entry_aisle_path_failed")

        # Phase 4: retain the original bounded legs, targeting only ground.
        previous = None
        for index in range(self.max_legs):
            ox, oy, _oz, _yaw = self.odom
            remaining = math.hypot(entry_ground[0]-ox, entry_ground[1]-oy)
            self.records.append({"name": "entry_measurement_%02d" % index,
                                 "distance_m": remaining, "odom": self.odom})
            if remaining <= max(self.executor_tolerance,
                                self.entry_arrival_tolerance):
                # Stop at the detected free-floor entrance.  The previous
                # staging-and-return manoeuvre could reverse the robot into
                # the wall and did not reliably leave it facing the first
                # step, so reaching the entrance is now the terminal action.
                self.records.append({
                    "name": "entry_ground_reached_stop",
                    "entry_ground": entry_ground,
                    "first_step": first_step,
                    "remaining_m": remaining,
                    "odom": self.odom,
                })
                return self.save(True, "stair_entry_ground_reached")
            if previous is not None and remaining > previous-self.minimum_progress:
                return self.save(False, "entry_ground_distance_not_decreasing")
            previous = remaining
            # The existing goal executor already closes heading and position
            # together.  Do not call the local-rescan interface here: that
            # interface is a full observation scan in this project and can
            # turn 180 degrees even when this ground leg needs only a small
            # heading correction.
            leg = min(self.max_leg, max(self.min_leg,
                      remaining-self.executor_tolerance))
            ratio = leg / max(remaining, 1e-6)
            gx = ox + (entry_ground[0]-ox)*ratio
            gy = oy + (entry_ground[1]-oy)*ratio
            ga = math.atan2(gy-oy, gx-ox)
            if not self.send_goal("entry_ground_leg_%02d" % index, gx, gy, ga):
                return self.save(False, "entry_ground_leg_failed")
        return self.save(False, "entry_ground_leg_limit")


if __name__ == "__main__":
    rospy.init_node("simenv_stair_return_turn_entry")
    ReturnTurnEntry().run()
