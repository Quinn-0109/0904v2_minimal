#!/usr/bin/env python3
"""Detect red spheres, localize them, and write the competition result."""

import json
import math
import os
import sys
import tempfile
import time
import threading
from collections import deque

import cv2
import message_filters
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detection_tracking import (
    MultiViewTracker,
    PoseSample,
    interpolate_pose,
    yaw_from_tuple,
)
from first_floor_core import classify_sphere_or_plane


def rotate_by_quaternion_tuple(point, quaternion):
    vector = np.asarray(point, dtype=np.float64)
    q = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-9:
        return vector
    q /= norm
    xyz = q[:3]
    return (
        vector
        + 2.0 * q[3] * np.cross(xyz, vector)
        + 2.0 * np.cross(xyz, np.cross(xyz, vector))
    )


class DangerDetector:
    def __init__(self):
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._odom = None
        self._pose_history = deque(maxlen=400)
        self._localization_healthy = False
        self._start_position = None
        self._start_time = None
        self._max_departure = 0.0
        self._finish_seen = False
        self._written = False

        self._result_file = rospy.get_param(
            "~result_file", os.path.join(os.getcwd(), "results", "detected_danger.json")
        )
        self._relative_result_file = rospy.get_param(
            "~relative_result_file",
            os.path.join(
                os.path.dirname(os.path.abspath(self._result_file)),
                "detected_danger_relative.json",
            ),
        )
        self._timeout = float(rospy.get_param("~timeout_seconds", 150.0))
        self._room_only_gate = bool(rospy.get_param("~room_only_gate", False))
        self._require_localization_healthy = bool(rospy.get_param(
            "~require_localization_healthy", True))
        self._odom_topic = rospy.get_param("~odom_topic", "/state_estimation")
        self._rgb_topic = rospy.get_param(
            "~rgb_topic", "/real_sense/rgb/image_raw")
        self._depth_topic = rospy.get_param(
            "~depth_topic", "/real_sense/depth/image_raw")
        self._camera_fov_deg = float(rospy.get_param("~camera_fov_deg", 60.0))
        self._room_detection_enabled = not self._room_only_gate
        self._room_context = {"enabled": self._room_detection_enabled,
                              "room_id": None, "scheduler_state": "STARTUP"}
        self._diagnostic_file = rospy.get_param(
            "~diagnostic_file", os.path.join(
                os.path.dirname(os.path.abspath(self._result_file)),
                "danger_detection_diagnostics.json"))
        self._diagnostic_dir = rospy.get_param(
            "~diagnostic_dir", os.path.join(
                os.path.dirname(os.path.abspath(self._result_file)),
                "danger_detection_debug"))
        self._max_diagnostic_frames = int(rospy.get_param(
            "~max_diagnostic_frames", 16))
        # RGB-D arrives at about 8 Hz while the robot is moving.  Five
        # independent frames per second still gives several observations over
        # a normal room transit, but avoids competing with mapping/control on
        # every rendered frame.
        self._processing_rate_hz = float(rospy.get_param(
            "~processing_rate_hz", 5.0))
        self._last_processed_image_stamp = -math.inf
        self._diagnostic_frames_written = 0
        self._diagnostic = {
            "frames_outside_room_skipped": 0,
            "frames_in_room_processed": 0,
            "frames_with_red_pixels": 0,
            "red_contours_seen": 0,
            "candidate_events": [],
            "rooms": {},
        }
        self._timeout_fallback = float(
            rospy.get_param("~timeout_fallback_seconds", 5.0)
        )
        self._deadline_margin = float(rospy.get_param("~deadline_margin_s", 0.30))
        self._home_tolerance = float(rospy.get_param("~home_tolerance", 0.7))
        self._dedup_radius = float(rospy.get_param("~dedup_radius", 0.45))
        self._minimum_hits = int(rospy.get_param("~minimum_hits", 3))
        self._minimum_views = int(rospy.get_param("~minimum_views", 2))
        self._minimum_view_angle = math.radians(
            float(rospy.get_param("~minimum_view_angle_deg", 12.0))
        )
        self._minimum_viewpoint_baseline = float(
            rospy.get_param("~minimum_viewpoint_baseline_m", 0.4)
        )
        self._pose_history_seconds = float(
            rospy.get_param("~pose_history_seconds", 2.5)
        )
        self._max_pose_skew = float(rospy.get_param("~max_pose_skew_s", 0.12))
        self._tracker = MultiViewTracker(
            radius=self._dedup_radius, min_view_angle=self._minimum_view_angle
        )
        self._minimum_area = float(rospy.get_param("~minimum_area", 40.0))
        self._minimum_circularity = float(
            rospy.get_param("~minimum_circularity", 0.72)
        )
        self._minimum_fill = float(rospy.get_param("~minimum_fill_ratio", 0.68))
        self._minimum_extent = float(rospy.get_param("~minimum_extent", 0.48))
        self._min_depth_samples = int(rospy.get_param("~min_depth_samples", 8))
        self._max_depth = float(rospy.get_param("~max_depth", 6.5))
        self._aspect_min = float(rospy.get_param("~aspect_min", 0.72))
        self._aspect_max = float(rospy.get_param("~aspect_max", 1.35))
        self._hsv_s_min = int(rospy.get_param("~hsv_s_min", 90))
        self._hsv_v_min = int(rospy.get_param("~hsv_v_min", 55))
        self._diameter_min = float(rospy.get_param("~diameter_min", 0.10))
        self._diameter_max = float(rospy.get_param("~diameter_max", 0.42))
        self._world_z_min = float(rospy.get_param("~world_z_min", 0.04))
        self._world_z_max = float(rospy.get_param("~world_z_max", 0.55))
        self._max_detection_range = float(rospy.get_param("~max_detection_range", 4.5))
        # Do not encode the seed-77 building rectangle in the online detector.
        self._world_x_min = float(rospy.get_param("~world_x_min", float("-inf")))
        self._world_x_max = float(rospy.get_param("~world_x_max", float("inf")))
        self._world_y_min = float(rospy.get_param("~world_y_min", float("-inf")))
        self._world_y_max = float(rospy.get_param("~world_y_max", float("inf")))
        self._camera_x = float(rospy.get_param("~camera_x", 0.28))
        self._camera_y = float(rospy.get_param("~camera_y", 0.0))
        self._camera_z = float(rospy.get_param("~camera_z", 0.043))
        self._image_callbacks = 0
        self._reject_counts = {
            "area": 0,
            "shape": 0,
            "depth": 0,
            "size": 0,
            "height": 0,
            "range": 0,
            "bounds": 0,
            "accepted": 0,
            "dr_skip": 0,
            "pose_sync": 0,
            "localization": 0,
            "geometry": 0,
            "outside_room": 0,
        }
        self._dead_reckon = False
        self._skip_dead_reckon = bool(
            rospy.get_param("~skip_dead_reckon_detections", False)
        )
        self._last_status_log = rospy.Time(0)
        self._wall_start = time.time()

        self._complete_pub = rospy.Publisher(
            "/simenv/mission_complete", Bool, queue_size=1, latch=True
        )
        self._candidate_pub = rospy.Publisher(
            "/simenv/danger_candidate", PointStamped, queue_size=5
        )
        rospy.Subscriber(self._odom_topic, Odometry, self._on_odom, queue_size=10)
        rospy.Subscriber(
            "/simenv/dead_reckon_active", Bool, self._on_dr_flag, queue_size=1
        )
        rospy.Subscriber(
            "/simenv/localization_healthy", Bool, self._on_localization_health, queue_size=1
        )
        rospy.Subscriber(
            "/simenv/finalize_result", Bool, self._on_finalize, queue_size=2
        )
        rospy.Subscriber(
            "/simenv/room_detection_enabled", Bool, self._on_room_gate,
            queue_size=2
        )
        rospy.Subscriber(
            "/simenv/room_detection_context", String, self._on_room_context,
            queue_size=2
        )

        rgb = message_filters.Subscriber(self._rgb_topic, Image)
        depth = message_filters.Subscriber(self._depth_topic, Image)
        # Gazebo's depth plugin advertises camera_info but, in this scene,
        # never publishes it.  RGB and depth share the same optical frame and
        # calibrated 60-degree FOV, so synchronize the streams that actually
        # arrive and derive intrinsics from each image's dimensions.
        sync = message_filters.ApproximateTimeSynchronizer(
            [rgb, depth], queue_size=12, slop=0.12
        )
        sync.registerCallback(self._on_images)
        self._sync = sync
        rospy.Timer(rospy.Duration(0.10), self._on_timer)
        rospy.loginfo("Danger RGB-D detector: odom=%s rgb=%s depth=%s room_gate=%s health_gate=%s",
                      self._odom_topic, self._rgb_topic, self._depth_topic,
                      self._room_only_gate, self._require_localization_healthy)

    def _on_room_gate(self, message):
        with self._lock:
            self._room_detection_enabled = bool(message.data)
            self._room_context["enabled"] = bool(message.data)

    def _on_room_context(self, message):
        try:
            context = json.loads(message.data)
        except (TypeError, ValueError):
            return
        with self._lock:
            self._room_context = dict(context)
            self._room_detection_enabled = bool(context.get("enabled", False))

    def _room_bucket_locked(self):
        room_id = self._room_context.get("room_id") or "unassigned_room"
        bucket = self._diagnostic["rooms"].setdefault(str(room_id), {
            "processed_frames": 0, "frames_with_red_pixels": 0,
            "red_contours": 0, "accepted_candidates": 0,
            "confirmed_detections": 0, "rejections": {},
        })
        return bucket

    def _note_rejection(self, reason):
        with self._lock:
            self._reject_counts[reason] = self._reject_counts.get(reason, 0) + 1
            bucket = self._room_bucket_locked()
            bucket["rejections"][reason] = (
                int(bucket["rejections"].get(reason, 0)) + 1)

    def _write_diagnostics(self, final=False):
        with self._lock:
            payload = {
                "schema": "simenv_danger_detection_diagnostics_v1",
                "room_only_gate": self._room_only_gate,
                "current_room_context": dict(self._room_context),
                "final": bool(final),
                "image_callbacks": self._image_callbacks,
                "rejection_counts": dict(self._reject_counts),
                "diagnostics": json.loads(json.dumps(self._diagnostic)),
                "confirmed_clusters": len(self._tracker.confirmed(
                    self._minimum_hits, self._minimum_views,
                    self._minimum_viewpoint_baseline)),
            }
        self._atomic_write(self._diagnostic_file, payload)

    def _save_evidence_frame(self, image, mask, observations, stamp):
        if not observations:
            return
        with self._lock:
            if self._diagnostic_frames_written >= self._max_diagnostic_frames:
                return
            self._diagnostic_frames_written += 1
            frame_index = self._diagnostic_frames_written
            context = dict(self._room_context)
        try:
            os.makedirs(self._diagnostic_dir, exist_ok=True)
            canvas = image.copy()
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)[0]
            cv2.drawContours(canvas, contours, -1, (0, 255, 255), 2)
            labels = ", ".join(observations[:4])
            cv2.putText(canvas, "room={} {}".format(
                context.get("room_id"), labels[:110]), (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1,
                cv2.LINE_AA)
            filename = "frame_{:02d}_{:.3f}.png".format(frame_index, stamp)
            cv2.imwrite(os.path.join(self._diagnostic_dir, filename), canvas)
        except (OSError, cv2.error):
            pass

    def _on_dr_flag(self, message):
        with self._lock:
            self._dead_reckon = bool(message.data)

    def _on_localization_health(self, message):
        with self._lock:
            self._localization_healthy = bool(message.data)

    def _on_odom(self, message):
        with self._lock:
            self._odom = message
            position = message.pose.pose.position
            orientation = message.pose.pose.orientation
            stamp = message.header.stamp.to_sec()
            if stamp <= 0.0:
                stamp = rospy.Time.now().to_sec()
            self._pose_history.append(
                PoseSample(
                    stamp=stamp,
                    position=(position.x, position.y, position.z),
                    quaternion=(
                        orientation.x,
                        orientation.y,
                        orientation.z,
                        orientation.w,
                    ),
                )
            )
            cutoff = stamp - self._pose_history_seconds
            while self._pose_history and self._pose_history[0].stamp < cutoff:
                self._pose_history.popleft()
            current = np.array([position.x, position.y, position.z])
            if self._start_position is None:
                self._start_position = current
                self._start_time = rospy.Time.now()
            self._max_departure = max(
                self._max_departure,
                float(np.linalg.norm(current[:2] - self._start_position[:2])),
            )
            should_write = (
                self._finish_seen
                and self._max_departure >= 0.5
                and np.linalg.norm(current[:2] - self._start_position[:2])
                <= self._home_tolerance
            )
        if should_write:
            self._write_result("completed")

    def _on_finish(self, message):
        if message.data:
            with self._lock:
                self._finish_seen = True

    def _on_finalize(self, message):
        if not message.data:
            return
        with self._lock:
            self._finish_seen = True
        # The mission manager only finalizes after return-home or a hard
        # deadline.  Persist results before acknowledging mission_complete.
        self._write_result("finalized")

    def _on_timer(self, _event):
        with self._lock:
            if self._start_time is None or self._written:
                return
            elapsed = (rospy.Time.now() - self._start_time).to_sec()
            clusters = len(self._tracker.clusters)
            hits = sum(cluster.hits for cluster in self._tracker.clusters)
            departure = self._max_departure
            images = self._image_callbacks
        wall_elapsed = time.time() - self._wall_start
        if elapsed >= max(
            0.1, self._timeout + self._timeout_fallback - self._deadline_margin
        ) or wall_elapsed >= self._timeout + 30.0:
            rospy.logerr(
                "First-floor benchmark exceeded %.1f seconds (sim=%.1f wall=%.1f)",
                self._timeout,
                elapsed,
                wall_elapsed,
            )
            self._write_result("timeout")
            return
        now = rospy.Time.now()
        if (now - self._last_status_log).to_sec() >= 5.0:
            self._last_status_log = now
            rejects = dict(self._reject_counts)
            rospy.loginfo(
                "status t=%.1fs departure=%.2fm images=%d clusters=%d hits=%d "
                "rej(area=%d shape=%d depth=%d size=%d z=%d dr=%d ok=%d)",
                elapsed,
                departure,
                images,
                clusters,
                hits,
                rejects["area"],
                rejects["shape"],
                rejects["depth"],
                rejects.get("size", 0),
                rejects.get("height", 0),
                rejects.get("dr_skip", 0),
                rejects["accepted"],
            )
            self._write_diagnostics(final=False)

    @staticmethod
    def _depth_meters(depth_image):
        depth = np.asarray(depth_image)
        if depth.dtype == np.uint16:
            return depth.astype(np.float32) * 0.001
        return depth.astype(np.float32)

    def _on_images(self, rgb_message, depth_message):
        with self._lock:
            pose_history = list(self._pose_history)
            dead_reckon = self._dead_reckon
            localization_healthy = self._localization_healthy
            room_enabled = self._room_detection_enabled
            self._image_callbacks += 1
            if self._room_only_gate and not room_enabled:
                self._reject_counts["outside_room"] += 1
                self._diagnostic["frames_outside_room_skipped"] += 1
                return
            image_stamp = rgb_message.header.stamp.to_sec()
            if image_stamp <= 0.0:
                image_stamp = depth_message.header.stamp.to_sec()
            minimum_period = 1.0 / max(self._processing_rate_hz, 0.1)
            if image_stamp - self._last_processed_image_stamp < minimum_period:
                return
            self._last_processed_image_stamp = image_stamp
            self._diagnostic["frames_in_room_processed"] += 1
            bucket = self._room_bucket_locked()
            bucket["processed_frames"] += 1
        if not pose_history:
            return
        # Skip world projection while /state_estimation is coasting on cmd_vel.
        if dead_reckon and self._skip_dead_reckon:
            self._note_rejection("dr_skip")
            return
        if self._require_localization_healthy and not localization_healthy:
            self._note_rejection("localization")
            return
        pose_sample = interpolate_pose(
            pose_history, image_stamp, max_skew=self._max_pose_skew
        )
        if pose_sample is None:
            self._note_rejection("pose_sync")
            return
        try:
            image = self._bridge.imgmsg_to_cv2(rgb_message, "bgr8")
            depth = self._depth_meters(
                self._bridge.imgmsg_to_cv2(depth_message, "passthrough")
            )
        except CvBridgeError as error:
            rospy.logwarn_throttle(5.0, "Image conversion failed: %s", error)
            return

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        s_min, v_min = self._hsv_s_min, self._hsv_v_min
        # Narrow red band: reject orange-ish distractors / warm walls.
        red_low = cv2.inRange(hsv, (0, s_min, v_min), (8, 255, 255))
        red_high = cv2.inRange(hsv, (170, s_min, v_min), (179, 255, 255))
        mask = cv2.morphologyEx(
            red_low | red_high,
            cv2.MORPH_OPEN,
            np.ones((5, 5), dtype=np.uint8),
        )
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        observations = []
        with self._lock:
            if contours:
                self._diagnostic["frames_with_red_pixels"] += 1
                self._diagnostic["red_contours_seen"] += len(contours)
                bucket = self._room_bucket_locked()
                bucket["frames_with_red_pixels"] += 1
                bucket["red_contours"] += len(contours)
        image_height, image_width = image.shape[:2]
        fov_rad = math.radians(max(1.0, min(self._camera_fov_deg, 179.0)))
        fx = 0.5 * float(image_width) / math.tan(0.5 * fov_rad)
        fy = fx
        cx = 0.5 * float(image_width - 1)
        cy = 0.5 * float(image_height - 1)
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if area < self._minimum_area or perimeter <= 0.0:
                observations.append("reject:area")
                self._note_rejection("area")
                continue
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            x, y, width, height = cv2.boundingRect(contour)
            aspect = float(width) / max(float(height), 1.0)
            (_, _), radius = cv2.minEnclosingCircle(contour)
            circle_area = math.pi * max(radius, 1.0) ** 2
            fill_ratio = area / circle_area
            extent = area / max(float(width * height), 1.0)
            # Red boxes (distractors) fail circularity/fill; spheres pass.
            if (
                circularity < self._minimum_circularity
                or fill_ratio < self._minimum_fill
                or extent < self._minimum_extent
                or not self._aspect_min <= aspect <= self._aspect_max
            ):
                observations.append("reject:shape")
                self._note_rejection("shape")
                continue

            contour_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            contour_mask = cv2.erode(contour_mask, np.ones((3, 3), np.uint8))
            samples = depth[
                (contour_mask > 0)
                & np.isfinite(depth)
                & (depth > 0.25)
                & (depth < self._max_depth)
            ]
            if samples.size < self._min_depth_samples:
                observations.append("reject:depth_samples")
                self._note_rejection("depth")
                continue
            valid_pixels = (
                (contour_mask > 0)
                & np.isfinite(depth)
                & (depth > 0.25)
                & (depth < self._max_depth)
            )
            pixel_y, pixel_x = np.nonzero(valid_pixels)
            if len(pixel_x) > 700:
                indices = np.linspace(0, len(pixel_x) - 1, 700).astype(np.int32)
                pixel_x, pixel_y = pixel_x[indices], pixel_y[indices]
            pixel_depth = depth[pixel_y, pixel_x]
            geometry_points = np.column_stack(
                (
                    (pixel_x - cx) * pixel_depth / fx,
                    (pixel_y - cy) * pixel_depth / fy,
                    pixel_depth,
                )
            )
            geometry = classify_sphere_or_plane(geometry_points)
            if geometry.label != "sphere":
                observations.append("reject:geometry")
                self._note_rejection("geometry")
                continue
            # Depth must be compact (sphere surface), not a red wall patch.
            depth_span = float(np.percentile(samples, 80) - np.percentile(samples, 20))
            # Spheres present a tight depth shell; boxes/walls span deeper.
            if depth_span > 0.22:
                observations.append("reject:depth_span")
                self._note_rejection("depth")
                continue
            distance = float(np.percentile(samples, 40))
            # Metric diameter from image radius + depth (~0.15m danger spheres).
            diameter = 2.0 * radius * distance / fx
            if not (self._diameter_min <= diameter <= self._diameter_max):
                observations.append("reject:size")
                self._note_rejection("size")
                continue
            # Reject elongated box faces that sneak past circularity.
            if radius > 0.0 and (max(width, height) / (2.0 * radius)) > 1.15:
                observations.append("reject:shape")
                self._note_rejection("shape")
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            u = moments["m10"] / moments["m00"]
            v = moments["m01"] / moments["m00"]
            optical = np.array(
                [
                    (u - cx) * distance / fx,
                    (v - cy) * distance / fy,
                    distance,
                ]
            )
            # ROS optical (right, down, forward) -> A1 base (forward, left, up).
            base = np.array(
                [
                    optical[2] + self._camera_x,
                    -optical[0] + self._camera_y,
                    -optical[1] + self._camera_z,
                ]
            )
            world = rotate_by_quaternion_tuple(base, pose_sample.quaternion) + np.array(
                pose_sample.position
            )
            if not (self._world_z_min <= float(world[2]) <= self._world_z_max):
                observations.append("reject:height")
                self._note_rejection("height")
                continue
            robot_xy = np.array(pose_sample.position[:2], dtype=np.float64)
            if np.linalg.norm(world[:2] - robot_xy) > self._max_detection_range:
                observations.append("reject:range")
                self._note_rejection("range")
                continue
            if not (
                self._world_x_min <= float(world[0]) <= self._world_x_max
                and self._world_y_min <= float(world[1]) <= self._world_y_max
            ):
                observations.append("reject:bounds")
                self._note_rejection("bounds")
                continue
            with self._lock:
                self._reject_counts["accepted"] += 1
                self._room_bucket_locked()["accepted_candidates"] += 1
            confirmed = self._add_detection(
                world,
                yaw_from_tuple(pose_sample.quaternion),
                pose_sample.stamp,
                pose_sample.position,
            )
            observations.append("confirmed" if confirmed else "candidate")
            if confirmed:
                with self._lock:
                    self._room_bucket_locked()["confirmed_detections"] += 1
        self._save_evidence_frame(image, mask, observations, image_stamp)

    def _add_detection(self, position, view_yaw, stamp, viewpoint):
        with self._lock:
            cluster = self._tracker.add(
                position, view_yaw, stamp, viewpoint=viewpoint
            )
            confirmed = (
                cluster.hits >= self._minimum_hits
                and cluster.view_count >= self._minimum_views
                and cluster.viewpoint_baseline >= self._minimum_viewpoint_baseline
            )
            events = self._diagnostic["candidate_events"]
            if len(events) < 80:
                events.append({
                    "stamp": round(float(stamp), 3),
                    "room_id": self._room_context.get("room_id"),
                    "position": [round(float(value), 3) for value in cluster.position],
                    "hits": int(cluster.hits), "views": int(cluster.view_count),
                    "viewpoint_baseline_m": round(float(cluster.viewpoint_baseline), 3),
                    "status": "confirmed" if confirmed else "candidate",
                })
        if not confirmed:
            candidate = PointStamped()
            candidate.header.stamp = rospy.Time.from_sec(stamp)
            candidate.header.frame_id = "map"
            candidate.point.x = float(cluster.position[0])
            candidate.point.y = float(cluster.position[1])
            candidate.point.z = float(cluster.position[2])
            self._candidate_pub.publish(candidate)
        return confirmed
        return confirmed

    def _write_result(self, reason):
        with self._lock:
            if self._written:
                return
            self._written = True
            elapsed = (
                0.0
                if self._start_time is None
                else (rospy.Time.now() - self._start_time).to_sec()
            )
            detections = [
                {"position": [round(float(value), 4) for value in cluster.position]}
                for cluster in self._tracker.confirmed(
                    self._minimum_hits,
                    self._minimum_views,
                    self._minimum_viewpoint_baseline,
                )
            ]
            start_position = (
                None if self._start_position is None else self._start_position.copy()
            )
        result = {
            "exploration_time": round(elapsed, 3),
            "detected_danger_sources": detections,
            "benchmark_status": reason,
        }
        self._atomic_write(self._result_file, result)
        relative_detections = []
        if start_position is not None:
            for detection in detections:
                position = detection["position"]
                relative_detections.append(
                    {
                        "position": [
                            round(float(position[index]) - float(start_position[index]), 4)
                            for index in range(3)
                        ]
                    }
                )
        self._atomic_write(
            self._relative_result_file,
            {
                "schema": "simenv_danger_relative_v1",
                "frame_id": "start",
                "exploration_time": round(elapsed, 3),
                "detected_danger_sources": relative_detections,
                "benchmark_status": reason,
            },
        )
        self._write_diagnostics(final=True)
        rospy.loginfo(
            "Wrote %d danger detections to %s (%s, %.1fs)",
            len(detections),
            self._result_file,
            reason,
            elapsed,
        )
        self._complete_pub.publish(Bool(data=True))

    @staticmethod
    def _atomic_write(path, payload):
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(prefix=".danger-", dir=directory)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)


if __name__ == "__main__":
    rospy.init_node("danger_detector")
    DangerDetector()
    rospy.spin()
