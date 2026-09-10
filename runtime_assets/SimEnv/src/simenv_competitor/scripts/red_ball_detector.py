#!/usr/bin/env python3
"""Online RGB-D red-ball observations.

This node intentionally has no knowledge of Gazebo entities or room layout.
It converts compact red image regions into map-frame observations and leaves
multi-frame confirmation to ``hazard_candidate_tracker.py``.
"""
import json
import math
import os
import sys
import threading
import time
from collections import deque

import cv2
import message_filters
import numpy as np
# catkin's devel-space relay executes this source while leaving sys.path at
# devel/lib/<package>. Re-add the real script directory for pure helpers.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from red_ball_detector_core import (
    edge_clipped_partial_sphere_hint, oblique_sphere_metric_gate,
    oblique_sphere_shape_gate,
)
import rospy
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class RedBallDetector:
    def __init__(self):
        # OpenCV inherits the host CPU count and may create a very large
        # worker pool for each small camera operation. On the competition
        # host this was 128 workers (143 process threads total); during run57
        # the detector alone consumed more than one CPU core and contended
        # with Gazebo/FAST-LIO. These 640x480 operations are too small to
        # amortise that fan-out. A single OpenCV worker preserves the input,
        # detector rate, thresholds and pixel results while avoiding global
        # oversubscription.
        requested_opencv_threads = max(
            1, int(rospy.get_param("~opencv_threads", 1)))
        cv2.setNumThreads(requested_opencv_threads)
        self.opencv_threads = int(cv2.getNumThreads())
        self.bridge, self.lock = CvBridge(), threading.Lock()
        self.pose = None
        # RGB and Odometry arrive on independent ROS queues.  Retaining a
        # short history lets each RGB-D ray use the pose at the image stamp,
        # rather than whichever odometry message happened to arrive last.
        self.pose_history = deque(maxlen=300)
        self.maximum_pose_sync_error = float(rospy.get_param(
            "~maximum_pose_sync_error_sec", .20))
        # A finite but discontinuous FAST-LIO pose is not a usable map pose:
        # it would project an ordinary RGB contour many metres from reality.
        # The motion executor independently stops on this same condition.
        self.maximum_pose_jump = float(rospy.get_param(
            "~maximum_pose_jump_m", .75))
        self.pose_jump_hold_seconds = float(rospy.get_param(
            "~pose_jump_hold_seconds", 1.0))
        self.pose_hold_until = -math.inf
        self.depth = None
        self.depth_stamp = None
        self.camera = None
        self.context = {"enabled": False, "room_id": None}
        self.room_search_stats = {}
        self.last_stamp = -math.inf
        self.rate = float(rospy.get_param("~processing_rate_hz", 8.0))
        # The Gazebo RGB and depth images are emitted by the same sensor, but
        # independent rospy callbacks do not preserve their arrival order.
        # At 2 rad/s, pairing an RGB frame with the previous 10 Hz depth frame
        # mixes views about 11 degrees apart.  run31 consequently measured a
        # real 5.34 m ball at 1.96 m in the preceding frame.  Synchronise on
        # sensor stamps before any geometry or metric-diameter test.
        self.maximum_depth_sync_error = max(0.0, float(rospy.get_param(
            "~maximum_depth_sync_error_sec", .035)))
        self.room_only = bool(rospy.get_param("~room_only_gate", True))
        self.min_area = float(rospy.get_param("~minimum_area", 35.0))
        self.min_circularity = float(rospy.get_param("~minimum_circularity", .80))
        self.min_fill = float(rospy.get_param("~minimum_fill_ratio", .58))
        self.max_fill = float(rospy.get_param("~maximum_fill_ratio", .83))
        # A real ball viewed close to the edge of the RGB image can have a
        # clipped/aliased outline.  Keep this fallback deliberately narrower
        # than a general shape relaxation: it is only for an almost solid,
        # still-round red blob.  Red cuboid faces seen in prior runs had
        # fill ratios around .86-.87 and therefore do not enter this path.
        self.oblique_min_aspect = float(rospy.get_param(
            "~oblique_ball_min_aspect", .65))
        self.oblique_max_aspect = float(rospy.get_param(
            "~oblique_ball_max_aspect", 1.55))
        self.oblique_min_circularity = float(rospy.get_param(
            "~oblique_ball_min_circularity", .70))
        self.oblique_min_fill = float(rospy.get_param(
            "~oblique_ball_min_fill_ratio", .88))
        # Low-poly simulator spheres can render as an almost solid square at
        # room scale (fix15 D6: fill .855--.934, circularity .761--.807).
        # Admit that silhouette to the RGB-D surface-convexity gate below;
        # high-fill red box faces remain planar and are rejected there.
        self.oblique_max_fill = float(rospy.get_param(
            "~oblique_ball_max_fill_ratio", .96))
        # A red cuboid viewed obliquely can have the same high image fill as
        # a clipped sphere.  Its depth is nevertheless planar, whereas a
        # sphere has a nearer centre and a farther contour boundary.  Apply
        # this check only to the permissive oblique branch: the stricter
        # round/medium-fill branch keeps its current recall for small or
        # partly occluded balls.
        self.oblique_minimum_depth_convexity = float(rospy.get_param(
            "~oblique_minimum_depth_convexity_m", .018))
        self.min_diameter = float(rospy.get_param("~minimum_ball_diameter_m", .18))
        # Competition spheres have 0.30 m diameter.  Keep a generous metric
        # tolerance for depth error, but do not admit the much larger apparent
        # red cuboid faces that passed the oblique image-shape branch.
        self.max_diameter = float(rospy.get_param("~maximum_ball_diameter_m", .40))
        # Only the deliberately permissive oblique branch needs a tighter size
        # envelope.  Historical true spheres measured .265--.306 m, whereas
        # run96's red-box face repeatedly measured .365--.378 m.  The strict
        # sphere branch retains the wider global tolerance above.
        self.oblique_max_diameter = float(rospy.get_param(
            "~oblique_maximum_ball_diameter_m", .32))
        self.edge_hint_margin_px = int(rospy.get_param(
            "~edge_hint_margin_px", 20))
        self.max_depth_mad = float(rospy.get_param("~maximum_depth_mad_m", .075))
        # D6 was visibly circular in the diagnostic run but lay just beyond
        # the former 6.5 m RGB-D gate.  The sphere/box checks below and the
        # multi-frame tracker remain mandatory, so retain useful room-scale
        # observations out to 8 m instead of discarding them before 3-D
        # validation.
        self.max_depth = float(rospy.get_param("~max_depth", 8.0))
        # Gazebo clips missing rays at the configured far plane (8.0 m), not
        # at 95% of it.  The old 7.60 m cutoff rejected two stable, correctly
        # sized strict-sphere frames in run31.  Retain a small far-plane guard
        # while allowing real room-scale observations up to 7.92 m.
        self.maximum_valid_depth_fraction = min(.999, max(.95, float(
            rospy.get_param("~maximum_valid_depth_fraction", .99))))
        self.fallback_depth_mad = float(rospy.get_param(
            "~fallback_maximum_depth_mad_m", .18))
        self.camera_yaw = math.radians(float(rospy.get_param("~camera_yaw_offset_deg", 0.0)))
        self.camera_z = float(rospy.get_param("~camera_height_m", 0.32))
        self.output_dir = rospy.get_param("~output_dir", os.getcwd())
        self.log_path = os.path.join(self.output_dir, "logs", "red_ball_observations.jsonl")
        # A successful observation log alone cannot diagnose a miss: it loses
        # frames in which the ball was outside the camera view *and* frames in
        # which a red region was rejected by one of the geometric checks.
        # Keep a compact per-frame funnel for post-run analysis.  It contains
        # no simulator truth or layout information.
        self.funnel_log_path = os.path.join(self.output_dir, "logs", "red_ball_detection_funnel.jsonl")
        self.debug_dir = os.path.join(self.output_dir, "logs", "red_ball_debug_frames")
        self.debug_frame_limit = int(rospy.get_param("~debug_frame_limit", 24))
        self.debug_frame_interval = float(rospy.get_param("~debug_frame_interval_s", 2.0))
        self.debug_frames_saved = 0
        self.last_debug_stamp = -math.inf
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.pub = rospy.Publisher("/simenv/red_ball_observations", String, queue_size=30)
        self.search_status_pub = rospy.Publisher(
            "/simenv/red_ball_search_status", String, queue_size=10)
        rospy.Subscriber(rospy.get_param("~odom_topic", "/Odometry"), Odometry, self.on_odom, queue_size=30)
        rgb_topic = rospy.get_param("~rgb_topic", "/real_sense/rgb/image_raw")
        depth_topic = rospy.get_param(
            "~depth_topic", "/real_sense/depth/image_raw")
        rospy.Subscriber(rospy.get_param("~camera_info_topic", "/real_sense/rgb/camera_info"), CameraInfo, self.on_info, queue_size=2)
        rospy.Subscriber("/simenv/room_detection_context", String, self.on_context, queue_size=5)
        self.rgb_subscriber = message_filters.Subscriber(
            rgb_topic, Image, queue_size=8)
        self.depth_subscriber = message_filters.Subscriber(
            depth_topic, Image, queue_size=8)
        self.rgbd_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_subscriber, self.depth_subscriber], queue_size=12,
            slop=self.maximum_depth_sync_error, allow_headerless=False)
        self.rgbd_synchronizer.registerCallback(self.on_rgbd)
        rospy.loginfo(
            "red_ball_detector: RGB-D map observations enabled "
            "(OpenCV threads=%d)", self.opencv_threads)

    def on_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        pose = (p.x, p.y, p.z, yaw_from_quaternion(q), msg.header.stamp.to_sec())
        with self.lock:
            previous = self.pose_history[-1] if self.pose_history else None
            if (not all(math.isfinite(float(value)) for value in pose[:4]) or
                    (previous is not None and
                     math.hypot(float(pose[0]) - float(previous[0]),
                                float(pose[1]) - float(previous[1])) >
                     self.maximum_pose_jump)):
                self.pose_hold_until = max(
                    self.pose_hold_until,
                    time.monotonic() + self.pose_jump_hold_seconds)
                # Never interpolate a camera image across incompatible map
                # frames after FAST-LIO re-registration.
                self.pose_history.clear()
            self.pose = pose
            self.pose_history.append(pose)

    @staticmethod
    def _interpolate_pose(first, second, stamp):
        """Interpolate xy/z and shortest-path yaw at an RGB frame stamp."""
        span = float(second[4]) - float(first[4])
        if span <= 1e-6:
            return first
        ratio = min(1.0, max(0.0, (float(stamp) - float(first[4])) / span))
        yaw_delta = (float(second[3]) - float(first[3]) + math.pi) % (2.0 * math.pi) - math.pi
        return tuple(float(first[index]) + ratio *
                     (float(second[index]) - float(first[index]))
                     for index in range(3)) + \
               (float(first[3]) + ratio * yaw_delta, float(stamp))

    def _pose_at_stamp(self, stamp):
        """Return a time-synchronised map pose and its residual in seconds."""
        history = list(self.pose_history)
        if not history:
            return self.pose, None
        before = [item for item in history if item[4] <= stamp]
        after = [item for item in history if item[4] >= stamp]
        if before and after:
            left, right = before[-1], after[0]
            residual = max(abs(float(stamp) - left[4]), abs(right[4] - float(stamp)))
            if residual <= self.maximum_pose_sync_error:
                return self._interpolate_pose(left, right, stamp), residual
        nearest = min(history, key=lambda item: abs(float(item[4]) - float(stamp)))
        residual = abs(float(nearest[4]) - float(stamp))
        if residual <= self.maximum_pose_sync_error:
            return nearest, residual
        return None, residual

    def on_depth(self, msg):
        """Decode one depth frame (kept public for focused ROS tests)."""
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if image.dtype == np.uint16:
                image = image.astype(np.float32) * .001
            else:
                image = image.astype(np.float32)
            # Some depth frames contain signalling NaNs. Numpy evaluates both
            # operands of a boolean mask, so an earlier isfinite() test alone
            # cannot suppress invalid-comparison warnings. Zero is outside all
            # valid-depth gates below and preserves the rejection behaviour.
            image[~np.isfinite(image)] = 0.0
            with self.lock:
                self.depth = image
                self.depth_stamp = float(msg.header.stamp.to_sec())
            return image
        except CvBridgeError:
            return None

    def on_rgbd(self, rgb_msg, depth_msg):
        """Process a stamp-paired RGB/depth observation."""
        # Gate before CvBridge depth conversion. Previously every synchronized
        # camera frame was decoded even outside rooms and above the configured
        # detector rate, although on_image then discarded it. This preserves
        # exactly the in-room frame/rate policy while removing rejected-frame
        # allocation and uint16-to-float conversion.
        rgb_stamp = float(rgb_msg.header.stamp.to_sec())
        with self.lock:
            context_enabled = bool(
                self.context.get("enabled") and
                self.context.get("entry_confirmed", True))
            last_stamp = float(self.last_stamp)
        if self.room_only and not context_enabled:
            return
        if rgb_stamp - last_stamp < 1.0 / max(.5, self.rate):
            return
        depth = self.on_depth(depth_msg)
        if depth is None:
            return
        rgb_stamp = float(rgb_msg.header.stamp.to_sec())
        depth_stamp = float(depth_msg.header.stamp.to_sec())
        sync_error = abs(rgb_stamp - depth_stamp)
        if sync_error > self.maximum_depth_sync_error + 1e-9:
            return
        self.on_image(rgb_msg, depth_override=depth,
                      depth_sync_error=sync_error)

    def on_info(self, msg):
        with self.lock:
            self.camera = (float(msg.K[0]), float(msg.K[4]), float(msg.K[2]), float(msg.K[5]))

    def on_context(self, msg):
        try:
            value = json.loads(msg.data)
            with self.lock:
                self.context = value if isinstance(value, dict) else self.context
        except (ValueError, TypeError):
            pass

    @staticmethod
    def median_depth(depth, u, v):
        h, w = depth.shape[:2]
        patch = depth[max(0, v - 3):min(h, v + 4), max(0, u - 3):min(w, u + 4)]
        values = patch[np.isfinite(patch) & (patch > .08) & (patch < 12.0)]
        return float(np.median(values)) if values.size >= 4 else None

    def write(self, value):
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(value, sort_keys=True) + "\n")
        except OSError as exc:
            rospy.logwarn_throttle(10.0, "red ball log unavailable: %s", exc)

    def write_funnel(self, value):
        """Append one bounded detector-funnel record for a room-camera frame."""
        try:
            with open(self.funnel_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(value, sort_keys=True) + "\n")
        except OSError as exc:
            rospy.logwarn_throttle(10.0, "red ball funnel log unavailable: %s", exc)

    def save_debug_frame(self, rgb, mask, contours, details, stamp, room_id):
        """Save a few failure overlays, never an unbounded camera recording."""
        if self.debug_frames_saved >= self.debug_frame_limit:
            return
        if stamp - self.last_debug_stamp < self.debug_frame_interval:
            return
        rejected = [item for item in details if item.get("rejection")]
        if not rejected:
            return
        try:
            os.makedirs(self.debug_dir, exist_ok=True)
            overlay = rgb.copy()
            for contour, item in zip(contours, details):
                color = (0, 0, 255) if item.get("rejection") else (0, 200, 0)
                cv2.drawContours(overlay, [contour], -1, color, 2)
                x, y, _, _ = cv2.boundingRect(contour)
                label = item.get("rejection", "accepted")
                cv2.putText(overlay, label, (x, max(14, y - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1, cv2.LINE_AA)
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            view = np.hstack((overlay, mask_bgr))
            safe_room = str(room_id or "unknown").replace("/", "_")
            name = "{0:04d}_{1}_{2:.3f}.png".format(self.debug_frames_saved, safe_room, stamp)
            if cv2.imwrite(os.path.join(self.debug_dir, name), view):
                self.debug_frames_saved += 1
                self.last_debug_stamp = stamp
        except (OSError, cv2.error) as exc:
            rospy.logwarn_throttle(10.0, "red ball debug frame unavailable: %s", exc)

    def on_image(self, msg, depth_override=None, depth_sync_error=None):
        stamp = msg.header.stamp.to_sec() or time.time()
        if stamp - self.last_stamp < 1.0 / max(.5, self.rate):
            return
        self.last_stamp = stamp
        with self.lock:
            pose, pose_sync_error = self._pose_at_stamp(stamp)
            depth = self.depth if depth_override is None else depth_override
            intrinsics, context = self.camera, dict(self.context)
            pose_held = time.monotonic() < self.pose_hold_until
        # Do not produce outside-room records by default; nevertheless record
        # why an enabled room detector had no usable RGB-D input.
        if (self.room_only and not (context.get("enabled") and
                                   context.get("entry_confirmed", True))):
            return
        room_id = context.get("room_id")
        if pose_held:
            self.write_funnel({"timestamp": round(stamp, 6), "room_id": room_id,
                               "status": "pose_jump_hold"})
            return
        if pose is None or depth is None:
            self.write_funnel({"timestamp": round(stamp, 6), "room_id": room_id,
                               "status": "missing_input", "has_pose": pose is not None,
                               "has_depth": depth is not None,
                               "pose_sync_error_sec": pose_sync_error})
            return
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError:
            self.write_funnel({"timestamp": round(stamp, 6), "room_id": room_id,
                               "status": "rgb_decode_failed"})
            return
        # CameraInfo is preferred; the simulator occasionally publishes the
        # first RGB frame before it.  Use the configured horizontal FOV only
        # as a short startup fallback, never Gazebo truth metadata.
        if intrinsics is None:
            height, width = rgb.shape[:2]
            fx = width / (2.0 * math.tan(math.radians(60.0) / 2.0))
            intrinsics = (fx, fx, width / 2.0, height / 2.0)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        # Red wraps around hue zero; both bands are mandatory.
        mask = cv2.inRange(hsv, (0, 85, 65), (10, 255, 255)) | cv2.inRange(hsv, (170, 85, 65), (179, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fx, fy, cx, cy = intrinsics
        rejection_counts = {"area": 0, "shape": 0, "edge_clipped": 0,
                            "depth_missing": 0,
                            "depth_out_of_range": 0, "depth_mad": 0,
                            "metric_diameter": 0, "depth_profile": 0}
        details, accepted = [], 0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            detail = {"area": round(area, 2)}
            if area < self.min_area:
                rejection_counts["area"] += 1
                detail["rejection"] = "area"
                details.append(detail)
                continue
            x, y, w, h = cv2.boundingRect(contour)
            aspect = float(w) / max(1.0, float(h))
            perimeter = float(cv2.arcLength(contour, True))
            circularity = 4.0 * math.pi * area / max(1e-6, perimeter * perimeter)
            fill = area / max(1.0, float(w * h))
            detail.update({"bbox": [int(x), int(y), int(w), int(h)], "aspect": round(aspect, 3),
                           "circularity": round(circularity, 3), "fill_ratio": round(fill, 3)})
            edge_clipped = (
                x <= self.edge_hint_margin_px or
                x + w >= rgb.shape[1] - self.edge_hint_margin_px)
            # The only red distractors are cuboids.  Their visible faces can
            # be red and compact, but retain a squarer fill and lower contour
            # roundness than a sphere.  These limits keep the real-sphere
            # observations from the last run while rejecting the confirmed
            # red-box false positive (roundness .74-.76, fill .86-.87).
            strict_shape = (.80 <= aspect <= 1.25 and
                            circularity >= self.min_circularity and
                            self.min_fill <= fill <= self.max_fill)
            # This is not a second generic red-object detector.  It accepts
            # only the high-fill oblique sphere case, then still requires the
            # same metric-depth checks and three-view tracker confirmation.
            oblique_ball_shape = oblique_sphere_shape_gate(
                aspect, circularity, fill, self.oblique_min_aspect,
                self.oblique_max_aspect, self.oblique_min_circularity,
                self.oblique_min_fill, self.oblique_max_fill)
            # A distant sphere partly hidden by furniture becomes a narrow
            # moving slice. Keep this branch below red-box area/fill and send
            # it through metric depth plus the tracker s five-view gate.
            partial_ball_shape = (
                120.0 <= area <= 1000.0 and
                .45 <= aspect <= .62 and
                .42 <= circularity <= .70 and
                .55 <= fill <= .83)
            if not (strict_shape or oblique_ball_shape or partial_ball_shape):
                rejection_counts["shape"] += 1
                detail["rejection"] = (
                    "edge_clipped_partial_sphere"
                    if edge_clipped_partial_sphere_hint(
                        area, aspect, circularity, fill, x, w,
                        rgb.shape[1], self.edge_hint_margin_px)
                    else "shape")
                details.append(detail)
                continue
            detail["shape_gate"] = (
                "strict" if strict_shape else
                ("oblique_sphere" if oblique_ball_shape else
                 "partial_sphere"))
            # A high-fill oblique contour clipped against the image boundary
            # has an unreliable outline and can make a red cuboid look like
            # a sphere.  Do not confirm it from one fan.  Its bounding box
            # is still included in the search-status edge hint, which causes
            # the manager to turn the existing camera fan toward the object
            # and obtain an unclipped, ordinary observation instead.
            if detail["shape_gate"] == "oblique_sphere" and edge_clipped:
                rejection_counts["edge_clipped"] += 1
                detail["rejection"] = "edge_clipped_oblique"
                details.append(detail)
                continue
            u, v = int(x + w / 2), int(y + h / 2)
            # A red box can be compact and red but has near-unity fill; a
            # partially visible box can still look round at its contour edge.
            # Metric diameter and compact in-contour depth recover the 3-D
            # sphere check without another RGB-D detector node.
            contour_mask = np.zeros(depth.shape[:2], dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
            depth_values = depth[(contour_mask > 0) & np.isfinite(depth) &
                                 (depth > .08) & (depth < self.max_depth)]
            center_depth = self.median_depth(depth, u, v)
            depth_source = "center"
            # A floor-level red sphere can have a small depth-image hole at
            # its centre while the immediately surrounding pixels are valid.
            # Use only a narrow exterior ring as a fallback, never a global
            # scene depth.  Its looser MAD gate is explicitly recorded and
            # remains protected by RGB shape + metric size + multi-view gates.
            if center_depth is None or center_depth > self.max_depth:
                ring_kernel = np.ones((17, 17), np.uint8)
                outer = cv2.dilate(contour_mask, ring_kernel, iterations=1)
                ring = (outer > 0) & (contour_mask == 0)
                ring_values = depth[ring & np.isfinite(depth) & (depth > .08) &
                                    (depth < self.max_depth)]
                if ring_values.size >= 8:
                    center_depth = float(np.median(ring_values))
                    depth_values = ring_values
                    depth_source = "contour_ring"
                elif center_depth is not None and center_depth > self.max_depth:
                    rejection_counts["depth_out_of_range"] += 1
                    detail.update({"depth_m": round(center_depth, 4),
                                   "rejection": "depth_out_of_range"})
                    details.append(detail)
                    continue
            z = center_depth
            if z is None:
                rejection_counts["depth_missing"] += 1
                detail["rejection"] = "depth_missing"
                details.append(detail)
                continue
            if z >= self.max_depth * self.maximum_valid_depth_fraction:
                rejection_counts["depth_out_of_range"] += 1
                detail.update({"depth_m": round(z, 4),
                               "rejection": "depth_saturated"})
                details.append(detail)
                continue
            if depth_values.size < 8:
                # A valid centre is enough to position a compact sphere; use
                # its local patch for the compactness check when possible.
                h_img, w_img = depth.shape[:2]
                patch = depth[max(0, v - 5):min(h_img, v + 6),
                              max(0, u - 5):min(w_img, u + 6)]
                depth_values = patch[np.isfinite(patch) & (patch > .08) &
                                     (patch < self.max_depth)]
            if depth_values.size < 8:
                rejection_counts["depth_missing"] += 1
                detail["rejection"] = "depth_missing"
                details.append(detail)
                continue
            depth_mad = float(np.median(np.abs(depth_values - np.median(depth_values))))
            diameter = 2.0 * math.sqrt(area / math.pi) * z / fx
            detail.update({"depth_m": round(z, 4), "depth_mad_m": round(depth_mad, 4),
                           "diameter_m": round(diameter, 4), "depth_source": depth_source})
            allowed_mad = (self.fallback_depth_mad if depth_source == "contour_ring"
                           else self.max_depth_mad)
            if depth_mad > allowed_mad:
                rejection_counts["depth_mad"] += 1
                detail["rejection"] = "depth_mad"
                details.append(detail)
                continue
            if detail["shape_gate"] == "oblique_sphere":
                # Compare an eroded central patch with the visible red
                # boundary.  This is a local RGB-D surface measurement, not
                # a Gazebo/model prior.  A flat red box has near-zero signed
                # centre-to-edge depth difference and is rejected before it
                # can accumulate three otherwise consistent observations.
                core = cv2.erode(contour_mask, np.ones((7, 7), np.uint8),
                                 iterations=1)
                edge = (contour_mask > 0) & (core == 0)
                core_values = depth[(core > 0) & np.isfinite(depth) &
                                    (depth > .08) & (depth < self.max_depth)]
                edge_values = depth[edge & np.isfinite(depth) &
                                    (depth > .08) & (depth < self.max_depth)]
                if core_values.size >= 8 and edge_values.size >= 8:
                    convexity = float(np.median(edge_values) -
                                      np.median(core_values))
                    # The depth sag of a 0.30 m sphere decreases with range.
                    # A fixed 18 mm threshold is physically unattainable at
                    # roughly 6 m (D9 measured 1--3 mm), while a planar box
                    # still has near-zero signed sag. Keep the ordinary gate
                    # nearby; relax it only for remote RGB-D geometry using a
                    # conservative inverse-range floor.
                    required_convexity = self.oblique_minimum_depth_convexity
                    if z >= 5.0:
                        required_convexity = min(required_convexity,
                                                 max(.001, .006 / z))
                    detail["depth_convexity_m"] = round(convexity, 4)
                    detail["required_depth_convexity_m"] = round(
                        required_convexity, 4)
                    if convexity < required_convexity:
                        # At long range, a real 0.30 m sphere's signed depth
                        # sag is below the RGB-D quantisation step. Retain an
                        # exceptionally compact and depth-stable contour as a
                        # low-confidence observation; the tracker applies a
                        # separate five-frame, 15-degree confirmation gate.
                        far_profile_relaxed = (
                            z >= 5.0 and .24 <= diameter <= .42 and
                            circularity >= .75 and fill >= .88 and
                            fill <= .90 and
                            depth_mad <= .015)
                        if not far_profile_relaxed:
                            rejection_counts["depth_profile"] += 1
                            detail["rejection"] = "depth_profile"
                            details.append(detail)
                            continue
                        detail["profile_relaxed"] = True
                else:
                    # Do not turn a sparse valid depth image into a false
                    # negative.  Existing depth-MAD/metric/multi-view gates
                    # remain active when the profile cannot be sampled.
                    detail["depth_convexity_m"] = None
            if (detail["shape_gate"] in ("oblique_sphere", "partial_sphere") and
                    not oblique_sphere_metric_gate(
                        diameter, self.min_diameter, self.max_diameter,
                        self.oblique_max_diameter)):
                rejection_counts["metric_diameter"] += 1
                detail["rejection"] = "oblique_metric_diameter"
                details.append(detail)
                continue
            if diameter < self.min_diameter or diameter > self.max_diameter:
                rejection_counts["metric_diameter"] += 1
                detail["rejection"] = "metric_diameter"
                details.append(detail)
                continue
            # Optical frame (x right, y down, z forward) -> planar robot/map.
            forward, lateral = z, -(u - cx) * z / fx
            yaw = pose[3] + self.camera_yaw
            mx = pose[0] + forward * math.cos(yaw) - lateral * math.sin(yaw)
            my = pose[1] + forward * math.sin(yaw) + lateral * math.cos(yaw)
            confidence = min(1.0, .40 + .25 * circularity + .20 * fill + min(.15, area / 1200.0))
            observation = {"timestamp": round(stamp, 6), "room_id": room_id,
                           "map_position": {"x": round(mx, 4), "y": round(my, 4), "z": round(pose[2] + self.camera_z, 4)},
                           "camera_position": {"x": round(pose[0], 4), "y": round(pose[1], 4), "yaw": round(yaw, 5)},
                           "pose_sync_error_sec": (None if pose_sync_error is None else
                                                     round(float(pose_sync_error), 4)),
                           "pixel": {"u": u, "v": v, "area": round(area, 2), "aspect": round(aspect, 3),
                                     "circularity": round(circularity, 3), "fill_ratio": round(fill, 3),
                                     "shape_gate": detail["shape_gate"]},
                           "depth_m": round(z, 4), "depth_mad_m": round(depth_mad, 4),
                           "diameter_m": round(diameter, 4), "depth_source": depth_source,
                           "profile_relaxed": bool(detail.get("profile_relaxed", False)),
                           "confidence": round(confidence, 3)}
            self.pub.publish(String(data=json.dumps(observation, sort_keys=True)))
            self.write(observation)
            accepted += 1
            detail["accepted"] = True
            details.append(detail)
        # Restrict contour details so a red-rich frame cannot turn this into a
        # high-volume image log.  Totals above always preserve the full count.
        record = {"timestamp": round(stamp, 6), "room_id": room_id, "status": "processed",
                  "red_pixels": int(np.count_nonzero(mask)), "contours": len(contours),
                  "accepted": accepted, "rejections": rejection_counts,
                  "depth_sync_error_sec": (None if depth_sync_error is None else
                                             round(float(depth_sync_error), 6)),
                  "valid_depth_fraction": round(float(np.count_nonzero(np.isfinite(depth) & (depth > .08) &
                                                          (depth < self.max_depth))) / max(1, depth.size), 4),
                  "camera_position": {"x": round(pose[0], 4), "y": round(pose[1], 4),
                                      "yaw": round(pose[3] + self.camera_yaw, 5)},
                  "contour_details": details[:12]}
        self.write_funnel(record)
        if room_id:
            stats = self.room_search_stats.setdefault(str(room_id), {
                "frames": 0, "red_pixel_frames": 0,
                "red_contours_seen": 0,
                "accepted_observations": 0, "left_edge_red_frames": 0,
                "right_edge_red_frames": 0,
                "strict_accepted_observations": 0,
                "left_edge_clipped_ball_frames": 0,
                "right_edge_clipped_ball_frames": 0})
            stats["frames"] += 1
            if record["red_pixels"] > 0:
                stats["red_pixel_frames"] += 1
            stats["red_contours_seen"] += int(record["contours"])
            # A compact red object clipped at an image edge is useful active
            # search evidence even if it is deliberately rejected as an
            # unreliable sphere shape.  Publish only its side, never a world
            # position, so the manager can turn the existing camera fan
            # toward it without using truth/layout information.
            height, width = rgb.shape[:2]
            left_clipped_ball = False
            right_clipped_ball = False
            for item in details:
                bbox = item.get("bbox") if isinstance(item, dict) else None
                if not isinstance(bbox, list) or len(bbox) < 4:
                    continue
                x, _, w, _ = bbox
                if int(x) <= self.edge_hint_margin_px:
                    stats["left_edge_red_frames"] += 1
                if int(x) + int(w) >= width - self.edge_hint_margin_px:
                    stats["right_edge_red_frames"] += 1
                # Only this rejection means the contour already passed the
                # permissive sphere-shape test and was withheld solely because
                # its outline touched the image edge.  It is high-value active
                # search evidence; generic red-at-edge evidence often comes
                # from a distractor and must not consume another room turn.
                if item.get("rejection") in (
                        "edge_clipped_oblique",
                        "edge_clipped_partial_sphere"):
                    left_clipped_ball |= int(x) <= self.edge_hint_margin_px
                    right_clipped_ball |= (int(x) + int(w) >=
                                           width - self.edge_hint_margin_px)
            stats["left_edge_clipped_ball_frames"] += int(left_clipped_ball)
            stats["right_edge_clipped_ball_frames"] += int(right_clipped_ball)
            stats["accepted_observations"] += int(accepted)
            stats["strict_accepted_observations"] += sum(
                1 for item in details if item.get("accepted") and
                item.get("shape_gate") == "strict")
            self.search_status_pub.publish(String(data=json.dumps({
                "room_id": str(room_id), "timestamp": round(stamp, 6),
                **stats}, sort_keys=True)))
        self.save_debug_frame(rgb, mask, contours[:len(details)], details, stamp, room_id)


if __name__ == "__main__":
    rospy.init_node("red_ball_detector")
    RedBallDetector()
    rospy.spin()
