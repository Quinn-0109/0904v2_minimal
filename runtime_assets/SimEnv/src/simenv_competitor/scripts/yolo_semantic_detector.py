#!/usr/bin/env python3
"""YOLOv8 RGB-D red-ball detector with the legacy tracker interface.

The neural network replaces HSV/contour candidate generation.  Accepted
YOLO boxes are paired with aligned depth, projected into the FAST-LIO map
frame and published on ``/simenv/red_ball_observations`` so the existing
multi-view hazard tracker and result files remain unchanged.
"""

import json
import math
import os
import sys
import threading
import time
import traceback
from collections import deque

# When this node is launched with the YOLO virtualenv Python, ROS Debian
# modules such as rospy/rospkg still live in the system dist-packages paths.
for _path in (
    "/usr/lib/python3/dist-packages",
    "/opt/ros/noetic/lib/python3/dist-packages",
):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.append(_path)

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class YoloSemanticDetector:
    def __init__(self):
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._pose = None
        self._pose_history = deque(maxlen=300)
        self._depth = None
        self._depth_stamp = None
        self._camera = None
        self._context = {"enabled": False, "room_id": None}
        self._room_search_stats = {}
        self._last_stamp = -math.inf
        self._pose_hold_until = -math.inf

        self._model_path = os.path.abspath(rospy.get_param("~model_path", ""))
        self._site_packages = rospy.get_param(
            "~python_site_packages", "")
        self._device = self._resolve_device(rospy.get_param("~device", "auto"))
        self._confidence = float(rospy.get_param("~confidence", 0.35))
        self._iou = float(rospy.get_param("~iou", 0.45))
        self._image_size = int(rospy.get_param("~image_size", 640))
        self._rate = float(rospy.get_param("~processing_rate_hz", 6.0))
        self._room_only = bool(rospy.get_param("~room_only_gate", True))
        self._publish_red_ball = bool(rospy.get_param(
            "~publish_red_ball_observations", True))
        self._maximum_pose_sync_error = float(rospy.get_param(
            "~maximum_pose_sync_error_sec", 0.20))
        self._maximum_depth_sync_error = float(rospy.get_param(
            "~maximum_depth_sync_error_sec", 0.25))
        self._maximum_pose_jump = float(rospy.get_param(
            "~maximum_pose_jump_m", 0.75))
        self._pose_jump_hold_seconds = float(rospy.get_param(
            "~pose_jump_hold_seconds", 1.0))
        self._camera_yaw = math.radians(float(rospy.get_param(
            "~camera_yaw_offset_deg", 0.0)))
        self._camera_z = float(rospy.get_param("~camera_height_m", 0.32))
        self._max_depth = float(rospy.get_param("~max_depth", 10.0))
        self._minimum_depth_samples = int(rospy.get_param(
            "~minimum_depth_samples", 8))
        self._min_confidence_by_label = {
            "red_ball": float(rospy.get_param("~red_ball_min_confidence", 0.70)),
            "stair": float(rospy.get_param("~stair_min_confidence", 0.80)),
            "elevator": float(rospy.get_param("~elevator_min_confidence", 0.80)),
            "corridor_entrance": float(rospy.get_param(
                "~corridor_entrance_min_confidence", 0.55)),
        }
        self._semantic_edge_margin_px = int(rospy.get_param(
            "~semantic_edge_margin_px", 8))
        self._semantic_max_area_ratio = float(rospy.get_param(
            "~semantic_max_area_ratio", 0.35))
        self._semantic_min_depth = float(rospy.get_param(
            "~semantic_min_depth_m", 0.80))
        self._semantic_max_depth = float(rospy.get_param(
            "~semantic_max_depth_m", 8.0))
        self._corridor_min_area_ratio = float(rospy.get_param(
            "~corridor_entrance_min_area_ratio", 0.002))
        self._corridor_max_area_ratio = float(rospy.get_param(
            "~corridor_entrance_max_area_ratio", 0.70))
        self._corridor_min_depth = float(rospy.get_param(
            "~corridor_entrance_min_depth_m", 0.45))
        self._corridor_max_depth = float(rospy.get_param(
            "~corridor_entrance_max_depth_m", 12.0))
        self._red_ball_min_area_ratio = float(rospy.get_param(
            "~red_ball_min_area_ratio", 0.0005))
        self._red_ball_max_area_ratio = float(rospy.get_param(
            "~red_ball_max_area_ratio", 0.12))
        self._red_ball_max_aspect_ratio = float(rospy.get_param(
            "~red_ball_max_aspect_ratio", 1.8))

        self._output_dir = os.path.abspath(rospy.get_param(
            "~output_dir", os.getcwd()))
        self._log_path = os.path.join(
            self._output_dir, "logs", "yolo_semantic_observations.jsonl")
        self._red_log_path = os.path.join(
            self._output_dir, "logs", "red_ball_observations.jsonl")
        self._funnel_log_path = os.path.join(
            self._output_dir, "logs", "yolo_red_ball_detection_funnel.jsonl")
        self._startup_log_path = os.path.join(
            self._output_dir, "logs", "yolo_semantic_startup.jsonl")
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)

        self._log_startup("LOADING_MODEL")
        try:
            self._model = self._load_model()
        except Exception as exc:
            self._log_startup(
                "MODEL_LOAD_FAILED", error_type=type(exc).__name__,
                error=str(exc), traceback=traceback.format_exc())
            raise
        self._log_startup("MODEL_READY")
        self._semantic_pub = rospy.Publisher(
            "/simenv/yolo_semantic_observations", String, queue_size=30)
        self._stair_pub = rospy.Publisher(
            "/simenv/stair_visual_candidates", String, queue_size=10)
        self._elevator_pub = rospy.Publisher(
            "/simenv/elevator_visual_candidates", String, queue_size=10)
        self._corridor_pub = rospy.Publisher(
            "/simenv/corridor_visual_candidates", String, queue_size=10)
        self._red_ball_pub = rospy.Publisher(
            "/simenv/red_ball_observations", String, queue_size=30)
        self._search_status_pub = rospy.Publisher(
            "/simenv/red_ball_search_status", String, queue_size=10)
        rospy.Subscriber(rospy.get_param("~odom_topic", "/Odometry"),
                         Odometry, self._on_odom, queue_size=30)
        rospy.Subscriber(rospy.get_param("~depth_topic", "/real_sense/depth/image_raw"),
                         Image, self._on_depth, queue_size=3)
        rospy.Subscriber(rospy.get_param("~camera_info_topic", "/real_sense/rgb/camera_info"),
                         CameraInfo, self._on_info, queue_size=2)
        rospy.Subscriber("/simenv/room_detection_context", String,
                         self._on_context, queue_size=5)
        rospy.Subscriber(rospy.get_param("~rgb_topic", "/real_sense/rgb/image_raw"),
                         Image, self._on_image, queue_size=3)
        self._log_startup("ROS_INTERFACES_READY")
        rospy.loginfo(
            "yolo_semantic_detector: model=%s device=%s rate=%.1fHz "
            "red_to_tracker=%s room_only=%s",
            self._model_path, self._device, self._rate,
            self._publish_red_ball, self._room_only)

    def _log_startup(self, phase, **details):
        payload = {
            "timestamp": round(time.time(), 6),
            "phase": str(phase),
            "pid": int(os.getpid()),
            "model_path": self._model_path,
            "device": self._device,
        }
        payload.update(details)
        try:
            self._append_jsonl(self._startup_log_path, payload)
        except OSError as exc:
            rospy.logwarn("YOLO startup log unavailable: %s", exc)

    @staticmethod
    def _resolve_device(requested):
        value = str(requested or "auto").strip()
        if value.lower() != "auto":
            return value
        try:
            import torch
            return "0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @staticmethod
    def _normalise_label(label):
        value = str(label).strip().lower().replace("-", "_").replace(" ", "_")
        return {
            "redball": "red_ball",
            "ball_red": "red_ball",
            "red_sphere": "red_ball",
        }.get(value, value)

    def _load_model(self):
        if self._site_packages and os.path.isdir(self._site_packages):
            sys.path.insert(0, self._site_packages)
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is unavailable; set ~python_site_packages to the YOLO venv"
            ) from exc
        if not self._model_path or not os.path.exists(self._model_path):
            raise RuntimeError("YOLO model file does not exist: %s" %
                               self._model_path)
        model = YOLO(self._model_path)
        labels = {self._normalise_label(value)
                  for value in dict(model.names).values()}
        if "red_ball" not in labels:
            raise RuntimeError(
                "YOLO model has no red_ball class: %s" % sorted(labels))
        return model

    def _on_odom(self, message):
        p = message.pose.pose.position
        q = message.pose.pose.orientation
        stamp = message.header.stamp.to_sec() or time.time()
        pose = (float(p.x), float(p.y), float(p.z),
                float(yaw_from_quaternion(q)), float(stamp))
        with self._lock:
            previous = self._pose_history[-1] if self._pose_history else None
            if (not all(math.isfinite(value) for value in pose[:4]) or
                    (previous is not None and
                     math.hypot(pose[0] - previous[0],
                                pose[1] - previous[1]) >
                     self._maximum_pose_jump)):
                self._pose_hold_until = max(
                    self._pose_hold_until,
                    time.monotonic() + self._pose_jump_hold_seconds)
                self._pose_history.clear()
            self._pose = pose
            self._pose_history.append(pose)

    def _on_depth(self, message):
        try:
            image = self._bridge.imgmsg_to_cv2(
                message, desired_encoding="passthrough")
        except CvBridgeError:
            return
        if image.dtype == np.uint16:
            image = image.astype(np.float32) * 0.001
        else:
            image = image.astype(np.float32)
        image[~np.isfinite(image)] = 0.0
        stamp = message.header.stamp.to_sec() or time.time()
        with self._lock:
            self._depth = image
            self._depth_stamp = float(stamp)

    def _on_info(self, message):
        with self._lock:
            self._camera = (float(message.K[0]), float(message.K[4]),
                            float(message.K[2]), float(message.K[5]))

    def _on_context(self, message):
        try:
            value = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if isinstance(value, dict):
            with self._lock:
                self._context = value

    @staticmethod
    def _interpolate_pose(first, second, stamp):
        span = float(second[4]) - float(first[4])
        if span <= 1e-6:
            return first
        ratio = min(1.0, max(0.0, (float(stamp) - float(first[4])) / span))
        yaw_delta = (float(second[3]) - float(first[3]) + math.pi) % (
            2.0 * math.pi) - math.pi
        return tuple(float(first[index]) + ratio *
                     (float(second[index]) - float(first[index]))
                     for index in range(3)) + (
                         float(first[3]) + ratio * yaw_delta, float(stamp))

    def _pose_at_stamp(self, stamp):
        history = list(self._pose_history)
        if not history:
            return self._pose, None
        before = [item for item in history if item[4] <= stamp]
        after = [item for item in history if item[4] >= stamp]
        if before and after:
            left, right = before[-1], after[0]
            residual = max(abs(float(stamp) - left[4]),
                           abs(right[4] - float(stamp)))
            if residual <= self._maximum_pose_sync_error:
                return self._interpolate_pose(left, right, stamp), residual
        nearest = min(history, key=lambda item: abs(float(item[4]) - stamp))
        residual = abs(float(nearest[4]) - stamp)
        if residual <= self._maximum_pose_sync_error:
            return nearest, residual
        return None, residual

    def _fallback_intrinsics(self, rgb):
        height, width = rgb.shape[:2]
        fx = width / (2.0 * math.tan(math.radians(60.0) / 2.0))
        return fx, fx, width / 2.0, height / 2.0

    def _depth_for_box(self, depth, x1, y1, x2, y2, rgb_shape=None):
        height, width = depth.shape[:2]
        if rgb_shape is not None:
            rgb_height, rgb_width = rgb_shape[:2]
            if rgb_width > 0 and rgb_height > 0:
                scale_x = float(width) / float(rgb_width)
                scale_y = float(height) / float(rgb_height)
                x1, x2 = x1 * scale_x, x2 * scale_x
                y1, y2 = y1 * scale_y, y2 * scale_y
        ix1 = max(0, min(width - 1, int(round(x1))))
        ix2 = max(0, min(width, int(round(x2))))
        iy1 = max(0, min(height - 1, int(round(y1))))
        iy2 = max(0, min(height, int(round(y2))))
        if ix2 <= ix1 or iy2 <= iy1:
            return None
        cx = int(round((ix1 + ix2) * 0.5))
        cy = int(round((iy1 + iy2) * 0.5))
        patch = depth[max(0, cy - 4):min(height, cy + 5),
                      max(0, cx - 4):min(width, cx + 5)]
        values = patch[np.isfinite(patch) & (patch > 0.08) &
                       (patch < self._max_depth)]
        if values.size < self._minimum_depth_samples:
            roi = depth[iy1:iy2, ix1:ix2]
            values = roi[np.isfinite(roi) & (roi > 0.08) &
                         (roi < self._max_depth)]
        if values.size < self._minimum_depth_samples:
            return None
        return float(np.median(values))

    def _map_position(self, pose, intrinsics, bbox, depth_m):
        fx, _, cx, _ = intrinsics
        x1, y1, x2, y2 = bbox
        u = 0.5 * (x1 + x2)
        v = 0.5 * (y1 + y2)
        forward = float(depth_m)
        lateral = -(u - cx) * forward / fx
        yaw = float(pose[3]) + self._camera_yaw
        mx = float(pose[0]) + forward * math.cos(yaw) - lateral * math.sin(yaw)
        my = float(pose[1]) + forward * math.sin(yaw) + lateral * math.cos(yaw)
        return {
            "x": round(mx, 4),
            "y": round(my, 4),
            "z": round(float(pose[2]) + self._camera_z, 4),
        }, {"u": int(round(u)), "v": int(round(v))}

    @staticmethod
    def _append_jsonl(path, payload):
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")

    def _log(self, payload):
        try:
            self._append_jsonl(self._log_path, payload)
        except OSError as exc:
            rospy.logwarn_throttle(10.0, "YOLO semantic log unavailable: %s", exc)

    def _log_funnel(self, payload):
        try:
            self._append_jsonl(self._funnel_log_path, payload)
        except OSError as exc:
            rospy.logwarn_throttle(10.0, "YOLO funnel log unavailable: %s", exc)

    def _publish(self, payload):
        encoded = json.dumps(payload, sort_keys=True)
        self._semantic_pub.publish(String(data=encoded))
        label = payload.get("class")
        if not payload.get("accepted", False):
            return
        if label == "stair":
            self._stair_pub.publish(String(data=encoded))
        elif label == "elevator":
            self._elevator_pub.publish(String(data=encoded))
        elif label == "corridor_entrance":
            self._corridor_pub.publish(String(data=encoded))
        elif label == "red_ball" and self._publish_red_ball:
            red_payload = {
                "timestamp": payload["timestamp"],
                "room_id": payload.get("room_id"),
                "map_position": payload["map_position"],
                "camera_position": payload["camera_position"],
                "pose_sync_error_sec": payload.get("pose_sync_error_sec"),
                "pixel": {
                    "u": payload["pixel"]["u"],
                    "v": payload["pixel"]["v"],
                    "area": payload["pixel"]["area"],
                    "shape_gate": "yolo",
                    "bbox": payload["pixel"]["bbox"],
                },
                "depth_m": payload["depth_m"],
                "confidence": payload["confidence"],
                "source": "yolov8_rgbd",
            }
            red_encoded = json.dumps(red_payload, sort_keys=True)
            self._red_ball_pub.publish(String(data=red_encoded))
            try:
                self._append_jsonl(self._red_log_path, red_payload)
            except OSError as exc:
                rospy.logwarn_throttle(
                    10.0, "YOLO red-ball log unavailable: %s", exc)

    def _quality_gate(self, label, confidence, bbox, depth_m, image_shape,
                      pose_sync_error):
        height, width = image_shape[:2]
        x1, y1, x2, y2 = bbox
        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        area_ratio = (box_width * box_height) / max(1.0, float(width * height))
        margin = float(self._semantic_edge_margin_px)
        edge_clipped = (
            x1 <= margin or y1 <= margin or
            x2 >= float(width) - margin or y2 >= float(height) - margin)
        reasons = []

        minimum_confidence = self._min_confidence_by_label.get(
            label, self._confidence)
        if confidence < minimum_confidence:
            reasons.append("low_confidence")
        if pose_sync_error is not None and pose_sync_error > self._maximum_pose_sync_error:
            reasons.append("stale_pose_sync")

        if label in ("stair", "elevator"):
            # The current model often mistakes partially visible door/wall
            # structures at the image border for stairs/elevators.  Keep those
            # observations in the raw log, but do not publish them downstream.
            if edge_clipped:
                reasons.append("edge_clipped_large_structure")
            if area_ratio > self._semantic_max_area_ratio:
                reasons.append("oversized_structure_box")
            if depth_m < self._semantic_min_depth or depth_m > self._semantic_max_depth:
                reasons.append("semantic_depth_out_of_range")
        elif label == "corridor_entrance":
            if area_ratio < self._corridor_min_area_ratio:
                reasons.append("corridor_box_too_small")
            if area_ratio > self._corridor_max_area_ratio:
                reasons.append("corridor_box_too_large")
            if depth_m < self._corridor_min_depth or depth_m > self._corridor_max_depth:
                reasons.append("corridor_depth_out_of_range")
        elif label == "red_ball":
            if area_ratio < self._red_ball_min_area_ratio:
                reasons.append("red_ball_box_too_small")
            if area_ratio > self._red_ball_max_area_ratio:
                reasons.append("red_ball_box_too_large")
            aspect = max(box_width, box_height) / max(1.0, min(box_width, box_height))
            if aspect > self._red_ball_max_aspect_ratio:
                reasons.append("red_ball_box_not_round")
            if depth_m <= 0.15 or depth_m > self._max_depth:
                reasons.append("red_ball_depth_out_of_range")

        return not reasons, {
            "area_ratio": round(float(area_ratio), 6),
            "edge_clipped": bool(edge_clipped),
            "rejection_reasons": reasons,
        }

    def _on_image(self, message):
        stamp = message.header.stamp.to_sec() or time.time()
        if stamp - self._last_stamp < 1.0 / max(0.5, self._rate):
            return
        self._last_stamp = stamp
        with self._lock:
            pose, pose_sync_error = self._pose_at_stamp(stamp)
            depth = self._depth
            depth_stamp = self._depth_stamp
            intrinsics = self._camera
            context = dict(self._context)
            pose_held = time.monotonic() < self._pose_hold_until
        if self._room_only and not context.get("enabled"):
            return
        room_id = context.get("room_id")
        depth_sync_error = (None if depth_stamp is None else
                            abs(float(stamp) - float(depth_stamp)))
        if pose_held:
            self._log_funnel({
                "timestamp": round(stamp, 6), "room_id": room_id,
                "status": "pose_jump_hold"})
            return
        if (pose is None or depth is None or depth_sync_error is None or
                depth_sync_error > self._maximum_depth_sync_error):
            self._log_funnel({
                "timestamp": round(stamp, 6), "room_id": room_id,
                "status": "missing_or_stale_input",
                "has_pose": pose is not None, "has_depth": depth is not None,
                "pose_sync_error_sec": pose_sync_error,
                "depth_sync_error_sec": depth_sync_error})
            return
        try:
            rgb = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except CvBridgeError:
            return
        if intrinsics is None:
            intrinsics = self._fallback_intrinsics(rgb)

        try:
            results = self._model.predict(
                source=rgb, imgsz=self._image_size, conf=self._confidence,
                iou=self._iou, device=self._device, verbose=False)
        except Exception as exc:  # keep one bad frame from killing the mission
            rospy.logerr_throttle(5.0, "YOLO red-ball inference failed: %s", exc)
            self._log_funnel({
                "timestamp": round(stamp, 6), "room_id": room_id,
                "status": "inference_failed", "error": str(exc)})
            return
        names = results[0].names
        frame_payloads = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = self._normalise_label(names.get(cls_id, cls_id))
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            depth_m = self._depth_for_box(
                depth, x1, y1, x2, y2, rgb.shape)
            if depth_m is None:
                frame_payloads.append({
                    "class": label, "confidence": confidence,
                    "accepted": False,
                    "rejection_reasons": ["red_ball_depth_unavailable"],
                    "pixel": {"bbox": [x1, y1, x2, y2]}})
                continue
            map_position, pixel_center = self._map_position(
                pose, intrinsics, (x1, y1, x2, y2), depth_m)
            accepted, quality = self._quality_gate(
                label, confidence, (x1, y1, x2, y2), depth_m, rgb.shape,
                pose_sync_error)
            payload = {
                "schema": "simenv_yolo_semantic_observation_v1",
                "timestamp": round(stamp, 6),
                "class": label,
                "confidence": round(confidence, 4),
                "accepted": bool(accepted),
                "rejection_reasons": quality["rejection_reasons"],
                "room_id": context.get("room_id"),
                "map_position": map_position,
                "camera_position": {
                    "x": round(float(pose[0]), 4),
                    "y": round(float(pose[1]), 4),
                    "yaw": round(float(pose[3]) + self._camera_yaw, 5),
                },
                "pose_sync_error_sec": (
                    None if pose_sync_error is None else
                    round(float(pose_sync_error), 4)),
                "pixel": {
                    "u": pixel_center["u"],
                    "v": pixel_center["v"],
                    "bbox": [round(x1, 1), round(y1, 1),
                             round(x2, 1), round(y2, 1)],
                    "area": round(max(0.0, x2 - x1) * max(0.0, y2 - y1), 2),
                    "area_ratio": quality["area_ratio"],
                    "edge_clipped": quality["edge_clipped"],
                },
                "depth_m": round(float(depth_m), 4),
                "depth_sync_error_sec": round(float(depth_sync_error), 4),
                "image_width": int(rgb.shape[1]),
                "image_height": int(rgb.shape[0]),
                "source": "yolov8_rgbd",
            }
            self._publish(payload)
            self._log(payload)
            frame_payloads.append(payload)

        accepted_red = [item for item in frame_payloads
                        if item.get("class") == "red_ball" and
                        item.get("accepted")]
        red_boxes = [item for item in frame_payloads
                     if item.get("class") == "red_ball"]
        self._log_funnel({
            "timestamp": round(stamp, 6), "room_id": room_id,
            "status": "processed", "backend": "yolov8s_rgbd",
            "detections": len(frame_payloads),
            "red_ball_boxes": len(red_boxes),
            "accepted": len(accepted_red),
            "depth_sync_error_sec": round(float(depth_sync_error), 4),
            "pose_sync_error_sec": (None if pose_sync_error is None else
                                     round(float(pose_sync_error), 4)),
        })
        if room_id:
            key = str(room_id)
            stats = self._room_search_stats.setdefault(key, {
                "frames": 0, "red_pixel_frames": 0,
                "red_contours_seen": 0, "accepted_observations": 0,
                "left_edge_red_frames": 0,
                "right_edge_red_frames": 0,
                "strict_accepted_observations": 0,
                "left_edge_clipped_ball_frames": 0,
                "right_edge_clipped_ball_frames": 0,
            })
            stats["frames"] += 1
            stats["red_pixel_frames"] += int(bool(red_boxes))
            stats["red_contours_seen"] += len(red_boxes)
            stats["accepted_observations"] += len(accepted_red)
            stats["strict_accepted_observations"] += len(accepted_red)
            left_edge = any(
                float((item.get("pixel") or {}).get("bbox", [999])[0]) <=
                self._semantic_edge_margin_px for item in red_boxes)
            right_edge = any(
                len((item.get("pixel") or {}).get("bbox", [])) >= 4 and
                float((item.get("pixel") or {})["bbox"][2]) >=
                rgb.shape[1] - self._semantic_edge_margin_px
                for item in red_boxes)
            accepted_left = any(
                bool((item.get("pixel") or {}).get("edge_clipped")) and
                float((item.get("pixel") or {}).get("bbox", [999])[0]) <=
                self._semantic_edge_margin_px for item in accepted_red)
            accepted_right = any(
                bool((item.get("pixel") or {}).get("edge_clipped")) and
                len((item.get("pixel") or {}).get("bbox", [])) >= 4 and
                float((item.get("pixel") or {})["bbox"][2]) >=
                rgb.shape[1] - self._semantic_edge_margin_px
                for item in accepted_red)
            stats["left_edge_red_frames"] += int(left_edge)
            stats["right_edge_red_frames"] += int(right_edge)
            stats["left_edge_clipped_ball_frames"] += int(accepted_left)
            stats["right_edge_clipped_ball_frames"] += int(accepted_right)
            self._search_status_pub.publish(String(data=json.dumps({
                "room_id": key, "timestamp": round(stamp, 6),
                "detector_backend": "yolov8s_rgbd", **stats,
            }, sort_keys=True)))


def main():
    rospy.init_node("yolo_semantic_detector")
    try:
        YoloSemanticDetector()
    except Exception as exc:
        # output=screen does not reliably preserve Python stderr in the ROS
        # node log.  Keep the traceback visible there as well as in the
        # startup JSONL written by the constructor.
        rospy.logfatal("YOLO semantic detector startup failed: %s\n%s",
                       exc, traceback.format_exc())
        raise
    rospy.spin()


if __name__ == "__main__":
    main()
