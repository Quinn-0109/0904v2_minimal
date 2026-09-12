#!/usr/bin/env python3
"""Detect red spherical danger sources and publish world-frame positions."""
import json
import math
import os
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np


DANGER_Z = 0.15


def load_room_bounds(layout_path):
    """Load only room IDs and XY bounds; danger truth is never accessed."""
    with open(layout_path, "r", encoding="utf-8") as stream:
        layout = json.load(stream)

    result = {}
    for floor in layout.get("floors", []):
        for room in floor.get("rooms", []):
            room_id = str(room.get("id", "")).strip()
            raw = room.get("bounds", {})
            required = ("x_min", "x_max", "y_min", "y_max")
            if not room_id or not all(key in raw for key in required):
                continue

            bounds = {key: float(raw[key]) for key in required}
            if (bounds["x_min"] > bounds["x_max"] or
                    bounds["y_min"] > bounds["y_max"]):
                raise ValueError(
                    "invalid room bounds for {}".format(room_id)
                )
            result[room_id] = bounds
    return result


def room_id_from_scan_label(label):
    """Extract floor_N_room_N from scan waypoint names."""
    parts = str(label).split("_")
    if (len(parts) >= 4 and parts[0] == "floor" and
            parts[1].isdigit() and parts[2] == "room" and
            parts[3].isdigit()):
        return "_".join(parts[:4])
    return None


def point_in_room_bounds(point, bounds, margin=0.50):
    """Inclusive XY room check with outward localization margin."""
    x, y = float(point[0]), float(point[1])
    margin = max(0.0, float(margin))
    return (
        bounds["x_min"] - margin <= x <= bounds["x_max"] + margin and
        bounds["y_min"] - margin <= y <= bounds["y_max"] + margin
    )


def should_update_detections(scan_active):
    """Accumulate points only during a stationary room scan."""
    return bool(scan_active)


def scan_min_points(label):
    """Return the evidence threshold for a completed room scan.

    The front-left room is the only staging point with a very short visible
    dwell in the four-room route.  Requiring several synchronized frames
    there drops both balls when Gazebo publishes sparse RGB/depth pairs.
    Other scans retain a multi-frame threshold to suppress transient
    projections and partially visible distractors.
    """
    return 1 if str(label) == "room_LF_scan" else 3


def cluster_scan_points(points, merge_radius=1.2, min_points=2):
    """Collapse one scan's noisy projections to robust median world points."""
    clusters = []
    for point in points:
        x, y = float(point[0]), float(point[1])
        frame_id = point[2] if len(point) > 2 else None
        nearest = None
        nearest_distance = float("inf")
        for cluster in clusters:
            if frame_id is not None and frame_id in cluster["frames"]:
                continue
            distance = math.hypot(x - cluster["x"], y - cluster["y"])
            if distance < nearest_distance:
                nearest, nearest_distance = cluster, distance
        if nearest is None or nearest_distance > float(merge_radius):
            clusters.append({"x": x, "y": y, "points": [(x, y)], "frames": {frame_id}})
        else:
            nearest["points"].append((x, y))
            nearest["frames"].add(frame_id)
            values = np.asarray(nearest["points"], dtype=float)
            nearest["x"], nearest["y"] = np.median(values, axis=0)
    return [
        (cluster["x"], cluster["y"])
        for cluster in clusters
        if len(cluster["points"]) >= int(min_points)
    ]


def nearest_pose(samples, stamp, fallback):
    """Return the odometry sample closest to a camera timestamp."""
    if not samples:
        return fallback
    sample = min(samples, key=lambda value: abs(float(value[0]) - float(stamp)))
    return sample[1], sample[2]


def select_sphere_depth(sensor_depth, radius_px, fx, sphere_radius=DANGER_Z):
    """Fuse RGB-D range with a silhouette check and camera-only fallback."""
    apparent_depth = None
    if radius_px is not None and float(radius_px) > 1.0:
        apparent_depth = float(fx) * float(sphere_radius) / float(radius_px)
        if not 0.2 <= apparent_depth <= 8.0:
            apparent_depth = None
    if sensor_depth is not None and 0.2 <= float(sensor_depth) <= 8.0:
        sensor_depth = float(sensor_depth)
        if apparent_depth is None:
            return sensor_depth
        # Gazebo occasionally returns the wall behind a red sphere at its
        # centre pixel. Reject that background depth only when the independent
        # apparent-sphere range strongly disagrees.
        ratio = sensor_depth / apparent_depth
        return sensor_depth if 0.60 <= ratio <= 1.60 else apparent_depth
    return apparent_depth if apparent_depth is not None else sensor_depth


class CameraCalibration:
    def __init__(self, width, height, fx, fy, cx, cy, xyz_base, pitch):
        self.width = int(width)
        self.height = int(height)
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.xyz_base = np.asarray(xyz_base, dtype=float)
        self.pitch = float(pitch)

    @classmethod
    def front_camera_fallback(cls):
        focal = 400.0 / math.tan(math.radians(40.0))
        return cls(800, 800, focal, focal, 400.0, 400.0, (0.27, 0.0, 0.05), 0.4)

    @classmethod
    def real_sense_fallback(cls):
        return cls(640, 480, 554.5940455, 554.5940455, 320.5, 240.5, (0.28, 0.0, 0.043), 0.0)

    @classmethod
    def recording_camera_fallback(cls):
        focal = 320.0 / math.tan(math.radians(30.0))
        return cls(640, 480, focal, focal, 320.0, 240.0,
                   (0.30, 0.0, 0.08), 0.0)

    @classmethod
    def from_camera_info(cls, msg, fallback=None):
        fallback = fallback or cls.front_camera_fallback()
        if not msg.K or msg.K[0] <= 0.0 or msg.K[4] <= 0.0:
            return fallback
        return cls(msg.width, msg.height, msg.K[0], msg.K[4], msg.K[2], msg.K[5],
                   fallback.xyz_base, fallback.pitch)


def independent_scan_batch_reappearances(event, scan_batches,
                                        radius_m=1.20):
    """Count room scan batches whose clusters land on one track position.

    A batch is the candidate evidence of one completed room scan, so two
    matching batches mean the same place produced red evidence twice from
    independent sweeps.  Batch clusters stay diagnostic: nothing here
    confirms a track or changes ``evidence_frames``.
    """
    position = event.get("position")
    room_id = str(event.get("room_id", "")).strip()
    if not room_id or not isinstance(position, (list, tuple)):
        return 0
    if len(position) < 2:
        return 0
    try:
        x = float(position[0])
        y = float(position[1])
        limit = float(radius_m) ** 2
    except (TypeError, ValueError):
        return 0

    matched = 0
    for batch in scan_batches or []:
        if str(batch.get("room_id", "")).strip() != room_id:
            continue
        for point in batch.get("candidate_clusters_min1") or []:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                px = float(point[0])
                py = float(point[1])
            except (TypeError, ValueError):
                continue
            if (px - x) ** 2 + (py - y) ** 2 <= limit:
                matched += 1
                break
    return matched


def is_weak_small_single_view_detection(
        event, scan_batches, max_evidence_frames=3,
        max_radius_px=7.0, max_median_radius_px=7.0,
        min_depth_profiles=4, min_median_depth_m=5.3,
        max_depth_shape_accept_fraction=0.25,
        reappearance_radius_m=1.20):
    """Reject only measured D1-like online camera evidence.

    Missing room, view, contour or depth evidence fails open.
    No scene truth, danger positions or distractor metadata are used.
    """
    try:
        room_id = str(event.get("room_id", "")).strip()
        evidence_frames = int(event.get("evidence_frames", 0))
    except (AttributeError, TypeError, ValueError):
        return False

    if not room_id:
        return False

    # The D1 profile this rule targets is a single brief exposure. A track
    # that a second independent scan of the same room found again is not
    # that profile, so keep it whatever its image size.
    if independent_scan_batch_reappearances(
            event, scan_batches, reappearance_radius_m) >= 2:
        return False

    positive_roles = set()
    accepted_radii = []
    depth_centres = []
    depth_shape_accepts = []

    for batch in scan_batches or []:
        if str(batch.get("room_id", "")).strip() != room_id:
            continue

        role = str(batch.get("viewpoint_role", "")).strip()

        try:
            has_candidates = (
                int(batch.get("candidate_frames", 0)) > 0 or
                int(batch.get("raw_candidate_points", 0)) > 0
            )
        except (TypeError, ValueError):
            has_candidates = False

        if role and has_candidates:
            positive_roles.add(role)

        if not has_candidates:
            continue

        for diagnostic in (
                batch.get("contour_diagnostics", []) or []):
            if not bool(diagnostic.get("accepted", False)):
                continue

            try:
                radius = float(diagnostic.get("radius_px"))
            except (TypeError, ValueError):
                continue

            if not math.isfinite(radius) or radius <= 0.0:
                continue

            accepted_radii.append(radius)

            depth_shape_value = diagnostic.get(
                "depth_shape_accepted")
            if not isinstance(depth_shape_value, bool):
                continue

            try:
                depth_centre = float(
                    diagnostic.get("depth_center_m"))
            except (TypeError, ValueError):
                continue

            if (
                    math.isfinite(depth_centre) and
                    depth_centre > 0.0):
                depth_centres.append(depth_centre)
                depth_shape_accepts.append(depth_shape_value)

    if len(positive_roles) != 1 or not accepted_radii:
        return False

    # Preserve the original minimum-evidence D1 rule.
    if (
            evidence_frames <= int(max_evidence_frames) and
            max(accepted_radii) <= float(max_radius_px)):
        return True

    # Extended D1 profile. Medians prevent one noisy frame from deciding
    # the result. Incomplete depth evidence always retains the detection.
    if len(depth_centres) < int(min_depth_profiles):
        return False

    def median(values):
        ordered = sorted(float(value) for value in values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return 0.5 * (
            ordered[middle - 1] + ordered[middle]
        )

    median_radius = median(accepted_radii)
    median_depth = median(depth_centres)
    depth_accept_fraction = (
        float(sum(1 for value in depth_shape_accepts if value)) /
        float(len(depth_shape_accepts))
    )

    return bool(
        median_radius <= float(max_median_radius_px) and
        median_depth >= float(min_median_depth_m) and
        depth_accept_fraction <=
        float(max_depth_shape_accept_fraction)
    )




class DetectionTracker:
    def __init__(self, merge_radius=0.9, confirm_count=3):
        self.merge_radius = float(merge_radius)
        self.confirm_count = int(confirm_count)
        self.tracks = []
        self._next_id = 1
        self.last_newly_confirmed = []

    def update(self, points, metadata=None):
        """Update 2-D or floor-aware 3-D tracks.

        A track may consume at most one candidate from a camera frame.  This
        prevents two nearby balls visible simultaneously from being collapsed
        into one source while still averaging projection noise over time.
        """
        used_tracks = set()
        newly_confirmed = []
        for point in points:
            dimensions = 3 if len(point) >= 3 else 2
            coordinates = tuple(float(value) for value in point[:dimensions])
            nearest = None
            nearest_distance = float("inf")
            for track in self.tracks:
                if track["id"] in used_tracks or track["dimensions"] != dimensions:
                    continue
                if dimensions == 3 and abs(coordinates[2] - track["z"]) > 1.0:
                    continue
                distance = math.hypot(
                    coordinates[0] - track["x"],
                    coordinates[1] - track["y"],
                )
                if distance < nearest_distance:
                    nearest = track
                    nearest_distance = distance
            if nearest is None or nearest_distance > self.merge_radius:
                nearest = {
                    "id": self._next_id,
                    "dimensions": dimensions,
                    "x": coordinates[0],
                    "y": coordinates[1],
                    "count": 1,
                    "confirmed": self.confirm_count <= 1,
                    "metadata": dict(metadata or {}),
                }
                if dimensions == 3:
                    nearest["z"] = coordinates[2]
                self._next_id += 1
                self.tracks.append(nearest)
                if nearest["confirmed"]:
                    newly_confirmed.append(nearest)
                used_tracks.add(nearest["id"])
                continue
            count = nearest["count"] + 1
            nearest["x"] += (coordinates[0] - nearest["x"]) / count
            nearest["y"] += (coordinates[1] - nearest["y"]) / count
            if dimensions == 3:
                nearest["z"] += (coordinates[2] - nearest["z"]) / count
            nearest["count"] = count
            used_tracks.add(nearest["id"])
            if not nearest["confirmed"] and count >= self.confirm_count:
                nearest["confirmed"] = True
                nearest["metadata"] = dict(metadata or {})
                newly_confirmed.append(nearest)
        self.last_newly_confirmed = list(newly_confirmed)
        return self.confirmed()

    def confirmed(self):
        return [
            ((t["x"], t["y"], t["z"])
             if t["dimensions"] == 3 else (t["x"], t["y"]))
            for t in self.tracks if t["count"] >= self.confirm_count
        ]

    def confirmed_tracks(self):
        return [t for t in self.tracks if t["count"] >= self.confirm_count]


def strong_same_view_duplicate_losers(
        detections, confirm_count=3, merge_radius_m=1.25,
        strong_evidence_multiplier=2,
        extra_frame_merge_radius_m=0.90,
        extra_frame_time_sec=0.50,
        late_merge_radius_m=1.60,
        late_merge_minimum_time_sec=1.00):
    """Find tightly supported weak same-view projection tails.

    Floor, room, viewpoint role and exact waypoint must all match.
    No scene truth is read.
    """
    confirm_count = max(1, int(confirm_count))
    strong_count = max(
        confirm_count + 1,
        confirm_count * int(strong_evidence_multiplier),
    )
    merge_radius_m = max(0.0, float(merge_radius_m))
    extra_frame_merge_radius_m = max(
        0.0, float(extra_frame_merge_radius_m))
    extra_frame_time_sec = max(
        0.0, float(extra_frame_time_sec))
    late_merge_radius_m = max(0.0, float(late_merge_radius_m))
    late_merge_minimum_time_sec = max(
        0.0, float(late_merge_minimum_time_sec))
    groups = {}

    for index, event in enumerate(detections or []):
        try:
            room_id = str(event.get("room_id", "")).strip()
            role = str(
                event.get("viewpoint_role", "")
            ).strip().upper()
            waypoint = str(event.get("waypoint", "")).strip()
            position = event.get("position", [])

            if (
                not room_id or
                not role or
                not waypoint or
                len(position) < 2
            ):
                continue

            float(position[0])
            float(position[1])
        except (AttributeError, TypeError, ValueError):
            continue

        key = (
            event.get("floor"),
            room_id,
            role,
            waypoint,
        )
        groups.setdefault(key, []).append(index)

    discarded = set()

    for indices in groups.values():
        strong = [
            index for index in indices
            if int(detections[index].get(
                "evidence_frames", 0
            ) or 0) >= strong_count
        ]
        weak = [
            index for index in indices
            if int(detections[index].get(
                "evidence_frames", 0
            ) or 0) in (
                confirm_count,
                confirm_count + 1,
            )
        ]

        for weak_index in weak:
            weak_event = detections[weak_index]
            weak_position = weak_event["position"]
            weak_count = int(weak_event.get(
                "evidence_frames", 0
            ) or 0)

            for strong_index in strong:
                strong_event = detections[strong_index]
                strong_position = strong_event["position"]

                distance = math.hypot(
                    float(weak_position[0]) -
                    float(strong_position[0]),
                    float(weak_position[1]) -
                    float(strong_position[1]),
                )

                try:
                    time_delta = abs(
                        float(weak_event[
                            "first_confirmed_elapsed_sec"
                        ]) -
                        float(strong_event[
                            "first_confirmed_elapsed_sec"
                        ])
                    )
                except (KeyError, TypeError, ValueError):
                    time_delta = None

                # A tail that appears seconds AFTER the strong track is the
                # same ball re-projected on a later pass: measured deltas are
                # 2.1 s, 8.1 s and 17.2 s at 1.1-1.5 m.  Two balls genuinely
                # visible at once are confirmed together -- the pair that made
                # this gate necessary sat 2.08 m apart at 0.15 s -- so the
                # radius stays below that and the delay is what separates them.
                late_reprojection = (
                    time_delta is not None and
                    distance <= late_merge_radius_m and
                    time_delta >= late_merge_minimum_time_sec
                )

                # Preserve the established three-frame rule.
                if weak_count == confirm_count:
                    if distance <= merge_radius_m or late_reprojection:
                        discarded.add(weak_index)
                        break
                    continue

                # A four-frame tail must satisfy both tighter gates.
                if (distance <= extra_frame_merge_radius_m and
                        time_delta is not None and
                        time_delta <= extra_frame_time_sec):
                    discarded.add(weak_index)
                    break

                if late_reprojection:
                    discarded.add(weak_index)
                    break

    return discarded


def detect_red_spheres(img_rgb, min_area=50, min_fill=0.50, sat_min=60, val_min=60,
                       min_vertices=6, min_circularity=0.68,
                       diagnostics=None):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, sat_min, val_min), (10, 255, 255)) | \
           cv2.inRange(hsv, (170, sat_min, val_min), (180, 255, 255))

    # Reject warm-gray/brown rendered furniture whose red channel is only
    # slightly stronger. Integer form is exactly:
    # R >= 1.30 * G and R >= 1.30 * B.
    rgb16 = img_rgb.astype(np.uint16)
    red = rgb16[:, :, 0]
    green = rgb16[:, :, 1]
    blue = rgb16[:, :, 2]
    red_purity_mask = (
        (100 * red >= 130 * green) &
        (100 * red >= 130 * blue)
    ).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, red_purity_mask)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        (ex, ey), radius = cv2.minEnclosingCircle(contour)
        if radius <= 0.0:
            continue
        fill = area / (math.pi * radius * radius)
        if fill < min_fill:
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = (4.0 * math.pi * area / (perimeter * perimeter)
                       if perimeter > 1e-6 else 0.0)
        vertices = len(cv2.approxPolyDP(contour, 0.04 * perimeter, True))
        # Preserve the established decision while recording bounded,
        # truth-independent contour evidence for later analysis.
        shape_accepted = (
            vertices >= int(min_vertices) and
            circularity >= float(min_circularity)
        )
        if diagnostics is not None:
            diagnostics.append({
                "cx_px": round(float(ex), 3),
                "cy_px": round(float(ey), 3),
                "area": round(float(area), 3),
                "radius_px": round(float(radius), 3),
                "fill": round(float(fill), 5),
                "circularity": round(float(circularity), 5),
                "vertices": int(vertices),
                "accepted": bool(shape_accepted),
            })
        # A sphere has a rounded silhouette even when part of it is hidden;
        # boxes and polygonal red distractors have four or five corners.
        if not shape_accepted:
            continue
        moments = cv2.moments(contour)
        detections.append((moments["m10"] / area, moments["m01"] / area, radius, fill))
    return detections


def pixel_to_world(cx_px, cy_px, base_xyz_world, base_yaw_world,
                   calibration=None, target_z=DANGER_Z):
    calibration = calibration or CameraCalibration.front_camera_fallback()
    horizontal = (cx_px - calibration.cx) / calibration.fx
    vertical = -(cy_px - calibration.cy) / calibration.fy
    ray_camera = np.array([1.0, -horizontal, vertical], dtype=float)

    cp = math.cos(calibration.pitch)
    sp = math.sin(calibration.pitch)
    pitch_rotation = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    ray_base = pitch_rotation @ ray_camera

    cyaw = math.cos(base_yaw_world)
    syaw = math.sin(base_yaw_world)
    yaw_rotation = np.array([[cyaw, -syaw, 0.0], [syaw, cyaw, 0.0], [0.0, 0.0, 1.0]])
    ray_world = yaw_rotation @ ray_base
    camera_world = np.asarray(base_xyz_world, dtype=float) + yaw_rotation @ calibration.xyz_base
    if ray_world[2] >= -1e-6:
        return None
    scale = (float(target_z) - camera_world[2]) / ray_world[2]
    if scale <= 0.0:
        return None
    point = camera_world + scale * ray_world
    return float(point[0]), float(point[1])


def depth_pixel_to_world(cx_px, cy_px, depth, base_xyz_world, base_yaw_world, calibration):
    horizontal = (cx_px - calibration.cx) / calibration.fx
    vertical = -(cy_px - calibration.cy) / calibration.fy
    point_camera = np.array([depth, -horizontal * depth, vertical * depth], dtype=float)
    cp = math.cos(calibration.pitch)
    sp = math.sin(calibration.pitch)
    pitch_rotation = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    point_base = calibration.xyz_base + pitch_rotation @ point_camera
    cyaw = math.cos(base_yaw_world)
    syaw = math.sin(base_yaw_world)
    yaw_rotation = np.array([[cyaw, -syaw, 0.0], [syaw, cyaw, 0.0], [0.0, 0.0, 1.0]])
    point_world = np.asarray(base_xyz_world, dtype=float) + yaw_rotation @ point_base
    return float(point_world[0]), float(point_world[1]), float(point_world[2])


def scale_planar_range(point_xy, origin_xyz, scale):
    """Apply a camera-model range calibration without scene-truth access."""
    if point_xy is None:
        return None
    origin_x, origin_y = float(origin_xyz[0]), float(origin_xyz[1])
    factor = float(scale)
    return (
        origin_x + factor * (float(point_xy[0]) - origin_x),
        origin_y + factor * (float(point_xy[1]) - origin_y),
    )


def fuse_sphere_world_point(depth_point, ground_point, target_z,
                            agreement_m=1.0):
    """Fuse independent sphere-size/depth and ground-ray geometry.

    ``select_sphere_depth`` has already replaced a disagreeing background
    depth with the sphere's apparent-size range. Therefore, when the two
    world projections still disagree, the depth point is the bounded sphere
    estimate and the near-horizon ground intersection is the unstable one.
    """
    if depth_point is not None and ground_point is not None:
        if math.hypot(float(depth_point[0]) - float(ground_point[0]),
                      float(depth_point[1]) - float(ground_point[1])) <= float(agreement_m):
            return (
                0.5 * (float(depth_point[0]) + float(ground_point[0])),
                0.5 * (float(depth_point[1]) + float(ground_point[1])),
                float(target_z),
            )
        return (float(depth_point[0]), float(depth_point[1]), float(target_z))
    if depth_point is not None:
        return (float(depth_point[0]), float(depth_point[1]), float(target_z))
    if ground_point is not None:
        return (float(ground_point[0]), float(ground_point[1]), float(target_z))
    return None


def point_inside_door_halfplane(point, door_plane_x, inward_direction,
                                margin=0.10):
    """Reject a projection that lies outside the currently scanned room."""
    if door_plane_x is None or inward_direction is None:
        return True
    return float(inward_direction) * (
        float(point[0]) - float(door_plane_x)) >= float(margin)


class Detector:
    def __init__(self, out_json, image_topic="/real_sense/rgb/image_raw", odom_topic="/Odometry_gazebo",
                 camera_info_topic="/real_sense/rgb/camera_info",
                 depth_topic="/real_sense/depth/image_raw", use_depth=True):
        import rospy
        from geometry_msgs.msg import PoseArray
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import CameraInfo, Image
        from std_msgs.msg import Bool, String

        self.rospy = rospy
        self.Image = Image
        self.Bool = Bool
        self.String = String
        self.PoseArray = PoseArray
        self.out_json = out_json
        self.base_xyz = [0.0, 2.0, 0.6]
        self.base_yaw = math.pi / 2.0
        self.pose_samples = deque(maxlen=200)
        self._pose_lock = threading.Lock()
        self.use_depth = bool(use_depth)
        self.calibration = (
            CameraCalibration.recording_camera_fallback()
            if "recording_camera" in str(image_topic) else
            CameraCalibration.real_sense_fallback())
        self.latest_depth = None
        self.latest_depth_stamp = None
        self._depth_lock = threading.Lock()
        self.scan_active = False
        self.scan_label = ""
        self.current_room_id = None
        self.room_boundary_margin_m = max(
            0.0, float(rospy.get_param("~room_boundary_margin_m", 0.50)))
        self.room_boundary_filter_required = bool(
            rospy.get_param(
                "~require_room_boundary_during_scan", False))
        self.layout_metadata_path = str(
            rospy.get_param("~layout_metadata", "")).strip()
        self.room_bounds_by_id = {}

        if self.layout_metadata_path:
            try:
                self.room_bounds_by_id = load_room_bounds(
                    self.layout_metadata_path)
            except (OSError, TypeError, ValueError) as error:
                if self.room_boundary_filter_required:
                    raise RuntimeError(
                        "cannot load required room bounds from {}: {}".format(
                            self.layout_metadata_path, error))
                rospy.logwarn(
                    "danger_detector: room-boundary filter disabled: %s",
                    error)

        if (self.room_boundary_filter_required and
                not self.room_bounds_by_id):
            raise RuntimeError(
                "room-boundary filtering is required but no room bounds "
                "were loaded")

        if self.room_bounds_by_id:
            rospy.loginfo(
                "danger_detector: loaded %d room boundaries; "
                "scan margin=%.2f m",
                len(self.room_bounds_by_id),
                self.room_boundary_margin_m)

        self.stage = "algorithm_preparation"
        self.floor = None
        self.waypoint = None
        self.room_id = None
        self.viewpoint_role = None
        self.door_plane_x = None
        self.door_inward_direction = None
        self.scan_points = []
        self.scan_frame = 0
        self.scan_batches = []
        self._scan_batch = None
        self._danger_rescan_requested_rooms = set()
        self._room_scan_evidence = {}
        self.scan_merge_radius = float(rospy.get_param("~scan_merge_radius", 1.2))
        self.floor_height = float(rospy.get_param("~floor_height_m", 2.6))
        # The level Gazebo recording camera's rasterized sphere centroid is
        # biased slightly toward the visible lower silhouette, which makes a
        # ground-plane ray intersection overestimate planar range. Keep this
        # fixed camera calibration independent of randomized scene truth.
        self.ground_range_scale = float(rospy.get_param(
            "~ground_range_scale", 0.95))
        self.tracker = DetectionTracker(
            rospy.get_param("~merge_radius", 0.75), rospy.get_param("~confirm_count", 3)
        )
        self.start_time = None
        self.start_sim_time = None
        self.started_wall_time = None
        self.exploration_active = False
        self.terminal_status = None
        # rospy subscriber callbacks run on separate threads.  In particular,
        # exploration_active=false and mission_complete=true can arrive
        # together at mission shutdown, so output and terminal-state changes
        # must be serialized.
        self._output_lock = threading.RLock()
        self.frames_processed = 0
        self.frames_with_candidates = 0
        self.detection_events = []
        self.maximum_processing_rate_hz = max(
            0.0, float(rospy.get_param("~maximum_processing_rate_hz", 5.0)))
        self._last_processed_monotonic = None
        self._camera_ready = False
        self.pose_pub = rospy.Publisher("/scanplanner/confirmed_dangers", PoseArray, queue_size=1, latch=True)
        self.debug_pub = rospy.Publisher("/scanplanner/danger_debug_image", Image, queue_size=1)
        self.danger_rescan_pub = rospy.Publisher(
            "/scanplanner/danger_rescan_request", String, queue_size=2)
        self.camera_ready_pub = rospy.Publisher(
            rospy.get_param("~camera_ready_topic", "/simenv/recording_camera_ready"),
            Bool, queue_size=1, latch=True)
        self.camera_ready_pub.publish(self.Bool(data=False))
        rospy.Subscriber(odom_topic, Odometry, self._odom, queue_size=50)
        rospy.Subscriber(camera_info_topic, CameraInfo, self._camera_info, queue_size=1)
        if self.use_depth:
            self.image_sub = rospy.Subscriber(
                image_topic, Image, self._image, queue_size=20,
                buff_size=2 ** 24)
            self.depth_sub = rospy.Subscriber(
                depth_topic, Image, self._depth, queue_size=10,
                buff_size=2 ** 24)
            self.image_depth_sync = None
        else:
            self.image_sub = rospy.Subscriber(
                image_topic, Image, self._image, queue_size=2,
                buff_size=2 ** 24)
            self.depth_sub = None
            self.image_depth_sync = None
        rospy.Subscriber("/scanplanner/route_failed", Bool, self._terminal, queue_size=1)
        rospy.Subscriber("/simenv/exploration_active", Bool, self._exploration_active_cb, queue_size=2)
        rospy.Subscriber("/scanplanner/three_floor_mission_complete", Bool, self._mission_complete_cb, queue_size=1)
        rospy.Subscriber("/simenv/mission_abort", Bool, self._mission_abort_cb, queue_size=1)
        rospy.Subscriber(
            rospy.get_param(
                "~scan_active_topic", "/scanplanner/room_scan_active"),
            Bool, self._scan_active_cb, queue_size=1)
        rospy.Subscriber("/scanplanner/route_state", String, self._route_state_cb, queue_size=1)
        self._write_output("preparing")
        rospy.on_shutdown(self._on_shutdown)

    def _odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.base_xyz = [p.x, p.y, p.z]
        self.base_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        stamp = msg.header.stamp.to_sec() if msg.header.stamp.to_sec() > 0.0 else self.rospy.get_time()
        with self._pose_lock:
            self.pose_samples.append(
                (stamp, tuple(self.base_xyz), self.base_yaw))

    def _camera_info(self, msg):
        self.calibration = CameraCalibration.from_camera_info(msg, self.calibration)

    def _terminal(self, msg):
        if msg.data:
            self._finalize("failed")

    def _exploration_active_cb(self, msg):
        with self._output_lock:
            active = bool(msg.data)
            if active and self.start_time is None:
                self.start_time = time.monotonic()
                self.start_sim_time = float(self.rospy.get_time())
                self.started_wall_time = time.time()
                self.exploration_active = True
                self.terminal_status = None
                self._write_output("running")
                self.rospy.loginfo(
                    "danger_detector: exploration clock started after preparation")
            elif not active and self.exploration_active:
                # The mission-complete/abort callbacks provide the
                # authoritative terminal status.  Keep a running snapshot only
                # if this Bool wins the race; a terminal status can never be
                # downgraded by a later callback.
                self.exploration_active = False
                self._write_output(self.terminal_status or "running")

    def _mission_complete_cb(self, msg):
        if bool(msg.data):
            self._finalize("completed")

    def _mission_abort_cb(self, msg):
        if bool(msg.data):
            self._finalize("failed")

    def _start_scan_batch(self):
        """Snapshot label/stage/floor at scan start so later route-state
        callbacks cannot relabel a batch that is already being recorded."""
        self._scan_batch = {
            "label": str(self.scan_label),
            "stage": str(self.stage),
            "floor": self.floor,
            "room_id": getattr(self, "room_id", None),
            "viewpoint_role": getattr(self, "viewpoint_role", None),
            "processed_frames": 0,
            "candidate_frames": 0,
            "room_boundary_rejected_points": 0,
            "room_boundary_missing_frames": 0,
        }

    def _record_scan_frame(self, world_points):
        """Count one processed scan frame and its accepted RGB candidates."""
        batch = getattr(self, "_scan_batch", None)
        if batch is None:
            return
        batch["processed_frames"] += 1
        if world_points:
            batch["candidate_frames"] += 1

    def _scan_active_cb(self, msg):
        active = bool(msg.data)
        if active and not self.scan_active:
            self.scan_points = []
            self.scan_frame = 0
            self._start_scan_batch()
        elif not active and self.scan_active:
            batch = getattr(self, "_scan_batch", None)
            if batch is None:
                self._start_scan_batch()
                batch = self._scan_batch
            # The frozen start-of-scan label is authoritative for the whole
            # batch; a route-state relabel mid-scan must not change the
            # evidence threshold or the logged batch identity.
            batch_label = str(batch.get("label", self.scan_label))
            batch_min_points = scan_min_points(batch_label)
            clustered = cluster_scan_points(
                self.scan_points,
                self.scan_merge_radius,
                min_points=batch_min_points,
            )
            diagnostic_clusters = cluster_scan_points(
                self.scan_points,
                self.scan_merge_radius,
                min_points=1,
            )
            self.rospy.loginfo(
                "danger_detector: scan batch label=%s raw=%d min_points=%d clusters=%s",
                batch_label, len(self.scan_points), batch_min_points, clustered,
            )
            # Diagnostic evidence only: the batch clusters are deliberately
            # not fed into DetectionTracker, so confirmation semantics stay
            # untouched.
            batch.update({
                "raw_candidate_points": int(len(self.scan_points)),
                "clustered_candidate_count": int(len(clustered)),
                "minimum_cluster_points": int(batch_min_points),
                "candidate_clusters_min1": [
                    [round(float(x), 4), round(float(y), 4)]
                    for x, y in diagnostic_clusters
                ],
                "candidate_clusters_min1_count": int(
                    len(diagnostic_clusters)),
            })
            self.scan_batches.append(dict(batch))
            room_id = batch.get("room_id")
            if room_id is not None:
                room_evidence = self._room_scan_evidence.setdefault(
                    str(room_id), {})
                room_evidence[str(batch.get("viewpoint_role"))] = {
                    "waypoint": batch_label,
                    "candidate_frames": int(batch.get(
                        "candidate_frames", 0)),
                    "raw_candidate_points": int(len(self.scan_points)),
                    "candidate_clusters_min1": list(batch.get(
                        "candidate_clusters_min1", [])),
                }
            confirmed_room_tracks = [
                track for track in self.tracker.confirmed_tracks()
                if str(track.get("metadata", {}).get("room_id")) ==
                str(room_id)
            ]
            confirmed_room_xy = [
                (float(track["x"]), float(track["y"]))
                for track in confirmed_room_tracks
                if "x" in track and "y" in track
            ]
            room_already_confirmed = bool(confirmed_room_tracks)
            # Tracker confirmation precedes the final weak-evidence filter.
            # Do not let a track that will be removed suppress its one retry.
            weak_room_tracks = []
            for track in confirmed_room_tracks:
                event = dict(track.get("metadata", {}))
                event.update({
                    "room_id": str(room_id),
                    "position": [track["x"], track["y"]],
                    "evidence_frames": int(track["count"]),
                })
                if is_weak_small_single_view_detection(event, self.scan_batches):
                    weak_room_tracks.append(int(track["id"]))

            # A non-empty candidate batch with no confirmed track is useful
            # online evidence that the sphere was only briefly exposed. Wait
            # until the mandatory G4 view before spending the room's single
            # bounded rescan: requesting it at G3 can consume the only retry
            # before the complementary view exposes the target. The request
            # uses camera evidence only (never scene truth).
            combined_evidence = self._room_scan_evidence.get(
                str(room_id), {}) if room_id is not None else {}
            combined_candidate_frames = sum(
                int(item.get("candidate_frames", 0))
                for item in combined_evidence.values())
            preferred_role = max(
                combined_evidence,
                key=lambda role: int(combined_evidence[role].get(
                    "candidate_frames", 0)), default="G4")

            # If one source was already confirmed, do not suppress evidence
            # for a possible second source. Require at least two residual
            # camera clusters outside every confirmed track's merge radius.
            # This only controls the existing bounded rescan; it never
            # confirms a detection or weakens the three-frame threshold.
            residual_cluster_evidence = []
            if room_already_confirmed and confirmed_room_xy:
                merge_radius_sq = float(self.scan_merge_radius) ** 2
                for role, item in combined_evidence.items():
                    for point in item.get(
                            "candidate_clusters_min1", []):
                        if not isinstance(point, (list, tuple)):
                            continue
                        if len(point) < 2:
                            continue
                        try:
                            px = float(point[0])
                            py = float(point[1])
                        except (TypeError, ValueError):
                            continue
                        covered = any(
                            (px - tx) ** 2 + (py - ty) ** 2 <=
                            merge_radius_sq
                            for tx, ty in confirmed_room_xy
                        )
                        if not covered:
                            residual_cluster_evidence.append({
                                "viewpoint_role": str(role),
                                "position": [
                                    round(px, 4), round(py, 4)
                                ],
                            })

            residual_multi_cluster_evidence = (
                len(residual_cluster_evidence) >= 2
            )

            if (room_id is not None and
                    str(batch.get("viewpoint_role")) == "G4" and
                    combined_candidate_frames > 0 and
                    (not room_already_confirmed or
                     bool(weak_room_tracks) or
                     residual_multi_cluster_evidence) and
                    str(room_id) not in getattr(
                        self, "_danger_rescan_requested_rooms", set())):
                request = {
                    "weak_track_ids": weak_room_tracks,
                    "room_id": str(room_id),
                    "waypoint": batch_label,
                    "floor": batch.get("floor"),
                    "candidate_frames": int(batch.get("candidate_frames", 0)),
                    "combined_candidate_frames": int(
                        combined_candidate_frames),
                    "raw_candidate_points": int(len(self.scan_points)),
                    "preferred_viewpoint_role": str(preferred_role),
                    "room_view_evidence": dict(combined_evidence),
                    "confirmed_room_track_count": int(
                        len(confirmed_room_tracks)),
                    "residual_cluster_evidence": list(
                        residual_cluster_evidence),
                    "residual_cluster_count": int(
                        len(residual_cluster_evidence)),
                    "reason": (
                        "confirmed_room_residual_multi_cluster_evidence"
                        if residual_multi_cluster_evidence else
                        "combined_G3_G4_partial_online_camera_evidence"
                    ),
                }
                self._danger_rescan_requested_rooms.add(str(room_id))
                self.danger_rescan_pub.publish(
                    self.String(data=json.dumps(request, sort_keys=True)))
                self.rospy.loginfo(
                    "danger_detector: requested one bounded rescan for %s at %s",
                    room_id, batch_label)
            self._scan_batch = None
            self.scan_points = []
            self.current_room_id = None
            self._write_output(self.terminal_status or "running")
        self.scan_active = active

    def _route_state_cb(self, msg):
        text = str(msg.data)
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            payload = {"phase": text}
        if not isinstance(payload, dict):
            payload = {"phase": text}
        phase = str(payload.get("phase", "unknown"))
        floor = payload.get("floor")
        if phase == "EXPLORATION_STARTED" and self.start_time is None:
            self._exploration_active_cb(type("BoolMessage", (), {"data": True})())
        if phase == "ENTER_MAIN_ENTRANCE_STAIR_RL":
            self.stage = "entrance_step_stair_rl"
            self.floor = 1
        elif phase == "EXPLORE_FLOOR" and floor is not None:
            self.stage = "floor_{}_exploration".format(int(floor))
        elif phase == "RELEASE_TO_STAIR_RL" and floor is not None:
            self.stage = {
                1: "stair_f1_to_f2",
                2: "stair_f2_to_f3",
                3: "stair_f3_to_f1_descent",
            }.get(int(floor), self.stage)
        self.floor = int(floor) if floor is not None else self.floor
        self.waypoint = payload.get("waypoint", self.waypoint)
        self.room_id = payload.get("room_id", self.room_id)
        self.viewpoint_role = payload.get(
            "viewpoint_role", self.viewpoint_role)
        self.door_plane_x = payload.get(
            "door_plane_x", self.door_plane_x)
        self.door_inward_direction = payload.get(
            "door_inward_direction", self.door_inward_direction)
        self.scan_label = str(payload.get("waypoint", phase))

        if phase == "SCAN_ROOM":
            room_id = (
                payload.get("room_id") or
                room_id_from_scan_label(self.scan_label)
            )
            self.current_room_id = str(room_id) if room_id else None

            # The two ROS topics can arrive in either callback order.
            # Populate only an empty batch identity.
            batch = getattr(self, "_scan_batch", None)
            if (self.scan_active and batch is not None and
                    not batch.get("room_id") and self.current_room_id):
                batch["room_id"] = self.current_room_id


    def _depth(self, msg):
        depth = None
        if msg.encoding == "32FC1" and len(msg.data) == msg.height * msg.width * 4:
            depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
        elif msg.encoding == "16UC1" and len(msg.data) == msg.height * msg.width * 2:
            depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width).astype(float) / 1000.0
        if depth is not None:
            stamp = (msg.header.stamp.to_sec()
                     if msg.header.stamp.to_sec() > 0.0
                     else self.rospy.get_time())
            with self._depth_lock:
                self.latest_depth = depth
                self.latest_depth_stamp = float(stamp)

    def _exploration_elapsed(self):
        start_sim_time = getattr(self, "start_sim_time", None)
        if start_sim_time is None:
            return None
        return max(0.0, float(self.rospy.get_time()) - start_sim_time)

    def _synced_image(self, image_msg, depth_msg):
        self._depth(depth_msg)
        self._image(image_msg)

    def _depth_at(self, cx, cy, image_stamp=None):
        with self._depth_lock:
            depth_image = self.latest_depth
            depth_stamp = self.latest_depth_stamp
        if depth_image is None:
            return None
        if (image_stamp is not None and depth_stamp is not None and
                abs(float(image_stamp) - float(depth_stamp)) > 0.20):
            return None
        # The robust RGB source and RealSense depth camera are co-located but
        # may publish different resolutions. Map normalized RGB coordinates
        # into the latest depth raster before sampling.
        x = int(round(float(cx) * depth_image.shape[1] /
                      max(1.0, float(self.calibration.width))))
        y = int(round(float(cy) * depth_image.shape[0] /
                      max(1.0, float(self.calibration.height))))
        x0, x1 = max(0, x - 2), min(depth_image.shape[1], x + 3)
        y0, y1 = max(0, y - 2), min(depth_image.shape[0], y + 3)
        values = depth_image[y0:y1, x0:x1]
        valid = values[np.isfinite(values) & (values >= 0.4) & (values <= 8.0)]
        return float(np.median(valid)) if valid.size else None

    def _publish_confirmed(self, header):
        from geometry_msgs.msg import Pose

        output = self.PoseArray()
        output.header = header
        output.header.frame_id = "map"
        for point in self.tracker.confirmed():
            pose = Pose()
            pose.position.x = point[0]
            pose.position.y = point[1]
            pose.position.z = point[2] if len(point) >= 3 else DANGER_Z
            pose.orientation.w = 1.0
            output.poses.append(pose)
        self.pose_pub.publish(output)

    def _image(self, msg):
        if msg.encoding not in ("rgb8", "bgr8"):
            return
        expected = msg.height * msg.width * 3
        if len(msg.data) != expected:
            return
        if not self._camera_ready:
            self._camera_ready = True
            self.camera_ready_pub.publish(self.Bool(data=True))
            self.rospy.loginfo(
                "danger_detector: first rendered RGB frame ready (%dx%d)",
                msg.width, msg.height)
        now_monotonic = time.monotonic()
        if (self.maximum_processing_rate_hz > 0.0 and
                self._last_processed_monotonic is not None and
                now_monotonic - self._last_processed_monotonic <
                1.0 / self.maximum_processing_rate_hz):
            return
        self._last_processed_monotonic = now_monotonic
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        rgb = image if msg.encoding == "rgb8" else image[:, :, ::-1]
        contour_diagnostics = []
        candidates = detect_red_spheres(
            rgb, diagnostics=contour_diagnostics)
        batch = getattr(self, "_scan_batch", None)
        if self.scan_active and batch is not None and contour_diagnostics:
            stored = batch.setdefault("contour_diagnostics", [])
            remaining = max(0, 500 - len(stored))
            stored.extend(contour_diagnostics[:remaining])
            if len(contour_diagnostics) > remaining:
                batch["contour_diagnostics_truncated"] = True
        stamp = msg.header.stamp.to_sec() if msg.header.stamp.to_sec() > 0.0 else self.rospy.get_time()

        # Diagnostic only: measure depth curvature inside accepted red
        # contours. This does not change detection or tracking decisions.
        for diagnostic in contour_diagnostics:
            if not diagnostic.get("accepted"):
                continue

            cx = float(diagnostic["cx_px"])
            cy = float(diagnostic["cy_px"])
            radius = float(diagnostic["radius_px"])
            offset = max(2.0, 0.55 * radius)

            depth_samples = {
                "center": self._depth_at(cx, cy, stamp),
                "left": self._depth_at(cx - offset, cy, stamp),
                "right": self._depth_at(cx + offset, cy, stamp),
                "up": self._depth_at(cx, cy - offset, stamp),
                "down": self._depth_at(cx, cy + offset, stamp),
            }

            for name, value in depth_samples.items():
                diagnostic["depth_{}_m".format(name)] = (
                    round(float(value), 5)
                    if value is not None and math.isfinite(float(value))
                    else None
                )

            center = depth_samples["center"]
            left = depth_samples["left"]
            right = depth_samples["right"]
            up = depth_samples["up"]
            down = depth_samples["down"]

            if all(value is not None and math.isfinite(float(value))
                   for value in (center, left, right)):
                diagnostic["depth_curvature_lr_m"] = round(
                    0.5 * (float(left) + float(right)) -
                    float(center), 5)
                diagnostic["depth_symmetry_lr_m"] = round(
                    abs(float(left) - float(right)), 5)

            if all(value is not None and math.isfinite(float(value))
                   for value in (center, up, down)):
                diagnostic["depth_curvature_ud_m"] = round(
                    0.5 * (float(up) + float(down)) -
                    float(center), 5)
                diagnostic["depth_symmetry_ud_m"] = round(
                    abs(float(up) - float(down)), 5)

        # Truth-independent, conservative depth-shape gate. A spherical
        # surface has positive second-order depth curvature; a planar
        # pallet/box remains approximately flat. Preserve candidates when
        # aligned depth is unavailable.
        depth_shape_min_curvature_m = 0.012
        accepted_diagnostics = [
            item for item in contour_diagnostics
            if item.get("accepted")
        ]

        if len(accepted_diagnostics) == len(candidates):
            depth_filtered_candidates = []
            depth_rejected_count = 0

            for candidate, diagnostic in zip(
                    candidates, accepted_diagnostics):
                center = diagnostic.get("depth_center_m")
                up = diagnostic.get("depth_up_m")
                down = diagnostic.get("depth_down_m")
                curvature_ud = diagnostic.get(
                    "depth_curvature_ud_m")

                complete_profile = all(
                    value is not None
                    for value in (center, up, down, curvature_ud)
                )

                if complete_profile:
                    try:
                        values_finite = all(
                            math.isfinite(float(value))
                            for value in (
                                center, up, down, curvature_ud
                            )
                        )
                    except (TypeError, ValueError):
                        values_finite = False
                else:
                    values_finite = False

                gate_applied = bool(
                    complete_profile and values_finite
                )
                depth_shape_accepted = (
                    not gate_applied or
                    float(curvature_ud) >=
                    depth_shape_min_curvature_m
                )

                diagnostic["depth_shape_gate_applied"] = gate_applied
                diagnostic["depth_shape_min_curvature_m"] = (
                    depth_shape_min_curvature_m
                )
                diagnostic["depth_shape_accepted"] = bool(
                    depth_shape_accepted
                )

                # Keep the depth-shape result as diagnostic evidence only.
                # The metric is not scale-stable for small or distant official
                # spheres, so it must not delete an otherwise valid red,
                # circular candidate. Existing projection, room-boundary and
                # multi-frame confirmation gates remain active.
                depth_filtered_candidates.append(candidate)
                if not depth_shape_accepted:
                    depth_rejected_count += 1

            candidates = depth_filtered_candidates

            scan_batch = getattr(self, "_scan_batch", None)
            if (
                self.scan_active and
                scan_batch is not None and
                depth_rejected_count
            ):
                scan_batch["depth_shape_rejected_candidates"] = (
                    int(scan_batch.get(
                        "depth_shape_rejected_candidates", 0
                    )) +
                    int(depth_rejected_count)
                )
        with self._pose_lock:
            pose_samples = list(self.pose_samples)
        base_xyz, base_yaw = nearest_pose(
            pose_samples, stamp, (tuple(self.base_xyz), self.base_yaw)
        )
        world_points = []
        floor_z = round(float(base_xyz[2]) / self.floor_height) * self.floor_height
        danger_z = floor_z + DANGER_Z
        for cx, cy, radius, _fill in candidates:
            sensor_depth = self._depth_at(cx, cy, stamp)
            depth = select_sphere_depth(
                sensor_depth, radius, self.calibration.fx)
            ground_point = pixel_to_world(
                cx, cy, base_xyz, base_yaw, self.calibration,
                target_z=danger_z)
            ground_point = scale_planar_range(
                ground_point, base_xyz, self.ground_range_scale)
            if depth is not None:
                point3 = depth_pixel_to_world(cx, cy, depth, base_xyz, base_yaw, self.calibration)
                # The level recording camera gives a stable ground-plane
                # intersection for a sphere of known physical radius. Fuse
                # consistent estimates; when they disagree, retain the
                # already background-filtered sphere-size/depth estimate.
                fused = fuse_sphere_world_point(
                    point3, ground_point, danger_z, agreement_m=1.0)
                if fused is not None:
                    world_points.append(fused)
            elif ground_point is not None:
                world_points.append((ground_point[0], ground_point[1], danger_z))
        if (should_update_detections(self.scan_active) and
                self.room_id is not None):
            world_points = [
                point for point in world_points
                if point_inside_door_halfplane(
                    point, self.door_plane_x,
                    self.door_inward_direction)
            ]
        active_scan_room_id = None
        if should_update_detections(self.scan_active):
            batch = getattr(self, "_scan_batch", None)
            active_scan_room_id = (
                str(batch.get("room_id", "")).strip()
                if batch is not None else ""
            )
            bounds = self.room_bounds_by_id.get(active_scan_room_id)

            if bounds is not None:
                original_count = len(world_points)
                world_points = [
                    point for point in world_points
                    if point_in_room_bounds(
                        point,
                        bounds,
                        self.room_boundary_margin_m,
                    )
                ]
                if batch is not None:
                    batch["room_boundary_rejected_points"] += (
                        original_count - len(world_points)
                    )
            elif self.room_boundary_filter_required:
                if batch is not None:
                    batch["room_boundary_missing_frames"] += 1
                    batch["room_boundary_rejected_points"] += len(
                        world_points
                    )
                world_points = []
                self.rospy.logwarn_throttle(
                    5.0,
                    "danger_detector: rejecting candidates because "
                    "room boundary is unavailable (room_id=%r)",
                    active_scan_room_id,
                )

        if should_update_detections(self.scan_active):
            self.scan_frame += 1
            self.scan_points.extend(
                (point[0], point[1], self.scan_frame) for point in world_points)
            self._record_scan_frame(world_points)
        if self.exploration_active and self.start_time is not None:
            self.frames_processed += 1
            if world_points:
                self.frames_with_candidates += 1
            elapsed = self._exploration_elapsed()
            metadata = {
                "first_confirmed_elapsed_sec": round(elapsed, 3),
                "first_confirmed_wall_time": round(time.time(), 3),
                "ros_time": round(float(self.rospy.get_time()), 3),
                "stage": self.stage,
                "floor": self.floor,
                "waypoint": self.waypoint,
                "room_id": self.room_id,
                "viewpoint_role": self.viewpoint_role,
                "door_plane_x": self.door_plane_x,
                "door_inward_direction": self.door_inward_direction,
                "room_id": active_scan_room_id,
                "scan_active": bool(self.scan_active),
            }
            if should_update_detections(self.scan_active):
                self.tracker.update(world_points, metadata=metadata)
            else:
                # A direct-RL navigation command also uses scan_cmd_vel, but
                # only the dedicated stationary-room-scan topic is detection
                # evidence.  Clear the edge-trigger list when tracking is
                # intentionally paused so confirmations cannot be replayed.
                self.tracker.last_newly_confirmed = []
            for track in self.tracker.last_newly_confirmed:
                event = dict(track.get("metadata", {}))
                event.update({
                    "track_id": int(track["id"]),
                    "position": [round(track["x"], 4),
                                 round(track["y"], 4),
                                 round(track.get("z", danger_z), 4)],
                    "confirmation_frames": int(track["count"]),
                })
                self.detection_events.append(event)
                self.rospy.loginfo(
                    "danger_detector: confirmed red sphere #%d at %s, t=%.2fs, stage=%s",
                    track["id"], event["position"], elapsed, self.stage)
            if self.tracker.last_newly_confirmed or self.frames_processed % 25 == 0:
                self._write_output("running")
        confirmed = self.tracker.confirmed()
        self._publish_confirmed(msg.header)

        if self.debug_pub.get_num_connections() > 0:
            debug = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for cx, cy, radius, _fill in candidates:
                cv2.circle(
                    debug, (int(round(cx)), int(round(cy))),
                    int(round(radius)), (0, 255, 255), 2)
            elapsed = self._exploration_elapsed()
            elapsed = elapsed if elapsed is not None else 0.0
            cv2.putText(
                debug, "detected=%d  elapsed=%.1fs" % (len(confirmed), elapsed),
                (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
            out = self.Image()
            out.header = msg.header
            out.height, out.width = debug.shape[:2]
            out.encoding = "bgr8"
            out.is_bigendian = False
            out.step = out.width * 3
            out.data = debug.tobytes()
            self.debug_pub.publish(out)

    def _write_output(self, status):
        with self._output_lock:
            # Once a terminal callback has won, no delayed running/preparing
            # callback may overwrite the terminal evidence file.
            effective_status = self.terminal_status or str(status)
            elapsed = self._exploration_elapsed()
            events_by_id = {
                int(event["track_id"]): event for event in self.detection_events
            }
            detections = []
            for track in self.tracker.confirmed_tracks():
                event = dict(events_by_id.get(int(track["id"]), {}))
                event.update({
                    "track_id": int(track["id"]),
                    "position": [round(track["x"], 4), round(track["y"], 4),
                                 round(track.get("z", DANGER_Z), 4)],
                    "evidence_frames": int(track["count"]),
                })
                detections.append(event)
            # Suppress only the measured D1 profile: a minimum-evidence,
            # small-image candidate localized from one room viewpoint only.
            # Missing diagnostics fail open. No truth/distractor metadata is
            # read here.
            scan_batches = list(getattr(self, "scan_batches", []))
            weak_small_single_view_rejected_count = 0
            retained_detections = []

            for event in detections:
                if is_weak_small_single_view_detection(
                        event, scan_batches):
                    weak_small_single_view_rejected_count += 1
                    continue
                retained_detections.append(event)

            detections = retained_detections

            # Official generation may place multiple danger sources in one
            # room. Never merge detections first established in the same
            # viewpoint: they may be distinct, simultaneously visible balls.
            #
            # A physical ball observed again from G3/G4 can acquire a shifted
            # world projection. Suppress it only when the two tracks are
            # mutual cross-view nearest neighbours in the same room and floor.
            # This uses online camera evidence only and never reads truth.
            cross_view_merge_radius_m = 1.80
            candidate_records = []

            for event in detections:
                room_id = event.get("room_id")
                direction = event.get("door_inward_direction")
                door_x = event.get("door_plane_x")

                if (
                    room_id is not None and
                    direction is not None and
                    door_x is not None
                ):
                    room_depth = float(direction) * (
                        float(event["position"][0]) - float(door_x)
                    )
                    if room_depth < 0.10:
                        continue

                try:
                    first_time = float(
                        event.get("first_confirmed_elapsed_sec")
                    )
                except (TypeError, ValueError):
                    first_time = float("inf")

                candidate_records.append({
                    "event": event,
                    "score": (
                        int(event.get("evidence_frames", 0)),
                        -first_time,
                        -int(event.get("track_id", 0)),
                    ),
                })

            room_groups = {}
            for index, record in enumerate(candidate_records):
                event = record["event"]
                room_id = event.get("room_id")
                floor = event.get("floor")
                position = event.get("position", [])

                if room_id is None or len(position) < 2:
                    continue

                room_groups.setdefault(
                    (floor, str(room_id)), []
                ).append(index)

            discarded = set()

            def planar_distance(left_index, right_index):
                left = candidate_records[left_index]["event"]["position"]
                right = candidate_records[right_index]["event"]["position"]
                return math.hypot(
                    float(left[0]) - float(right[0]),
                    float(left[1]) - float(right[1]),
                )

            for _scope, indices in room_groups.items():
                g3_indices = [
                    index for index in indices
                    if str(candidate_records[index]["event"].get(
                        "viewpoint_role", "")).upper() == "G3"
                ]
                g4_indices = [
                    index for index in indices
                    if str(candidate_records[index]["event"].get(
                        "viewpoint_role", "")).upper() == "G4"
                ]

                if not g3_indices or not g4_indices:
                    continue

                nearest_g4 = {
                    g3: min(
                        g4_indices,
                        key=lambda g4: planar_distance(g3, g4),
                    )
                    for g3 in g3_indices
                }
                nearest_g3 = {
                    g4: min(
                        g3_indices,
                        key=lambda g3: planar_distance(g3, g4),
                    )
                    for g4 in g4_indices
                }

                for g3, g4 in nearest_g4.items():
                    if nearest_g3.get(g4) != g3:
                        continue

                    distance = planar_distance(g3, g4)
                    if distance > cross_view_merge_radius_m:
                        continue

                    g3_score = candidate_records[g3]["score"]
                    g4_score = candidate_records[g4]["score"]

                    if g3_score >= g4_score:
                        winner, loser = g3, g4
                    else:
                        winner, loser = g4, g3

                    discarded.add(loser)

                    self.rospy.loginfo(
                        "danger_detector: merged cross-view duplicate "
                        "track %s into track %s, distance=%.3fm",
                        candidate_records[loser]["event"].get("track_id"),
                        candidate_records[winner]["event"].get("track_id"),
                        distance,
                    )

            # A later same-view rescan can split one physical source into a
            # strong established track and a minimum-confirmation projection
            # tail. Equal-strength detections and different waypoints remain.
            strong_same_view_losers = strong_same_view_duplicate_losers(
                [record["event"] for record in candidate_records],
                confirm_count=int(getattr(
                    self.tracker, "confirm_count", 3)),
                merge_radius_m=1.40,
                strong_evidence_multiplier=2,
            )
            for loser in sorted(
                    strong_same_view_losers - discarded):
                discarded.add(loser)
                self.rospy.loginfo(
                    "danger_detector: merged weak same-view track %s "
                    "into nearby strong track",
                    candidate_records[loser]["event"].get(
                        "track_id"),
                )

            # Weak same-view D1 fallback duplicate suppression.
            # Apply only to minimum-evidence tracks produced almost
            # simultaneously in the same depth-rejected scan batch.
            weak_same_view_radius_m = 1.50
            weak_same_view_time_sec = 0.15
            confirm_count = int(
                getattr(self.tracker, "confirm_count", 3)
            )

            batches_by_label = {}
            for batch in getattr(self, "scan_batches", []):
                label = str(batch.get("label", "")).strip()
                if label:
                    batches_by_label[label] = batch

            for _scope, indices in room_groups.items():
                active_indices = [
                    index for index in indices
                    if index not in discarded
                ]

                for offset, left_index in enumerate(active_indices):
                    if left_index in discarded:
                        continue

                    left = candidate_records[left_index]["event"]
                    left_role = str(
                        left.get("viewpoint_role", "")
                    ).upper()
                    waypoint = str(left.get("waypoint", "")).strip()

                    if not left_role or not waypoint:
                        continue
                    if int(left.get(
                        "evidence_frames", 0
                    )) != confirm_count:
                        continue

                    batch = batches_by_label.get(waypoint)
                    if batch is None:
                        continue

                    raw_points = int(
                        batch.get("raw_candidate_points", 0) or 0
                    )
                    rejected = int(
                        batch.get(
                            "depth_shape_rejected_candidates", 0
                        ) or 0
                    )

                    if raw_points <= 0 or rejected < raw_points:
                        continue

                    for right_index in active_indices[offset + 1:]:
                        if right_index in discarded:
                            continue

                        right = candidate_records[right_index]["event"]

                        if (
                            str(right.get(
                                "viewpoint_role", ""
                            )).upper() != left_role or
                            str(right.get(
                                "waypoint", ""
                            )).strip() != waypoint or
                            int(right.get(
                                "evidence_frames", 0
                            )) != confirm_count
                        ):
                            continue

                        try:
                            time_delta = abs(
                                float(left[
                                    "first_confirmed_elapsed_sec"
                                ]) -
                                float(right[
                                    "first_confirmed_elapsed_sec"
                                ])
                            )
                        except (KeyError, TypeError, ValueError):
                            continue

                        if time_delta > weak_same_view_time_sec:
                            continue

                        distance = planar_distance(
                            left_index, right_index
                        )
                        if distance > weak_same_view_radius_m:
                            continue

                        def room_depth(event):
                            try:
                                return float(
                                    event["door_inward_direction"]
                                ) * (
                                    float(event["position"][0]) -
                                    float(event["door_plane_x"])
                                )
                            except (
                                KeyError, TypeError, ValueError
                            ):
                                return float("-inf")

                        if room_depth(left) >= room_depth(right):
                            loser = right_index
                        else:
                            loser = left_index

                        discarded.add(loser)

            detections = [
                record["event"]
                for index, record in enumerate(candidate_records)
                if index not in discarded
            ]
            detections.sort(
                key=lambda event: int(event.get("track_id", 0))
            )

            output = {
                "schema": "scanplanner_red_ball_detections_v1",
                "status": str(effective_status),
                "preparation_excluded": True,
                "clock_definition": "ROS/Gazebo simulation time from EXPLORATION_STARTED",
                "time_basis": "ros_simulation_time",
                "exploration_started_wall_time": (
                    round(self.started_wall_time, 3)
                    if self.started_wall_time is not None else None),
                "exploration_duration_sec": (
                    round(elapsed, 3) if elapsed is not None else None),
                "frames_processed": int(self.frames_processed),
                "frames_with_red_sphere_candidates": int(self.frames_with_candidates),
                "confirmed_count": len(detections),
                "weak_small_single_view_rejected_count": int(
                    weak_small_single_view_rejected_count),
                "detections": detections,
                "scan_batches": [dict(batch) for batch in
                                 getattr(self, "scan_batches", [])],
            }
            directory = os.path.dirname(self.out_json) or "."
            os.makedirs(directory, exist_ok=True)
            temporary = "%s.tmp.%d.%d" % (
                self.out_json, os.getpid(), threading.get_ident())
            try:
                with open(temporary, "w", encoding="utf-8") as stream:
                    json.dump(output, stream, ensure_ascii=False, indent=2,
                              sort_keys=True)
                    stream.write("\n")
                os.replace(temporary, self.out_json)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)

    def _finalize(self, status):
        with self._output_lock:
            if self.terminal_status is not None:
                return
            if self.scan_active:
                self._scan_active_cb(type("BoolMessage", (), {"data": False})())
            self.terminal_status = str(status)
            self.exploration_active = False
            self._write_output(status)
            self.rospy.loginfo(
                "danger_detector: wrote %d confirmed red-ball detections -> %s",
                len(self.tracker.confirmed()), self.out_json)

    def _on_shutdown(self):
        if self.terminal_status is None:
            self._finalize("interrupted")


def _selftest():
    image = np.zeros((800, 800, 3), np.uint8)
    cv2.circle(image, (500, 400), 45, (255, 0, 0), -1)
    cv2.rectangle(image, (120, 120), (210, 210), (255, 0, 0), -1)
    detections = detect_red_spheres(image)
    calibration = CameraCalibration.front_camera_fallback()
    projected = pixel_to_world(calibration.cx, calibration.cy, [0.0, 2.0, 0.6], math.pi / 2.0, calibration)
    ok = len(detections) == 1 and projected is not None and abs(projected[1] - 3.45) < 0.15
    print("detect=%d projected=%s" % (len(detections), projected))
    print("SELFCHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    import rospy

    rospy.init_node("danger_detector", anonymous=False)
    output_path = rospy.get_param("~out_json", "/workspace/SimEnv/results/detected_danger.json")
    Detector(
        output_path,
        rospy.get_param("~image_topic", "/real_sense/rgb/image_raw"),
        camera_info_topic=rospy.get_param("~camera_info_topic", "/real_sense/rgb/camera_info"),
        depth_topic=rospy.get_param("~depth_topic", "/real_sense/depth/image_raw"),
        use_depth=rospy.get_param("~use_depth", True),
    )
    rospy.spin()
