#!/usr/bin/env python3
"""Maintain online RGB-D viewing-sector coverage independently of LiDAR map data."""
import json
import math
import os
import time

import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import String


def yaw(q):
    return math.atan2(2.0 * (q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))


class VisualCoverageMapper:
    def __init__(self):
        self.fov = math.radians(float(rospy.get_param("~camera_fov_deg", 60.0)))
        self.required = int(rospy.get_param("~required_view_sectors", 5))
        self.context, self.pose, self.map_seen = {"enabled": False}, None, False
        self.bridge, self.depth_valid_fraction = CvBridge(), 0.0
        self.minimum_depth_valid_fraction = float(rospy.get_param(
            "~minimum_depth_valid_fraction", .20))
        self.minimum_inside_depth = float(rospy.get_param(
            "~minimum_inside_depth_m", .15))
        self.anchor_radius = float(rospy.get_param(
            "~visual_anchor_radius_m", 3.2))
        self.rooms = {}
        output = rospy.get_param("~output_dir", os.getcwd())
        self.path = os.path.join(output, "logs", "visual_coverage.jsonl"); os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.pub = rospy.Publisher("/simenv/visual_coverage", String, queue_size=10, latch=True)
        rospy.Subscriber(rospy.get_param("~odom_topic", "/Odometry"), Odometry, self.on_odom, queue_size=20)
        rospy.Subscriber("/simenv/room_detection_context", String, self.on_context, queue_size=5)
        rospy.Subscriber(rospy.get_param("~occupancy_topic", "/simenv/voxel_floor_projection"), OccupancyGrid, self.on_map, queue_size=1)
        rospy.Subscriber(rospy.get_param("~depth_topic", "/real_sense/depth/image_raw"), Image, self.on_depth, queue_size=2)
        rospy.Timer(rospy.Duration(.5), self.tick)

    def on_odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.pose = (p.x, p.y, yaw(q))

    def on_map(self, _): self.map_seen = True

    def on_depth(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32) * .001
            else:
                depth = depth.astype(np.float32)
            # A yaw sector counts only when RGB-D supplied actual range
            # measurements.  Direction-only bookkeeping had reported 100%
            # coverage even when furniture/invalid depth hid the scene.
            self.depth_valid_fraction = float(np.mean(
                np.isfinite(depth) & (depth > .08) & (depth < 6.5)))
        except CvBridgeError:
            return

    def on_context(self, m):
        try: self.context = json.loads(m.data)
        except (ValueError, TypeError): pass

    def write(self, data):
        try:
            with open(self.path, "a", encoding="utf-8") as f: f.write(json.dumps(data, sort_keys=True) + "\n")
        except OSError: pass

    def tick(self, _):
        if not self.context.get("enabled") or self.pose is None:
            return
        room = str(self.context.get("room_id") or "active_room")
        anchor = self.context.get("visual_anchor") or {}
        center = anchor.get("door_center")
        inward = anchor.get("interior_direction")
        valid_anchor = (isinstance(center, (list, tuple)) and len(center) >= 2 and
                        isinstance(inward, (list, tuple)) and len(inward) >= 2)
        inside_anchor = False
        anchor_distance = None
        inward_yaw = None
        if valid_anchor:
            try:
                dx, dy = self.pose[0] - float(center[0]), self.pose[1] - float(center[1])
                nx, ny = float(inward[0]), float(inward[1])
                inward_norm = math.hypot(nx, ny)
                anchor_distance = math.hypot(dx, dy)
                inside_anchor = (inward_norm > 1e-4 and
                                 (dx * nx + dy * ny) / inward_norm >= self.minimum_inside_depth and
                                 anchor_distance <= self.anchor_radius)
                inward_yaw = math.atan2(ny, nx)
            except (TypeError, ValueError):
                valid_anchor = False
        state = self.rooms.setdefault(room, {"sectors": set(), "poses": [],
                                              "entry_yaw": inward_yaw if inward_yaw is not None else self.pose[2]})
        # Count a sector only when the camera is on the interior side of the
        # active door and near its first safe viewing region.  Previously a
        # turn while leaving the room could falsely complete the sector map,
        # even though the relevant side wall was never observed from inside.
        reference_yaw = inward_yaw if inward_yaw is not None else state["entry_yaw"]
        relative_yaw = (self.pose[2] - reference_yaw + math.pi) % (2*math.pi) - math.pi
        sector = min(4, max(0, int((relative_yaw + math.pi/2) / (math.pi/5))))
        # A door normal is estimated from a sparse local map and can flip by
        # a few decimetres after the portal is crossed.  Requiring that
        # normal again while a confirmed room-view role is active made a
        # genuine 360-degree turn contribute only one sector (0.2 coverage),
        # thereby forcing a needless second cross-room goal.  The scheduler
        # state is already the authoritative online proof that the robot is
        # executing a room-internal G1/G3/G4 view; use it as the primary gate.
        # ENTRY/EXIT/RETURN remain excluded, so corridor turns cannot claim
        # visual room coverage.
        room_view_role = str(self.context.get("scheduler_state") or "")
        scheduler_inside_view = room_view_role in (
            "G1_CENTER", "G3_LEFT", "G4_RIGHT")
        usable_viewpoint = (scheduler_inside_view or
                            (inside_anchor if valid_anchor else True))
        if usable_viewpoint and self.depth_valid_fraction >= self.minimum_depth_valid_fraction:
            state["sectors"].add(sector)
        if (usable_viewpoint and (not state["poses"] or
                math.hypot(state["poses"][-1][0]-self.pose[0], state["poses"][-1][1]-self.pose[1]) > .25)):
            state["poses"].append(self.pose)
        visual = len(state["sectors"]) / 5.0
        sector_centers = [(-.5 * math.pi) + (index + .5) * (math.pi / 5.0)
                          for index in range(5)]
        missing = [index for index in range(5) if index not in state["sectors"]]
        if missing:
            desired = min((sector_centers[index] for index in missing),
                          key=lambda value: abs((value - relative_yaw + math.pi) %
                                                (2.0 * math.pi) - math.pi))
            delta = (desired - relative_yaw + math.pi) % (2.0 * math.pi) - math.pi
            suggested_direction = 1.0 if delta >= 0.0 else -1.0
            suggested_angle = min(math.pi / 2.0,
                                  max(math.pi / 6.0, abs(delta) + .5 * self.fov))
        else:
            suggested_direction, suggested_angle = 1.0, 0.0
        record = {"timestamp": time.time(), "room_id": room,
                  "lidar_coverage_available": self.map_seen,
                  "visual_coverage": round(visual, 3), "view_sectors": sorted(state["sectors"]),
                  "relative_camera_yaw_deg": round(math.degrees(relative_yaw), 1),
                  "depth_valid_fraction": round(self.depth_valid_fraction, 3),
                  "inside_visual_anchor": bool(inside_anchor),
                  "scheduler_inside_view": bool(scheduler_inside_view),
                  "visual_anchor_distance_m": (None if anchor_distance is None else
                                               round(anchor_distance, 3)),
                  "missing_view_sectors": missing,
                  "suggested_sweep_direction": suggested_direction,
                  "suggested_sweep_angle_rad": round(suggested_angle, 3),
                  "visual_sweep_needed": bool(self.map_seen and visual < self.required / 5.0),
                  "camera_pose": {"x": round(self.pose[0],3), "y": round(self.pose[1],3), "yaw": round(self.pose[2],3)},
                  "camera_fov_deg": round(math.degrees(self.fov), 1)}
        self.pub.publish(String(data=json.dumps(record, sort_keys=True))); self.write(record)


if __name__ == "__main__":
    rospy.init_node("visual_coverage_mapper")
    VisualCoverageMapper(); rospy.spin()
