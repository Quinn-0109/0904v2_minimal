#!/usr/bin/env python3
"""Radar-guided camera attention using a FUEL-inspired frontier hierarchy.

Livox Mid-360 maintains an incremental frontier information structure (FIS).
Hierarchical policy:
  far  (> mid_range): radar-only tracking, do not stop for RGB
  mid  : yaw-blend while driving (local refine without full stop)
  near : short stop-hold to clear occlusions / furniture shadows

TARE owns global exploration. This node only preempts /cmd_vel briefly.
"""

import math
import os
import sys
import threading

import rospy
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud, PointCloud2
from std_msgs.msg import Bool, String

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frontier_information_structure import (  # noqa: E402
    FrontierInformationStructure,
    angle_diff,
)


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class RadarCameraAttention:
    def __init__(self):
        self._lock = threading.Lock()
        self._pose = None
        self._planner_command = Twist()
        self._cloud_stamp = rospy.Time(0)
        self._anchor = None
        self._mode = "drive"
        self._target_yaw = 0.0
        self._hold_until = rospy.Time(0)
        self._hold_seconds = 0.4
        self._active_cluster_id = None
        self._blend_until = rospy.Time(0)
        self._stopped = False

        sector_count = int(rospy.get_param("~sector_count", 36))
        self._cell_size = float(rospy.get_param("~camera_cell_size", 3.0))
        self._spacing = float(rospy.get_param("~attention_spacing", 3.5))
        camera_range = float(rospy.get_param("~camera_range", 7.5))
        camera_fov = math.radians(float(rospy.get_param("~camera_fov_deg", 60.0)))
        self._max_yaw = float(rospy.get_param("~attention_yaw_rate", 0.8))
        self._min_score = float(rospy.get_param("~min_viewpoint_score", 0.12))
        self._blend_seconds = float(rospy.get_param("~yaw_blend_seconds", 1.2))
        self._blend_linear_scale = float(
            rospy.get_param("~yaw_blend_linear_scale", 0.55)
        )

        self._fis = FrontierInformationStructure(
            sector_count=sector_count,
            camera_range=camera_range,
            camera_fov=camera_fov,
            radar_max_range=float(rospy.get_param("~radar_max_range", 20.0)),
            near_range=float(rospy.get_param("~near_range", 4.0)),
            mid_range=float(rospy.get_param("~mid_range", 9.0)),
            merge_yaw=math.radians(float(rospy.get_param("~merge_yaw_deg", 25.0))),
            stale_seconds=float(rospy.get_param("~frontier_stale_sec", 8.0)),
            min_points=int(rospy.get_param("~min_sector_points", 3)),
        )

        self._own_cmd = bool(rospy.get_param("~own_cmd_vel", True))
        self._cmd_pub = None
        if self._own_cmd:
            self._cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=5)
        self._status_pub = rospy.Publisher(
            "/camera_attention/status", String, queue_size=5, latch=True
        )
        self._summary_pub = rospy.Publisher(
            "/camera_attention/frontier_summary", String, queue_size=5, latch=True
        )
        rospy.Subscriber(
            "/tare/cmd_vel_stamped", TwistStamped, self._on_command, queue_size=5
        )
        rospy.Subscriber("/state_estimation", Odometry, self._on_state, queue_size=20)
        rospy.Subscriber("/scan", PointCloud, self._on_radar, queue_size=2)
        rospy.Subscriber(
            "/registered_scan", PointCloud2, self._on_registered, queue_size=2
        )
        rospy.Subscriber(
            "/simenv/mission_complete", Bool, self._on_complete, queue_size=1
        )
        rospy.Timer(rospy.Duration(0.05), self._on_timer)
        rospy.Timer(rospy.Duration(2.0), self._on_summary)
        rospy.loginfo(
            "FIS camera attention: near<=%.1fm mid<=%.1fm camera=%.1fm sectors=%d own_cmd=%s",
            self._fis.near_range,
            self._fis.mid_range,
            self._fis.camera_range,
            self._fis.sector_count,
            self._own_cmd,
        )

    def _on_command(self, message):
        with self._lock:
            self._planner_command = message.twist

    def _on_state(self, message):
        position = message.pose.pose.position
        yaw = yaw_from_quaternion(message.pose.pose.orientation)
        with self._lock:
            self._pose = (position.x, position.y, yaw)
            self._fis.mark_camera_view(
                position.x, position.y, yaw, cell_size=self._cell_size
            )
            if self._anchor is None:
                self._anchor = (position.x, position.y)

    def _ingest_body_points(self, points_xy, stamp):
        bins = [[] for _ in range(self._fis.sector_count)]
        for x, y in points_xy:
            planar = math.hypot(x, y)
            if planar < 0.4 or planar > self._fis.radar_max_range:
                continue
            index = self._fis.heading_bin(math.atan2(y, x))
            bins[index].append(planar)
        sectors = []
        for values in bins:
            if not values:
                sectors.append((0, 0.0))
                continue
            values.sort()
            percentile = values[min(len(values) - 1, int(0.7 * len(values)))]
            sectors.append((len(values), percentile))

        stamp_sec = stamp.to_sec() if hasattr(stamp, "to_sec") else float(stamp)
        with self._lock:
            pose = self._pose
            if pose is None:
                self._cloud_stamp = stamp if hasattr(stamp, "secs") else rospy.Time.now()
                return
            x, y, yaw = pose
            self._fis.update_from_radar(x, y, yaw, sectors, stamp_sec)
            self._cloud_stamp = (
                stamp if hasattr(stamp, "secs") else rospy.Time.from_sec(stamp_sec)
            )

    def _on_radar(self, message):
        points = [(point.x, point.y) for point in message.points]
        self._ingest_body_points(points, message.header.stamp)

    def _on_registered(self, message):
        """Map-frame LIO cloud -> body-frame sectors for FIS."""
        with self._lock:
            pose = self._pose
        if pose is None:
            return
        rx, ry, ryaw = pose
        cos_y = math.cos(ryaw)
        sin_y = math.sin(ryaw)
        points = []
        for px, py, _pz in pc2.read_points(
            message, field_names=("x", "y", "z"), skip_nans=True
        ):
            dx = px - rx
            dy = py - ry
            # World -> body
            bx = cos_y * dx + sin_y * dy
            by = -sin_y * dx + cos_y * dy
            points.append((bx, by))
            if len(points) >= 8000:
                break
        if points:
            self._ingest_body_points(points, message.header.stamp)

    def _on_complete(self, message):
        if message.data:
            with self._lock:
                self._stopped = True

    def _on_summary(self, _event):
        with self._lock:
            summary = self._fis.summary()
        self._summary_pub.publish(
            String(
                data="near={near} mid={mid} far={far} covered={covered}".format(
                    **summary
                )
            )
        )

    def _publish_cmd(self, command):
        if self._cmd_pub is not None:
            self._cmd_pub.publish(command)

    def _on_timer(self, _event):
        command = Twist()
        with self._lock:
            if self._stopped:
                self._publish_cmd(command)
                return
            pose = self._pose
            if pose is None:
                self._publish_cmd(command)
                return
            x, y, yaw = pose
            now = rospy.Time.now()

            # Status-only mode: still score viewpoints for structure explorer /
            # logging, but do not publish /cmd_vel (cmd_vel_guard owns motion).
            if not self._own_cmd:
                travelled = (
                    0.0
                    if self._anchor is None
                    else math.hypot(x - self._anchor[0], y - self._anchor[1])
                )
                cloud_fresh = (now - self._cloud_stamp).to_sec() < 1.5
                if travelled >= self._spacing and cloud_fresh:
                    decision = self._fis.select_camera_viewpoint(
                        x, y, yaw, cell_size=self._cell_size
                    )
                    if decision is not None and decision.score >= self._min_score:
                        self._status_pub.publish(String(data=decision.reason))
                        self._fis.mark_camera_view(x, y, decision.yaw, cell_size=self._cell_size)
                    self._anchor = (x, y)
                return

            if self._mode == "turn":
                error = angle_diff(self._target_yaw, yaw)
                if abs(error) <= 0.10:
                    self._mode = "hold"
                    self._hold_until = now + rospy.Duration(self._hold_seconds)
                    self._fis.mark_camera_view(x, y, yaw, cell_size=self._cell_size)
                    self._status_pub.publish(String(data="camera_hold"))
                else:
                    command.angular.z = max(
                        -self._max_yaw, min(self._max_yaw, 1.8 * error)
                    )
                self._publish_cmd(command)
                return

            if self._mode == "hold":
                if now >= self._hold_until:
                    self._fis.note_camera_visit(yaw, self._active_cluster_id)
                    self._active_cluster_id = None
                    self._mode = "drive"
                    self._anchor = (x, y)
                    self._status_pub.publish(String(data="planner_drive"))
                self._publish_cmd(command)
                return

            if self._mode == "yaw_blend":
                error = angle_diff(self._target_yaw, yaw)
                command = Twist()
                command.linear.x = self._planner_command.linear.x * self._blend_linear_scale
                command.linear.y = self._planner_command.linear.y * self._blend_linear_scale
                command.angular.z = max(
                    -self._max_yaw, min(self._max_yaw, 1.6 * error)
                )
                aligned = abs(error) <= 0.18
                timed_out = now >= self._blend_until
                if aligned or timed_out:
                    self._fis.mark_camera_view(x, y, yaw, cell_size=self._cell_size)
                    self._fis.note_camera_visit(yaw, self._active_cluster_id)
                    self._active_cluster_id = None
                    self._mode = "drive"
                    self._anchor = (x, y)
                    self._status_pub.publish(String(data="planner_drive"))
                self._publish_cmd(command)
                return

            travelled = (
                0.0
                if self._anchor is None
                else math.hypot(x - self._anchor[0], y - self._anchor[1])
            )
            cloud_fresh = (now - self._cloud_stamp).to_sec() < 1.5
            if travelled >= self._spacing and cloud_fresh:
                decision = self._fis.select_camera_viewpoint(
                    x, y, yaw, cell_size=self._cell_size
                )
                if decision is not None and decision.score >= self._min_score:
                    self._target_yaw = decision.yaw
                    self._hold_seconds = decision.hold_seconds
                    self._active_cluster_id = decision.cluster_id
                    if decision.preempt_mode == "stop_hold":
                        self._mode = "turn"
                    else:
                        self._mode = "yaw_blend"
                        self._blend_until = now + rospy.Duration(
                            max(self._blend_seconds, decision.hold_seconds)
                        )
                    self._status_pub.publish(String(data=decision.reason))
                    self._publish_cmd(command)
                    return
                self._anchor = (x, y)

            command = self._planner_command
        self._publish_cmd(command)


if __name__ == "__main__":
    rospy.init_node("radar_camera_attention")
    RadarCameraAttention()
    rospy.spin()
