#!/usr/bin/env python3
"""Adapt SimEnv FAST-LIO pose/scan streams to the official CMU inputs.

This node contains no exploration policy.  It republishes synchronized
odometry and registered scans.  Production runs use CMU terrain_analysis for
terrain_map generation.  The old OccupancyGrid-to-PointXYZI conversion remains
available only as an explicit diagnostic fallback:

  intensity < terrain_free_threshold: traversable
  intensity >= terrain_free_threshold: obstacle
"""

import copy
import math
import threading

import numpy as np
import rospy
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


FIELDS_XYZI = [
    PointField("x", 0, PointField.FLOAT32, 1),
    PointField("y", 4, PointField.FLOAT32, 1),
    PointField("z", 8, PointField.FLOAT32, 1),
    PointField("intensity", 12, PointField.FLOAT32, 1),
]


class TareInterface:
    def __init__(self):
        self._lock = threading.RLock()
        self._grid = None
        self._odom = None
        self._last_grid_stamp = None

        self._map_topic = rospy.get_param(
            "~map_topic", "/simenv/voxel_floor_projection")
        self._odom_topic = rospy.get_param("~odom_topic", "/Odometry")
        self._scan_topic = rospy.get_param(
            "~registered_scan_topic", "/cloud_registered")
        # Official TARE hard-codes its world frame to "map".  SimEnv's
        # FAST-LIO initial frame is the same metric world origin, so this is a
        # frame alias (header normalization), not a hidden pose transform.
        self._world_frame = rospy.get_param("~world_frame", "map")
        self._local_radius = float(rospy.get_param(
            "~local_terrain_radius", 15.0))
        self._free_stride = max(1, int(rospy.get_param(
            "~free_cell_stride", 2)))
        self._occupied_stride = max(1, int(rospy.get_param(
            "~occupied_cell_stride", 1)))
        self._occupied_threshold = int(rospy.get_param(
            "~occupied_threshold", 50))
        self._terrain_z_offset = float(rospy.get_param(
            "~terrain_z_offset", 0.0))
        self._publish_period = float(rospy.get_param(
            "~terrain_publish_period", 0.5))
        self._synthetic_terrain = bool(rospy.get_param(
            "~synthetic_terrain_enabled", False))
        # TARE builds its own rolling occupancy grid by ray tracing the
        # registered scan.  kSensorRange only limits viewpoint scoring; it
        # does not crop that input cloud.  Feeding the full 30 m FAST-LIO
        # cloud therefore let rays through a doorway mark most of a room from
        # the corridor.  Keep a local scan for exploration decisions so an
        # unknown room remains a frontier until the robot approaches/enters.
        self._scan_max_range = max(0.0, float(rospy.get_param(
            "~scan_max_range", 7.5)))
        self._scan_input_count = 0
        self._scan_output_count = 0

        self._state_pub = rospy.Publisher(
            "/tare/state_estimation_at_scan", Odometry, queue_size=5)
        self._scan_pub = rospy.Publisher(
            "/tare/registered_scan", PointCloud2, queue_size=2)
        self._terrain_pub = None
        self._terrain_ext_pub = None
        if self._synthetic_terrain:
            rospy.logwarn(
                "Synthetic OccupancyGrid terrain is enabled for diagnostics; "
                "CMU terrain_analysis should be used for exploration")
            self._terrain_pub = rospy.Publisher(
                "/tare/terrain_map", PointCloud2, queue_size=2)
            self._terrain_ext_pub = rospy.Publisher(
                "/tare/terrain_map_ext", PointCloud2, queue_size=2)
            rospy.Subscriber(
                self._map_topic, OccupancyGrid, self._on_grid, queue_size=1)
        rospy.Subscriber(
            self._odom_topic, Odometry, self._on_odom, queue_size=20)
        rospy.Subscriber(
            self._scan_topic, PointCloud2, self._on_scan, queue_size=2)
        self._terrain_timer = None
        if self._synthetic_terrain:
            self._terrain_timer = rospy.Timer(
                rospy.Duration(self._publish_period),
                self._publish_terrain)
        rospy.loginfo(
            "TARE pose/scan interface ready: odom=%s scan=%s frame=%s "
            "synthetic_terrain=%s scan_max_range=%.2fm",
            self._odom_topic, self._scan_topic, self._world_frame,
            self._synthetic_terrain, self._scan_max_range)

    def _on_grid(self, message):
        with self._lock:
            self._grid = message

    def _on_odom(self, message):
        output = copy.deepcopy(message)
        output.header.frame_id = self._world_frame
        with self._lock:
            self._odom = output

    def _on_scan(self, message):
        with self._lock:
            odom = copy.deepcopy(self._odom)
        if odom is None:
            return
        scan = self._crop_scan(message, odom)
        scan.header.frame_id = self._world_frame
        self._scan_pub.publish(scan)
        odom.header.stamp = scan.header.stamp
        self._state_pub.publish(odom)
        rospy.loginfo_throttle(
            10.0, "TARE local scan: kept=%d/%d max_range=%.2fm",
            self._scan_output_count, self._scan_input_count,
            self._scan_max_range)

    def _crop_scan(self, message, odom):
        """Crop an XYZ PointCloud2 without decoding/repacking every field."""
        if self._scan_max_range <= 0.0 or not message.data:
            self._scan_input_count = int(message.width * message.height)
            self._scan_output_count = self._scan_input_count
            return copy.deepcopy(message)

        fields = {field.name: field for field in message.fields}
        xyz = [fields.get(name) for name in ("x", "y", "z")]
        if any(field is None or field.datatype != PointField.FLOAT32 or
               field.count != 1 for field in xyz):
            rospy.logwarn_throttle(
                5.0, "Cannot range-crop registered scan: XYZ float32 fields "
                "are unavailable")
            return copy.deepcopy(message)

        point_step = int(message.point_step)
        if point_step <= 0:
            return copy.deepcopy(message)
        count = len(message.data) // point_step
        endian = ">" if message.is_bigendian else "<"
        dtype = np.dtype({
            "names": ["x", "y", "z"],
            "formats": [endian + "f4"] * 3,
            "offsets": [int(field.offset) for field in xyz],
            "itemsize": point_step,
        })
        points = np.frombuffer(message.data, dtype=dtype, count=count)
        position = odom.pose.pose.position
        dx = points["x"] - float(position.x)
        dy = points["y"] - float(position.y)
        dz = points["z"] - float(position.z)
        radius2 = self._scan_max_range * self._scan_max_range
        mask = (np.isfinite(dx) & np.isfinite(dy) & np.isfinite(dz) &
                (dx * dx + dy * dy + dz * dz <= radius2))

        raw_dtype = np.dtype(("V", point_step))
        raw_points = np.frombuffer(
            message.data, dtype=raw_dtype, count=count)
        selected = np.ascontiguousarray(raw_points[mask])
        output = copy.deepcopy(message)
        output.height = 1
        output.width = int(selected.size)
        output.row_step = output.width * point_step
        output.data = selected.tobytes()
        output.is_dense = True
        self._scan_input_count = count
        self._scan_output_count = output.width
        return output

    def _publish_terrain(self, _event):
        if not self._synthetic_terrain:
            return
        with self._lock:
            grid = copy.deepcopy(self._grid)
            odom = copy.deepcopy(self._odom)
        if grid is None or odom is None:
            return
        stamp_key = (grid.header.stamp.secs, grid.header.stamp.nsecs)
        if stamp_key == self._last_grid_stamp:
            return
        self._last_grid_stamp = stamp_key

        width, height = int(grid.info.width), int(grid.info.height)
        if width <= 0 or height <= 0 or len(grid.data) < width * height:
            rospy.logwarn_throttle(5.0, "TARE adapter received invalid grid")
            return
        resolution = float(grid.info.resolution)
        origin = grid.info.origin.position
        robot_x = float(odom.pose.pose.position.x)
        robot_y = float(odom.pose.pose.position.y)
        robot_z = float(odom.pose.pose.position.z) + self._terrain_z_offset
        radius2 = self._local_radius * self._local_radius
        local_points, extended_points = [], []

        for row in range(height):
            y = origin.y + (row + 0.5) * resolution
            base = row * width
            for column in range(width):
                value = int(grid.data[base + column])
                if value < 0:
                    continue
                obstacle = value >= self._occupied_threshold
                stride = (
                    self._occupied_stride if obstacle else self._free_stride)
                if row % stride or column % stride:
                    continue
                x = origin.x + (column + 0.5) * resolution
                intensity = 1.0 if obstacle else 0.0
                point = (x, y, robot_z, intensity)
                extended_points.append(point)
                if (x - robot_x) ** 2 + (y - robot_y) ** 2 <= radius2:
                    local_points.append(point)

        frame = self._world_frame
        header = Header(
            stamp=grid.header.stamp or rospy.Time.now(), frame_id=frame)
        local_cloud = point_cloud2.create_cloud(
            header, FIELDS_XYZI, local_points)
        extended_cloud = point_cloud2.create_cloud(
            header, FIELDS_XYZI, extended_points)
        self._terrain_pub.publish(local_cloud)
        self._terrain_ext_pub.publish(extended_cloud)
        rospy.loginfo_throttle(
            10.0, "TARE terrain adapter: local=%d extended=%d frame=%s",
            len(local_points), len(extended_points), frame)


if __name__ == "__main__":
    rospy.init_node("tare_interface")
    TareInterface()
    rospy.spin()
