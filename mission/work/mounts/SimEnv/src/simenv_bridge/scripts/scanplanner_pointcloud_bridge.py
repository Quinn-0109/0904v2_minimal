#!/usr/bin/env python3
"""Publish the simulated Livox cloud in the SCAN-Planner world frame.

The image's generic ``pointcloud2livox.py`` constructs a 2500-element Livox
``CustomMsg`` and a second PointCloud2 on every 10 Hz callback.  This mission
uses neither FAST-LIO's CustomMsg input nor its odometry: SCAN-Planner is
explicitly configured with Gazebo odometry and a world-frame cloud.  This
bridge therefore performs the same sensor->base->world rigid transform once,
publishes an intensity-bearing PointCloud2 at a bounded rate, and avoids the
unused per-point ROS-object copy.
"""

import math
import threading
import time

import numpy as np
import rospy
import tf
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud, PointCloud2, PointField


FLOAT32 = PointField.FLOAT32


def quaternion_matrix_xyzw(x, y, z, w):
    """Return the 3x3 rotation matrix for a finite xyzw quaternion."""
    norm = x * x + y * y + z * z + w * w
    if not math.isfinite(norm) or norm <= 1.0e-12:
        return np.eye(3, dtype=np.float32)
    scale = 2.0 / norm
    xx, yy, zz = x * x * scale, y * y * scale, z * z * scale
    xy, xz, yz = x * y * scale, x * z * scale, y * z * scale
    wx, wy, wz = w * x * scale, w * y * scale, w * z * scale
    return np.asarray([
        [1.0 - yy - zz, xy - wz, xz + wy],
        [xy + wz, 1.0 - xx - zz, yz - wx],
        [xz - wy, yz + wx, 1.0 - xx - yy],
    ], dtype=np.float32)


def pointcloud2_xyzi(points, header):
    """Build a packed XYZI PointCloud2 without one Python object per point."""
    xyz = np.asarray(points, dtype=np.float32).reshape((-1, 3))
    packed = np.empty((xyz.shape[0], 4), dtype=np.float32)
    packed[:, :3] = xyz
    packed[:, 3] = 1.0
    message = PointCloud2()
    message.header = header
    message.height = 1
    message.width = int(packed.shape[0])
    message.fields = [
        PointField(name="x", offset=0, datatype=FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=FLOAT32, count=1),
    ]
    message.is_bigendian = False
    message.point_step = 16
    message.row_step = 16 * message.width
    message.is_dense = True
    message.data = packed.tobytes(order="C")
    return message


class ScanPlannerPointCloudBridge:
    def __init__(self):
        self._input = str(rospy.get_param("~input", "/scan"))
        self._output = str(rospy.get_param("~output", "/livox/Pointcloud2"))
        self._odom_topic = str(
            rospy.get_param("~odom_topic", "/Odometry_gazebo"))
        self._base_frame = str(rospy.get_param("~base_frame", "base"))
        self._sensor_frame = str(
            rospy.get_param("~sensor_frame", "laser_livox"))
        self._world_frame = str(rospy.get_param("~world_frame", "odom"))
        maximum_rate = max(0.1, float(rospy.get_param(
            "~maximum_rate_hz", 5.0)))
        self._minimum_period = 1.0 / maximum_rate
        self._blind = max(0.0, float(rospy.get_param("~laser_blind", 0.5)))
        self._minimum_angle = math.radians(float(rospy.get_param(
            "~minimum_vertical_angle_deg", -7.0)))
        self._maximum_angle = math.radians(float(rospy.get_param(
            "~maximum_vertical_angle_deg", 55.0)))

        self._lock = threading.RLock()
        self._odom = None
        self._sensor_rotation = None
        self._sensor_translation = None
        self._last_callback_wall = 0.0
        self._published = 0
        self._listener = tf.TransformListener()
        self._publisher = rospy.Publisher(
            self._output, PointCloud2, queue_size=2)
        rospy.Subscriber(
            self._odom_topic, Odometry, self._on_odom, queue_size=10)
        rospy.Subscriber(
            self._input, PointCloud, self._on_cloud, queue_size=1)
        rospy.loginfo(
            "scanplanner_pointcloud_bridge: %s -> %s at <= %.1f Hz",
            self._input, self._output, maximum_rate)

    def _on_odom(self, message):
        with self._lock:
            self._odom = message

    def _sensor_transform(self):
        with self._lock:
            if self._sensor_rotation is not None:
                return self._sensor_rotation, self._sensor_translation
        try:
            translation, quaternion = self._listener.lookupTransform(
                self._base_frame, self._sensor_frame, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException,
                tf.ExtrapolationException):
            return None
        matrix = tf.transformations.quaternion_matrix(quaternion)[:3, :3]
        rotation = np.asarray(matrix, dtype=np.float32)
        offset = np.asarray(translation, dtype=np.float32)
        with self._lock:
            self._sensor_rotation = rotation
            self._sensor_translation = offset
        return rotation, offset

    def _on_cloud(self, message):
        now = time.monotonic()
        with self._lock:
            if now - self._last_callback_wall < self._minimum_period:
                return
            odom = self._odom
            self._last_callback_wall = now
        if odom is None:
            return
        sensor_transform = self._sensor_transform()
        if sensor_transform is None:
            return
        if not message.points:
            return

        points = np.fromiter(
            (coordinate for point in message.points
             for coordinate in (point.x, point.y, point.z)),
            dtype=np.float32,
            count=3 * len(message.points),
        ).reshape((-1, 3))
        planar = np.linalg.norm(points[:, :2], axis=1)
        vertical = np.arctan2(points[:, 2], planar)
        ranges = np.linalg.norm(points, axis=1)
        keep = (
            np.isfinite(points).all(axis=1) &
            (vertical >= self._minimum_angle) &
            (vertical <= self._maximum_angle) &
            (ranges >= self._blind)
        )
        points = points[keep]
        if points.size == 0:
            return

        sensor_rotation, sensor_translation = sensor_transform
        base_points = points.dot(sensor_rotation.T) + sensor_translation
        pose = odom.pose.pose
        q = pose.orientation
        world_rotation = quaternion_matrix_xyzw(q.x, q.y, q.z, q.w)
        world_translation = np.asarray(
            [pose.position.x, pose.position.y, pose.position.z],
            dtype=np.float32)
        world_points = base_points.dot(world_rotation.T) + world_translation

        header = message.header
        header.frame_id = self._world_frame
        self._publisher.publish(pointcloud2_xyzi(world_points, header))
        self._published += 1
        if self._published % 50 == 0:
            rospy.loginfo(
                "scanplanner_pointcloud_bridge: published %d clouds (%d points)",
                self._published, world_points.shape[0])


def main():
    rospy.init_node("scanplanner_pointcloud_bridge", anonymous=False)
    ScanPlannerPointCloudBridge()
    rospy.spin()


if __name__ == "__main__":
    main()
