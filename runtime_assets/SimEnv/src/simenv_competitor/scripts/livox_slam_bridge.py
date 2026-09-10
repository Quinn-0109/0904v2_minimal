#!/usr/bin/env python3
"""Expose sensor-only SLAM and Livox data through the TARE topic contract."""

import copy
import math

import rospy
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud, PointCloud2, PointField
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud


class LivoxSlamBridge:
    def __init__(self):
        self._map_frame = rospy.get_param("~map_frame", "map")
        self._base_frame = rospy.get_param("~base_frame", "base")
        self._voxel = float(rospy.get_param("~voxel", 0.10))
        self._max_range = float(rospy.get_param("~max_range", 20.0))
        self._max_points = int(rospy.get_param("~max_points", 12000))
        self._tf = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self._tf_listener = tf2_ros.TransformListener(self._tf)

        self._state_pub = rospy.Publisher(
            "/state_estimation", Odometry, queue_size=10
        )
        self._scan_state_pub = rospy.Publisher(
            "/state_estimation_at_scan", Odometry, queue_size=5
        )
        self._raw_cloud_pub = rospy.Publisher(
            "/livox/points2", PointCloud2, queue_size=2
        )
        self._registered_pub = rospy.Publisher(
            "/registered_scan", PointCloud2, queue_size=2
        )

        self._latest_state = None
        rospy.Subscriber("/rtabmap/odom", Odometry, self._on_odom, queue_size=20)
        rospy.Subscriber("/scan", PointCloud, self._on_livox, queue_size=2)
        rospy.loginfo(
            "Sensor-only bridge: /rtabmap/odom + /scan -> TARE contract"
        )

    def _on_odom(self, message):
        output = copy.deepcopy(message)
        output.header.frame_id = self._map_frame
        output.child_frame_id = "sensor"
        try:
            transform = self._tf.lookup_transform(
                self._map_frame,
                self._base_frame,
                message.header.stamp,
                rospy.Duration(0.05),
            )
            output.pose.pose.position.x = transform.transform.translation.x
            output.pose.pose.position.y = transform.transform.translation.y
            output.pose.pose.position.z = transform.transform.translation.z
            output.pose.pose.orientation = transform.transform.rotation
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            # Before the first loop closure, RTAB-Map's odom frame starts at
            # the mission origin and is a valid local map frame.
            pass
        self._latest_state = output
        self._state_pub.publish(output)

    def _on_livox(self, message):
        channel_by_name = {channel.name: channel.values for channel in message.channels}
        intensities = channel_by_name.get("intensity", [])
        inv_voxel = 1.0 / max(self._voxel, 1e-3)
        seen = set()
        points = []
        for index, point in enumerate(message.points):
            distance = math.sqrt(point.x * point.x + point.y * point.y + point.z * point.z)
            if distance < 0.15 or distance > self._max_range:
                continue
            key = (
                int(point.x * inv_voxel),
                int(point.y * inv_voxel),
                int(point.z * inv_voxel),
            )
            if key in seen:
                continue
            seen.add(key)
            intensity = intensities[index] if index < len(intensities) else point.z
            points.append((point.x, point.y, point.z, float(intensity)))
            if len(points) >= self._max_points:
                break

        header = copy.deepcopy(message.header)
        header.frame_id = message.header.frame_id or "laser_livox"
        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("intensity", 12, PointField.FLOAT32, 1),
        ]
        cloud = pc2.create_cloud(header, fields, points)
        self._raw_cloud_pub.publish(cloud)

        try:
            transform = self._tf.lookup_transform(
                self._map_frame,
                header.frame_id,
                rospy.Time(0),
                rospy.Duration(0.10),
            )
            registered = do_transform_cloud(cloud, transform)
            registered.header.frame_id = self._map_frame
            self._registered_pub.publish(registered)
            if self._latest_state is not None:
                scan_state = copy.deepcopy(self._latest_state)
                scan_state.header.stamp = header.stamp
                self._scan_state_pub.publish(scan_state)
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as error:
            rospy.logwarn_throttle(5.0, "Livox map transform unavailable: %s", error)


if __name__ == "__main__":
    rospy.init_node("livox_slam_bridge")
    LivoxSlamBridge()
    rospy.spin()
