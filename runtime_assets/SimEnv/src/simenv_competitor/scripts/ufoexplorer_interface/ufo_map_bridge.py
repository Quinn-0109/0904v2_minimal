#!/usr/bin/env python3
"""Bridge FAST-LIO odometry/registered cloud into UFOMap inputs.

Only FAST-LIO data is used. Registered points already live in FAST-LIO's
world coordinates, which this bridge aliases to the configured map frame.
"""

import copy
import math
import threading

import rospy
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
import tf2_ros


class UfoMapBridge:
    def __init__(self):
        self._lock = threading.Lock()
        self._map_frame = rospy.get_param("~map_frame", "map")
        self._base_frame = rospy.get_param("~base_frame", "base_link")
        self._range = float(rospy.get_param("~ufo_cloud_range", 8.0))
        self._min_z = float(rospy.get_param("~min_z", -0.5))
        self._max_z = float(rospy.get_param("~max_z", 2.5))
        self._cloud_period = 1.0 / max(
            0.1, float(rospy.get_param("~maximum_cloud_rate", 5.0)))
        self._last_cloud_time = -1e9
        self._cloud_pub = rospy.Publisher(
            rospy.get_param("~output_cloud_topic", "/simenv/ufo/cloud_filtered"),
            PointCloud2, queue_size=2)
        self._odom_pub = rospy.Publisher(
            rospy.get_param("~output_odom_topic", "/simenv/ufo/odometry"),
            Odometry, queue_size=10)
        self._status_pub = rospy.Publisher(
            "/simenv/ufo_map_bridge_status", String, queue_size=2, latch=True)
        self._tf = tf2_ros.TransformBroadcaster()
        rospy.Subscriber(rospy.get_param("~cloud_topic", "/cloud_registered"),
                         PointCloud2, self._on_cloud, queue_size=1,
                         buff_size=32 * 1024 * 1024)
        rospy.Subscriber(rospy.get_param("~odom_topic", "/Odometry"),
                         Odometry, self._on_odom, queue_size=20)
        self._cloud_count = 0
        self._odom_count = 0
        self._sensor_origin = None

    def _on_odom(self, message):
        output = copy.deepcopy(message)
        output.header.frame_id = self._map_frame
        output.child_frame_id = self._base_frame
        self._odom_pub.publish(output)
        transform = TransformStamped()
        transform.header = output.header
        transform.child_frame_id = self._base_frame
        transform.transform.translation.x = output.pose.pose.position.x
        transform.transform.translation.y = output.pose.pose.position.y
        transform.transform.translation.z = output.pose.pose.position.z
        transform.transform.rotation = output.pose.pose.orientation
        self._tf.sendTransform(transform)
        with self._lock:
            self._sensor_origin = (
                output.pose.pose.position.x, output.pose.pose.position.y,
                output.pose.pose.position.z)
        self._odom_count += 1

    def _on_cloud(self, message):
        now = rospy.get_time()
        with self._lock:
            if now - self._last_cloud_time < self._cloud_period:
                return
            self._last_cloud_time = now
        names = [field.name for field in message.fields]
        try:
            x_index, y_index, z_index = names.index("x"), names.index("y"), names.index("z")
        except ValueError:
            rospy.logwarn_throttle(2.0, "UFO bridge cloud has no xyz fields")
            return
        limit2 = self._range * self._range
        with self._lock:
            origin = self._sensor_origin
        if origin is None:
            return
        kept = []
        for point in pc2.read_points(message, skip_nans=True):
            x, y, z = point[x_index], point[y_index], point[z_index]
            relative_z = z - origin[2]
            if (self._min_z <= relative_z <= self._max_z and
                    (x - origin[0]) ** 2 + (y - origin[1]) ** 2 +
                    relative_z ** 2 <= limit2):
                kept.append(point)
        header = copy.deepcopy(message.header)
        header.frame_id = self._map_frame
        self._cloud_pub.publish(pc2.create_cloud(header, message.fields, kept))
        self._cloud_count += 1
        self._status_pub.publish(String(data=(
            '{"state":"READY","cloud_count":%d,"odom_count":%d,'
            '"input_points":%d,"filtered_points":%d}' %
            (self._cloud_count, self._odom_count,
             message.width * message.height, len(kept)))))


if __name__ == "__main__":
    rospy.init_node("ufo_map_bridge")
    UfoMapBridge()
    rospy.spin()
