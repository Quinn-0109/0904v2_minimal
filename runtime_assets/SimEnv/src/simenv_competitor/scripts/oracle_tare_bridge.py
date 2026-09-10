#!/usr/bin/env python3
"""Adapter for the first-floor planner benchmark.

This bridge intentionally consumes referee odometry. It is only an oracle
baseline for validating TARE, locomotion, perception, and timing. It must not
be used for a competition submission.
"""

import copy

import rospy
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2


class OracleTareBridge:
    def __init__(self):
        self._latest_odom = None
        self._state_pub = rospy.Publisher("/state_estimation", Odometry, queue_size=5)
        self._scan_state_pub = rospy.Publisher(
            "/state_estimation_at_scan", Odometry, queue_size=5
        )
        self._scan_pub = rospy.Publisher("/registered_scan", PointCloud2, queue_size=2)
        self._cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=5)

        rospy.Subscriber("/Odometry_gazebo", Odometry, self._on_odom, queue_size=10)
        rospy.Subscriber(
            "/livox/Pointcloud2", PointCloud2, self._on_cloud, queue_size=2
        )
        rospy.Subscriber(
            "/tare/cmd_vel_stamped", TwistStamped, self._on_cmd, queue_size=5
        )

    @staticmethod
    def _tare_odom(message, stamp=None):
        output = copy.deepcopy(message)
        output.header.frame_id = "map"
        output.child_frame_id = "sensor"
        if stamp is not None:
            output.header.stamp = stamp
        return output

    def _on_odom(self, message):
        self._latest_odom = message
        self._state_pub.publish(self._tare_odom(message))

    def _on_cloud(self, message):
        if self._latest_odom is None:
            return
        cloud = copy.deepcopy(message)
        cloud.header.frame_id = "map"
        self._scan_pub.publish(cloud)
        self._scan_state_pub.publish(
            self._tare_odom(self._latest_odom, message.header.stamp)
        )

    def _on_cmd(self, message):
        command = copy.deepcopy(message.twist)
        max_linear = rospy.get_param("~max_linear_speed", 0.8)
        max_yaw = rospy.get_param("~max_yaw_rate", 1.0)
        command.linear.x = max(-max_linear, min(max_linear, command.linear.x))
        command.linear.y = max(-max_linear, min(max_linear, command.linear.y))
        command.angular.z = max(-max_yaw, min(max_yaw, command.angular.z))
        self._cmd_pub.publish(command)


if __name__ == "__main__":
    rospy.init_node("oracle_tare_bridge")
    OracleTareBridge()
    rospy.spin()
