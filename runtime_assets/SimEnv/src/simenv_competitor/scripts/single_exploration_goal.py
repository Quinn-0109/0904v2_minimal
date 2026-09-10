#!/usr/bin/env python3
"""Publish one reproducible PoseStamped goal after FAST-LIO becomes live."""

import math

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


class SingleGoalPublisher:
    def __init__(self):
        self._have_odom = False
        self._delay = float(rospy.get_param("~delay_after_odometry", 3.0))
        self._x = float(rospy.get_param("~x", 15.225))
        self._y = float(rospy.get_param("~y", 8.625))
        self._z = float(rospy.get_param("~z", 0.476))
        self._yaw = float(rospy.get_param("~yaw", 1.495467))
        self._frame = rospy.get_param("~frame_id", "camera_init")
        self._publisher = rospy.Publisher("/exploration_goal", PoseStamped, queue_size=1, latch=True)
        rospy.Subscriber(rospy.get_param("~odom_topic", "/Odometry"),
                         Odometry, self._on_odom, queue_size=1)
        rospy.Timer(rospy.Duration(0.25), self._tick)
        self._odom_time = None
        self._published = False

    def _on_odom(self, _message):
        if self._odom_time is None:
            self._odom_time = rospy.Time.now()

    def _tick(self, _event):
        if self._published or self._odom_time is None:
            return
        if (rospy.Time.now() - self._odom_time).to_sec() < self._delay:
            return
        message = PoseStamped()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = self._frame
        message.pose.position.x = self._x
        message.pose.position.y = self._y
        message.pose.position.z = self._z
        message.pose.orientation.z = math.sin(self._yaw * 0.5)
        message.pose.orientation.w = math.cos(self._yaw * 0.5)
        self._publisher.publish(message)
        self._published = True
        rospy.loginfo("Published single exploration goal (%.3f, %.3f, %.3f)",
                      self._x, self._y, self._z)


if __name__ == "__main__":
    rospy.init_node("single_exploration_goal")
    SingleGoalPublisher()
    rospy.spin()
