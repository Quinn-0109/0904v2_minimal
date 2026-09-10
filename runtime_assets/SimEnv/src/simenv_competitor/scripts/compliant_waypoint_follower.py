#!/usr/bin/env python3
"""Follow TARE /way_point with pure pursuit on aligned /state_estimation.

Bypasses CMU localPlanner, which often stalls on sparse Livox + RealSense
door-frame clouds in SimEnv.  Still unknown exploration: goals come from TARE.

Periodically spins in place so RealSense can catch red spheres that sit
off the drive heading (danger sources are room-interior only).
"""

import math

import rospy
from geometry_msgs.msg import PointStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_diff(target, source):
    return (target - source + math.pi) % (2.0 * math.pi) - math.pi


class CompliantWaypointFollower:
    def __init__(self):
        self._pose = None
        self._waypoint = None
        self._egress_active = False
        self._structure_active = False
        self._start = rospy.Time.now()
        self._start_delay = float(rospy.get_param("~start_delay", 3.0))
        self._goal_tolerance = float(rospy.get_param("~goal_tolerance", 0.65))
        self._max_speed = float(rospy.get_param("~max_speed", 0.28))
        self._max_yaw = float(rospy.get_param("~max_yaw_rate", 0.22))
        self._slow_yaw = float(rospy.get_param("~heading_align_rad", 0.45))
        self._min_speed = float(rospy.get_param("~min_speed", 0.08))
        self._scan_enable = bool(rospy.get_param("~scan_enable", True))
        self._scan_period = float(rospy.get_param("~scan_period", 12.0))
        self._scan_duration = float(rospy.get_param("~scan_duration", 2.2))
        self._scan_yaw = float(rospy.get_param("~scan_yaw_rate", 0.35))
        self._last_scan = rospy.Time.now()
        self._scanning_until = rospy.Time(0)
        self._cmd_pub = rospy.Publisher(
            "/simenv/planner_cmd_vel", TwistStamped, queue_size=5
        )
        rospy.Subscriber("/state_estimation", Odometry, self._on_odom, queue_size=5)
        rospy.Subscriber("/way_point", PointStamped, self._on_waypoint, queue_size=5)
        rospy.Subscriber("/simenv/egress_active", Bool, self._on_egress, queue_size=1)
        rospy.Subscriber(
            "/simenv/structure_active", Bool, self._on_structure, queue_size=1
        )
        rospy.Timer(rospy.Duration(0.05), self._on_timer)
        rospy.loginfo(
            "Compliant waypoint follower: max_speed=%.2f min_speed=%.2f tol=%.2fm scan=%s",
            self._max_speed,
            self._min_speed,
            self._goal_tolerance,
            self._scan_enable,
        )

    def _on_odom(self, message):
        self._pose = message

    def _on_waypoint(self, message):
        self._waypoint = message

    def _on_egress(self, message):
        self._egress_active = bool(message.data)

    def _on_structure(self, message):
        self._structure_active = bool(message.data)

    def _on_timer(self, _event):
        # Structure explorer owns motion after lobby egress.
        if self._egress_active or self._structure_active or self._pose is None:
            return
        now = rospy.Time.now()
        if (now - self._start).to_sec() < self._start_delay:
            return

        command = Twist()
        if self._scan_enable and now < self._scanning_until:
            command.angular.z = self._scan_yaw
            self._publish(command)
            return

        if (
            self._scan_enable
            and (now - self._last_scan).to_sec() >= self._scan_period
        ):
            self._last_scan = now
            self._scanning_until = now + rospy.Duration(self._scan_duration)
            command.angular.z = self._scan_yaw
            self._publish(command)
            return

        if self._waypoint is None:
            return

        position = self._pose.pose.pose.position
        yaw = yaw_from_quaternion(self._pose.pose.pose.orientation)
        target_x = float(self._waypoint.point.x)
        target_y = float(self._waypoint.point.y)
        dx = target_x - position.x
        dy = target_y - position.y
        distance = math.hypot(dx, dy)

        if distance <= self._goal_tolerance:
            self._publish(command)
            return

        heading = math.atan2(dy, dx)
        err = angle_diff(heading, yaw)
        command.angular.z = max(-self._max_yaw, min(self._max_yaw, 1.8 * err))

        align_scale = max(0.35, 1.0 - abs(err) / max(self._slow_yaw, 1e-3))
        cruise = min(self._max_speed, 0.12 + 0.22 * distance)
        command.linear.x = max(self._min_speed, cruise * align_scale)

        self._publish(command)

    def _publish(self, twist):
        stamped = TwistStamped()
        stamped.header.stamp = rospy.Time.now()
        stamped.header.frame_id = "vehicle"
        stamped.twist = twist
        self._cmd_pub.publish(stamped)


if __name__ == "__main__":
    rospy.init_node("compliant_waypoint_follower")
    CompliantWaypointFollower()
    rospy.spin()
