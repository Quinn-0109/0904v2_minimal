#!/usr/bin/env python3
"""Safe /cmd_vel gate for compliant quadruped exploration.

Priority (highest first):
  1. hold (fall recovery)
  2. lobby egress
  3. structure-aware explorer
  4. planner / waypoint follower

Also applies startup delay/ramp (bypassed for egress/structure), clamps, and
periodic republish for junior_ctrl.
"""

import os
import sys
import threading

import rospy
from geometry_msgs.msg import Twist, TwistStamped
from std_msgs.msg import Bool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion_safety import longitudinal_slew


class CmdVelGuard:
    def __init__(self):
        self._lock = threading.Lock()
        self._hold = False
        if rospy.get_param("/use_sim_time", False):
            while not rospy.is_shutdown() and rospy.Time.now().to_sec() < 1.0:
                rospy.sleep(0.05)
        self._start = rospy.Time.now()
        self._start_delay = float(rospy.get_param("~start_delay", 4.0))
        self._ramp_seconds = float(rospy.get_param("~ramp_seconds", 10.0))
        self._max_linear = float(rospy.get_param("~max_linear_speed", 0.4))
        self._max_yaw = float(rospy.get_param("~max_yaw_rate", 0.45))
        self._max_lateral = float(rospy.get_param("~max_lateral_speed", 0.0))
        self._turn_linear_scale = float(rospy.get_param("~turn_linear_scale", 0.55))
        self._turn_yaw_threshold = float(rospy.get_param("~turn_yaw_threshold", 0.25))
        self._priority_turn_scale = float(
            rospy.get_param("~egress_turn_linear_scale", 0.75)
        )
        self._max_acceleration = float(
            rospy.get_param("~max_linear_acceleration", 0.70)
        )
        self._max_deceleration = float(
            rospy.get_param("~max_linear_deceleration", 2.50)
        )
        self._egress_active = False
        self._structure_active = False
        self._last_egress_cmd = rospy.Time(0)
        self._last_structure_cmd = rospy.Time(0)
        self._last_command = Twist()
        self._last_command_time = rospy.Time(0)
        self._last_applied_linear = 0.0
        self._last_slew_time = rospy.Time.now()

        self._pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        rospy.Subscriber(
            "/simenv/planner_cmd_vel", TwistStamped, self._on_planner_cmd, queue_size=10
        )
        rospy.Subscriber(
            "/simenv/egress_cmd_vel", TwistStamped, self._on_egress_cmd, queue_size=10
        )
        rospy.Subscriber(
            "/simenv/structure_cmd_vel",
            TwistStamped,
            self._on_structure_cmd,
            queue_size=10,
        )
        rospy.Subscriber("/simenv/cmd_vel_hold", Bool, self._on_hold, queue_size=1)
        rospy.Subscriber("/simenv/egress_active", Bool, self._on_egress_active, queue_size=1)
        rospy.Subscriber(
            "/simenv/structure_active", Bool, self._on_structure_active, queue_size=1
        )
        rospy.Timer(rospy.Duration(0.05), self._on_republish)
        rospy.loginfo(
            "cmd_vel guard: delay=%.1fs ramp=%.1fs vmax=%.2f (egress>structure>planner)",
            self._start_delay,
            self._ramp_seconds,
            self._max_linear,
        )

    def _on_hold(self, message):
        with self._lock:
            self._hold = bool(message.data)

    def _on_egress_active(self, message):
        with self._lock:
            self._egress_active = bool(message.data)

    def _on_structure_active(self, message):
        with self._lock:
            self._structure_active = bool(message.data)

    def _speed_scale(self, now, priority):
        if priority:
            return 1.0
        elapsed = (now - self._start).to_sec()
        if elapsed < 0.0:
            return 1.0
        if elapsed < self._start_delay:
            return 0.0
        if self._ramp_seconds <= 0.0:
            return 1.0
        return min(1.0, (elapsed - self._start_delay) / self._ramp_seconds)

    def _on_egress_cmd(self, message):
        self._last_egress_cmd = rospy.Time.now()
        self._apply(message.twist, priority=True)

    def _on_structure_cmd(self, message):
        now = rospy.Time.now()
        with self._lock:
            if self._egress_active or (now - self._last_egress_cmd).to_sec() < 0.25:
                return
            structure_active = self._structure_active
        self._last_structure_cmd = now
        self._apply(message.twist, priority=structure_active)

    def _on_planner_cmd(self, message):
        now = rospy.Time.now()
        with self._lock:
            if self._egress_active or (now - self._last_egress_cmd).to_sec() < 0.25:
                return
            if self._structure_active or (now - self._last_structure_cmd).to_sec() < 0.25:
                return
        self._apply(message.twist, priority=False)

    def _apply(self, twist, priority=False, now=None):
        if now is None:
            now = rospy.Time.now()
        with self._lock:
            hold = self._hold
        if hold:
            with self._lock:
                self._last_applied_linear = 0.0
                self._last_slew_time = now
            self._last_command = Twist()
            self._last_command_time = now
            self._pub.publish(Twist())
            return

        scale = self._speed_scale(now, priority)
        if scale <= 0.0:
            with self._lock:
                self._last_applied_linear = 0.0
                self._last_slew_time = now
            self._last_command = Twist()
            self._last_command_time = now
            self._pub.publish(Twist())
            return

        command = Twist()
        command.linear.x = twist.linear.x * scale
        command.linear.y = twist.linear.y * scale
        command.angular.z = twist.angular.z * scale
        command.linear.y = max(
            -self._max_lateral, min(self._max_lateral, command.linear.y)
        )
        command.linear.x = max(
            -self._max_linear, min(self._max_linear, command.linear.x)
        )
        command.angular.z = max(-self._max_yaw, min(self._max_yaw, command.angular.z))

        turn_scale = self._priority_turn_scale if priority else self._turn_linear_scale
        if abs(command.angular.z) >= self._turn_yaw_threshold:
            command.linear.x *= turn_scale
            command.linear.y *= turn_scale

        with self._lock:
            dt = max(0.0, (now - self._last_slew_time).to_sec())
            command.linear.x = longitudinal_slew(
                self._last_applied_linear,
                command.linear.x,
                dt,
                acceleration=self._max_acceleration,
                deceleration=self._max_deceleration,
                immediate_zero=True,
            )
            self._last_applied_linear = command.linear.x
            self._last_slew_time = now

        self._last_command = command
        self._last_command_time = now
        self._pub.publish(command)

    def _on_republish(self, _event):
        now = rospy.Time.now()
        with self._lock:
            hold = self._hold
            command = self._last_command
            last_time = self._last_command_time
        if hold or (now - last_time).to_sec() > 0.5:
            return
        self._pub.publish(command)


if __name__ == "__main__":
    rospy.init_node("cmd_vel_guard")
    CmdVelGuard()
    rospy.spin()
