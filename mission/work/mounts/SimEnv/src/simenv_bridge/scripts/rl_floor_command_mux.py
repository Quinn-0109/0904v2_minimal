#!/usr/bin/env python3
"""Give SCAN-Planner exclusive access to the plane RL gait on flat floors.

The SCAN closed-loop controller publishes to an isolated topic.  This node
forwards a bounded, slew-limited command to the Unitree RL controller only
while the three-floor sequencer owns a flat floor.  When floor ownership is
released it publishes one stop and then remains silent, allowing the stair
transition/descent managers to own ``/cmd_vel`` without a competing zero
publisher.
"""

import math
import threading
import time

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32, String


def clamp(value, lower, upper):
    return max(float(lower), min(float(upper), float(value)))


def approach(previous, target, maximum_delta):
    delta = clamp(float(target) - float(previous), -maximum_delta, maximum_delta)
    return float(previous) + delta


def shape_command(vx, vy, wz, speed_limit, max_forward=0.65,
                  max_reverse=0.20, max_lateral=0.18, max_yaw=0.55,
                  yaw_slowdown=0.22, yaw_translation_stop=0.42):
    """Clamp a SCAN body command to the stable plane-policy envelope."""
    limit = max(0.05, min(float(speed_limit), float(max_forward)))
    vx = clamp(vx, -float(max_reverse), limit)
    vy = clamp(vy, -float(max_lateral), float(max_lateral))
    wz = clamp(wz, -float(max_yaw), float(max_yaw))

    planar = math.hypot(vx, vy)
    if planar > limit:
        scale = limit / planar
        vx *= scale
        vy *= scale

    yaw_magnitude = abs(wz)
    if yaw_magnitude >= float(yaw_translation_stop):
        vx = 0.0
        vy = 0.0
    elif yaw_magnitude > float(yaw_slowdown):
        span = max(1e-6, float(yaw_translation_stop) - float(yaw_slowdown))
        scale = clamp((float(yaw_translation_stop) - yaw_magnitude) / span,
                      0.20, 1.0)
        vx *= scale
        vy *= scale
    return vx, vy, wz


class RLFloorCommandMux:
    def __init__(self):
        self._lock = threading.RLock()
        self._rate = max(5.0, float(rospy.get_param("~rate", 30.0)))
        self._command_timeout = max(
            0.05, float(rospy.get_param("~command_timeout_sec", 0.40)))
        self._max_forward = float(rospy.get_param("~max_forward", 0.65))
        self._max_reverse = float(rospy.get_param("~max_reverse", 0.20))
        self._max_lateral = float(rospy.get_param("~max_lateral", 0.18))
        self._max_yaw = float(rospy.get_param("~max_yaw", 0.55))
        self._yaw_slowdown = float(rospy.get_param("~yaw_slowdown", 0.22))
        self._yaw_translation_stop = float(
            rospy.get_param("~yaw_translation_stop", 0.42))
        self._linear_acceleration = float(
            rospy.get_param("~linear_acceleration", 0.45))
        self._linear_deceleration = float(
            rospy.get_param("~linear_deceleration", 0.80))
        self._yaw_acceleration = float(
            rospy.get_param("~yaw_acceleration", 1.20))
        self._speed_limit = self._max_forward

        self._enabled = False
        self._enabled_at = None
        self._ready = False
        self._planner_command = (0.0, 0.0, 0.0)
        self._planner_at = None
        self._scan_active = False
        self._scan_command = (0.0, 0.0, 0.0)
        self._scan_at = None
        self._applied = (0.0, 0.0, 0.0)
        self._last_tick = time.monotonic()
        self._last_source = None

        self._command_pub = rospy.Publisher(
            rospy.get_param("~output_topic", "/cmd_vel"), Twist, queue_size=2)
        self._applied_pub = rospy.Publisher(
            "/scanplanner/applied_cmd_vel", Twist, queue_size=5)
        self._source_pub = rospy.Publisher(
            "/scanplanner/floor_command_source", String,
            queue_size=1, latch=True)

        rospy.Subscriber(
            rospy.get_param("~planner_topic", "/scanplanner/planner_cmd_vel"),
            Twist, self._on_planner, queue_size=5)
        rospy.Subscriber(
            "/scanplanner/scan_cmd_vel", Twist, self._on_scan, queue_size=5)
        rospy.Subscriber(
            "/scanplanner/scan_active", Bool, self._on_scan_active, queue_size=2)
        rospy.Subscriber(
            "/scanplanner/floor_control_enabled", Bool,
            self._on_enabled, queue_size=2)
        rospy.Subscriber(
            "/scanplanner/floor_speed_limit", Float32,
            self._on_speed_limit, queue_size=2)
        rospy.Subscriber(
            "/locomotion_ready", Bool, self._on_ready, queue_size=3)
        rospy.Timer(rospy.Duration(1.0 / self._rate), self._on_timer)
        rospy.on_shutdown(self._on_shutdown)
        self._source_pub.publish(String(data="disabled"))

    @staticmethod
    def _tuple(message):
        return (float(message.linear.x), float(message.linear.y),
                float(message.angular.z))

    @staticmethod
    def _message(values):
        message = Twist()
        message.linear.x, message.linear.y, message.angular.z = values
        return message

    def _publish(self, values, source):
        message = self._message(values)
        self._command_pub.publish(message)
        self._applied_pub.publish(message)
        if source != self._last_source:
            self._last_source = source
            self._source_pub.publish(String(data=source))

    def _publish_stop_once(self, source):
        with self._lock:
            self._applied = (0.0, 0.0, 0.0)
        self._publish(self._applied, source)

    def _on_planner(self, message):
        with self._lock:
            self._planner_command = self._tuple(message)
            self._planner_at = time.monotonic()

    def _on_scan(self, message):
        with self._lock:
            self._scan_command = self._tuple(message)
            self._scan_at = time.monotonic()

    def _on_scan_active(self, message):
        with self._lock:
            self._scan_active = bool(message.data)
            if not self._scan_active:
                self._scan_at = None

    def _on_speed_limit(self, message):
        with self._lock:
            self._speed_limit = clamp(message.data, 0.05, self._max_forward)

    def _on_ready(self, message):
        with self._lock:
            self._ready = bool(message.data)

    def _on_enabled(self, message):
        enabled = bool(message.data)
        publish_stop = False
        with self._lock:
            if enabled == self._enabled:
                return
            self._enabled = enabled
            self._applied = (0.0, 0.0, 0.0)
            self._last_tick = time.monotonic()
            self._planner_at = None
            self._scan_at = None
            self._scan_active = False
            if enabled:
                self._enabled_at = self._last_tick
            else:
                self._enabled_at = None
                publish_stop = True
        if publish_stop:
            # This is the final command from the floor mux until ownership is
            # explicitly re-enabled on the next landing.
            self._publish_stop_once("released_to_stair_controller")

    def _on_timer(self, _event):
        now = time.monotonic()
        with self._lock:
            if not self._enabled:
                return
            dt = clamp(now - self._last_tick, 0.0, 0.10)
            self._last_tick = now
            if not self._ready:
                target = (0.0, 0.0, 0.0)
                source = "waiting_for_plane_rl"
            elif (self._scan_active and self._scan_at is not None and
                  now - self._scan_at <= self._command_timeout):
                target = shape_command(
                    *self._scan_command, self._speed_limit,
                    max_forward=self._max_forward,
                    max_reverse=self._max_reverse,
                    max_lateral=self._max_lateral,
                    max_yaw=self._max_yaw,
                    yaw_slowdown=self._yaw_slowdown,
                    yaw_translation_stop=self._yaw_translation_stop)
                source = "room_scan"
            elif (not self._scan_active and self._planner_at is not None and
                  self._enabled_at is not None and
                  self._planner_at >= self._enabled_at and
                  now - self._planner_at <= self._command_timeout):
                target = shape_command(
                    *self._planner_command, self._speed_limit,
                    max_forward=self._max_forward,
                    max_reverse=self._max_reverse,
                    max_lateral=self._max_lateral,
                    max_yaw=self._max_yaw,
                    yaw_slowdown=self._yaw_slowdown,
                    yaw_translation_stop=self._yaw_translation_stop)
                source = "scanplanner"
            else:
                target = (0.0, 0.0, 0.0)
                source = "stale_command_stop"

            decelerating = math.hypot(target[0], target[1]) < math.hypot(
                self._applied[0], self._applied[1])
            linear_rate = (self._linear_deceleration if decelerating else
                           self._linear_acceleration)
            maximum_linear_delta = max(0.0, linear_rate * dt)
            maximum_yaw_delta = max(0.0, self._yaw_acceleration * dt)
            self._applied = (
                approach(self._applied[0], target[0], maximum_linear_delta),
                approach(self._applied[1], target[1], maximum_linear_delta),
                approach(self._applied[2], target[2], maximum_yaw_delta),
            )
            applied = self._applied
        self._publish(applied, source)

    def _on_shutdown(self):
        self._publish_stop_once("shutdown")


def main():
    rospy.init_node("rl_floor_command_mux", anonymous=False)
    RLFloorCommandMux()
    rospy.spin()


if __name__ == "__main__":
    main()
