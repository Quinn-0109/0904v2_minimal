#!/usr/bin/env python3
"""The sole /cmd_vel publisher for the refactored first-floor stack."""

import json
import math
import os
import sys
import threading

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud
from std_msgs.msg import Bool, String

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from first_floor_core import (
    adaptive_speed_limit,
    clustered_front_clearance,
    predicted_footprint_collision,
    proportional_yaw_rate,
    supported_obstacle_points,
    vertically_supported_obstacles,
)
from motion_safety import longitudinal_slew


class AdaptiveSpeedGuard:
    def __init__(self):
        self._lock = threading.Lock()
        self._maximum = float(rospy.get_param("~maximum_speed", 2.0))
        self._max_yaw = float(rospy.get_param("~maximum_yaw_rate", 0.65))
        self._unpitch = float(rospy.get_param("~lidar_unpitch_rad", 0.785))
        self._acceleration = float(rospy.get_param("~acceleration", 0.7))
        self._deceleration = float(rospy.get_param("~deceleration", 2.5))
        self._desired = Twist()
        self._desired_stamp = rospy.Time(0)
        self._scan_stamp = rospy.Time(0)
        self._scan_points = []
        self._front_clearance = 0.0
        self._localization_healthy = False
        self._fault_since = None
        self._state = {}
        self._hold = False
        self._last_linear = 0.0
        self._last_tick = rospy.Time.now()
        self._output = rospy.Publisher("/cmd_vel", Twist, queue_size=5)
        self._status = rospy.Publisher(
            "/simenv/speed_gate_status", String, queue_size=2
        )
        rospy.Subscriber(
            "/simenv/desired_cmd_vel", Twist, self._on_desired, queue_size=10
        )
        rospy.Subscriber("/scan", PointCloud, self._on_scan, queue_size=2)
        rospy.Subscriber(
            "/simenv/localization_healthy", Bool, self._on_health, queue_size=2
        )
        rospy.Subscriber(
            "/simenv/mission_state", String, self._on_state, queue_size=2
        )
        rospy.Subscriber("/simenv/cmd_vel_hold", Bool, self._on_hold, queue_size=2)
        rospy.Timer(rospy.Duration(0.05), self._on_timer)
        rospy.on_shutdown(lambda: self._output.publish(Twist()))
        rospy.loginfo("Adaptive speed guard is the only /cmd_vel owner (vmax=%.1f)", self._maximum)

    def _on_desired(self, message):
        with self._lock:
            self._desired = message
            self._desired_stamp = rospy.Time.now()

    def _on_health(self, message):
        now = rospy.Time.now()
        with self._lock:
            healthy = bool(message.data)
            if not healthy and self._localization_healthy:
                self._fault_since = now
            elif healthy:
                self._fault_since = None
            elif not healthy and self._fault_since is None:
                self._fault_since = now
            self._localization_healthy = healthy

    def _on_state(self, message):
        try:
            state = json.loads(message.data)
        except (TypeError, ValueError):
            return
        with self._lock:
            self._state = state

    def _on_hold(self, message):
        with self._lock:
            self._hold = bool(message.data)

    def _on_scan(self, message):
        if not message.points:
            return
        raw = np.asarray(
            [(point.x, point.y, point.z) for point in message.points], dtype=np.float64
        )
        cosine, sine = math.cos(self._unpitch), math.sin(self._unpitch)
        points = raw.copy()
        points[:, 0] = cosine * raw[:, 0] + sine * raw[:, 2]
        points[:, 2] = -sine * raw[:, 0] + cosine * raw[:, 2]
        finite = np.all(np.isfinite(points), axis=1)
        height = np.logical_and(points[:, 2] >= -0.12, points[:, 2] <= 1.0)
        candidate = vertically_supported_obstacles(points[np.logical_and(finite, height)])
        distance = np.linalg.norm(candidate[:, :2], axis=1) if len(candidate) else np.asarray([])
        # Reject the stable A1 self-return ring before footprint prediction.
        keep = distance >= 0.82 if len(candidate) else np.asarray([], dtype=bool)
        planar = candidate[keep, :2] if len(candidate) else candidate.reshape((-1, 3))[:, :2]
        supported = supported_obstacle_points(planar)
        clearance = clustered_front_clearance(supported)
        with self._lock:
            self._scan_points = supported.tolist()
            self._front_clearance = clearance
            self._scan_stamp = rospy.Time.now()

    def _on_timer(self, _event):
        now = rospy.Time.now()
        with self._lock:
            desired = self._desired
            desired_age = (now - self._desired_stamp).to_sec()
            scan_age = (now - self._scan_stamp).to_sec()
            points = list(self._scan_points)
            clearance = self._front_clearance
            healthy = self._localization_healthy
            fault_since = self._fault_since
            state = dict(self._state)
            hold = self._hold
            dt = max(0.0, (now - self._last_tick).to_sec())
            self._last_tick = now

        command = Twist()
        reason = "normal"
        yaw_error = float(state.get("yaw_error_rad", abs(desired.angular.z)))
        center_error = float(state.get("center_error_m", 0.0))
        goal_distance = float(state.get("goal_distance_m", 0.0))
        stage = str(state.get("state", "INIT"))
        speed_mode = str(state.get("speed_mode", "STOP"))
        corridor_acquired = bool(state.get("corridor_acquired", False))
        in_room = bool(state.get("in_room", stage in ("ROOM_FIRST_PASS", "ROOM_REVISIT")))
        near_door = bool(state.get("near_door", False))
        sharp_turn = bool(state.get("sharp_turn", abs(desired.angular.z) >= 0.32))
        prolonged_fault = fault_since is not None and (now - fault_since).to_sec() > 2.0

        collision = predicted_footprint_collision(points, desired.linear.x)
        if hold:
            reason = "hold"
        elif desired_age > 0.40:
            reason = "desired_stale"
        elif scan_age > 0.50:
            reason = "scan_stale"
        elif prolonged_fault:
            reason = "localization_recovery"
        elif collision:
            reason = "predicted_footprint_collision"
        else:
            limit = adaptive_speed_limit(
                clearance,
                healthy,
                yaw_error,
                center_error,
                in_room=in_room,
                near_door=near_door,
                sharp_turn=sharp_turn,
                maximum=self._maximum,
            )
            if speed_mode == "CORRIDOR_CRUISE" and not corridor_acquired:
                limit = min(limit, 0.6)
            requested = float(desired.linear.x)
            # DWA is deliberately conservative while a just-built occupancy
            # grid is sparse.  Once DWA has selected a collision-free forward
            # trajectory, promote it to the discrete safety tier; keep DWA's
            # sign/steering and let the full-footprint gate veto it instantly.
            stopping_gate = max(0.65, 0.65 * limit)
            promotable = (
                requested > 0.005
                and goal_distance > stopping_gate
                and abs(float(desired.angular.z)) < 0.20
            )
            if promotable:
                command.linear.x = limit
            elif requested > 0.005 and goal_distance > 0.25:
                command.linear.x = min(limit, 0.30)
            else:
                command.linear.x = max(-0.30, min(limit, requested))
            command.angular.z = max(-self._max_yaw, min(self._max_yaw, float(desired.angular.z)))
            if abs(yaw_error) > 0.05:
                # Mission goal yaw is authoritative.  A proportional command
                # removes the old >=0.8 rad/s discontinuity that repeatedly
                # overshot the target and triggered move_base rotate recovery.
                command.angular.z = proportional_yaw_rate(
                    yaw_error, maximum_rate=min(self._max_yaw, 0.8)
                )
            if abs(yaw_error) > 0.30:
                command.linear.x = max(-0.15, min(0.15, command.linear.x))
            if speed_mode in ("SCAN", "TURN_IN_PLACE", "STOP"):
                command.linear.x = 0.0
            if not healthy:
                reason = "localization_limited"
            elif in_room or near_door:
                reason = "room_or_door_limit"
            elif sharp_turn:
                reason = "turn_limit"
            else:
                reason = "clearance_tier_{:.1f}".format(limit)

        immediate_stop = reason in (
            "hold",
            "desired_stale",
            "scan_stale",
            "localization_recovery",
            "predicted_footprint_collision",
        )
        command.linear.x = longitudinal_slew(
            self._last_linear,
            command.linear.x,
            dt,
            acceleration=self._acceleration,
            deceleration=self._deceleration,
            immediate_zero=immediate_stop,
        )
        self._last_linear = command.linear.x
        self._output.publish(command)
        payload = {
            "stamp": round(now.to_sec(), 3),
            "state": stage,
            "reason": reason,
            "requested_speed_mps": round(float(desired.linear.x), 4),
            "published_speed_mps": round(float(command.linear.x), 4),
            "front_clearance_m": round(float(clearance), 3),
            "localization_healthy": healthy,
            "yaw_error_rad": round(yaw_error, 4),
            "center_error_m": round(center_error, 4),
            "goal_distance_m": round(goal_distance, 3),
            "tier_promoted": bool(locals().get("promotable", False)),
            "speed_mode": speed_mode,
            "collision_predicted": collision,
        }
        self._status.publish(String(data=json.dumps(payload, sort_keys=True)))


if __name__ == "__main__":
    rospy.init_node("adaptive_speed_guard")
    AdaptiveSpeedGuard()
    rospy.spin()
