#!/usr/bin/env python3
"""Drive out of the lobby using LIO odometry + body-frame /scan structure.

Publishes:
  - /way_point toward corridor center (for TARE / direct follower)
  - /simenv/egress_cmd_vel forward crawl (bypasses planner while active)
  - /simenv/egress_active while seeding

Layout (seed 77): lobby y∈[0, 7.85], corridor mouth x∈[-1.1, 1.1] at y≈7.85.
Completion requires clearing that mouth — never hand off while still wall-following
the lobby north wall.
"""

import json
import math

import rospy
from geometry_msgs.msg import PointStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud
from std_msgs.msg import Bool, String

from motion_safety import clearance_speed_limit, straight_boost_allowed


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wait_for_sim_time(timeout=30.0):
    """Avoid Time(0) -> large sim clock jump that instantly trips max_seconds."""
    deadline = rospy.Time.now() + rospy.Duration(timeout)
    if not rospy.get_param("/use_sim_time", False):
        return rospy.Time.now()
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        now = rospy.Time.now()
        if now.to_sec() > 1.0:
            return now
        if now > deadline:
            return now
        rate.sleep()
    return rospy.Time.now()


class LobbyEgressSeed:
    def __init__(self):
        self._pose = None
        self._start_xy = None
        self._done = False
        self._exit_y = float(rospy.get_param("~exit_y", 10.0))
        self._exit_distance = float(rospy.get_param("~exit_distance", 8.0))
        self._goal_y = float(rospy.get_param("~goal_y", 14.0))
        self._goal_x = float(rospy.get_param("~goal_x", 0.0))
        self._max_seconds = float(rospy.get_param("~max_seconds", 45.0))
        self._crawl_speed = float(rospy.get_param("~crawl_speed", 0.22))
        self._straight_speed = float(
            rospy.get_param("~straight_speed", self._crawl_speed)
        )
        self._yaw_gain = float(rospy.get_param("~yaw_gain", 1.2))
        self._max_yaw = float(rospy.get_param("~max_yaw", 0.38))
        self._motion_delay = float(rospy.get_param("~motion_delay", 3.5))
        self._corridor_yaw = float(rospy.get_param("~corridor_yaw", 1.5708))
        self._reverse_after = float(rospy.get_param("~reverse_if_no_progress_s", 6.0))
        self._progress_dy = float(rospy.get_param("~progress_dy", 0.6))
        # Structure gate: do not trust absolute LIO Y alone.
        self._min_travel_m = float(rospy.get_param("~min_travel_m", 7.0))
        self._corridor_half_min = float(rospy.get_param("~corridor_half_min", 0.55))
        self._corridor_half_max = float(rospy.get_param("~corridor_half_max", 1.70))
        self._corridor_width_max = float(rospy.get_param("~corridor_width_max", 3.4))
        self._front_clear = float(rospy.get_param("~front_clear_m", 2.2))
        self._narrow_front_clear = float(
            rospy.get_param("~narrow_front_clear_m", 1.6)
        )
        self._structure_hits_needed = int(rospy.get_param("~structure_hits_needed", 8))
        # Mouth geometry (lobby -> corridor). Past this, explorer can take over.
        self._mouth_y = float(rospy.get_param("~mouth_y", 8.2))
        self._mouth_half_width = float(rospy.get_param("~mouth_half_width", 0.95))
        self._min_north_finish = float(rospy.get_param("~min_north_finish", 6.2))
        self._x_gain = float(rospy.get_param("~x_gain", 0.85))
        self._heading_flipped = False
        self._best_y = None
        self._start_time = wait_for_sim_time()
        self._motion_start_y = None
        self._motion_began = None
        self._path_length = 0.0
        self._last_xy = None
        self._left = None
        self._right = None
        self._front = None
        self._narrow_front = None
        self._localization_healthy = False
        self._structure_hits = 0
        self._stuck_since = None
        self._last_progress_y = None
        self._wp_pub = rospy.Publisher("/way_point", PointStamped, queue_size=1)
        self._cmd_pub = rospy.Publisher(
            "/simenv/egress_cmd_vel", TwistStamped, queue_size=5
        )
        self._active_pub = rospy.Publisher(
            "/simenv/egress_active", Bool, queue_size=1, latch=True
        )
        self._speed_gate_pub = rospy.Publisher(
            "/simenv/speed_gate_status", String, queue_size=2
        )
        self._active_pub.publish(Bool(data=True))
        rospy.Subscriber("/state_estimation", Odometry, self._on_odom, queue_size=5)
        rospy.Subscriber("/scan", PointCloud, self._on_scan, queue_size=2)
        rospy.Subscriber(
            "/simenv/localization_healthy", Bool, self._on_localization_health, queue_size=1
        )
        rospy.Timer(rospy.Duration(0.1), self._on_timer)
        rospy.loginfo(
            "Lobby egress seed armed (exit_y=%.1f dist=%.1f min_travel=%.1f "
            "crawl=%.2f straight=%.2f goal=(%.1f,%.1f) mouth_y=%.1f)",
            self._exit_y,
            self._exit_distance,
            self._min_travel_m,
            self._crawl_speed,
            self._straight_speed,
            self._goal_x,
            self._goal_y,
            self._mouth_y,
        )

    def _finish(self, reason, position_y, elapsed):
        self._done = True
        self._cmd_pub.publish(TwistStamped(twist=Twist()))
        self._active_pub.publish(Bool(data=False))
        rospy.loginfo(
            "Lobby egress seed done (%s y=%.2f path=%.2f hits=%d elapsed=%.1fs)",
            reason,
            position_y,
            self._path_length,
            self._structure_hits,
            elapsed,
        )

    def _publish_speed_gate(self, yaw_error, center_error, requested_speed, reason):
        payload = {
            "stage": "lobby_egress",
            "reason": reason,
            "requested_speed_mps": round(float(requested_speed), 4),
            "front_clearance_m": self._front,
            "narrow_front_m": self._narrow_front,
            "localization_healthy": self._localization_healthy,
            "yaw_error_rad": round(float(yaw_error), 4),
            "center_error_m": round(float(center_error), 4),
        }
        self._speed_gate_pub.publish(String(data=json.dumps(payload, sort_keys=True)))

    def _distance(self):
        if self._pose is None or self._start_xy is None:
            return 0.0
        p = self._pose.pose.pose.position
        return math.hypot(p.x - self._start_xy[0], p.y - self._start_xy[1])

    def _on_odom(self, message):
        self._pose = message
        p = message.pose.pose.position
        xy = (p.x, p.y)
        if self._last_xy is not None:
            step = math.hypot(xy[0] - self._last_xy[0], xy[1] - self._last_xy[1])
            if step < 1.0:
                self._path_length += step
        self._last_xy = xy

    def _on_localization_health(self, message):
        self._localization_healthy = bool(message.data)

    def _on_scan(self, message):
        if not message.points:
            return
        left, right, front, narrow_front = [], [], [], []
        for point in message.points:
            x, y = float(point.x), float(point.y)
            rng = math.hypot(x, y)
            if not math.isfinite(rng) or rng < 0.35 or rng > 12.0:
                continue
            deg = math.degrees(math.atan2(y, x))
            if abs(deg) <= 25.0:
                front.append(rng)
                if abs(deg) <= 10.0:
                    narrow_front.append(rng)
            elif 60.0 <= deg <= 120.0:
                left.append(rng)
            elif -120.0 <= deg <= -60.0:
                right.append(rng)
        self._left = float(sorted(left)[len(left) // 4]) if left else None
        self._right = float(sorted(right)[len(right) // 4]) if right else None
        # Broad p82 estimates how far a sizeable straight corridor remains.
        # The narrow p20 is a separate compact-obstacle veto.  Requiring a
        # percentile (rather than the minimum) rejects isolated body/leg rays.
        self._front = self._percentile(front, 0.82)
        self._narrow_front = self._percentile(narrow_front, 0.20)
        if self._corridor_like():
            self._structure_hits += 1
        else:
            self._structure_hits = max(0, self._structure_hits - 1)

    @staticmethod
    def _percentile(values, fraction):
        if not values:
            return None
        ordered = sorted(values)
        index = int(round((len(ordered) - 1) * max(0.0, min(1.0, fraction))))
        return float(ordered[index])

    def _obstacle_front(self):
        """Conservative forward range for the existing blockage protections."""
        if self._narrow_front is not None:
            return self._narrow_front
        return self._front

    def _front_path_clear(self):
        broad_clear = self._front is None or self._front >= self._front_clear
        narrow_clear = (
            self._narrow_front is None
            or self._narrow_front >= self._narrow_front_clear
        )
        return broad_clear and narrow_clear

    def _corridor_like(self):
        if self._left is None or self._right is None:
            return False
        width = self._left + self._right
        if width > self._corridor_width_max or width < 1.1:
            return False
        if not (
            self._corridor_half_min <= self._left <= self._corridor_half_max
            and self._corridor_half_min <= self._right <= self._corridor_half_max
        ):
            return False
        if not self._front_path_clear():
            return False
        return True

    def _desired_yaw(self):
        yaw = self._corridor_yaw
        if self._heading_flipped:
            yaw = math.atan2(math.sin(yaw + math.pi), math.cos(yaw + math.pi))
        return yaw

    def _north_progress(self, p):
        if self._start_xy is None:
            return 0.0
        best = self._best_y if self._best_y is not None else p.y
        return max(0.0, best - self._start_xy[1], p.y - self._start_xy[1])

    def _past_mouth(self, p):
        """True once we have cleared the lobby->corridor opening."""
        north = self._north_progress(p)
        in_mouth_x = abs(p.x - self._goal_x) <= self._mouth_half_width + 0.35
        return (p.y >= self._mouth_y and in_mouth_x) or north >= self._min_north_finish + 1.5

    def _ready_to_finish(self, p, dist, elapsed):
        traveled = max(dist, self._path_length)
        north = self._north_progress(p)
        front_ok = self._front_path_clear()
        structure_ok = self._structure_hits >= self._structure_hits_needed and front_ok
        past = self._past_mouth(p)

        # Never finish while still in the lobby (run6-9 wall-follow trap).
        # Cap so a jam cannot consume the whole 150s budget.
        if not past and north < self._min_north_finish:
            if elapsed >= self._max_seconds + 20.0:
                return "hard_timeout"
            return None

        if past and structure_ok and north >= 5.5:
            return "structure+mouth"
        if past and traveled >= 6.0 and north >= self._min_north_finish:
            return "mouth+north"
        # FAST-LIO can loop back after it has already observed the mouth.  The
        # peak north progress plus continuous travelled distance is enough to
        # hand control to the scan-reactive explorer; requiring the *current*
        # drifting pose to remain past the mouth kept egress active for ~60 s.
        if traveled >= 6.0 and north >= self._min_north_finish:
            return "peak+north+travel"
        if traveled >= self._min_travel_m and north >= self._min_north_finish and structure_ok:
            return "structure+travel"
        if traveled >= self._exit_distance and front_ok and north >= self._min_north_finish:
            return "progress+front"
        if elapsed >= self._max_seconds and past and north >= 5.5 and traveled >= 6.0:
            return "timeout"
        # Absolute last resort — only if we somehow cleared the mouth.
        if elapsed >= self._max_seconds + 25.0:
            return "hard_timeout"
        return None

    def _goal_yaw(self, p, yaw):
        """Prefer north, with strong pull toward corridor centerline x=goal_x."""
        # Point slightly ahead on the centerline so we thread the mouth.
        target_y = max(p.y + 3.5, min(self._goal_y, p.y + 8.0))
        if p.y < self._mouth_y:
            # Still in lobby: aim at mouth center, not a far north point that
            # lets large |x| error produce a shallow diagonal into the wall.
            target_x = self._goal_x
            target_y = max(self._mouth_y + 1.0, p.y + 4.0)
        else:
            target_x = self._goal_x
        bearing = math.atan2(target_y - p.y, target_x - p.x)
        # Blend bearing with pure corridor yaw so we do not face SE/SW in lobby.
        north = self._desired_yaw()
        bx = math.cos(bearing)
        by = math.sin(bearing)
        nx = math.cos(north)
        ny = math.sin(north)
        # Heavier north weight while |x| is small; heavier bearing when off-center.
        w_bear = min(0.75, abs(p.x - self._goal_x) * self._x_gain)
        w_north = 1.0 - w_bear
        mx = w_north * nx + w_bear * bx
        my = w_north * ny + w_bear * by
        if abs(mx) + abs(my) < 1e-3:
            return north
        return math.atan2(my, mx)

    def _on_timer(self, _event):
        if self._done or self._pose is None:
            return
        now = rospy.Time.now()
        if self._start_time.to_sec() < 1.0 and now.to_sec() > 1.0:
            self._start_time = now
            return
        elapsed = (now - self._start_time).to_sec()
        if elapsed < 0.0:
            self._start_time = now
            return
        if elapsed < self._motion_delay:
            return

        p = self._pose.pose.pose.position
        if self._start_xy is None:
            self._start_xy = (p.x, p.y)
            self._best_y = p.y
            self._motion_start_y = p.y
            self._motion_began = now
            self._last_progress_y = p.y
            rospy.loginfo(
                "Lobby egress start at (%.2f, %.2f) t=%.1f",
                self._start_xy[0],
                self._start_xy[1],
                now.to_sec(),
            )

        if p.y > self._best_y:
            self._best_y = p.y
        if self._last_progress_y is None or p.y > self._last_progress_y + 0.15:
            self._last_progress_y = p.y
            self._stuck_since = None
        else:
            if self._stuck_since is None:
                self._stuck_since = now

        # Only reverse if jammed into a wall with almost no travel (wrong face).
        if (
            not self._heading_flipped
            and self._motion_began is not None
            and (now - self._motion_began).to_sec() >= self._reverse_after + 4.0
            and self._path_length < 1.2
            and self._obstacle_front() is not None
            and self._obstacle_front() < 0.9
            and (self._left is None or self._left > 2.0)
            and (self._right is None or self._right > 2.0)
        ):
            self._heading_flipped = True
            self._motion_began = now
            self._motion_start_y = p.y
            rospy.logwarn(
                "Lobby egress: wall jam (F=%.2f path=%.2f); reversing heading",
                self._obstacle_front(),
                self._path_length,
            )

        dist = self._distance()
        reason = self._ready_to_finish(p, dist, elapsed)
        if reason is not None:
            self._finish(reason, p.y, elapsed)
            return

        wp = PointStamped()
        wp.header.stamp = now
        wp.header.frame_id = "map"
        # Keep waypoint on corridor centerline ahead of us.
        wp.point.x = self._goal_x
        wp.point.y = max(self._goal_y, p.y + 4.0)
        wp.point.z = 0.15
        self._wp_pub.publish(wp)

        yaw = yaw_from_quaternion(self._pose.pose.pose.orientation)
        desired = self._goal_yaw(p, yaw)

        # Soft corridor centering only AFTER we are past the lobby mouth.
        # In the open lobby, L/R openness wall-follows the north wall (run9).
        if (
            self._past_mouth(p)
            and self._left is not None
            and self._right is not None
        ):
            center = max(-0.30, min(0.30, 0.35 * (self._left - self._right)))
            desired = desired + center

        yaw_err = math.atan2(math.sin(desired - yaw), math.cos(desired - yaw))
        cmd = TwistStamped()
        cmd.header.stamp = now
        cmd.header.frame_id = "vehicle"
        cmd.twist.angular.z = max(
            -self._max_yaw, min(self._max_yaw, self._yaw_gain * yaw_err)
        )

        front = self._front if self._front is not None else 0.0
        obstacle_front = self._obstacle_front()
        obstacle_front = obstacle_front if obstacle_front is not None else 0.0
        x_err = p.x - self._goal_x
        in_lobby = p.y < self._mouth_y and self._north_progress(p) < self._min_north_finish

        # Front blocked in lobby: NEVER turn toward "more open" (east/west along
        # the north wall). Always re-center on the corridor mouth.
        # Do NOT reverse — run10 body returns triggered reverse and drove south
        # out the entrance to y≈0.4.
        blocked = obstacle_front < 0.95 or (
            obstacle_front < 1.35 and abs(x_err) > 0.70
        )
        if in_lobby and blocked:
            mouth_bearing = math.atan2(
                (self._mouth_y + 1.5) - p.y, self._goal_x - p.x
            )
            yaw_err = math.atan2(
                math.sin(mouth_bearing - yaw), math.cos(mouth_bearing - yaw)
            )
            cmd.twist.angular.z = max(
                -self._max_yaw, min(self._max_yaw, 1.4 * yaw_err)
            )
            if abs(yaw_err) > 0.55:
                cmd.twist.linear.x = 0.0
            elif abs(x_err) > 0.45:
                cmd.twist.linear.x = 0.12
            else:
                cmd.twist.linear.x = 0.20
            self._publish_speed_gate(
                yaw_err, x_err, cmd.twist.linear.x, "near_obstacle_or_recenter"
            )
            self._cmd_pub.publish(cmd)
            return

        # Off-center in lobby with clear front: bias turn to kill |x| before racing north.
        if in_lobby and abs(x_err) > 0.40:
            bias = 0.55 * math.atan2(-x_err, 3.0)
            blend = math.atan2(
                math.sin(self._corridor_yaw + bias - yaw),
                math.cos(self._corridor_yaw + bias - yaw),
            )
            cmd.twist.angular.z = max(
                -self._max_yaw, min(self._max_yaw, self._yaw_gain * blend)
            )
            yaw_err = blend

        turn_scale = max(0.35, 1.0 - abs(yaw_err) / 1.1)
        speed = self._crawl_speed * turn_scale
        boost_allowed = straight_boost_allowed(
            self._localization_healthy,
            yaw_err,
            x_err,
            self._narrow_front,
            min_narrow_front=self._narrow_front_clear,
        )
        gate_reason = "straight_boost" if boost_allowed else "crawl_gate"
        if boost_allowed:
            speed = clearance_speed_limit(
                self._front,
                self._straight_speed,
                cruise_speed=self._crawl_speed,
            )
        if obstacle_front < 1.6:
            speed = min(speed, self._crawl_speed)
            speed *= max(0.25, (obstacle_front - 0.6) / 1.0)
        # If the estimator stops making north progress, keep moving whenever
        # the body-frame scan says the path is clear.  LIO loop closures must
        # not turn a healthy straight walk into a permanent 0.22 m/s crawl.
        if (
            self._stuck_since is not None
            and (now - self._stuck_since).to_sec() > 4.0
            and in_lobby
        ):
            north_err = math.atan2(
                math.sin(self._desired_yaw() - yaw),
                math.cos(self._desired_yaw() - yaw),
            )
            cmd.twist.angular.z = max(
                -self._max_yaw, min(self._max_yaw, 1.5 * north_err)
            )
            if abs(north_err) > 0.35 and obstacle_front < 1.8:
                speed = 0.0
            else:
                speed = self._crawl_speed if abs(x_err) < 0.8 else max(
                    0.22, self._crawl_speed * 0.65
                )

        # Hard clamp: never command reverse, and never drive south of spawn.
        if self._start_xy is not None and p.y < self._start_xy[1] - 0.25:
            north_err = math.atan2(
                math.sin(self._desired_yaw() - yaw),
                math.cos(self._desired_yaw() - yaw),
            )
            cmd.twist.angular.z = max(
                -self._max_yaw, min(self._max_yaw, 1.6 * north_err)
            )
            speed = 0.28 if abs(north_err) < 0.45 else 0.0

        cmd.twist.linear.x = max(0.0, speed)
        self._publish_speed_gate(
            yaw_err, x_err, cmd.twist.linear.x, gate_reason
        )
        self._cmd_pub.publish(cmd)


if __name__ == "__main__":
    rospy.init_node("lobby_egress_seed")
    LobbyEgressSeed()
    rospy.spin()
