#!/usr/bin/env python3
"""Competition-oriented TARE bridge.

Consumes ONLY allowed sensors via an external LIO:
  - /Odometry
  - /cloud_registered

Does NOT subscribe to /Odometry_gazebo or other referee topics.
Aligns the LIO origin to competition robot_start from team_scene_info.json
(allowed public file), planar-stabilizes pose, and rejects impossible jumps.

When FAST-LIO stalls or freezes in place (common in Gazebo Mid-360),
integrates /cmd_vel so exploration can continue with a smooth
/state_estimation stream. Stagnant LIO poses are not allowed to overwrite
the dead-reckoned pose while motion commands are active.
"""

import copy
import json
import math
import os
import threading

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud2
from std_msgs.msg import Bool


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw):
    return (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5))


def _load_robot_start():
    root = rospy.get_param(
        "~workspace_root",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),
    )
    scene_path = rospy.get_param(
        "~team_scene_info",
        os.path.join(root, "generated_building", "team_scene_info.json"),
    )
    with open(scene_path, encoding="utf-8") as stream:
        scene = json.load(stream)
    start = scene.get("robot_start", {})
    return (
        float(start.get("x", 0.0)),
        float(start.get("y", 2.0)),
        float(start.get("yaw", 1.5708)),
    )


class CompliantTareBridge:
    def __init__(self):
        self._lock = threading.Lock()
        self._latest_odom = None
        self._latest_cloud = None
        self._last_output = None
        self._good_xy = None
        self._good_yaw = 0.0
        self._reject_count = 0
        self._lio_origin = None
        self._spawn_x, self._spawn_y, self._spawn_yaw = _load_robot_start()
        self._odom_topic = rospy.get_param("~lio_odom_topic", "/Odometry")
        self._cloud_topic = rospy.get_param("~lio_cloud_topic", "/cloud_registered")
        self._flip_axes = bool(rospy.get_param("~flip_loam_axes", False))
        self._forward_cmd = bool(rospy.get_param("~forward_cmd_vel", True))
        self._planar = bool(rospy.get_param("~planar_stabilize", True))
        self._lock_z = float(rospy.get_param("~lock_z", 0.28))
        self._max_jump = float(rospy.get_param("~max_pose_jump_m", 0.8))
        self._max_yaw_jump = float(rospy.get_param("~max_yaw_jump_rad", 0.8))
        self._use_imu_yaw = bool(rospy.get_param("~use_imu_yaw", True))
        # Reject LIVO scale blowups / Y-flips that arrive as many small steps.
        self._max_lio_speed = float(rospy.get_param("~max_lio_speed_mps", 1.15))
        self._dr_enable = bool(rospy.get_param("~cmd_vel_dead_reckon", True))
        self._dr_stale = float(rospy.get_param("~dr_stale_seconds", 0.70))
        self._dr_max_speed = float(rospy.get_param("~dr_max_speed", 0.32))
        # Cap continuous dead-reckon so sparse LIO cannot invent wall-crossing paths.
        self._dr_max_burst = float(rospy.get_param("~dr_max_burst_seconds", 1.2))
        self._base_frame = str(rospy.get_param("~base_frame", "base"))
        # Prefer LIO when it moves; only DR to cover short stalls.
        self._dr_lio_move_eps = float(rospy.get_param("~dr_lio_move_eps", 0.02))
        self._dr_active = False
        self._dr_burst_start = None
        self._last_lio_time = rospy.Time(0)
        self._last_lio_move_time = rospy.Time(0)
        self._last_accept_time = None
        self._last_lio_raw_xy = None
        self._last_pose_accepted = False
        self._imu_yaw = None
        self._imu_yaw_offset = None
        self._last_tick_time = rospy.Time.now()
        self._last_cmd = Twist()
        self._tf_broadcaster = tf2_ros.TransformBroadcaster()

        self._state_pub = rospy.Publisher("/state_estimation", Odometry, queue_size=5)
        self._scan_state_pub = rospy.Publisher(
            "/state_estimation_at_scan", Odometry, queue_size=5
        )
        self._scan_pub = rospy.Publisher("/registered_scan", PointCloud2, queue_size=2)
        self._dr_pub = rospy.Publisher(
            "/simenv/dead_reckon_active", Bool, queue_size=1, latch=True
        )
        self._dr_pub.publish(Bool(data=False))
        self._last_dr_flag = False
        self._cmd_pub = None
        if self._forward_cmd:
            self._cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=5)

        rospy.Subscriber(self._odom_topic, Odometry, self._on_odom, queue_size=10)
        rospy.Subscriber(self._cloud_topic, PointCloud2, self._on_cloud, queue_size=2)
        rospy.Subscriber("/trunk_imu", Imu, self._on_imu, queue_size=20)
        rospy.Subscriber("/cmd_vel", Twist, self._on_cmd_vel, queue_size=10)
        if self._forward_cmd:
            rospy.Subscriber(
                "/tare/cmd_vel_stamped", TwistStamped, self._on_cmd, queue_size=5
            )
        rospy.Timer(rospy.Duration(0.05), self._on_timer)
        rospy.loginfo(
            "Compliant TARE bridge ready (odom=%s cloud=%s planar=%s dr=%s "
            "spawn=(%.2f, %.2f, %.2f))",
            self._odom_topic,
            self._cloud_topic,
            self._planar,
            self._dr_enable,
            self._spawn_x,
            self._spawn_y,
            self._spawn_yaw,
        )

    def _to_world(self, x, y, yaw):
        if self._lio_origin is None:
            self._lio_origin = (x, y, yaw)
        ox, oy, oyaw = self._lio_origin
        dx, dy = x - ox, y - oy
        alignment_yaw = self._spawn_yaw - oyaw
        cos_s, sin_s = math.cos(alignment_yaw), math.sin(alignment_yaw)
        wx = self._spawn_x + cos_s * dx - sin_s * dy
        wy = self._spawn_y + sin_s * dx + cos_s * dy
        wyaw = yaw + alignment_yaw
        wyaw = math.atan2(math.sin(wyaw), math.cos(wyaw))
        return wx, wy, wyaw

    def _accept_pose(self, x, y, yaw):
        """Accept physically plausible FAST-LIO motion in one static SE(2)."""
        now = rospy.Time.now()
        if not all(math.isfinite(value) for value in (x, y, yaw)):
            self._reject_count += 1
            return False
        if self._good_xy is None:
            self._good_xy = (x, y)
            self._good_yaw = yaw
            self._last_accept_time = now
            return True
        jump = math.hypot(x - self._good_xy[0], y - self._good_xy[1])
        dt = 0.05
        if self._last_accept_time is not None:
            dt = max(0.02, (now - self._last_accept_time).to_sec())
        implied_speed = jump / dt
        # The old command-direction and in-place-turn gates repeatedly moved
        # the map translation for valid centimetre innovations.  Only a hard
        # nonphysical step is rejected now; turns are treated exactly like any
        # other FAST-LIO motion.
        physical_step = max(0.15, (self._max_lio_speed + 0.40) * dt)
        if jump > self._max_jump or jump > physical_step:
            self._reject_count += 1
            if self._reject_count % 20 == 1:
                rospy.logwarn_throttle(
                    2.0,
                    "Rejecting nonphysical LIO step=%.2fm speed=%.2fm/s "
                    "limit=%.2fm (keeping last good)",
                    jump,
                    implied_speed,
                    physical_step,
                )
            return False
        dyaw = abs(
            math.atan2(math.sin(yaw - self._good_yaw), math.cos(yaw - self._good_yaw))
        )
        self._good_xy = (x, y)
        if dyaw <= self._max_yaw_jump:
            self._good_yaw = yaw
        self._last_accept_time = now
        return True

    def _make_output(self, message, x, y, yaw, twist=None):
        output = copy.deepcopy(message)
        output.header.frame_id = "map"
        output.child_frame_id = self._base_frame
        output.header.stamp = rospy.Time.now()
        output.pose.pose.position.x = x
        output.pose.pose.position.y = y
        if self._planar:
            qx, qy, qz, qw = quaternion_from_yaw(yaw)
            output.pose.pose.orientation.x = qx
            output.pose.pose.orientation.y = qy
            output.pose.pose.orientation.z = qz
            output.pose.pose.orientation.w = qw
            output.pose.pose.position.z = self._lock_z
            output.twist.twist.linear.z = 0.0
            output.twist.twist.angular.x = 0.0
            output.twist.twist.angular.y = 0.0
        if twist is not None:
            output.twist.twist = copy.deepcopy(twist)
        return output

    def _raw_world_pose(self, message):
        x = float(message.pose.pose.position.x)
        y = float(message.pose.pose.position.y)
        if self._flip_axes:
            x, y = y, x
        yaw = yaw_from_quaternion(message.pose.pose.orientation)
        return self._to_world(x, y, yaw)

    def _stabilize(self, message):
        x, y, yaw = self._raw_world_pose(message)
        if self._use_imu_yaw and self._imu_yaw is not None:
            # FAST-LIO supplies metric translation; the body IMU supplies the
            # achieved quadruped heading.  Using LIO yaw here under-corrected
            # a repeatable 0.08 rad gait bias and drove the robot into a wall
            # before the first door band.
            yaw = self._imu_yaw
        accepted = self._accept_pose(x, y, yaw)
        self._last_pose_accepted = accepted
        if not accepted:
            x, y = self._good_xy
            yaw = self._good_yaw
            twist = Twist()
        else:
            twist = message.twist.twist
        return self._make_output(message, x, y, yaw, twist)

    def _publish_state(self, output):
        self._last_output = output
        self._state_pub.publish(output)
        transform = TransformStamped()
        transform.header.stamp = output.header.stamp
        transform.header.frame_id = "map"
        transform.child_frame_id = self._base_frame
        transform.transform.translation.x = output.pose.pose.position.x
        transform.transform.translation.y = output.pose.pose.position.y
        transform.transform.translation.z = output.pose.pose.position.z
        transform.transform.rotation = copy.deepcopy(output.pose.pose.orientation)
        self._tf_broadcaster.sendTransform(transform)

    def _cmd_active(self):
        return abs(self._last_cmd.linear.x) + abs(self._last_cmd.angular.z) > 1e-3

    def _should_dead_reckon(self, now):
        if not self._dr_enable or self._latest_odom is None or not self._cmd_active():
            return False
        lio_stale = (now - self._last_lio_time).to_sec()
        move_stale = (now - self._last_lio_move_time).to_sec()
        if not (lio_stale >= self._dr_stale or move_stale >= self._dr_stale):
            return False
        if self._dr_burst_start is None:
            return True
        return (now - self._dr_burst_start).to_sec() <= self._dr_max_burst

    def _on_odom(self, message):
        now = rospy.Time.now()
        with self._lock:
            self._latest_odom = message
            self._last_lio_time = now
            wx, wy, wyaw = self._raw_world_pose(message)
            moved = 0.0
            had_raw_pose = self._last_lio_raw_xy is not None
            if had_raw_pose:
                moved = math.hypot(
                    wx - self._last_lio_raw_xy[0], wy - self._last_lio_raw_xy[1]
                )
            output = self._stabilize(message)

            # Only accepted LIO motion is evidence of estimator health.  Raw
            # rejected jumps previously refreshed this clock forever and
            # prevented dead reckoning exactly when it was needed.
            if self._last_pose_accepted:
                self._last_lio_raw_xy = (wx, wy)
                if not had_raw_pose or moved >= self._dr_lio_move_eps:
                    self._last_lio_move_time = now
                self._dr_active = False
                self._dr_burst_start = None
            elif self._dr_enable and self._cmd_active() and self._good_xy is not None:
                self._dr_active = True
                return
        self._publish_state(output)

    def _on_cloud(self, message):
        with self._lock:
            self._latest_cloud = message
            output = self._last_output
            origin = self._lio_origin
        if output is None or origin is None:
            return
        cloud = copy.deepcopy(message)
        fields = {field.name: field for field in cloud.fields}
        if all(name in fields for name in ("x", "y", "z")) and cloud.point_step > 0:
            # Registered points are expressed in FAST-LIO's initial world
            # frame.  Rewriting frame_id alone is wrong: apply the exact same
            # alignment used by _to_world while preserving every extra field.
            mutable = bytearray(cloud.data)
            count = int(cloud.width) * int(cloud.height)
            endian = ">f4" if cloud.is_bigendian else "<f4"
            x_view = np.ndarray(
                (count,), dtype=endian, buffer=mutable,
                offset=fields["x"].offset, strides=(cloud.point_step,),
            )
            y_view = np.ndarray(
                (count,), dtype=endian, buffer=mutable,
                offset=fields["y"].offset, strides=(cloud.point_step,),
            )
            raw_x = x_view.copy()
            raw_y = y_view.copy()
            if self._flip_axes:
                raw_x, raw_y = raw_y, raw_x
            ox, oy, oyaw = origin
            alignment_yaw = self._spawn_yaw - oyaw
            cosine, sine = math.cos(alignment_yaw), math.sin(alignment_yaw)
            dx, dy = raw_x - ox, raw_y - oy
            x_view[:] = self._spawn_x + cosine * dx - sine * dy
            y_view[:] = self._spawn_y + sine * dx + cosine * dy
            cloud.data = bytes(mutable)
        cloud.header.frame_id = "map"
        self._scan_pub.publish(cloud)
        scan_state = copy.deepcopy(output)
        scan_state.header.stamp = message.header.stamp
        self._scan_state_pub.publish(scan_state)

    def _on_cmd_vel(self, message):
        with self._lock:
            self._last_cmd = copy.deepcopy(message)

    def _on_imu(self, message):
        raw = yaw_from_quaternion(message.orientation)
        with self._lock:
            if self._imu_yaw_offset is None:
                self._imu_yaw_offset = math.atan2(
                    math.sin(self._spawn_yaw - raw),
                    math.cos(self._spawn_yaw - raw),
                )
            self._imu_yaw = math.atan2(
                math.sin(raw + self._imu_yaw_offset),
                math.cos(raw + self._imu_yaw_offset),
            )

    def _integrate_cmd_vel(self, dt):
        if self._good_xy is None or dt <= 0.0:
            return None
        dt = min(dt, 0.2)
        vx = max(
            -self._dr_max_speed,
            min(self._dr_max_speed, float(self._last_cmd.linear.x)),
        )
        vy = max(
            -self._dr_max_speed,
            min(self._dr_max_speed, float(self._last_cmd.linear.y)),
        )
        wz = float(self._last_cmd.angular.z)
        x, y = self._good_xy
        # Commanded yaw rate is not achieved instantaneously by the RL gait;
        # integrating it over a 90/180-degree sweep caused several metres of
        # lateral DR error. The onboard IMU supplies the achieved heading.
        yaw = self._imu_yaw if self._imu_yaw is not None else self._good_yaw
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        x += (cos_y * vx - sin_y * vy) * dt
        y += (sin_y * vx + cos_y * vy) * dt
        if self._imu_yaw is None:
            yaw = math.atan2(math.sin(yaw + wz * dt), math.cos(yaw + wz * dt))
        self._good_xy = (x, y)
        self._good_yaw = yaw
        twist = Twist()
        twist.linear.x = vx
        twist.linear.y = vy
        twist.angular.z = wz
        template = self._latest_odom if self._latest_odom is not None else Odometry()
        return self._make_output(template, x, y, yaw, twist)

    def _on_timer(self, _event):
        now = rospy.Time.now()
        with self._lock:
            dt = (now - self._last_tick_time).to_sec()
            self._last_tick_time = now
            move_stale = (now - self._last_lio_move_time).to_sec()
            last_output = self._last_output
            do_dr = self._should_dead_reckon(now)

        if dt <= 0.0:
            return

        output = None
        if do_dr:
            with self._lock:
                if self._dr_burst_start is None:
                    self._dr_burst_start = now
                burst = (now - self._dr_burst_start).to_sec()
                if burst <= self._dr_max_burst:
                    self._dr_active = True
                    output = self._integrate_cmd_vel(dt)
                else:
                    self._dr_active = False
                    rospy.logwarn_throttle(
                        5.0,
                        "Dead-reckon burst capped at %.1fs (LIO stagnant %.1fs)",
                        self._dr_max_burst,
                        move_stale,
                    )
            if output is not None and move_stale > 1.0:
                rospy.logwarn_throttle(
                    5.0,
                    "LIO position stagnant %.1fs; dead-reckoning from /cmd_vel",
                    move_stale,
                )
        else:
            with self._lock:
                self._dr_active = False
                self._dr_burst_start = None

        if output is None and last_output is not None:
            output = copy.deepcopy(last_output)
            output.header.stamp = now

        if output is not None:
            self._publish_state(output)
        with self._lock:
            dr_flag = self._dr_active
        if dr_flag != self._last_dr_flag:
            self._last_dr_flag = dr_flag
            self._dr_pub.publish(Bool(data=dr_flag))

    def _on_cmd(self, message):
        if self._cmd_pub is None:
            return
        command = copy.deepcopy(message.twist)
        max_linear = float(rospy.get_param("~max_linear_speed", 0.8))
        max_yaw = float(rospy.get_param("~max_yaw_rate", 0.6))
        command.linear.x = max(-max_linear, min(max_linear, command.linear.x))
        command.linear.y = max(-max_linear, min(max_linear, command.linear.y))
        command.angular.z = max(-max_yaw, min(max_yaw, command.angular.z))
        self._cmd_pub.publish(command)


if __name__ == "__main__":
    rospy.init_node("compliant_tare_bridge")
    CompliantTareBridge()
    rospy.spin()
