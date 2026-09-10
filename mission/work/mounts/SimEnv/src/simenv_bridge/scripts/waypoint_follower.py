#!/usr/bin/env python3
"""
waypoint_follower.py — 跟随 TARE 的航点，输出 A1 RL 控制器可消费的 /cmd_vel。

TARE 发布 /way_point (geometry_msgs/PointStamped, frame=map)。
A1 的 RL 状态机订阅 /cmd_vel (geometry_msgs/Twist):
    linear.x = 前进速度, linear.y = 侧移, angular.z = 转向。

本节点用纯追踪(pure-pursuit)思路:
  1. 用 tf2 把航点从 map 变换到 base 系
  2. 计算朝向误差 dyaw 和距离 dist
  3. linear.x = clip(k_v * dist, 0, v_max); angular.z = clip(k_w * dyaw, -w_max, w_max)
  4. 到点(dist < goal_tol)发零速；超时无航点发零速(原地待命)

注意: 机器狗必须先进入 RL 模式 (junior_ctrl 按键 2->6)，/cmd_vel 才会被执行。
      map==odom==base 链由 state_from_gazebo 发布。

用法: rosrun simenv_bridge waypoint_follower.py
        _waypoint:=/way_point  _cmd_vel:=/cmd_vel
"""
import math
import rospy
import tf2_ros
from geometry_msgs.msg import PointStamped, Twist


def compute_cmd_from_base_point(x, y, k_v, k_w, v_max, w_max, goal_tol):
    dist = math.hypot(x, y)
    if dist <= goal_tol:
        return 0.0, 0.0, True
    dyaw = math.atan2(y, x)
    vx = max(0.0, min(k_v * dist, v_max))
    wz = max(-w_max, min(k_w * dyaw, w_max))
    return vx, wz, False


def yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def main():
    rospy.init_node("waypoint_follower")
    wp_topic = rospy.get_param("~waypoint", "/way_point")
    cmd_topic = rospy.get_param("~cmd_vel", "/cmd_vel")
    target_frame = rospy.get_param("~target_frame", "base")  # A1 基座
    v_max = rospy.get_param("~v_max", 1.0)
    w_max = rospy.get_param("~w_max", 0.6)
    k_v = rospy.get_param("~k_v", 0.8)
    k_w = rospy.get_param("~k_w", 1.2)
    goal_tol = rospy.get_param("~goal_tol", 0.3)
    wp_timeout = rospy.get_param("~wp_timeout", 3.0)  # 无航点超时(秒)

    tfbuf = tf2_ros.Buffer()
    tfl = tf2_ros.TransformListener(tfbuf)
    pub = rospy.Publisher(cmd_topic, Twist, queue_size=1)
    rate = rospy.Rate(20)

    latest_wp = {"pt": None, "t": rospy.Time(0)}

    def on_wp(msg):
        latest_wp["pt"] = msg
        latest_wp["t"] = rospy.Time.now()

    rospy.Subscriber(wp_topic, PointStamped, on_wp, queue_size=5)

    rospy.loginfo("waypoint_follower: %s -> %s (frame=%s)", wp_topic, cmd_topic, target_frame)
    while not rospy.is_shutdown():
        cmd = Twist()
        now = rospy.Time.now()
        if latest_wp["pt"] is not None and (now - latest_wp["t"]).to_sec() < wp_timeout:
            wp = latest_wp["pt"]
            src = wp.header.frame_id if wp.header.frame_id else "map"
            try:
                trans = tfbuf.lookup_transform(target_frame, src, rospy.Time(0), rospy.Duration(0.2))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                rate.sleep(); continue
            yaw = yaw_from_quat(trans.transform.rotation)
            tx = trans.transform.translation.x
            ty = trans.transform.translation.y
            base_x = math.cos(yaw) * wp.point.x - math.sin(yaw) * wp.point.y + tx
            base_y = math.sin(yaw) * wp.point.x + math.cos(yaw) * wp.point.y + ty
            vx, wz, done = compute_cmd_from_base_point(
                base_x, base_y, k_v, k_w, v_max, w_max, goal_tol
            )
            if not done:
                cmd.linear.x = vx
                cmd.angular.z = wz
        pub.publish(cmd)
        rate.sleep()


if __name__ == "__main__":
    main()
