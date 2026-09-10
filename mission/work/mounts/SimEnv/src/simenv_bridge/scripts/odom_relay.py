#!/usr/bin/env python3
"""
odom_relay.py — 把 SimEnv 的里程计转发成 TARE 需要的话题名/帧。

SimEnv 发布: /Odometry_gazebo  (nav_msgs/Odometry, frame=odom, child=base)
TARE  需要:  /state_estimation_at_scan (nav_msgs/Odometry)

两者消息类型一致，只是话题名不同。SimEnv 中 map->odom 是单位变换
(state_from_gazebo 发布)，因此直接转发并把 frame_id 设为 map 即可，
让 TARE 认为位姿就在 map 系。

用法: rosrun simenv_bridge odom_relay.py
        _input:=/Odometry_gazebo  _output:=/state_estimation_at_scan
"""
import rospy
from nav_msgs.msg import Odometry


def prepare_odom(msg, frame, lock_z=None):
    msg.header.frame_id = frame
    if lock_z is not None:
        msg.pose.pose.position.z = float(lock_z)
    return msg


def main():
    rospy.init_node("odom_relay")
    inp = rospy.get_param("~input", "/Odometry_gazebo")
    out = rospy.get_param("~output", "/state_estimation_at_scan")
    frame = rospy.get_param("~frame", "map")  # TARE 世界系
    lock_z = rospy.get_param("~lock_z", None)
    if lock_z is not None:
        lock_z = float(lock_z)

    pub = rospy.Publisher(out, Odometry, queue_size=10)

    def cb(msg):
        pub.publish(prepare_odom(msg, frame, lock_z))  # map == odom (单位变换)

    rospy.Subscriber(inp, Odometry, cb)
    rospy.loginfo("odom_relay: %s -> %s (frame=%s lock_z=%s)", inp, out, frame, lock_z)
    rospy.spin()


if __name__ == "__main__":
    main()
