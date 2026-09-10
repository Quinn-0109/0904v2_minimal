#!/usr/bin/env python3
"""
Deterministically move the simulated A1 model from /cmd_vel.

This is a reproduction/debug helper for mapping. The learned A1 controller can
respond too slowly for short mapping runs on this server, so this node applies
the commanded planar velocity directly to Gazebo's model pose while keeping the
sensor rig attached to the robot model.
"""

import math
import threading

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion, quaternion_from_euler


def make_next_state(model_name, pose, vx, vy, wz, dt, command_frame, min_z, lock_z=None):
    q = pose.orientation
    yaw = euler_from_quaternion((q.x, q.y, q.z, q.w))[2]
    if command_frame == "world":
        dx = vx * dt
        dy = vy * dt
    else:
        dx = (math.cos(yaw) * vx - math.sin(yaw) * vy) * dt
        dy = (math.sin(yaw) * vx + math.cos(yaw) * vy) * dt
    yaw += wz * dt

    new_state = ModelState()
    new_state.model_name = model_name
    new_state.reference_frame = "world"
    new_state.pose = pose
    new_state.pose.position.x += dx
    new_state.pose.position.y += dy
    if lock_z is None:
        new_state.pose.position.z = max(min_z, pose.position.z)
    else:
        new_state.pose.position.z = max(min_z, float(lock_z))
    quat = quaternion_from_euler(0.0, 0.0, yaw)
    new_state.pose.orientation.x = quat[0]
    new_state.pose.orientation.y = quat[1]
    new_state.pose.orientation.z = quat[2]
    new_state.pose.orientation.w = quat[3]
    new_state.twist.linear.x = vx
    new_state.twist.linear.y = vy
    new_state.twist.angular.z = wz
    return new_state


class CmdVelModelDriver:
    def __init__(self):
        self.model_name = rospy.get_param("~model_name", "a1_gazebo")
        self.cmd_topic = rospy.get_param("~cmd_vel", "/cmd_vel")
        self.rate_hz = float(rospy.get_param("~rate", 20.0))
        self.max_vx = float(rospy.get_param("~max_vx", 1.0))
        self.max_vy = float(rospy.get_param("~max_vy", 1.0))
        self.max_wz = float(rospy.get_param("~max_wz", 0.60))
        self.timeout = float(rospy.get_param("~cmd_timeout", 0.5))
        self.min_z = float(rospy.get_param("~min_z", 0.35))
        self.lock_z = rospy.get_param("~lock_z", 0.6)
        if self.lock_z is not None:
            self.lock_z = float(self.lock_z)
        self.command_frame = rospy.get_param("~command_frame", "body")
        self.use_topic = bool(rospy.get_param("~use_topic", True))

        self.lock = threading.Lock()
        self.cmd = Twist()
        self.cmd_stamp = rospy.Time(0)

        rospy.wait_for_service("/gazebo/get_model_state")
        if not self.use_topic:
            rospy.wait_for_service("/gazebo/set_model_state")
        self.get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        self.set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState) if not self.use_topic else None
        self.state_pub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1) if self.use_topic else None
        rospy.Subscriber(self.cmd_topic, Twist, self.on_cmd, queue_size=1)

    def on_cmd(self, msg):
        with self.lock:
            self.cmd = msg
            self.cmd_stamp = rospy.Time.now()

    def clipped_cmd(self):
        with self.lock:
            age = (rospy.Time.now() - self.cmd_stamp).to_sec()
            cmd = self.cmd
        if age > self.timeout:
            return 0.0, 0.0, 0.0
        vx = max(-self.max_vx, min(self.max_vx, cmd.linear.x))
        vy = max(-self.max_vy, min(self.max_vy, cmd.linear.y))
        wz = max(-self.max_wz, min(self.max_wz, cmd.angular.z))
        return vx, vy, wz

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        last = rospy.Time.now()
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            dt = max(0.0, min(0.2, (now - last).to_sec()))
            last = now

            vx, vy, wz = self.clipped_cmd()
            state = self.get_state(self.model_name, "")
            if not state.success:
                rate.sleep()
                continue

            new_state = make_next_state(
                self.model_name, state.pose, vx, vy, wz, dt,
                self.command_frame, self.min_z, self.lock_z,
            )
            if self.use_topic:
                self.state_pub.publish(new_state)
            else:
                try:
                    self.set_state(new_state)
                except rospy.ServiceException as exc:
                    rospy.logwarn_throttle(2.0, "cmd_vel_to_model_state: set_model_state failed: %s", exc)
            rate.sleep()


def main():
    rospy.init_node("cmd_vel_to_model_state")
    driver = CmdVelModelDriver()
    rospy.loginfo("cmd_vel_to_model_state: %s -> %s", driver.cmd_topic, driver.model_name)
    driver.run()


if __name__ == "__main__":
    main()
