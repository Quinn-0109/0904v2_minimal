#!/usr/bin/env python3
"""Debug-only bridge for the isolated F1-to-stair integration test.

It converts Gazebo model states to the same /Odometry contract consumed by
the online stair manager and emits the normal F1 handoff state.  Production
never launches this node: FAST-LIO remains the sole odometry provider there.
"""
import rospy
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry
from std_msgs.msg import String


class Bridge:
    def __init__(self):
        self.robot = rospy.get_param("~robot_model", "a1_gazebo")
        self.sent = False
        self.odom = rospy.Publisher("/Odometry", Odometry, queue_size=5)
        self.state = rospy.Publisher("/simenv/baseline_state", String,
                                     queue_size=1, latch=True)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.on_models,
                         queue_size=5)

    def on_models(self, message):
        try:
            index = message.name.index(self.robot)
        except ValueError:
            return
        output = Odometry()
        output.header.stamp = rospy.Time.now()
        output.header.frame_id = "map"
        output.child_frame_id = "base"
        output.pose.pose = message.pose[index]
        output.twist.twist = message.twist[index]
        self.odom.publish(output)
        if not self.sent:
            self.state.publish(String(data="STAIR_WAIT_ZONE|debug_f1_handoff"))
            self.sent = True
            rospy.loginfo("[F1-STAIR-TEST] Published STAIR_WAIT_ZONE")


if __name__ == "__main__":
    rospy.init_node("f1_stair_test_odom_bridge")
    Bridge()
    rospy.spin()
