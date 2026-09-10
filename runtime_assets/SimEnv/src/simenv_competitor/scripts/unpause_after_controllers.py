#!/usr/bin/env python3
"""Give ros_control a paused load window, then safely start physics."""

import time

import rospy
from controller_manager_msgs.srv import ListControllers
from std_srvs.srv import Empty


def main():
    rospy.init_node("unpause_after_controllers")
    namespace = rospy.get_param("~controller_namespace", "/a1_gazebo")
    timeout = float(rospy.get_param("~timeout", 120.0))
    load_window = float(rospy.get_param("~load_window_seconds", 2.0))
    list_service = namespace.rstrip("/") + "/controller_manager/list_controllers"
    deadline = time.monotonic() + timeout
    rospy.wait_for_service(list_service, timeout=timeout)
    rospy.wait_for_service("/gazebo/unpause_physics", timeout=timeout)
    list_controllers = rospy.ServiceProxy(list_service, ListControllers)
    unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
    required = {"joint_state_controller"}
    required.update(
        "{}_{}_controller".format(leg, joint)
        for leg in ("FL", "FR", "RL", "RR")
        for joint in ("hip", "thigh", "calf"))
    # controller_manager's switch service waits for an update-cycle boundary,
    # so waiting for state=running while Gazebo is paused deadlocks.  The
    # standard spawner loads all controllers before it invokes switch; give it
    # a bounded wall-clock load window, unpause, then verify the running state.
    time.sleep(max(0.0, load_window))
    unpause()
    rospy.loginfo(
        "Gazebo unpaused after %.1f s paused controller-load window",
        load_window)
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        try:
            response = list_controllers()
            running = {controller.name for controller in response.controller
                       if controller.state == "running"}
            if required <= running:
                rospy.loginfo(
                    "All %d A1 controllers reached running state",
                    len(required))
                return
        except (rospy.ServiceException, rospy.exceptions.ROSInterruptException) as error:
            if rospy.is_shutdown():
                return
            rospy.logwarn_throttle(
                5.0, "Waiting for running A1 controllers: %s", str(error))
        time.sleep(0.05)
    raise SystemExit("controller_startup_timeout: controllers not running")


if __name__ == "__main__":
    main()
