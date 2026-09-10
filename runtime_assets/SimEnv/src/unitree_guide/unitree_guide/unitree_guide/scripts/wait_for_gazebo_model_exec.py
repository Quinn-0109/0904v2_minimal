#!/usr/bin/env python3
"""Exec a roslaunch child only after its Gazebo model really exists."""

import os
import sys
import time

import rospy
from gazebo_msgs.srv import GetModelState


def main():
    model_name = os.environ.get("WAIT_FOR_GAZEBO_MODEL", "a1_gazebo")
    timeout = float(os.environ.get("WAIT_FOR_GAZEBO_MODEL_TIMEOUT", "150"))
    settle = float(os.environ.get("WAIT_FOR_GAZEBO_MODEL_SETTLE", "1.0"))
    sim_settle = float(os.environ.get(
        "WAIT_FOR_GAZEBO_MODEL_SIM_SETTLE", "0.30"))
    deadline = time.monotonic() + timeout
    rospy.init_node("wait_for_gazebo_model_exec", anonymous=True, disable_signals=True)
    rospy.wait_for_service("/gazebo/get_model_state", timeout=timeout)
    get_model_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
    model_ready_wall = None
    model_ready_sim = None
    while time.monotonic() < deadline and not rospy.is_shutdown():
        try:
            if get_model_state(model_name, "world").success:
                now_wall = time.monotonic()
                now_sim = rospy.Time.now().to_sec()
                if model_ready_wall is None:
                    model_ready_wall = now_wall
                    model_ready_sim = now_sim
                wall_ready = now_wall - model_ready_wall >= settle
                sim_ready = (sim_settle <= 0.0 or
                             now_sim - model_ready_sim >= sim_settle)
                # A model can be reported by get_model_state before all of its
                # Gazebo plugins have completed initialization.  Loading an
                # effort controller in that window can block Gazebo's update
                # thread permanently.  Wall time alone is not sufficient when
                # a cold start has a very low RTF, so also require actual
                # simulation-clock progress before invoking the spawner.
                if wall_ready and sim_ready:
                    os.execvp(sys.argv[1], sys.argv[1:])
            else:
                model_ready_wall = None
                model_ready_sim = None
        except rospy.ServiceException:
            model_ready_wall = None
            model_ready_sim = None
        time.sleep(0.1)
    raise SystemExit("Gazebo model did not become ready: " + model_name)


if __name__ == "__main__":
    main()
