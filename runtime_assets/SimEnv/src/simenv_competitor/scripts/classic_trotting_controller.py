#!/usr/bin/env python3
"""Run junior_ctrl from the workspace root so its model assets resolve."""

import os
import sys
import threading
import time

import rospy
from controller_manager_msgs.srv import ListControllers
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, JointState


class StartupState:
    """Wall-clock startup health; never waits on potentially frozen sim time."""

    def __init__(self):
        self.lock = threading.Lock()
        self.joint_wall = None
        self.imu_wall = None
        self.clock_wall = None
        self.clock_progress_wall = None
        self.clock_value = None

    def joint(self, _message):
        with self.lock:
            self.joint_wall = time.monotonic()

    def imu(self, _message):
        with self.lock:
            self.imu_wall = time.monotonic()

    def clock(self, message):
        now = time.monotonic()
        value = message.clock.to_sec()
        with self.lock:
            self.clock_wall = now
            if self.clock_value is None or value > self.clock_value + 1e-6:
                self.clock_value = value
                self.clock_progress_wall = now

    def snapshot(self):
        with self.lock:
            return (self.joint_wall, self.imu_wall, self.clock_wall,
                    self.clock_progress_wall, self.clock_value)


def wait_for_startup_feedback(timeout, clock_stall_timeout):
    rospy.init_node("classic_controller_startup_gate", anonymous=True,
                    disable_signals=True)
    state = StartupState()
    subscribers = (
        rospy.Subscriber("/a1_gazebo/joint_states", JointState, state.joint,
                         queue_size=1),
        rospy.Subscriber("/trunk_imu", Imu, state.imu, queue_size=1),
        rospy.Subscriber("/clock", Clock, state.clock, queue_size=2),
    )
    started = time.monotonic()
    deadline = started + float(timeout)
    while time.monotonic() < deadline and not rospy.is_shutdown():
        now = time.monotonic()
        joint_wall, imu_wall, clock_wall, progress_wall, clock_value = state.snapshot()
        if (now - started >= float(clock_stall_timeout) and
                (progress_wall is None or
                 now - progress_wall >= float(clock_stall_timeout))):
            raise SystemExit(
                "gazebo_clock_stalled: no simulation progress for {:.1f}s "
                "(last_sim_time={})".format(
                    float(clock_stall_timeout),
                    "none" if clock_value is None else "{:.3f}".format(clock_value)))
        if joint_wall is not None and imu_wall is not None:
            return subscribers
        time.sleep(0.05)
    missing = []
    joint_wall, imu_wall, _, _, _ = state.snapshot()
    if joint_wall is None:
        missing.append("/a1_gazebo/joint_states")
    if imu_wall is None:
        missing.append("/trunk_imu")
    raise SystemExit("controller_state_timeout: missing " + ", ".join(missing))


def wait_for_running_controllers(deadline):
    service = "/a1_gazebo/controller_manager/list_controllers"
    remaining = max(0.1, deadline - time.monotonic())
    try:
        rospy.wait_for_service(service, timeout=remaining)
    except rospy.ROSException:
        raise SystemExit("controller_state_timeout: list_controllers unavailable")
    proxy = rospy.ServiceProxy(service, ListControllers, persistent=False)
    joint_controllers = {
        "{}_{}_controller".format(leg, joint)
        for leg in ("FL", "FR", "RL", "RR")
        for joint in ("hip", "thigh", "calf")
    }
    while time.monotonic() < deadline and not rospy.is_shutdown():
        try:
            response = proxy()
            running = {item.name for item in response.controller
                       if item.state == "running"}
            if joint_controllers <= running:
                return
        except rospy.ServiceException:
            pass
        time.sleep(0.10)
    raise SystemExit("controller_state_timeout: 12 joint controllers not running")

def main():
    workspace = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")
    )
    # Keep the locomotion process in the same isolated catkin profile as the
    # launch files and Python nodes.  The former devel/devel_tools fallback
    # silently selected a binary compiled from /restore/SimEnv when this
    # /restore/0803b/SimEnv workspace was launched.  That made stair behaviour
    # depend on unrelated builds left on disk.  Fail explicitly instead of
    # crossing workspace boundaries when the local controller is not built.
    executable = os.path.join(
        workspace, "devel_0803b", ".private", "unitree_guide",
        "lib", "unitree_guide", "junior_ctrl")
    if not os.path.isfile(executable):
        raise SystemExit(
            "0803b junior_ctrl is not built: {} (run catkin build "
            "unitree_guide --profile relocated_0803b)".format(executable))
    print("Using junior_ctrl: {}".format(executable))
    policy = os.environ.get("UNITREE_RL_POLICY", "").strip()
    if policy:
        resolved_policy = (policy if os.path.isabs(policy)
                           else os.path.join(workspace, policy))
        if not os.path.isfile(resolved_policy):
            # Recover the known legacy launch layout without accepting an
            # arbitrary similarly named model elsewhere on the filesystem.
            canonical_policy = os.path.join(
                workspace, "src", "unitree_guide", "logs",
                os.path.basename(policy))
            if os.path.isfile(canonical_policy):
                print("Corrected legacy RL policy path: {} -> {}".format(
                    policy, canonical_policy))
                resolved_policy = canonical_policy
            else:
                raise SystemExit(
                    "rl_policy_not_found: {} (also checked {})".format(
                        resolved_policy, canonical_policy))
        if not os.access(resolved_policy, os.R_OK):
            raise SystemExit("rl_policy_not_readable: {}".format(resolved_policy))
        os.environ["UNITREE_RL_POLICY"] = resolved_policy
    affinity = os.environ.get("UNITREE_CPU_AFFINITY", "").strip()
    if affinity and hasattr(os, "sched_setaffinity"):
        cores = {int(value) for value in affinity.split(",") if value.strip()}
        os.sched_setaffinity(0, cores)
        print("Pinned classic junior_ctrl to CPU core(s): {}".format(sorted(cores)))
    # roslaunch starts siblings concurrently. Loading Torch before Gazebo has
    # live controllers is occasionally fatal inside the legacy controller.
    # Use one ROS subscriber gate: spawning a new `rostopic` process every two
    # seconds leaked temporary nodes and hid a frozen simulation for minutes.
    print("Waiting for Gazebo A1 joint state and trunk IMU before junior_ctrl")
    startup_timeout = float(os.environ.get("UNITREE_STARTUP_TIMEOUT", "150"))
    clock_stall_timeout = float(os.environ.get(
        "UNITREE_CLOCK_STALL_TIMEOUT", "12.0"))
    deadline = time.monotonic() + startup_timeout
    subscribers = wait_for_startup_feedback(startup_timeout, clock_stall_timeout)
    # A joint-state message can arrive before controller_spawner has switched
    # the twelve effort controllers.  junior_ctrl assumes that every command
    # topic already has a live controller and may segfault if started in that
    # short window, so inspect controller_manager instead of using a fixed
    # sleep.
    wait_for_running_controllers(deadline)
    # The legacy controller also reads the first settled low-level state in its
    # constructor.  Starting immediately after the switch can race Gazebo's
    # spawn/configuration transaction even though all controllers report
    # running, so let the simulated robot settle before constructing it.
    time.sleep(5.0)
    print("A1 sensors and 12 joint controllers are ready; starting classic junior_ctrl")
    for subscriber in subscribers:
        subscriber.unregister()
    os.chdir(workspace)
    os.execv(executable, [executable] + sys.argv[1:])


if __name__ == "__main__":
    main()
