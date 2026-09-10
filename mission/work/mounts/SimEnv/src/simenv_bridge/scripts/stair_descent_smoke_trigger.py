#!/usr/bin/env python3
"""Trigger an isolated descent only after Gazebo truth and odometry exist."""

import json
import math
import os
import tempfile
import time

import rospy
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion


TERMINAL_MARKERS = (
    "FIRST_FLOOR_RETURNED",
    "FAILED",
    "FAILURE",
    "TIMEOUT",
    "ATTITUDE_LOST",
    "ALIGNMENT_LOST",
    "FALL",
)


def atomic_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=".descent-smoke-", suffix=".json", dir=os.path.dirname(path)
    )
    try:
        with os.fdopen(handle, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


class Trigger:
    def __init__(self):
        self.output_dir = os.path.abspath(rospy.get_param("~output_dir"))
        self.token = str(rospy.get_param("~trigger_token", "DESCENT_SMOKE_START"))
        self.delay = float(rospy.get_param("~trigger_delay_sec", 15.0))
        trigger_topic = rospy.get_param(
            "~trigger_topic", "/simenv/descent_smoke_trigger"
        )
        state_topic = rospy.get_param(
            "~state_topic", "/simenv/third_to_first_floor_stair_state"
        )
        odom_topic = rospy.get_param("~odom_topic", "/Odometry_gazebo")

        self.booted_at = time.monotonic()
        self.ready_at = None
        self.last_publish_at = 0.0
        self.trigger_accepted = False
        self.odom_seen = False
        self.truth_pose = None
        self.state = "WAIT_FOR_INPUTS"
        self.observation_written = False

        self.publisher = rospy.Publisher(
            trigger_topic, String, queue_size=1, latch=True
        )
        rospy.Subscriber(odom_topic, Odometry, self.on_odom, queue_size=2)
        rospy.Subscriber(
            "/gazebo/model_states", ModelStates, self.on_models, queue_size=2
        )
        rospy.Subscriber(state_topic, String, self.on_state, queue_size=10)
        rospy.Timer(rospy.Duration(0.25), self.tick)

    def on_odom(self, _message):
        self.odom_seen = True

    def on_models(self, message):
        try:
            index = message.name.index("a1_gazebo")
            pose = message.pose[index]
        except (ValueError, IndexError):
            return
        quaternion = (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        roll, pitch, yaw = euler_from_quaternion(quaternion)
        self.truth_pose = {
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "z": float(pose.position.z),
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
            "upright_z": float(math.cos(roll) * math.cos(pitch)),
        }

    def on_state(self, message):
        self.state = str(message.data).strip()
        if self.state and self.state != "WAIT_F3":
            self.trigger_accepted = True
        if any(marker in self.state for marker in TERMINAL_MARKERS):
            self.write_observation()

    def write_observation(self):
        if self.observation_written:
            return
        atomic_json(
            os.path.join(self.output_dir, "stair_descent_smoke_observation.json"),
            {
                "schema": "stair_descent_physical_smoke_observation_v1",
                "state": self.state,
                "truth_pose": self.truth_pose,
                "trigger_accepted": self.trigger_accepted,
                "wall_time": round(time.time(), 3),
            },
        )
        self.observation_written = True

    def tick(self, _event):
        if self.trigger_accepted:
            return
        if not self.odom_seen or self.truth_pose is None:
            return
        now = time.monotonic()
        if self.ready_at is None:
            self.ready_at = now
            rospy.loginfo(
                "Descent smoke prerequisites ready; trigger in %.1f s.", self.delay
            )
            return
        if now - self.ready_at < self.delay or now - self.last_publish_at < 0.5:
            return
        self.publisher.publish(String(data=self.token))
        self.last_publish_at = now
        rospy.loginfo_throttle(2.0, "Publishing descent smoke trigger.")


def main():
    rospy.init_node("stair_descent_smoke_trigger")
    Trigger()
    rospy.spin()


if __name__ == "__main__":
    main()
