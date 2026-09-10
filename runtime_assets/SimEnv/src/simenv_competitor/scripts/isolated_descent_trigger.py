#!/usr/bin/env python3
"""Gate the isolated F3->F1 descent on a physically valid F3 spawn."""

import json
import math
import os
import time

import rospy
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Bool, String
from tf.transformations import euler_from_quaternion


def pose_is_ready(pose, attitude, locomotion_ready, bounds, tilt_limit):
    if pose is None or attitude is None or not locomotion_ready:
        return False
    x, y, z, _yaw = pose
    roll, pitch = attitude
    return (bounds[0] <= x <= bounds[1] and
            bounds[2] <= y <= bounds[3] and
            bounds[4] <= z <= bounds[5] and
            abs(roll) <= tilt_limit and abs(pitch) <= tilt_limit)


class IsolatedDescentTrigger:
    def __init__(self):
        self.output_dir = os.path.abspath(rospy.get_param("~output_dir"))
        os.makedirs(os.path.join(self.output_dir, "logs"), exist_ok=True)
        self.log_path = os.path.join(
            self.output_dir, "logs", "isolated_descent_trigger.json")
        self.model_name = rospy.get_param("~model_name", "a1_gazebo")
        self.bounds = (
            float(rospy.get_param("~minimum_x", -2.05)),
            float(rospy.get_param("~maximum_x", -1.83)),
            float(rospy.get_param("~minimum_y", 1.10)),
            float(rospy.get_param("~maximum_y", 1.34)),
            float(rospy.get_param("~minimum_z", 5.50)),
            float(rospy.get_param("~maximum_z", 5.68)),
        )
        self.tilt_limit = float(rospy.get_param("~maximum_tilt_rad", 0.25))
        self.stable_seconds = float(rospy.get_param(
            "~stable_ros_seconds", 1.0))
        self.ready_timeout = float(rospy.get_param(
            "~ready_ros_timeout_seconds", 45.0))
        self.wall_timeout = float(rospy.get_param(
            "~ready_wall_timeout_seconds", 180.0))
        self.terminal_grace = float(rospy.get_param(
            "~terminal_artifact_wall_timeout_seconds", 95.0))
        self.trigger_token = rospy.get_param(
            "~trigger_token", "THIRD_FLOOR_STAIR_RETURN_READY")
        self.failure_token = rospy.get_param(
            "~failure_token", "ISOLATED_F3_SPAWN_NOT_READY")
        self.trigger_topic = rospy.get_param(
            "~trigger_topic", "/simenv/third_floor_state")
        self.descent_state_topic = rospy.get_param(
            "~descent_state_topic",
            "/simenv/third_to_first_floor_stair_state")
        self.trigger_pub = rospy.Publisher(
            self.trigger_topic, String, queue_size=1, latch=True)
        self.state_pub = rospy.Publisher(
            "/simenv/isolated_descent_trigger_state", String,
            queue_size=1, latch=True)
        self.pose = None
        self.attitude = None
        self.locomotion_ready = False
        self.first_ros_time = None
        self.stable_since = None
        self.stable_ros_time = None
        self.trigger_ros_time = None
        self.triggered = False
        self.terminal_state = None
        self.terminal_seen_wall = None
        self.started_wall = time.monotonic()
        self.trace = []
        rospy.Subscriber("/gazebo/model_states", ModelStates,
                         self._on_models, queue_size=2)
        rospy.Subscriber("/locomotion_ready", Bool,
                         self._on_locomotion_ready, queue_size=2)
        rospy.Subscriber(self.descent_state_topic, String,
                         self._on_descent_state, queue_size=20)
        rospy.Timer(rospy.Duration(0.05), self._tick)
        rospy.on_shutdown(self._write)

    def _on_models(self, message):
        try:
            index = message.name.index(self.model_name)
            pose = message.pose[index]
        except (ValueError, IndexError):
            return
        q = pose.orientation
        roll, pitch, _ = euler_from_quaternion((q.x, q.y, q.z, q.w))
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))
        self.pose = (pose.position.x, pose.position.y, pose.position.z, yaw)
        self.attitude = (roll, pitch)

    def _on_locomotion_ready(self, message):
        self.locomotion_ready = bool(message.data)

    @staticmethod
    def _is_terminal(token):
        if token in ("FIRST_FLOOR_START_RETURNED",
                     "ISOLATED_F3_SPAWN_NOT_READY"):
            return True
        return any(marker in token for marker in (
            "_TIMEOUT", "_FAILED", "_ABORT", "_STALLED", "FALL_DETECTED",
            "ALIGNMENT_LOST", "GAIN_UP_DETECTED",
            "NO_DROP_SEAM_FAILED", "POLICY_MISSING"))

    def _on_descent_state(self, message):
        token = str(message.data).strip()
        self.trace.append({"ros_time": rospy.Time.now().to_sec(),
                           "state": token})
        if self._is_terminal(token) and self.terminal_state is None:
            self.terminal_state = token
            self.terminal_seen_wall = time.monotonic()
            self._write()

    def _snapshot(self, phase):
        return {
            "phase": phase,
            "ros_time": round(rospy.Time.now().to_sec(), 3),
            "wall_time": round(time.time(), 3),
            "truth_pose": list(self.pose) if self.pose is not None else None,
            "truth_attitude": (list(self.attitude)
                               if self.attitude is not None else None),
            "locomotion_ready": self.locomotion_ready,
        }

    def _write(self):
        payload = self._snapshot(
            self.terminal_state or
            ("TRIGGERED" if self.triggered else "WAITING_FOR_STABLE_SPAWN"))
        payload.update({
            "stable_ros_time": self.stable_ros_time,
            "trigger_ros_time": self.trigger_ros_time,
            "trigger_publish_count": 1 if self.triggered else 0,
            "terminal_state": self.terminal_state,
            "bounds": list(self.bounds),
            "trace": self.trace,
        })
        try:
            with open(self.log_path, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
        except OSError as error:
            rospy.logerr("Cannot write isolated descent trigger log: %s",
                         error)

    def _fail_spawn(self):
        if self.terminal_state is not None:
            return
        self.terminal_state = self.failure_token
        self.terminal_seen_wall = time.monotonic()
        self.state_pub.publish(String(data=self.failure_token))
        self.trigger_pub.publish(String(data=self.failure_token))
        self._write()
        rospy.logerr("Isolated F3 spawn did not become physically ready.")

    def _artifacts_ready(self):
        return (os.path.isfile(os.path.join(
                    self.output_dir, "logs", "stair_descent.json")) and
                os.path.isfile(os.path.join(
                    self.output_dir, "visualization", "roundtrip_return",
                    "30_third_floor_to_first_floor_start_truth_return.json")))

    def _tick(self, _event):
        now_ros = rospy.Time.now().to_sec()
        if now_ros > 0.0 and self.first_ros_time is None:
            self.first_ros_time = now_ros
        if self.terminal_state is not None:
            if (self._artifacts_ready() or
                    time.monotonic() - self.terminal_seen_wall >=
                    self.terminal_grace):
                self._write()
                rospy.signal_shutdown("isolated_descent_terminal")
            return
        if self.triggered:
            return
        ready = pose_is_ready(self.pose, self.attitude,
                              self.locomotion_ready, self.bounds,
                              self.tilt_limit)
        if ready:
            if self.stable_since is None:
                self.stable_since = now_ros
            if now_ros - self.stable_since >= self.stable_seconds:
                self.stable_ros_time = round(now_ros, 3)
                self.trigger_ros_time = round(now_ros, 3)
                self.triggered = True
                self.state_pub.publish(String(data="ISOLATED_F3_SPAWN_READY"))
                self.trigger_pub.publish(String(data=self.trigger_token))
                self._write()
                rospy.loginfo("Published isolated descent trigger once at "
                              "ROS %.3f.", now_ros)
                return
        else:
            self.stable_since = None
        ros_elapsed = (now_ros - self.first_ros_time
                       if self.first_ros_time is not None else 0.0)
        if (ros_elapsed >= self.ready_timeout or
                time.monotonic() - self.started_wall >= self.wall_timeout):
            self._fail_spawn()


def main():
    rospy.init_node("isolated_descent_trigger")
    IsolatedDescentTrigger()
    rospy.spin()


if __name__ == "__main__":
    main()
