#!/usr/bin/env python3
"""Record /state_estimation (or Odometry_gazebo) to a trajectory JSON file."""

import json
import math
import os
import threading

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class TrajectoryRecorder:
    def __init__(self):
        self._lock = threading.Lock()
        self._poses = []
        self._gt_poses = []
        self._stopped = False
        self._min_spacing = float(rospy.get_param("~min_spacing", 0.08))
        self._output_file = rospy.get_param(
            "~output_file",
            os.path.join(os.getcwd(), "results", "exploration_trajectory.json"),
        )
        self._gt_output_file = rospy.get_param("~gt_output_file", "")
        topic = rospy.get_param("~odom_topic", "/state_estimation")
        gt_topic = rospy.get_param("~gt_odom_topic", "")
        rospy.Subscriber(topic, Odometry, self._on_odom, queue_size=20)
        if gt_topic and self._gt_output_file:
            rospy.Subscriber(gt_topic, Odometry, self._on_gt_odom, queue_size=20)
            rospy.loginfo(
                "Also recording GT %s -> %s", gt_topic, self._gt_output_file
            )
        rospy.Subscriber(
            "/simenv/mission_complete", Bool, self._on_complete, queue_size=1
        )
        rospy.on_shutdown(self._flush)
        rospy.Timer(rospy.Duration(5.0), lambda _event: self._flush())
        rospy.loginfo("Recording trajectory from %s -> %s", topic, self._output_file)

    def _append_pose(self, bucket, message):
        position = message.pose.pose.position
        yaw = yaw_from_quaternion(message.pose.pose.orientation)
        stamp = message.header.stamp.to_sec()
        sample = {
            "t": round(stamp, 3),
            "x": round(position.x, 4),
            "y": round(position.y, 4),
            "z": round(position.z, 4),
            "yaw": round(yaw, 4),
        }
        with self._lock:
            if self._stopped:
                return
            if bucket:
                previous = bucket[-1]
                distance = math.hypot(
                    sample["x"] - previous["x"], sample["y"] - previous["y"]
                )
                if (
                    distance < self._min_spacing
                    and abs(sample["yaw"] - previous["yaw"]) < 0.15
                ):
                    return
            if not bucket:
                sample["t0"] = sample["t"]
            bucket.append(sample)

    def _on_odom(self, message):
        self._append_pose(self._poses, message)

    def _on_gt_odom(self, message):
        self._append_pose(self._gt_poses, message)

    def _on_complete(self, message):
        if not message.data:
            return
        with self._lock:
            self._stopped = True
        self._flush()

    @staticmethod
    def _payload_from_poses(poses):
        t0 = poses[0].get("t0", poses[0]["t"])
        return {
            "schema": "simenv_exploration_trajectory_v1",
            "frame_id": "world",
            "start_stamp": t0,
            "start": {"x": poses[0]["x"], "y": poses[0]["y"], "yaw": poses[0]["yaw"]},
            "end": {"x": poses[-1]["x"], "y": poses[-1]["y"], "yaw": poses[-1]["yaw"]},
            "duration_sec": round(poses[-1]["t"] - t0, 3),
            "path_length_m": round(
                sum(
                    math.hypot(
                        poses[i]["x"] - poses[i - 1]["x"],
                        poses[i]["y"] - poses[i - 1]["y"],
                    )
                    for i in range(1, len(poses))
                ),
                3,
            ),
            "poses": [
                {
                    "t": p["t"] - t0,
                    "x": p["x"],
                    "y": p["y"],
                    "z": p["z"],
                    "yaw": p["yaw"],
                }
                for p in poses
            ],
        }

    @staticmethod
    def _write_json(path, payload):
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)

    def _flush(self):
        with self._lock:
            poses = list(self._poses)
            gt_poses = list(self._gt_poses)
        if poses:
            self._write_json(self._output_file, self._payload_from_poses(poses))
        if gt_poses and self._gt_output_file:
            self._write_json(self._gt_output_file, self._payload_from_poses(gt_poses))


if __name__ == "__main__":
    rospy.init_node("trajectory_recorder")
    TrajectoryRecorder()
    rospy.spin()
