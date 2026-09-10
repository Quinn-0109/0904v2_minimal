#!/usr/bin/env python3
"""Publish online LiDAR/OccupancyGrid corridor geometry for UFO path ranking."""

import json
import os
import sys
import threading
import time

import rospy
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import String

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_SCRIPT_DIR = os.path.dirname(SCRIPT_DIR)
for module_dir in (SCRIPT_DIR, PARENT_SCRIPT_DIR):
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

from corridor_semantic_core import (corridor_observations_match,
                                    detect_parallel_wall_corridor)
from hierarchical_ros_utils import occupancy_from_message


class CorridorSemanticNode:
    def __init__(self):
        self._lock = threading.Lock()
        self._pose = None
        self._grid = None
        self._previous = None
        self._confirmations = 0
        self._required = int(rospy.get_param("~confirmation_frames", 3))
        self._output_dir = os.path.abspath(rospy.get_param("~output_dir", "/tmp/ufo"))
        self._log_path = os.path.join(
            self._output_dir, "logs", "ufo_corridor_semantics.jsonl")
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        self._publisher = rospy.Publisher(
            "/simenv/ufo_corridor_semantics", String, queue_size=5, latch=True)
        rospy.Subscriber(rospy.get_param("~odom_topic", "/simenv/ufo/odometry"),
                         Odometry, self._on_odom, queue_size=10)
        rospy.Subscriber(rospy.get_param(
            "~map_topic", "/simenv/voxel_floor_projection"),
            OccupancyGrid, self._on_map, queue_size=2)
        rospy.Timer(rospy.Duration(0.5), self._on_timer)

    def _on_odom(self, message):
        with self._lock:
            self._pose = (message.pose.pose.position.x,
                          message.pose.pose.position.y)

    def _on_map(self, message):
        try:
            grid = occupancy_from_message(message)
        except (IndexError, TypeError, ValueError):
            return
        with self._lock:
            self._grid = grid

    def _on_timer(self, _event):
        with self._lock:
            pose, grid = self._pose, self._grid
        if pose is None or grid is None:
            return
        observation = detect_parallel_wall_corridor(
            grid, pose,
            search_radius=float(rospy.get_param("~search_radius", 8.0)),
            minimum_width=float(rospy.get_param("~minimum_width", 1.0)),
            maximum_width=float(rospy.get_param("~maximum_width", 4.0)),
            minimum_length=float(rospy.get_param("~minimum_visible_length", 4.0)),
            angle_step_degrees=float(rospy.get_param("~angle_step_degrees", 10.0)))
        if corridor_observations_match(self._previous, observation):
            self._confirmations += 1
        else:
            self._confirmations = 1 if observation is not None else 0
        self._previous = observation
        payload = dict(state="NO_CORRIDOR", confirmed=False,
                       confirmation_count=self._confirmations,
                       timestamp=time.time(), robot_pose=list(pose))
        if observation is not None:
            payload.update(observation)
            payload["state"] = ("CORRIDOR_CONFIRMED" if
                                self._confirmations >= self._required else
                                "CORRIDOR_CANDIDATE")
            payload["confirmed"] = self._confirmations >= self._required
        message = json.dumps(payload, sort_keys=True)
        self._publisher.publish(String(data=message))
        with open(self._log_path, "a", encoding="utf-8") as stream:
            stream.write(message + "\n")


if __name__ == "__main__":
    rospy.init_node("ufo_corridor_semantic")
    CorridorSemanticNode()
    rospy.spin()
