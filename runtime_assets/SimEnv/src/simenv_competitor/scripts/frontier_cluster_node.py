#!/usr/bin/env python3
"""Detect and publish reachable frontier clusters from the online 2-D map."""

import json
import os
import sys
import threading

import rospy
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import String

# catkin's devel-space relay does not add this source directory to sys.path.
# The frontier detector and serialization helpers are adjacent pure modules.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from exploration_graph import FrontierClusterDetector
from hierarchical_ros_utils import (
    atomic_json, cluster_to_dict, occupancy_from_message)


class FrontierClusterNode:
    def __init__(self):
        self.lock = threading.RLock()
        self.pose = None
        self.sequence = 0
        self.history = []
        self.output_dir = rospy.get_param("~output_dir", "/tmp/simenv")
        self.history_file = os.path.join(
            self.output_dir, "logs", "frontier_history.json")
        self.detector = FrontierClusterDetector(
            minimum_cluster_cells=rospy.get_param(
                "~minimum_cluster_cells", 8),
            merge_distance=rospy.get_param("~merge_distance", 1.25),
            merge_connectivity_radius=rospy.get_param(
                "~merge_connectivity_radius", 2.5),
            unknown_area_radius=rospy.get_param(
                "~unknown_area_radius", 2.5),
            goal_search_radius=rospy.get_param(
                "~goal_search_radius", 1.5),
            goal_clearance=rospy.get_param("~goal_clearance", 0.35),
        )
        self.publisher = rospy.Publisher(
            "/simenv/frontier_clusters", String, queue_size=1, latch=True)
        rospy.Subscriber(
            rospy.get_param("~odom_topic", "/Odometry"),
            Odometry, self._on_odom, queue_size=20)
        rospy.Subscriber(
            rospy.get_param(
                "~map_topic", "/simenv/voxel_floor_projection"),
            OccupancyGrid, self._on_map, queue_size=1)

    def _on_odom(self, message):
        with self.lock:
            self.pose = (
                float(message.pose.pose.position.x),
                float(message.pose.pose.position.y))

    def _on_map(self, message):
        with self.lock:
            pose = self.pose
        if pose is None:
            return
        try:
            grid = occupancy_from_message(message)
            clusters = self.detector.detect(grid, pose)
        except (ValueError, IndexError) as error:
            rospy.logwarn_throttle(
                5.0, "frontier cluster update failed: %s", error)
            return
        self.sequence += 1
        payload = {
            "schema": "simenv_frontier_clusters_v1",
            "sequence": self.sequence,
            "stamp": message.header.stamp.to_sec(),
            "frame_id": message.header.frame_id,
            "robot_pose": list(pose),
            "clusters": [cluster_to_dict(item) for item in clusters],
        }
        self.publisher.publish(String(data=json.dumps(payload)))
        self.history.append({
            "sequence": self.sequence,
            "stamp": payload["stamp"],
            "robot_pose": list(pose),
            "cluster_count": len(clusters),
            "clusters": payload["clusters"],
        })
        maximum = int(rospy.get_param("~maximum_history_updates", 10000))
        if len(self.history) > maximum:
            self.history = self.history[-maximum:]
        atomic_json(self.history_file, {
            "schema": "simenv_frontier_history_v1",
            "updates": self.history,
        })


if __name__ == "__main__":
    rospy.init_node("frontier_cluster_node")
    FrontierClusterNode()
    rospy.spin()
