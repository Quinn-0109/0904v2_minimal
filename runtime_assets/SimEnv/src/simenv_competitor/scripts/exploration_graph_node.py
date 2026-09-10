#!/usr/bin/env python3
"""Maintain and publish persistent region/frontier exploration topology."""

import json
import os
import threading

import rospy
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import String

from exploration_graph import ExplorationGraph
from hierarchical_ros_utils import (
    atomic_json, cluster_from_dict, occupancy_from_message)


class ExplorationGraphNode:
    def __init__(self):
        self.lock = threading.RLock()
        self.grid = None
        self.pose = None
        self.output_dir = rospy.get_param("~output_dir", "/tmp/simenv")
        self.graph_file = os.path.join(
            self.output_dir, "logs", "exploration_graph.json")
        self.coverage_threshold = rospy.get_param(
            "~coverage_threshold", 0.90)
        self.graph = ExplorationGraph(
            region_core_clearance=rospy.get_param(
                "~region_core_clearance", 0.75),
            minimum_region_core_cells=rospy.get_param(
                "~minimum_region_core_cells", 20),
            region_association_radius=rospy.get_param(
                "~region_association_radius", 3.0),
            region_match_distance=rospy.get_param(
                "~region_match_distance", 5.0),
            graph_connection_distance=rospy.get_param(
                "~graph_connection_distance", 15.0),
        )
        self.publisher = rospy.Publisher(
            "/simenv/exploration_graph", String, queue_size=1, latch=True)
        rospy.Subscriber(
            rospy.get_param(
                "~map_topic", "/simenv/voxel_floor_projection"),
            OccupancyGrid, self._on_map, queue_size=1)
        rospy.Subscriber(
            rospy.get_param("~odom_topic", "/Odometry"),
            Odometry, self._on_odom, queue_size=20)
        rospy.Subscriber(
            "/simenv/frontier_clusters", String,
            self._on_clusters, queue_size=1)
        rospy.Subscriber(
            "/simenv/region_visit_update", String,
            self._on_visit, queue_size=10)

    def _on_map(self, message):
        try:
            grid = occupancy_from_message(message)
        except (ValueError, IndexError) as error:
            rospy.logwarn_throttle(5.0, "invalid occupancy map: %s", error)
            return
        with self.lock:
            self.grid = grid

    def _on_odom(self, message):
        with self.lock:
            self.pose = (
                float(message.pose.pose.position.x),
                float(message.pose.pose.position.y))

    def _on_visit(self, message):
        try:
            update = json.loads(message.data)
        except (TypeError, ValueError):
            return
        with self.lock:
            self.graph.mark_region(
                str(update.get("region_id", "")),
                float(update.get("coverage_ratio", 0.0)),
                float(update.get("time", rospy.Time.now().to_sec())),
                entered=bool(update.get("entered", False)),
                exited=bool(update.get("exited", False)),
                coverage_threshold=self.coverage_threshold,
                coverage_eligible=bool(
                    update.get("coverage_eligible", True)))
            self._publish_locked({})

    def _on_clusters(self, message):
        try:
            payload = json.loads(message.data)
            clusters = [
                cluster_from_dict(item)
                for item in payload.get("clusters", [])]
        except (KeyError, TypeError, ValueError) as error:
            rospy.logwarn_throttle(5.0, "invalid frontier payload: %s", error)
            return
        with self.lock:
            if self.grid is None or self.pose is None:
                return
            stats = self.graph.update(
                self.grid, clusters, self.pose,
                float(payload.get("stamp", rospy.Time.now().to_sec())))
            self._publish_locked(stats)

    def _publish_locked(self, stats):
        payload = self.graph.to_dict(include_metadata=False)
        payload["stamp"] = rospy.Time.now().to_sec()
        payload["stats"] = stats
        serialized = json.dumps(payload)
        self.publisher.publish(String(data=serialized))
        atomic_json(self.graph_file, payload)


if __name__ == "__main__":
    rospy.init_node("exploration_graph_node")
    ExplorationGraphNode()
    rospy.spin()
