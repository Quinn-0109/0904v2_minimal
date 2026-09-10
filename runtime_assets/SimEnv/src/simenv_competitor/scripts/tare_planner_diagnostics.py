#!/usr/bin/env python3
"""Record Official TARE frontier/viewpoint/path decisions without changing them."""

import json
import math
import os
import threading

import rospy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2


def _yaw(quaternion):
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z +
               quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y +
                     quaternion.z * quaternion.z))


class TarePlannerDiagnostics:
    """Compact telemetry for the unmodified decisions made inside TARE."""

    def __init__(self):
        self._lock = threading.RLock()
        self._pose = None
        self._records = []
        self._last_cloud_time = {}
        self._started = rospy.Time.now().to_sec()
        output_dir = os.path.abspath(rospy.get_param(
            "~output_dir", "/tmp/simenv/tare_far"))
        self._path = os.path.join(
            output_dir, "logs", "tare_planner_diagnostics.json")
        self._cloud_period = max(
            0.2, float(rospy.get_param("~cloud_sample_period", 1.0)))
        self._maximum_records = max(
            100, int(rospy.get_param("~maximum_records", 3000)))
        self._maximum_sample_points = max(
            20, int(rospy.get_param("~maximum_sample_points", 200)))

        rospy.Subscriber(
            rospy.get_param("~odom_topic", "/tare/state_estimation_at_scan"),
            Odometry, self._on_odom, queue_size=20)
        rospy.Subscriber(
            rospy.get_param("~waypoint_topic", "/tare/way_point"),
            PointStamped, self._on_waypoint, queue_size=5)
        rospy.Subscriber(
            rospy.get_param(
                "~path_topic", "/sensor_coverage_planner/exploration_path"),
            Path, self._on_path, queue_size=2)

        cloud_topics = {
            "filtered_frontier": rospy.get_param(
                "~filtered_frontier_topic",
                "/sensor_coverage_planner/filtered_frontier_cloud"),
            "uncovered_frontier": rospy.get_param(
                "~uncovered_frontier_topic",
                "/sensor_coverage_planner/uncovered_frontier_cloud"),
            "selected_viewpoint": rospy.get_param(
                "~selected_viewpoint_topic",
                "/sensor_coverage_planner/selected_viewpoint_vis_cloud"),
        }
        self._subscribers = []
        for name, topic in cloud_topics.items():
            self._subscribers.append(rospy.Subscriber(
                topic, PointCloud2,
                lambda message, label=name: self._on_cloud(label, message),
                queue_size=1))

        self._timer = rospy.Timer(rospy.Duration(2.0), self._save)
        rospy.on_shutdown(self._save)
        self._save()
        rospy.loginfo(
            "Official TARE diagnostics ready: output=%s", self._path)

    def _elapsed(self):
        return max(0.0, rospy.Time.now().to_sec() - self._started)

    def _append(self, record):
        record["elapsed_sec"] = self._elapsed()
        with self._lock:
            self._records.append(record)
            if len(self._records) > self._maximum_records:
                self._records = self._records[-self._maximum_records:]

    def _on_odom(self, message):
        position = message.pose.pose.position
        pose = (float(position.x), float(position.y),
                float(position.z), _yaw(message.pose.pose.orientation))
        with self._lock:
            self._pose = pose

    def _on_waypoint(self, message):
        self._append({
            "kind": "waypoint",
            "stamp": message.header.stamp.to_sec(),
            "frame_id": message.header.frame_id,
            "point": [message.point.x, message.point.y, message.point.z],
        })

    def _on_path(self, message):
        points = [[pose.pose.position.x, pose.pose.position.y,
                   pose.pose.position.z] for pose in message.poses]
        length = sum(math.hypot(
            second[0] - first[0], second[1] - first[1])
                     for first, second in zip(points[:-1], points[1:]))
        self._append({
            "kind": "exploration_path",
            "stamp": message.header.stamp.to_sec(),
            "frame_id": message.header.frame_id,
            "point_count": len(points),
            "length_m": length,
            "points": points,
        })

    def _on_cloud(self, name, message):
        now = rospy.Time.now().to_sec()
        if now - self._last_cloud_time.get(name, -math.inf) < \
                self._cloud_period:
            return
        self._last_cloud_time[name] = now
        points = [(float(x), float(y), float(z)) for x, y, z in
                  point_cloud2.read_points(
                      message, field_names=("x", "y", "z"),
                      skip_nans=True)]
        with self._lock:
            pose = self._pose
        record = {
            "kind": name,
            "stamp": message.header.stamp.to_sec(),
            "frame_id": message.header.frame_id,
            "point_count": len(points),
        }
        if points:
            record["bounds"] = {
                "min": [min(point[axis] for point in points)
                        for axis in range(3)],
                "max": [max(point[axis] for point in points)
                        for axis in range(3)],
            }
            stride = max(1, len(points) // self._maximum_sample_points)
            record["sample_points"] = [
                list(point) for point in points[::stride]
                [:self._maximum_sample_points]]
        if pose is not None and points:
            cosine, sine = math.cos(pose[3]), math.sin(pose[3])
            relative = []
            for x, y, _ in points:
                dx, dy = x - pose[0], y - pose[1]
                relative.append((cosine * dx + sine * dy,
                                 -sine * dx + cosine * dy))
            record["relative_counts"] = {
                "left_beyond_1_5m": sum(lat > 1.5 for _, lat in relative),
                "right_beyond_1_5m": sum(lat < -1.5 for _, lat in relative),
                "center_within_1_5m": sum(
                    abs(lat) <= 1.5 for _, lat in relative),
                "ahead": sum(forward > 0.0 for forward, _ in relative),
                "behind": sum(forward <= 0.0 for forward, _ in relative),
            }
            record["robot_pose"] = list(pose)
        self._append(record)

    def _save(self, _event=None):
        with self._lock:
            records = list(self._records)
        directory = os.path.dirname(self._path)
        os.makedirs(directory, exist_ok=True)
        temporary = self._path + ".tmp"
        payload = {
            "schema": "simenv_official_tare_diagnostics_v1",
            "policy_effect": "none_observation_only",
            "records": records,
        }
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
        os.replace(temporary, self._path)


if __name__ == "__main__":
    rospy.init_node("tare_planner_diagnostics")
    TarePlannerDiagnostics()
    rospy.spin()
