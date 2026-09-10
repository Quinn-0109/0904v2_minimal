#!/usr/bin/env python3
"""Record time-synchronised LiDAR ray coverage in Gazebo world coordinates.

This node is an offline evaluator only.  It never publishes a planning topic
and its Gazebo truth input is not available to the exploration stack.  The
result lets post-run figures show what the simulated LiDAR rays actually
observed instead of treating a drifted final OctoMap free projection as a
sensor-coverage mask.
"""

import json
import math
import os
import threading
from collections import deque

import numpy as np
import rospy
from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import PointCloud


def quaternion_matrix(q):
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    return np.asarray([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
         2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
         1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


class TruthLidarCoverageLogger:
    def __init__(self):
        output_dir = os.path.abspath(rospy.get_param("~output_dir"))
        self._output_file = os.path.join(output_dir, "truth_lidar_coverage.json")
        self._model = str(rospy.get_param("~model_name", "a1_gazebo"))
        self._resolution = max(0.05, float(rospy.get_param("~resolution", 0.15)))
        self._maximum_range = max(0.5, float(rospy.get_param("~maximum_range", 30.0)))
        self._minimum_range = max(0.0, float(rospy.get_param("~minimum_range", 0.25)))
        self._minimum_scan_interval = max(
            0.0, float(rospy.get_param("~minimum_scan_interval", 0.20)))
        self._ray_stride = max(1, int(rospy.get_param("~ray_stride", 4)))
        self._maximum_sync_error = max(
            0.001, float(rospy.get_param("~maximum_sync_error", 0.05)))
        self._sensor_translation = np.asarray(
            rospy.get_param("~sensor_translation", [0.2, 0.0, 0.08]),
            dtype=np.float64)
        sensor_pitch = float(rospy.get_param("~sensor_pitch", 0.785))
        cosine, sine = math.cos(sensor_pitch), math.sin(sensor_pitch)
        self._sensor_rotation = np.asarray(
            [[cosine, 0.0, sine], [0.0, 1.0, 0.0],
             [-sine, 0.0, cosine]], dtype=np.float64)
        self._truth = deque(maxlen=250)
        self._covered = set()
        self._hits = set()
        self._lock = threading.RLock()
        self._last_scan_stamp = None
        self._scan_count = 0
        self._synchronised_scan_count = 0
        self._dropped_unsynchronised = 0
        self._ray_count = 0
        self._maximum_sync_error_seen = 0.0
        self._closed = False
        os.makedirs(output_dir, exist_ok=True)
        rospy.Subscriber("/gazebo/model_states", ModelStates,
                         self._on_truth, queue_size=20)
        rospy.Subscriber("/scan", PointCloud, self._on_scan, queue_size=2)
        # The exploration manager starts detached post-processing before ROS
        # has necessarily finished shutting every node down.  Keep a recent
        # atomic snapshot on disk so that visualization never races shutdown.
        rospy.Timer(rospy.Duration(2.0), self._write_snapshot)
        rospy.on_shutdown(self.close)
        rospy.loginfo(
            "Truth LiDAR coverage evaluator active: resolution=%.2fm range=%.1fm "
            "scan_interval=%.2fs stride=%d (offline only)",
            self._resolution, self._maximum_range,
            self._minimum_scan_interval, self._ray_stride)

    def _on_truth(self, message):
        try:
            index = message.name.index(self._model)
        except ValueError:
            rospy.logwarn_throttle(5.0, "Coverage evaluator cannot find model %s",
                                   self._model)
            return
        with self._lock:
            self._truth.append((rospy.Time.now().to_sec(), message.pose[index]))

    def _cell(self, x, y):
        return (int(math.floor(float(x) / self._resolution)),
                int(math.floor(float(y) / self._resolution)))

    def _on_scan(self, message):
        stamp = message.header.stamp.to_sec()
        with self._lock:
            self._scan_count += 1
            if (self._last_scan_stamp is not None and
                    stamp - self._last_scan_stamp < self._minimum_scan_interval):
                return
            if not self._truth:
                self._dropped_unsynchronised += 1
                return
            truth_stamp, pose = min(self._truth,
                                    key=lambda item: abs(item[0] - stamp))
            sync_error = abs(truth_stamp - stamp)
            if sync_error > self._maximum_sync_error:
                self._dropped_unsynchronised += 1
                return
            self._last_scan_stamp = stamp
            self._maximum_sync_error_seen = max(
                self._maximum_sync_error_seen, sync_error)

        rotation_world = quaternion_matrix(pose.orientation)
        translation_world = np.asarray(
            [pose.position.x, pose.position.y, pose.position.z],
            dtype=np.float64)
        sensor_origin = rotation_world.dot(self._sensor_translation) + translation_world
        covered = set()
        hits = set()
        ray_count = 0
        for point in message.points[::self._ray_stride]:
            local = np.asarray([point.x, point.y, point.z], dtype=np.float64)
            distance = float(np.linalg.norm(local))
            if (not math.isfinite(distance) or distance < self._minimum_range or
                    distance > self._maximum_range):
                continue
            endpoint_base = self._sensor_rotation.dot(local) + self._sensor_translation
            endpoint = rotation_world.dot(endpoint_base) + translation_world
            dx, dy = endpoint[0] - sensor_origin[0], endpoint[1] - sensor_origin[1]
            planar_distance = math.hypot(dx, dy)
            steps = max(1, int(math.ceil(planar_distance / self._resolution)))
            for step in range(steps + 1):
                ratio = step / float(steps)
                covered.add(self._cell(sensor_origin[0] + ratio * dx,
                                       sensor_origin[1] + ratio * dy))
            hits.add(self._cell(endpoint[0], endpoint[1]))
            ray_count += 1
        with self._lock:
            self._covered.update(covered)
            self._hits.update(hits)
            self._ray_count += ray_count
            self._synchronised_scan_count += 1

    def _payload(self):
        with self._lock:
            covered = sorted(self._covered)
            hits = sorted(self._hits)
            payload = {
                "schema": "simenv_truth_lidar_coverage_v1",
                "coordinate_frame": "gazebo_world",
                "online_planning_input": False,
                "source": "/scan rays + nearest timestamped /gazebo/model_states pose",
                "resolution": self._resolution,
                "maximum_range": self._maximum_range,
                "minimum_range": self._minimum_range,
                "ray_stride": self._ray_stride,
                "minimum_scan_interval": self._minimum_scan_interval,
                "maximum_allowed_sync_error": self._maximum_sync_error,
                "maximum_sync_error_seen": self._maximum_sync_error_seen,
                "received_scan_count": self._scan_count,
                "synchronised_scan_count": self._synchronised_scan_count,
                "dropped_unsynchronised_scan_count": self._dropped_unsynchronised,
                "processed_ray_count": self._ray_count,
                "covered_cell_count": len(covered),
                "hit_cell_count": len(hits),
                "covered_cells": covered,
                "hit_cells": hits,
            }
        return payload, covered

    def _write_snapshot(self, _event=None, announce=False):
        payload, covered = self._payload()
        # Timer and shutdown callbacks can overlap during roslaunch teardown.
        # A shared ``.tmp`` name lets one callback replace/remove the file
        # underneath another, losing the final coverage snapshot.
        temporary = "{}.tmp.{}.{}".format(
            self._output_file, os.getpid(), threading.get_ident())
        try:
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, separators=(",", ":"), sort_keys=True)
                stream.write("\n")
            os.replace(temporary, self._output_file)
            if announce:
                rospy.loginfo("Truth LiDAR ray coverage saved to %s (%d cells)",
                              self._output_file, len(covered))
        except OSError as error:
            rospy.logerr("Failed to save truth LiDAR coverage: %s", error)
            try:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            except OSError:
                pass

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._write_snapshot(announce=True)


if __name__ == "__main__":
    rospy.init_node("truth_lidar_coverage_logger")
    TruthLidarCoverageLogger()
    rospy.spin()
