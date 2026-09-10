#!/usr/bin/env python3
"""Aggregate UFO backend health into a machine-readable run summary."""

import json
import math
import os
import threading
import time

import rospy
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool, String
from geometry_msgs.msg import PoseStamped


class StatusMonitor:
    def __init__(self):
        self._lock = threading.Lock()
        self._output_dir = os.path.abspath(rospy.get_param("~output_dir", "/tmp/ufo"))
        os.makedirs(os.path.join(self._output_dir, "logs"), exist_ok=True)
        self._started = time.monotonic()
        self._counts = dict(ufo_path_count=0, ufo_empty_path_count=0,
                            ufo_scan_reject_count=0, ufo_fallback_count=0,
                            total_executed_waypoints=0, stall_count=0)
        self._last_pose = None
        self._distance = 0.0
        self._complete = False
        self._latest = {}
        rospy.Subscriber("/simenv/ufo_exploration_path",
                         Path, self._on_path, queue_size=10)
        rospy.Subscriber("/simenv/ufoexplorer_status",
                         String, self._on_ufo, queue_size=20)
        rospy.Subscriber("/simenv/ufo_path_adapter_status",
                         String, self._on_adapter, queue_size=20)
        rospy.Subscriber("/simenv/velocity_watchdog_status",
                         String, self._on_watchdog, queue_size=20)
        rospy.Subscriber("/simenv/ufo_executed_waypoint",
                         PoseStamped, self._on_waypoint, queue_size=20)
        rospy.Subscriber(rospy.get_param("~odom_topic", "/simenv/ufo/odometry"),
                         Odometry, self._on_odom, queue_size=20)
        rospy.Subscriber("/simenv/mission_complete",
                         Bool, self._on_complete, queue_size=2)
        rospy.Timer(rospy.Duration(2.0), self._on_timer)
        rospy.on_shutdown(self._write)

    @staticmethod
    def _json(message):
        try:
            return json.loads(message.data)
        except (TypeError, ValueError):
            return {}

    def _on_path(self, _message):
        with self._lock:
            self._counts["ufo_path_count"] += 1

    def _on_ufo(self, message):
        data = self._json(message)
        with self._lock:
            self._latest["ufo"] = data
            state = data.get("state", "")
            if state in ("UFO_PATH_EMPTY", "UFO_NO_PATH", "UFO_PATH_TOO_SHORT"):
                self._counts["ufo_empty_path_count"] += 1
            if data.get("fallback_active") and state == "PATH_READY":
                self._counts["ufo_fallback_count"] += 1

    def _on_adapter(self, message):
        data = self._json(message)
        with self._lock:
            self._latest["adapter"] = data
            if data.get("state") == "SCAN_REJECTED":
                self._counts["ufo_scan_reject_count"] += 1

    def _on_watchdog(self, message):
        data = self._json(message)
        with self._lock:
            self._latest["watchdog"] = data
            if data.get("event") == "STALLED":
                self._counts["stall_count"] += 1

    def _on_waypoint(self, _message):
        with self._lock:
            self._counts["total_executed_waypoints"] += 1

    def _on_odom(self, message):
        point = (message.pose.pose.position.x, message.pose.pose.position.y)
        with self._lock:
            if self._last_pose is not None:
                step = math.hypot(point[0] - self._last_pose[0],
                                  point[1] - self._last_pose[1])
                if step < 2.0:
                    self._distance += step
            self._last_pose = point

    def _on_complete(self, message):
        if message.data:
            self._complete = True
            self._write()

    def _on_timer(self, _event):
        self._write()

    def _write(self):
        with self._lock:
            payload = dict(total_runtime=time.monotonic() - self._started,
                           total_distance=self._distance,
                           room_entry_count=None,
                           room_entry_count_source="offline_truth_only",
                           mission_complete=self._complete,
                           latest_status=dict(self._latest))
            payload.update(self._counts)
        # Keep the requested canonical log location and a root-level
        # compatibility copy used by older result-analysis scripts.
        for directory in (os.path.join(self._output_dir, "logs"),
                          self._output_dir):
            temporary = os.path.join(directory, "ufoexplorer_summary.json.tmp")
            final = os.path.join(directory, "ufoexplorer_summary.json")
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
            os.replace(temporary, final)


if __name__ == "__main__":
    rospy.init_node("ufoexplorer_status_monitor")
    StatusMonitor()
    rospy.spin()
