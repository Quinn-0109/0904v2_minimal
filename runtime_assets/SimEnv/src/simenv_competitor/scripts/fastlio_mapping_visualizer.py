#!/usr/bin/env python3
"""Expose FAST-LIO2 odometry and a bounded accumulated cloud for RViz.

FAST-LIO2's ``/cloud_registered`` is the registered current scan.  This node
voxel-accumulates those already-map-frame points and republishes a latched
``/cloud_map`` (plus the requested ``/map`` alias).  No navigation, planning,
ground truth, or command generation is performed here.
"""

import copy
import math
import threading

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Imu, PointCloud, PointCloud2
from std_msgs.msg import Header


class FastlioMappingVisualizer:
    def __init__(self):
        self._lock = threading.RLock()
        self._voxel_size = max(0.03, float(rospy.get_param("~voxel_size", 0.10)))
        self._maximum_voxels = max(
            10000, int(rospy.get_param("~maximum_voxels", 350000))
        )
        self._maximum_scan_points = max(
            1000, int(rospy.get_param("~maximum_scan_points", 12000))
        )
        self._publish_rate = max(0.2, float(rospy.get_param("~publish_rate", 1.0)))
        self._voxels = {}
        self._received = {
            "lidar": False,
            "imu": False,
            "odometry": False,
            "registered_cloud": False,
        }
        self._ready_logged = False
        self._last_cloud_stamp = rospy.Time(0)
        self._last_frame = "camera_init"
        self._started_wall = rospy.get_time()

        self._odom_pub = rospy.Publisher("/odometry", Odometry, queue_size=10)
        self._state_pub = rospy.Publisher(
            "/state_estimation", Odometry, queue_size=10
        )
        self._map_pub = rospy.Publisher(
            "/cloud_map", PointCloud2, queue_size=1, latch=True
        )
        self._map_alias_pub = rospy.Publisher(
            "/map", PointCloud2, queue_size=1, latch=True
        )

        rospy.Subscriber("/scan", PointCloud, self._on_lidar, queue_size=1)
        rospy.Subscriber("/livox/imu", Imu, self._on_imu, queue_size=20)
        rospy.Subscriber("/Odometry", Odometry, self._on_odometry, queue_size=20)
        rospy.Subscriber(
            "/cloud_registered", PointCloud2, self._on_registered, queue_size=2
        )
        rospy.Timer(rospy.Duration(1.0 / self._publish_rate), self._publish_map)
        rospy.Timer(rospy.Duration(1.0), self._report_readiness)

    def _mark(self, key):
        with self._lock:
            self._received[key] = True

    def _on_lidar(self, _message):
        self._mark("lidar")

    def _on_imu(self, _message):
        self._mark("imu")

    def _on_odometry(self, message):
        self._mark("odometry")
        output = copy.deepcopy(message)
        # map is defined as an identity alias of FAST-LIO's camera_init by the
        # static TF in mapping_test.launch, so this is a coordinate-preserving
        # frame normalization rather than an untransformed frame rewrite.
        output.header.frame_id = "map"
        output.child_frame_id = message.child_frame_id or "body"
        self._odom_pub.publish(output)
        self._state_pub.publish(output)

    def _voxel_key(self, x, y, z):
        scale = 1.0 / self._voxel_size
        return (
            int(math.floor(x * scale)),
            int(math.floor(y * scale)),
            int(math.floor(z * scale)),
        )

    def _on_registered(self, message):
        self._mark("registered_cloud")
        points = list(
            point_cloud2.read_points(
                message, field_names=("x", "y", "z"), skip_nans=True
            )
        )
        if not points:
            return
        if len(points) > self._maximum_scan_points:
            stride = int(math.ceil(len(points) / float(self._maximum_scan_points)))
            points = points[::stride]
        with self._lock:
            self._last_cloud_stamp = message.header.stamp
            self._last_frame = message.header.frame_id or "camera_init"
            for x, y, z in points:
                if not all(math.isfinite(value) for value in (x, y, z)):
                    continue
                self._voxels[self._voxel_key(x, y, z)] = (
                    float(x),
                    float(y),
                    float(z),
                )
            overflow = len(self._voxels) - self._maximum_voxels
            for _ in range(max(0, overflow)):
                # Dict insertion order gives a deterministic bounded rolling
                # map if a very long manual session reaches the memory cap.
                del self._voxels[next(iter(self._voxels))]

    def _publish_map(self, _event):
        with self._lock:
            if not self._voxels:
                return
            points = list(self._voxels.values())
            stamp = self._last_cloud_stamp
            frame = self._last_frame
        header = Header()
        header.stamp = stamp if stamp != rospy.Time(0) else rospy.Time.now()
        header.frame_id = frame
        cloud = point_cloud2.create_cloud_xyz32(header, points)
        self._map_pub.publish(cloud)
        self._map_alias_pub.publish(cloud)

    def _report_readiness(self, _event):
        with self._lock:
            received = dict(self._received)
            voxel_count = len(self._voxels)
        if all(received.values()) and not self._ready_logged:
            self._ready_logged = True
            rospy.loginfo(
                "FAST-LIO initialized: LiDAR + IMU + /Odometry + "
                "/cloud_registered are live; accumulated map has %d voxels",
                voxel_count,
            )
        elif not self._ready_logged:
            missing = [name for name, ready in received.items() if not ready]
            rospy.loginfo_throttle(
                5.0, "Waiting for FAST-LIO streams: %s", ", ".join(missing)
            )


if __name__ == "__main__":
    rospy.init_node("fastlio_mapping_visualizer")
    FastlioMappingVisualizer()
    rospy.spin()
