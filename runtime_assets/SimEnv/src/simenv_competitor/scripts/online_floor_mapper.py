#!/usr/bin/env python3
"""Online first-floor occupancy and occlusion-aware camera coverage mapper.

Only onboard Livox, RealSense and the FAST-LIO-derived /state_estimation are
consumed.  Layout and Gazebo truth are deliberately absent from this node.
"""

import json
import math
import os
import sys
import threading

import numpy as np
import rospy
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, PointCloud, PointCloud2
from std_msgs.msg import Bool, Header, String

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from first_floor_core import (
    RayOccupancyGrid,
    bresenham_cells,
    remove_isolated_occupied_cells,
    vertically_supported_obstacles,
)


def yaw_from_quaternion(quaternion):
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


class OnlineFloorMapper:
    def __init__(self):
        self._lock = threading.RLock()
        self._resolution = float(rospy.get_param("~resolution", 0.15))
        self._span = float(rospy.get_param("~map_span_m", 50.0))
        self._maximum_range = float(rospy.get_param("~lidar_range", 20.0))
        self._camera_range = float(rospy.get_param("~camera_range", 6.0))
        self._camera_fov = math.radians(float(rospy.get_param("~camera_fov_deg", 60.0)))
        self._unpitch = float(rospy.get_param("~lidar_unpitch_rad", 0.785))
        self._sensor_x = float(rospy.get_param("~lidar_offset_x", 0.20))
        self._max_rays = int(rospy.get_param("~max_rays_per_scan", 1800))
        self._pose_topic = rospy.get_param("~pose_topic", "/state_estimation")
        self._scan_topic = rospy.get_param("~scan_topic", "/scan")
        self._map_frame = rospy.get_param("~map_frame", "map")
        self._pose = None
        self._home = None
        self._grid = None
        self._camera_checked = None
        self._last_camera_update = rospy.Time(0)
        self._last_scan_stamp = rospy.Time(0)
        self._output_png = rospy.get_param(
            "~online_map_file", os.path.join(os.getcwd(), "results", "online_map.png")
        )
        self._coverage_file = rospy.get_param(
            "~coverage_file",
            os.path.join(os.getcwd(), "results", "coverage_status.json"),
        )

        self._map_pub = rospy.Publisher(
            "/simenv/first_floor_map", OccupancyGrid, queue_size=1, latch=True
        )
        self._coverage_pub = rospy.Publisher(
            "/simenv/coverage_status", String, queue_size=1, latch=True
        )
        self._obstacle_pub = rospy.Publisher(
            "/simenv/local_obstacles", PointCloud2, queue_size=1
        )
        rospy.Subscriber(self._pose_topic, Odometry, self._on_pose, queue_size=20)
        rospy.Subscriber(self._scan_topic, PointCloud, self._on_scan, queue_size=2)
        rospy.Subscriber(
            "/real_sense/rgb/image_raw", Image, self._on_camera, queue_size=1
        )
        rospy.Subscriber(
            "/simenv/finalize_result", Bool, self._on_finalize, queue_size=1
        )
        rospy.Timer(rospy.Duration(0.5), self._publish)
        rospy.on_shutdown(self._write_artifacts)

    def _occupancy(self):
        return remove_isolated_occupied_cells(self._grid.occupancy(), 3)

    def _initialise_grid(self, x, y):
        cells = int(math.ceil(self._span / self._resolution))
        origin_x = float(x) - 0.5 * cells * self._resolution
        origin_y = float(y) - 0.5 * cells * self._resolution
        self._grid = RayOccupancyGrid(
            cells, cells, self._resolution, origin_x, origin_y
        )
        self._camera_checked = np.zeros((cells, cells), dtype=np.bool_)
        rospy.loginfo(
            "Online map initialised: %dx%d @ %.2fm, origin=(%.2f, %.2f)",
            cells,
            cells,
            self._resolution,
            origin_x,
            origin_y,
        )

    def _on_pose(self, message):
        position = message.pose.pose.position
        pose = (
            float(position.x),
            float(position.y),
            yaw_from_quaternion(message.pose.pose.orientation),
        )
        if not all(math.isfinite(value) for value in pose):
            return
        with self._lock:
            self._pose = pose
            if not self._map_frame and message.header.frame_id:
                self._map_frame = message.header.frame_id
            if self._home is None:
                self._home = pose
                self._initialise_grid(pose[0], pose[1])

    def _level_points(self, message):
        raw = np.asarray(
            [(point.x, point.y, point.z) for point in message.points], dtype=np.float64
        )
        if raw.size == 0:
            return raw.reshape((-1, 3))
        cosine, sine = math.cos(self._unpitch), math.sin(self._unpitch)
        # Same +Ry(unpitch) rotation used by fastlio_pointcloud_adapter.py.
        levelled = raw.copy()
        levelled[:, 0] = cosine * raw[:, 0] + sine * raw[:, 2]
        levelled[:, 2] = -sine * raw[:, 0] + cosine * raw[:, 2]
        return levelled

    def _on_scan(self, message):
        if not message.points:
            return
        levelled = self._level_points(message)
        ranges = np.linalg.norm(levelled, axis=1)
        planar_ranges = np.linalg.norm(levelled[:, :2], axis=1)
        valid = (
            np.all(np.isfinite(levelled), axis=1)
            # Simulated Mid-360 returns include the A1 forebody around
            # 0.45--0.55 m after unpitching.  They are not scene obstacles.
            & (planar_ranges >= 0.82)
            & (ranges <= self._maximum_range)
            # Static occupancy uses the body-height obstacle slice.  Floor,
            # ceiling and grazing returns remain available to FAST-LIO but do
            # not create cross-corridor walls in the navigation map.
            & (levelled[:, 2] >= -0.05)
            & (levelled[:, 2] <= 1.20)
        )
        levelled = vertically_supported_obstacles(levelled[valid])
        if len(levelled) == 0:
            return
        if len(levelled) > self._max_rays:
            indices = np.linspace(0, len(levelled) - 1, self._max_rays).astype(np.int32)
            levelled = levelled[indices]

        with self._lock:
            pose = self._pose
            grid = self._grid
            if pose is None or grid is None:
                return
            x, y, yaw = pose
            cosine, sine = math.cos(yaw), math.sin(yaw)
            origin = (
                x + cosine * self._sensor_x,
                y + sine * self._sensor_x,
            )
            local_x = levelled[:, 0] + self._sensor_x
            local_y = levelled[:, 1]
            world_x = x + cosine * local_x - sine * local_y
            world_y = y + sine * local_x + cosine * local_y
            for endpoint_x, endpoint_y in zip(world_x, world_y):
                grid.update_ray(origin, (endpoint_x, endpoint_y), hit=True)
            obstacle_mask = np.logical_and(levelled[:, 2] >= -0.05, levelled[:, 2] <= 1.0)
            obstacle_points = [
                (float(px), float(py), 0.30)
                for px, py in zip(local_x[obstacle_mask], local_y[obstacle_mask])
            ]
            self._last_scan_stamp = message.header.stamp

        if obstacle_points:
            # The rolling local costmap needs the sensor origin at the robot,
            # not map=(0,0).  Points stay in the levelled base frame here;
            # only the persistent occupancy endpoints above use world space.
            header = Header(stamp=rospy.Time.now(), frame_id="base")
            self._obstacle_pub.publish(
                point_cloud2.create_cloud_xyz32(header, obstacle_points)
            )

    def _on_camera(self, message):
        now = message.header.stamp if message.header.stamp != rospy.Time(0) else rospy.Time.now()
        if (now - self._last_camera_update).to_sec() < 0.35:
            return
        self._last_camera_update = now
        with self._lock:
            if self._pose is None or self._grid is None:
                return
            x, y, yaw = self._pose
            grid = self._grid
            occupancy = self._occupancy()
            camera = grid.world_to_cell(x + 0.28 * math.cos(yaw), y + 0.28 * math.sin(yaw))
            if camera is None:
                return
            angular_samples = max(31, int(math.degrees(self._camera_fov)) + 1)
            for relative in np.linspace(-0.5 * self._camera_fov, 0.5 * self._camera_fov, angular_samples):
                angle = yaw + float(relative)
                endpoint = grid.world_to_cell(
                    x + self._camera_range * math.cos(angle),
                    y + self._camera_range * math.sin(angle),
                )
                if endpoint is None:
                    continue
                for ix, iy in bresenham_cells(camera, endpoint):
                    if occupancy[iy, ix] == 100:
                        break
                    if occupancy[iy, ix] == 0:
                        self._camera_checked[iy, ix] = True

    def _payload(self):
        if self._grid is None:
            return {
                "schema": "simenv_online_coverage_v2",
                "camera_coverage_pct": 0.0,
                "mapped_free_cells": 0,
                "camera_checked_cells": 0,
            }
        occupancy = self._occupancy()
        free = occupancy == 0
        free_count = int(np.count_nonzero(free))
        checked_count = int(np.count_nonzero(np.logical_and(free, self._camera_checked)))
        return {
            "schema": "simenv_online_coverage_v2",
            "online_only": True,
            "resolution_m": self._resolution,
            "camera_fov_deg": round(math.degrees(self._camera_fov), 3),
            "camera_range_m": self._camera_range,
            "mapped_free_cells": free_count,
            "camera_checked_cells": checked_count,
            "camera_coverage_pct": round(100.0 * checked_count / max(1, free_count), 3),
        }

    def _publish(self, _event):
        with self._lock:
            if self._grid is None:
                return
            occupancy = self._occupancy()
            message = OccupancyGrid()
            message.header.stamp = rospy.Time.now()
            message.header.frame_id = self._map_frame or "map"
            message.info.resolution = self._resolution
            message.info.width = self._grid.width
            message.info.height = self._grid.height
            message.info.origin.position.x = self._grid.origin_x
            message.info.origin.position.y = self._grid.origin_y
            message.info.origin.orientation.w = 1.0
            message.data = occupancy.ravel().astype(np.int8).tolist()
            payload = self._payload()
        self._map_pub.publish(message)
        self._coverage_pub.publish(String(data=json.dumps(payload, sort_keys=True)))

    @staticmethod
    def _write_json(path, payload):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)

    def _write_artifacts(self):
        with self._lock:
            if self._grid is None:
                return
            occupancy = self._occupancy().copy()
            checked = self._camera_checked.copy()
            payload = self._payload()
        self._write_json(self._coverage_file, payload)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            display = np.ones((*occupancy.shape, 3), dtype=np.float32)
            display[occupancy == -1] = (0.55, 0.55, 0.55)
            display[occupancy == 100] = (0.05, 0.05, 0.05)
            display[np.logical_and(checked, occupancy == 0)] = (0.30, 0.65, 1.0)
            os.makedirs(os.path.dirname(os.path.abspath(self._output_png)), exist_ok=True)
            fig, axis = plt.subplots(figsize=(9, 9), dpi=130)
            axis.imshow(
                display,
                origin="lower",
                extent=(
                    self._grid.origin_x,
                    self._grid.origin_x + self._grid.width * self._resolution,
                    self._grid.origin_y,
                    self._grid.origin_y + self._grid.height * self._resolution,
                ),
            )
            axis.set_aspect("equal")
            axis.set_title(
                "Online map only | camera checked={:.1f}%".format(
                    payload["camera_coverage_pct"]
                )
            )
            axis.set_xlabel("map x [m]")
            axis.set_ylabel("map y [m]")
            fig.tight_layout()
            fig.savefig(self._output_png, bbox_inches="tight")
            plt.close(fig)
        except Exception as error:
            rospy.logwarn("Could not save online map: %s", error)

    def _on_finalize(self, message):
        if message.data:
            self._write_artifacts()


if __name__ == "__main__":
    rospy.init_node("online_floor_mapper")
    OnlineFloorMapper()
    rospy.spin()
