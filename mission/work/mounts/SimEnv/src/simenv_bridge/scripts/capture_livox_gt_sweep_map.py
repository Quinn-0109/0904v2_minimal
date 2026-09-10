#!/usr/bin/env python3
"""
Capture a full-scene point cloud by sweeping known Gazebo poses.

The script places the A1 model at scan stations derived from the generated
building layout, collects filtered /livox/lidar2 frames, projects them with the
known model pose, voxel-deduplicates, and writes a PLY map. It is intended for
dataset/debug map generation, not physically realistic robot motion.
"""

import argparse
import json
import math
import os
import struct
import time

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Pose, Quaternion, Twist
from sensor_msgs.msg import PointCloud2


def yaw_to_quat(yaw):
    return Quaternion(0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5))


def quat_to_rot_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def transform_points(points, translation, quaternion):
    rot = quat_to_rot_matrix(quaternion)
    return (rot @ points.T).T + np.asarray(translation, dtype=np.float64)


def voxel_unique(points, voxel_size):
    if points.size == 0:
        return points.reshape((0, 3))
    quantized = np.floor(points / voxel_size).astype(np.int64)
    _, idx = np.unique(quantized, axis=0, return_index=True)
    idx.sort()
    return points[idx]


def write_ply_xyz(path, points):
    with open(path, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))
        for x, y, z in points:
            f.write(struct.pack("<fff", float(x), float(y), float(z)))


def linspace_inclusive(start, stop, step):
    values = []
    cur = start
    while cur <= stop + 1e-6:
        values.append(round(cur, 4))
        cur += step
    if not values or abs(values[-1] - stop) > step * 0.5:
        values.append(round(stop, 4))
    return values


def load_stations(layout_path, grid_step):
    layout = json.loads(open(layout_path, "r", encoding="utf-8").read())
    floor = layout["floors"][0]
    stations = []

    def add_grid(bounds, margin, label):
        xs = linspace_inclusive(bounds["x_min"] + margin, bounds["x_max"] - margin, grid_step)
        ys = linspace_inclusive(bounds["y_min"] + margin, bounds["y_max"] - margin, grid_step)
        for x in xs:
            for y in ys:
                stations.append((x, y, label))

    add_grid(floor["lobby_bounds"], 1.2, "lobby")
    add_grid(floor["corridor_bounds"], 0.25, "corridor")
    for room in floor["rooms"]:
        add_grid(room["bounds"], 1.0, room["id"])
        gx, gy = room["goal_pose"][:2]
        stations.append((gx, gy, room["id"] + "_goal"))
        dx, dy = room["door_pose"][:2]
        offset = -0.7 if dx < 0 else 0.7
        stations.append((dx + offset, dy, room["id"] + "_door"))

    seen = set()
    unique = []
    for x, y, label in stations:
        key = (round(x, 2), round(y, 2))
        if key in seen:
            continue
        seen.add(key)
        unique.append((float(x), float(y), label))
    return unique


class SweepCapturer:
    def __init__(self):
        self.model_name = rospy.get_param("~model_name", "a1_gazebo")
        self.layout = rospy.get_param("~layout", "/workspace/SimEnv/generated_building/layout_metadata.json")
        self.lidar_topic = rospy.get_param("~lidar", "/livox/lidar2")
        self.out_path = rospy.get_param("~out", "/workspace/SimEnv/results/uf_map/livox_gt_sweep_map.ply")
        self.robot_z = float(rospy.get_param("~robot_z", 0.6))
        self.grid_step = float(rospy.get_param("~grid_step", 5.0))
        self.frames_per_yaw = int(rospy.get_param("~frames_per_yaw", 3))
        self.settle_sec = float(rospy.get_param("~settle_sec", 0.25))
        self.voxel = float(rospy.get_param("~voxel", 0.06))
        self.max_range = float(rospy.get_param("~max_range", 35.0))
        self.yaws = [float(v) for v in rospy.get_param("~yaws", "0,1.5707963268,3.1415926536,-1.5707963268").split(",")]
        self.base_to_laser_translation = np.array([0.2, 0.0, 0.08], dtype=np.float64)
        self.base_to_laser_quaternion = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.point_chunks = []
        self.frames = 0
        self.stations_used = 0
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)

    def set_static_transform(self):
        listener = tf.TransformListener()
        try:
            listener.waitForTransform("base", "laser_livox", rospy.Time(0), rospy.Duration(3.0))
            trans, quat = listener.lookupTransform("base", "laser_livox", rospy.Time(0))
            self.base_to_laser_translation = np.array(trans, dtype=np.float64)
            self.base_to_laser_quaternion = np.array(quat, dtype=np.float64)
        except Exception as exc:
            rospy.logwarn("using default base->laser_livox transform: %s", exc)

    def set_model_pose(self, x, y, yaw):
        state = ModelState()
        state.model_name = self.model_name
        state.reference_frame = "world"
        state.pose = Pose()
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = self.robot_z
        state.pose.orientation = yaw_to_quat(yaw)
        state.twist = Twist()
        self.set_state(state)

    def collect_frame(self, x, y, yaw):
        msg = rospy.wait_for_message(self.lidar_topic, PointCloud2, timeout=5.0)
        pts = np.array(
            list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)),
            dtype=np.float64,
        )
        if pts.size == 0:
            return
        pts = pts.reshape((-1, 3))
        ranges = np.linalg.norm(pts, axis=1)
        pts = pts[np.isfinite(ranges) & (ranges <= self.max_range)]
        if pts.size == 0:
            return
        base_pts = transform_points(
            pts,
            self.base_to_laser_translation,
            self.base_to_laser_quaternion,
        )
        q = np.array([0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)], dtype=np.float64)
        odom_pts = transform_points(base_pts, np.array([x, y, self.robot_z], dtype=np.float64), q)
        self.frames += 1
        self.point_chunks.append(odom_pts.astype(np.float32))

    def run(self):
        self.set_static_transform()
        rospy.wait_for_service("/gazebo/set_model_state", timeout=10.0)
        self.set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        stations = load_stations(self.layout, self.grid_step)
        rospy.loginfo("capture_livox_gt_sweep_map: stations=%d yaws=%d", len(stations), len(self.yaws))
        for index, (x, y, label) in enumerate(stations):
            if rospy.is_shutdown():
                break
            for yaw in self.yaws:
                self.set_model_pose(x, y, yaw)
                time.sleep(self.settle_sec)
                for _ in range(self.frames_per_yaw):
                    self.collect_frame(x, y, yaw)
            self.stations_used += 1
            if self.stations_used % 5 == 0:
                rospy.loginfo(
                    "sweep station %d/%d label=%s points=%d frames=%d",
                    index + 1,
                    len(stations),
                    label,
                    sum(len(chunk) for chunk in self.point_chunks),
                    self.frames,
                )
        self.dump(stations)

    def dump(self, stations):
        if not self.point_chunks:
            raise RuntimeError("no points captured")
        points = voxel_unique(np.vstack(self.point_chunks), self.voxel).astype(np.float32)
        if points.size == 0:
            raise RuntimeError("no points captured")
        write_ply_xyz(self.out_path, points)
        meta_path = os.path.splitext(self.out_path)[0] + ".txt"
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        with open(meta_path, "w", encoding="ascii") as f:
            f.write(f"stations_total={len(stations)}\n")
            f.write(f"stations_used={self.stations_used}\n")
            f.write(f"frames={self.frames}\n")
            f.write(f"points={len(points)}\n")
            f.write(f"voxel={self.voxel}\n")
            f.write(f"grid_step={self.grid_step}\n")
            f.write(f"frames_per_yaw={self.frames_per_yaw}\n")
            f.write("bbox_min=%.6f %.6f %.6f\n" % tuple(bbox_min))
            f.write("bbox_max=%.6f %.6f %.6f\n" % tuple(bbox_max))
        rospy.loginfo("wrote %d points to %s", len(points), self.out_path)


def self_test():
    pts = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    moved = transform_points(pts, [1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(moved, [[2.0, 2.0, 3.0]])
    assert len(linspace_inclusive(0.0, 1.0, 0.5)) == 3
    print("capture_livox_gt_sweep_map self-test passed")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--self-test", action="store_true")
    args, _ = parser.parse_known_args()
    if args.self_test:
        self_test()
        return
    rospy.init_node("capture_livox_gt_sweep_map")
    SweepCapturer().run()


if __name__ == "__main__":
    main()
