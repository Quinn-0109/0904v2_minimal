#!/usr/bin/env python3
"""
Export a global map by projecting filtered /livox/lidar2 with Gazebo GT odometry.

This bypasses Ultra-Fusion odometry. It uses /Odometry_gazebo as the trusted
base pose and the fixed base->laser_livox transform to write a voxel-deduped
PLY in the odom frame.
"""

import argparse
import bisect
import math
import os
import struct
import threading

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2


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


class OdomBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.lock = threading.Lock()
        self.stamps = []
        self.poses = []

    def add(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        stamp = msg.header.stamp.to_sec()
        pose = (
            np.array([p.x, p.y, p.z], dtype=np.float64),
            np.array([q.x, q.y, q.z, q.w], dtype=np.float64),
        )
        with self.lock:
            self.stamps.append(stamp)
            self.poses.append(pose)
            if len(self.stamps) > self.max_size:
                overflow = len(self.stamps) - self.max_size
                del self.stamps[:overflow]
                del self.poses[:overflow]

    def nearest(self, stamp, max_dt):
        with self.lock:
            if not self.stamps:
                return None, None
            idx = bisect.bisect_left(self.stamps, stamp)
            candidates = []
            if idx < len(self.stamps):
                candidates.append(idx)
            if idx > 0:
                candidates.append(idx - 1)
            best = min(candidates, key=lambda i: abs(self.stamps[i] - stamp))
            dt = abs(self.stamps[best] - stamp)
            if dt > max_dt:
                return None, dt
            return self.poses[best], dt


class LivoxGTMapExporter:
    def __init__(self):
        self.lock = threading.Lock()
        self.odom = OdomBuffer(max_size=int(rospy.get_param("~odom_buffer", 5000)))
        self.points = np.zeros((0, 3), dtype=np.float32)
        self.frames = 0
        self.used_frames = 0
        self.dropped_no_odom = 0
        self.max_observed_dt = 0.0
        self.path_length = 0.0
        self.last_pose_xy = None
        self.voxel = float(rospy.get_param("~voxel", 0.08))
        self.max_frames = int(rospy.get_param("~max_frames", 500))
        self.min_frames = int(rospy.get_param("~min_frames", 20))
        self.max_odom_dt = float(rospy.get_param("~max_odom_dt", 0.05))
        self.max_points_per_frame = int(rospy.get_param("~max_points_per_frame", 0))
        self.out_path = rospy.get_param(
            "~out",
            "/workspace/SimEnv/results/uf_map/livox_gt_global_map.ply",
        )
        self.base_to_laser_translation = np.array([0.2, 0.0, 0.08], dtype=np.float64)
        self.base_to_laser_quaternion = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
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

    def odom_callback(self, msg):
        self.odom.add(msg)

    def cloud_callback(self, msg):
        pts = np.array(
            list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)),
            dtype=np.float64,
        )
        self.frames += 1
        if pts.size == 0:
            return
        pts = pts.reshape((-1, 3))
        if self.max_points_per_frame > 0 and len(pts) > self.max_points_per_frame:
            pts = pts[:self.max_points_per_frame]

        pose, dt = self.odom.nearest(msg.header.stamp.to_sec(), self.max_odom_dt)
        if dt is not None:
            self.max_observed_dt = max(self.max_observed_dt, dt)
        if pose is None:
            self.dropped_no_odom += 1
            return

        base_points = transform_points(
            pts,
            self.base_to_laser_translation,
            self.base_to_laser_quaternion,
        )
        odom_translation, odom_quaternion = pose
        odom_points = transform_points(base_points, odom_translation, odom_quaternion)

        pose_xy = odom_translation[:2]
        if self.last_pose_xy is not None:
            self.path_length += float(np.linalg.norm(pose_xy - self.last_pose_xy))
        self.last_pose_xy = pose_xy.copy()

        should_exit = False
        with self.lock:
            self.used_frames += 1
            if self.points.size == 0:
                self.points = odom_points.astype(np.float32)
            else:
                self.points = voxel_unique(
                    np.vstack((self.points, odom_points.astype(np.float32))),
                    self.voxel,
                ).astype(np.float32)
            if self.used_frames % 25 == 0:
                rospy.loginfo(
                    "export_livox_gt_map: used_frames=%d points=%d path=%.3fm dropped=%d",
                    self.used_frames,
                    len(self.points),
                    self.path_length,
                    self.dropped_no_odom,
                )
            should_exit = self.max_frames > 0 and self.used_frames >= self.max_frames

        if should_exit:
            self.dump()
            os._exit(0)

    def dump(self):
        with self.lock:
            points = self.points.copy()
            frames = self.used_frames
            dropped = self.dropped_no_odom
            path_length = self.path_length
            max_dt = self.max_observed_dt
        if frames < self.min_frames:
            rospy.logwarn("only %d frames collected, below min_frames=%d", frames, self.min_frames)
        if points.size == 0:
            rospy.logwarn("no points to write")
            return False
        write_ply_xyz(self.out_path, points)
        meta_path = os.path.splitext(self.out_path)[0] + ".txt"
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        with open(meta_path, "w", encoding="ascii") as f:
            f.write(f"frames={frames}\n")
            f.write(f"input_frames={self.frames}\n")
            f.write(f"dropped_no_odom={dropped}\n")
            f.write(f"points={len(points)}\n")
            f.write(f"voxel={self.voxel}\n")
            f.write(f"path_length_xy={path_length}\n")
            f.write(f"max_odom_dt={max_dt}\n")
            f.write("bbox_min=%.6f %.6f %.6f\n" % tuple(bbox_min))
            f.write("bbox_max=%.6f %.6f %.6f\n" % tuple(bbox_max))
        rospy.loginfo("wrote %d points from %d frames to %s", len(points), frames, self.out_path)
        return True


def self_test():
    pts = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    moved = transform_points(pts, [1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(moved, [[2.0, 2.0, 3.0]])
    dedup = voxel_unique(np.array([[0.01, 0.01, 0.0], [0.02, 0.02, 0.0], [0.2, 0.0, 0.0]]), 0.1)
    assert len(dedup) == 2, dedup
    print("export_livox_gt_map self-test passed")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--self-test", action="store_true")
    args, _ = parser.parse_known_args()
    if args.self_test:
        self_test()
        return

    rospy.init_node("export_livox_gt_map")
    exporter = LivoxGTMapExporter()
    exporter.set_static_transform()
    lidar_topic = rospy.get_param("~lidar", "/livox/lidar2")
    odom_topic = rospy.get_param("~odom", "/Odometry_gazebo")
    rospy.Subscriber(odom_topic, Odometry, exporter.odom_callback, queue_size=200)
    rospy.Subscriber(lidar_topic, PointCloud2, exporter.cloud_callback, queue_size=20)
    rospy.on_shutdown(exporter.dump)
    rospy.loginfo(
        "export_livox_gt_map: lidar=%s odom=%s out=%s voxel=%.3f max_frames=%d",
        lidar_topic,
        odom_topic,
        exporter.out_path,
        exporter.voxel,
        exporter.max_frames,
    )
    rospy.spin()


if __name__ == "__main__":
    main()
