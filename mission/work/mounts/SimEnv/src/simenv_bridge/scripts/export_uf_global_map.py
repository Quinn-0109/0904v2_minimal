#!/usr/bin/env python3
"""
Accumulate Ultra-Fusion /curr_cloud into a real global point cloud map.

The current UF binary in this environment publishes /curr_cloud in the `world`
frame, but /voxel_map_cloud` is advertised without emitting messages. This
script builds a global map by voxel-deduplicating consecutive /curr_cloud
frames and exporting a PLY artifact.

Usage:
  rosrun simenv_bridge export_uf_global_map.py
  rosrun simenv_bridge export_uf_global_map.py _input:=/curr_cloud _out:=/workspace/SimEnv/results/uf_map/uf_global_cloud.ply
  python3 export_uf_global_map.py --self-test
"""

import argparse
import os
import struct
import threading

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2


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


class UFGlobalMapExporter:
    def __init__(self):
        self.lock = threading.Lock()
        self.frames = 0
        self.last_stamp = None
        self.points = np.zeros((0, 3), dtype=np.float32)
        self.voxel = float(rospy.get_param("~voxel", 0.10))
        self.out_path = rospy.get_param(
            "~out",
            "/workspace/SimEnv/results/uf_map/uf_global_cloud.ply",
        )
        self.min_frames = int(rospy.get_param("~min_frames", 20))
        self.max_frames = int(rospy.get_param("~max_frames", 0))
        self.dump_on_shutdown = bool(rospy.get_param("~dump_on_shutdown", True))
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)

    def add_cloud(self, msg):
        pts = np.array(
            list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)),
            dtype=np.float32,
        )
        if pts.size == 0:
            return
        pts = pts.reshape((-1, 3))

        with self.lock:
            self.frames += 1
            self.last_stamp = msg.header.stamp
            if self.points.size == 0:
                self.points = pts
            else:
                merged = np.vstack((self.points, pts))
                self.points = voxel_unique(merged, self.voxel)

            if self.frames % 20 == 0:
                rospy.loginfo(
                    "export_uf_global_map: frames=%d points=%d frame_id=%s",
                    self.frames,
                    self.points.shape[0],
                    msg.header.frame_id,
                )

            if self.max_frames > 0 and self.frames >= self.max_frames:
                self.dump_locked()
                rospy.signal_shutdown("reached max_frames")
                os._exit(0)

    def dump_locked(self):
        points = self.points.copy()
        frames = self.frames
        stamp = self.last_stamp
        if frames < self.min_frames:
            rospy.logwarn(
                "export_uf_global_map: only %d frames collected, below min_frames=%d",
                frames,
                self.min_frames,
            )
        if points.size == 0:
            rospy.logwarn("export_uf_global_map: no points to write")
            return False

        write_ply_xyz(self.out_path, points)
        meta_path = os.path.splitext(self.out_path)[0] + ".txt"
        with open(meta_path, "w", encoding="ascii") as f:
            f.write(f"frames={frames}\n")
            f.write(f"points={points.shape[0]}\n")
            f.write(f"voxel={self.voxel}\n")
            if stamp is not None:
                f.write(f"stamp={stamp.secs}.{stamp.nsecs:09d}\n")
        rospy.loginfo(
            "export_uf_global_map: wrote %d points from %d frames to %s",
            points.shape[0],
            frames,
            self.out_path,
        )
        return True

    def dump(self):
        with self.lock:
            return self.dump_locked()


def self_test():
    pts = np.array(
        [
            [0.01, 0.01, 0.00],
            [0.02, 0.02, 0.00],
            [0.11, 0.01, 0.00],
            [0.11, 0.02, 0.01],
        ],
        dtype=np.float32,
    )
    dedup = voxel_unique(pts, 0.1)
    assert dedup.shape[0] == 2, dedup

    out = "/tmp/export_uf_global_map_test.ply"
    write_ply_xyz(out, dedup)
    with open(out, "rb") as f:
        data = f.read()
    assert b"element vertex 2" in data
    assert data.endswith(struct.pack("<fff", *dedup[-1]))
    print("export_uf_global_map self-test passed")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--self-test", action="store_true")
    args, _ = parser.parse_known_args()
    if args.self_test:
        self_test()
        return

    rospy.init_node("export_uf_global_map")
    exporter = UFGlobalMapExporter()
    topic = rospy.get_param("~input", "/curr_cloud")
    rospy.Subscriber(topic, PointCloud2, exporter.add_cloud, queue_size=10)
    rospy.loginfo(
        "export_uf_global_map: input=%s out=%s voxel=%.3f min_frames=%d max_frames=%d",
        topic,
        exporter.out_path,
        exporter.voxel,
        exporter.min_frames,
        exporter.max_frames,
    )
    if exporter.dump_on_shutdown:
        rospy.on_shutdown(exporter.dump)
    rospy.spin()


if __name__ == "__main__":
    main()
