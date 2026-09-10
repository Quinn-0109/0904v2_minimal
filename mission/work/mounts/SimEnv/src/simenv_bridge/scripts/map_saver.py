#!/usr/bin/env python3
"""
map_saver.py — 累积 /registered_scan 构建 TARE 探索地图并渲染俯视图。

订阅 /registered_scan (map 系 PointCloud2)，按机器人位姿配准后累积到一张
全局点云，降采样，每 dump_period 秒输出一次:
  - PCD 文件 (供 rviz/CloudCompare 查看)
  - 俯视 PNG 占据图 (高度配色 + 机器人轨迹)

用法: rosrun simenv_bridge map_saver.py
        _input:=/registered_scan  _odom:=/Odometry_gazebo
        _out_dir:=/workspace/SimEnv/results/map
"""
import os
import math
import struct
import threading
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


class MapBuilder:
    def __init__(self):
        self.lock = threading.Lock()
        # 累积点: Nx3 (xyz) + intensity; 用 voxel 降采样去重
        self.pts = np.zeros((0, 3), dtype=np.float32)
        self.traj = []  # [(x,y), ...]
        self.voxel = float(rospy.get_param("~voxel", 0.15))
        self.out_dir = rospy.get_param("~out_dir", "/workspace/SimEnv/results/map")
        os.makedirs(self.out_dir, exist_ok=True)

    def add_cloud(self, msg):
        gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        arr = np.array(list(gen), dtype=np.float32)
        if arr.size == 0:
            return
        with self.lock:
            self.pts = np.vstack([self.pts, arr[:, :3]])

    def add_odom(self, msg):
        self.traj.append((msg.pose.pose.position.x, msg.pose.pose.position.y))

    def _downsample(self):
        if self.pts.shape[0] == 0:
            return self.pts
        v = self.voxel
        quantized = np.floor(self.pts / v).astype(np.int64)
        uniq, idx = np.unique(quantized, axis=0, return_index=True)
        return self.pts[idx]

    def dump(self, _event=None):
        with self.lock:
            ds = self._downsample()
            traj = list(self.traj)
        if ds.shape[0] == 0:
            rospy.loginfo("map_saver: no points yet")
            return
        # PCD (ascii)
        pcd = os.path.join(self.out_dir, "tare_map.pcd")
        with open(pcd, "w") as f:
            f.write("# .PCD v0.7 - Point Cloud Data file format\n")
            f.write("VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n")
            f.write("WIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA ascii\n"
                    % (ds.shape[0], ds.shape[0]))
            np.savetxt(f, ds, fmt="%.3f")
        # PNG top-down
        if HAS_MPL:
            fig, ax = plt.subplots(figsize=(10, 8))
            # 颜色按高度 z
            z = ds[:, 2]
            sc = ax.scatter(ds[:, 0], ds[:, 1], c=z, cmap="terrain", s=1, alpha=0.8)
            if len(traj) > 1:
                tx = [p[0] for p in traj]; ty = [p[1] for p in traj]
                ax.plot(tx, ty, "-r", lw=1.2, alpha=0.7, label="robot path")
                ax.scatter(tx[0], ty[0], c="lime", s=40, zorder=5, label="start")
                ax.scatter(tx[-1], ty[-1], c="red", marker="x", s=60, zorder=5, label="now")
            ax.set_aspect("equal")
            ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
            ax.set_title("TARE exploration map (%d pts, path %.1f m)"
                         % (ds.shape[0], self._path_len(traj)))
            ax.legend(loc="upper right", fontsize=8)
            fig.colorbar(sc, ax=ax, label="height z [m]", shrink=0.8)
            png = os.path.join(self.out_dir, "tare_map.png")
            fig.savefig(png, dpi=120, bbox_inches="tight")
            plt.close(fig)
        rospy.loginfo("map_saver: dumped %d pts -> %s (PNG=%s)", ds.shape[0], pcd, HAS_MPL)

    @staticmethod
    def _path_len(traj):
        if len(traj) < 2:
            return 0.0
        a = np.array(traj)
        return float(np.sum(np.hypot(np.diff(a[:, 0]), np.diff(a[:, 1]))))


def main():
    rospy.init_node("map_saver")
    mb = MapBuilder()
    inp = rospy.get_param("~input", "/registered_scan")
    odom = rospy.get_param("~odom", "/Odometry_gazebo")
    period = float(rospy.get_param("~dump_period", 15.0))
    rospy.Subscriber(inp, PointCloud2, mb.add_cloud, queue_size=10)
    rospy.Subscriber(odom, Odometry, mb.add_odom, queue_size=50)
    rospy.Timer(rospy.Duration(period), mb.dump)
    rospy.loginfo("map_saver: accum %s + %s -> %s (every %.0fs)", inp, odom, mb.out_dir, period)
    rospy.on_shutdown(lambda: mb.dump())
    rospy.spin()


if __name__ == "__main__":
    main()
