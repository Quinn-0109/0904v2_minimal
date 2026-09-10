#!/usr/bin/env python3
"""
grid_map.py — 把累积的 map 系点云渲染成 2D 占据栅格地图 (标准 SLAM 地图风格)。

订阅 /registered_scan 累积全场景点云, 每次 dump:
  1. 切出"墙体高度"切片 (z ∈ [z_min, z_max]), 排除地面/天花板
  2. 在 XY 平面做 2D 直方图 (分辨率 grid_res), 点数 >= occ_thr 的格子 = 占据(墙)
  3. 渲染 PNG: 占据=黑, 空闲=白, 未知=浅灰; 叠加浅色栅格线 + 机器人轨迹
  4. 也存 .pcd (全部点) 和 .yaml (地图元信息, 供 nav_msgs/OccupancyGrid 用)

用法: rosrun simenv_bridge grid_map.py
        _input:=/registered_scan  _odom:=/Odometry_gazebo
        _out_dir:=/workspace/SimEnv/results/map
"""
import os
import math
import threading
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class GridMapper:
    def __init__(self):
        self.lock = threading.Lock()
        self.pts = np.zeros((0, 3), dtype=np.float32)
        self.traj = []
        self.voxel = float(rospy.get_param("~voxel", 0.08))      # 累积去重体素
        self.grid_res = float(rospy.get_param("~grid_res", 0.05)) # 栅格分辨率(m)
        self.z_min = float(rospy.get_param("~z_min", 0.15))       # 墙体下沿
        self.z_max = float(rospy.get_param("~z_max", 1.30))       # 墙体上沿
        self.occ_thr = int(rospy.get_param("~occ_thr", 3))        # 格内点数阈值
        self.out_dir = rospy.get_param("~out_dir", "/workspace/SimEnv/results/map")
        self.show_path = bool(rospy.get_param("~show_path", False))  # 默认不画轨迹(teleport 轨迹会穿墙)
        os.makedirs(self.out_dir, exist_ok=True)

    def add_cloud(self, msg):
        arr = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)), dtype=np.float32)
        if arr.size:
            # 裁剪离群点: Livox 偶发返回最大量程(40m+)或 teleport 瞬间配准错位会产生
            # 极远点(几百米外), 把栅格撑到几百米, 建筑缩成一坨。按建筑边界 + 余量裁剪。
            xb = float(rospy.get_param("~x_max", 15.0))
            inb = (arr[:,0] >= -xb) & (arr[:,0] <= xb) & (arr[:,1] >= -5) & (arr[:,1] <= 40) & (arr[:,2] >= -1) & (arr[:,2] <= 5)
            arr = arr[inb]
            if arr.size:
                with self.lock:
                    self.pts = np.vstack([self.pts, arr[:, :3]])

    def add_odom(self, msg):
        self.traj.append((msg.pose.pose.position.x, msg.pose.pose.position.y))

    def _downsample(self):
        if self.pts.shape[0] == 0:
            return self.pts
        q = np.floor(self.pts / self.voxel).astype(np.int64)
        _, idx = np.unique(q, axis=0, return_index=True)
        return self.pts[idx]

    def dump(self, _event=None):
        with self.lock:
            ds = self._downsample()
            traj = list(self.traj)
        if ds.shape[0] < 10:
            rospy.loginfo("grid_map: not enough points (%d)", ds.shape[0])
            return
        # 墙体切片
        wall = ds[(ds[:, 2] >= self.z_min) & (ds[:, 2] <= self.z_max)]
        if wall.shape[0] < 10:
            rospy.loginfo("grid_map: no wall-slice points")
            return
        # XY 范围 (用全部点+轨迹定边界, 让地图框含整个走过的区域)
        allxy = ds[:, :2]
        if traj:
            tj = np.array(traj)
            allxy = np.vstack([allxy, tj])
        xmin, ymin = allxy.min(axis=0) - 1.0
        xmax, ymax = allxy.max(axis=0) + 1.0
        g = self.grid_res
        nx = int(math.ceil((xmax - xmin) / g))
        ny = int(math.ceil((ymax - ymin) / g))
        # 2D 直方图 (image 行=y, 列=x)
        H, _, _ = np.histogram2d(wall[:, 1], wall[:, 0],
                                 bins=[ny, nx],
                                 range=[[ymin, ymax], [xmin, xmax]])
        # 占据: 0=空闲(白), 1=占据(黑), 2=未知(灰)
        occ = np.full((ny, nx), 2, dtype=np.uint8)  # 默认未知
        occ[H >= self.occ_thr] = 1                   # 墙
        # 空闲: 在轨迹周围 radius 内且非墙 → 自由
        if traj:
            for (px, py) in traj:
                ix = int((px - xmin) / g); iy = int((py - ymin) / g)
                r = int(round(3.0 / g))  # 传感器半径 ~3m
                y0, y1 = max(0, iy - r), min(ny, iy + r + 1)
                x0, x1 = max(0, ix - r), min(nx, ix + r + 1)
                sub = occ[y0:y1, x0:x1]
                sub[sub != 1] = 0  # 非墙格标为空闲
        # 渲染
        cmap = ListedColormap(["white", "black", "#bbbbbb"])  # 0白/1黑/2灰
        fig, ax = plt.subplots(figsize=(ny / 50.0 + 2, nx / 50.0 + 2))
        ax.imshow(occ, cmap=cmap, vmin=0, vmax=2, origin="lower",
                  extent=[xmin, xmax, ymin, ymax], interpolation="nearest")
        # 浅色栅格线
        for gx in np.arange(math.floor(xmin), math.ceil(xmax), 1.0):
            ax.axvline(gx, color="#cccccc", lw=0.3, alpha=0.5)
        for gy in np.arange(math.floor(ymin), math.ceil(ymax), 1.0):
            ax.axhline(gy, color="#cccccc", lw=0.3, alpha=0.5)
        if self.show_path and len(traj) > 1:
            tx = [p[0] for p in traj]; ty = [p[1] for p in traj]
            ax.plot(tx, ty, "-", color="royalblue", lw=1.0, alpha=0.7)
            ax.scatter(tx[0], ty[0], c="lime", s=25, zorder=5)
            ax.scatter(tx[-1], ty[-1], c="red", marker="x", s=40, zorder=5)
        ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        n_occ = int((occ == 1).sum()); n_free = int((occ == 0).sum())
        ax.set_title("Occupancy grid (%.2fm, %dx%d)  wall=%d free=%d pts=%d"
                     % (g, nx, ny, n_occ, n_free, ds.shape[0]))
        png = os.path.join(self.out_dir, "grid_map.png")
        fig.savefig(png, dpi=150, bbox_inches="tight"); plt.close(fig)
        # PCD (全部点)
        pcd = os.path.join(self.out_dir, "grid_map.pcd")
        with open(pcd, "w") as f:
            f.write("# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n")
            f.write("WIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA ascii\n"
                    % (ds.shape[0], ds.shape[0]))
            np.savetxt(f, ds, fmt="%.3f")
        rospy.loginfo("grid_map: %dx%d grid, wall=%d free=%d, pts=%d -> %s",
                      nx, ny, n_occ, n_free, ds.shape[0], png)

    @staticmethod
    def _path_len(traj):
        if len(traj) < 2: return 0.0
        a = np.array(traj); return float(np.sum(np.hypot(np.diff(a[:, 0]), np.diff(a[:, 1]))))


def main():
    rospy.init_node("grid_map")
    gm = GridMapper()
    inp = rospy.get_param("~input", "/registered_scan")
    odom = rospy.get_param("~odom", "/Odometry_gazebo")
    period = float(rospy.get_param("~dump_period", 15.0))
    rospy.Subscriber(inp, PointCloud2, gm.add_cloud, queue_size=20)
    rospy.Subscriber(odom, Odometry, gm.add_odom, queue_size=100)
    rospy.Timer(rospy.Duration(period), gm.dump)
    rospy.loginfo("grid_map: accum %s, res=%.2f z=[%.2f,%.2f] -> %s", inp, gm.grid_res, gm.z_min, gm.z_max, gm.out_dir)
    rospy.on_shutdown(lambda: gm.dump())
    rospy.spin()


if __name__ == "__main__":
    main()
