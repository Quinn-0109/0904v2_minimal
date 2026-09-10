#!/usr/bin/env python3
"""Record TARE-driven exploration and build a GT-pose global map."""
import csv
import json
import math
import os
import struct
import threading
import time

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2


def yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


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
    q = np.floor(points / voxel_size).astype(np.int64)
    _, idx = np.unique(q, axis=0, return_index=True)
    idx.sort()
    return points[idx]


def xy_cell(point_xy, grid_res):
    return (int(math.floor(point_xy[0] / grid_res)), int(math.floor(point_xy[1] / grid_res)))


def raycast_xy_cells(origin_xy, point_xy, grid_res):
    x0, y0 = xy_cell(origin_xy, grid_res)
    x1, y1 = xy_cell(point_xy, grid_res)
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    free = set()
    while (x, y) != (x1, y1):
        free.add((x, y))
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return free, (x1, y1)


def accumulate_scan_occupancy(free_cells, occupied_cells, origin_xy, world_points,
                              grid_res, z_min, z_max, max_rays=1200):
    if world_points.size == 0:
        return
    wall = world_points[
        np.isfinite(world_points).all(axis=1) &
        (world_points[:, 2] >= z_min) &
        (world_points[:, 2] <= z_max)
    ]
    if len(wall) == 0:
        return
    if len(wall) > max_rays:
        step = int(math.ceil(len(wall) / float(max_rays)))
        wall = wall[::step]
    for point in wall:
        ray_free, endpoint = raycast_xy_cells(origin_xy, (float(point[0]), float(point[1])), grid_res)
        free_cells.update(ray_free)
        occupied_cells.add(endpoint)


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


def filter_global_points(points, layout, xy_margin=0.5, z_margin=0.8):
    if points.size == 0:
        return points.reshape((0, 3))
    wall_height = float(layout.get("wall_height", 3.0))
    valid = (
        np.isfinite(points).all(axis=1) &
        (points[:, 2] >= -z_margin) &
        (points[:, 2] <= wall_height + z_margin)
    )
    region_mask = np.zeros(len(points), dtype=bool)
    for _name, bounds in iter_regions(layout):
        region_mask |= (
            (points[:, 0] >= bounds["x_min"] - xy_margin) &
            (points[:, 0] <= bounds["x_max"] + xy_margin) &
            (points[:, 1] >= bounds["y_min"] - xy_margin) &
            (points[:, 1] <= bounds["y_max"] + xy_margin)
        )
    return points[valid & region_mask]


def iter_regions(layout):
    floor = layout["floors"][0]
    yield "lobby", floor["lobby_bounds"]
    yield "corridor", floor["corridor_bounds"]
    for room in floor.get("rooms", []):
        yield room.get("id", "room"), room["bounds"]


def point_region(layout, x, y, margin=0.0):
    for name, bounds in iter_regions(layout):
        if (bounds["x_min"] - margin <= x <= bounds["x_max"] + margin and
                bounds["y_min"] - margin <= y <= bounds["y_max"] + margin):
            return name
    return "outside"


def region_bounds(layout, region_name):
    for name, bounds in iter_regions(layout):
        if name == region_name:
            return bounds
    return None


def filter_points_to_robot_region(points, layout, robot_x, robot_y, margin=0.2):
    if points.size == 0:
        return points.reshape((0, 3))
    region = point_region(layout, robot_x, robot_y, margin=margin)
    bounds = region_bounds(layout, region)
    if bounds is None:
        return points[:0].reshape((0, 3))
    mask = (
        (points[:, 0] >= bounds["x_min"] - margin) &
        (points[:, 0] <= bounds["x_max"] + margin) &
        (points[:, 1] >= bounds["y_min"] - margin) &
        (points[:, 1] <= bounds["y_max"] + margin)
    )
    return points[mask]


def path_length_xy(rows):
    if len(rows) < 2:
        return 0.0
    total = 0.0
    last = rows[0]
    for row in rows[1:]:
        total += math.hypot(row[1] - last[1], row[2] - last[2])
        last = row
    return total


def _cell_center(cell, grid_res):
    return ((cell[0] + 0.5) * grid_res, (cell[1] + 0.5) * grid_res)


def _cells_in_bounds(cells, bounds, grid_res):
    selected = set()
    for cell in cells:
        x, y = _cell_center(cell, grid_res)
        if bounds["x_min"] <= x <= bounds["x_max"] and bounds["y_min"] <= y <= bounds["y_max"]:
            selected.add(cell)
    return selected


def _total_region_cells(bounds, grid_res):
    nx = max(1, int(math.ceil((bounds["x_max"] - bounds["x_min"]) / grid_res)))
    ny = max(1, int(math.ceil((bounds["y_max"] - bounds["y_min"]) / grid_res)))
    return nx * ny


def compute_region_metrics(layout, trajectory, free_cells, occupied_cells, grid_res):
    metrics = {}
    for name, bounds in iter_regions(layout):
        rows = [row for row in trajectory if point_region(layout, row[1], row[2], margin=0.2) == name]
        total_cells = _total_region_cells(bounds, grid_res)
        region_free = _cells_in_bounds(free_cells, bounds, grid_res)
        region_occ = _cells_in_bounds(occupied_cells, bounds, grid_res)
        if rows:
            xs = [row[1] for row in rows]
            ys = [row[2] for row in rows]
            x_cov = min(1.0, max(0.0, (max(xs) - min(xs)) / max(bounds["x_max"] - bounds["x_min"], 1e-6)))
            y_cov = min(1.0, max(0.0, (max(ys) - min(ys)) / max(bounds["y_max"] - bounds["y_min"], 1e-6)))
        else:
            x_cov = 0.0
            y_cov = 0.0
        metrics[name] = {
            "trajectory_samples": len(rows),
            "trajectory_path": path_length_xy(rows),
            "trajectory_x_coverage": x_cov,
            "trajectory_y_coverage": y_cov,
            "free_cell_coverage": len(region_free) / float(total_cells),
            "occupied_cell_coverage": len(region_occ) / float(total_cells),
            "observed_cell_coverage": len(region_free | region_occ) / float(total_cells),
        }
    return metrics


class TareTraceMap:
    def __init__(self):
        self.out_dir = rospy.get_param("~out_dir", "/workspace/SimEnv/results/tare_trace_map")
        self.layout_path = rospy.get_param("~layout", "/workspace/SimEnv/generated_building/layout_metadata.json")
        self.odom_topic = rospy.get_param("~odom", "/Odometry_gazebo")
        self.scan_topic = rospy.get_param("~scan", "/scan")
        self.waypoint_topic = rospy.get_param("~waypoint", "/way_point")
        self.cmd_topic = rospy.get_param("~cmd_vel", "/cmd_vel")
        self.duration = float(rospy.get_param("~duration", 120.0))
        self.dump_period = float(rospy.get_param("~dump_period", 10.0))
        self.voxel = float(rospy.get_param("~voxel", 0.08))
        self.frame_voxel = float(rospy.get_param("~frame_voxel", max(self.voxel, 0.12)))
        self.max_points_per_frame = int(rospy.get_param("~max_points_per_frame", 2500))
        self.grid_res = float(rospy.get_param("~grid_res", 0.08))
        self.occ_thr = int(rospy.get_param("~occ_thr", 2))
        self.z_min = float(rospy.get_param("~z_min", 0.15))
        self.z_max = float(rospy.get_param("~z_max", 1.6))
        self.max_range = float(rospy.get_param("~max_range", 35.0))
        self.robot_z_offset = float(rospy.get_param("~robot_z_offset", 0.0))
        self.base_to_laser_translation = np.array(rospy.get_param("~base_to_laser_translation", [0.2, 0.0, 0.08]), dtype=np.float64)
        self.base_to_laser_quaternion = np.array(rospy.get_param("~base_to_laser_quaternion", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64)
        self.layout = json.load(open(self.layout_path, "r", encoding="utf-8"))

        self.lock = threading.Lock()
        self.pose = None
        self.point_chunks = []
        self.free_cells = set()
        self.occupied_cells = set()
        self.trajectory = []
        self.waypoints = []
        self.cmds = []
        self.frames = 0
        os.makedirs(self.out_dir, exist_ok=True)

    def on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q)
        stamp = msg.header.stamp.to_sec() if msg.header.stamp else rospy.Time.now().to_sec()
        row = [stamp, p.x, p.y, p.z, yaw]
        with self.lock:
            self.pose = (p.x, p.y, p.z, yaw, (q.x, q.y, q.z, q.w), stamp)
            if not self.trajectory or stamp - self.trajectory[-1][0] >= 0.1:
                self.trajectory.append(row)

    def on_waypoint(self, msg):
        p = msg.point
        stamp = msg.header.stamp.to_sec() if msg.header.stamp else rospy.Time.now().to_sec()
        with self.lock:
            self.waypoints.append([stamp, p.x, p.y, p.z, msg.header.frame_id or ""])

    def on_cmd(self, msg):
        stamp = rospy.Time.now().to_sec()
        with self.lock:
            self.cmds.append([stamp, msg.linear.x, msg.linear.y, msg.angular.z])

    def on_scan(self, msg):
        with self.lock:
            pose = self.pose
        if pose is None:
            return
        pts = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)), dtype=np.float64)
        if pts.size == 0:
            return
        pts = pts.reshape((-1, 3))
        ranges = np.linalg.norm(pts, axis=1)
        pts = pts[np.isfinite(ranges) & (ranges <= self.max_range)]
        if pts.size == 0:
            return
        x, y, z, _yaw, quat, _stamp = pose
        base_pts = transform_points(pts, self.base_to_laser_translation, self.base_to_laser_quaternion)
        world_pts = transform_points(base_pts, [x, y, z + self.robot_z_offset], quat)
        world_pts = filter_global_points(world_pts, self.layout, xy_margin=0.5, z_margin=0.8)
        if world_pts.size == 0:
            return
        accumulate_scan_occupancy(
            self.free_cells,
            self.occupied_cells,
            (x, y),
            filter_points_to_robot_region(world_pts, self.layout, x, y, margin=0.2),
            self.grid_res,
            self.z_min,
            self.z_max,
        )
        world_pts = voxel_unique(world_pts, self.frame_voxel)
        if len(world_pts) > self.max_points_per_frame:
            step = int(math.ceil(len(world_pts) / float(self.max_points_per_frame)))
            world_pts = world_pts[::step]
        with self.lock:
            self.point_chunks.append(world_pts.astype(np.float32))
            self.frames += 1

    def snapshot(self):
        with self.lock:
            chunks = list(self.point_chunks)
            trajectory = list(self.trajectory)
            waypoints = list(self.waypoints)
            cmds = list(self.cmds)
            frames = self.frames
            free_cells = set(self.free_cells)
            occupied_cells = set(self.occupied_cells)
        points = np.zeros((0, 3), dtype=np.float32)
        if chunks:
            points = voxel_unique(np.vstack(chunks), self.voxel).astype(np.float32)
        return points, trajectory, waypoints, cmds, frames, free_cells, occupied_cells

    def write_csv(self, name, rows, header):
        with open(os.path.join(self.out_dir, name), "w", newline="", encoding="ascii") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    def dump_grid(self, points, trajectory, free_cells=None, occupied_cells=None):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        if len(points) < 10:
            return None
        free_cells = free_cells or set()
        occupied_cells = occupied_cells or set()
        wall = points[(points[:, 2] >= self.z_min) & (points[:, 2] <= self.z_max)]
        if len(wall) < 10 and not occupied_cells:
            return None
        allxy = points[:, :2]
        if trajectory:
            allxy = np.vstack([allxy, np.asarray([[r[1], r[2]] for r in trajectory], dtype=np.float32)])
        xmin, ymin = allxy.min(axis=0) - 1.0
        xmax, ymax = allxy.max(axis=0) + 1.0
        nx = max(1, int(math.ceil((xmax - xmin) / self.grid_res)))
        ny = max(1, int(math.ceil((ymax - ymin) / self.grid_res)))
        occ = np.full((ny, nx), 2, dtype=np.uint8)
        if free_cells or occupied_cells:
            for cx, cy in free_cells:
                ix = int((cx * self.grid_res - xmin) / self.grid_res)
                iy = int((cy * self.grid_res - ymin) / self.grid_res)
                if 0 <= ix < nx and 0 <= iy < ny:
                    occ[iy, ix] = 0
            for cx, cy in occupied_cells:
                ix = int((cx * self.grid_res - xmin) / self.grid_res)
                iy = int((cy * self.grid_res - ymin) / self.grid_res)
                if 0 <= ix < nx and 0 <= iy < ny:
                    occ[iy, ix] = 1
        else:
            hist, _, _ = np.histogram2d(wall[:, 1], wall[:, 0], bins=[ny, nx], range=[[ymin, ymax], [xmin, xmax]])
            occ[hist >= self.occ_thr] = 1
        if trajectory and not free_cells:
            radius = int(round(2.0 / self.grid_res))
            for _, x, y, *_ in trajectory:
                ix = int((x - xmin) / self.grid_res)
                iy = int((y - ymin) / self.grid_res)
                sub = occ[max(0, iy-radius):min(ny, iy+radius+1), max(0, ix-radius):min(nx, ix+radius+1)]
                sub[sub != 1] = 0
        cmap = ListedColormap(["white", "black", "#bbbbbb"])
        fig, ax = plt.subplots(figsize=(max(5, (xmax - xmin) / 2.5), max(6, (ymax - ymin) / 2.5)))
        ax.imshow(occ, cmap=cmap, vmin=0, vmax=2, origin="lower", extent=[xmin, xmax, ymin, ymax], interpolation="nearest")
        if len(trajectory) > 1:
            tx = [r[1] for r in trajectory]
            ty = [r[2] for r in trajectory]
            ax.plot(tx, ty, color="royalblue", lw=1.2)
            ax.scatter(tx[0], ty[0], c="lime", s=25, zorder=5)
            ax.scatter(tx[-1], ty[-1], c="red", marker="x", s=35, zorder=5)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title("TARE occupancy grid")
        out = os.path.join(self.out_dir, "occupancy_grid.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out

    def dump_gt_overlay(self, trajectory, waypoints):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        fig, ax = plt.subplots(figsize=(8, 14))
        for name, bounds in iter_regions(self.layout):
            ax.add_patch(Rectangle(
                (bounds["x_min"], bounds["y_min"]),
                bounds["x_max"] - bounds["x_min"],
                bounds["y_max"] - bounds["y_min"],
                facecolor="#f3f3f3",
                edgecolor="black",
                lw=1.4,
                zorder=0,
            ))
            ax.text((bounds["x_min"] + bounds["x_max"]) / 2.0, (bounds["y_min"] + bounds["y_max"]) / 2.0,
                    name, ha="center", va="center", fontsize=7, color="#444444")
        if len(trajectory) > 1:
            tx = [r[1] for r in trajectory]
            ty = [r[2] for r in trajectory]
            ax.plot(tx, ty, "-", color="#0b5bd3", lw=1.8, label="odom trajectory")
            ax.scatter(tx[0], ty[0], c="lime", edgecolors="black", s=45, zorder=5, label="start")
            ax.scatter(tx[-1], ty[-1], c="red", marker="x", s=55, zorder=5, label="end")
        finite_wp = [w for w in waypoints if all(math.isfinite(v) for v in w[1:4])]
        if finite_wp:
            ax.scatter([w[1] for w in finite_wp], [w[2] for w in finite_wp], c="#f28e2b", s=8, alpha=0.6, label="TARE waypoints")
        bounds = [b for _, b in iter_regions(self.layout)]
        ax.set_xlim(min(b["x_min"] for b in bounds) - 2, max(b["x_max"] for b in bounds) + 2)
        ax.set_ylim(min(b["y_min"] for b in bounds) - 2, max(b["y_max"] for b in bounds) + 2)
        ax.set_aspect("equal")
        ax.grid(True, color="#dddddd", lw=0.4)
        ax.legend(loc="upper right")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title("TARE trajectory on GT layout")
        out = os.path.join(self.out_dir, "tare_on_gt_layout.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out

    def dump(self):
        points, trajectory, waypoints, cmds, frames, free_cells, occupied_cells = self.snapshot()
        self.write_csv("trajectory.csv", trajectory, ["t", "x", "y", "z", "yaw"])
        self.write_csv("waypoints.csv", waypoints, ["t", "x", "y", "z", "frame"])
        self.write_csv("cmd_vel.csv", cmds, ["t", "linear_x", "linear_y", "angular_z"])
        if len(points):
            write_ply_xyz(os.path.join(self.out_dir, "tare_global_map.ply"), points)
        grid = self.dump_grid(points, trajectory, free_cells, occupied_cells)
        overlay = self.dump_gt_overlay(trajectory, waypoints)

        region_counts = {name: 0 for name, _ in iter_regions(self.layout)}
        region_counts["outside"] = 0
        for row in trajectory:
            region_counts[point_region(self.layout, row[1], row[2], margin=0.2)] += 1
        region_metrics = compute_region_metrics(
            self.layout, trajectory, free_cells, occupied_cells, self.grid_res,
        )
        finite_wp = sum(1 for w in waypoints if all(math.isfinite(v) for v in w[1:4]))
        nan_wp = len(waypoints) - finite_wp
        with open(os.path.join(self.out_dir, "summary.txt"), "w", encoding="ascii") as f:
            f.write(f"frames={frames}\n")
            f.write(f"points={len(points)}\n")
            f.write(f"trajectory_samples={len(trajectory)}\n")
            f.write(f"waypoint_samples={len(waypoints)}\n")
            f.write(f"waypoint_finite={finite_wp}\n")
            f.write(f"waypoint_nan_or_inf={nan_wp}\n")
            f.write(f"cmd_samples={len(cmds)}\n")
            f.write(f"path_xy={path_length_xy(trajectory):.6f}\n")
            for key, value in region_counts.items():
                f.write(f"trajectory_region_{key}={value}\n")
            for key, metrics in region_metrics.items():
                safe_key = key.replace("/", "_")
                f.write(f"coverage_{safe_key}_trajectory_path={metrics['trajectory_path']:.6f}\n")
                f.write(f"coverage_{safe_key}_trajectory_x={metrics['trajectory_x_coverage']:.6f}\n")
                f.write(f"coverage_{safe_key}_trajectory_y={metrics['trajectory_y_coverage']:.6f}\n")
                f.write(f"coverage_{safe_key}_free_cells={metrics['free_cell_coverage']:.6f}\n")
                f.write(f"coverage_{safe_key}_occupied_cells={metrics['occupied_cell_coverage']:.6f}\n")
                f.write(f"coverage_{safe_key}_observed_cells={metrics['observed_cell_coverage']:.6f}\n")
            if len(points):
                f.write("bbox_min=%.6f %.6f %.6f\n" % tuple(points.min(axis=0)))
                f.write("bbox_max=%.6f %.6f %.6f\n" % tuple(points.max(axis=0)))
            if grid:
                f.write(f"occupancy_grid={grid}\n")
            if overlay:
                f.write(f"gt_overlay={overlay}\n")
        rospy.loginfo("tare_trace_map: wrote %s", self.out_dir)

    def run(self):
        rospy.Subscriber(self.odom_topic, Odometry, self.on_odom, queue_size=100)
        rospy.Subscriber(self.scan_topic, PointCloud2, self.on_scan, queue_size=20)
        rospy.Subscriber(self.waypoint_topic, PointStamped, self.on_waypoint, queue_size=50)
        rospy.Subscriber(self.cmd_topic, Twist, self.on_cmd, queue_size=100)
        rospy.on_shutdown(self.dump)
        start = time.monotonic()
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            if time.monotonic() - start >= self.duration:
                break
            rate.sleep()
        self.dump()


def main():
    rospy.init_node("tare_trace_map")
    node = TareTraceMap()
    rospy.loginfo("tare_trace_map: duration=%.1fs out=%s", node.duration, node.out_dir)
    node.run()


if __name__ == "__main__":
    main()
