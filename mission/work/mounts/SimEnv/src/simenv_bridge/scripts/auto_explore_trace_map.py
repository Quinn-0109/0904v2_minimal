#!/usr/bin/env python3
"""
Autonomous kinematic exploration with trace logging and GT-projected mapping.

This node publishes /cmd_vel for a deterministic layout-covering route. Pair it
with cmd_vel_to_model_state.py for stable kinematic motion. It accumulates
filtered /livox/lidar2 using /Odometry_gazebo, writes trace CSV files, and
exports a PLY map when the route finishes.
"""

import csv
from dataclasses import dataclass
import json
import math
import os
import struct
import threading
import time

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2

try:
    from unitree_guide.msg import CustomMsg
except ImportError:
    CustomMsg = None


@dataclass
class Waypoint:
    x: float
    y: float
    label: str


@dataclass
class Cmd:
    vx: float
    vy: float
    wz: float


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def clamp(value, low, high):
    return max(low, min(high, value))


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


def extract_xyz_points(msg):
    if hasattr(msg, "points"):
        return np.array([(p.x, p.y, p.z) for p in msg.points], dtype=np.float64).reshape((-1, 3))
    return np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)), dtype=np.float64).reshape((-1, 3))


def voxel_unique(points, voxel_size):
    if points.size == 0:
        return points.reshape((0, 3))
    quantized = np.floor(points / voxel_size).astype(np.int64)
    _, idx = np.unique(quantized, axis=0, return_index=True)
    idx.sort()
    return points[idx]


def filter_global_points(points, layout, z_margin=0.8):
    if points.size == 0:
        return points.reshape((0, 3))
    wall_height = float(layout.get("wall_height", 3.0))
    finite = np.isfinite(points).all(axis=1)
    height_ok = (points[:, 2] >= -z_margin) & (points[:, 2] <= wall_height + z_margin)
    return points[finite & height_ok]


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


# --- ponytail: Bresenham free-space clearing (ported from tare_trace_map.py). ---
# Ray from base(xy) to each body-level hit: ray cells -> FREE, hit cell -> OCCUPIED.
# This explicitly marks door-gap cells free (structural fix: density-only grids relied on
# absence-of-points, which left door gaps unreliable -> path_clear over-skip).
def xy_cell(xy, res):
    return (int(math.floor(xy[0] / res)), int(math.floor(xy[1] / res)))


def raycast_xy_cells(origin_xy, point_xy, grid_res):
    x0, y0 = xy_cell(origin_xy, grid_res)
    x1, y1 = xy_cell(point_xy, grid_res)
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err, x, y = dx - dy, x0, y0
    free = set()
    while (x, y) != (x1, y1):
        free.add((x, y))
        e2 = 2 * err
        if e2 > -dy:
            err -= dy; x += sx
        if e2 < dx:
            err += dx; y += sy
    return free, (x1, y1)


def accumulate_scan_occupancy(free_cells, occupied_cells, origin_xy, world_points,
                              grid_res, z_min, z_max, max_rays=8000):
    if world_points.size == 0:
        return
    wall = world_points[np.isfinite(world_points).all(axis=1) &
                        (world_points[:, 2] >= z_min) & (world_points[:, 2] <= z_max)]
    if len(wall) == 0:
        return
    if len(wall) > max_rays:                       # deterministic subsample
        wall = wall[np.linspace(0, len(wall) - 1, max_rays).astype(int)]
    ox, oy = origin_xy
    for p in wall:
        ray_free, endpoint = raycast_xy_cells((ox, oy), (float(p[0]), float(p[1])), grid_res)
        free_cells.update(ray_free)
        occupied_cells.add(endpoint)


def add_sweep_waypoints(waypoints, bounds, label, x_values, y_step):
    y_min = bounds["y_min"] + 1.0
    y_max = bounds["y_max"] - 1.0
    if y_min > y_max:
        y_min = (bounds["y_min"] + bounds["y_max"]) * 0.5
        y_max = y_min
    y = y_min
    row = 0
    while y <= y_max + 1e-6:
        xs = x_values if row % 2 == 0 else list(reversed(x_values))
        for x in xs:
            waypoints.append(Waypoint(float(x), float(y), f"{label}_{row}"))
        y += y_step
        row += 1
    if not waypoints or waypoints[-1].y < y_max - y_step * 0.5:
        for x in x_values:
            waypoints.append(Waypoint(float(x), float(y_max), f"{label}_end"))


def plan_compact_waypoints(layout):
    floor = layout["floors"][0]
    lobby = floor["lobby_bounds"]
    corridor = floor["corridor_bounds"]
    waypoints = [
        Waypoint(0.0, max(0.5, lobby["y_min"] + 1.0), "entrance"),
        Waypoint(0.0, (lobby["y_min"] + lobby["y_max"]) * 0.5, "lobby_center"),
        Waypoint(0.0, lobby["y_max"] - 0.7, "lobby_exit"),
    ]
    sorted_rooms = sorted(floor["rooms"], key=lambda room: room["door_pose"][1])
    for room in sorted_rooms:
        door_x, door_y = room["door_pose"][:2]
        gx, gy = room["goal_pose"][:2]
        waypoints.append(Waypoint(0.0, float(door_y), f"{room['id']}_corridor"))
        waypoints.append(Waypoint(float(door_x), float(door_y), f"{room['id']}_door"))
        waypoints.append(Waypoint(float(gx), float(gy), f"{room['id']}_goal"))
        waypoints.append(Waypoint(0.0, float(door_y), f"{room['id']}_return_corridor"))
    waypoints.append(Waypoint(0.0, corridor["y_max"] - 1.0, "corridor_end"))
    return waypoints


def plan_exploration_waypoints(layout, y_step=6.0, route_mode="dense"):
    if route_mode == "compact":
        return plan_compact_waypoints(layout)

    floor = layout["floors"][0]
    waypoints = []
    lobby = floor["lobby_bounds"]
    waypoints.append(Waypoint(0.0, max(0.5, lobby["y_min"] + 1.0), "entrance"))
    waypoints.append(Waypoint(0.0, (lobby["y_min"] + lobby["y_max"]) * 0.5, "lobby_center"))
    waypoints.append(Waypoint(-4.5, (lobby["y_min"] + lobby["y_max"]) * 0.5, "lobby_left"))
    waypoints.append(Waypoint(4.5, (lobby["y_min"] + lobby["y_max"]) * 0.5, "lobby_right"))
    waypoints.append(Waypoint(0.0, lobby["y_max"] - 0.7, "lobby_exit"))

    corridor = floor["corridor_bounds"]
    add_sweep_waypoints(waypoints, corridor, "corridor", [0.0], y_step)
    for room in floor["rooms"]:
        bounds = room["bounds"]
        door_x, door_y = room["door_pose"][:2]
        inside_x = door_x - 1.0 if door_x < 0 else door_x + 1.0
        waypoints.append(Waypoint(0.0, float(door_y), f"{room['id']}_door_outside"))  # P3: was door_x (±1.1 wall plane -> A* goal blocked -> room skipped). Now corridor center.
        waypoints.append(Waypoint(float(inside_x), float(door_y), f"{room['id']}_door_inside"))
        x_mid = (bounds["x_min"] + bounds["x_max"]) * 0.5
        add_sweep_waypoints(waypoints, bounds, room["id"], [inside_x, x_mid], y_step)
        gx, gy = room["goal_pose"][:2]
        waypoints.append(Waypoint(float(gx), float(gy), f"{room['id']}_goal"))
        waypoints.append(Waypoint(0.0, float(door_y), f"{room['id']}_return_corridor"))
    return waypoints


def compute_cmd_to_waypoint(pose, target, v_max, w_max, goal_tol, slow_radius, command_frame="body"):
    x, y, yaw = pose
    dx = target.x - x
    dy = target.y - y
    dist = math.hypot(dx, dy)
    if dist <= goal_tol:
        return Cmd(0.0, 0.0, 0.0), True
    target_yaw = math.atan2(dy, dx)
    yaw_err = normalize_angle(target_yaw - yaw)
    speed = min(v_max, v_max * dist / max(slow_radius, 1e-3))
    if command_frame == "world":
        vx = clamp(0.8 * dx, -speed, speed)
        vy = clamp(0.8 * dy, -speed, speed)
    else:
        bx = math.cos(yaw) * dx + math.sin(yaw) * dy
        by = -math.sin(yaw) * dx + math.cos(yaw) * dy
        vx = clamp(0.8 * bx, -speed, speed)
        vy = clamp(0.8 * by, -speed, speed)
    wz = clamp(1.5 * yaw_err, -w_max, w_max)
    return Cmd(vx, vy, wz), False


class AutoExploreTraceMap:
    def __init__(self):
        self.layout_path = rospy.get_param("~layout", "/workspace/SimEnv/generated_building/layout_metadata.json")
        self.out_dir = rospy.get_param("~out_dir", "/workspace/SimEnv/results/auto_explore_trace")
        self.lidar_topic = rospy.get_param("~lidar", "/livox/lidar2")
        self.odom_topic = rospy.get_param("~odom", "/Odometry_gazebo")
        self.cmd_topic = rospy.get_param("~cmd_vel", "/cmd_vel")
        self.y_step = float(rospy.get_param("~y_step", 6.0))
        self.route_mode = rospy.get_param("~route_mode", "dense")
        self.v_max = float(rospy.get_param("~v_max", 1.0))
        self.w_max = float(rospy.get_param("~w_max", 0.7))
        self.goal_tol = float(rospy.get_param("~goal_tol", 0.6))
        self.slow_radius = float(rospy.get_param("~slow_radius", 2.0))
        self.stuck_timeout = float(rospy.get_param("~stuck_timeout", 20.0))   # ponytail: skip a waypoint not reached within N s. Without it the robot oscillates at goal_tol forever (1.6M samples in one room) + balloons the run -> recorder O(n^2) dumps -> RTF death-spiral.
        self.command_frame = rospy.get_param("~command_frame", "body")
        self.max_duration = float(rospy.get_param("~max_duration", 240.0))
        self.dump_period = float(rospy.get_param("~dump_period", 10.0))
        self.voxel = float(rospy.get_param("~voxel", 0.08))
        self.max_range = float(rospy.get_param("~max_range", 35.0))
        self.robot_z_offset = float(rospy.get_param("~robot_z_offset", 0.0))
        self.reset_model = bool(rospy.get_param("~reset_model", True))
        self.model_name = rospy.get_param("~model_name", "a1_gazebo")
        self.start_x = float(rospy.get_param("~start_x", 0.0))
        self.start_y = float(rospy.get_param("~start_y", -2.2))
        self.start_z = float(rospy.get_param("~start_z", 0.6))
        self.start_yaw = float(rospy.get_param("~start_yaw", 1.5708))
        self.base_to_laser_translation = np.array([0.2, 0.0, 0.08], dtype=np.float64)
        self.base_to_laser_quaternion = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.lock = threading.Lock()
        self.pose = None
        self.point_chunks = []
        self.trace_rows = []
        self.cmd_rows = []
        self.coverage_rows = []
        self.frames = 0
        self.last_dump_wall = 0.0
        # ponytail: Bresenham free-space-cleared occupancy (replaces density grid). free_cells (ray-cleared)
        # + occ_cells (hit endpoints). Door gaps are explicitly free -> A* (P2) can thread them; density
        # grid couldn't (door gap is topological, not density). P1 = dump grid + verify door gaps free.
        self.free_cells = set()
        self.occ_cells = set()
        self.occ_res = float(rospy.get_param("~occ_res", 0.15))   # 0.15: door 1.2m = 8 cells; coarser = denser per-cell for sparse Livox
        self.occ_zmin = float(rospy.get_param("~occ_zmin", 0.15))
        self.occ_zmax = float(rospy.get_param("~occ_zmax", 1.2))
        self.inflation = float(rospy.get_param("~inflation", 0.30))  # footprint; MUST be < door_width/2
        os.makedirs(self.out_dir, exist_ok=True)
        layout = json.load(open(self.layout_path, "r", encoding="utf-8"))
        self.layout = layout
        self.waypoints = plan_exploration_waypoints(layout, self.y_step, self.route_mode)
        # plan window = union of building bounds + 1m margin (clip planning/dump to building, avoid void)
        f0 = layout["floors"][0]
        _xs, _ys = [], []
        for key in ("lobby_bounds", "corridor_bounds", "elevator_bounds"):
            b = f0.get(key) or {}
            if b:
                _xs += [b["x_min"], b["x_max"]]; _ys += [b["y_min"], b["y_max"]]
        for r in f0.get("rooms", []):
            b = r.get("bounds", {})
            if b:
                _xs += [b["x_min"], b["x_max"]]; _ys += [b["y_min"], b["y_max"]]
        self.plan_window = (min(_xs) - 1.0, min(_ys) - 1.0, max(_xs) + 1.0, max(_ys) + 1.0)

    def reset_model_pose(self):
        if not self.reset_model:
            return
        from gazebo_msgs.msg import ModelState
        from gazebo_msgs.srv import SetModelState
        from geometry_msgs.msg import Pose, Quaternion, Twist as GazeboTwist

        rospy.wait_for_service("/gazebo/set_model_state", timeout=10.0)
        set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        state = ModelState()
        state.model_name = self.model_name
        state.reference_frame = "world"
        state.pose = Pose()
        state.pose.position.x = self.start_x
        state.pose.position.y = self.start_y
        state.pose.position.z = self.start_z
        state.pose.orientation = Quaternion(
            0.0,
            0.0,
            math.sin(self.start_yaw * 0.5),
            math.cos(self.start_yaw * 0.5),
        )
        state.twist = GazeboTwist()
        set_state(state)
        rospy.sleep(0.5)

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q)
        with self.lock:
            self.pose = (p.x, p.y, yaw, p.z, (q.x, q.y, q.z, q.w), msg.header.stamp.to_sec())

    def cloud_callback(self, msg):
        with self.lock:
            pose = self.pose
        if pose is None:
            return
        pts = extract_xyz_points(msg)
        if pts.size == 0:
            return
        ranges = np.linalg.norm(pts, axis=1)
        pts = pts[np.isfinite(ranges) & (ranges <= self.max_range)]
        if pts.size == 0:
            return
        x, y, _yaw, z, quat, _stamp = pose
        base_pts = transform_points(pts, self.base_to_laser_translation, self.base_to_laser_quaternion)
        odom_pts = transform_points(base_pts, [x, y, z + self.robot_z_offset], quat)
        with self.lock:
            self.point_chunks.append(odom_pts.astype(np.float32))
            self.frames += 1
            # Bresenham free-space clearing: ray base(xy) -> each body-level hit (free ray, occ endpoint)
            accumulate_scan_occupancy(self.free_cells, self.occ_cells, (x, y), odom_pts,
                                      self.occ_res, self.occ_zmin, self.occ_zmax)

    def region_counts(self, points):
        floor = self.layout["floors"][0]
        regions = [("lobby", floor["lobby_bounds"]), ("corridor", floor["corridor_bounds"])]
        regions.extend((room["id"], room["bounds"]) for room in floor["rooms"])
        counts = {name: 0 for name, _ in regions}
        for x, y, z in points:
            if not (-0.3 <= z <= self.layout["wall_height"] + 0.8):
                continue
            for name, bounds in regions:
                if bounds["x_min"] - 0.6 <= x <= bounds["x_max"] + 0.6 and bounds["y_min"] - 0.6 <= y <= bounds["y_max"] + 0.6:
                    counts[name] += 1
                    break
        return counts

    def dump_occupancy_png(self, path):
        # P1 verification: windowed grid PNG (free=white, occ=black, unknown=gray), y-up.
        # Goal: confirm door gaps (x=±1.1) are FREE so A* (P2) can thread them.
        try:
            import cv2
        except ImportError:
            return
        xmin, ymin, xmax, ymax = self.plan_window
        res = self.occ_res
        x0 = int(math.floor(xmin / res)); y0 = int(math.floor(ymin / res))
        nx = int(math.ceil((xmax - xmin) / res)); ny = int(math.ceil((ymax - ymin) / res))
        img = np.full((ny, nx), 127, np.uint8)  # unknown=gray
        for (cx, cy) in self.free_cells:
            i, j = cy - y0, cx - x0
            if 0 <= i < ny and 0 <= j < nx:
                img[i, j] = 255  # free=white
        for (cx, cy) in self.occ_cells:
            i, j = cy - y0, cx - x0
            if 0 <= i < ny and 0 <= j < nx:
                img[i, j] = 0  # occ=black (overrides free)
        cv2.imwrite(path, img[::-1, :])  # flip so y increases upward

    def dump(self):
        with self.lock:
            chunks = list(self.point_chunks)
            trace_rows = list(self.trace_rows)
            cmd_rows = list(self.cmd_rows)
            coverage_rows = list(self.coverage_rows)
            frames = self.frames
        points = np.zeros((0, 3), dtype=np.float32)
        if chunks:
            points = filter_global_points(np.vstack(chunks), self.layout).astype(np.float32)
            points = voxel_unique(points, self.voxel).astype(np.float32)
        map_path = os.path.join(self.out_dir, "auto_explore_map.ply")
        if len(points):
            write_ply_xyz(map_path, points)
        for name, rows, header in [
            ("trajectory.csv", trace_rows, ["t", "x", "y", "yaw", "waypoint_index", "waypoint_label"]),
            ("cmd_vel.csv", cmd_rows, ["t", "vx", "vy", "wz", "waypoint_index", "waypoint_label"]),
            ("coverage.csv", coverage_rows, ["t", "frames", "raw_points", "waypoint_index", "waypoint_label"]),
        ]:
            with open(os.path.join(self.out_dir, name), "w", newline="", encoding="ascii") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)
        meta_path = os.path.join(self.out_dir, "summary.txt")
        counts = self.region_counts(points) if len(points) else {}
        with open(meta_path, "w", encoding="ascii") as f:
            f.write(f"waypoints={len(self.waypoints)}\n")
            f.write(f"frames={frames}\n")
            f.write(f"points={len(points)}\n")
            f.write(f"voxel={self.voxel}\n")
            if len(points):
                f.write("bbox_min=%.6f %.6f %.6f\n" % tuple(points.min(axis=0)))
                f.write("bbox_max=%.6f %.6f %.6f\n" % tuple(points.max(axis=0)))
            for key, value in counts.items():
                f.write(f"region_{key}_points={value}\n")
        try:
            self.dump_occupancy_png(os.path.join(self.out_dir, "occupancy_bresenham.png"))
        except Exception as e:
            rospy.logwarn("auto_explore_trace_map: occupancy png failed: %s" % e)
        rospy.loginfo("auto_explore_trace_map: wrote trace/map to %s", self.out_dir)

    def is_occupied(self, x, y):
        return (int(math.floor(x / self.occ_res)), int(math.floor(y / self.occ_res))) in self.occ_cells

    def path_clear(self, x0, y0, x1, y1):
        # ponytail: ray-cast the straight segment (x0,y0)->(x1,y1) at cell resolution; False if any
        # cell occupied (density-thresholded). The kinematic driver teleports the base in straight
        # lines ignoring collisions, so without this the robot drives THROUGH walls/furniture (穿墙).
        if not self.occ_cells:
            return True
        d = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(d / (self.occ_res * 0.5)))
        for i in range(n + 1):
            t = i / n
            if self.is_occupied(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t):
                return False
        return True

    def build_traversable(self):
        """Return (blocked[ny,nx], (x0,y0,nx,ny)) for the plan window.
        blocked=1 where occupied (after denoise+morph open + inflate by footprint)."""
        import cv2
        xmin, ymin, xmax, ymax = self.plan_window
        res = self.occ_res
        x0 = int(math.floor(xmin / res)); y0 = int(math.floor(ymin / res))
        nx = int(math.ceil((xmax - xmin) / res)); ny = int(math.ceil((ymax - ymin) / res))
        occ = np.zeros((ny, nx), dtype=np.uint8)
        for (cx, cy) in self.occ_cells:
            i, j = cy - y0, cx - x0
            if 0 <= i < ny and 0 <= j < nx:
                occ[i, j] = 1
        for (cx, cy) in self.stuck_blacklist:  # stuck cells = permanent obstacles
            i, j = cy - y0, cx - x0
            if 0 <= i < ny and 0 <= j < nx:
                occ[i, j] = 1
        occ = cv2.morphologyEx(occ, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # denoise
        r = max(1, int(round(self.inflation / res)))
        occ = cv2.dilate(occ, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        # P3: force-clear known door gaps (from layout) so A* can ALWAYS thread doors
        # regardless of how dense the grid + inflation gets. Without this, later rooms become
        # unreachable as walls are sensed more densely and inflation closes the gaps.
        for room in self.layout["floors"][0].get("rooms", []):
            dp = room.get("door_pose")
            if not dp:
                continue
            dw = float(room.get("door_width", 1.2)) / 2.0 + 0.15
            for wx in np.arange(dp[0] - 0.6, dp[0] + 0.61, res * 0.5):  # wider: cover full inflated wall
                j = int(math.floor(wx / res)) - x0
                for wy in np.arange(dp[1] - dw, dp[1] + dw, res * 0.5):
                    i = int(math.floor(wy / res)) - y0
                    if 0 <= i < ny and 0 <= j < nx:
                        occ[i, j] = 0
        return occ, (x0, y0, nx, ny)

    def _astar(self, sx, sy, gx, gy, blocked, nx, ny):
        import heapq
        if not (0 <= gy < ny and 0 <= gx < nx) or blocked[gy, gx]:
            return None
        if not (0 <= sy < ny and 0 <= sx < nx) or blocked[sy, sx]:
            blocked[sy, sx] = 0  # robot in occ cell — clear a bubble
        h0 = [(0, sx, sy)]; came = {(sx, sy): None}; g = {(sx, sy): 0}
        nb = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        while h0:
            _, cx, cy = heapq.heappop(h0)
            if (cx, cy) == (gx, gy):
                p = []; n = (cx, cy)
                while n is not None:
                    p.append(n); n = came[n]
                return p[::-1]
            for dx, dy in nb:
                ax, ay = cx + dx, cy + dy
                if not (0 <= ay < ny and 0 <= ax < nx) or blocked[ay, ax]:
                    continue
                w = 1.414 if (dx and dy) else 1.0
                t = g[(cx, cy)] + w
                if (ax, ay) not in g or t < g[(ax, ay)]:
                    g[(ax, ay)] = t
                    heapq.heappush(h0, (t + math.hypot(ax - gx, ay - gy), ax, ay))
                    came[(ax, ay)] = (cx, cy)
        return None

    def plan_path(self, pose_xy, goal_xy):
        """A* from pose to goal on the traversable grid. Returns [(x,y),...] sub-waypoints, or None."""
        if len(self.occ_cells) < 2000:
            return [(goal_xy[0], goal_xy[1])]  # grid too young (<~2 frames) -> straight line (A* unreliable: sparse occ + inflation traps start)
        blocked, (x0, y0, nx, ny) = self.build_traversable()
        res = self.occ_res
        sx = int(math.floor(pose_xy[0] / res)) - x0
        sy = int(math.floor(pose_xy[1] / res)) - y0
        gx = int(math.floor(goal_xy[0] / res)) - x0
        gy = int(math.floor(goal_xy[1] / res)) - y0
        # sanitize goal: if blocked, pull to nearest free cell
        if not (0 <= gy < ny and 0 <= gx < nx) or blocked[gy, gx]:
            found = False
            for r in range(1, 5):
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        gx2, gy2 = gx + dx, gy + dy
                        if 0 <= gy2 < ny and 0 <= gx2 < nx and not blocked[gy2, gx2]:
                            gx, gy = gx2, gy2; found = True; break
                    if found:
                        break
                if found:
                    break
            if not found:
                return None
        cells = self._astar(sx, sy, gx, gy, blocked, nx, ny)
        if cells is None:
            return None
        return [((cx + x0) * res + res / 2, (cy + y0) * res + res / 2) for (cx, cy) in cells[1:]]

    def run(self):
        self.reset_model_pose()
        cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=50)
        lidar_msg_type = CustomMsg if CustomMsg is not None and "livox" in self.lidar_topic else PointCloud2
        rospy.Subscriber(self.lidar_topic, lidar_msg_type, self.cloud_callback, queue_size=20)
        rate = rospy.Rate(10)
        start_wall = time.time()
        self.last_dump_wall = start_wall
        wp_index = 0
        wp_start_wall = start_wall
        last_coverage = start_wall
        self.astar_path = None
        self.sub_index = 0
        self.stuck_blacklist = set()  # cells where robot got stuck -> permanent obstacles for A*
        while not rospy.is_shutdown() and wp_index < len(self.waypoints):
            wall_now = time.time()
            now = rospy.Time.now().to_sec()
            if wall_now - start_wall > self.max_duration:
                rospy.logwarn("auto_explore_trace_map: max_duration reached")
                break
            with self.lock:
                pose = self.pose
                raw_points = sum(len(chunk) for chunk in self.point_chunks)
                frames = self.frames
            if pose is None:
                rate.sleep()
                continue
            # P2: A* path planning — follow sub-waypoints through free cells (no 穿墙)
            if self.astar_path is None or self.sub_index >= len(self.astar_path):
                wp = self.waypoints[wp_index]
                self.astar_path = self.plan_path(pose[:2], (wp.x, wp.y))
                self.sub_index = 0
                wp_start_wall = wall_now
                if not self.astar_path:
                    rospy.logwarn("auto_explore: skip unreachable wp %d/%d (%s)" %
                                  (wp_index, len(self.waypoints), wp.label))
                    wp_index += 1
                    self.astar_path = None
                    continue
                rospy.loginfo("auto_explore: A* %d sub-wps -> wp %d (%s)" %
                              (len(self.astar_path), wp_index, wp.label))
            tgt = self.astar_path[self.sub_index]
            target = Waypoint(tgt[0], tgt[1], self.waypoints[wp_index].label)
            cmd, done = compute_cmd_to_waypoint(
                pose[:3], target, self.v_max, self.w_max,
                self.goal_tol, self.slow_radius, self.command_frame,
            )
            if done:
                self.sub_index += 1
                if self.sub_index >= len(self.astar_path):
                    wp_index += 1
                    self.astar_path = None
                wp_start_wall = wall_now
                continue
            if wall_now - wp_start_wall > self.stuck_timeout:
                rospy.logwarn("auto_explore: stuck at wp %d (%s) (%.1f,%.1f), blacklist+replan" %
                              (wp_index, target.label, tgt[0], tgt[1]))
                self.stuck_blacklist.add((int(math.floor(tgt[0] / self.occ_res)),
                                         int(math.floor(tgt[1] / self.occ_res))))
                self.astar_path = None
                wp_start_wall = wall_now
                continue
            msg = Twist()
            msg.linear.x = cmd.vx
            msg.linear.y = cmd.vy
            msg.angular.z = cmd.wz
            cmd_pub.publish(msg)
            self.trace_rows.append([now, pose[0], pose[1], pose[2], wp_index, target.label])
            self.cmd_rows.append([now, cmd.vx, cmd.vy, cmd.wz, wp_index, target.label])
            if wall_now - last_coverage >= 2.0:
                self.coverage_rows.append([now, frames, raw_points, wp_index, target.label])
                last_coverage = wall_now
            if wall_now - self.last_dump_wall >= self.dump_period:
                self.dump()
                self.last_dump_wall = wall_now
            rate.sleep()
        cmd_pub.publish(Twist())
        self.dump()


def main():
    rospy.init_node("auto_explore_trace_map")
    node = AutoExploreTraceMap()
    rospy.on_shutdown(node.dump)
    rospy.loginfo("auto_explore_trace_map: waypoints=%d out=%s", len(node.waypoints), node.out_dir)
    node.run()


if __name__ == "__main__":
    main()
