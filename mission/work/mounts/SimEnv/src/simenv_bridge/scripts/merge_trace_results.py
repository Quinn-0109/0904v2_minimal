#!/usr/bin/env python3
"""Merge consecutive tare_trace_map result directories into one evidence bundle."""
import argparse
import csv
import json
import math
import os
import shutil
import struct

import numpy as np


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


def path_length_xy(rows):
    return sum(math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(rows, rows[1:]))


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


def _xy_cell(x, y, grid_res):
    return (int(math.floor(x / grid_res)), int(math.floor(y / grid_res)))


def compute_region_metrics(layout, trajectory, points, grid_res):
    wall = points[(points[:, 2] >= 0.15) & (points[:, 2] <= 1.6)] if len(points) else points
    occupied_cells = {_xy_cell(float(point[0]), float(point[1]), grid_res) for point in wall}
    free_cells = set()
    radius = int(round(2.0 / grid_res))
    for _, x, y, *_ in trajectory:
        cx, cy = _xy_cell(x, y, grid_res)
        for ix in range(cx - radius, cx + radius + 1):
            for iy in range(cy - radius, cy + radius + 1):
                if math.hypot(ix - cx, iy - cy) <= radius:
                    free_cells.add((ix, iy))

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
            "trajectory_path": path_length_xy(rows),
            "trajectory_x_coverage": x_cov,
            "trajectory_y_coverage": y_cov,
            "free_cell_coverage": len(region_free) / float(total_cells),
            "occupied_cell_coverage": len(region_occ) / float(total_cells),
            "observed_cell_coverage": len(region_free | region_occ) / float(total_cells),
        }
    return metrics


def read_ply_xyz(path):
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode("ascii")
            if line.strip() == "end_header":
                break
        return np.frombuffer(f.read(), dtype="<f4").reshape((-1, 3)).copy()


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


def voxel_unique(points, voxel_size):
    if points.size == 0:
        return points.reshape((0, 3))
    q = np.floor(points / voxel_size).astype(np.int64)
    _, idx = np.unique(q, axis=0, return_index=True)
    idx.sort()
    return points[idx]


def read_numeric_csv(result_dirs, name, text_columns=None):
    text_columns = set(text_columns or [])
    header = None
    rows = []
    for result_dir in result_dirs:
        with open(os.path.join(result_dir, name), newline="", encoding="ascii") as f:
            reader = csv.reader(f)
            file_header = next(reader)
            if header is None:
                header = file_header
            for row in reader:
                rows.append([
                    value if index in text_columns else float(value)
                    for index, value in enumerate(row)
                ])
    rows.sort(key=lambda row: float(row[0]))
    return header, rows


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="ascii") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def dump_grid(out_dir, points, trajectory, grid_res=0.10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    wall = points[(points[:, 2] >= 0.15) & (points[:, 2] <= 1.6)]
    allxy = points[:, :2]
    if trajectory:
        allxy = np.vstack([allxy, np.asarray([[r[1], r[2]] for r in trajectory], dtype=np.float32)])
    xmin, ymin = allxy.min(axis=0) - 1.0
    xmax, ymax = allxy.max(axis=0) + 1.0
    nx = max(1, int(math.ceil((xmax - xmin) / grid_res)))
    ny = max(1, int(math.ceil((ymax - ymin) / grid_res)))
    hist, _, _ = np.histogram2d(wall[:, 1], wall[:, 0], bins=[ny, nx], range=[[ymin, ymax], [xmin, xmax]])
    occ = np.full((ny, nx), 2, dtype=np.uint8)
    occ[hist >= 2] = 1
    radius = int(round(2.0 / grid_res))
    for _, x, y, *_ in trajectory:
        ix = int((x - xmin) / grid_res)
        iy = int((y - ymin) / grid_res)
        sub = occ[max(0, iy - radius):min(ny, iy + radius + 1), max(0, ix - radius):min(nx, ix + radius + 1)]
        sub[sub != 1] = 0
    fig, ax = plt.subplots(figsize=(max(5, (xmax - xmin) / 2.5), max(6, (ymax - ymin) / 2.5)))
    ax.imshow(occ, cmap=ListedColormap(["white", "black", "#bbbbbb"]), vmin=0, vmax=2,
              origin="lower", extent=[xmin, xmax, ymin, ymax], interpolation="nearest")
    if len(trajectory) > 1:
        ax.plot([r[1] for r in trajectory], [r[2] for r in trajectory], color="royalblue", lw=1.2)
        ax.scatter(trajectory[0][1], trajectory[0][2], c="lime", s=25, zorder=5)
        ax.scatter(trajectory[-1][1], trajectory[-1][2], c="red", marker="x", s=35, zorder=5)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("TARE occupancy grid")
    path = os.path.join(out_dir, "occupancy_grid.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def dump_gt_overlay(out_dir, layout, trajectory, waypoints):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(8, 14))
    for name, bounds in iter_regions(layout):
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
        ax.plot([r[1] for r in trajectory], [r[2] for r in trajectory], "-", color="#0b5bd3", lw=1.8,
                label="odom trajectory")
        ax.scatter(trajectory[0][1], trajectory[0][2], c="lime", edgecolors="black", s=45, zorder=5, label="start")
        ax.scatter(trajectory[-1][1], trajectory[-1][2], c="red", marker="x", s=55, zorder=5, label="end")
    finite_wp = [w for w in waypoints if all(math.isfinite(v) for v in w[1:4])]
    if finite_wp:
        ax.scatter([w[1] for w in finite_wp], [w[2] for w in finite_wp], c="#f28e2b", s=8, alpha=0.6,
                   label="TARE waypoints")
    bounds = [b for _, b in iter_regions(layout)]
    ax.set_xlim(min(b["x_min"] for b in bounds) - 2, max(b["x_max"] for b in bounds) + 2)
    ax.set_ylim(min(b["y_min"] for b in bounds) - 2, max(b["y_max"] for b in bounds) + 2)
    ax.set_aspect("equal")
    ax.grid(True, color="#dddddd", lw=0.4)
    ax.legend(loc="upper right")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("TARE trajectory on GT layout")
    path = os.path.join(out_dir, "tare_on_gt_layout.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def read_summary_value(result_dir, key, default=0):
    with open(os.path.join(result_dir, "summary.txt"), encoding="ascii") as f:
        for line in f:
            if line.startswith(key + "="):
                return int(line.split("=", 1)[1].strip())
    return default


def merge(result_dirs, out_dir, layout_path, voxel_size):
    layout = json.load(open(layout_path, encoding="utf-8"))
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    traj_header, trajectory = read_numeric_csv(result_dirs, "trajectory.csv")
    waypoint_header, waypoints = read_numeric_csv(result_dirs, "waypoints.csv", text_columns={4})
    cmd_header, cmds = read_numeric_csv(result_dirs, "cmd_vel.csv")
    write_csv(os.path.join(out_dir, "trajectory.csv"), traj_header, trajectory)
    write_csv(os.path.join(out_dir, "waypoints.csv"), waypoint_header, waypoints)
    write_csv(os.path.join(out_dir, "cmd_vel.csv"), cmd_header, cmds)
    points = voxel_unique(np.vstack([read_ply_xyz(os.path.join(d, "tare_global_map.ply")) for d in result_dirs]), voxel_size)
    write_ply_xyz(os.path.join(out_dir, "tare_global_map.ply"), points)
    grid = dump_grid(out_dir, points, trajectory)
    overlay = dump_gt_overlay(out_dir, layout, trajectory, waypoints)
    region_counts = {name: 0 for name, _ in iter_regions(layout)}
    region_counts["outside"] = 0
    for row in trajectory:
        region_counts[point_region(layout, row[1], row[2], margin=0.2)] += 1
    region_metrics = compute_region_metrics(layout, trajectory, points, 0.10)
    finite_wp = sum(1 for w in waypoints if all(math.isfinite(v) for v in w[1:4]))
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="ascii") as f:
        f.write("source_dirs=%s\n" % ",".join(result_dirs))
        f.write("frames=%d\n" % sum(read_summary_value(d, "frames") for d in result_dirs))
        f.write("points=%d\n" % len(points))
        f.write("trajectory_samples=%d\n" % len(trajectory))
        f.write("waypoint_samples=%d\n" % len(waypoints))
        f.write("waypoint_finite=%d\n" % finite_wp)
        f.write("waypoint_nan_or_inf=%d\n" % (len(waypoints) - finite_wp))
        f.write("cmd_samples=%d\n" % len(cmds))
        f.write("path_xy=%.6f\n" % path_length_xy(trajectory))
        for key, value in region_counts.items():
            f.write("trajectory_region_%s=%d\n" % (key, value))
        for key, metrics in region_metrics.items():
            f.write("coverage_%s_trajectory_path=%.6f\n" % (key, metrics["trajectory_path"]))
            f.write("coverage_%s_trajectory_x=%.6f\n" % (key, metrics["trajectory_x_coverage"]))
            f.write("coverage_%s_trajectory_y=%.6f\n" % (key, metrics["trajectory_y_coverage"]))
            f.write("coverage_%s_free_cells=%.6f\n" % (key, metrics["free_cell_coverage"]))
            f.write("coverage_%s_occupied_cells=%.6f\n" % (key, metrics["occupied_cell_coverage"]))
            f.write("coverage_%s_observed_cells=%.6f\n" % (key, metrics["observed_cell_coverage"]))
        f.write("bbox_min=%.6f %.6f %.6f\n" % tuple(points.min(axis=0)))
        f.write("bbox_max=%.6f %.6f %.6f\n" % tuple(points.max(axis=0)))
        f.write("occupancy_grid=%s\n" % grid)
        f.write("gt_overlay=%s\n" % overlay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--voxel", type=float, default=0.10)
    parser.add_argument("result_dirs", nargs="+")
    args = parser.parse_args()
    merge(args.result_dirs, args.out, args.layout, args.voxel)


if __name__ == "__main__":
    main()
