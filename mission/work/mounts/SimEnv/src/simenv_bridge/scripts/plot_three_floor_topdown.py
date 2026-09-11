#!/usr/bin/env python3
"""Render one top-down executed-route plot for each mission floor."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Patch, Polygon, Rectangle


FLOOR_TOUR_FILES = (
    "tour_summary.json",
    "second_floor/tour_summary.json",
    "third_floor/tour_summary.json",
)

FURNITURE_COLORS = {
    "chair": "#2dd4bf",
    "desk": "#8b5cf6",
    "side_table": "#c084fc",
    "meeting_table": "#7c3aed",
    "coffee_table": "#a78bfa",
    "cabinet": "#ca8a04",
    "storage_rack": "#64748b",
    "bookshelf": "#475569",
    "sofa": "#0ea5e9",
    "planter": "#22c55e",
    "pallet": "#b45309",
}


def read_json(path, required=True):
    path = Path(path)
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return {}
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def bounds_rectangle(bounds, **kwargs):
    return Rectangle(
        (float(bounds["x_min"]), float(bounds["y_min"])),
        float(bounds["x_max"]) - float(bounds["x_min"]),
        float(bounds["y_max"]) - float(bounds["y_min"]),
        **kwargs,
    )


def rotated_footprint_vertices(pose, size):
    """Return the four world-XY corners of an oriented box footprint."""
    centre_x, centre_y = float(pose[0]), float(pose[1])
    yaw = float(pose[5]) if len(pose) >= 6 else 0.0
    half_x, half_y = 0.5 * float(size[0]), 0.5 * float(size[1])
    cosine, sine = math.cos(yaw), math.sin(yaw)
    vertices = []
    for local_x, local_y in (
            (-half_x, -half_y), (half_x, -half_y),
            (half_x, half_y), (-half_x, half_y)):
        vertices.append((
            centre_x + cosine * local_x - sine * local_y,
            centre_y + sine * local_x + cosine * local_y,
        ))
    return vertices


def draw_furniture(ax, room):
    """Draw run-local furniture using its true footprint and yaw."""
    count = 0
    for item in room.get("furniture", []):
        pose, size = item.get("pose"), item.get("size")
        if (not isinstance(pose, list) or len(pose) < 2 or
                not isinstance(size, list) or len(size) < 2):
            continue
        kind = str(item.get("kind", "object"))
        color = FURNITURE_COLORS.get(kind, "#6b7280")
        ax.add_patch(Polygon(
            rotated_footprint_vertices(pose, size), closed=True,
            facecolor=color, edgecolor="#1f2937", linewidth=0.65,
            alpha=0.82, zorder=3))
        label = kind.replace("_", " ")
        ax.text(
            float(pose[0]), float(pose[1]), label,
            ha="center", va="center", fontsize=4.0, color="#111827",
            rotation=math.degrees(float(pose[5])) if len(pose) >= 6 else 0.0,
            rotation_mode="anchor", zorder=4,
            bbox={"boxstyle": "round,pad=0.08", "facecolor": "white",
                  "edgecolor": "none", "alpha": 0.62})
        count += 1
    return count


def waypoint_xy(tour):
    points = []
    for waypoint in tour.get("waypoints", []):
        pose = waypoint.get("reached_pose")
        if isinstance(pose, list) and len(pose) >= 2:
            points.append((float(pose[0]), float(pose[1]), waypoint))
    return points


def trace_xy(payload, segment=None):
    points = []
    for sample in payload.get("trace", []):
        if segment is not None and int(sample.get("segment", -1)) != segment:
            continue
        if sample.get("x") is None or sample.get("y") is None:
            continue
        points.append((float(sample["x"]), float(sample["y"])))
    return points


def plot_polyline(ax, points, color, label, linewidth=2.2, alpha=0.95,
                  linestyle="-"):
    if len(points) < 2:
        return
    ax.plot(
        [point[0] for point in points],
        [point[1] for point in points],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        linestyle=linestyle,
        solid_capstyle="round",
        label=label,
        zorder=7,
    )


def stage_duration(stage_timing, name):
    for stage in stage_timing.get("stages", []):
        if stage.get("name") == name:
            return stage.get("duration_sec")
    return None


def draw_layout(ax, floor):
    footprint = {
        "x_min": -10.0,
        "x_max": 10.0,
        "y_min": 0.0,
        "y_max": 36.0,
    }
    ax.add_patch(bounds_rectangle(
        footprint, facecolor="#fafafa", edgecolor="#111827",
        linewidth=2.0, zorder=0))
    ax.add_patch(bounds_rectangle(
        floor["lobby_bounds"], facecolor="#dbeafe", edgecolor="#64748b",
        linewidth=1.1, zorder=1))
    ax.add_patch(bounds_rectangle(
        floor["corridor_bounds"], facecolor="#e5e7eb", edgecolor="#64748b",
        linewidth=1.1, zorder=1))

    room_colors = ("#fef3c7", "#dcfce7", "#ffedd5", "#f3e8ff")
    for room_index, room in enumerate(floor.get("rooms", [])):
        ax.add_patch(bounds_rectangle(
            room["bounds"], facecolor=room_colors[room_index % 4],
            edgecolor="#475569", linewidth=1.0, alpha=0.72, zorder=1))
        bounds = room["bounds"]
        draw_furniture(ax, room)
        x = (float(bounds["x_min"]) + float(bounds["x_max"])) / 2.0
        y = float(bounds["y_max"]) - 0.32
        room_type = str(room.get("room_type", "room")).replace("_", " ")
        ax.text(
            x, y, "Room {}\n{}".format(room_index + 1, room_type),
            ha="center", va="top", color="#334155", fontsize=7,
            alpha=0.95, zorder=5,
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white",
                  "edgecolor": "#cbd5e1", "alpha": 0.76})

    stair = floor.get("stair_bounds")
    if stair:
        ax.add_patch(bounds_rectangle(
            stair, facecolor="#fed7aa", edgecolor="#c2410c",
            linewidth=1.3, hatch="////", alpha=0.7, zorder=2))
        ax.text(
            (stair["x_min"] + stair["x_max"]) / 2.0,
            (stair["y_min"] + stair["y_max"]) / 2.0,
            "STAIRS", ha="center", va="center", fontsize=7,
            color="#9a3412", rotation=90, zorder=3)
    elevator = floor.get("elevator_bounds")
    if elevator:
        ax.add_patch(bounds_rectangle(
            elevator, facecolor="#cbd5e1", edgecolor="#475569",
            linewidth=1.0, hatch="xx", alpha=0.75, zorder=2))
        ax.text(
            (elevator["x_min"] + elevator["x_max"]) / 2.0,
            (elevator["y_min"] + elevator["y_max"]) / 2.0,
            "LIFT\n(not used)", ha="center", va="center", fontsize=6,
            color="#475569", zorder=3)


def draw_two_view_contracts(ax, floor_number, evidence, layout_floor,
                            include_entry_exit=True):
    """Overlay planned/actual G3-G4 pairs and physical door crossings."""
    layout_rooms = {
        str(room.get("id")): room for room in layout_floor.get("rooms", [])}
    count = 0
    for record in evidence.get("rooms", []):
        if int(record.get("floor_number", -1)) != int(floor_number):
            continue
        room_id = str(record.get("room_id", "room"))
        room_short = room_id.replace("floor_{}_room_".format(floor_number - 1), "R")
        viewpoints = record.get("viewpoints", {})
        door = record.get("door") or {}
        direction = float(door.get("inward_direction", 1.0))
        door_x = float((door.get("centre") or [0.0])[0])
        layout_room = layout_rooms.get(room_id, {})
        primary_obstacle = record.get("primary_obstacle") or {}
        obstacle_id = str(primary_obstacle.get("id", ""))
        inflation = float(primary_obstacle.get(
            "door_axis_inflation_m", 0.60))
        for furniture in layout_room.get("furniture", []):
            if str(furniture.get("id", "")) != obstacle_id:
                continue
            pose, size = furniture.get("pose", []), furniture.get("size", [])
            if len(pose) >= 2 and len(size) >= 2:
                inflated = [float(size[0]) + 2.0 * inflation,
                            float(size[1]) + 2.0 * inflation]
                ax.add_patch(Polygon(
                    rotated_footprint_vertices(pose, inflated), closed=True,
                    facecolor="none", edgecolor="#dc2626", linewidth=1.2,
                    linestyle=":", alpha=0.85, zorder=8))
            break
        path_colors = {
            "entry_to_g3": "#2563eb",
            "g3_to_g4": "#111827",
            "g4_to_return": "#7c3aed",
        }
        for path_name, path in (record.get("planned_room_paths") or {}).items():
            if not isinstance(path, list) or len(path) < 2:
                continue
            color = path_colors.get(path_name, "#64748b")
            ax.plot([float(point[0]) for point in path],
                    [float(point[1]) for point in path],
                    color=color, linewidth=0.9, linestyle=":",
                    alpha=0.7, zorder=7)
            ax.annotate(
                "", xy=(float(path[-1][0]), float(path[-1][1])),
                xytext=(float(path[-2][0]), float(path[-2][1])),
                arrowprops={"arrowstyle": "->", "color": color,
                            "lw": 1.0, "alpha": 0.75}, zorder=8)
        actual = {}
        for role, marker, color in (("G3", "s", "#0891b2"),
                                    ("G4", "D", "#db2777")):
            item = viewpoints.get(role, {})
            planned, reached = item.get("planned_pose"), item.get("actual_pose")
            if isinstance(planned, list) and len(planned) >= 2:
                ax.scatter(
                    [planned[0]], [planned[1]], marker=marker, s=70,
                    facecolor="none", edgecolor=color, linewidth=1.2,
                    alpha=0.65, zorder=10)
            if isinstance(reached, list) and len(reached) >= 2:
                actual[role] = (float(reached[0]), float(reached[1]))
                semantic = str(item.get("viewpoint_semantic") or role.lower())
                depth = direction * (float(reached[0]) - door_x)
                ax.scatter(
                    [reached[0]], [reached[1]], marker=marker, s=58,
                    facecolor=color, edgecolor="#0f172a", linewidth=0.8,
                    zorder=12)
                ax.annotate(
                    "{} {} {}\ndepth {:.2f} m | scan {:.0f}°".format(
                        room_short, role, semantic,
                        depth,
                        math.degrees(float(item.get("scan_rotation_rad", 0.0)))),
                    (reached[0], reached[1]), xytext=(5, 5),
                    textcoords="offset points", fontsize=5.7,
                    color=color, weight="bold", zorder=13,
                    bbox={"boxstyle": "round,pad=0.12", "facecolor": "white",
                          "edgecolor": color, "alpha": 0.78})
                ax.add_patch(Arc(
                    (float(reached[0]), float(reached[1])), 1.05, 1.05,
                    theta1=0.0, theta2=min(
                        359.0, math.degrees(float(
                            item.get("scan_rotation_rad", 0.0)))),
                    color=color, linewidth=1.0, alpha=0.75, zorder=11))
        if {"G3", "G4"}.issubset(actual):
            x3, y3 = actual["G3"]
            x4, y4 = actual["G4"]
            ax.annotate(
                "", xy=(x4, y4), xytext=(x3, y3),
                arrowprops={"arrowstyle": "->", "color": "#111827",
                            "lw": 1.25, "linestyle": "--", "alpha": 0.8},
                zorder=9)
            midpoint = ((x3 + x4) / 2.0, (y3 + y4) / 2.0)
            separation = float(record.get(
                "viewpoint_separation_m", math.hypot(x4 - x3, y4 - y3)))
            angle = record.get("actual_viewpoint_line_angle_deg")
            angle_text = (" | θ={:.1f}°".format(float(angle))
                          if angle is not None else "")
            ax.text(
                midpoint[0], midpoint[1], "{:.2f} m{}".format(
                    separation, angle_text),
                fontsize=5.3, ha="center", va="center", zorder=13,
                bbox={"facecolor": "white", "edgecolor": "#64748b",
                      "alpha": 0.78, "pad": 0.7})

        if include_entry_exit:
            phases = record.get("phases", {})
            entry = phases.get("ENTRY", {}).get("actual_pose")
            returned = phases.get("RETURN", {}).get("actual_pose")
            crossing = record.get("door_plane_exit_evidence") or {}
            inside, outside = crossing.get("inside_pose"), crossing.get("outside_pose")
            if isinstance(entry, list) and len(entry) >= 2:
                ax.scatter([entry[0]], [entry[1]], marker=">", s=48,
                           color="#f59e0b", edgecolor="#78350f", zorder=12)
                if "G3" in actual:
                    ax.annotate(
                        "", xy=actual["G3"], xytext=(entry[0], entry[1]),
                        arrowprops={"arrowstyle": "->", "color": "#2563eb",
                                    "lw": 1.15, "alpha": 0.8}, zorder=9)
            if ("G4" in actual and isinstance(returned, list) and
                    len(returned) >= 2):
                ax.annotate(
                    "", xy=(returned[0], returned[1]), xytext=actual["G4"],
                    arrowprops={"arrowstyle": "->", "color": "#7c3aed",
                                "lw": 1.15, "alpha": 0.8}, zorder=9)
                ax.scatter([returned[0]], [returned[1]], marker="v", s=45,
                           color="#7c3aed", edgecolor="#4c1d95", zorder=12)
            if (isinstance(inside, list) and len(inside) >= 2 and
                    isinstance(outside, list) and len(outside) >= 2):
                ax.annotate(
                    "", xy=(outside[0], outside[1]), xytext=(inside[0], inside[1]),
                    arrowprops={"arrowstyle": "->", "color": "#16a34a",
                                "lw": 1.5}, zorder=12)
                ax.scatter([outside[0]], [outside[1]], marker="<", s=48,
                           color="#16a34a", edgecolor="#14532d", zorder=12)
        policy = str(record.get("viewpoint_policy", ""))
        fallback = str(record.get("viewpoint_fallback_level", "primary"))
        room_bounds = layout_room.get("bounds", {})
        if room_bounds:
            ax.text(
                0.5 * (float(room_bounds["x_min"]) +
                       float(room_bounds["x_max"])),
                float(room_bounds["y_min"]) + 0.28,
                "{} | {}".format(policy, fallback),
                fontsize=4.8, ha="center", va="bottom", color="#334155",
                zorder=13, bbox={"facecolor": "white", "edgecolor": "#cbd5e1",
                                "alpha": 0.72, "pad": 0.5})
        count += 1
    return count


def draw_two_view_floor(ax, floor_index, layout, evidence):
    floor = layout["floors"][floor_index]
    draw_layout(ax, floor)
    count = draw_two_view_contracts(
        ax, floor_index + 1, evidence, floor, include_entry_exit=True)
    truth = [item for item in layout.get("danger_red_spheres", [])
             if int(item.get("floor_index", -1)) == floor_index]
    distractors = [item for item in layout.get("red_distractors", [])
                   if abs(float(item.get("pose", [0, 0, -99])[2]) -
                          (float(floor.get("elevation", 0.0)) + 0.15)) < 0.45]
    if distractors:
        ax.scatter([item["pose"][0] for item in distractors],
                   [item["pose"][1] for item in distractors],
                   marker="s", s=28, facecolor="#ef4444",
                   edgecolor="#7f1d1d", linewidth=0.7, zorder=8)
    if truth:
        ax.scatter(
            [item["pose"][0] for item in truth],
            [item["pose"][1] for item in truth], marker="*", s=115,
            facecolor="#dc2626", edgecolor="#7f1d1d", linewidth=0.9,
            zorder=14)
        records = {
            str(item.get("room_id")): item
            for item in evidence.get("rooms", [])
            if int(item.get("floor_number", -1)) == floor_index + 1
        }
        for hazard in truth:
            position = hazard.get("pose") or []
            record = records.get(str(hazard.get("room_id")), {})
            viewpoints = record.get("viewpoints", {})
            view_evidence = hazard.get("view_evidence") or []
            evidence_by_scan = {
                str(item.get("scan_id")): item for item in view_evidence}
            evidence_by_role = {
                str(item.get("viewpoint_role", "")).upper(): item
                for item in view_evidence
                if item.get("viewpoint_role")}
            for role, color in (("G3", "#0891b2"),
                                ("G4", "#db2777")):
                viewpoint = viewpoints.get(role, {})
                origin = viewpoint.get("actual_pose")
                if not (len(position) >= 2 and isinstance(origin, list) and
                        len(origin) >= 2):
                    continue
                planned_id = str(viewpoint.get("waypoint") or "")
                # Runtime evidence uses the executed waypoint name while layout
                # metadata uses the stable G3/G4 role.  Prefer the role so that
                # a harmless waypoint-name rewrite cannot turn a verified clear
                # sightline into a red "unreliable" line in the final figure.
                online_contract = (evidence_by_role.get(role) or
                                   evidence_by_scan.get(planned_id))
                reliable = (online_contract is not None and bool(
                    online_contract.get("uncertainty_safe", True)) and bool(
                    online_contract.get(
                        "red_distractor_projection_safe", True)))
                ax.plot(
                    [float(origin[0]), float(position[0])],
                    [float(origin[1]), float(position[1])],
                    color=color if reliable else "#dc2626",
                    linewidth=0.85, linestyle="-" if reliable else "--",
                    alpha=0.72, zorder=9)
            parallax = hazard.get("dual_view_parallax_deg")
            if parallax is not None and len(position) >= 2:
                ax.annotate(
                    "dual parallax {:.1f}°".format(float(parallax)),
                    (float(position[0]), float(position[1])),
                    xytext=(5, -11), textcoords="offset points",
                    fontsize=5.4, color="#7f1d1d", zorder=15,
                    bbox={"facecolor": "white", "edgecolor": "#fecaca",
                          "alpha": 0.76, "pad": 0.5})
    ax.set_title(
        "Floor {} physical two-view selection\n{} rooms: planned outlines, actual solid markers".format(
            floor_index + 1, count), fontsize=11, fontweight="bold")
    ax.set_xlim(-10.7, 10.7)
    ax.set_ylim(-0.7, 36.6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.grid(True, color="#94a3b8", alpha=0.22, linewidth=0.6)
    ax.legend(handles=[
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#0891b2",
               markeredgecolor="#0f172a", markersize=7,
               label="actual G3 deep scan"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#db2777",
               markeredgecolor="#0f172a", markersize=7,
               label="actual G4 near scan"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="none",
               markeredgecolor="#0891b2", markersize=7, label="planned viewpoint"),
        Line2D([0], [0], marker=">", color="none", markerfacecolor="#f59e0b",
               markeredgecolor="#78350f", markersize=7, label="room entry"),
        Line2D([0], [0], marker="<", color="none", markerfacecolor="#16a34a",
               markeredgecolor="#14532d", markersize=7, label="door-plane exit"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#dc2626",
               markeredgecolor="#7f1d1d", markersize=9, label="truth danger source"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#ef4444",
               markeredgecolor="#7f1d1d", markersize=6,
               label="red non-danger distractor"),
        Line2D([0], [0], color="#0891b2", linewidth=1.0,
               label="reliable G3 sightline"),
        Line2D([0], [0], color="#db2777", linewidth=1.0,
               label="reliable G4 sightline"),
        Line2D([0], [0], color="#dc2626", linewidth=1.0, linestyle="--",
               label="blocked / unreliable sightline"),
        Patch(facecolor="#94a3b8", edgecolor="#1f2937", alpha=0.82,
              label="furniture / obstacle"),
    ], loc="upper right", fontsize=6.6, framealpha=0.92)


def draw_floor(ax, floor_index, layout, tours, red_record, mission,
               timing, ascent_traces, descent_trace):
    floor = layout["floors"][floor_index]
    tour_points = waypoint_xy(tours[floor_index])
    draw_layout(ax, floor)

    plane_points = [(x, y) for x, y, _waypoint in tour_points]
    if floor_index == 0 and len(plane_points) >= 3:
        spawn = timing.get("spawn_truth_pose_at_start") or [0.0, -3.2]
        entrance_points = [(float(spawn[0]), float(spawn[1]))] + plane_points[:3]
        plot_polyline(
            ax, entrance_points, "#ea580c", "stair RL (entrance)",
            linewidth=2.8)
        plane_points = plane_points[2:]
    elif floor_index > 0 and ascent_traces[floor_index - 1]:
        plane_points = [ascent_traces[floor_index - 1][-1]] + plane_points

    plot_polyline(ax, plane_points, "#2563eb", "plane RL", linewidth=2.5)

    if floor_index < 2:
        plot_polyline(
            ax, ascent_traces[floor_index], "#ea580c", "stair RL (up)",
            linewidth=2.6)

    descent_segment = 0 if floor_index == 2 else (1 if floor_index == 1 else None)
    if floor_index == 2:
        down_points = trace_xy(descent_trace, segment=0)
    elif floor_index == 1:
        # Show both the upper arrival and lower departure around the F2 landing.
        down_points = trace_xy(descent_trace, segment=0) + trace_xy(
            descent_trace, segment=1)
    else:
        down_points = trace_xy(descent_trace, segment=1)
    plot_polyline(
        ax, down_points, "#7c3aed", "stair RL (down)", linewidth=2.2,
        alpha=0.85)

    if floor_index == 0:
        returned = []
        down_all = trace_xy(descent_trace)
        if down_all:
            returned.append(down_all[-1])
        for waypoint in mission.get("return_waypoints", []):
            pose = waypoint.get("reached_pose")
            if isinstance(pose, list) and len(pose) >= 2:
                returned.append((float(pose[0]), float(pose[1])))
        plot_polyline(
            ax, returned, "#059669", "plane RL (lobby return)",
            linewidth=2.8)

    if tour_points:
        ax.scatter(
            [point[0] for point in tour_points],
            [point[1] for point in tour_points],
            s=18, facecolor="white", edgecolor="#1d4ed8", linewidth=0.9,
            zorder=9)
        for order, (x, y, waypoint) in enumerate(tour_points, start=1):
            ax.annotate(
                str(order), (x, y), xytext=(3, 3), textcoords="offset points",
                fontsize=5.5, color="#1e3a8a", zorder=10)
            if waypoint.get("scan_completed"):
                note = str(waypoint.get("note", ""))
                role = ("G3" if note.endswith("_g3") else
                        "G4" if note.endswith("_g4") else "scan")
                marker, color = ({
                    "G3": ("s", "#0891b2"),
                    "G4": ("D", "#db2777"),
                    "scan": ("*", "#06b6d4"),
                }[role])
                ax.scatter(
                    [x], [y], marker=marker, s=62, facecolor=color,
                    edgecolor="#0f172a", linewidth=0.7, zorder=10)
                if role in ("G3", "G4"):
                    ax.annotate(
                        role, (x, y), xytext=(4, -8),
                        textcoords="offset points", fontsize=5.3,
                        color=color, weight="bold", zorder=11)

    detections = [
        item for item in red_record.get("detections", [])
        if int(item.get("floor", -1)) == floor_index + 1
    ]
    if detections:
        ax.scatter(
            [item["position"][0] for item in detections],
            [item["position"][1] for item in detections],
            marker="o", s=42, facecolor="#ef4444", edgecolor="#7f1d1d",
            linewidth=0.8, zorder=11, label="RGB red-ball confirmation")

    duration = stage_duration(timing, "floor_{}_exploration".format(floor_index + 1))
    duration_text = "{:.2f} s".format(float(duration)) if duration is not None else "running"
    ax.set_title(
        "Floor {} top-down executed route\nexploration={} | RGB confirmations={}".format(
            floor_index + 1, duration_text, len(detections)),
        fontsize=11, fontweight="bold")
    ax.set_xlim(-10.7, 10.7)
    ax.set_ylim(-3.8 if floor_index == 0 else -0.7, 36.6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.grid(True, color="#94a3b8", alpha=0.22, linewidth=0.6)
    ax.text(
        0.01, 0.01,
        "Lines connect actual reached poses; stair traces are dense Gazebo truth.",
        transform=ax.transAxes, fontsize=6.5, color="#475569",
        ha="left", va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8,
              "edgecolor": "#cbd5e1"})
    handles = [
        Line2D([0], [0], color="#2563eb", lw=2.5, label="plane RL"),
        Line2D([0], [0], color="#ea580c", lw=2.6, label="stair RL up/entrance"),
        Line2D([0], [0], color="#7c3aed", lw=2.2, label="stair RL down"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#0891b2",
               markeredgecolor="#0f172a", markersize=7, label="actual G3 scan"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#db2777",
               markeredgecolor="#0f172a", markersize=7, label="actual G4 scan"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#ef4444",
               markeredgecolor="#7f1d1d", markersize=6,
               label="RGB red-ball confirmation"),
        Patch(facecolor="#94a3b8", edgecolor="#1f2937", alpha=0.82,
              label="room furniture / obstacle (type labelled)"),
    ]
    if floor_index == 0:
        handles.insert(3, Line2D(
            [0], [0], color="#059669", lw=2.8, label="lobby return"))
    ax.legend(handles=handles, loc="upper right", fontsize=6.8, framealpha=0.92)


def generate(results_dir, output_dir):
    results_dir = Path(results_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    layout = read_json(results_dir / "layout_metadata.json")
    if not layout.get("danger_red_spheres"):
        # An official scene's layout carries no sphere: the run is not told
        # where they are.  Draw the referee copy the runner saved beside the
        # results, so the picture still shows what was there to find.
        referee = results_dir / "official_danger_truth.json"
        if referee.is_file():
            truth_payload = read_json(referee, required=False) or {}
            layout["danger_red_spheres"] = [
                dict(source, pose=list(source.get("position", [])),
                     id=source.get("model_name", source.get("id")))
                for source in truth_payload.get("danger_sources", [])]
            layout["red_distractors"] = [
                dict(item, pose=list(item.get("position", [])))
                for item in truth_payload.get("distraction_sources", [])
                if str(item.get("color")) == "red"]
    # Failed missions may stop before later-floor tour files exist.  Render
    # the available trace and planned layout instead of losing every top-down
    # diagnostic precisely when it is most useful.
    tours = [read_json(results_dir / relative, required=False)
             for relative in FLOOR_TOUR_FILES]
    red_record = read_json(results_dir / "red_ball_detections.json")
    mission = read_json(results_dir / "three_floor_rl_mission_summary.json")
    timing = read_json(results_dir / "mission_stage_timing.json")
    two_view_evidence = read_json(
        results_dir / "physical_two_view_room_evidence.json", required=False)
    ascent_traces = [
        trace_xy(read_json(
            results_dir / "logs/stair_transition.json", required=False)),
        trace_xy(read_json(
            results_dir / "logs/second_to_third_floor_stair_transition.json",
            required=False)),
    ]
    descent_trace = read_json(
        results_dir / "logs/stair_descent.json", required=False)

    outputs = []
    for floor_index in range(3):
        figure, axis = plt.subplots(figsize=(7.2, 11.2))
        draw_floor(
            axis, floor_index, layout, tours, red_record, mission, timing,
            ascent_traces, descent_trace)
        figure.tight_layout()
        path = output_dir / "topdown_floor_{}.png".format(floor_index + 1)
        figure.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        outputs.append(path)

    figure, axes = plt.subplots(1, 3, figsize=(19.5, 11.2), sharex=True)
    for floor_index, axis in enumerate(axes):
        draw_floor(
            axis, floor_index, layout, tours, red_record, mission, timing,
            ascent_traces, descent_trace)
    total = timing.get("total_duration_sec")
    figure.suptitle(
        "Three-floor RL exploration — {} — total={} s".format(
            results_dir.name,
            "{:.2f}".format(float(total)) if total is not None else "running"),
        fontsize=14, fontweight="bold", y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    combined = output_dir / "three_floor_topdown_trajectories.png"
    figure.savefig(combined, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    outputs.append(combined)

    for floor_index in range(3):
        figure, axis = plt.subplots(figsize=(7.2, 11.2))
        draw_two_view_floor(axis, floor_index, layout, two_view_evidence)
        figure.tight_layout()
        path = output_dir / "two_view_floor_{}.png".format(floor_index + 1)
        figure.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        outputs.append(path)

    figure, axes = plt.subplots(1, 3, figsize=(19.5, 11.2), sharex=True)
    for floor_index, axis in enumerate(axes):
        draw_two_view_floor(axis, floor_index, layout, two_view_evidence)
    figure.suptitle(
        "Physical G3/G4 selections and verified room crossings — {}".format(
            results_dir.name), fontsize=14, fontweight="bold", y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    two_view_combined = output_dir / "two_view_all_floors.png"
    figure.savefig(
        two_view_combined, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    outputs.append(two_view_combined)

    manifest = {
        "schema": "scanplanner_three_floor_topdown_v1",
        "results_dir": str(results_dir),
        "source_status": timing.get("status"),
        "source_total_duration_sec": timing.get("total_duration_sec"),
        "source_red_record_status": red_record.get("status"),
        "source_red_confirmed_count": red_record.get("confirmed_count"),
        "physical_two_view_completed_room_count": two_view_evidence.get(
            "completed_room_count"),
        "furniture_footprints_rendered": sum(
            len(room.get("furniture", []))
            for floor in layout.get("floors", [])
            for room in floor.get("rooms", [])),
        "plots": [str(path) for path in outputs],
    }
    manifest_path = output_dir / "three_floor_topdown_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return outputs, manifest_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="completed mission result directory")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    results = Path(args.results)
    output = Path(args.output_dir) if args.output_dir else results / "visualization"
    outputs, manifest = generate(results, output)
    for path in outputs:
        print(path)
    print(manifest)


if __name__ == "__main__":
    main()
