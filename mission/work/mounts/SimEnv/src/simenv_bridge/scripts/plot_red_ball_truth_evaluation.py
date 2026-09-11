#!/usr/bin/env python3
"""Plot offline red-ball truth matching for a completed or failed run."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle


FURNITURE_COLORS = {
    "chair": "#2dd4bf", "desk": "#8b5cf6", "side_table": "#c084fc",
    "meeting_table": "#7c3aed", "coffee_table": "#a78bfa",
    "cabinet": "#ca8a04", "storage_rack": "#64748b",
    "bookshelf": "#475569", "sofa": "#0ea5e9", "planter": "#22c55e",
    "pallet": "#b45309",
}


def read_json(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def maximum_truth_matching(detections, truth, tolerance):
    """Mirror check_three_floor_rl_mission.py's one-to-one matching."""
    adjacency = {}
    distances = {}
    for detection_index, detection in enumerate(detections):
        candidates = []
        for truth_index, expected in enumerate(truth):
            if int(detection["floor"]) != int(expected["floor_index"]) + 1:
                continue
            distance = math.hypot(
                float(detection["position"][0]) - float(expected["pose"][0]),
                float(detection["position"][1]) - float(expected["pose"][1]),
            )
            distances[(detection_index, truth_index)] = distance
            if distance <= tolerance:
                candidates.append((distance, truth_index))
        adjacency[detection_index] = [item[1] for item in sorted(candidates)]

    truth_to_detection = {}

    def augment(detection_index, visited):
        for truth_index in adjacency[detection_index]:
            if truth_index in visited:
                continue
            visited.add(truth_index)
            previous = truth_to_detection.get(truth_index)
            if previous is None or augment(previous, visited):
                truth_to_detection[truth_index] = detection_index
                return True
        return False

    for detection_index in sorted(adjacency, key=lambda value: len(adjacency[value])):
        augment(detection_index, set())
    return truth_to_detection, distances


def draw_bounds(axis, bounds, color, alpha, label=None, hatch=None):
    x_min, x_max = float(bounds["x_min"]), float(bounds["x_max"])
    y_min, y_max = float(bounds["y_min"]), float(bounds["y_max"])
    axis.add_patch(Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        facecolor=color, edgecolor="#455a64", linewidth=1.0,
        alpha=alpha, hatch=hatch,
    ))
    if label:
        axis.text((x_min + x_max) / 2.0, (y_min + y_max) / 2.0,
                  label, ha="center", va="center", fontsize=8,
                  color="#455a64")


def rotated_footprint_vertices(pose, size):
    """Return an object's four footprint corners in world XY coordinates."""
    centre_x, centre_y = float(pose[0]), float(pose[1])
    yaw = float(pose[5]) if len(pose) >= 6 else 0.0
    half_x, half_y = float(size[0]) / 2.0, float(size[1]) / 2.0
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return [
        (centre_x + cosine * x - sine * y,
         centre_y + sine * x + cosine * y)
        for x, y in ((-half_x, -half_y), (half_x, -half_y),
                     (half_x, half_y), (-half_x, half_y))
    ]


def draw_furniture(axis, room):
    """Draw furniture and obstacles using recorded pose, size, and yaw."""
    count = 0
    for item in room.get("furniture", []):
        pose, size = item.get("pose"), item.get("size")
        if (not isinstance(pose, list) or len(pose) < 2 or
                not isinstance(size, list) or len(size) < 2):
            continue
        kind = str(item.get("kind", "object"))
        axis.add_patch(Polygon(
            rotated_footprint_vertices(pose, size), closed=True,
            facecolor=FURNITURE_COLORS.get(kind, "#6b7280"),
            edgecolor="#1f2937", linewidth=0.65, alpha=0.82, zorder=3))
        axis.text(
            float(pose[0]), float(pose[1]), kind.replace("_", " "),
            ha="center", va="center", fontsize=4.0, color="#111827",
            rotation=math.degrees(float(pose[5])) if len(pose) >= 6 else 0.0,
            rotation_mode="anchor", zorder=4,
            bbox={"boxstyle": "round,pad=0.08", "facecolor": "white",
                  "edgecolor": "none", "alpha": 0.62})
        count += 1
    return count


def draw_floor(axis, floor, floor_number, truth, detections,
               matching, distances, tolerance, red_distractors=None):
    draw_bounds(axis, floor["lobby_bounds"], "#bbdefb", 0.42, "Lobby")
    draw_bounds(axis, floor["corridor_bounds"], "#cfd8dc", 0.55, "Corridor")
    if floor.get("stair_bounds"):
        draw_bounds(axis, floor["stair_bounds"], "#ffcc80", 0.38, "Stairs", "////")
    if floor.get("elevator_bounds"):
        draw_bounds(axis, floor["elevator_bounds"], "#b0bec5", 0.35, "Lift", "xx")
    room_colors = ("#fff3e0", "#e8f5e9", "#f3e5f5", "#fffde7")
    for room_index, room in enumerate(floor.get("rooms", [])):
        draw_bounds(axis, room["bounds"], room_colors[room_index % 4], 0.55)
        draw_furniture(axis, room)
        bounds = room["bounds"]
        label = "{}\n{}".format(room.get("id", "room").split("_")[-1],
                                 room.get("room_type", "room").replace("_", " "))
        axis.text(
            (float(bounds["x_min"]) + float(bounds["x_max"])) / 2.0,
            float(bounds["y_max"]) - 0.32, label,
            ha="center", va="top", fontsize=7, color="#455a64", zorder=5,
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white",
                  "edgecolor": "#cbd5e1", "alpha": 0.76})

    floor_truth = [index for index, item in enumerate(truth)
                   if int(item["floor_index"]) + 1 == floor_number]
    floor_detections = [index for index, item in enumerate(detections)
                        if int(item["floor"]) == floor_number]
    matched_detection_indices = set(matching.values())

    floor_elevation = float(floor.get("elevation", 0.0))
    floor_distractors = [
        item for item in (red_distractors or [])
        if abs(float(item.get("pose", [0, 0, -99])[2]) -
               (floor_elevation + 0.15)) < 0.45]
    if floor_distractors:
        axis.scatter(
            [item["pose"][0] for item in floor_distractors],
            [item["pose"][1] for item in floor_distractors],
            marker="s", s=42, color="#ef4444", edgecolor="#7f1d1d",
            linewidth=0.7, zorder=6)

    for truth_index in floor_truth:
        expected = truth[truth_index]
        tx, ty = float(expected["pose"][0]), float(expected["pose"][1])
        axis.scatter(tx, ty, marker="*", s=180, color="#d32f2f",
                     edgecolor="#7f0000", linewidth=0.8, zorder=7)
        label = expected.get("id", str(truth_index)).replace("danger_red_sphere_", "T")
        axis.annotate(label, (tx, ty), xytext=(5, 7), textcoords="offset points",
                      fontsize=7, color="#7f0000", weight="bold")
        if truth_index not in matching:
            axis.scatter(tx, ty, marker="x", s=230, color="#212121",
                         linewidth=2.5, zorder=10)

    for detection_index in floor_detections:
        detection = detections[detection_index]
        dx, dy = float(detection["position"][0]), float(detection["position"][1])
        if detection_index in matched_detection_indices:
            axis.scatter(dx, dy, marker="o", s=105, facecolor="none",
                         edgecolor="#00a152", linewidth=2.4, zorder=9)
        else:
            axis.scatter(dx, dy, marker="x", s=160, color="#8e24aa",
                         linewidth=2.5, zorder=9)
        axis.annotate("D{}".format(detection_index + 1), (dx, dy),
                      xytext=(5, -11), textcoords="offset points",
                      fontsize=7, color="#1b5e20" if detection_index in matched_detection_indices else "#6a1b9a")

    floor_distances = []
    for truth_index, detection_index in matching.items():
        if truth_index not in floor_truth:
            continue
        expected, detection = truth[truth_index], detections[detection_index]
        tx, ty = float(expected["pose"][0]), float(expected["pose"][1])
        dx, dy = float(detection["position"][0]), float(detection["position"][1])
        distance = distances[(detection_index, truth_index)]
        floor_distances.append(distance)
        axis.plot([tx, dx], [ty, dy], color="#00a152", linestyle="--",
                  linewidth=1.0, alpha=0.85, zorder=6)
        axis.text((tx + dx) / 2.0, (ty + dy) / 2.0,
                  "{:.2f}m".format(distance), fontsize=6, color="#00695c",
                  bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 0.5})

    tp = sum(1 for truth_index in floor_truth if truth_index in matching)
    fp = sum(1 for index in floor_detections if index not in matched_detection_indices)
    fn = len(floor_truth) - tp
    axis.set_title(
        "Floor {} offline danger-source evaluation\nTruth={} | TP={} | FP={} | FN={} | tolerance={:.2f}m".format(
            floor_number, len(floor_truth), tp, fp, fn, tolerance),
        fontsize=11, weight="bold",
    )
    axis.set_xlabel("world x (m)")
    axis.set_ylabel("world y (m)")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.2)
    axis.legend(handles=[
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#d32f2f",
               markeredgecolor="#7f0000", markersize=13, label="Truth danger source"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#ef4444",
               markeredgecolor="#7f1d1d", markersize=7,
               label="Red non-danger distractor"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor="#00a152", markeredgewidth=2.2, markersize=9,
               label="TP: correct detection"),
        Line2D([0], [0], marker="x", color="#8e24aa", markersize=9,
               markeredgewidth=2.2, label="FP: false detection"),
        Line2D([0], [0], marker="x", color="#212121", markersize=10,
               markeredgewidth=2.4, label="FN: missed truth"),
        Patch(facecolor="#94a3b8", edgecolor="#1f2937", alpha=0.82,
              label="Room furniture / obstacle (type labelled)"),
    ], loc="upper right", fontsize=8)
    return {"floor": floor_number, "truth": len(floor_truth), "tp": tp,
            "fp": fp, "fn": fn,
            "mean_match_distance_m": (sum(floor_distances) / len(floor_distances)
                                      if floor_distances else None),
            "maximum_match_distance_m": max(floor_distances) if floor_distances else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    results = Path(args.results).resolve()
    output = Path(args.output_dir).resolve() if args.output_dir else results / "visualization"
    output.mkdir(parents=True, exist_ok=True)

    layout = read_json(results / "layout_metadata.json")
    detection_record = read_json(results / "red_ball_detections.json")
    config = read_json(results / "mission_config.json")
    truth = layout.get("danger_red_spheres", [])
    red_distractors = layout.get("red_distractors", [])
    if not truth:
        # An official scene's layout carries no sphere: the run is not told
        # where they are.  Plot against the referee copy the runner saved
        # beside the results, which is also what the score is computed from.
        referee = results / "official_danger_truth.json"
        if referee.is_file():
            truth = []
            for source in read_json(referee).get("danger_sources", []):
                record = dict(source)
                record["pose"] = list(source.get("position", []))
                truth.append(record)
            red_distractors = [
                dict(item, pose=list(item.get("position", [])))
                for item in read_json(referee).get("distraction_sources", [])
                if str(item.get("color")) == "red"]
    detections = detection_record.get("detections", [])
    tolerance = float(config["red_ball_detection"]["matching_tolerance_m"])
    matching, distances = maximum_truth_matching(detections, truth, tolerance)

    floors = sorted(layout.get("floors", []), key=lambda item: int(item["floor_index"]))
    reports = []
    combined, axes = plt.subplots(1, len(floors), figsize=(21, 12), squeeze=False)
    for column, floor in enumerate(floors):
        floor_number = int(floor["floor_index"]) + 1
        report = draw_floor(axes[0][column], floor, floor_number, truth,
                            detections, matching, distances, tolerance,
                            red_distractors)
        reports.append(report)
        single, single_axis = plt.subplots(figsize=(8, 11))
        draw_floor(single_axis, floor, floor_number, truth, detections,
                   matching, distances, tolerance, red_distractors)
        single.tight_layout()
        single.savefig(output / "danger_truth_evaluation_floor_{}.png".format(floor_number),
                       dpi=180, bbox_inches="tight")
        plt.close(single)
    combined.suptitle("Three-floor red-ball truth matching (offline evaluation only)",
                      fontsize=15, weight="bold", y=0.99)
    combined.tight_layout(rect=(0, 0, 1, 0.92))
    combined.savefig(output / "danger_truth_evaluation_all_floors.png",
                     dpi=180, bbox_inches="tight")
    plt.close(combined)

    matched_detection_indices = set(matching.values())
    payload = {
        "schema": "three_floor_red_ball_offline_truth_evaluation_v1",
        "matching_tolerance_m": tolerance,
        "truth_count": len(truth),
        "detection_count": len(detections),
        "true_positive_count": len(matching),
        "false_positive_count": len(detections) - len(matched_detection_indices),
        "false_negative_count": len(truth) - len(matching),
        "furniture_footprints_rendered": sum(
            len(room.get("furniture", []))
            for floor in floors for room in floor.get("rooms", [])),
        "floors": reports,
        "matches": [
            {"truth_id": truth[truth_index].get("id"),
             "room_id": truth[truth_index].get("room_id"),
             "detection_track_id": detections[detection_index].get("track_id"),
             "distance_m": distances[(detection_index, truth_index)]}
            for truth_index, detection_index in sorted(matching.items())
        ],
    }
    with open(output / "danger_truth_evaluation.json", "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
