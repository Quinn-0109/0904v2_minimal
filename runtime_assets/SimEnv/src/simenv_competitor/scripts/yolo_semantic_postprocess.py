#!/usr/bin/env python3
"""Post-process YOLO semantic observations into reportable confirmed results."""

import argparse
import collections
import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


CLASS_STYLE = {
    "red_ball": {"color": "#dc2626", "marker": "o", "label": "confirmed red ball"},
    "stair": {"color": "#f59e0b", "marker": "^", "label": "confirmed stair"},
    "elevator": {"color": "#7c3aed", "marker": "s", "label": "confirmed elevator"},
}

DISTRACTOR_STYLE = {
    "red_box": {"color": "#ef4444", "marker": "s", "label": "red box distractor"},
    "green_sphere": {"color": "#22c55e", "marker": "o", "label": "green sphere distractor"},
}


def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return default


def load_jsonl(path):
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return rows


def observation_gate(row):
    label = row.get("class")
    confidence = float(row.get("confidence", 0.0))
    depth_m = float(row.get("depth_m", 0.0))
    pixel = row.get("pixel") or {}
    bbox = pixel.get("bbox") or [0, 0, 0, 0]
    width = max(1.0, float(bbox[2]) - float(bbox[0]))
    height = max(1.0, float(bbox[3]) - float(bbox[1]))
    image_width = float(row.get("image_width", 640.0))
    image_height = float(row.get("image_height", 480.0))
    area_ratio = float(pixel.get(
        "area_ratio", (width * height) / max(1.0, image_width * image_height)))
    if "edge_clipped" in pixel:
        edge_clipped = bool(pixel.get("edge_clipped", False))
    else:
        margin = 8.0
        edge_clipped = (
            float(bbox[0]) <= margin or float(bbox[1]) <= margin or
            float(bbox[2]) >= image_width - margin or
            float(bbox[3]) >= image_height - margin)
    aspect = max(width, height) / max(1.0, min(width, height))

    reasons = []
    if label == "red_ball":
        if confidence < 0.70:
            reasons.append("low_confidence")
        if area_ratio < 0.0005:
            reasons.append("red_ball_box_too_small")
        if area_ratio > 0.12:
            reasons.append("red_ball_box_too_large")
        if aspect > 1.8:
            reasons.append("red_ball_box_not_round")
        if depth_m <= 0.15 or depth_m > 10.0:
            reasons.append("red_ball_depth_out_of_range")
    elif label in ("stair", "elevator"):
        if confidence < 0.80:
            reasons.append("low_confidence")
        if edge_clipped:
            reasons.append("edge_clipped_large_structure")
        if area_ratio > 0.35:
            reasons.append("oversized_structure_box")
        if depth_m < 0.80 or depth_m > 8.0:
            reasons.append("semantic_depth_out_of_range")
    else:
        reasons.append("unsupported_class")
    return not reasons, reasons


def cluster_rows(rows, radius):
    clusters = []
    for row in rows:
        mp = row.get("map_position") or {}
        if "x" not in mp or "y" not in mp:
            continue
        x = float(mp["x"])
        y = float(mp["y"])
        matched = None
        for cluster in clusters:
            cx = sum(float((r.get("map_position") or {})["x"]) for r in cluster) / len(cluster)
            cy = sum(float((r.get("map_position") or {})["y"]) for r in cluster) / len(cluster)
            if math.hypot(x - cx, y - cy) <= radius:
                matched = cluster
                break
        if matched is None:
            clusters.append([row])
        else:
            matched.append(row)
    return clusters


def summarize_cluster(label, cluster):
    xs = [float((row.get("map_position") or {})["x"]) for row in cluster]
    ys = [float((row.get("map_position") or {})["y"]) for row in cluster]
    zs = [float((row.get("map_position") or {}).get("z", 0.0)) for row in cluster]
    confs = [float(row.get("confidence", 0.0)) for row in cluster]
    stamps = [float(row.get("timestamp", 0.0)) for row in cluster]
    return {
        "class": label,
        "observations": len(cluster),
        "max_confidence": round(max(confs), 4),
        "mean_confidence": round(sum(confs) / len(confs), 4),
        "position": {
            "x": round(sum(xs) / len(xs), 4),
            "y": round(sum(ys) / len(ys), 4),
            "z": round(sum(zs) / len(zs), 4),
        },
        "first_timestamp": round(min(stamps), 4),
        "last_timestamp": round(max(stamps), 4),
    }


def workspace_root_from_script():
    return Path(__file__).resolve().parents[2]


def load_world_distractors(run_dir):
    candidates = [
        run_dir / "competition_scene.world",
        run_dir.parent.parent / "generated_building" / "competition_scene.world",
        workspace_root_from_script() / "generated_building" / "competition_scene.world",
    ]
    world = next((path for path in candidates if path.exists()), None)
    if world is None:
        return []
    try:
        root = ET.parse(world).getroot()
    except ET.ParseError:
        return []

    objects = []
    for model in root.iter("model"):
        name = model.get("name") or ""
        if not name.startswith("distractor_"):
            continue
        pose_text = (model.findtext("pose") or "").strip()
        parts = pose_text.split()
        if len(parts) < 2:
            continue
        world_x, world_y = float(parts[0]), float(parts[1])
        if "red_box" in name:
            kind = "red_box"
        elif "green_sphere" in name:
            kind = "green_sphere"
        else:
            kind = "other"
        objects.append({
            "name": name,
            "kind": kind,
            "position": {
                "x": world_y,
                "y": -world_x,
            },
        })
    return objects


def confirmed_semantics(run_dir):
    observations = load_jsonl(run_dir / "logs" / "yolo_semantic_observations.jsonl")
    accepted = []
    rejected = []
    for row in observations:
        ok, reasons = observation_gate(row)
        row = dict(row)
        row["accepted_by_postprocess"] = ok
        row["postprocess_rejection_reasons"] = reasons
        if ok:
            accepted.append(row)
        else:
            rejected.append(row)

    confirmed = []
    for label in ("stair", "elevator"):
        label_rows = [row for row in accepted if row.get("class") == label]
        for cluster in cluster_rows(label_rows, radius=0.75):
            # Structure classes need several non-edge observations; otherwise
            # they remain candidates because the current model confuses walls
            # and door frames with stairs/elevators.
            if len(cluster) >= 4 and max(float(r.get("confidence", 0.0)) for r in cluster) >= 0.85:
                confirmed.append(summarize_cluster(label, cluster))

    # Red-ball reporting remains driven by the existing RGB-D hazard tracker,
    # which already clusters and multi-frame-confirms sphere candidates.
    danger = load_json(run_dir / "detected_danger.json", {})
    for hazard in danger.get("confirmed_hazards", []):
        position = hazard.get("position")
        if not isinstance(position, dict):
            continue
        confirmed.append({
            "class": "red_ball",
            "observations": int(hazard.get("observations", 0)),
            "max_confidence": float(hazard.get("confidence", 0.0)),
            "mean_confidence": float(hazard.get("confidence", 0.0)),
            "position": {
                "x": float(position.get("x", 0.0)),
                "y": float(position.get("y", 0.0)),
                "z": float(position.get("z", 0.0)),
            },
            "source_id": hazard.get("id"),
            "room_id": hazard.get("room_id"),
        })
    return observations, accepted, rejected, confirmed


def draw_map(run_dir, confirmed, rejected):
    vis_dir = run_dir / "visualization"
    vis_dir.mkdir(parents=True, exist_ok=True)
    layout = load_json(vis_dir / "room_layout_projection.json", {})
    traj = load_json(run_dir / "trajectory_path.json", {}).get("samples", [])
    goals = load_json(run_dir / "exploration_goal_history.json", [])

    fig, ax = plt.subplots(figsize=(14, 8), dpi=180)
    pgm = vis_dir / "room_layout_projection.pgm"
    if pgm.exists() and layout:
        img = np.asarray(Image.open(pgm).convert("L"))
        origin_x = float(layout["origin_x"])
        origin_y = float(layout["origin_y"])
        res = float(layout["resolution"])
        width = int(layout["width"])
        height = int(layout["height"])
        extent = [origin_x, origin_x + width * res, origin_y, origin_y + height * res]
        ax.imshow(img, cmap="gray", origin="lower", extent=extent, alpha=0.72)
    else:
        ax.set_facecolor("#f4f4f4")

    if traj:
        xs = [sample["position_x"] for sample in traj]
        ys = [sample["position_y"] for sample in traj]
        ax.plot(xs, ys, color="#111827", lw=1.8, label="robot trajectory", zorder=3)
        ax.scatter(xs[0], ys[0], s=80, marker="o", color="#16a34a",
                   edgecolor="white", linewidth=1.2, label="start", zorder=6)
        ax.scatter(xs[-1], ys[-1], s=90, marker="X", color="#111827",
                   edgecolor="white", linewidth=1.2, label="end", zorder=6)

    goal_xy = []
    for goal in goals:
        point = goal.get("goal")
        if isinstance(point, list) and len(point) >= 2:
            goal_xy.append((point[0], point[1]))
    if goal_xy:
        gx, gy = zip(*goal_xy)
        ax.scatter(gx, gy, s=28, marker="D", color="#2563eb", alpha=0.45,
                   label="exploration goals", zorder=4)

    distractors = load_world_distractors(run_dir)
    for kind, style in DISTRACTOR_STYLE.items():
        rows = [item for item in distractors if item["kind"] == kind]
        if not rows:
            continue
        xs = [item["position"]["x"] for item in rows]
        ys = [item["position"]["y"] for item in rows]
        ax.scatter(xs, ys, s=62, marker=style["marker"], facecolors="none",
                   edgecolors=style["color"], linewidths=1.8,
                   label=f"{style['label']} ({len(rows)})", zorder=6)
        for item in rows:
            label = item["name"].replace("distractor_", "")
            ax.annotate(label, xy=(item["position"]["x"], item["position"]["y"]),
                        xytext=(6, -10), textcoords="offset points",
                        fontsize=7, color=style["color"], zorder=7)

    rejected_semantic = [row for row in rejected if row.get("class") in ("stair", "elevator")]
    if rejected_semantic:
        rx = [float((row.get("map_position") or {})["x"]) for row in rejected_semantic
              if "x" in (row.get("map_position") or {})]
        ry = [float((row.get("map_position") or {})["y"]) for row in rejected_semantic
              if "y" in (row.get("map_position") or {})]
        if rx and ry:
            ax.scatter(rx, ry, s=34, marker="x", color="#6b7280", alpha=0.55,
                       label="rejected stair/elevator raw candidates", zorder=4)

    counts = collections.Counter(item["class"] for item in confirmed)
    for label in ("red_ball", "stair", "elevator"):
        rows = [item for item in confirmed if item["class"] == label]
        if not rows:
            continue
        style = CLASS_STYLE[label]
        x = [item["position"]["x"] for item in rows]
        y = [item["position"]["y"] for item in rows]
        sizes = [90 + 15 * min(10, int(item.get("observations", 1))) for item in rows]
        ax.scatter(x, y, s=sizes, marker=style["marker"], color=style["color"],
                   edgecolor="white", linewidth=1.0,
                   label=f"{style['label']} ({counts[label]})", zorder=7)
        for index, item in enumerate(rows, 1):
            ax.annotate(f"{label} {index}\\n{item.get('max_confidence', 0):.2f}",
                        xy=(item["position"]["x"], item["position"]["y"]),
                        xytext=(8, 8), textcoords="offset points", fontsize=8,
                        color=style["color"],
                        bbox=dict(boxstyle="round,pad=0.22", fc="white",
                                  ec=style["color"], alpha=0.84),
                        zorder=8)

    ax.set_title("Filtered semantic results during 4-room mapping run",
                 fontsize=14, weight="bold")
    ax.set_xlabel("map x (m)")
    ax.set_ylabel("map y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d1d5db", linewidth=0.5, alpha=0.45)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    summary = ", ".join(f"{label}: {counts.get(label, 0)} confirmed"
                        for label in ("red_ball", "stair", "elevator"))
    ax.text(0.01, 0.01, summary, transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec="#9ca3af", alpha=0.9))
    output = vis_dir / "17_yolo_semantic_filtered_map.png"
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="SimEnv result directory")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    observations, accepted, rejected, confirmed = confirmed_semantics(run_dir)
    image = draw_map(run_dir, confirmed, rejected)

    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    rejected_path = logs_dir / "yolo_semantic_rejected_candidates.jsonl"
    with rejected_path.open("w", encoding="utf-8") as stream:
        for row in rejected:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "schema": "simenv_yolo_semantic_postprocess_v1",
        "source_observations": str(run_dir / "logs" / "yolo_semantic_observations.jsonl"),
        "visualization": str(image),
        "raw_counts": dict(collections.Counter(row.get("class") for row in observations)),
        "accepted_counts": dict(collections.Counter(row.get("class") for row in accepted)),
        "rejected_counts": dict(collections.Counter(row.get("class") for row in rejected)),
        "confirmed_counts": dict(collections.Counter(item["class"] for item in confirmed)),
        "confirmed": confirmed,
        "rejected_candidates_log": str(rejected_path),
    }
    summary_path = run_dir / "visualization" / "17_yolo_semantic_filtered_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True),
                            encoding="utf-8")
    print(image)
    print(summary_path)
    print(json.dumps(summary["confirmed_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
