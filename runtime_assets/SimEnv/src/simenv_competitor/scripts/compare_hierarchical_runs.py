#!/usr/bin/env python3
"""Aggregate exploration metrics from baseline or hierarchical result folders."""

import argparse
import csv
import json
import os


def load_json(path):
    try:
        with open(path, encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return {}


def visit_metrics(directory):
    path = os.path.join(directory, "logs", "region_visit.csv")
    rows = []
    try:
        with open(path, newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except OSError:
        pass
    counts = {}
    for row in rows:
        region_id = row.get("region_id", "")
        counts[region_id] = counts.get(region_id, 0) + 1
    return len(rows), sum(max(0, count - 1) for count in counts.values())


def metrics(directory):
    hierarchical = load_json(os.path.join(
        directory, "hierarchical_summary.json"))
    baseline = load_json(os.path.join(directory, "baseline_summary.json"))
    summary = hierarchical or baseline
    graph = load_json(os.path.join(
        directory, "logs", "exploration_graph.json"))
    visits, repeated = visit_metrics(directory)
    nodes = graph.get("nodes", [])
    regions = [
        node for node in nodes
        if node.get("type") in ("region", "visited")]
    covered = [
        node for node in regions
        if node.get("visited") or
        float(node.get("coverage_ratio", 0.0)) >= 0.90]
    return {
        "run": os.path.abspath(directory),
        "elapsed_sec": summary.get(
            "elapsed_sec", summary.get("duration_sec")),
        "path_length_m": summary.get(
            "total_path_length", summary.get("trajectory_length_m")),
        "goal_count": summary.get(
            "goal_count", summary.get("goals_attempted")),
        "region_visits": visits,
        "repeated_region_entries": repeated,
        "region_count": len(regions),
        "covered_region_count": len(covered),
        "region_coverage_ratio": (
            len(covered) / float(len(regions)) if regions else None),
        "unknown_ratio": graph.get("stats", {}).get("unknown_ratio"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs="+")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = {"runs": [metrics(path) for path in args.result_dirs]}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

