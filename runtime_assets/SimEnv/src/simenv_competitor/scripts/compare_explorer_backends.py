#!/usr/bin/env python3
"""Compare baseline, TARE and UFOExplorer result directories."""

import argparse
import json
import math
import os


def load_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return default


def load_jsonl(path):
    rows = []
    try:
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    rows.append(json.loads(line))
    except (OSError, ValueError):
        pass
    return rows


def nested_number(documents, names):
    names = set(names)
    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in names:
                    try:
                        number = float(child)
                        if math.isfinite(number):
                            return number
                    except (TypeError, ValueError):
                        pass
                result = visit(child)
                if result is not None:
                    return result
        elif isinstance(value, list):
            for child in value:
                result = visit(child)
                if result is not None:
                    return result
        return None
    return visit(documents)


def metrics(directory):
    names = ("baseline_summary.json", "hierarchical_summary.json",
             "ufoexplorer_summary.json", "map_statistics.json",
             "goal_execution_metrics.json", "telemetry_performance.json")
    documents = [load_json(os.path.join(directory, name), {}) for name in names]
    ufo = load_json(os.path.join(
        directory, "logs", "ufoexplorer_summary.json"), None)
    if ufo is None:
        ufo = load_json(os.path.join(directory, "ufoexplorer_summary.json"), {})
    goal_rows = load_jsonl(os.path.join(
        directory, "logs", "ufoexplorer_goal_history.jsonl"))
    stall_rows = load_jsonl(os.path.join(
        directory, "logs", "velocity_watchdog.jsonl"))
    map_stats = load_json(os.path.join(directory, "map_statistics.json"), {})
    result = {
        "total_distance": nested_number(documents, (
            "total_distance", "total_distance_m", "distance_m")),
        "runtime": nested_number(documents, (
            "total_runtime", "runtime", "duration_sec", "elapsed_sec")),
        "cmd_vel_zero_time_ratio": nested_number(documents, (
            "cmd_vel_zero_time_ratio", "zero_velocity_ratio",
            "stationary_time_ratio")),
        "stall_count": int(ufo.get("stall_count", len(stall_rows))),
        "goal_count": int(nested_number(documents, (
            "goal_count", "total_goals", "goals_attempted")) or
            sum(row.get("ufo_raw_output_type") == "execution"
                for row in goal_rows)),
        "path_count": int(ufo.get("ufo_path_count",
                                  sum(row.get("ufo_raw_output_type") == "path"
                                      for row in goal_rows))),
        "fallback_count": int(ufo.get("ufo_fallback_count",
                                      sum(bool(row.get("fallback_used"))
                                          for row in goal_rows))),
        "scan_lite_reject_count": int(ufo.get(
            "ufo_scan_reject_count",
            sum(row.get("scan_lite_status") == "REJECTED"
                for row in goal_rows))),
        "room_entry_count": nested_number(documents, (
            "room_entry_count", "rooms_entered")),
        "explored_free_area": nested_number(
            [map_stats] + documents, ("explored_free_area",
                                      "explored_free_area_m2",
                                      "observed_free_area_m2")),
        "unknown_ratio_final": nested_number(
            [map_stats] + documents, ("unknown_ratio_final",
                                      "unknown_ratio")),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_result_dir")
    parser.add_argument("tare_result_dir")
    parser.add_argument("ufo_result_dir")
    parser.add_argument("--output",
                        default="results/explorer_backend_comparison.json")
    args = parser.parse_args()
    payload = {
        "schema": "simenv_explorer_backend_comparison_v1",
        "online_truth_used": False,
        "room_entry_count_source": "offline result metrics only",
        "backends": {
            "baseline": metrics(os.path.abspath(args.baseline_result_dir)),
            "tare": metrics(os.path.abspath(args.tare_result_dir)),
            "ufoexplorer": metrics(os.path.abspath(args.ufo_result_dir)),
        },
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(output)


if __name__ == "__main__":
    main()
