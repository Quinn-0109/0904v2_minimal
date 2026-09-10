#!/usr/bin/env python3
"""Validate the stable20 geometry and danger contracts without Gazebo."""
import argparse
import importlib.util
import json
import multiprocessing
import os
import time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(SCRIPT_DIR)
_WORKER_RANDOMIZER = None
_WORKER_CHECKER = None
_WORKER_LAYOUT = None
_WORKER_CONFIG = None
_WORKER_RED_DISTRACTORS = None


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(SCRIPT_DIR, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def initialize_worker(layout_path, config_path, world_path):
    global _WORKER_RANDOMIZER, _WORKER_CHECKER, _WORKER_LAYOUT, _WORKER_CONFIG
    global _WORKER_RED_DISTRACTORS
    _WORKER_RANDOMIZER = load_module(
        "stable20_randomizer_{}".format(os.getpid()),
        "randomize_three_floor_scene.py")
    _WORKER_CHECKER = load_module(
        "stable20_checker_{}".format(os.getpid()),
        "check_three_floor_rl_mission.py")
    _WORKER_LAYOUT = _WORKER_RANDOMIZER.read_json(layout_path)
    _WORKER_CONFIG = _WORKER_RANDOMIZER.read_json(config_path)
    _WORKER_RED_DISTRACTORS = (
        _WORKER_RANDOMIZER.read_red_distractors_from_world(world_path))


def validate_seed(seed):
    started = time.monotonic()
    try:
        layout, config, _furniture, _danger, truth, _scans, _seeds = (
            _WORKER_RANDOMIZER.build_randomized_scene(
                _WORKER_LAYOUT, _WORKER_CONFIG, seed,
                _WORKER_RED_DISTRACTORS))
        errors = _WORKER_CHECKER.validate_config(config)
        errors += _WORKER_CHECKER.validate_layout_clearance(config, layout)
        errors += _WORKER_CHECKER.validate_randomized_layout(config, layout)
        route = float(layout["metadata"]["runtime_floor_route_m"])
        minimum_separation = min(
            float(room["room_contract"]["planned_viewpoint_separation_m"])
            for floor in layout["floors"] for room in floor["rooms"])
        minimum_hazard_parallax = min(
            float(item["dual_view_parallax_deg"]) for item in truth)
        minimum_distractor_gap = min(
            float(view["minimum_red_distractor_edge_gap_rad"])
            for item in truth for view in item["view_evidence"])
        minimum_vertical_slack = min(
            float(view["vertical_projection_slack_rad"])
            for item in truth for view in item["view_evidence"])
        minimum_range_margin = min(
            float(view["range_margin_m"])
            for item in truth for view in item["view_evidence"])
        return {
            "seed": seed, "passed": not errors, "errors": errors,
            "danger_source_count": len(truth),
            "route_m": round(route, 3),
            "minimum_planned_separation_m": round(minimum_separation, 3),
            "minimum_hazard_parallax_deg": round(
                minimum_hazard_parallax, 3),
            "minimum_red_distractor_edge_gap_deg": round(
                minimum_distractor_gap * 180.0 / 3.141592653589793, 3),
            "minimum_vertical_projection_slack_deg": round(
                minimum_vertical_slack * 180.0 / 3.141592653589793, 3),
            "minimum_hazard_range_margin_m": round(
                minimum_range_margin, 3),
            "elapsed_wall_sec": round(time.monotonic() - started, 3),
        }
    except Exception as error:
        return {
            "seed": seed, "passed": False,
            "errors": ["{}: {}".format(type(error).__name__, error)],
            "elapsed_wall_sec": round(time.monotonic() - started, 3),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", required=True)
    parser.add_argument("--world", required=True)
    parser.add_argument("--config", default=os.path.join(
        PACKAGE_DIR, "config", "three_floor_rl_mission.json"))
    parser.add_argument("--seed-first", type=int, default=40)
    parser.add_argument("--seed-last", type=int, default=139)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    args = parser.parse_args()
    records = []
    started = time.monotonic()
    passed = True
    seeds = list(range(args.seed_first, args.seed_last + 1))
    with multiprocessing.Pool(
            processes=max(1, args.workers), initializer=initialize_worker,
            initargs=(args.layout, args.config, args.world)) as pool:
        for record in pool.imap_unordered(validate_seed, seeds):
            records.append(record)
            passed = passed and bool(record["passed"])
            print(json.dumps(record, sort_keys=True), flush=True)
    records.sort(key=lambda item: int(item["seed"]))
    payload = {
        "schema": "stable20_static_randomization_v1",
        "seed_first": args.seed_first,
        "seed_last": args.seed_last,
        "completed_seed_count": len(records),
        "passed": bool(passed and len(records) ==
                       args.seed_last - args.seed_first + 1),
        "elapsed_wall_sec": round(time.monotonic() - started, 3),
        "workers": max(1, args.workers),
        "records": records,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    temporary = args.output + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, args.output)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
