#!/usr/bin/env python3
"""Score a finished run against the official referee truth.

The mission is never told how many danger spheres the scene holds, so
whether it missed one is not a question it can answer about itself.  This
answers it afterwards, the way a referee does: the run's confirmed
detections against danger_truth.json.

    score_official_danger_truth.py --results <dir> --truth <danger_truth.json>
"""

import argparse
import importlib.util
import json
import os
import sys


def _load_checker():
    """Reuse the acceptance checker's one-to-one same-floor matching."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "check_three_floor_rl_mission.py")
    spec = importlib.util.spec_from_file_location(
        "check_three_floor_rl_mission_for_scoring", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True,
                        help="Run results directory holding the detection record.")
    parser.add_argument("--truth", required=True,
                        help="Official danger_truth.json written for the referee.")
    parser.add_argument("--detections", default="red_ball_detections.json")
    parser.add_argument("--tolerance", type=float, default=1.5)
    parser.add_argument("--output")
    args = parser.parse_args()

    checker = _load_checker()
    payload = read_json(os.path.join(args.results, args.detections))
    detections = [item for item in payload.get("detections", [])
                  if isinstance(item.get("position"), (list, tuple))]
    # The truth records carry "position"; the matcher reads "pose".
    truth = []
    for source in read_json(args.truth).get("danger_sources", []):
        record = dict(source)
        record["pose"] = list(source.get("position", []))
        truth.append(record)

    matching = checker._maximum_truth_matching(
        detections, truth, float(args.tolerance))
    matched_truth = set(matching)
    missed = [truth[index].get("model_name", truth[index].get("id", index))
              for index in range(len(truth)) if index not in matched_truth]
    false_positives = max(0, len(detections) - len(matching))
    rooms_missed = sorted({str(truth[index].get("room_id"))
                           for index in range(len(truth))
                           if index not in matched_truth})

    score = {
        "schema": "official_danger_truth_score_v1",
        "truth_file": os.path.abspath(args.truth),
        "danger_sources": len(truth),
        "confirmed_detections": len(detections),
        "true_positives": len(matching),
        "false_positives": false_positives,
        "false_negatives": len(missed),
        "missed_sources": missed,
        "rooms_with_a_missed_source": rooms_missed,
        "matching_tolerance_m": float(args.tolerance),
        "passed": not missed and not false_positives,
    }
    destination = args.output or os.path.join(
        args.results, "official_danger_truth_score.json")
    with open(destination, "w", encoding="utf-8") as stream:
        json.dump(score, stream, indent=2)
        stream.write("\n")
    print(json.dumps(score, indent=2))
    return 0 if score["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
