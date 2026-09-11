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


PREFIX = "referee scoring: "


def update_acceptance(path, score):
    """Fill the verdict the run itself could not reach.

    The mission is judged on the published range while it runs, so these four
    checks come out of it as null.  They are answerable once the truth is
    open, which is what this settles.  Rerunning replaces the previous
    verdict rather than stacking a second copy of it.
    """
    payload = read_json(path)
    checks = payload.setdefault("checks", {})
    no_miss = score["false_negatives"] == 0
    no_alarm = score["false_positives"] == 0
    checks["all_scene_red_balls_detected"] = no_miss
    checks["zero_false_negative_dangers"] = no_miss
    checks["zero_false_positive_dangers"] = no_alarm
    checks["every_selected_danger_room_detected"] = not score[
        "rooms_with_a_missed_source"]
    checks.pop("red_ball_truth_scoring_deferred_to_referee", None)
    checks["red_ball_truth_scored_after_run"] = True

    failures = [reason for reason in payload.get("failure_reasons", [])
                if not str(reason).startswith(PREFIX)]
    if score["missed_sources"]:
        failures.append("{}{} danger source(s) never confirmed: {}".format(
            PREFIX, score["false_negatives"],
            ", ".join(str(item) for item in score["missed_sources"])))
    if score["false_positives"]:
        failures.append("{}{} confirmed detection(s) match no danger "
                        "source".format(PREFIX, score["false_positives"]))
    payload["failure_reasons"] = failures
    # A run that passed everything it could check itself still fails here if
    # it missed a sphere or invented one.
    payload["passed"] = bool(payload.get("passed")) and score["passed"]
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True,
                        help="Run results directory holding the detection record.")
    parser.add_argument("--truth", required=True,
                        help="Official danger_truth.json written for the referee.")
    parser.add_argument("--detections", default="red_ball_detections.json")
    parser.add_argument("--tolerance", type=float, default=1.5)
    parser.add_argument("--output")
    parser.add_argument(
        "--acceptance",
        help="Acceptance record to write the verdict back into. Defaults to "
             "three_floor_rl_acceptance.json in the results directory.")
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

    acceptance = args.acceptance or os.path.join(
        args.results, "three_floor_rl_acceptance.json")
    if os.path.isfile(acceptance):
        update_acceptance(acceptance, score)
        score["acceptance_updated"] = os.path.abspath(acceptance)

    print(json.dumps(score, indent=2))
    return 0 if score["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
