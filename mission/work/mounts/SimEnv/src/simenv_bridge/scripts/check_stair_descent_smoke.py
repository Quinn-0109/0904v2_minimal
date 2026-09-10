#!/usr/bin/env python3
"""Strict, machine-readable acceptance for the physical two-segment descent."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile


SUCCESS = "FIRST_FLOOR_RETURNED"
SEGMENT_ONE_SUCCESS = "SECOND_FLOOR_DESCENT_REACHED"


def load_json(path, failures):
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, ValueError) as error:
        failures.append("cannot read {}: {}".format(path, error))
        return None
    if not isinstance(value, dict):
        failures.append("{} must contain a JSON object".format(path))
        return None
    return value


def number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def check_results(results, expected_policy_basename="policy_act_inference_stair.pt"):
    root = Path(results)
    failures = []
    checks = {}

    primary = load_json(root / "logs" / "stair_descent.json", failures)
    segment_one = load_json(
        root / "logs" / "third_to_second_floor_stair_transition.json", failures
    )
    segment_two = load_json(
        root / "logs" / "second_to_first_floor_stair_transition.json", failures
    )
    returned = load_json(root / "first_floor_returned.json", failures)

    checks["manager_terminal"] = bool(primary and primary.get("phase") == SUCCESS)
    checks["segment_one_terminal"] = bool(
        segment_one and segment_one.get("phase") == SEGMENT_ONE_SUCCESS
    )
    checks["segment_two_terminal"] = bool(
        segment_two and segment_two.get("phase") == SUCCESS
    )
    checks["return_marker"] = bool(returned and returned.get("phase") == SUCCESS)

    if primary and not checks["manager_terminal"]:
        failures.append(
            "stair_descent terminal {!r}, expected {!r}".format(
                primary.get("phase"), SUCCESS
            )
        )
    if segment_one and not checks["segment_one_terminal"]:
        failures.append("F3->F2 segment did not complete")
    if segment_two and not checks["segment_two_terminal"]:
        failures.append("F2->F1 segment did not complete")
    if returned and not checks["return_marker"]:
        failures.append("first_floor_returned.json has no success terminal")

    policy = str(primary.get("policy", "")) if primary else ""
    checks["stair_policy"] = os.path.basename(policy) == expected_policy_basename
    if primary and not checks["stair_policy"]:
        failures.append(
            "descent policy {!r}, expected basename {!r}".format(
                policy, expected_policy_basename
            )
        )

    trace = primary.get("trace", []) if primary else []
    if not isinstance(trace, list):
        trace = []
    trace_items = [item for item in trace if isinstance(item, dict)]
    phases_by_segment = {0: set(), 1: set()}
    for item in trace_items:
        segment = item.get("segment")
        if segment in phases_by_segment:
            phases_by_segment[segment].add(str(item.get("phase", "")))
    flight_phases = {"STAIR_DESCENT_FLIGHT_B", "STAIR_DESCENT_FLIGHT_A"}
    checks["both_segments_traced"] = all(
        flight_phases.issubset(phases_by_segment[index]) for index in (0, 1)
    )
    checks["intermediate_landing_turn_traced"] = any(
        item.get("phase") == "STAIR_DESCENT_SEGMENT_TURN"
        for item in trace_items
    )
    if not checks["both_segments_traced"]:
        failures.append("trace does not contain flight A and B on both segments")
    if not checks["intermediate_landing_turn_traced"]:
        failures.append("trace does not contain the F2 intermediate landing turn")

    bad_markers = (
        "FAILED", "FAILURE", "TIMEOUT", "ATTITUDE_LOST", "FALL", "UNSTABLE"
    )
    bad_phases = sorted({
        str(item.get("phase", ""))
        for item in trace_items
        if any(marker in str(item.get("phase", "")) for marker in bad_markers)
    })
    checks["no_failure_phase"] = not bad_phases
    if bad_phases:
        failures.append("failure phase(s) in trace: {}".format(", ".join(bad_phases)))

    truth_items = [
        item for item in trace_items
        if number(item.get("truth_z")) is not None
    ]
    start_z = number(truth_items[0].get("truth_z")) if truth_items else None
    final = truth_items[-1] if truth_items else None
    final_z = number(final.get("truth_z")) if final else None
    final_x = number(final.get("truth_x", final.get("x"))) if final else None
    final_y = number(final.get("truth_y", final.get("y"))) if final else None
    vertical_drop = (
        start_z - final_z if start_z is not None and final_z is not None else None
    )
    checks["gazebo_truth_trace"] = bool(
        truth_items and all(item.get("pose_source") == "gazebo_truth" for item in truth_items)
    )
    checks["physical_two_floor_drop"] = bool(
        start_z is not None
        and final_z is not None
        and start_z >= 5.0
        # The learned A1 gait's measured flat standing trunk height is about
        # 0.31 m (the nominal spawn height is not its controlled posture).
        and 0.20 <= final_z <= 1.35
        and vertical_drop >= 4.0
    )
    checks["final_pose_on_first_floor_landing"] = bool(
        final_x is not None
        and final_y is not None
        and -4.76 <= final_x <= -1.74
        and 1.05 <= final_y <= 2.05
    )
    if not checks["gazebo_truth_trace"]:
        failures.append("descent is not evidenced by a Gazebo-truth trace")
    if not checks["physical_two_floor_drop"]:
        failures.append(
            "truth vertical motion is not F3->F1: start_z={!r}, final_z={!r}, drop={!r}".format(
                start_z, final_z, vertical_drop
            )
        )
    if not checks["final_pose_on_first_floor_landing"]:
        failures.append(
            "final Gazebo-truth xy is outside the F1 stair landing: x={!r}, y={!r}".format(
                final_x, final_y
            )
        )

    roll = number(final.get("roll")) if final else None
    pitch = number(final.get("pitch")) if final else None
    upright_z = (
        math.cos(roll) * math.cos(pitch)
        if roll is not None and pitch is not None
        else None
    )
    checks["final_pose_upright"] = bool(upright_z is not None and upright_z >= 0.55)
    if not checks["final_pose_upright"]:
        failures.append("final Gazebo-truth pose is not upright: upright_z={!r}".format(upright_z))

    reported_drop = number(returned.get("descent_total_drop")) if returned else None
    checks["manager_drop_gate"] = bool(reported_drop is not None and reported_drop >= 2.2)
    if returned and not checks["manager_drop_gate"]:
        failures.append("manager final-segment drop is below 2.2 m")

    return {
        "schema": "stair_descent_physical_smoke_acceptance_v1",
        "passed": not failures and all(checks.values()),
        "checks": checks,
        "metrics": {
            "trace_samples": len(trace_items),
            "start_truth_z": start_z,
            "final_truth_z": final_z,
            "final_truth_x": final_x,
            "final_truth_y": final_y,
            "truth_vertical_drop": vertical_drop,
            "final_upright_z": upright_z,
            "manager_final_segment_drop": reported_drop,
        },
        "failure_reasons": failures,
    }


def write_json(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".descent-acceptance-", suffix=".json", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output")
    parser.add_argument(
        "--expected-policy-basename", default="policy_act_inference_stair.pt"
    )
    args = parser.parse_args(argv)
    acceptance = check_results(args.results, args.expected_policy_basename)
    if args.output:
        write_json(args.output, acceptance)
    print(json.dumps(acceptance, indent=2, sort_keys=True))
    return 0 if acceptance["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
