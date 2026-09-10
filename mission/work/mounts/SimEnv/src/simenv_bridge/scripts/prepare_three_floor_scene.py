#!/usr/bin/env python3
"""Create validated, run-local copies of the three-floor Gazebo assets.

The archived ``simenv-exploration:threefloor`` image contains a malformed
three-floor scene: its full-footprint ``roof`` collision is at the third-floor
slab height and therefore seals the stair opening.  Never mutate the image's
shared source in place.  This helper derives the correct roof height from the
layout metadata, patches run-local XML copies, and records exactly what was
changed.
"""

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import xml.etree.ElementTree as ET


FLIGHT_RE = re.compile(r"^stair_flight_([ab])_floor_(\d+)_step_(\d+)$")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pose_values(link, path):
    pose = link.find("pose")
    if pose is None or not pose.text:
        raise ValueError("{}: link {!r} has no pose".format(path, link.get("name")))
    values = pose.text.split()
    if len(values) != 6:
        raise ValueError("{}: link {!r} pose must have six values".format(
            path, link.get("name")))
    try:
        return pose, [float(value) for value in values]
    except ValueError as exc:
        raise ValueError("{}: link {!r} has a nonnumeric pose".format(
            path, link.get("name"))) from exc


def _roof_thickness(roof, path):
    sizes = []
    for collision in roof.findall("collision"):
        size = collision.findtext("geometry/box/size")
        if size:
            values = size.split()
            if len(values) == 3:
                sizes.append(float(values[2]))
    if len(sizes) != 1 or sizes[0] <= 0.0:
        raise ValueError("{}: roof must contain one positive box collision".format(path))
    return sizes[0]


def _validate_stairs(links, floor_count, path):
    actual = set()
    for link in links:
        match = FLIGHT_RE.match(link.get("name", ""))
        if match:
            actual.add((match.group(1), int(match.group(2)), int(match.group(3))))
    expected = {
        (flight, floor, step)
        for flight in ("a", "b")
        for floor in range(floor_count - 1)
        for step in range(10)
    }
    missing = sorted(expected - actual)
    if missing:
        raise ValueError("{}: missing physical stair links: {}".format(
            path, ", ".join("flight_{}_floor_{}_step_{}".format(*item)
                            for item in missing)))
    landing_names = {link.get("name", "") for link in links}
    for floor in range(floor_count):
        name = "stair_floor_landing_floor_{}".format(floor)
        if name not in landing_names:
            raise ValueError("{}: missing {}".format(path, name))


def _atomic_write_xml(tree, destination):
    directory = os.path.dirname(destination)
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".three-floor-scene-", suffix=".xml", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            tree.write(stream, encoding="utf-8", xml_declaration=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def prepare_asset(source, destination, floor_count, expected_roof_z):
    tree = ET.parse(source)
    links = list(tree.getroot().iter("link"))
    roofs = [link for link in links if link.get("name") == "roof"]
    if len(roofs) != 1:
        raise ValueError("{}: expected exactly one roof link, found {}".format(
            source, len(roofs)))
    _validate_stairs(links, floor_count, source)
    pose, values = _pose_values(roofs[0], source)
    before_z = values[2]
    values[2] = expected_roof_z
    pose.text = "{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(*values)
    _atomic_write_xml(tree, destination)

    verification = ET.parse(destination)
    copied_roofs = [link for link in verification.getroot().iter("link")
                    if link.get("name") == "roof"]
    _, copied_pose = _pose_values(copied_roofs[0], destination)
    if not math.isclose(copied_pose[2], expected_roof_z, abs_tol=1e-6):
        raise ValueError("{}: copied roof height verification failed".format(destination))
    return {
        "source": source,
        "source_sha256": _sha256(source),
        "output": destination,
        "output_sha256": _sha256(destination),
        "roof_z_before": before_z,
        "roof_z_after": copied_pose[2],
        "roof_was_corrected": not math.isclose(
            before_z, expected_roof_z, abs_tol=1e-6),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layout", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    with open(args.layout, "r", encoding="utf-8") as stream:
        layout = json.load(stream)
    floors = layout.get("floors")
    if not isinstance(floors, list) or len(floors) != 3:
        raise ValueError("three-floor scene requires exactly three metadata floors")
    elevations = [float(floor["elevation"]) for floor in floors]
    wall_height = float(layout["wall_height"])

    model_tree = ET.parse(args.model)
    model_roofs = [link for link in model_tree.getroot().iter("link")
                   if link.get("name") == "roof"]
    if len(model_roofs) != 1:
        raise ValueError("{}: expected exactly one roof link".format(args.model))
    roof_thickness = _roof_thickness(model_roofs[0], args.model)
    expected_roof_z = max(elevations) + wall_height + roof_thickness / 2.0

    os.makedirs(args.output_dir, exist_ok=True)
    world_output = os.path.join(
        args.output_dir, "competition_scene_with_dangers.world")
    model_output = os.path.join(args.output_dir, "model.sdf")
    assets = {
        "world": prepare_asset(
            args.world, world_output, len(floors), expected_roof_z),
        "model": prepare_asset(
            args.model, model_output, len(floors), expected_roof_z),
    }
    manifest = {
        "schema": "simenv_three_floor_scene_preparation_v1",
        "passed": True,
        "floor_count": len(floors),
        "floor_elevations": elevations,
        "wall_height": wall_height,
        "roof_thickness": roof_thickness,
        "expected_roof_z": expected_roof_z,
        "assets": assets,
    }
    manifest_path = os.path.join(args.output_dir, "scene_preparation.json")
    descriptor, temporary = tempfile.mkstemp(
        prefix=".scene-preparation-", suffix=".json", dir=args.output_dir)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, manifest_path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
