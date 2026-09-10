#!/usr/bin/env python3
import re
from pathlib import Path
import unittest


def _tare_package_root():
    for parent in Path(__file__).resolve().parents:
        for rel in ("tare_planner/src/tare_planner", "src/tare_planner"):
            candidate = parent / rel
            if (candidate / "config" / "simenv.yaml").exists():
                return candidate
    raise AssertionError("could not locate tare_planner package root")


def _read_scalar(text, key):
    match = re.search(r"^{}\s*:\s*([^#\n]+)".format(re.escape(key)), text, re.MULTILINE)
    if not match:
        raise AssertionError("missing {}".format(key))
    value = match.group(1).strip()
    try:
        return int(value)
    except ValueError:
        return float(value)


class TareSimenvConfigTest(unittest.TestCase):
    def test_grid_world_neighbor_window_updates_adjacent_xy_cells(self):
        config = _tare_package_root() / "config" / "simenv.yaml"
        text = config.read_text()

        cell_size = _read_scalar(text, "kGridWorldCellSize")
        nearby_grid_num = _read_scalar(text, "kGridWorldNearbyGridNum")

        self.assertLessEqual(cell_size, 4.8)
        self.assertGreaterEqual(
            nearby_grid_num,
            5,
            "GridWorld uses KNearbyGridNum / 2 as the XY radius. A value of 1 "
            "updates only the robot's current XY cell, so SimEnv four-room "
            "exploration degenerates into local coverage and never promotes "
            "adjacent corridor/room cells.",
        )

    def test_grid_world_cell_size_is_configurable_for_wide_lobby(self):
        source = (
            _tare_package_root() / "src" / "grid_world" / "grid_world.cpp"
        ).read_text()

        self.assertIn("kGridWorldCellSize", source)

    def test_sensor_range_reaches_next_room_door_in_four_room_layout(self):
        config = _tare_package_root() / "config" / "simenv.yaml"
        text = config.read_text()

        self.assertGreaterEqual(_read_scalar(text, "kSensorRange"), 12.0)


if __name__ == "__main__":
    unittest.main()
