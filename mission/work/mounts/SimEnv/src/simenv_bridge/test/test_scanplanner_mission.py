#!/usr/bin/env python3
import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest


def _install_ros_stubs():
    sys.modules["rospy"] = types.ModuleType("rospy")

    geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")
    geometry_msgs_msg.PoseStamped = type("PoseStamped", (), {})
    geometry_msgs_msg.Twist = type("Twist", (), {})
    nav_msgs_msg = types.ModuleType("nav_msgs.msg")
    nav_msgs_msg.Odometry = type("Odometry", (), {})
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.Bool = type("Bool", (), {})
    std_msgs_msg.String = type("String", (), {})

    for pkg, msg in (
        ("geometry_msgs", geometry_msgs_msg),
        ("nav_msgs", nav_msgs_msg),
        ("std_msgs", std_msgs_msg),
    ):
        sys.modules[pkg] = types.ModuleType(pkg)
        sys.modules[pkg + ".msg"] = msg


def _load_module():
    _install_ros_stubs()
    script = Path(__file__).resolve().parents[1] / "scripts" / "scanplanner_goal_sequencer.py"
    spec = importlib.util.spec_from_file_location("scanplanner_goal_sequencer_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScanplannerMissionTest(unittest.TestCase):
    @staticmethod
    def _layout():
        rooms = []
        for index, (side, door_y) in enumerate(
            (("left", 14.865), ("right", 14.865), ("left", 28.895), ("right", 28.895))
        ):
            sign = -1.0 if side == "left" else 1.0
            rooms.append({
                "id": "floor_0_room_%d" % index,
                "side": side,
                "door_pose": [sign * 1.1, door_y],
                "goal_pose": [sign * 4.8, door_y],
                "bounds": {
                    "x_min": -9.5 if side == "left" else 1.1,
                    "x_max": -1.1 if side == "left" else 9.5,
                    "y_min": door_y - 7.015,
                    "y_max": door_y + 7.015,
                },
            })
        return {"floors": [{"floor_index": 0, "rooms": rooms}]}

    def test_route_covers_every_layout_room_without_danger_positions(self):
        mod = _load_module()
        route = mod.build_route(self._layout())
        labels = [action["label"] for action in route]
        for room_index in range(4):
            prefix = "floor_0_room_%d" % room_index
            self.assertIn(prefix + "_entry", labels)
            self.assertIn(prefix + "_scan_center", labels)
            self.assertIn(prefix + "_scan_low", labels)
            self.assertIn(prefix + "_scan_high", labels)
            self.assertIn(prefix + "_return_corridor", labels)
        self.assertFalse(any("danger" in label or "truth" in label for label in labels))
        scans = [a for a in route if a["kind"] == "scan"]
        self.assertEqual(len(scans), 12)
        for scan in scans:
            self.assertAlmostEqual(scan["turn"], 2.0 * math.pi)

        translations = [a for a in route if a["kind"] == "goal"]
        path_length = 0.0
        for previous, current in zip(translations, translations[1:]):
            distance = math.hypot(current["x"] - previous["x"], current["y"] - previous["y"])
            path_length += distance
            self.assertLessEqual(distance, 15.0 + 1e-9, (previous["label"], current["label"]))
            for wall_x in (-1.1, 1.1):
                dx = current["x"] - previous["x"]
                if abs(dx) < 1e-9:
                    continue
                u = (wall_x - previous["x"]) / dx
                if 0.0 < u < 1.0:
                    cross_y = previous["y"] + u * (current["y"] - previous["y"])
                    door_y = min((14.865, 28.895), key=lambda y: abs(y - cross_y))
                    self.assertLessEqual(abs(cross_y - door_y), 0.7)
        self.assertLessEqual(path_length, 150.0)

    def test_accumulated_rotation_handles_pi_wrap(self):
        mod = _load_module()
        delta = mod.accumulated_rotation(math.radians(179), math.radians(-179))
        self.assertAlmostEqual(delta, math.radians(2), places=6)

    def test_room_goal_tolerance_is_larger_than_corridor_tolerance(self):
        mod = _load_module()
        self.assertAlmostEqual(mod.goal_tolerance({"label": "room_LB_center"}, 1.0), 1.5)
        self.assertAlmostEqual(mod.goal_tolerance({"label": "corridor_back"}, 1.0), 1.0)

    def test_launch_and_runner_contract(self):
        root = Path(__file__).resolve().parents[1]
        launch = (root / "launch" / "scanplanner_simenv.launch").read_text()
        runner = (root / "scripts" / "run_scanplanner_simenv.sh").read_text()
        self.assertIn('name="danger_detector"', launch)
        self.assertIn('name="camera_video_recorder"', launch)
        self.assertIn("scanplanner_camera_fixed.mp4", launch)
        self.assertIn("layout_metadata.json", launch)
        self.assertIn('max_acc" value="2.0"', launch)
        self.assertIn("MAX_DURATION=200", runner)
        self.assertIn("/real_sense/rgb/image_raw", runner)
        self.assertIn("/livox/Pointcloud2", runner)
        self.assertIn("/Odometry_gazebo", runner)
        self.assertLess(runner.index("source /opt/ros/noetic/setup.bash"), runner.index("set -u"))
        self.assertIn("/gazebo/get_world_properties", runner)
        self.assertIn("generated_building", runner)
        self.assertIn("danger_red_sphere", runner)
        self.assertIn("applied_command_speed", runner)


if __name__ == "__main__":
    unittest.main()
