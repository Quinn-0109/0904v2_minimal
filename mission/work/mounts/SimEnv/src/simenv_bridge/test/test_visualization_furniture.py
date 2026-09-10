#!/usr/bin/env python3
import importlib.util
import json
import math
from pathlib import Path
import sys
import types
import unittest


def load_script(name):
    script = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", "_under_test"), script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_three_floor_sequencer():
    sys.modules["rospy"] = types.ModuleType("rospy")
    for package, names in (
            ("geometry_msgs", ("PoseStamped", "Twist")),
            ("nav_msgs", ("Odometry",)),
            ("sensor_msgs", ("PointCloud2",)),
            ("std_msgs", ("Bool", "Float32", "String"))):
        message_module = types.ModuleType(package + ".msg")
        for name in names:
            setattr(message_module, name, type(name, (), {}))
        sys.modules[package] = types.ModuleType(package)
        sys.modules[package + ".msg"] = message_module
    return load_script("scanplanner_three_floor_goal_sequencer.py")


def randomization_settings():
    path = (Path(__file__).resolve().parents[1] / "config" /
            "three_floor_rl_mission.json")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)["scene_randomization"]


class VisualizationFurnitureTest(unittest.TestCase):
    def test_topdown_rotates_object_footprint_from_layout_metadata(self):
        module = load_script("plot_three_floor_topdown.py")
        vertices = module.rotated_footprint_vertices(
            [1.0, 2.0, 0.0, 0.0, 0.0, math.pi / 2.0], [2.0, 1.0, 0.8])
        expected = [(1.5, 1.0), (1.5, 3.0), (0.5, 3.0), (0.5, 1.0)]
        for actual, wanted in zip(vertices, expected):
            self.assertAlmostEqual(actual[0], wanted[0])
            self.assertAlmostEqual(actual[1], wanted[1])

    def test_danger_evaluation_uses_same_oriented_footprint(self):
        module = load_script("plot_red_ball_truth_evaluation.py")
        vertices = module.rotated_footprint_vertices(
            [1.0, 2.0, 0.0, 0.0, 0.0, 0.0], [2.0, 1.0, 0.8])
        self.assertEqual(vertices, [(0.0, 1.5), (2.0, 1.5),
                                    (2.0, 2.5), (0.0, 2.5)])

    def test_open_room_uses_middle_deep_then_near_oblique_pair(self):
        module = load_script("randomize_three_floor_scene.py")
        layout = {"floors": [{
            "floor_index": 0,
            "elevation": 0.0,
            "rooms": [{
                "id": "floor_0_room_0", "side": "left",
                "door_pose": [-1.1, 14.865, 1.2, 0.0, 0.0, 0.0],
                "bounds": {"x_min": -9.5, "x_max": -1.1,
                           "y_min": 7.85, "y_max": 21.88},
                "furniture": [],
            }],
        }]}
        definitions = module.derive_physical_viewpoints(
            layout, {}, randomization_settings())["floor_0_room_0"]
        g3, g4 = definitions
        self.assertEqual(
            g3["viewpoint_policy"], "open_middle_deep_then_near_oblique")
        self.assertEqual(g3["viewpoint_semantic"], "deep")
        self.assertEqual(g4["viewpoint_semantic"], "near")
        self.assertGreater(g3["door_relative_depth_m"],
                           g4["door_relative_depth_m"])
        self.assertGreaterEqual(g3["door_relative_depth_m"], 4.0)
        self.assertGreaterEqual(g4["door_relative_depth_m"], 2.5)
        angle = math.degrees(math.atan2(
            abs(g4["door_relative_lateral_m"] -
                g3["door_relative_lateral_m"]),
            abs(g4["door_relative_depth_m"] -
                g3["door_relative_depth_m"])))
        self.assertGreaterEqual(angle, 20.0)
        self.assertLessEqual(angle, 70.0)
        self.assertGreaterEqual(abs(
            g4["door_relative_lateral_m"] -
            g3["door_relative_lateral_m"]), 0.8)
        self.assertGreaterEqual(
            math.hypot(g4["pose"][0] - g3["pose"][0],
                       g4["pose"][1] - g3["pose"][1]), 1.2)

    def test_large_central_table_blocks_diagonal_room_classification(self):
        module = load_script("randomize_three_floor_scene.py")
        room = {
            "id": "floor_0_room_1", "side": "right",
            "bounds": {"x_min": 1.1, "x_max": 9.5,
                       "y_min": 7.85, "y_max": 21.88},
            "furniture": [{
                "id": "table", "kind": "meeting_table",
                "pose": [4.8, 14.865, 0.45, 0.0, 0.0, 0.0],
                "size": [2.2, 1.0, 0.75],
            }],
        }
        room_type, policy, _geometry, obstacle, _envelopes = (
            module.classify_room_geometry(room, {}))
        self.assertEqual(room_type, "shallow_obstacle")
        self.assertEqual(policy, "front_obstacle_side_bypass_oblique")
        self.assertEqual(obstacle["kind"], "meeting_table")

    def test_small_central_chair_remains_open_but_collision_visible(self):
        module = load_script("randomize_three_floor_scene.py")
        room = {
            "id": "floor_0_room_1", "side": "right",
            "bounds": {"x_min": 1.1, "x_max": 9.5,
                       "y_min": 7.85, "y_max": 21.88},
            "furniture": [{
                "id": "chair", "kind": "chair",
                "pose": [3.1, 14.865, 0.28, 0.0, 0.0, 0.0],
                "size": [0.5, 0.5, 0.55],
            }],
        }
        room_type, policy, _geometry, obstacle, envelopes = (
            module.classify_room_geometry(room, {}))
        self.assertEqual(room_type, "open")
        self.assertEqual(policy, "open_middle_deep_then_near_oblique")
        self.assertEqual(obstacle["id"], "chair")
        self.assertEqual(len(envelopes), 1)

    def test_actual_oblique_geometry_checks_both_room_sides_and_boundaries(self):
        module = load_three_floor_sequencer()
        limits = {
            "g3_depth_minimum": 4.0, "g3_depth_maximum": 5.5,
            "g4_depth_minimum": 2.5, "g4_depth_maximum": 4.0,
            "minimum_depth_delta": 1.0, "minimum_lateral_delta": 0.8,
            "angle_minimum": 20.0, "angle_maximum": 70.0,
            "central_half_width_m": 4.2,
        }
        for direction, door_x in ((1.0, 1.1), (-1.0, -1.1)):
            door = {"inward_direction": direction,
                    "centre": [door_x, 14.865, 0.0]}
            deep = [door_x + direction * 5.0, 16.365, 0.0]
            near = [door_x + direction * 3.0, 14.865, 0.0]
            result = module.oblique_actual_geometry(
                deep, near, door, limits=limits)
            self.assertTrue(result["passed"])
            too_shallow = [door_x + direction * 3.9, 16.365, 0.0]
            self.assertFalse(module.oblique_actual_geometry(
                too_shallow, near, door, limits=limits)["passed"])

        door = {"inward_direction": 1.0, "centre": [1.1, 0.0, 0.0]}
        near = [4.1, 0.0, 0.0]
        for angle, depth_delta in ((20.0, 2.20), (70.0, 1.00)):
            deep = [4.1 + depth_delta,
                    depth_delta * math.tan(math.radians(angle)), 0.0]
            self.assertTrue(module.oblique_actual_geometry(
                deep, near, door, limits=limits)["passed"])

    def test_deep_obstacle_pair_stays_before_obstacle(self):
        module = load_script("randomize_three_floor_scene.py")
        room = self._obstacle_room("deep", 6.0)
        layout = {"floors": [{"floor_index": 0, "elevation": 0.0,
                              "rooms": [room]}]}
        points = module.derive_physical_viewpoints(
            layout, {}, randomization_settings())["deep"]
        obstacle = room["room_contract"]["primary_obstacle"]
        limit = obstacle["front_depth_m"] - 0.72
        self.assertEqual(room["room_contract"]["viewpoint_policy"],
                         "deep_obstacle_front_oblique_pair")
        self.assertTrue(all(point["door_relative_depth_m"] <= limit + 1e-9
                            for point in points))

    def test_front_obstacle_pair_bypasses_to_back_and_opposite_side(self):
        module = load_script("randomize_three_floor_scene.py")
        room = self._obstacle_room("front", 4.8)
        layout = {"floors": [{"floor_index": 0, "elevation": 0.0,
                              "rooms": [room]}]}
        g3, g4 = module.derive_physical_viewpoints(
            layout, {}, randomization_settings())["front"]
        obstacle = room["room_contract"]["primary_obstacle"]
        self.assertEqual(room["room_contract"]["viewpoint_policy"],
                         "front_obstacle_side_bypass_oblique")
        self.assertGreaterEqual(g3["door_relative_depth_m"],
                                obstacle["back_depth_m"] + 0.72)
        self.assertLess(
            (g3["door_relative_lateral_m"] - obstacle["lateral_m"]) *
            (g4["door_relative_lateral_m"] - obstacle["lateral_m"]), 0.0)

    def test_open_pair_uses_bounded_same_type_fallback(self):
        module = load_script("randomize_three_floor_scene.py")
        room = self._obstacle_room("fallback", None)
        room["furniture"] = []
        layout = {"floors": [{"floor_index": 0, "elevation": 0.0,
                              "rooms": [room]}]}
        settings = randomization_settings()
        settings["open_view_central_width_fraction"] = 0.04
        settings["room_view_fallback_central_width_fraction"] = 0.70
        module.derive_physical_viewpoints(layout, {}, settings)
        self.assertEqual(room["room_contract"]["viewpoint_fallback_level"],
                         "bounded_relaxation")
        self.assertEqual(room["room_contract"]["viewpoint_fallback_reason"],
                         "primary_oblique_pair_infeasible")

    @staticmethod
    def _obstacle_room(room_id, table_x):
        furniture = [] if table_x is None else [{
            "id": "table", "kind": "meeting_table",
            "pose": [table_x, 14.865, 0.45, 0.0, 0.0, 0.0],
            "size": [2.2, 1.0, 0.75],
        }]
        return {
            "id": room_id, "side": "right",
            "bounds": {"x_min": 1.1, "x_max": 9.5,
                       "y_min": 7.85, "y_max": 21.88},
            "furniture": furniture,
        }

    def test_audit_refresh_and_danger_rescan_are_bounded_independently(self):
        module = load_script("check_three_floor_rl_mission.py")
        room = {
            "extra_scan_count": 2,
            "replan_count": 0,
            "recovery_count": 0,
            "runtime_3d_audits": [
                {"waypoint": "room_g3", "passed": False},
                {"waypoint": "room_g3", "passed": True},
            ],
        }
        self.assertTrue(module.bounded_room_recovery(room))
        room["extra_scan_count"] = 3
        self.assertFalse(module.bounded_room_recovery(room))


if __name__ == "__main__":
    unittest.main()
