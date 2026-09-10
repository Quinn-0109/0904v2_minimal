#!/usr/bin/env python3
import importlib.util
import json
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_randomizer():
    path = ROOT / "scripts" / "randomize_three_floor_scene.py"
    spec = importlib.util.spec_from_file_location(
        "stable20_randomizer_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Stable20ViewContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_randomizer()
        cls.config = json.loads((
            ROOT / "config" / "three_floor_rl_mission.json").read_text())
        cls.settings = cls.config["scene_randomization"]

    @staticmethod
    def scan(scan_id, role, x, y, target_x, target_y):
        bearing = math.atan2(target_y - y, target_x - x)
        return {
            "id": scan_id,
            "role": role,
            "pose": [x, y, 0.0, bearing - 1.5],
            "scan_yaw_rate": 0.86,
            "scan_turn_rad": math.radians(215.0),
            "detection_heading_margin_rad": 0.35,
        }

    def test_config_keeps_primary_and_fallback_baselines_strict(self):
        settings = self.settings
        self.assertGreaterEqual(
            settings["physical_viewpoint_minimum_separation_m"], 2.45)
        self.assertGreaterEqual(
            settings["physical_viewpoint_fallback_minimum_separation_m"], 2.30)
        self.assertGreaterEqual(
            settings["physical_viewpoint_actual_minimum_separation_m"], 2.20)
        self.assertGreaterEqual(
            settings["physical_viewpoint_fallback_actual_minimum_separation_m"],
            2.10)
        self.assertEqual(settings["minimum_dual_view_parallax_deg"], 30.0)
        self.assertEqual(settings["physical_viewpoint_navigation_speed"], 0.90)
        self.assertLessEqual(settings["physical_corridor_cruise_speed"], 2.25)
        self.assertLessEqual(
            settings["physical_corridor_maximum_lateral_speed_mps"], 0.45)
        self.assertEqual(settings["physical_plane_distance_gain"], 1.45)
        self.assertLessEqual(
            settings["physical_stair_side_opening_speed_mps"], 1.85)
        self.assertLessEqual(
            settings["physical_stair_side_opening_maximum_lateral_speed_mps"],
            0.35)
        self.assertLessEqual(
            settings["physical_stair_handoff_maximum_yaw_rate_rad_s"], 1.00)
        self.assertGreaterEqual(
            settings["physical_stair_side_opening_settle_sec"], 0.20)
        self.assertLessEqual(
            settings["physical_viewpoint_maximum_lateral_speed_mps"], 0.55)
        self.assertLessEqual(
            settings["physical_viewpoint_maximum_yaw_rate_rad_s"], 1.50)

    def test_stair_side_opening_stops_and_handoff_turns_before_translation(self):
        mission = {"floors": [{"waypoints": [
            {
                "note": "floor_0_stair_side_opening",
                "controller": "direct_plane_rl",
                "speed": 0.5,
                "yaw": math.pi,
            },
            {
                "note": "floor_0_stair_handoff",
                "controller": "direct_plane_rl",
                "speed": 0.55,
                "yaw": math.pi / 2.0,
            },
        ]}]}
        self.mod.rewrite_physical_room_route(
            mission, {"floors": []}, {}, self.settings)
        side, handoff = mission["floors"][0]["waypoints"]
        self.assertEqual(side["speed"], 1.85)
        self.assertTrue(side["speed_scale_exempt"])
        self.assertFalse(side["pass_through"])
        self.assertTrue(side["align_yaw"])
        self.assertEqual(side["direct_plane_max_lateral_speed_mps"], 0.35)
        self.assertEqual(side["alignment_yaw_rate"], 1.0)
        self.assertEqual(side["direct_plane_distance_gain"], 1.45)
        self.assertGreaterEqual(side["post_reach_settle_sec"], 0.20)
        self.assertTrue(handoff["plane_forward_only"])
        self.assertEqual(handoff["alignment_yaw_rate"], 1.0)
        self.assertEqual(handoff["direct_plane_distance_gain"], 1.45)

    def test_target_requires_two_uncertainty_safe_views_and_parallax(self):
        target = (0.0, 2.0)
        scans = [
            self.scan("room_scan_a", "G3", -1.25, 0.0, *target),
            self.scan("room_scan_b", "G4", 1.25, 0.0, *target),
        ]
        room = {"furniture": []}
        evidence = [self.mod.robust_detection_view(
            *target, scan, room, self.settings) for scan in scans]
        self.assertTrue(all(item and item["uncertainty_safe"]
                            for item in evidence))
        self.assertGreaterEqual(
            self.mod.dual_view_parallax_deg(*target, *scans), 30.0)
        self.assertTrue(all(item["projected_radius_px"] >= 10.0
                            for item in evidence))
        self.assertTrue(all(item["vertical_projection_slack_rad"] >= 0.0
                            for item in evidence))

    def test_arrival_envelope_and_small_furniture_occlusion_are_not_relaxed(self):
        # The 1.15 m reliable minimum is evaluated after the +/-0.12 m
        # arrival envelope, so a 1.20 m centre range is correctly rejected.
        target = (0.0, 1.20)
        scan = self.scan("room_scan_a", "G3", 0.0, 0.0, *target)
        self.assertIsNone(self.mod.robust_detection_view(
            *target, scan, {"furniture": []}, self.settings))

        # A point technically inside the 3.80 m hard limit but with only a
        # centimetre-scale arrival margin must also be rejected.
        marginal_target = (0.0, 3.65)
        marginal_scan = self.scan(
            "room_scan_margin", "G3", 0.0, 0.0, *marginal_target)
        self.assertIsNone(self.mod.robust_detection_view(
            *marginal_target, marginal_scan,
            {"furniture": []}, self.settings))

        target = (0.0, 2.0)
        scan = self.scan("room_scan_a", "G3", -1.25, 0.0, *target)
        room = {"furniture": [{
            "id": "small_coffee_table",
            "pose": [-0.62, 1.0, 0.0, 0.0, 0.0, 0.0],
            "size": [0.45, 0.45, 0.45],
        }]}
        self.assertIsNone(self.mod.robust_detection_view(
            *target, scan, room, self.settings))

    def test_red_distractor_projected_silhouettes_must_not_merge(self):
        target = (-6.369036575, 13.179340285)
        scan = self.scan("floor_2_room_0_g3", "G3", -5.605625,
                         15.543166667, *target)
        distractor = [{
            "id": "distractor_red_box_01_floor_2",
            "pose": [-6.841, 12.773, 5.35, 0.0, 0.0, 0.0],
            "size": [0.3, 0.3, 0.3],
        }]
        self.assertIsNone(self.mod.red_distractor_projection_evidence(
            *target, 5.35, scan, distractor, self.settings))

        distractor[0]["pose"][0] = -8.8
        evidence = self.mod.red_distractor_projection_evidence(
            *target, 5.35, scan, distractor, self.settings)
        self.assertTrue(evidence["red_distractor_projection_safe"])
        self.assertGreaterEqual(
            evidence["minimum_red_distractor_edge_gap_rad"],
            evidence["required_red_distractor_edge_gap_rad"])


if __name__ == "__main__":
    unittest.main()
