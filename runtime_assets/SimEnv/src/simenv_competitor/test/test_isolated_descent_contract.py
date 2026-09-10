#!/usr/bin/env python3
"""Static and ROS-light checks for the isolated F3->F1 acceptance path."""

import inspect
import json
import os
import sys
import unittest
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from unittest import mock


SCRIPT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "scripts"))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from isolated_descent_trigger import pose_is_ready
from stair_descent_manager import StairDescent


class IsolatedDescentContractTest(unittest.TestCase):
    def test_official_three_floor_roof_does_not_seal_f3_stairwell(self):
        workspace = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", ".."))
        scene = os.path.join(
            workspace, "generated_building",
            "elevator_three_floor_official_count", "model.sdf")
        metadata = os.path.join(
            workspace, "generated_building",
            "elevator_three_floor_official_count", "layout_metadata.json")
        root = ET.parse(scene).getroot()
        roof = root.find(".//link[@name='roof']")
        self.assertIsNotNone(roof)
        roof_z = float(roof.find("pose").text.split()[2])
        roof_thickness = float(
            roof.find("collision/geometry/box/size").text.split()[2])
        with open(metadata, encoding="utf-8") as stream:
            layout = json.load(stream)
        highest_floor = max(float(floor["elevation"])
                            for floor in layout["floors"])
        minimum_roof_center = (highest_floor + float(layout["wall_height"])
                               + roof_thickness / 2.0)
        self.assertAlmostEqual(roof_z, minimum_roof_center, places=3)
        self.assertGreater(roof_z - roof_thickness / 2.0,
                           highest_floor + 2.0)

    def test_spawn_gate_requires_pose_tilt_and_locomotion(self):
        bounds = (-2.05, -1.83, 1.10, 1.34, 5.50, 5.68)
        self.assertTrue(pose_is_ready(
            (-1.94, 1.22, 5.58, 2.95), (0.02, -0.03), True,
            bounds, 0.25))
        self.assertFalse(pose_is_ready(
            (-1.94, 1.22, 5.58, 2.95), (0.02, -0.03), False,
            bounds, 0.25))
        self.assertFalse(pose_is_ready(
            (-1.94, 1.22, 5.75, 2.95), (0.02, -0.03), True,
            bounds, 0.25))
        self.assertFalse(pose_is_ready(
            (-1.94, 1.22, 5.58, 2.95), (0.30, 0.0), True,
            bounds, 0.25))

    def test_top_lip_gate_uses_height_and_planar_progress(self):
        source = inspect.getsource(StairDescent._flight_tick)
        gate = source[source.index("'F3_TOP_LIP_CROSSED'"):
                      source.index("# Advancing nearly four treads")]
        self.assertIn("top_lip_minimum_drop", gate)
        self.assertIn("top_lip_minimum_advance", gate)
        self.assertIn("_upright_for_milestone", gate)

    def test_floor_landing_gate_uses_truth_height_region_and_tilt(self):
        source = inspect.getsource(StairDescent._target_floor_landing_envelope)
        self.assertIn("second_floor_landing_maximum_z", source)
        self.assertIn("first_floor_landing_maximum_z", source)
        self.assertIn("_flight_a_center_x", source)
        self.assertIn("landing_maximum_tilt", source)

    def test_success_file_is_only_written_after_home_gate(self):
        source = inspect.getsource(StairDescent._finish_home_return)
        self.assertIn("first_floor_start_returned.json", source)
        home_tick = inspect.getsource(StairDescent._home_return_tick)
        self.assertIn("home_position_tolerance", home_tick)
        self.assertIn("home_heading_tolerance", home_tick)
        self.assertIn("home_arrival_stable_seconds", home_tick)

    def test_home_route_leaves_stairwell_through_east_opening(self):
        source = inspect.getsource(StairDescent.__init__)
        self.assertIn("first_floor_stair_clear_x", source)
        self.assertIn("first_floor_stair_clear_y", source)
        begin = inspect.getsource(StairDescent._begin_home_return)
        self.assertNotIn("self.home_clear_x=(self.truth_pose", begin)

    def test_home_return_fall_gets_one_bounded_physical_recovery_wait(self):
        source = inspect.getsource(StairDescent._home_return_tick)
        self.assertIn("_home_pose_upright", source)
        self.assertIn("_begin_home_fall_recovery_wait", source)
        recovery = inspect.getsource(
            StairDescent._home_fall_recovery_tick)
        self.assertIn("home_fall_recovery_timeout", recovery)
        self.assertIn("self.locomotion_ready", recovery)
        self.assertIn("_begin_home_return", recovery)
        bounded = inspect.getsource(
            StairDescent._begin_home_fall_recovery_wait)
        self.assertIn("home_fall_recovery_max_attempts", bounded)
        self.assertIn("FIRST_FLOOR_HOME_RETURN_FALL_DETECTED", bounded)
        self.assertNotIn("set_model_state", bounded)

    def test_acceptance_defaults_prohibit_post_trigger_teleport(self):
        launch = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "launch",
            "stair_third_to_first_isolated.launch"))
        with open(launch, encoding="utf-8") as stream:
            text = stream.read()
        self.assertIn(
            'name="truth_descent_no_drop_correction_enabled" value="false"',
            text)
        self.assertIn(
            'name="truth_descent_flight_max_stall_recoveries" value="2"',
            text)
        self.assertIn(
            'name="enable_compliant_fall_recovery" value="false"', text)
        self.assertIn('name="enable_fastlio_stack" value="false"', text)
        self.assertIn('name="enable_voxel_mapper" value="false"', text)
        self.assertIn(
            'name="enable_local_doorway_detector" value="false"', text)
        self.assertIn(
            'name="localization_truth_recovery_enabled" value="false"',
            text)

    def test_fullflow_exports_the_accepted_descent_profile(self):
        launch_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "launch"))
        for filename in ("baseline_fastlio_exploration.launch",
                         "fuel_semantic_fastlio_exploration.launch"):
            with open(os.path.join(launch_dir, filename),
                      encoding="utf-8") as stream:
                text = stream.read()
            for expected in (
                    'name="descent_speed_mps" default="0.52"',
                    'name="descent_maximum_speed_mps" default="0.52"',
                    'name="descent_segment_timeout_sec" default="120.0"',
                    'name="descent_pre_stand_seconds" default="1.5"',
                    'name="truth_descent_landing_heading_tolerance_rad" default="0.28"',
                    'name="truth_descent_landing_settle_seconds" default="1.5"',
                    'name="truth_descent_landing_yaw_stall_sec" default="2.5"',
                    'name="truth_descent_step_pause_seconds" default="0.10"',
                    'name="first_floor_home_return_wall_timeout_sec" default="180.0"'):
                self.assertIn(expected, text, filename)

    def test_fullflow_goal_executor_yields_to_descent_state(self):
        source = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "scripts", "goal_executor.py"))
        with open(source, encoding="utf-8") as stream:
            text = stream.read()
        self.assertIn(
            '"/simenv/third_to_first_floor_stair_state"', text)
        self.assertIn(
            'state.strip().upper().startswith("STAIR_DESCENT_")', text)
        self.assertIn(
            '"event": "f3_truth_context_released_for_descent"', text)

    def test_fullflow_descent_terminal_is_required_to_end_roslaunch(self):
        launch = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "launch",
            "baseline_fastlio_exploration.launch"))
        root = ET.parse(launch).getroot()
        node = next(item for item in root.findall("node")
                    if item.get("name") == "stair_descent_manager")
        self.assertEqual(node.get("required"), "true")

    def test_waiting_descent_ignores_global_ascent_policy_ack(self):
        descent = StairDescent.__new__(StairDescent)
        descent.phase = "WAIT_F3"
        descent.active_policy = "/tmp/policy_stair.pt"
        descent.policy_loaded = False
        descent.state = mock.Mock()

        descent.on_policy_status(SimpleNamespace(
            data="policy_reloaded:/tmp/policy_stair.pt"))

        self.assertFalse(descent.policy_loaded)
        descent.state.publish.assert_not_called()

    def test_second_segment_east_align_overcomes_policy_dead_zone(self):
        source = inspect.getsource(StairDescent._east_align_tick)
        self.assertIn("east_align_minimum_speed", source)
        self.assertIn("segment_turn_position_tolerance", source)

    def test_segment_turn_exhaustion_uses_physical_fixedstand_recovery(self):
        begin = inspect.getsource(
            StairDescent._begin_segment_turn_fixed_stand_recovery)
        tick = inspect.getsource(StairDescent._segment_turn_tick)
        recovery = inspect.getsource(
            StairDescent._segment_turn_fixed_stand_tick)
        self.assertIn("yaw_stall_replants_exhausted", tick)
        self.assertIn("turn_timeout", tick)
        self.assertIn("stand.buttons[1]=1", recovery)
        self.assertIn("STAIR_DESCENT_SEGMENT_POLICY_LOADING", recovery)
        self.assertIn("self.hold_rl()", recovery)
        loading = inspect.getsource(
            StairDescent._segment_policy_loading_tick)
        self.assertIn("segment_policy_loading_timeout", loading)
        self.assertIn("segment_policy_loading_max_requests", loading)
        self.assertIn("F1_LANDING_REACHED_DEGRADED_POLICY_ACK_TIMEOUT",
                      loading)
        self.assertNotIn("ServiceProxy", begin)
        self.assertNotIn("ModelState()", begin)

    def test_segment_turn_policy_reload_ack_is_consumed(self):
        descent = StairDescent.__new__(StairDescent)
        descent.phase = "STAIR_DESCENT_SEGMENT_POLICY_LOADING"
        descent.active_policy = "/tmp/policy_stair.pt"
        descent.policy_loaded = False
        descent.state = mock.Mock()

        descent.on_policy_status(SimpleNamespace(
            data="policy_reloaded:/tmp/policy_stair.pt"))

        self.assertTrue(descent.policy_loaded)
        descent.state.publish.assert_called_once()

    def test_landing_recenter_cannot_command_an_extra_full_turn(self):
        source = inspect.getsource(StairDescent._landing_turn_tick)
        self.assertIn("direction_release_error", source)
        self.assertIn("turn_error=(error if abs(error)", source)

    def test_full_drop_edge_drift_is_captured_without_model_reset(self):
        source = inspect.getsource(StairDescent._flight_tick)
        start = source.index("edge_drift_capture_gate = bool(")
        end = source.index("self.landing_started_at", start)
        gate = source[start:end]
        self.assertIn("drop >= self.flight_b_descent_drop", gate)
        self.assertIn("self.edge_drift_landing_capture_margin", gate)
        self.assertIn("abs(center_error) > self.max_center_error", gate)
        self.assertNotIn("set_model_state", gate)
        self.assertNotIn("_correct_no_drop_stair_seam", gate)

    def test_visualizer_emits_two_floor_to_floor_stage_images(self):
        visualizer = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "scripts",
            "visualize_roundtrip_return.py"))
        with open(visualizer, encoding="utf-8") as stream:
            text = stream.read()
        self.assertIn("31_third_floor_to_second_floor_truth_descent", text)
        self.assertIn("32_second_floor_to_first_floor_truth_descent", text)

    def test_performance_baseline_ignores_incomplete_runs(self):
        visualizer = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "scripts",
            "visualize_roundtrip_return.py"))
        with open(visualizer, encoding="utf-8") as stream:
            text = stream.read()
        baseline = text[text.index("def previous_isolated_total"):
                        text.index("def write_subflow_summary")]
        self.assertIn('not payload.get("complete")', baseline)
        self.assertIn('FIRST_FLOOR_START_RETURNED', baseline)


if __name__ == "__main__":
    unittest.main()
