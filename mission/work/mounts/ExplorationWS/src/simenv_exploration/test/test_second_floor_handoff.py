#!/usr/bin/env python3
"""ROS-light regression tests for the isolated F2 truth handoff policy."""

import os
import sys
import threading
import unittest
import json
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from unittest import mock

import numpy as np


SCRIPT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "scripts"))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from second_floor_exploration_manager import SecondFloorMission
import baseline_exploration_manager as baseline_manager_module
from baseline_exploration_manager import BaselineExplorationManager
from lightweight_room_core import LightweightRoomScheduler
from goal_executor import GoalExecutor
from stair_transition_manager import StairTransition


class SecondFloorHandoffTest(unittest.TestCase):
    def test_truth_long_return_route_stays_on_corridor_before_lobby_turn(self):
        transition = StairTransition.__new__(StairTransition)
        transition.truth_pose = (-0.56, 29.0, 0.32, -np.pi / 2.0)
        transition.truth_side_entry_offset = 1.65
        transition.truth_corridor_lateral_offset = 1.80
        transition.truth_corridor_longitudinal_offset = 3.80
        corridor, side = transition._truth_entry_route_targets(
            (-4.015, 1.75), np.pi / 2.0)
        self.assertAlmostEqual(corridor[0], -0.565, places=6)
        self.assertAlmostEqual(corridor[1], 5.55, places=6)
        self.assertAlmostEqual(side[0], -2.365, places=6)
        self.assertAlmostEqual(side[1], 1.75, places=6)

    def test_run101_upper_floor_route_stays_east_of_stair_void(self):
        transition = StairTransition.__new__(StairTransition)
        transition.source_floor_index = 1
        transition.offline_stair_model = os.path.abspath(os.path.join(
            SCRIPT_DIR, "..", "..", "..", "generated_building",
            "elevator_three_floor_debug", "model.sdf"))
        transition.truth_upper_floor_exit_clearance = .80
        transition.truth_pose = (0.357, 12.454, 2.913, -1.109)
        transition.truth_side_entry_offset = 1.65
        transition.truth_corridor_lateral_offset = 1.80
        transition.truth_corridor_longitudinal_offset = 3.80
        corridor, side = transition._truth_entry_route_targets(
            (-4.015, 1.75), np.pi / 2.0)
        self.assertAlmostEqual(corridor[0], -0.94, places=6)
        self.assertAlmostEqual(corridor[1], 1.55, places=6)
        self.assertAlmostEqual(side[0], -2.365, places=6)
        self.assertAlmostEqual(side[1], 1.75, places=6)
        # The long leg stays east of opening x=-1.65. The short leg
        # then crosses only the solid 3.02 x 1.00 m landing.
        self.assertGreater(min(transition.truth_pose[0], corridor[0]), -1.65)

    def test_upper_floor_outbound_defers_failed_near_door_retry(self):
        overrides = SecondFloorMission._second_floor_only_parameter_overrides()
        self.assertTrue(
            overrides["known_unvisited_door_retry_only_on_terminal_return"])
        self.assertEqual(overrides["corridor_partial_return_reserve_seconds"],
                         75.0)
        self.assertEqual(
            overrides[
                "corridor_partial_return_room_minimum_remaining_seconds"],
            45.0)


    def test_outbound_retry_gate_returns_before_touching_scheduler(self):
        manager = BaselineExplorationManager.__new__(BaselineExplorationManager)
        manager.enable_known_unvisited_door_retry = True
        manager.known_unvisited_door_retry_only_on_terminal_return = True
        manager.corridor_terminal_return_latched = False
        manager.corridor_door_detection_armed = True
        manager.room_scheduler = SimpleNamespace(active_door=None)
        self.assertIsNone(manager._plan_known_unvisited_door_retry_goal())


    def test_run99_far_corridor_pose_triggers_truth_return_takeover(self):
        transition = StairTransition.__new__(StairTransition)
        transition.enable_truth_return_gate = True
        transition.truth_entry_guide = True
        transition.return_transit_armed = True
        transition.return_gate_published = False
        transition.phase = "WAIT_F1"
        transition.truth_pose = (-0.558, 28.999, 0.32, 0.05)
        transition.truth_step_pose = (-4.015, 2.18, 0.0, 0.0)
        transition.truth_step_next_pose = (-4.015, 2.30)
        transition.truth_staging_standoff = 0.43
        transition.truth_return_gate_radius = 35.0
        transition.return_gate_pub = mock.Mock()
        transition._maybe_publish_truth_return_gate()
        self.assertTrue(transition.return_gate_published)
        transition.return_gate_pub.publish.assert_called_once()

    def test_truth_long_return_launch_takes_over_before_fastlio_drift(self):
        launch = ET.parse(os.path.join(
            SCRIPT_DIR, "..", "launch",
            "baseline_fastlio_exploration.launch"))
        node = next(item for item in launch.getroot().findall("node")
                    if item.get("name") == "stair_transition_manager")
        params = {item.get("name"): item.get("value")
                  for item in node.findall("param")}
        self.assertEqual(params["truth_return_gate_radius_m"], "35.0")
        self.assertEqual(
            params["truth_corridor_lateral_offset_m"],
            "$(arg stair_truth_corridor_lateral_offset_m)")
        self.assertEqual(
            params["truth_corridor_longitudinal_offset_m"],
            "$(arg stair_truth_corridor_longitudinal_offset_m)")

    def test_pre_ascent_alignment_watchdog_refreshes_while_yaw_improves(self):
        transition = StairTransition.__new__(StairTransition)
        transition.pre_ascent_align_timeout = 20.0
        transition.pre_ascent_align_watchdog_progress = 0.08
        transition.pre_ascent_align_watchdog_anchor_error = None
        transition.pre_ascent_align_deadline = None
        self.assertTrue(transition._refresh_pre_ascent_align_progress_watchdog(
            1.44, now=0.0))
        self.assertAlmostEqual(transition.pre_ascent_align_deadline, 20.0)
        self.assertTrue(transition._refresh_pre_ascent_align_progress_watchdog(
            1.35, now=19.9))
        self.assertAlmostEqual(transition.pre_ascent_align_deadline, 39.9)
        self.assertFalse(transition._refresh_pre_ascent_align_progress_watchdog(
            1.31, now=25.0))
        self.assertAlmostEqual(transition.pre_ascent_align_deadline, 39.9)

    def test_truth_entry_watchdog_refreshes_while_distance_improves(self):
        transition = StairTransition.__new__(StairTransition)
        transition.truth_entry_timeout = 60.0
        transition.truth_entry_watchdog_progress = 0.25
        transition.truth_entry_watchdog_anchor_distance = 9.10
        transition.truth_entry_deadline = 60.0
        self.assertTrue(transition._refresh_truth_entry_progress_watchdog(
            8.80, now=59.9))
        self.assertAlmostEqual(transition.truth_entry_deadline, 119.9)
        self.assertFalse(transition._refresh_truth_entry_progress_watchdog(
            8.70, now=70.0))
        self.assertAlmostEqual(transition.truth_entry_deadline, 119.9)

    def test_flight_a_center_error_corrects_back_to_tread_axis(self):
        # +Y stair: x below the step centre must command +X correction.
        error = StairTransition._truth_stair_center_error(
            (-4.29, 3.05), (-4.015, 1.85), np.pi / 2.0)
        self.assertAlmostEqual(error, 0.275, places=6)
        # The same geometry rotated onto a +X stair remains valid.
        rotated = StairTransition._truth_stair_center_error(
            (3.05, 4.29), (1.85, 4.015), 0.0)
        self.assertAlmostEqual(rotated, 0.275, places=6)

    def test_flight_a_control_combines_forward_and_centering(self):
        transition = StairTransition.__new__(StairTransition)
        transition.truth_pose = (-4.29, 3.05, 0.95, 1.40)
        transition.truth_step_pose = (-4.015, 1.85, 0.0, 0.0)
        transition.truth_stair_heading = np.pi / 2.0
        transition.truth_flight_a_center_deadband = 0.04
        transition.truth_flight_a_center_gain = 0.55
        transition.truth_flight_a_center_speed = 0.12
        transition.truth_flight_a_heading_gain = 0.35
        transition.truth_flight_a_max_yaw_rate = 0.10
        transition.truth_flight_a_recovery_center_error = 0.14
        transition.truth_flight_a_recovery_heading_error = 0.20
        transition.truth_flight_a_recovery_forward_speed = 0.12
        transition.truth_flight_a_recovery_center_speed = 0.18
        transition.truth_flight_a_recovery_yaw_rate = 0.16
        transition.ascent_speed = 0.80
        vx, vy, yaw_rate, center_error, heading_error = (
            transition._truth_flight_a_control())
        self.assertAlmostEqual(center_error, 0.275, places=6)
        self.assertAlmostEqual(vx, 0.18, places=6)
        self.assertAlmostEqual(vy, 0.12, places=6)
        self.assertGreater(heading_error, 0.0)
        self.assertAlmostEqual(yaw_rate, 0.0597787, places=5)
        self.assertTrue(transition.truth_flight_a_recovery_active)

    def test_flight_a_recovery_crosses_yaw_deadband_before_guard(self):
        transition = StairTransition.__new__(StairTransition)
        transition.truth_pose = (-3.84, 3.53, 1.14, 2.01)
        transition.truth_step_pose = (-4.015, 1.85, 0.0, 0.0)
        transition.truth_stair_heading = np.pi / 2.0
        transition.truth_flight_a_center_deadband = 0.04
        transition.truth_flight_a_center_gain = 0.55
        transition.truth_flight_a_center_speed = 0.12
        transition.truth_flight_a_heading_gain = 0.35
        transition.truth_flight_a_max_yaw_rate = 0.10
        transition.truth_flight_a_recovery_center_error = 0.14
        transition.truth_flight_a_recovery_heading_error = 0.20
        transition.truth_flight_a_recovery_forward_speed = 0.12
        transition.truth_flight_a_recovery_center_speed = 0.18
        transition.truth_flight_a_recovery_yaw_rate = 0.16
        transition.ascent_speed = 0.20
        vx, vy, yaw_rate, center_error, heading_error = (
            transition._truth_flight_a_control())
        self.assertLess(center_error, -0.14)
        self.assertLess(heading_error, -0.20)
        self.assertAlmostEqual(vx, -0.18, places=6)
        self.assertAlmostEqual(vy, 0.12, places=6)
        self.assertAlmostEqual(yaw_rate, -0.16, places=6)
        self.assertTrue(transition.truth_flight_a_recovery_active)

    def test_flight_b_gets_budget_after_slow_landing_turn(self):
        transition = StairTransition.__new__(StairTransition)
        transition.started = 0.0
        transition.ascent_timeout = 45.0
        transition.phase = "STAIR_ASCENT_B"
        transition.flight_b_started_at = 41.6
        transition.flight_b_timeout = 22.0
        transition.total_height_gain = 2.20
        transition.second_floor_height_reached_at = None
        transition.upper_landing_clearance_grace = 15.0
        # Reproduce run62: healthy second-flight gain at the former global
        # deadline must continue instead of being stopped after 3.4 seconds.
        self.assertFalse(transition._truth_ascent_timed_out(45.0, 1.59))
        self.assertFalse(transition._truth_ascent_timed_out(63.59, 2.19))
        self.assertTrue(transition._truth_ascent_timed_out(63.61, 2.19))

    def test_global_stair_timeout_still_bounds_pre_flight_b_stall(self):
        transition = StairTransition.__new__(StairTransition)
        transition.started = 0.0
        transition.ascent_timeout = 45.0
        transition.phase = "STAIR_LANDING_TURN"
        transition.flight_b_started_at = None
        transition.flight_b_timeout = 22.0
        transition.total_height_gain = 2.20
        transition.second_floor_height_reached_at = None
        transition.upper_landing_clearance_grace = 15.0
        self.assertTrue(transition._truth_ascent_timed_out(45.01, 1.30))

    def test_run84_landing_timeout_enters_bounded_recovery(self):
        transition = StairTransition.__new__(StairTransition)
        transition.truth_entry_guide = True
        transition.truth_fixed_two_flight_profile = True
        transition.truth_pose = (-2.5904, 4.9143, 1.6330, -1.3587)
        transition.truth_ascent_start_z = 0.3207
        transition.truth_landing_position_error = 0.1054
        transition.truth_landing_heading_error = -0.2121
        transition.landing_recovery_max_position_error = 0.40
        transition.landing_recovery_max_heading_error = 0.65
        transition.landing_recovery_max_attempts = 2
        transition.landing_recovery_attempts = 0
        transition.landing_recovery_active = False
        transition.landing_recovery_timeout = 12.0
        transition.landing_timeout = 30.0
        transition.landing_started_at = 100.0
        transition.cmd = SimpleNamespace(publish=mock.Mock())
        transition.state = SimpleNamespace(publish=mock.Mock())
        transition._record_trace = mock.Mock()
        transition.hold_rl = mock.Mock()
        with mock.patch(
                "stair_transition_manager.time.monotonic",
                return_value=130.0), mock.patch(
                    "stair_transition_manager.rospy.signal_shutdown") as shutdown:
            handled = transition._landing_turn_timed_out(30.0)
        self.assertTrue(handled)
        self.assertEqual(transition.phase, "STAIR_LANDING_RECOVERY")
        self.assertEqual(transition.landing_recovery_attempts, 1)
        self.assertEqual(transition.landing_started_at, 130.0)
        transition.hold_rl.assert_called_once_with()
        shutdown.assert_not_called()

    def test_landing_recovery_crosses_run84_yaw_dead_zone(self):
        transition = StairTransition.__new__(StairTransition)
        transition.landing_turn_speed = 0.45
        transition.truth_landing_minimum_yaw_rate = 0.24
        transition.landing_recovery_minimum_yaw_rate = 0.34
        transition.landing_heading_tolerance = 0.15
        transition.landing_recovery_active = False
        ordinary = transition._truth_landing_turn_rate(-0.2121)
        transition.landing_recovery_active = True
        recovery = transition._truth_landing_turn_rate(-0.2121)
        self.assertAlmostEqual(ordinary, -0.24, places=6)
        self.assertAlmostEqual(recovery, -0.34, places=6)

    def test_unsafe_landing_pose_is_not_retried(self):
        transition = StairTransition.__new__(StairTransition)
        transition.truth_entry_guide = True
        transition.truth_fixed_two_flight_profile = True
        transition.truth_pose = (-3.20, 4.91, 1.63, -1.36)
        transition.truth_ascent_start_z = 0.32
        transition.truth_landing_position_error = 0.715
        transition.truth_landing_heading_error = -0.21
        transition.landing_recovery_max_position_error = 0.40
        transition.landing_recovery_max_heading_error = 0.65
        self.assertFalse(transition._landing_recovery_safe())
        transition.landing_recovery_max_attempts = 2
        transition.landing_recovery_attempts = 0
        transition.landing_recovery_active = False
        transition.landing_recovery_timeout = 12.0
        transition.landing_timeout = 30.0
        transition.cmd = SimpleNamespace(publish=mock.Mock())
        transition.state = SimpleNamespace(publish=mock.Mock())
        transition._record_trace = mock.Mock()
        with mock.patch(
                "stair_transition_manager.rospy.signal_shutdown") as shutdown:
            handled = transition._landing_turn_timed_out(30.0)
        self.assertTrue(handled)
        self.assertEqual(transition.phase, "STAIR_LANDING_TIMEOUT")
        self.assertEqual(transition.landing_recovery_attempts, 0)
        shutdown.assert_called_once_with("stair_landing_turn_timeout")




    def test_long_corridor_leg_gets_distance_based_budget(self):
        # run36 required about 50 s for its 7.48 m corridor leg.  The former
        # shared 45 s deadline expired after the preceding stair-exit leg.
        budget = SecondFloorMission._truth_route_stage_timeout(
            45.0, 7.48, 0.10, 6.0)
        self.assertGreaterEqual(budget, 80.8)

    def test_short_leg_keeps_configured_minimum_budget(self):
        budget = SecondFloorMission._truth_route_stage_timeout(
            45.0, 1.55, 0.10, 6.0)
        self.assertEqual(budget, 45.0)

    def test_truth_height_requires_second_floor_elevation(self):
        self.assertTrue(SecondFloorMission._truth_height_is_second_floor(
            (-2.5, 1.7, 2.91, 0.0), 2.60, 0.15))
        self.assertFalse(SecondFloorMission._truth_height_is_second_floor(
            (-2.5, 1.7, 0.31, 0.0), 2.60, 0.15))

    def test_second_floor_context_translates_first_floor_bounds(self):
        context = SecondFloorMission._second_floor_executor_context(2.60)
        self.assertAlmostEqual(context["minimum_pose_z"], 1.80)
        self.assertAlmostEqual(context["maximum_pose_z"], 3.80)
        self.assertEqual(
            GoalExecutor._validated_floor_context(context), (1.80, 3.80))

    def test_third_floor_context_reuses_same_validated_executor_contract(self):
        context = SecondFloorMission._second_floor_executor_context(
            5.20, 2, "third_floor")
        self.assertEqual(context["mode"], "third_floor_exploration")
        self.assertEqual(context["floor_index"], 2)
        self.assertAlmostEqual(context["minimum_pose_z"], 4.40)
        self.assertAlmostEqual(context["maximum_pose_z"], 6.40)
        self.assertEqual(
            GoalExecutor._validated_floor_context(context), (4.40, 6.40))

    def test_three_floor_scene_preserves_lower_layout_and_has_second_stair(self):
        workspace = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", ".."))
        old_dir = os.path.join(
            workspace, "generated_building", "elevator_two_floor_debug")
        new_dir = os.path.join(
            workspace, "generated_building", "elevator_three_floor_debug")
        with open(os.path.join(old_dir, "layout_metadata.json")) as stream:
            old = json.load(stream)
        with open(os.path.join(new_dir, "layout_metadata.json")) as stream:
            new = json.load(stream)
        self.assertEqual(old["floors"], new["floors"][:2])
        self.assertEqual(new["floors"][2]["floor_index"], 2)
        links = {link.get("name") for link in ET.parse(
            os.path.join(new_dir, "model.sdf")).getroot().findall(".//link")}
        self.assertIn("stair_flight_a_floor_1_step_0", links)
        self.assertIn("stair_flight_b_floor_1_step_9", links)
        self.assertIn("stair_floor_landing_floor_2", links)

    @staticmethod
    def _monotonic_probe_manager(occupied=False, unknown_queries=0):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.enable_corridor_monotonic_forward_probe = True
        manager.corridor_monotonic_probe_count = 0
        manager.corridor_monotonic_probe_limit = 24
        manager.corridor_monotonic_probe_distance = 0.60
        manager.corridor_monotonic_probe_sample_spacing = 0.10
        manager.corridor_monotonic_probe_max_unknown_queries = 8
        manager.corridor_monotonic_probe_mode_active = False
        manager.room_target_count = 4
        manager.room_scheduler = SimpleNamespace(
            active_door=None,
            detector=SimpleNamespace(doors=[]))
        manager.corridor_station_axis = np.asarray([1.0, 0.0])
        manager.corridor_station_origin = np.asarray([0.0, 0.0])
        manager.corridor_anchor = np.asarray([0.0, 0.0])
        manager.corridor_sweep_history = []
        manager.reached_tolerance = 0.15
        manager._missing_room_retrace_active = mock.Mock(return_value=False)
        manager._forward_corridor_wall_observed = mock.Mock(
            return_value=False)
        manager._scan_check = mock.Mock(return_value=SimpleNamespace(
            map_available=True,
            occupied_collision=occupied,
            unknown_queries=unknown_queries,
            status="test"))
        manager.elapsed = mock.Mock(return_value=12.0)
        return manager

    def test_f2_monotonic_probe_advances_on_dense_3d_clearance(self):
        manager = self._monotonic_probe_manager()
        goal, path = manager._plan_corridor_monotonic_forward_probe(
            9, (10.0, 0.12, 2.4, 0.0), object(),
            np.asarray([1.0, 0.0]))
        self.assertIsNotNone(goal)
        self.assertEqual(goal["source"],
                         "corridor_monotonic_forward_probe")
        self.assertTrue(goal["verified_corridor_3d_probe"])
        self.assertAlmostEqual(goal["position"][0], 10.60)
        self.assertAlmostEqual(goal["position"][1], 0.0)
        self.assertEqual(manager.corridor_monotonic_probe_count, 1)
        self.assertGreaterEqual(manager._scan_check.call_count, 6)
        self.assertEqual(path["execution_waypoints"], [(10.6, 0.0)])

    def test_f2_monotonic_probe_never_crosses_3d_occupied_sample(self):
        manager = self._monotonic_probe_manager(occupied=True)
        goal, path = manager._plan_corridor_monotonic_forward_probe(
            9, (10.0, 0.0, 2.4, 0.0), object(),
            np.asarray([1.0, 0.0]))
        self.assertIsNone(goal)
        self.assertIsNone(path)
        self.assertEqual(manager.corridor_monotonic_probe_count, 0)
        self.assertEqual(
            manager.corridor_sweep_history[-1]["event"],
            "CORRIDOR_MONOTONIC_PROBE_3D_REJECTED")

    def test_dense_3d_probe_is_not_rejected_again_by_stale_2d_map(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        result = manager._refine_path(
            9,
            {"success": True,
             "path": [(0.0, 0.0), (0.3, 0.0), (0.6, 0.0)]},
            {"verified_corridor_3d_probe": True})
        self.assertTrue(result["success"])
        self.assertEqual(
            result["refinement_status"],
            "dense_live_3d_corridor_probe_verified")
        self.assertAlmostEqual(result["refined_path_length"], 0.6)

    def test_executor_rejects_context_without_f2_gate(self):
        self.assertIsNone(GoalExecutor._validated_floor_context({
            "mode": "first_floor", "floor_index": 0,
            "minimum_pose_z": 1.80, "maximum_pose_z": 3.80,
        }))

    def test_second_floor_localization_uses_relative_frame_disagreement(self):
        result = SecondFloorMission._localization_disagreement(
            (0.0, 8.8, 2.9, 1.5),
            (0.02, 8.81, 2.9, 1.54),
            (24.0, -0.2, 2.9, -2.7),
            (24.03, -0.19, 2.9, -2.65))
        self.assertLess(result["relative_planar_disagreement_m"], 0.02)
        self.assertAlmostEqual(
            result["relative_yaw_disagreement_rad"], 0.01, places=6)

    def test_second_floor_localization_detects_stationary_lio_drift(self):
        result = SecondFloorMission._localization_disagreement(
            (0.0, 8.8, 2.9, 1.5),
            (0.0, 8.8, 2.9, 1.5),
            (24.0, -0.2, 2.9, -2.7),
            (29.2, -0.2, 2.9, -2.7))
        self.assertAlmostEqual(
            result["relative_planar_disagreement_m"], 5.2, places=6)

    def test_second_floor_localization_detects_missed_stair_height(self):
        result = SecondFloorMission._localization_disagreement(
            (-0.13, 8.60, 2.912, 1.39),
            (-0.13, 8.60, 2.912, 1.39),
            (22.82, 6.18, 1.020, -0.19),
            (22.82, 6.18, 1.020, -0.19))
        self.assertAlmostEqual(
            result["absolute_vertical_disagreement_m"], 1.892, places=6)

    def test_second_floor_localization_accepts_normal_sensor_height_offset(self):
        result = SecondFloorMission._localization_disagreement(
            (-0.11, 8.60, 2.912, 1.40),
            (-0.11, 8.60, 2.912, 1.40),
            (22.03, -0.07, 2.500, -0.19),
            (22.03, -0.07, 2.500, -0.19))
        self.assertLess(
            result["absolute_vertical_disagreement_m"], 0.65)

    def test_second_floor_truth_pose_is_relayed_for_runtime_guard(self):
        mission = SecondFloorMission.__new__(SecondFloorMission)
        mission._lock = threading.RLock()
        mission._truth_pose = None
        mission._truth_pose_received_at = -1.0
        mission._truth_correction_pub = mock.Mock()
        pose = SimpleNamespace(
            position=SimpleNamespace(x=1.0, y=2.0, z=2.9),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0))
        twist = SimpleNamespace(
            linear=SimpleNamespace(x=0.1, y=0.2, z=0.0),
            angular=SimpleNamespace(x=0.0, y=0.0, z=0.3))
        message = SimpleNamespace(
            name=["ground", "a1_gazebo"],
            pose=[pose, pose], twist=[twist, twist])
        with mock.patch(
                "second_floor_exploration_manager.rospy.Time.now",
                return_value=mock.Mock()):
            SecondFloorMission._on_truth_states(mission, message)
        mission._truth_correction_pub.publish.assert_called_once()
        relayed = mission._truth_correction_pub.publish.call_args.args[0]
        self.assertEqual(relayed.header.frame_id, "world")
        self.assertEqual(relayed.child_frame_id, "a1_gazebo")
        self.assertAlmostEqual(relayed.pose.pose.position.z, 2.9)
        self.assertAlmostEqual(relayed.twist.twist.linear.y, 0.2)

    def test_second_floor_corridor_guards_are_process_local(self):
        overrides = SecondFloorMission._second_floor_only_parameter_overrides()
        self.assertEqual(overrides["corridor_door_post_latch_motion_m"], 0.35)
        self.assertEqual(
            overrides["first_room_minimum_corridor_station_m"], 0.75)
        self.assertNotIn(
            "corridor_forward_recovery_minimum_advance_m", overrides)
        self.assertNotIn("corridor_forward_lateral_search_m", overrides)
        self.assertNotIn("room_require_portal_preflight_for_entry", overrides)
        self.assertNotIn("room_entry_retry_limit", overrides)
        self.assertTrue(
            overrides["corridor_short_door_commit_after_establishment"])
        self.assertFalse(
            overrides["enforce_room_phase_target_deadline"])
        self.assertNotIn("corridor_search_speed", overrides)
        self.assertNotIn("corridor_forward_distance", overrides)
        self.assertTrue(
            overrides["enable_corridor_monotonic_forward_probe"])
        self.assertAlmostEqual(
            overrides["corridor_monotonic_probe_distance_m"], 0.60)
        self.assertAlmostEqual(
            overrides["corridor_monotonic_probe_sample_spacing_m"], 0.10)
        self.assertEqual(overrides["corridor_monotonic_probe_limit"], 24)
        self.assertNotIn(
            "corridor_side_candidate_max_forward_station_m", overrides)
        self.assertNotIn("enable_terminal_door_context_fallback", overrides)
        self.assertFalse(
            overrides["room_exit_require_raw_corridor_confirmation"])
        self.assertFalse(
            overrides["room_exit_require_station_corridor_confirmation"])
        self.assertTrue(
            overrides["room_accept_reversed_entry_trace_exit"])
        self.assertEqual(
            overrides["room_corridor_side_confirmation_count"], 2)
        self.assertAlmostEqual(
            overrides["room_door_minimum_centerline_lateral_m"], 0.25)
        self.assertAlmostEqual(
            overrides["room_reversed_entry_trace_exit_tolerance_m"], 0.30)
        self.assertTrue(overrides["enable_terminal_missing_room_retrace"])
        self.assertAlmostEqual(
            overrides[
                "terminal_missing_room_retrace_minimum_outbound_progress_m"],
            12.0)
        self.assertAlmostEqual(
            overrides["room_door_cooldown_seconds"], 20.0)
        self.assertTrue(overrides["enable_known_unvisited_door_retry"])
        self.assertAlmostEqual(
            overrides["known_unvisited_door_retry_distance_m"], 6.5)
        self.assertFalse(
            overrides["terminal_return_blocked_by_known_unvisited_door"])

    def test_run75_upper_floor_rejects_corridor_internal_fake_door(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.room_door_minimum_centerline_lateral = 0.25
        manager.corridor_station_origin = np.asarray(
            [23.5458, 1.2090], dtype=float)
        manager.corridor_station_axis = np.asarray(
            [0.9848, 0.17365], dtype=float)
        manager.corridor_sweep_history = []
        manager.elapsed = lambda: 180.555

        # The real room-0 centre lies across the corridor wall, while run75's
        # estimated room-02 centre lies on the online centreline.
        self.assertTrue(manager._door_centerline_lateral_admissible(
            [26.1887, 2.1422], "test", "real"))
        self.assertFalse(manager._door_centerline_lateral_admissible(
            [27.3414, 1.9250], "test", "run75-fake"))
        event = manager.corridor_sweep_history[-1]
        self.assertEqual(
            event["event"],
            "DOOR_REJECTED_INSIDE_CORRIDOR_CENTER_BAND")
        self.assertLess(event["centerline_lateral_m"], 0.10)

        # Default/F1 remains disabled and admits the same geometry.
        manager.room_door_minimum_centerline_lateral = 0.0
        self.assertTrue(manager._door_centerline_lateral_admissible(
            [27.3414, 1.9250], "test", "f1-default"))

    def test_upper_floor_terminal_return_waits_for_registered_missing_door(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.terminal_return_blocked_by_known_unvisited_door = True
        manager.room_scheduler = SimpleNamespace(
            detector=SimpleNamespace(doors=[
                SimpleNamespace(door_id="door_1", visited=True,
                                completed=False),
                SimpleNamespace(door_id="door_2", visited=False,
                                completed=False),
            ]))
        self.assertTrue(
            manager._terminal_return_has_blocking_known_door(False))
        self.assertFalse(
            manager._terminal_return_has_blocking_known_door(True))
        manager.terminal_return_blocked_by_known_unvisited_door = False
        self.assertFalse(
            manager._terminal_return_has_blocking_known_door(False))

    def test_failed_second_floor_result_cannot_arm_next_stair(self):
        self.assertFalse(SecondFloorMission._exploration_handoff_authorized(
            "ROOM_EXIT_BLOCKED", 1, 4, False))
        self.assertFalse(SecondFloorMission._exploration_handoff_authorized(
            "TIME_LIMIT", 4, 4, False))

    def test_completed_return_arms_next_stair_after_four_exits(self):
        self.assertTrue(SecondFloorMission._exploration_handoff_authorized(
            "STAIR_CORRIDOR_EXIT_HANDOFF", 4, 4, False))

    def test_upper_stair_trigger_requires_exact_success_token(self):
        token = "SECOND_FLOOR_EXPLORATION_COMPLETE"
        self.assertTrue(StairTransition._mission_trigger_matches(token, token))
        self.assertFalse(StairTransition._mission_trigger_matches(
            token + "_DIAGNOSTIC", token))
        self.assertFalse(StairTransition._mission_trigger_matches(
            "SECOND_FLOOR_EXPLORATION_FAILED", token))

    def test_third_floor_waiter_latches_exact_source_floor_failure(self):
        mission = SecondFloorMission.__new__(SecondFloorMission)
        mission._lock = threading.RLock()
        mission._source_floor_failure_token = \
            "SECOND_FLOOR_EXPLORATION_FAILED"
        mission._source_floor_failed = False
        mission._on_source_floor_state(SimpleNamespace(
            data="SECOND_FLOOR_EXPLORATION_COMPLETE"))
        self.assertFalse(mission._source_floor_failed)
        mission._on_source_floor_state(SimpleNamespace(
            data="SECOND_FLOOR_EXPLORATION_FAILED"))
        self.assertTrue(mission._source_floor_failed)

    def test_terminal_return_policy_remains_valid_with_partial_room_count(self):
        self.assertTrue(SecondFloorMission._exploration_handoff_authorized(
            "STAIR_CORRIDOR_EXIT_HANDOFF_TRUTH_GATE", 3, 4, True))
        self.assertFalse(SecondFloorMission._exploration_handoff_authorized(
            "STAIR_CORRIDOR_EXIT_HANDOFF_TRUTH_GATE", 3, 4, False))

    def test_run80_terminal_return_ignores_failed_portal(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.terminal_return_blocked_by_known_unvisited_door = False
        manager.room_scheduler = SimpleNamespace(
            detector=SimpleNamespace(doors=[SimpleNamespace(
                door_id="failed_far_portal", visited=False,
                completed=False)]))
        self.assertFalse(
            manager._terminal_return_has_blocking_known_door(False))
        self.assertTrue(
            SecondFloorMission._exploration_handoff_authorized(
                "STAIR_CORRIDOR_EXIT_HANDOFF_TRUTH_GATE", 3, 4, True))

    def test_second_floor_terminal_retrace_requires_outbound_progress(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.terminal_missing_room_retrace_minimum_outbound_progress = 12.0
        manager.corridor_station_axis = np.asarray([1.0, 0.0])
        manager.corridor_station_origin = np.asarray([3.0, -2.0])
        manager.corridor_forward_station_sign = 1.0
        self.assertFalse(
            manager._terminal_missing_room_retrace_progress_met(
                (14.9, -2.0, 2.8)))
        self.assertTrue(
            manager._terminal_missing_room_retrace_progress_met(
                (15.1, -2.0, 2.8)))

    def test_run77_axis_relock_synchronizes_station_frame(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        old_axis = np.asarray([0.979626310785161, -0.2008290099001019])
        old_axis /= np.linalg.norm(old_axis)
        old_origin = np.asarray([15.752156853302639,
                                 -0.39958992868390775])
        manager.corridor_station_axis = old_axis.copy()
        manager.corridor_station_origin = old_origin.copy()
        manager.corridor_axis = old_axis.copy()
        manager.corridor_anchor = old_origin.copy()
        manager.corridor_forward_station_sign = 1.0
        manager.corridor_forward_station_high_water = 7.0
        manager.corridor_station_centerline_rebased = False
        sync = manager._synchronize_corridor_station_frame_after_relock(
            [1.0, 0.0], [18.45, -0.93],
            (18.45, -0.93, 2.73, 0.0))
        self.assertEqual(sync["new_axis"], [1.0, 0.0])
        self.assertTrue(np.allclose(
            manager.corridor_station_axis, [1.0, 0.0]))
        self.assertAlmostEqual(manager.corridor_station_origin[1], -0.93)
        observed_exit = np.asarray([22.09326083401102,
                                    -0.9070237629626354])
        old_lateral = abs(float(np.dot(
            observed_exit - old_origin, [-old_axis[1], old_axis[0]])))
        new_lateral = abs(float(np.dot(
            observed_exit - manager.corridor_station_origin, [0.0, 1.0])))
        self.assertGreater(old_lateral, 0.75)
        self.assertLess(new_lateral, 0.05)

    def test_run78_upper_floor_keeps_opposite_door_during_recenter(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.pending_paired_opposite_door = {
            "candidate_id": "local-door-1-track-127",
            "corridor_side": [29.8555, 0.6764],
            "stored_elapsed": 104.995,
        }
        manager.paired_opposite_recenter_pending_timeout = 30.0
        manager.lock = threading.RLock()
        manager.pose = (29.8555, 0.6764, 2.4334, 0.0)
        manager.grid = object()
        manager.reached_tolerance = 0.15
        manager.corridor_sweep_history = []
        manager.local_candidate_attempts = {}
        manager.elapsed = lambda: 122.097
        scheduler = SimpleNamespace(active_door=None)
        scheduler.consider_local_door_candidate = (
            lambda *_: SimpleNamespace(door_id="estimated_door_02"))
        scheduler.next_goal = lambda *_: {
            "position": [29.9, 1.7, 0.0],
            "source": "lightweight_room_entry",
        }
        manager.room_scheduler = scheduler

        goal = manager._plan_pending_paired_opposite_door()

        self.assertIsNotNone(goal)
        self.assertEqual(goal["scheduler_phase"],
                         "PAIRED_OPPOSITE_DOOR_ENTRY")
        self.assertEqual(goal["door_takeover_source"],
                         "paired_opposite_door_recenter")
        self.assertIsNone(manager.pending_paired_opposite_door)
        self.assertEqual(
            SecondFloorMission._second_floor_only_parameter_overrides()[
                "paired_opposite_recenter_pending_timeout_sec"], 30.0)

    def test_run79_pre_ascent_alignment_has_slow_yaw_budget(self):
        launch_path = os.path.abspath(os.path.join(
            SCRIPT_DIR, "..", "launch",
            "baseline_fastlio_exploration.launch"))
        root = ET.parse(launch_path).getroot()
        argument = next(item for item in root.findall("arg")
                        if item.get("name") ==
                        "pre_ascent_align_timeout_sec")
        self.assertEqual(float(argument.get("default")), 20.0)
        stair_nodes = [node for node in root.findall(".//node")
                       if node.get("type") ==
                       "stair_transition_manager.py"]
        self.assertGreaterEqual(len(stair_nodes), 2)
        for node in stair_nodes:
            parameter = next(item for item in node.findall("param")
                             if item.get("name") ==
                             "pre_ascent_align_timeout_sec")
            self.assertEqual(parameter.get("value"),
                             "$(arg pre_ascent_align_timeout_sec)")

    def test_second_floor_raw_corridor_uses_short_door_commit(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.corridor_station_origin = np.asarray([0.0, 0.0])
        manager.corridor_anchor = np.asarray([0.0, 0.0])
        manager.corridor_door_forward_pending = True
        manager.corridor_established_from_local_door = False
        manager.corridor_short_door_commit_after_establishment = True
        manager.corridor_door_detection_armed = False
        manager.corridor_door_forward_commit_distance = 1.5
        manager.corridor_forward_distance = 7.0
        manager.corridor_forward_recovery_minimum_advance = 0.35
        manager.corridor_forward_lateral_search = 0.0
        manager.corridor_minimum_advance = 2.5
        manager.clearance = 0.38
        manager.reached_tolerance = 0.15
        manager.corridor_reversed = False
        with mock.patch.object(
                baseline_manager_module, "astar_safe_path",
                return_value={"success": True,
                              "path": [(0.0, 0.0), (1.5, 0.0)]}):
            goal, _ = manager._corridor_forward_plan(
                1, (0.0, 0.0, 2.8), object(), np.asarray([1.0, 0.0]))
        self.assertEqual(goal["source"], "corridor_door_forward_commit")
        self.assertAlmostEqual(goal["corridor_advance_m"], 1.5)

    def test_run81_first_floor_far_pair_search_uses_short_segments(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.corridor_station_origin = np.asarray([0.0, 0.0])
        manager.corridor_anchor = np.asarray([0.0, 0.0])
        manager.corridor_door_forward_pending = False
        manager.corridor_forward_distance = 7.0
        manager.corridor_far_pair_search_advance = 3.0
        manager.corridor_far_pair_search_after_exits = 2
        manager.corridor_forward_recovery_minimum_advance = 0.35
        manager.corridor_forward_lateral_search = 0.0
        manager.corridor_minimum_advance = 2.5
        manager.clearance = 0.38
        manager.reached_tolerance = 0.15
        manager.corridor_reversed = False
        manager.corridor_terminal_return_latched = False
        manager.corridor_sweep_history = []
        manager.elapsed = mock.Mock(return_value=120.0)
        manager._missing_room_retrace_active = lambda: False
        manager.room_scheduler = SimpleNamespace(
            active_door=None, detector=SimpleNamespace(doors=[
                SimpleNamespace(visited=index < 2) for index in range(4)
            ]))
        with mock.patch.object(
                baseline_manager_module, "astar_safe_path",
                return_value={"success": True,
                              "path": [(0.0, 0.0), (3.0, 0.0)]}):
            goal, _ = manager._corridor_forward_plan(
                1, (0.0, 0.0, 2.8), object(), np.asarray([1.0, 0.0]))
        self.assertAlmostEqual(goal["corridor_advance_m"], 3.0)
        self.assertEqual(manager.corridor_sweep_history[-1]["event"],
                         "CORRIDOR_FAR_PAIR_SEARCH_ADVANCE_CLAMPED")
        overrides = SecondFloorMission._second_floor_only_parameter_overrides()
        self.assertEqual(
            overrides["corridor_far_pair_search_maximum_advance_m"], 0.0)
        self.assertEqual(
            overrides["corridor_partial_return_minimum_outbound_progress_m"],
            20.0)
        self.assertEqual(
            overrides["corridor_partial_return_reserve_seconds"], 75.0)
        self.assertEqual(
            overrides["corridor_partial_return_room_minimum_remaining_seconds"],
            45.0)

        continuity = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        continuity.corridor_partial_return_minimum_outbound_progress = 20.0
        continuity.corridor_partial_return_reserve = 160.0
        continuity.corridor_forward_station_high_water = 19.8
        continuity.maximum_duration = 400.0
        continuity.elapsed = lambda: 200.0
        self.assertEqual(continuity._partial_corridor_return_trigger(),
                         "online_outbound_progress")
        continuity.corridor_forward_station_high_water = 5.0
        continuity.elapsed = lambda: 240.0
        self.assertEqual(continuity._partial_corridor_return_trigger(),
                         "next_floor_time_reserve")

    def test_active_corridor_goal_can_arm_confirmed_door_preemption(self):
        """A long in-flight goal must not starve the room scheduler."""
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)

        manager.enable_local_doorway_detector = True
        manager.corridor_door_detection_armed = False
        manager.room_scheduler = SimpleNamespace(active_door=None)
        manager.lock = threading.RLock()
        manager.local_entry_status = {"candidates": []}
        manager.pose = (10.0, 0.0, 0.0, 0.0)
        manager.grid = object()
        manager.last_local_entry_received = 0.5
        manager.local_candidate_freshness = 2.0
        manager.corridor_sweep_history = []
        manager.elapsed = mock.Mock(return_value=1.0)

        def arm_during_refresh(_pose, _context):
            manager.corridor_door_detection_armed = True
            return True

        phase_context = {
            "axis": [1.0, 0.0], "raw_is_corridor": True,
            "centerline_point": [10.0, 0.0],
        }
        with mock.patch.object(
                manager, "_corridor_context", return_value=phase_context), \
                mock.patch.object(
                    manager, "_update_corridor_door_phase",
                    side_effect=arm_during_refresh) as update_phase, \
                mock.patch.object(
                    manager, "_door_search_corridor_context",
                    return_value=None):
            candidate = manager._local_door_preemption_candidate()

        self.assertIsNone(candidate)
        self.assertTrue(manager.corridor_door_detection_armed)
        update_phase.assert_called_once_with(manager.pose, phase_context)
        self.assertEqual(
            manager.corridor_sweep_history[-1]["event"],
            "CORRIDOR_DOOR_PHASE_ARMED_DURING_ACTIVE_GOAL")

    def test_second_floor_exit_requires_live_raw_corridor(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.room_exit_require_raw_corridor_confirmation = True
        manager.room_exit_require_station_corridor_confirmation = False
        manager.corridor_station_centerline_rebased = False
        with mock.patch.object(
                manager, "_corridor_context",
                return_value={"raw_is_corridor": False}):
            confirmed, reason = manager._room_exit_corridor_confirmed(
                (1.0, 2.0, 2.8, 0.0), object())
        self.assertFalse(confirmed)
        self.assertEqual(reason, "exit_raw_corridor_geometry_unconfirmed")

    def test_first_floor_exit_guard_default_is_non_interfering(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.room_exit_require_raw_corridor_confirmation = False
        manager.room_exit_require_station_corridor_confirmation = False
        confirmed, reason = manager._room_exit_corridor_confirmed(None, None)
        self.assertTrue(confirmed)
        self.assertIsNone(reason)

    def test_run67_false_exit_is_outside_established_first_floor_band(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.room_exit_require_raw_corridor_confirmation = False
        manager.room_exit_require_station_corridor_confirmation = True
        manager.corridor_station_centerline_rebased = True
        manager.corridor_centerline_membership_tolerance = 0.75
        manager.room_exit_station_corridor_tolerance = 1.0
        manager.corridor_station_origin = np.asarray(
            [7.8780187518297, -1.4396349173917236])
        manager.corridor_station_axis = np.asarray(
            [0.9995096416055005, 0.03131255878467945])
        # Actual run67 fourth-room endpoint: it crossed the estimated oblique
        # door plane but remained in the physical far room.
        confirmed, reason = manager._room_exit_corridor_confirmed(
            (26.1071, -3.4975, 0.0, 0.0), object())
        self.assertFalse(confirmed)
        self.assertEqual(reason, "exit_outside_established_corridor_band")
        # The preceding physical far-room exit lies on the same online line.
        confirmed, reason = manager._room_exit_corridor_confirmed(
            (26.4317, -0.7596, 0.0, 0.0), object())
        self.assertTrue(confirmed)
        self.assertIsNone(reason)

    def test_run76_real_far_exit_uses_dedicated_executor_margin(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.room_exit_require_raw_corridor_confirmation = False
        manager.room_exit_require_station_corridor_confirmation = True
        manager.corridor_station_centerline_rebased = True
        manager.corridor_centerline_membership_tolerance = 0.75
        manager.room_exit_station_corridor_tolerance = 1.0
        manager.corridor_station_origin = np.asarray(
            [7.907010630742262, -0.4753680599130929])
        manager.corridor_station_axis = np.asarray(
            [0.9997656327482375, 0.021649008649734652])
        run76_exit_points = [
            (28.098851111433678, -0.8617079489616708, 0.0, 0.0),
            (28.11177007985996, -0.8183295937153728, 0.0, 0.0),
            (28.13188979279445, -0.8830464686489193, 0.0, 0.0),
        ]
        # All three were beyond the generic 0.75 m transit band but inside
        # the physical corridor and within the EXIT/executor-specific margin.
        self.assertFalse(manager._on_station_corridor_centerline(
            run76_exit_points[0]))
        for point in run76_exit_points:
            confirmed, reason = manager._room_exit_corridor_confirmed(
                point, object())
            self.assertTrue(confirmed)
            self.assertIsNone(reason)

    def test_second_floor_local_door_rejects_axis_only_room_context(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.enable_terminal_door_context_fallback = False
        with mock.patch.object(
                manager, "_corridor_context",
                return_value={"raw_is_corridor": False}), \
                mock.patch.object(
                    manager, "_on_established_corridor_centerline",
                    return_value=True):
            context = manager._door_search_corridor_context(
                (1.0, 2.0, 2.8, 0.0), object(), "test")
        self.assertIsNone(context)


    def test_return_pass_entry_requests_immediate_verified_exit(self):
        scheduler = LightweightRoomScheduler.__new__(
            LightweightRoomScheduler)
        scheduler.active_door = SimpleNamespace(door_id="estimated_door_02")
        scheduler.entry_confirmed = True
        scheduler.adaptive_route_queue = [{"role": "G1"}]
        scheduler.next_role_index = 0
        scheduler.completion_mode = None
        scheduler.state = "G1_CENTER"
        scheduler.events = []
        scheduler.active_room_id = "floor_2_estimated_room_02"
        self.assertTrue(scheduler.request_immediate_exit(
            281.623, "return_pass_entry_observation_complete"))
        self.assertEqual(scheduler.state, "ROOM_RETURN")
        self.assertEqual(scheduler.next_role_index, 3)
        self.assertEqual(scheduler.adaptive_route_queue, [])
        self.assertEqual(scheduler.events[-1]["event"],
                         "ROOM_RETURN_PASS_IMMEDIATE_EXIT_REQUESTED")


if __name__ == "__main__":
    unittest.main()
