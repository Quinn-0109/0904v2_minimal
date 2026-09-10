#!/usr/bin/env python3
"""ROS-light regression tests for the isolated F2 truth handoff policy."""

import os
import sys
import math
import inspect
import threading
import tempfile
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
from lightweight_room_core import EstimatedDoorway
from lightweight_room_core import portal_lateral_preserving_exit
from lightweight_room_core import select_near_door_side_second_visual_view
from goal_executor import GoalExecutor
from stair_transition_manager import StairTransition


class SecondFloorHandoffTest(unittest.TestCase):
    @staticmethod
    def _prefetch_test_manager(record):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.room_goal_prefetch_lock = threading.Lock()
        manager.room_goal_prefetch_token = int(record.get("token", 1))
        manager.room_goal_prefetch = record
        manager.room_goal_prefetch_thread = None
        manager.room_scheduler = SimpleNamespace(
            active_door=SimpleNamespace(door_id="door-1"))
        manager.corridor_sweep_history = []
        manager.elapsed = lambda: 42.0
        return manager

    def test_room_prefetch_commits_only_released_matching_scheduler_copy(self):
        copied_scheduler = SimpleNamespace(
            active_door=SimpleNamespace(door_id="door-1"))
        record = {
            "token": 1,
            "released": True,
            "finished": True,
            "room_id": "room-1",
            "door_id": "door-1",
            "completed_role": "G3",
            "scheduler": copied_scheduler,
            "goal": {
                "room_id": "room-1", "room_role": "G4",
                "_preplanned_path_result": {
                    "success": True, "reason": "path_found",
                    "path": [(0.0, 0.0), (1.0, 0.0)],
                },
            },
            "error": None,
            "grid_generation": 12,
            "release_grid_updates": 7,
            "planning_wall_sec": 0.4,
        }
        manager = self._prefetch_test_manager(record)

        goal = manager._consume_room_goal_prefetch((0.0, 0.0, 0.0))

        self.assertIs(manager.room_scheduler, copied_scheduler)
        self.assertEqual(goal["room_role"], "G4")
        self.assertTrue(goal["room_next_role_prefetched_during_scan"])
        self.assertNotIn("_preplanned_path_result", goal)
        self.assertIsNone(manager.room_goal_prefetch)
        self.assertEqual(
            manager.corridor_sweep_history[-1]["event"],
            "ROOM_NEXT_ROLE_PREFETCH_CONSUMED")

    def test_room_prefetch_late_worker_is_invalidated_before_live_planning(self):
        record = {
            "token": 3,
            "released": True,
            "finished": False,
            "room_id": "room-1",
            "door_id": "door-1",
        }
        manager = self._prefetch_test_manager(record)

        self.assertIsNone(
            manager._consume_room_goal_prefetch((0.0, 0.0, 0.0)))
        self.assertIsNone(manager.room_goal_prefetch)
        self.assertEqual(manager.room_goal_prefetch_token, 4)

    def test_room_prefetch_release_requires_scan_and_fresh_map(self):
        record = {
            "token": 5,
            "released": False,
            "finished": True,
            "room_id": "room-1",
            "door_id": "door-1",
            "completed_role": "G3",
        }
        manager = self._prefetch_test_manager(record)

        manager._release_room_goal_prefetch(
            "G3", "room-1", visual_sweep_completed=True,
            fresh_grid_updates=0)

        self.assertIsNone(manager.room_goal_prefetch)
        self.assertEqual(
            manager.corridor_sweep_history[-1]["event"],
            "ROOM_NEXT_ROLE_PREFETCH_DISCARDED")

    def test_f3_turn_rate_stays_outside_run133_plane_dead_zone(self):
        mission = SecondFloorMission.__new__(SecondFloorMission)
        mission._truth_heading_tolerance = 0.18
        mission._f3_turn_minimum_yaw_rate = 0.65
        mission._f3_turn_maximum_yaw_rate = 0.72

        self.assertEqual(mission._f3_effective_turn_rate(0.10), 0.0)
        self.assertAlmostEqual(
            mission._f3_effective_turn_rate(0.20), 0.65, places=6)
        self.assertAlmostEqual(
            mission._f3_effective_turn_rate(-0.55), -0.65, places=6)
        self.assertAlmostEqual(
            mission._f3_effective_turn_rate(1.50), 0.72, places=6)

    def test_f3_turn_assist_is_single_attempt_and_has_physical_arc(self):
        source = inspect.getsource(
            SecondFloorMission._guide_f3_stair_platform_to_corridor)
        assist = inspect.getsource(
            SecondFloorMission._f3_stationary_turn_assist)
        self.assertIn("turn_assist_attempted = False", source)
        self.assertIn("turn_assist_attempted = True", source)
        self.assertNotIn(
            "not turn_assisted and deadband_proven", source)
        self.assertIn("arc_recovery_used = True", assist)
        self.assertIn("_publish_f3_world_command", assist)
        self.assertIn("recovery_target=None", assist)
        self.assertIn("target_distance <= arrival_tolerance", assist)
        self.assertIn(
            "target_distance > arrival_tolerance", assist)
        self.assertIn("route_arc_active", assist)
        self.assertIn("max(0.08, 0.35 * target_distance)", assist)
        self.assertIn("world_x = arc_speed * dx / norm", assist)
        self.assertIn("world_y = arc_speed * dy / norm", assist)
        self.assertNotIn("set_model_state", assist)

    def test_f3_ingress_physically_crosses_lip_and_uses_entry_contract(self):
        source = inspect.getsource(
            SecondFloorMission._guide_f3_stair_platform_to_corridor)
        route_source = inspect.getsource(
            SecondFloorMission._second_floor_truth_route)
        self.assertIn(
            'corridor_entry_stage = "{}_corridor_entry".format(self._floor_slug)',
            source)
        self.assertIn(
            '"stage": "{}_corridor_entry".format(self._floor_slug)',
            route_source)
        self.assertIn(
            'waypoint.get("stage") == corridor_entry_stage', source)
        self.assertNotIn(
            'waypoint.get("stage") == "second_floor_corridor_entry"',
            source)
        self.assertNotIn(
            "f3_east_bridge_supported or f3_corridor_lip", source)
        self.assertIn(
            "remaining <= completion_tolerance or\n"
            "                            f3_east_bridge_supported", source)

    def test_failed_f3_descent_pre_align_has_bounded_physical_recovery(self):
        source_path = os.path.join(SCRIPT_DIR, "stair_descent_manager.py")
        with open(source_path, "r", encoding="utf-8") as stream:
            source = stream.read()
        method = source[
            source.index("    def _pre_align_tick"):
            source.index("    def _policy_loading_tick")]
        self.assertIn("pre_align_timeout_ros_sec", method)
        self.assertIn("pre_align_wall_timeout_sec", method)
        self.assertIn("STAIR_DESCENT_PRE_ALIGN_TIMEOUT", method)
        self.assertIn("pre_align_arc_used=True", method)
        self.assertIn("pre_align_minimum_yaw_rate", method)
        self.assertNotIn("set_model_state", method)

    def test_exit_centerline_release_requires_truth_portal_crossing(self):
        door = SimpleNamespace(
            truth_contract_center=(28.895, -1.10),
            truth_contract_normal_direction=-math.pi / 2.0)
        door.contract_depth = lambda point: -(
            float(point[1]) - door.truth_contract_center[1])

        # run133's false centreline pose was still just inside the truth
        # portal.  A real corridor pose lies at least 0.35 m outward.
        self.assertFalse(
            BaselineExplorationManager._exit_pose_physically_corridor_side(
                (27.94, -1.14), door))
        self.assertTrue(
            BaselineExplorationManager._exit_pose_physically_corridor_side(
                (28.89, -0.20), door))

    def test_truth_contract_claims_use_detector_door_registry(self):
        source = inspect.getsource(
            BaselineExplorationManager._assign_truth_viewpoint_contract)
        self.assertIn(
            'detector = getattr(self.room_scheduler, "detector", None)',
            source)
        self.assertIn('getattr(detector, "doors", [])', source)
        self.assertNotIn(
            'getattr(self.room_scheduler, "doors", [])', source)

    def test_verified_handoff_accepts_run125_legacy_stable_snapshot(self):
        status = {
            "base_z": 0.241846,
            "joint_position_error": 0.0317996,
            "joint_velocity_rms": 1.30453e-05,
            "pitch": -0.0236488,
            "roll": -2.46015e-05,
            "stable_duration": 2.49998,
            "stable_now": True,
        }
        self.assertTrue(
            SecondFloorMission._verified_handoff_acceleration_valid(status))

    def test_verified_handoff_rejects_missing_unstable_or_bad_acceleration(self):
        self.assertFalse(
            SecondFloorMission._verified_handoff_acceleration_valid({
                "stable_now": False, "stable_duration": 3.0}))
        self.assertFalse(
            SecondFloorMission._verified_handoff_acceleration_valid({
                "stable_now": True, "stable_duration": 3.0,
                "acceleration_norm": 3.0}))
        self.assertTrue(
            SecondFloorMission._verified_handoff_acceleration_valid({
                "stable_now": False, "acceleration_norm": 9.81}))

    def test_fullflow_mapper_persists_only_initial_and_explicit_milestones(self):
        launch_path = os.path.abspath(os.path.join(
            SCRIPT_DIR, "..", "launch",
            "baseline_fastlio_exploration.launch"))
        root = ET.parse(launch_path).getroot()
        mapper = next(node for node in root.findall("node")
                      if node.get("name") == "baseline_voxel_mapper")
        params = {item.get("name"): item.get("value")
                  for item in mapper.findall("param")}
        self.assertEqual(params.get("initial_save_updates"), "20")
        self.assertEqual(params.get("save_every_updates"), "0")

    def test_upper_floor_reuses_only_physically_verified_stair_plane_handoff(self):
        init_source = inspect.getsource(SecondFloorMission.__init__)
        helper_source = inspect.getsource(
            SecondFloorMission._verified_stair_plane_handoff_ready)
        restore_source = inspect.getsource(
            SecondFloorMission._restore_plane_policy)
        mission_source = inspect.getsource(SecondFloorMission._run_mission)

        self.assertIn("reuse_verified_stair_plane_handoff", init_source)
        self.assertIn("self._locomotion_ready", helper_source)
        self.assertIn("self._truth_pose_received_at", helper_source)
        self.assertIn("truth_age <= 0.5", helper_source)
        self.assertIn("joint_velocity_rms", helper_source)
        self.assertIn("joint_position_error", helper_source)
        self.assertIn("acceleration_norm", helper_source)
        self.assertIn("VERIFIED_STAIR_PLANE_HANDOFF_REUSED", helper_source)
        self.assertIn("VERIFIED_STAIR_PLANE_HANDOFF_REJECTED", helper_source)
        self.assertIn("rejection_reason", helper_source)
        self.assertIn(
            "if self._verified_stair_plane_handoff_ready():",
            restore_source)
        self.assertIn(
            "self._verified_stair_plane_handoff_reused", restore_source)
        precheck = mission_source.index(
            'self._mission_stage = "PRE_RESET_STAIR_HANDOFF_CHECK"')
        reset = mission_source.index(
            'self._mission_stage = "LANDING_RESET"')
        self.assertLess(precheck, reset)
        self.assertIn("publish_ready=False", mission_source)
        self.assertIn(
            "if self._verified_stair_plane_handoff_reused:",
            mission_source)
        self.assertIn("self._recover_fixed_stand()", restore_source)

    def test_f3_ingress_uses_stage_local_fast_caps_and_bounded_rl_arm(self):
        init_source = inspect.getsource(SecondFloorMission.__init__)
        guide_source = inspect.getsource(
            SecondFloorMission._guide_f3_stair_platform_to_corridor)
        restore_source = inspect.getsource(
            SecondFloorMission._restore_plane_policy)

        self.assertIn("third_floor_landing_clear_speed_mps", init_source)
        self.assertIn("third_floor_corridor_ingress_speed_mps", init_source)
        self.assertIn("third_floor_turn_deadband_assist_delay_sec", init_source)
        self.assertIn("self._f3_landing_clear_speed", guide_source)
        self.assertIn("self._f3_corridor_ingress_speed", guide_source)
        self.assertIn("deadband_progress < 0.12", guide_source)
        self.assertIn("settle_until = time.monotonic() + 0.50", guide_source)
        self.assertIn("arm_until = time.monotonic() + 0.75", restore_source)
        self.assertIn("_wait_for_real_locomotion_ready", restore_source)

    def test_f3_guide_motion_deadline_uses_sim_time_with_wall_watchdog(self):
        """Low RTF must not terminate a physically progressing F3 guide."""
        source = inspect.getsource(
            SecondFloorMission._guide_f3_stair_platform_to_corridor)
        self.assertIn("stage_sim_started = rospy.Time.now()", source)
        self.assertIn("stage_sim_deadline_sec", source)
        self.assertIn("stage_sim_hard_deadline_sec", source)
        self.assertIn("stage_wall_hard_deadline", source)
        self.assertIn(
            "(rospy.Time.now() - stage_sim_started).to_sec()", source)
        self.assertIn("last_progress_sim = stage_sim_started", source)
        self.assertIn("last_progress_sim = rospy.Time.now()", source)
        self.assertIn(
            "(rospy.Time.now() - last_progress_sim).to_sec()", source)
        self.assertNotIn("now - last_progress", source)
        self.assertNotIn("time.monotonic() < stage_deadline", source)

    def test_generated_truth_portal_keeps_live_corridor_lateral_offset(self):
        axis = np.asarray([0.9997573516, 0.0220281153])
        origin = np.asarray([7.6830307013, -0.1582318237])

        center = BaselineExplorationManager._generated_truth_portal_center(
            axis, origin, 14.865, -1, wall_offset=1.19)

        direction = axis / np.linalg.norm(axis)
        tangent = np.asarray([-direction[1], direction[0]])
        origin_lateral = float(np.dot(origin, tangent))
        self.assertAlmostEqual(float(np.dot(center, direction)), 14.865,
                               places=5)
        self.assertAlmostEqual(float(np.dot(center, tangent)),
                               origin_lateral - 1.19, places=5)

    def test_upper_floor_truth_portal_discards_lateral_fastlio_origin_bias(self):
        biased_origin = np.asarray([-0.606, 13.31])
        self.assertIsNone(
            BaselineExplorationManager.
            _generated_truth_portal_origin_for_floor(2, biased_origin))
        self.assertIsNone(
            BaselineExplorationManager.
            _generated_truth_portal_origin_for_floor(3, biased_origin))
        self.assertIs(
            BaselineExplorationManager.
            _generated_truth_portal_origin_for_floor(1, biased_origin),
            biased_origin)
        center = BaselineExplorationManager._generated_truth_portal_center(
            np.asarray([0.0, 1.0]), None, 28.895, -1,
            wall_offset=1.19)
        self.assertAlmostEqual(center[0], 1.19, places=6)
        self.assertAlmostEqual(center[1], 28.895, places=6)

    def test_f3_partial_return_is_irreversible_after_safe_exit_threshold(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.floor_number = 3
        manager.allow_partial_floor_handoff = True
        manager.room_target_count = 4
        manager.emergency_partial_floor_handoff_minimum_exited_rooms = 3
        doors = [SimpleNamespace(visited=index < 3) for index in range(4)]
        manager.room_scheduler = SimpleNamespace(
            active_door=None, detector=SimpleNamespace(doors=doors))

        self.assertTrue(
            manager._f3_irreversible_partial_return_allowed())
        manager.floor_number = 2
        self.assertFalse(
            manager._f3_irreversible_partial_return_allowed())
        manager.floor_number = 3
        doors[2].visited = False
        self.assertFalse(
            manager._f3_irreversible_partial_return_allowed())

    def test_terminal_retrace_rearms_one_truth_portal_attempt_per_side(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.enable_terminal_missing_room_retrace = True
        manager.upper_floor_truth_room_fallback_active = True
        manager.truth_room_fallback_retry_limit = 3
        manager.f3_truth_room_fallback_attempts = {
            "near:-1": 3, "near:1": 1, "far:-1": 3, "far:1": 3}
        manager.truth_room_fallback_last_grid_update = {
            key: 100 for key in manager.f3_truth_room_fallback_attempts}
        manager.truth_room_fallback_prescans = {
            "near:-1:2", "far:-1:2", "far:1:2"}
        manager.corridor_station_axis = np.asarray([0.0, 1.0])
        manager.corridor_station_origin = np.asarray([0.0, 0.0])
        manager.corridor_forward_station_sign = 1.0
        manager.corridor_sweep_history = []
        manager.start_time = 0.0
        manager.elapsed = mock.Mock(return_value=400.0)

        manager._record_missing_room_retrace_start((0.0, 35.0, 5.5))

        self.assertEqual(manager.terminal_missing_room_retrace_start_station,
                         35.0)
        self.assertEqual(manager.f3_truth_room_fallback_attempts, {
            "near:-1": 2, "near:1": 1, "far:-1": 2, "far:1": 2})
        self.assertNotIn(
            "near:-1", manager.truth_room_fallback_last_grid_update)
        self.assertFalse(manager.truth_room_fallback_prescans)
        self.assertEqual(
            manager.corridor_sweep_history[-1]["event"],
            "TERMINAL_RETRACE_TRUTH_PORTAL_SINGLE_RETRY_REARMED")

    def test_truth_open_room_disconnect_route_is_bounded_and_furniture_clear(self):
        layout = {
            "floors": [{}, {
                "rooms": [{
                    "id": "floor_1_room_0",
                    "bounds": {"x_min": -9.5, "x_max": -1.1,
                               "y_min": 7.85, "y_max": 21.88},
                    "furniture": [{"id": "rack", "pose": [-5.0, 17.0],
                                   "size": [1.0, 1.0]}],
                }],
            }],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as stream:
            json.dump(layout, stream)
            stream.flush()
            door = SimpleNamespace(
                door_id="estimated_door_02",
                truth_room_id="floor_1_room_0",
                entry_truth_crossing_confirmed=True,
                viewpoint_contract_locked_before_entry=True)
            manager = BaselineExplorationManager.__new__(
                BaselineExplorationManager)
            manager.floor_number = 2
            manager.allow_truth_exploration_when_registration_lost = True
            manager.offline_truth_layout_metadata = stream.name
            manager._offline_truth_layout_cache = None
            manager.room_scheduler = SimpleNamespace(
                active_door=door, active_room_id="room")
            goal = {
                "source": "lightweight_room_semantic",
                "room_role": "G3",
                "estimated_door_id": door.door_id,
                "room_goal_diagnostic": {
                    "truth_layout_centerline_fallback": True,
                    "live_3d_audit_required": True,
                    "two_pose_strategy":
                        "truth_layout_open_centerline_deep_near",
                    "two_pose_obstacle": None,
                },
            }
            route, detail = manager._truth_layout_open_room_segment_clear(
                (-2.1, 14.4), (-7.35, 14.4), goal)
            self.assertIsNotNone(route)
            self.assertEqual(
                detail["reason"],
                "truth_open_room_bounds_and_furniture_clear")
            blocked, blocked_detail = \
                manager._truth_layout_open_room_segment_clear(
                    (-2.1, 17.0), (-7.35, 17.0), goal)
            self.assertIsNone(blocked)
            self.assertEqual(
                blocked_detail["reason"],
                "truth_furniture_footprint_blocked")

    def test_f3_obstacle_gap_stale_2d_endpoint_uses_bounded_truth_chord(self):
        """Run9 G4 is clear in layout truth despite a stale occupied cell."""
        room = {
            "id": "floor_2_room_2", "side": "left",
            "bounds": {"x_min": -9.5, "x_max": -1.1,
                       "y_min": 21.88, "y_max": 35.91},
            "door_pose": [-1.1, 28.895],
            "furniture": [
                {"id": "coffee_table", "pose": [-5.3, 28.895],
                 "size": [0.8, 0.6]},
                {"id": "sofa", "pose": [-3.6, 25.8375],
                 "size": [1.8, 0.8]},
                {"id": "bookshelf", "pose": [-3.6, 31.9525],
                 "size": [0.45, 1.2]},
            ],
        }
        layout = {"floors": [{}, {}, {"rooms": [room]}]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as stream:
            json.dump(layout, stream)
            stream.flush()
            door = SimpleNamespace(
                door_id="estimated_door_04",
                truth_room_id="floor_2_room_2",
                viewpoint_contract="obstacle_front_opposite_sides",
                viewpoint_contract_locked_before_entry=True,
                entry_truth_crossing_confirmed=True)
            manager = BaselineExplorationManager.__new__(
                BaselineExplorationManager)
            manager.floor_number = 3
            manager.allow_truth_exploration_when_registration_lost = True
            manager.offline_truth_layout_metadata = stream.name
            manager._offline_truth_layout_cache = None
            manager.room_scheduler = SimpleNamespace(
                active_door=door, active_room_id="room")
            goal = {
                "source": "lightweight_room_semantic",
                "room_role": "G4",
                "estimated_door_id": door.door_id,
                "room_goal_diagnostic": {
                    "two_pose_strategy":
                        "obstacle_front_gap_opposite_side",
                    "two_pose_obstacle": {"front_edge_depth_m": 3.8},
                },
            }
            route, detail = \
                manager._truth_layout_obstacle_gap_segment_clear(
                    (-3.426, 27.602), (-3.327, 29.513), goal)
            self.assertIsNotNone(route)
            self.assertEqual(
                detail["reason"],
                "truth_obstacle_front_gap_and_furniture_clear")
            self.assertGreaterEqual(detail["length_m"], 1.80)
            same_side, same_side_detail = \
                manager._truth_layout_obstacle_gap_segment_clear(
                    (-3.426, 27.602), (-3.327, 27.20), goal)
            self.assertIsNone(same_side)
            self.assertEqual(
                same_side_detail["reason"],
                "truth_obstacle_gap_contract_not_met")

            # run59's first obstacle viewpoint starts on the doorway
            # centreline.  It is the initial move to one side, not a failed
            # same-side G4 chord, and must be truth-auditable before the stale
            # live stripe can reject it.
            initial_goal = dict(goal, room_role="G3")
            initial, initial_detail = \
                manager._truth_layout_obstacle_gap_segment_clear(
                    (-2.0, 28.90), (-2.40, 27.82), initial_goal)
            self.assertIsNotNone(initial)
            self.assertTrue(initial_detail["initial_side_leg"])

    def test_run63_f3_first_obstacle_g3_is_truth_gap_clear(self):
        """The generated clear first side must survive a stale F3 raster."""
        room = {
            "id": "floor_2_room_1", "side": "right",
            "bounds": {"x_min": 1.1, "x_max": 9.5,
                       "y_min": 7.85, "y_max": 21.88},
            "door_pose": [1.1, 14.865],
            "furniture": [
                {"id": "table", "pose": [5.3, 14.865],
                 "size": [2.2, 1.0]},
                {"id": "chair_west", "pose": [3.15, 14.865],
                 "size": [0.5, 0.5]},
            ],
        }
        layout = {"floors": [{}, {}, {"rooms": [room]}]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as stream:
            json.dump(layout, stream)
            stream.flush()
            door = SimpleNamespace(
                door_id="estimated_door_01",
                truth_room_id="floor_2_room_1",
                viewpoint_contract="obstacle_front_opposite_sides",
                viewpoint_contract_locked_before_entry=True,
                entry_truth_crossing_confirmed=True)
            manager = BaselineExplorationManager.__new__(
                BaselineExplorationManager)
            manager.floor_number = 3
            manager.allow_truth_exploration_when_registration_lost = True
            manager.offline_truth_layout_metadata = stream.name
            manager._offline_truth_layout_cache = None
            manager.room_scheduler = SimpleNamespace(
                active_door=door,
                active_room_id="floor_3_estimated_room_01")
            goal = {
                "source": "lightweight_room_semantic",
                "room_role": "G3",
                "estimated_door_id": door.door_id,
                "room_goal_diagnostic": {
                    "two_pose_strategy":
                        "obstacle_front_gap_first_truth_bounded"},
            }

            route, detail = \
                manager._truth_layout_obstacle_gap_segment_clear(
                    (2.024595245, 14.832604632),
                    (2.40, 15.945), goal)

            self.assertIsNotNone(route, detail)
            self.assertEqual(
                detail["reason"],
                "truth_obstacle_front_gap_and_furniture_clear")
            self.assertTrue(detail["initial_side_leg"])

    def test_run70_f1_rotated_obstacle_gap_uses_world_coordinates(self):
        room = {
            "id": "floor_0_room_1", "side": "right",
            "bounds": {"x_min": 1.1, "x_max": 9.5,
                       "y_min": 7.85, "y_max": 21.88},
            "door_pose": [1.1, 14.865],
            "furniture": [
                {"id": "table", "pose": [5.3, 14.865],
                 "size": [2.2, 1.0]},
                {"id": "chair", "pose": [3.15, 14.865],
                 "size": [0.5, 0.5]},
            ],
        }
        layout = {"floors": [{"rooms": [room]}]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as stream:
            json.dump(layout, stream)
            stream.flush()
            door = SimpleNamespace(
                door_id="estimated_door_01",
                truth_room_id="floor_0_room_1",
                truth_coordinate_transform="map_yaw_plus_90",
                viewpoint_contract="obstacle_front_opposite_sides",
                viewpoint_contract_locked_before_entry=True,
                entry_truth_crossing_confirmed=True)
            manager = BaselineExplorationManager.__new__(
                BaselineExplorationManager)
            manager.floor_number = 1
            manager.offline_truth_layout_metadata = stream.name
            manager._offline_truth_layout_cache = None
            manager.room_scheduler = SimpleNamespace(
                active_door=door, active_room_id="estimated_room_01")
            goal = {
                "source": "lightweight_room_semantic",
                "room_role": "G3",
                "estimated_door_id": door.door_id,
                "room_goal_diagnostic": {
                    "two_pose_strategy": "obstacle_front_gap_first"},
            }

            route, detail = manager._truth_layout_obstacle_gap_segment_clear(
                (14.758, -1.760), (15.945, -2.400), goal)

            self.assertIsNotNone(route, detail)
            self.assertEqual(detail["coordinate_transform"],
                             "map_yaw_plus_90")
            self.assertTrue(detail["initial_side_leg"])
            self.assertAlmostEqual(detail["target_depth_m"], 1.3, places=2)

    def test_f1_rotated_open_room_truth_route_uses_world_coordinates(self):
        layout = {
            "floors": [{
                "rooms": [{
                    "id": "floor_0_room_2",
                    "bounds": {"x_min": -9.5, "x_max": -1.1,
                               "y_min": 21.88, "y_max": 35.91},
                    "furniture": [],
                }],
            }],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as stream:
            json.dump(layout, stream)
            stream.flush()
            door = SimpleNamespace(
                door_id="estimated_door_03",
                truth_room_id="floor_0_room_2",
                truth_coordinate_transform="map_yaw_plus_90",
                entry_truth_crossing_confirmed=True,
                viewpoint_contract_locked_before_entry=True)
            manager = BaselineExplorationManager.__new__(
                BaselineExplorationManager)
            manager.floor_number = 1
            manager.offline_truth_layout_metadata = stream.name
            manager._offline_truth_layout_cache = None
            manager.room_scheduler = SimpleNamespace(
                active_door=door, active_room_id="room")
            goal = {
                "source": "lightweight_room_semantic",
                "room_role": "G3",
                "estimated_door_id": door.door_id,
                "room_goal_diagnostic": {
                    "truth_layout_centerline_fallback": True,
                    "live_3d_audit_required": True,
                    "two_pose_strategy":
                        "truth_layout_open_centerline_deep_near",
                    "two_pose_obstacle": None,
                },
            }

            route, detail = manager._truth_layout_open_room_segment_clear(
                (28.068, 1.736), (27.995, 7.250), goal)

            self.assertIsNotNone(route)
            self.assertEqual(detail["coordinate_transform"],
                             "map_yaw_plus_90")
            self.assertEqual(detail["reason"],
                             "truth_open_room_bounds_and_furniture_clear")

    def test_near_portal_exit_preserves_current_lateral_coordinate(self):
        door = EstimatedDoorway(
            door_id="door", center=(1.16, 28.90),
            normal_direction=0.0, width=1.4,
            left_frame_point=(1.16, 29.60),
            right_frame_point=(1.16, 28.20),
            corridor_side=(0.56, 28.90), interior_side=(2.16, 28.90),
            confidence=1.0, entered_at=0.0)
        target = portal_lateral_preserving_exit(
            door, (1.16, 28.53), 0.60)
        self.assertIsNotNone(target)
        self.assertAlmostEqual(target[0], 0.56, places=6)
        self.assertAlmostEqual(target[1], 28.53, places=6)

    def test_near_door_second_view_reports_defined_planning_clearance(self):
        grid = SimpleNamespace(
            width=1, height=1, resolution=0.1,
            data=np.asarray([[0]], dtype=np.int8),
            cell_to_world=lambda _cell: (1.0, 2.5))
        door = SimpleNamespace(
            depth=lambda point: float(point[0]) + 1.0,
            lateral=lambda point: float(point[1]))
        config = SimpleNamespace(
            goal_clearance=0.25,
            minimum_goal_separation=0.50,
            semantic_completion_tolerance=0.20)
        path = {"success": True, "path": [(0.0, 0.0), (1.0, 2.5)]}
        with mock.patch(
                "lightweight_room_core._clearance", return_value=1.0), \
                mock.patch(
                    "lightweight_room_core.astar_safe_path",
                    return_value=path):
            result = select_near_door_side_second_visual_view(
                grid, door, (0.0, 0.0), [(0.0, 0.0)], {}, config)
        self.assertIsNotNone(result)
        self.assertEqual(result["role"], "G4")
        self.assertAlmostEqual(result["path_planning_clearance_m"], 0.25)

    def test_f3_initial_truth_seed_stops_at_near_portal_station(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.floor_number = 3
        manager.f3_truth_degraded_active = True
        manager.f3_truth_seed_goal_pending = True
        manager.f3_truth_degraded_retry_count = 0
        manager.f3_truth_degraded_retry_limit = 2
        manager.f3_truth_seed_goal_selected_count = 0
        manager.room_scheduler = SimpleNamespace(
            active_door=None, detector=SimpleNamespace(doors=[]))
        manager.lock = threading.RLock()
        manager.pose = (0.0, 13.323, 2.914, math.pi / 2.0)
        manager.grid = object()
        manager.corridor_forward_axis_hint = np.asarray([0.0, 1.0])
        manager.corridor_sweep_history = []
        manager.elapsed = lambda: 1.0
        manager._plan_f3_truth_room_fallback = mock.Mock(return_value=None)

        goal = manager._plan_f3_initial_truth_corridor_seed(1)

        self.assertIsNotNone(goal)
        self.assertAlmostEqual(goal["position"][1], 14.415, places=6)
        self.assertLess(goal["corridor_advance_m"], 1.10)
        clamp = next(
            event for event in manager.corridor_sweep_history
            if event.get("event") ==
            "F3_TRUTH_FIRST_GOAL_CLAMPED_TO_PORTAL_STATION")
        self.assertAlmostEqual(clamp["portal_standoff_m"], 0.45)
        self.assertAlmostEqual(clamp["staged_station"], 14.415)
        self.assertTrue(any(
            event.get("event") ==
            "F3_TRUTH_FIRST_GOAL_CLAMPED_TO_PORTAL_STATION"
            for event in manager.corridor_sweep_history))

    def test_f3_initial_truth_seed_geometry_retries_do_not_repeat_endpoint(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.floor_number = 3
        manager.f3_truth_degraded_active = True
        manager.f3_truth_seed_goal_pending = True
        manager.f3_truth_degraded_retry_limit = 2
        manager.f3_truth_seed_goal_selected_count = 0
        manager.room_scheduler = SimpleNamespace(
            active_door=None, detector=SimpleNamespace(doors=[]))
        manager.lock = threading.RLock()
        manager.pose = (0.0, 13.323, 2.914, math.pi / 2.0)
        manager.grid = object()
        manager.corridor_forward_axis_hint = np.asarray([0.0, 1.0])
        manager.corridor_sweep_history = []
        manager.elapsed = lambda: 1.0
        manager._plan_f3_truth_room_fallback = mock.Mock(return_value=None)

        endpoints = []
        for retry in range(3):
            manager.f3_truth_degraded_retry_count = retry
            endpoints.append(
                manager._plan_f3_initial_truth_corridor_seed(retry + 1)
                ["position"][1])

        self.assertEqual(len(set(round(value, 6) for value in endpoints)), 3)
        self.assertEqual(endpoints, sorted(endpoints, reverse=True))
        self.assertAlmostEqual(endpoints[0], 14.415, places=6)
        self.assertAlmostEqual(endpoints[2], 14.015, places=6)

    def test_f3_initial_truth_seed_gives_portal_first_refusal(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.floor_number = 3
        manager.f3_truth_degraded_active = True
        manager.f3_truth_seed_goal_pending = True
        manager.room_scheduler = SimpleNamespace(
            active_door=None, detector=SimpleNamespace(doors=[]))
        manager.lock = threading.RLock()
        manager.pose = (0.0, 14.865, 2.914, math.pi / 2.0)
        manager.grid = object()
        manager.corridor_sweep_history = []
        manager.elapsed = lambda: 2.0
        portal = {"source": "lightweight_room_entry",
                  "estimated_door_id": "estimated_door_01"}
        manager._plan_f3_truth_room_fallback = mock.Mock(
            return_value=portal)

        self.assertIs(
            manager._plan_f3_initial_truth_corridor_seed(2), portal)
        self.assertEqual(
            manager.corridor_sweep_history[-1]["event"],
            "F3_TRUTH_FIRST_GOAL_PREEMPTED_BY_PORTAL")

    def test_f3_truth_seed_is_not_erased_by_empty_paired_door_queue(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager._plan_pending_paired_opposite_door = mock.Mock(
            return_value=None)
        seed = {"source": "f3_initial_truth_corridor_seed",
                "position": [0.0, 14.865, 2.91]}

        selected = manager._pending_paired_goal_or_existing(seed)

        self.assertIs(selected, seed)
        manager._plan_pending_paired_opposite_door.assert_not_called()

    def test_pending_paired_door_is_considered_when_no_goal_selected(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        pending = {"source": "paired_opposite_recenter"}
        manager._plan_pending_paired_opposite_door = mock.Mock(
            return_value=pending)

        selected = manager._pending_paired_goal_or_existing(None)

        self.assertIs(selected, pending)
        manager._plan_pending_paired_opposite_door.assert_called_once_with()

    def test_finite_stair_workers_are_not_required_but_f3_is_terminal(self):
        launch = ET.parse(os.path.join(
            SCRIPT_DIR, "..", "launch",
            "baseline_fastlio_exploration.launch"))
        nodes = {item.get("name"): item
                 for item in launch.getroot().findall("node")}
        self.assertEqual(
            nodes["stair_transition_manager"].get("required"),
            "$(arg first_floor_stair_required)")
        args = {item.get("name"): item.get("default")
                for item in launch.getroot().findall("arg")}
        self.assertEqual(args["first_floor_stair_required"], "false")
        self.assertEqual(
            nodes["second_to_third_floor_stair_manager"].get("required"),
            "false")
        self.assertEqual(
            nodes["third_floor_exploration_manager"].get("required"),
            "$(eval not arg('enable_third_floor_descent'))")
        f2 = nodes["second_floor_exploration_manager"]
        emergency = next(
            item for item in f2.findall("param")
            if item.get("name") ==
            "emergency_partial_handoff_minimum_exited_rooms")
        self.assertEqual(emergency.get("value"), "3")

    def test_terminal_stair_failure_does_not_wait_for_handoff_timeout(self):
        terminal = SecondFloorMission._terminal_stair_failure_state
        self.assertTrue(terminal("STAIR_FLIGHT_A_ALIGNMENT_LOST"))
        self.assertTrue(terminal("STAIR_FLIGHT_A_RECOVERY_FAILED"))
        self.assertTrue(terminal("STAIR_FLIGHT_B_FALL_DETECTED"))
        self.assertTrue(terminal("STAIR_ASCENT_TIMEOUT"))
        self.assertFalse(terminal("STAIR_POLICY_WARMUP"))
        self.assertFalse(terminal("STAIR_LANDING_RECOVERY"))
        self.assertFalse(terminal("THIRD_FLOOR_REACHED"))

    def test_bounded_one_room_emergency_continues_without_strict_credit(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        door = SimpleNamespace(visited=True, completed=True)
        manager.room_scheduler = SimpleNamespace(
            active_door=None,
            detector=SimpleNamespace(doors=[door]))
        manager.require_all_rooms_for_floor_handoff = True
        manager.allow_partial_floor_handoff = True
        manager.room_target_count = 4
        manager.partial_floor_handoff_minimum_exited_rooms = 3
        manager.emergency_partial_floor_handoff_minimum_exited_rooms = 1
        manager.emergency_partial_handoff_armed = True
        manager.corridor_established = True
        manager.lock = threading.RLock()
        manager.pose = (0.7, 11.6, 0.31, 0.0)
        pending = SimpleNamespace(visited=False, completed=False)
        manager._known_unvisited_room_doors = mock.Mock(
            return_value=[pending])

        # Strict scoring remains false, while the separately armed bounded
        # continuation gate allows the physical stair route to proceed.
        self.assertFalse(manager._floor_handoff_room_requirement_met())
        with mock.patch.object(
                baseline_manager_module.rospy, "get_param",
                return_value=0.0):
            self.assertTrue(manager._multi_floor_truth_handoff_safe())

        self.assertTrue(SecondFloorMission._exploration_handoff_authorized(
            "STAIR_CORRIDOR_EXIT_HANDOFF_TRUTH_REQUESTED",
            exited_room_count=1,
            room_target_count=4,
            terminal_return_latched=True,
            allow_partial_floor_handoff=True,
            partial_minimum_exited_rooms=3,
            emergency_minimum_exited_rooms=1))

        # The same pending portal must still block an ordinary (not bounded)
        # partial transfer.  Only the explicitly armed emergency path is
        # independent from strict room completion.
        manager.emergency_partial_handoff_armed = False
        with mock.patch.object(
                baseline_manager_module.rospy, "get_param",
                return_value=0.0):
            self.assertFalse(manager._multi_floor_truth_handoff_safe())

    def test_bounded_emergency_arms_transit_without_strict_credit(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.emergency_partial_handoff_armed = True
        manager.stair_return_transit_announced = False
        manager.stair_return_transit_pub = mock.Mock()
        manager._set_state = mock.Mock()
        manager._floor_handoff_room_requirement_met = mock.Mock(
            return_value=False)

        self.assertIsNone(manager._announce_stair_return_transit(
            "bounded_emergency"))
        self.assertTrue(manager.stair_return_transit_announced)
        manager.stair_return_transit_pub.publish.assert_called_once()
        manager._set_state.assert_called_once_with(
            "STAIR_RETURN_TRANSIT", "bounded_emergency")

    def test_finalize_delivery_race_accepts_late_transit_and_lobby(self):
        transition = StairTransition.__new__(StairTransition)
        transition.phase = "WAIT_F1"
        transition.handoff_seen = False
        transition.state = mock.Mock()
        transition.finalization_deadline = None
        transition.require_return_transit_arm_for_state_handoff = True
        transition.return_transit_armed = False
        transition.pending_state_handoff = False
        transition._maybe_publish_truth_return_gate = mock.Mock()
        transition._begin_handoff = mock.Mock()

        with mock.patch(
                "stair_transition_manager.rospy.get_param",
                return_value=2.0), mock.patch(
                    "stair_transition_manager.time.monotonic",
                    return_value=10.0):
            transition.on_finalize(SimpleNamespace(data=True))
        self.assertEqual(transition.phase, "FIRST_FLOOR_FINALIZING")
        self.assertEqual(transition.finalization_deadline, 12.0)

        with mock.patch(
                "stair_transition_manager.rospy.logwarn_throttle"):
            transition.on_state(SimpleNamespace(
                data="STAIR_LOBBY_HANDOFF"))
        self.assertEqual(transition.phase, "WAIT_F1")
        self.assertTrue(transition.pending_state_handoff)
        transition.on_return_transit_armed(SimpleNamespace(data=True))
        self.assertTrue(transition.return_transit_armed)
        transition._begin_handoff.assert_called_once_with()

    def test_f3_active_room_release_is_continuity_not_strict_credit(self):
        source = inspect.getsource(SecondFloorMission._run_mission)
        self.assertIn(
            "f3_return_continuity_after_unconfirmed_exit", source)
        self.assertIn("_truth_return_to_corridor_anchor", source)
        self.assertIn("_truth_reposition_to_corridor", source)
        release = source[source.index(
            "f3_return_continuity_after_unconfirmed_exit") - 1200:
            source.index("exited_room_count = sum(")]
        self.assertNotIn(".visited = True", release)
        self.assertNotIn(".completed = True", release)

    def test_no_valid_frontier_is_a_bounded_continuity_terminal(self):
        source = inspect.getsource(BaselineExplorationManager.run)
        self.assertGreaterEqual(
            source.count('"NO_VALID_FRONTIER_CONFIRMED"'), 4)
        wrapper = inspect.getsource(SecondFloorMission._run_mission)
        effective = wrapper.index("effective_emergency_minimum = min(")
        fallback = wrapper.index("# Flow-continuity recovery")
        self.assertLess(effective, fallback)

    def test_g2_missing_rooms_is_a_bounded_continuity_terminal(self):
        """A measured G2 return must continue floors without strict credit."""
        source = inspect.getsource(BaselineExplorationManager.run)
        self.assertIn('"G2_REACHED_WITH_MISSING_ROOMS"', source)
        start = source.index("bounded_handoff_reasons = (")
        end = source.index("elif self.termination_reason == \"TIME_LIMIT\"")
        epilogue = source[start:end]
        self.assertIn('"G2_REACHED_WITH_MISSING_ROOMS"', epilogue)
        self.assertIn(
            '"G2_RETURN_REACHED_WITH_STRICT_ROOMS_MISSING"', epilogue)
        # Continuity rewrites only the terminal token; it must not fabricate
        # visited/completed room evidence for either partial room.
        self.assertNotIn(".visited = True", epilogue)
        self.assertNotIn(".completed = True", epilogue)

    def test_all_direct_room_dispatches_lock_contract_before_entry(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.offline_truth_layout_metadata = "layout.json"
        door = EstimatedDoorway(
            door_id="direct_upper_door", center=(1.1, 28.895),
            normal_direction=math.pi, width=1.2,
            left_frame_point=(1.1, 28.295),
            right_frame_point=(1.1, 29.495),
            corridor_side=(0.5, 28.895), interior_side=(2.1, 28.895),
            confidence=0.9, entered_at=1.0)
        events = []

        def assign_contract(active):
            active.viewpoint_contract = "open_deep_near"
            active.viewpoint_contract_source = "test_truth_geometry"
            active.truth_room_id = "floor_2_room_3"

        def next_goal(_grid, _current, _now):
            self.assertTrue(door.viewpoint_contract_locked_before_entry)
            return {"room_role": "ENTRY"}

        manager._assign_truth_viewpoint_contract = assign_contract
        manager.room_scheduler = SimpleNamespace(
            active_door=door, entry_confirmed=False, events=events,
            _room_id=lambda: "room", abort_active_room=lambda *_: None,
            next_goal=next_goal)

        goal = manager._next_room_goal_with_contract(None, (0.0, 0.0), 2.0)

        self.assertEqual(goal["room_role"], "ENTRY")
        self.assertTrue(any(event.get("event") ==
                            "ROOM_VIEWPOINT_CONTRACT_LOCKED_BEFORE_ENTRY"
                            for event in events))

    def test_f3_truth_portal_fallback_is_available_from_zero_rooms(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.upper_floor_truth_room_fallback_active = True
        manager.floor_number = 3
        manager.f3_truth_degraded_active = False
        manager.room_scheduler = SimpleNamespace(active_door=None)
        manager.corridor_axis = np.asarray([0.0, 1.0])
        manager.truth_room_fallback_station_tolerance = 1.15
        manager.truth_room_fallback_retry_limit = 3
        manager.grid_update_count = 1
        retry_gate = mock.Mock(return_value=False)
        manager._truth_room_fallback_retry_allowed = retry_gate

        self.assertIsNone(manager._plan_f3_truth_room_fallback(
            (0.0, 0.0, 5.58, 0.0), object()))
        retry_gate.assert_called_once()

    def test_terminal_opposite_fourth_room_stages_before_expensive_search(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.upper_floor_truth_room_fallback_active = True
        manager.floor_number = 2
        manager.offline_truth_layout_metadata = ""
        manager.corridor_established = True
        manager.corridor_axis = np.asarray([0.0, 1.0])
        manager.corridor_station_axis = np.asarray([0.0, 1.0])
        manager.corridor_station_origin = np.asarray([0.0, 0.0])
        manager.truth_room_fallback_station_tolerance = 1.15
        manager.truth_room_fallback_retry_limit = 3
        manager.grid_update_count = 10
        manager.f3_truth_room_fallback_attempts = {}
        manager.truth_room_fallback_last_grid_update = {}
        manager.truth_room_fallback_prescans = set()
        manager.room_target_count = 4
        manager.room_camera_sweep_speed = 2.1
        manager.lock = threading.RLock()
        manager.grid = object()
        manager.corridor_sweep_history = []
        manager.elapsed = lambda: 170.0
        manager._perform_local_rescan = lambda **_kwargs: True
        manager._recent_room_exit_opposite_side = lambda _pose: -1
        manager._truth_room_fallback_retry_allowed = \
            lambda *_args: True
        prepared_search = mock.Mock(side_effect=AssertionError(
            "generic rolling-grid search must not run first"))
        staging = mock.Mock(side_effect=lambda _g, door, _c, _n: door)
        manager.room_scheduler = SimpleNamespace(
            active_door=None,
            detector=SimpleNamespace(doors=[
                SimpleNamespace(visited=True) for _ in range(3)]),
            activate_truth_portal_staging=staging,
            _prepare_and_activate=prepared_search,
            _clear_active=lambda: None,
        )
        manager._next_room_goal_with_contract = \
            lambda _g, _c, _n: {
                "position": [-1.95, 28.895, 0.0],
                "room_role": "ENTRY",
            }

        goal = manager._plan_f3_truth_room_fallback(
            (0.0, 28.90, 2.91, 0.0), object())

        self.assertIsNotNone(goal)
        self.assertEqual(
            goal["door_takeover_source"],
            "terminal_opposite_truth_staging_fast_path")
        staging.assert_called_once()
        prepared_search.assert_not_called()
        self.assertTrue(any(
            event.get("event") ==
            "TERMINAL_OPPOSITE_TRUTH_STAGING_FAST_PATH"
            for event in manager.corridor_sweep_history))

    def test_mirrored_pair_uses_corridor_center_not_approach_anchor(self):
        door = EstimatedDoorway(
            door_id="estimated_door_right", center=(1.189, 14.865),
            normal_direction=0.0, width=1.4,
            left_frame_point=(1.189, 15.565),
            right_frame_point=(1.189, 14.165),
            corridor_side=(0.589, 14.865),
            interior_side=(2.689, 14.865), confidence=0.9,
            entered_at=1.0)
        centre = BaselineExplorationManager._inferred_corridor_centerline_from_door(
            door)
        self.assertAlmostEqual(float(centre[0]), -0.011, places=3)
        self.assertAlmostEqual(float(centre[1]), 14.865, places=3)

    def test_f2_room2_truth_contract_is_open_deep_near(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.floor_number = 2
        manager.offline_truth_layout_metadata = os.path.abspath(os.path.join(
            SCRIPT_DIR, "..", "..", "..", "generated_building",
            "layout_metadata.json"))
        manager._offline_truth_layout_cache = None
        manager.room_scheduler = SimpleNamespace(events=[])
        manager.elapsed = lambda: 1.0
        door = EstimatedDoorway(
            door_id="estimated_f2_room2", center=(-1.10, 28.895),
            normal_direction=math.pi, width=1.2,
            left_frame_point=(-1.10, 28.295),
            right_frame_point=(-1.10, 29.495),
            corridor_side=(-0.50, 28.895),
            interior_side=(-2.10, 28.895), confidence=0.9,
            entered_at=1.0)

        contract = manager._assign_truth_viewpoint_contract(door)

        self.assertEqual(contract, "open_deep_near")
        self.assertEqual(door.truth_room_id, "floor_1_room_2")

    def test_f1_rotated_map_door_gets_obstacle_gap_contract(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.floor_number = 1
        manager.offline_truth_layout_metadata = os.path.abspath(os.path.join(
            SCRIPT_DIR, "..", "..", "..", "generated_building",
            "layout_metadata.json"))
        manager._offline_truth_layout_cache = None
        manager.room_scheduler = SimpleNamespace(events=[])
        manager.elapsed = lambda: 1.0
        # world door (-1.10, 14.865) appears as camera_init
        # (14.865, +1.10) on F1.
        door = EstimatedDoorway(
            door_id="estimated_f1_room0", center=(14.865, 1.10),
            normal_direction=math.pi / 2.0, width=1.2,
            left_frame_point=(14.265, 1.10),
            right_frame_point=(15.465, 1.10),
            corridor_side=(14.865, 0.50),
            interior_side=(14.865, 2.10), confidence=0.9,
            entered_at=1.0)

        contract = manager._assign_truth_viewpoint_contract(door)

        self.assertEqual(contract, "obstacle_front_opposite_sides")
        self.assertEqual(door.truth_room_id, "floor_0_room_0")
        self.assertEqual(door.truth_coordinate_transform,
                         "map_yaw_plus_90")
        self.assertAlmostEqual(door.truth_obstacle_front_depth_m, 3.8)

    def test_f3_deep_obstacle_contract_clears_secondary_side_occluders(self):
        """run87 D83: side peeks must clear sofa/bookshelf silhouettes."""
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.floor_number = 3
        manager.offline_truth_layout_metadata = os.path.abspath(os.path.join(
            SCRIPT_DIR, "..", "..", "..", "generated_building",
            "elevator_three_floor_official_count",
            "layout_metadata.json"))
        manager._offline_truth_layout_cache = None
        manager.room_scheduler = SimpleNamespace(events=[])
        manager.elapsed = lambda: 1.0
        door = EstimatedDoorway(
            door_id="estimated_f3_room2", center=(-1.10, 28.895),
            normal_direction=math.pi, width=1.4,
            left_frame_point=(-1.10, 28.195),
            right_frame_point=(-1.10, 29.595),
            corridor_side=(-0.50, 28.895),
            interior_side=(-2.10, 28.895), confidence=0.9,
            entered_at=1.0)

        contract = manager._assign_truth_viewpoint_contract(door)

        self.assertEqual(contract, "obstacle_front_opposite_sides")
        self.assertEqual(door.truth_room_id, "floor_2_room_2")
        self.assertEqual(door.truth_obstacle_view_policy,
                         "deep_outer_side_peek")
        self.assertLessEqual(
            door.truth_obstacle_visibility_minimum_lateral_m, -3.65)
        self.assertGreaterEqual(
            door.truth_obstacle_visibility_maximum_lateral_m, 3.45)
        event = manager.room_scheduler.events[-1]
        self.assertTrue(event["visibility_occluders"])

    def test_f1_long_meeting_table_requires_deep_side_peeks(self):
        """run153 D6: a long table is not a shallow near-door chord."""
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.floor_number = 1
        manager.offline_truth_layout_metadata = os.path.abspath(os.path.join(
            SCRIPT_DIR, "..", "..", "..", "generated_building",
            "elevator_three_floor_official_count",
            "layout_metadata.json"))
        manager._offline_truth_layout_cache = None
        manager.room_scheduler = SimpleNamespace(events=[])
        manager.elapsed = lambda: 1.0
        # floor_0_room_1 world door (1.10, 14.865) appears as
        # camera_init (14.865, -1.10) on F1.
        door = EstimatedDoorway(
            door_id="estimated_f1_meeting_room", center=(14.865, -1.10),
            normal_direction=-math.pi / 2.0, width=1.4,
            left_frame_point=(15.565, -1.10),
            right_frame_point=(14.165, -1.10),
            corridor_side=(14.865, -0.50),
            interior_side=(14.865, -2.10), confidence=0.9,
            entered_at=1.0)

        contract = manager._assign_truth_viewpoint_contract(door)

        self.assertEqual(contract, "obstacle_front_opposite_sides")
        self.assertEqual(door.truth_room_id, "floor_0_room_1")
        self.assertAlmostEqual(door.truth_obstacle_front_depth_m, 3.1)
        self.assertAlmostEqual(door.truth_obstacle_rear_depth_m, 5.3)
        self.assertEqual(
            door.truth_obstacle_view_policy, "deep_outer_side_peek")

    def test_completed_truth_portal_is_skipped_before_prescan(self):
        source = inspect.getsource(
            BaselineExplorationManager._plan_f3_truth_room_fallback)
        completed_gate = source.index("completed_contract_match = None")
        prescan = source.index("prescan_key =")
        self.assertLess(completed_gate, prescan)
        self.assertIn(
            "COMPLETED_TRUTH_PORTAL_PRESCAN_SKIPPED",
            source[completed_gate:prescan])
        self.assertIn("room_transaction_complete",
                      source[completed_gate:prescan])

    def test_f3_truth_portal_retry_waits_for_station_and_fresh_map(self):
        allowed = BaselineExplorationManager._truth_room_fallback_retry_allowed
        # The corridor ingress at y=13.31 must not consume a near-door retry.
        self.assertFalse(allowed(13.31, 14.865, 1.15, 0, 3, 100, -1000000))
        # Once physically beside the door, one fresh-map attempt is allowed.
        self.assertTrue(allowed(14.20, 14.865, 1.15, 0, 3, 100, 97))
        # A rejection cannot spin on the same raster generation.
        self.assertFalse(allowed(14.20, 14.865, 1.15, 1, 3, 101, 100))
        self.assertTrue(allowed(14.20, 14.865, 1.15, 1, 3, 102, 100))
        self.assertFalse(allowed(14.20, 14.865, 1.15, 3, 3, 104, 100))
        # run35's final generated-corridor probe stopped at y=27.31, 1.585 m
        # before the far portal.  The next cycle falsely latched the terminal
        # wall before it could clamp a further probe to the doorway station.
        # The wider observation band only authorizes live portal preflight;
        # ENTRY still requires A*/SCAN-lite and a physical plane crossing.
        self.assertTrue(allowed(27.31, 28.895, 1.75,
                                0, 3, 200, -1000000))

    def test_truth_portal_axis_is_canonical_when_sweep_reverses(self):
        axis = BaselineExplorationManager._canonical_truth_portal_axis(
            [0.0004, -0.9999999])
        station = float(np.dot(np.asarray([0.0, 14.865]), axis))
        self.assertAlmostEqual(station, 14.865, places=3)

    def test_generated_corridor_probe_keeps_advancing_past_midpoint(self):
        advances = BaselineExplorationManager._truth_probe_advances_generated_corridor
        self.assertTrue(advances((0.0, 22.0), (0.0, 25.0)))
        self.assertTrue(advances((0.0, 25.0), (0.0, 28.0)))
        self.assertFalse(advances((0.0, 25.0), (0.0, 22.0)))

    def test_generated_corridor_missing_room_retrace_moves_toward_lobby(self):
        retraces = BaselineExplorationManager._truth_probe_retraces_generated_corridor
        # run55 reached the real F3 end at y=34.79, but the bounded reverse
        # target at y=31.79 was incorrectly rejected as a forward-wall probe.
        self.assertTrue(retraces((0.0, 34.79), (0.0, 31.79)))
        self.assertFalse(retraces((0.0, 31.79), (0.0, 34.79)))

    def test_sparse_truth_portal_rejections_select_corridor_probe(self):
        recoverable = BaselineExplorationManager._truth_portal_probe_rejection_recoverable
        self.assertTrue(recoverable("candidate_no_observed_free_anchor"))
        self.assertTrue(recoverable("no_safe_entry_candidate"))
        self.assertFalse(recoverable("inward_ray_overlaps_visited_room"))

    def test_truth_portal_probe_stays_on_corridor_side(self):
        grid = SimpleNamespace()
        with mock.patch.object(
                baseline_manager_module, "astar_safe_path",
                return_value={"success": True,
                              "path": [(0.60, 14.86),
                                       (0.00, 14.86),
                                       (-0.59, 14.86)]}):
            goal = BaselineExplorationManager._truth_portal_probe_goal(
                grid, (0.60, 14.86), (-0.59, 14.86), 2.91)
        self.assertIsNotNone(goal)
        self.assertEqual(goal["source"],
                         "upper_floor_truth_portal_corridor_probe")
        self.assertEqual(goal["position"], [-0.59, 14.86, 2.91])
        self.assertEqual(goal["scheduler_phase"], "CORRIDOR_SWEEP")

    def test_fullflow_f2_to_f3_default_is_not_approach_only(self):
        launch = ET.parse(os.path.join(
            SCRIPT_DIR, "..", "launch",
            "fuel_semantic_fastlio_exploration.launch"))
        args = {item.get("name"): item.get("default")
                for item in launch.getroot().findall("arg")}
        self.assertEqual(args["second_to_third_stair_approach_only"],
                         "false")

    @staticmethod
    def _upper_landing_transition(pose, linear=(0.0, 0.0, 0.0),
                                  angular=(0.0, 0.0, 0.0)):
        transition = StairTransition.__new__(StairTransition)
        transition.truth_pose = pose
        transition.source_floor_index = 1
        transition.truth_flight_b_pose = (-2.485, 4.520, 0.0, 0.0)
        transition.truth_flight_b_next_pose = (-2.485, 4.260, 0.0, 0.0)
        transition.truth_flight_b_heading = -np.pi / 2.0
        transition.truth_second_floor_clearance = 0.30
        transition.truth_upper_landing_clearance_tolerance = 0.0
        transition.truth_upper_landing_center_tolerance = 0.35
        transition.truth_upper_landing_heading_tolerance = 0.35
        transition.truth_upper_landing_linear_speed = 0.20
        transition.truth_upper_landing_angular_speed = 0.35
        transition.truth_upper_landing_attitude_recovery_enabled = True
        transition.truth_upper_landing_attitude_recovered = False
        transition.truth_upper_landing_clearance_recovery_enabled = True
        transition.truth_upper_landing_clearance_recovered = False
        transition.truth_upper_landing_clearance_recovery_started = None
        transition.truth_upper_landing_clearance_recovery_shortfall = 0.35
        transition.truth_upper_landing_clearance_recovery_stall = 3.0
        transition.truth_upper_landing_clearance_recovery_planar_speed = 0.12
        transition.truth_f2_f3_landing_recovery_margin = 0.50
        transition.total_height_gain = 2.05
        transition.truth_model_pose = SimpleNamespace(position=SimpleNamespace(
            x=pose[0], y=pose[1], z=pose[2]))
        transition.truth_model_twist = SimpleNamespace(
            linear=SimpleNamespace(x=linear[0], y=linear[1], z=linear[2]),
            angular=SimpleNamespace(x=angular[0], y=angular[1], z=angular[2]))
        return transition

    def test_fix18_negative_final_tread_margin_cannot_handoff(self):
        transition = self._upper_landing_transition(
            (-2.485, 1.8014, 5.374, -np.pi / 2.0))
        cleared, margin = transition._truth_upper_landing_clearance()
        self.assertAlmostEqual(margin, -0.0514, places=4)
        self.assertFalse(cleared)

    def test_fix20_positive_clearance_bad_yaw_gets_one_safe_recovery(self):
        transition = self._upper_landing_transition(
            (-2.463, 1.200, 5.401, 2.585))
        cleared, margin = transition._truth_upper_landing_clearance()
        self.assertGreater(margin, 0.50)
        self.assertTrue(cleared)
        self.assertTrue(transition._truth_upper_landing_centered())
        self.assertFalse(transition._truth_upper_landing_aligned())
        transition.truth_model_pose.orientation = SimpleNamespace(
            x=0.0, y=0.0, z=np.sin(-np.pi / 4.0),
            w=np.cos(-np.pi / 4.0))
        response = SimpleNamespace(success=True)
        with mock.patch(
                "stair_transition_manager.rospy.wait_for_service"), mock.patch(
                "stair_transition_manager.rospy.ServiceProxy",
                return_value=mock.Mock(return_value=response)) as proxy:
            self.assertTrue(transition._recover_truth_upper_landing_attitude(
                2.10, cleared))
        state = proxy.return_value.call_args.args[0]
        self.assertAlmostEqual(state.pose.orientation.z,
                               np.sin(-np.pi / 4.0), places=6)
        self.assertAlmostEqual(state.pose.orientation.w,
                               np.cos(-np.pi / 4.0), places=6)
        self.assertTrue(transition.truth_upper_landing_attitude_recovered)
        self.assertFalse(transition._recover_truth_upper_landing_attitude(
            2.10, cleared))

    def test_plane_handoff_preserves_stable_flight_b_heading(self):
        transition = self._upper_landing_transition(
            (-2.485, 1.70, 5.35, -np.pi / 2.0))
        transition.truth_model_pose.orientation = SimpleNamespace(
            x=0.0, y=0.0, z=np.sin(-np.pi / 4.0),
            w=np.cos(-np.pi / 4.0))
        with mock.patch(
                "stair_transition_manager.rospy.wait_for_service"), mock.patch(
                "stair_transition_manager.rospy.ServiceProxy") as proxy:
            self.assertTrue(transition._align_truth_upper_landing_exit())
        proxy.assert_not_called()
        self.assertAlmostEqual(transition.truth_model_pose.orientation.z,
                               np.sin(-np.pi / 4.0), places=6)
        self.assertAlmostEqual(transition.truth_model_pose.orientation.w,
                               np.cos(-np.pi / 4.0), places=6)

    def test_run53_small_stalled_shortfall_gets_positive_clearance_recheck(self):
        transition = self._upper_landing_transition(
            (-2.485, 1.82, 5.35, -np.pi / 2.0))
        cleared, margin = transition._truth_upper_landing_clearance()
        self.assertFalse(cleared)
        self.assertGreater(margin, -0.15)
        self.assertFalse(transition._recover_truth_upper_landing_clearance(
            2.44, cleared, margin, now=10.0))
        response = SimpleNamespace(success=True)
        with mock.patch(
                "stair_transition_manager.rospy.wait_for_service"), mock.patch(
                "stair_transition_manager.rospy.ServiceProxy",
                return_value=mock.Mock(return_value=response)) as proxy:
            self.assertTrue(transition._recover_truth_upper_landing_clearance(
                2.44, cleared, margin, now=13.1))
        state = proxy.return_value.call_args.args[0]
        self.assertLess(state.pose.position.y, 1.751)
        self.assertTrue(transition.truth_upper_landing_clearance_recovered)

    def test_vertical_gait_bounce_does_not_reset_terminal_recovery(self):
        transition = self._upper_landing_transition(
            (-2.485, 1.82, 5.35, -np.pi / 2.0),
            linear=(0.03, 0.04, 0.50))
        cleared, margin = transition._truth_upper_landing_clearance()
        self.assertFalse(transition._recover_truth_upper_landing_clearance(
            2.44, cleared, margin, now=10.0))
        self.assertEqual(
            transition.truth_upper_landing_clearance_recovery_started, 10.0)

    def test_large_final_tread_shortfall_cannot_use_clearance_recovery(self):
        transition = self._upper_landing_transition(
            (-2.485, 2.05, 5.35, -np.pi / 2.0))
        cleared, margin = transition._truth_upper_landing_clearance()
        self.assertLess(margin, -0.15)
        with mock.patch("stair_transition_manager.rospy.ServiceProxy") as proxy:
            self.assertFalse(transition._recover_truth_upper_landing_clearance(
                2.44, cleared, margin, now=20.0))
        proxy.assert_not_called()

    def test_negative_clearance_never_allows_attitude_recovery(self):
        transition = self._upper_landing_transition(
            (-2.485, 1.8014, 5.374, 2.585))
        cleared, _ = transition._truth_upper_landing_clearance()
        with mock.patch(
                "stair_transition_manager.rospy.ServiceProxy") as proxy:
            self.assertFalse(transition._recover_truth_upper_landing_attitude(
                2.10, cleared))
        proxy.assert_not_called()

    def test_upper_landing_rejects_displaced_or_moving_pose(self):
        displaced = self._upper_landing_transition(
            (-3.622, 1.70, 5.374, 0.023))
        self.assertFalse(displaced._truth_upper_landing_stable())
        moving = self._upper_landing_transition(
            (-2.485, 1.70, 5.374, -np.pi / 2.0), linear=(0.25, 0.0, 0.0))
        self.assertFalse(moving._truth_upper_landing_stable())
        stable = self._upper_landing_transition(
            (-2.485, 1.70, 5.374, -np.pi / 2.0))
        self.assertTrue(stable._truth_upper_landing_stable())

    def test_f2_f3_handoff_rechecks_all_landing_safety_predicates(self):
        unsafe = (
            self._upper_landing_transition(
                (-2.485, 1.8014, 5.374, -np.pi / 2.0)),
            self._upper_landing_transition(
                (-3.200, 1.200, 5.374, -np.pi / 2.0)),
            self._upper_landing_transition(
                (-2.485, 1.200, 5.374, -np.pi / 2.0),
                linear=(0.40, 0.0, 0.0)),
        )
        for transition in unsafe:
            transition.cmd = SimpleNamespace(publish=lambda _message: None)
            transition.truth_f2_f3_handoff_stable_since = 0.0
            transition.truth_f2_f3_handoff_stable_seconds = 0.0
            transition._publish_second_floor_reached = mock.Mock()
            self.assertFalse(transition._handoff_f2_f3_interior_landing(
                2.44, 0.50, already_pinned=True))
            transition._publish_second_floor_reached.assert_not_called()

    def test_f2_f3_handoff_requires_continuous_safe_dwell(self):
        transition = self._upper_landing_transition(
            (-2.485, 1.200, 5.374, -np.pi / 2.0))
        transition.cmd = SimpleNamespace(publish=lambda _message: None)
        transition.truth_f2_f3_handoff_stable_since = None
        transition.truth_f2_f3_handoff_stable_seconds = 3.0
        transition._record_trace = lambda force=False: None
        transition._publish_second_floor_reached = mock.Mock()
        with mock.patch("stair_transition_manager.time.monotonic",
                        return_value=10.0):
            self.assertFalse(transition._handoff_f2_f3_interior_landing(
                2.44, 0.50, already_pinned=True))
        transition._publish_second_floor_reached.assert_not_called()
        transition.truth_f2_f3_handoff_stable_since = 5.0
        with mock.patch("stair_transition_manager.time.monotonic",
                        return_value=10.0):
            self.assertTrue(transition._handoff_f2_f3_interior_landing(
                2.44, 0.50, already_pinned=True))
        transition._publish_second_floor_reached.assert_called_once_with(2.44)


    def test_f1_f2_landing_uses_yaw_rate_and_tilt_not_rl_joint_spin(self):
        transition = self._upper_landing_transition(
            (-2.485, 1.70, 2.91, -np.pi / 2.0),
            angular=(1.20, 0.80, 0.02))
        transition.source_floor_index = 0
        transition.truth_f1_f2_landing_tilt_tolerance = 0.30
        transition.truth_model_pose.orientation = SimpleNamespace(
            x=0.0, y=0.0, z=np.sin(-np.pi / 4.0),
            w=np.cos(-np.pi / 4.0))
        self.assertTrue(transition._truth_upper_landing_motion_settled())

        transition.truth_model_pose.orientation = SimpleNamespace(
            x=np.sin(0.36 / 2.0), y=0.0, z=0.0,
            w=np.cos(0.36 / 2.0))
        self.assertFalse(transition._truth_upper_landing_motion_settled())

    def test_goal_executor_relinquishes_cmd_vel_during_stairs(self):
        active_states = (
            "TRUTH_ENTRY_GUIDE", "STAIR_ASCENT_B",
            "STAIR_FLIGHT_B_STALL_RECOVERY", "THIRD_FLOOR_REACHED")
        for state in active_states:
            self.assertTrue(GoalExecutor._stair_state_owns_command(state))
        released_states = (
            "", "WAIT_F1", "FIRST_FLOOR_FINALIZING",
            "STAIR_POLICY_READY",
            "SECOND_FLOOR_HANDOFF_COMPLETE", "STAIR_ASCENT_TIMEOUT",
            "THIRD_FLOOR_EXPLORATION_START")
        for state in released_states:
            self.assertFalse(GoalExecutor._stair_state_owns_command(state))

    def test_third_floor_exploration_releases_only_upper_stair_latch(self):
        executor = GoalExecutor.__new__(GoalExecutor)
        executor._lock = threading.RLock()
        executor._stair_command_states = {
            "/simenv/stair_transition_state":
                "SECOND_FLOOR_HANDOFF_COMPLETE",
            "/simenv/second_to_third_floor_stair_state":
                "THIRD_FLOOR_REACHED",
        }
        executor._stair_command_owned = True
        executor._events = []
        executor._last_command = [0.2, 0.0, 0.0]
        executor._on_third_floor_state(SimpleNamespace(
            data="THIRD_FLOOR_EXPLORATION_START"))
        self.assertFalse(executor._stair_command_owned)
        self.assertEqual(
            executor._stair_command_states[
                "/simenv/second_to_third_floor_stair_state"],
            "THIRD_FLOOR_EXPLORATION_START")

    def test_run113_future_stair_readiness_does_not_block_second_floor(self):
        executor = GoalExecutor.__new__(GoalExecutor)
        executor._lock = threading.RLock()
        executor._stair_command_states = {}
        executor._stair_command_owned = False
        executor._events = []
        executor._last_command = [0.0, 0.0, 0.0]
        executor._on_stair_command_state(
            SimpleNamespace(data="TRUTH_ENTRY_GUIDE"),
            "/simenv/stair_transition_state")
        executor._on_stair_command_state(
            SimpleNamespace(data="STAIR_POLICY_READY"),
            "/simenv/second_to_third_floor_stair_state")
        self.assertTrue(executor._stair_command_owned)
        executor._on_stair_command_state(
            SimpleNamespace(data="SECOND_FLOOR_HANDOFF_COMPLETE"),
            "/simenv/stair_transition_state")
        self.assertFalse(executor._stair_command_owned)

    def test_waiting_stair_manager_publishes_wait_phase_on_global_policy_ack(self):
        transition = StairTransition.__new__(StairTransition)
        transition.policy = "/tmp/policy_act_inference_stair.pt"
        transition.policy_loaded = False
        transition.phase = "WAIT_F1"
        transition.state = mock.Mock()
        transition.on_policy_status(SimpleNamespace(
            data="policy_reloaded:policy_act_inference_stair.pt"))
        self.assertTrue(transition.policy_loaded)
        published = transition.state.publish.call_args.args[0]
        self.assertEqual(published.data, "WAIT_F1")

    def test_upper_floor_ready_edge_is_latched_until_reached(self):
        transition = StairTransition.__new__(StairTransition)
        transition.upper_floor_ready_token = "SECOND_FLOOR_EXPLORATION_READY"
        transition.upper_floor_recovery_token = "SECOND_FLOOR_STAND_RECOVERY"
        transition.upper_floor_reached_phase = "SECOND_FLOOR_REACHED"
        transition.upper_floor_handoff_complete_phase = \
            "SECOND_FLOOR_HANDOFF_COMPLETE"
        transition.upper_floor_ready_pending = False
        transition.upper_floor_recovery_pending = False
        transition.upper_floor_recovery_requested = False
        transition.phase = "STAIR_LANDING_TURN"
        transition.state = mock.Mock()
        transition.cmd = mock.Mock()
        transition.source_floor_index = 0

        transition.on_second_floor_state(SimpleNamespace(
            data="SECOND_FLOOR_EXPLORATION_READY"))
        self.assertTrue(transition.upper_floor_ready_pending)
        self.assertEqual(transition.phase, "STAIR_LANDING_TURN")

        transition.phase = "SECOND_FLOOR_REACHED"
        self.assertTrue(transition._complete_upper_floor_handoff())
        self.assertEqual(transition.phase,
                         "SECOND_FLOOR_HANDOFF_COMPLETE")

    def test_goal_executor_does_not_publish_zero_while_stair_owns_command(self):
        executor = GoalExecutor.__new__(GoalExecutor)
        executor._lock = threading.RLock()
        executor._stair_command_states = {}
        executor._stair_command_owned = False
        executor._events = []
        executor._last_command = [0.1, 0.0, 0.0]
        executor._goal_generation = 0
        executor._command_pub = mock.Mock()
        executor._desired_command_pub = mock.Mock()
        executor._on_stair_command_state(
            SimpleNamespace(data="STAIR_ASCENT_B"),
            "/simenv/second_to_third_floor_stair_state")
        self.assertTrue(executor._stair_command_owned)
        self.assertFalse(executor._publish_zero())
        executor._command_pub.publish.assert_not_called()
        executor._on_stair_command_state(
            SimpleNamespace(data="THIRD_FLOOR_HANDOFF_COMPLETE"),
            "/simenv/second_to_third_floor_stair_state")
        self.assertFalse(executor._stair_command_owned)
        self.assertTrue(executor._publish_zero())
        executor._command_pub.publish.assert_called_once()

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
        overrides = SecondFloorMission._second_floor_only_parameter_overrides(
            unified_fast_profile=True)
        self.assertTrue(
            overrides["known_unvisited_door_retry_only_on_terminal_return"])
        self.assertNotIn(
            "corridor_partial_return_reserve_seconds", overrides)
        self.assertNotIn(
            "corridor_partial_return_room_minimum_remaining_seconds",
            overrides)
        self.assertAlmostEqual(
            overrides["upper_floor_truth_corridor_probe_distance_m"], 1.20)


    def test_fix09_corridor_heading_uses_truth_odom_frame_transform(self):
        truth = (0.0005, 13.3221, 2.9122, 1.5861926524)
        odom = (-0.5775, 2.3134, 2.9122, -1.8644133187)
        heading = SecondFloorMission._corridor_heading_from_truth_odom(
            truth, odom)
        self.assertAlmostEqual(heading, -1.8798096443, places=6)
        # Cardinal snapping to -pi/2 would add 1.90 m of lateral error over
        # the 6.18 m fix09 corridor chord.
        self.assertGreater(abs(heading + math.pi / 2.0), 0.30)

    def test_fix19_upper_floor_truth_probe_uses_station_centerline(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.corridor_guided_entry_door_context = True
        manager.corridor_established = True
        manager.room_scheduler = SimpleNamespace(active_door=None)
        manager.lock = threading.RLock()
        manager.pose = (0.60, 19.70, 2.91, 0.0)
        manager.corridor_axis = np.asarray([0.17, 0.99])
        manager.corridor_station_axis = np.asarray([0.0, 1.0])
        manager.corridor_forward_axis_hint = np.asarray([0.0, 1.0])
        manager.corridor_station_origin = np.asarray([-0.29, 9.37])
        manager.corridor_forward_distance = 2.5
        manager.corridor_sweep_history = []
        manager.elapsed = lambda: 184.0
        with mock.patch(
                "baseline_exploration_manager.rospy.get_param",
                return_value=2):
            goal = manager._plan_upper_floor_truth_corridor_probe(10)
        self.assertIsNotNone(goal)
        self.assertAlmostEqual(goal["position"][0], -0.29, places=6)
        # The truth segment deliberately lands 0.15 m beyond the ordinary
        # minimum-advance gate, avoiding a 2.49 m odometry measurement being
        # classified as a backward goal.
        self.assertAlmostEqual(goal["position"][1], 22.35, places=6)
        self.assertAlmostEqual(goal["corridor_advance_m"], 2.65, places=6)
        self.assertLess(goal["position"][0], 1.10)
        self.assertEqual(goal["source"],
                         "upper_floor_truth_corridor_probe")

    def test_run148_preportal_wall_cannot_reverse_f3_corridor(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.floor_number = 3
        manager.completed_near_pair_station_m = 14.865
        manager.truth_portal_observation_standoff_m = 0.45

        self.assertFalse(manager._upper_floor_terminal_reverse_allowed(
            (-0.80, 11.16, 5.51, math.pi / 2.0),
            np.asarray([0.0, 1.0]), exited_rooms=0))
        self.assertTrue(manager._upper_floor_terminal_reverse_allowed(
            (-0.10, 14.42, 5.51, math.pi / 2.0),
            np.asarray([0.0, 1.0]), exited_rooms=0))
        self.assertTrue(manager._upper_floor_terminal_reverse_allowed(
            (-0.80, 11.16, 5.51, math.pi / 2.0),
            np.asarray([0.0, 1.0]), exited_rooms=1))

    def test_run149_portal_clamp_reserves_centerline_recovery_distance(self):
        source = inspect.getsource(
            BaselineExplorationManager._plan_upper_floor_truth_corridor_probe)
        self.assertIn(
            'self, "reached_tolerance", 0.30', source)
        self.assertIn(
            "start_station + minimum_station_advance", source)
        self.assertIn(
            '"minimum_station_advance_m"', source)

    def test_f3_post_exit_truth_probe_is_short_and_one_shot(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.corridor_guided_entry_door_context = True
        manager.room_scheduler = SimpleNamespace(
            active_door=None, detector=SimpleNamespace(doors=[]))
        manager.lock = threading.RLock()
        manager.pose = (-0.74, 14.90, 2.91, -2.21)
        manager.grid = None
        manager.corridor_axis = np.asarray([0.0, 1.0])
        manager.corridor_station_axis = np.asarray([0.0, 1.0])
        manager.corridor_forward_axis_hint = np.asarray([0.0, 1.0])
        manager.corridor_station_origin = np.asarray([-0.66, 13.30])
        manager.corridor_forward_distance = 2.5
        manager.upper_floor_truth_probe_distance = 3.0
        manager.upper_floor_truth_probe_selected_count = 3
        manager.upper_floor_truth_probe_prescan = False
        manager.upper_floor_truth_room_fallback_active = False
        manager.upper_floor_post_exit_truth_probe_pending = True
        manager.post_entry_observed_cells = {(0, 0)}
        manager.corridor_sweep_history = []
        manager.elapsed = lambda: 82.0

        with mock.patch(
                "baseline_exploration_manager.rospy.get_param",
                return_value=3):
            goal = manager._plan_upper_floor_truth_corridor_probe(9)

        self.assertIsNotNone(goal)
        self.assertTrue(goal["post_exit_short_truth_probe"])
        self.assertAlmostEqual(goal["corridor_advance_m"], 1.20)
        self.assertAlmostEqual(goal["position"][0], -0.66)
        self.assertAlmostEqual(goal["position"][1], 16.10)
        self.assertFalse(manager.upper_floor_post_exit_truth_probe_pending)

    def test_completed_near_pair_uses_one_portal_clamped_far_row_transit(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.corridor_guided_entry_door_context = True
        near_doors = [
            SimpleNamespace(center=np.asarray([-1.0, 14.865]), visited=True),
            SimpleNamespace(center=np.asarray([1.0, 14.865]), visited=True),
        ]
        manager.room_scheduler = SimpleNamespace(
            active_door=None, detector=SimpleNamespace(doors=near_doors))
        manager.lock = threading.RLock()
        manager.pose = (0.0, 16.10, 5.51, math.pi / 2.0)
        manager.grid = None
        manager.corridor_axis = np.asarray([0.0, 1.0])
        manager.corridor_station_axis = np.asarray([0.0, 1.0])
        manager.corridor_forward_axis_hint = np.asarray([0.0, 1.0])
        manager.corridor_station_origin = np.asarray([0.0, 13.30])
        manager.corridor_forward_distance = 2.5
        manager.upper_floor_truth_probe_distance = 3.0
        manager.upper_floor_truth_probe_selected_count = 4
        manager.upper_floor_truth_probe_prescan = False
        manager.upper_floor_truth_room_fallback_active = True
        # The second completed near-row EXIT normally arms a short 1.20 m
        # map-growth probe.  The direct far-row chord must consume it instead
        # of forcing an extra stop before the same physical transit.
        manager.upper_floor_post_exit_truth_probe_pending = True
        manager.truth_room_fallback_retry_limit = 2
        manager.f3_truth_room_fallback_attempts = {}
        manager.room_target_count = 4
        manager.post_entry_observed_cells = {(0, 0)}
        manager.corridor_sweep_history = []
        manager.elapsed = lambda: 130.0

        with mock.patch(
                "baseline_exploration_manager.rospy.get_param",
                return_value=3):
            goal = manager._plan_upper_floor_truth_corridor_probe(12)

        self.assertIsNotNone(goal)
        self.assertTrue(goal["completed_near_pair_fast_row_transit"])
        self.assertAlmostEqual(goal["position"][1], 28.445, places=6)
        self.assertAlmostEqual(goal["corridor_advance_m"], 12.345,
                               places=6)
        self.assertFalse(manager.upper_floor_post_exit_truth_probe_pending)
        transit_event = next(
            item for item in manager.corridor_sweep_history
            if item.get("event") ==
            "COMPLETED_NEAR_PAIR_FAST_ROW_TRANSIT_ARMED")
        self.assertTrue(transit_event["post_exit_probe_consumed"])
        self.assertTrue(any(
            item.get("event") ==
            "COMPLETED_NEAR_PAIR_FAST_ROW_TRANSIT_ARMED"
            for item in manager.corridor_sweep_history))

    def test_f3_room_translation_uses_f1_f2_executor_envelope(self):
        source = inspect.getsource(SecondFloorMission._copy_first_floor_parameters)
        self.assertIn('rospy.set_param("~room_entry_speed", 1.35)', source)
        self.assertIn(
            'rospy.set_param("~room_open_transition_speed", 1.35)', source)
        self.assertIn('rospy.set_param("~room_exit_speed", 1.35)', source)
        self.assertIn(
            'rospy.set_param("~room_fast_exit_approach_speed", 1.35)',
            source)
        # Lateral/yaw stability guards remain tighter than F1/F2.
        self.assertIn(
            'rospy.set_param("~room_lateral_speed_limit", 0.25)', source)
        self.assertIn('rospy.set_param("~room_maximum_yaw_rate", 0.50)', source)

    def test_pre_riser_acceptance_logs_do_not_claim_pose_restore(self):
        source = inspect.getsource(StairTransition._tick_impl)
        self.assertNotIn("post-policy pre-riser pose force-restored", source)
        self.assertIn("starting ascent without Gazebo pose write", source)

    def test_completed_pair_truth_corridor_chord_requires_dense_clearance(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.floor_number = 3
        manager._offline_truth_layout_cache = None
        manager._room_contract_transform_for_floor = lambda: "world_aligned"
        manager._contract_map_point_to_truth_world = \
            lambda point, _transform: point
        layout = {
            "floors": [{}, {}, {
                "corridor_bounds": {
                    "x_min": -1.1, "x_max": 1.1,
                    "y_min": 7.85, "y_max": 35.91,
                },
                "rooms": [{
                    "furniture": [{
                        "id": "outside_corridor",
                        "pose": [-4.0, 22.0],
                        "size": [1.0, 1.0],
                    }],
                }],
            }],
        }
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False) as stream:
            json.dump(layout, stream)
            manager.offline_truth_layout_metadata = stream.name
        try:
            samples, detail = manager._truth_layout_corridor_segment_clear(
                (0.0, 16.1), (0.0, 28.445))
            self.assertIsNotNone(samples)
            self.assertGreater(len(samples), 150)
            self.assertEqual(
                detail["reason"],
                "truth_corridor_bounds_and_furniture_clear")

            samples, detail = manager._truth_layout_corridor_segment_clear(
                (0.85, 16.1), (0.85, 28.445))
            self.assertIsNone(samples)
            self.assertEqual(
                detail["reason"],
                "sample_outside_body_margined_corridor")
        finally:
            os.unlink(manager.offline_truth_layout_metadata)

    def test_upper_floor_truth_probe_stops_before_pending_near_portal(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.corridor_guided_entry_door_context = True
        manager.room_scheduler = SimpleNamespace(
            active_door=None, detector=SimpleNamespace(doors=[]))
        manager.lock = threading.RLock()
        manager.pose = (0.0, 12.804, 2.91, math.pi / 2.0)
        manager.grid = None
        manager.corridor_axis = np.asarray([0.0, 1.0])
        manager.corridor_station_axis = np.asarray([0.0, 1.0])
        manager.corridor_forward_axis_hint = np.asarray([0.0, 1.0])
        manager.corridor_station_origin = np.asarray([0.0, 12.804])
        manager.corridor_forward_distance = 2.5
        manager.upper_floor_truth_probe_distance = 3.0
        manager.upper_floor_truth_probe_selected_count = 0
        manager.upper_floor_truth_probe_prescan = False
        manager.upper_floor_truth_room_fallback_active = True
        manager.truth_room_fallback_retry_limit = 2
        manager.f3_truth_room_fallback_attempts = {}
        manager.room_target_count = 4
        manager.post_entry_observed_cells = set()
        manager.corridor_sweep_history = []
        manager.elapsed = lambda: 5.0

        with mock.patch(
                "baseline_exploration_manager.rospy.get_param",
                return_value=2):
            goal = manager._plan_upper_floor_truth_corridor_probe(1)

        self.assertIsNotNone(goal)
        self.assertAlmostEqual(goal["position"][1], 14.415, places=6)
        self.assertAlmostEqual(goal["corridor_advance_m"], 1.611, places=6)
        event = next(
            item for item in manager.corridor_sweep_history
            if item.get("event") ==
            "UPPER_FLOOR_TRUTH_PROBE_CLAMPED_TO_PORTAL_STATION")
        self.assertEqual(event["station_name"], "near")
        self.assertTrue(event["physical_entry_still_required"])

    def test_upper_floor_truth_probe_does_not_loop_at_exhausted_portal(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.corridor_guided_entry_door_context = True
        manager.room_scheduler = SimpleNamespace(
            active_door=None, detector=SimpleNamespace(doors=[]))
        manager.lock = threading.RLock()
        manager.pose = (0.0, 12.804, 2.91, math.pi / 2.0)
        manager.grid = None
        manager.corridor_axis = np.asarray([0.0, 1.0])
        manager.corridor_station_axis = np.asarray([0.0, 1.0])
        manager.corridor_forward_axis_hint = np.asarray([0.0, 1.0])
        manager.corridor_station_origin = np.asarray([0.0, 12.804])
        manager.corridor_forward_distance = 2.5
        manager.upper_floor_truth_probe_distance = 3.0
        manager.upper_floor_truth_probe_selected_count = 0
        manager.upper_floor_truth_probe_prescan = False
        manager.upper_floor_truth_room_fallback_active = True
        manager.truth_room_fallback_retry_limit = 2
        manager.f3_truth_room_fallback_attempts = {
            "near:-1": 2, "near:1": 2}
        manager.room_target_count = 4
        manager.post_entry_observed_cells = set()
        manager.corridor_sweep_history = []
        manager.elapsed = lambda: 5.0

        with mock.patch(
                "baseline_exploration_manager.rospy.get_param",
                return_value=2):
            goal = manager._plan_upper_floor_truth_corridor_probe(1)

        self.assertIsNotNone(goal)
        self.assertAlmostEqual(goal["position"][1], 15.804, places=6)
        self.assertFalse(any(
            item.get("event") ==
            "UPPER_FLOOR_TRUTH_PROBE_CLAMPED_TO_PORTAL_STATION"
            for item in manager.corridor_sweep_history))

    def test_f1_truth_probe_stops_before_pending_far_portal(self):
        """run109 must not jump from station 26.9 past the 28.895 doors."""
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.corridor_guided_entry_door_context = False
        manager.floor_number = 1
        manager.offline_truth_layout_metadata = "layout.json"
        manager.corridor_established = True
        manager.room_scheduler = SimpleNamespace(
            active_door=None,
            detector=SimpleNamespace(doors=[
                SimpleNamespace(visited=True, center=(14.2, -1.0)),
                SimpleNamespace(visited=True, center=(14.3, 1.0))]))
        manager.lock = threading.RLock()
        manager.pose = (26.905, 0.47, 0.32, 0.0)
        manager.grid = None
        manager.corridor_axis = np.asarray([1.0, 0.0])
        manager.corridor_station_axis = np.asarray([1.0, 0.0])
        manager.corridor_forward_axis_hint = np.asarray([1.0, 0.0])
        manager.corridor_station_origin = np.asarray([7.688, 0.23])
        manager.corridor_forward_distance = 7.0
        manager.upper_floor_truth_probe_distance = 7.15
        manager.upper_floor_truth_probe_selected_count = 0
        manager.upper_floor_truth_probe_prescan = False
        manager.upper_floor_truth_room_fallback_active = False
        manager.truth_room_fallback_retry_limit = 3
        manager.f3_truth_room_fallback_attempts = {}
        manager.room_target_count = 4
        manager.post_entry_observed_cells = set()
        manager.corridor_sweep_history = []
        manager.elapsed = lambda: 220.0

        with mock.patch(
                "baseline_exploration_manager.rospy.get_param",
                return_value=1):
            goal = manager._plan_upper_floor_truth_corridor_probe(19)

        self.assertIsNotNone(goal)
        self.assertAlmostEqual(goal["position"][0], 28.445, places=6)
        self.assertAlmostEqual(goal["corridor_advance_m"], 1.54, places=6)
        event = next(
            item for item in manager.corridor_sweep_history
            if item.get("event") ==
            "UPPER_FLOOR_TRUTH_PROBE_CLAMPED_TO_PORTAL_STATION")
        self.assertEqual(event["floor_number"], 1)
        self.assertEqual(event["station_name"], "far")
        self.assertTrue(event["physical_entry_still_required"])

    def test_f1_completed_near_pair_skips_near_row_and_stages_far_row(self):
        """Completed near rooms use one chord and stage before the far doors."""
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.corridor_guided_entry_door_context = False
        manager.floor_number = 1
        manager.offline_truth_layout_metadata = "layout.json"
        manager.corridor_established = True
        manager.room_scheduler = SimpleNamespace(
            active_door=None,
            detector=SimpleNamespace(doors=[
                SimpleNamespace(visited=True, center=(14.2, -1.0)),
                SimpleNamespace(visited=True, center=(14.3, 1.0))]))
        manager.lock = threading.RLock()
        manager.pose = (14.30, 0.31, 0.32, 0.0)
        manager.grid = None
        manager.corridor_axis = np.asarray([1.0, 0.0])
        manager.corridor_station_axis = np.asarray([1.0, 0.0])
        manager.corridor_forward_axis_hint = np.asarray([1.0, 0.0])
        manager.corridor_station_origin = np.asarray([7.688, 0.23])
        manager.corridor_forward_distance = 7.0
        manager.upper_floor_truth_probe_distance = 7.15
        manager.upper_floor_truth_probe_selected_count = 0
        manager.upper_floor_truth_probe_prescan = False
        manager.upper_floor_truth_room_fallback_active = False
        manager.truth_room_fallback_retry_limit = 3
        manager.f3_truth_room_fallback_attempts = {}
        manager.room_target_count = 4
        manager.post_entry_observed_cells = set()
        manager.corridor_sweep_history = []
        manager.elapsed = lambda: 170.0

        with mock.patch(
                "baseline_exploration_manager.rospy.get_param",
                return_value=1):
            goal = manager._plan_upper_floor_truth_corridor_probe(20)

        self.assertIsNotNone(goal)
        # The fast chord is intentionally clamped 0.45 m before the still
        # unvisited far doorway row; it must not jump beyond both portals.
        self.assertAlmostEqual(goal["position"][0], 27.88, places=6)
        self.assertTrue(any(
            item.get("event") == "COMPLETED_NEAR_PAIR_FAST_ROW_TRANSIT_ARMED"
            for item in manager.corridor_sweep_history))
        self.assertGreater(goal["corridor_advance_m"], 13.5)

    def test_truth_probe_grows_map_past_local_grid_edge_inside_corridor(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.floor_number = 3
        manager.corridor_guided_entry_door_context = True
        manager.room_scheduler = SimpleNamespace(active_door=None)
        manager.lock = threading.RLock()
        manager.pose = (0.14, 25.33, 5.58, math.pi / 2.0)
        manager.grid = SimpleNamespace(world_to_cell=lambda _point: None)
        manager.corridor_axis = np.asarray([0.0, 1.0])
        manager.corridor_station_axis = np.asarray([0.0, 1.0])
        manager.corridor_forward_axis_hint = np.asarray([0.0, 1.0])
        manager.corridor_station_origin = np.asarray([0.14, 13.35])
        manager.corridor_forward_distance = 2.5
        manager.upper_floor_truth_probe_distance = 2.65
        manager.upper_floor_truth_probe_selected_count = 4
        manager.upper_floor_truth_probe_prescan = False
        manager.enable_local_doorway_detector = False
        manager.post_entry_observed_cells = {(0, 0)}
        manager.corridor_sweep_history = []
        manager.elapsed = lambda: 171.9
        manager._offline_truth_layout_cache = None
        layout = {"floors": [
            {"corridor_bounds": {"x_min": -1.1, "x_max": 1.1,
                                  "y_min": 8.0, "y_max": 36.0}}
            for _ in range(3)]}
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False) as stream:
            json.dump(layout, stream)
            layout_path = stream.name
        manager.offline_truth_layout_metadata = layout_path
        try:
            with mock.patch(
                    "baseline_exploration_manager.rospy.get_param",
                    return_value=3):
                goal = manager._plan_upper_floor_truth_corridor_probe(27)
        finally:
            os.unlink(layout_path)
        self.assertIsNotNone(goal)
        self.assertAlmostEqual(goal["position"][1], 27.98, places=2)
        events = [event["event"]
                  for event in manager.corridor_sweep_history]
        self.assertIn(
            "UPPER_FLOOR_TRUTH_PROBE_ALLOWED_OUTSIDE_LIVE_GRID", events)

    def test_fix13_blocked_upper_probes_refresh_without_partial_return(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.corridor_sweep_history = []
        manager.corridor_terminal_return_latched = False
        manager.corridor_partial_return_trigger_reason = None
        manager.room_target_count = 4
        manager.room_camera_sweep_speed = 2.1
        manager.elapsed = lambda: 192.0
        rescans = []
        manager._perform_local_rescan = lambda **kwargs: rescans.append(kwargs)
        for _index in range(2):
            self.assertTrue(
                manager._handle_upper_floor_truth_probe_result(
                    False, "upper_floor_truth_corridor_probe",
                    "goal_footprint_blocked", 2))
            self.assertFalse(manager.corridor_terminal_return_latched)
        self.assertEqual(manager.upper_floor_truth_probe_failure_count, 2)
        self.assertTrue(
            manager._handle_upper_floor_truth_probe_result(
                False, "upper_floor_truth_corridor_probe",
                "goal_footprint_blocked", 2))
        self.assertFalse(manager.corridor_terminal_return_latched)
        self.assertEqual(manager.upper_floor_truth_probe_failure_count, 0)
        self.assertEqual(len(rescans), 1)
        self.assertEqual(
            rescans[0]["reason"], "upper_floor_blocked_probe_map_refresh")
        self.assertEqual(manager.corridor_sweep_history[-1]["event"],
                         "UPPER_FLOOR_BLOCKED_PROBE_MAP_REFRESH")

    def test_fix06_zero_room_probe_failure_does_not_poison_far_row(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.corridor_sweep_history = []
        manager.corridor_terminal_return_latched = False
        manager.corridor_partial_return_trigger_reason = None
        manager.elapsed = lambda: 20.0
        self.assertTrue(manager._handle_upper_floor_truth_probe_result(
            False, "upper_floor_truth_corridor_probe",
            "goal_footprint_blocked", 0))
        self.assertEqual(manager.upper_floor_truth_probe_failure_count, 0)
        manager.elapsed = lambda: 158.0
        self.assertTrue(manager._handle_upper_floor_truth_probe_result(
            False, "upper_floor_truth_corridor_probe",
            "start_footprint_blocked", 2))
        self.assertEqual(manager.upper_floor_truth_probe_failure_count, 1)
        self.assertFalse(manager.corridor_terminal_return_latched)
        self.assertFalse(any(
            event.get("event") == "UPPER_FLOOR_BLOCKED_PROBE_RETURN_LATCHED"
            for event in manager.corridor_sweep_history))

    def test_truth_probe_no_motion_is_counted_by_failure_watchdog(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.corridor_sweep_history = []
        manager.elapsed = lambda: 42.0

        handled = manager._handle_upper_floor_truth_probe_result(
            False, "upper_floor_truth_corridor_probe",
            "truth_probe_physical_advance_unconfirmed", 0)

        self.assertFalse(handled)
        self.assertEqual(manager.upper_floor_truth_probe_failure_count, 1)
        self.assertEqual(
            manager.corridor_sweep_history[-1]["event"],
            "UPPER_FLOOR_TRUTH_PROBE_NO_MOTION_COUNTED")

    def test_outbound_retry_gate_returns_before_touching_scheduler(self):
        manager = BaselineExplorationManager.__new__(BaselineExplorationManager)
        manager.enable_known_unvisited_door_retry = True
        manager.known_unvisited_door_retry_only_on_terminal_return = True
        manager.corridor_terminal_return_latched = False
        manager.corridor_door_detection_armed = True
        manager.room_scheduler = SimpleNamespace(active_door=None)
        self.assertIsNone(manager._plan_known_unvisited_door_retry_goal())


    def test_run115_truth_return_waits_until_stair_lobby(self):
        transition = StairTransition.__new__(StairTransition)
        transition.enable_truth_return_gate = True
        transition.truth_entry_guide = True
        transition.return_transit_armed = True
        transition.return_gate_published = False
        transition.phase = "WAIT_F1"
        transition.truth_pose = (-3.535, 17.407, 0.32, 0.05)
        transition.truth_step_pose = (-4.015, 2.18, 0.0, 0.0)
        transition.truth_step_next_pose = (-4.015, 2.30)
        transition.truth_staging_standoff = 0.43
        transition.truth_return_gate_radius = 5.4
        transition.return_gate_pub = mock.Mock()
        transition._maybe_publish_truth_return_gate()
        self.assertFalse(transition.return_gate_published)
        transition.return_gate_pub.publish.assert_not_called()
        transition.truth_pose = (-0.565, 5.55, 0.32, 0.05)
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
        self.assertEqual(params["truth_return_gate_radius_m"], "5.40")
        self.assertEqual(
            params["truth_corridor_lateral_offset_m"],
            "$(arg stair_truth_corridor_lateral_offset_m)")
        self.assertEqual(
            params["truth_corridor_longitudinal_offset_m"],
            "$(arg stair_truth_corridor_longitudinal_offset_m)")

    def test_second_to_third_uses_stricter_b_flight_alignment_profile(self):
        launch = ET.parse(os.path.join(
            SCRIPT_DIR, "..", "launch",
            "baseline_fastlio_exploration.launch"))
        root = launch.getroot()
        args = {item.get("name"): item.get("default")
                for item in root.findall("arg")}
        first = next(
            item for item in root.findall("node")
            if item.get("name") == "stair_transition_manager")
        second = next(
            item for item in root.findall("node")
            if item.get("name") == "second_to_third_floor_stair_manager")
        f1 = {item.get("name"): item.get("value")
              for item in first.findall("param")}
        f3 = {item.get("name"): item.get("value")
              for item in second.findall("param")}
        self.assertEqual(f1["flight_b_timeout_sec"],
                         "$(arg stair_flight_b_timeout_sec)")
        self.assertEqual(f3["flight_b_timeout_sec"],
                         "$(arg second_to_third_flight_b_timeout_sec)")
        self.assertEqual(f3["landing_heading_tolerance_rad"],
                         "$(arg second_to_third_landing_heading_tolerance_rad)")
        self.assertEqual(f3["truth_flight_b_heading_gain"],
                         "$(arg second_to_third_flight_b_heading_gain)")
        self.assertEqual(f3["truth_flight_b_heading_bias_rad"],
                         "$(arg second_to_third_flight_b_heading_bias_rad)")
        self.assertAlmostEqual(
            float(args["second_to_third_flight_b_heading_bias_rad"]), 0.0)
        self.assertEqual(
            f3["truth_flight_b_recovery_heading_error_rad"],
            "$(arg second_to_third_flight_b_recovery_heading_error_rad)")
        self.assertEqual(
            f3["truth_flight_b_recovery_yaw_rate_rps"],
            "$(arg second_to_third_flight_b_recovery_yaw_rate_rps)")
        self.assertLess(float(args["second_to_third_landing_heading_tolerance_rad"]),
                        float(args["landing_heading_tolerance_rad"]))
        self.assertAlmostEqual(
            float(args["second_to_third_landing_heading_tolerance_rad"]),
            0.05)
        self.assertGreater(float(args["second_to_third_flight_b_timeout_sec"]),
                           float(args["stair_flight_b_timeout_sec"]))
        self.assertGreater(
            float(args["second_to_third_flight_b_recovery_yaw_rate_rps"]),
            float(args["second_to_third_flight_b_max_yaw_rate_rps"]))
        self.assertEqual(
            f1["truth_flight_a_stall_recovery_limit"],
            "$(arg truth_flight_a_stall_recovery_limit)")
        self.assertEqual(
            f3["truth_flight_a_stall_recovery_limit"],
            "$(arg truth_flight_a_stall_recovery_limit)")
        self.assertEqual(
            f3["truth_flight_b_stall_recovery_limit"],
            "$(arg second_to_third_flight_b_stall_recovery_limit)")
        self.assertEqual(
            args["second_to_third_flight_b_stall_recovery_limit"], "2")
        self.assertEqual(
            f3["truth_flight_b_stall_recovery_distance_m"],
            "$(arg second_to_third_flight_b_stall_recovery_distance_m)")
        self.assertEqual(args["second_to_third_flight_b_stall_recovery_distance_m"], "0.28")
        self.assertEqual(
            f3["truth_flight_b_stall_forward_burst_speed_mps"],
            "$(arg second_to_third_flight_b_stall_forward_burst_speed_mps)")
        self.assertEqual(
            args["second_to_third_flight_b_stall_forward_burst_speed_mps"], "0.0")
        self.assertEqual(
            f3["truth_flight_b_stall_policy_reset"],
            "$(arg second_to_third_flight_b_stall_policy_reset)")
        self.assertEqual(args["second_to_third_flight_b_stall_policy_reset"], "true")

        self.assertEqual(f3["truth_flight_b_stall_wrench_enabled"],
                         "$(arg second_to_third_flight_b_stall_wrench_enabled)")
        self.assertEqual(args["second_to_third_flight_b_stall_wrench_enabled"], "false")


    def _run109_flight_b_transition(
            self, yaw=-2.0318, recovery_active=False):
        transition = StairTransition.__new__(StairTransition)
        transition.truth_pose = (-2.5544, 3.4419, 4.8778, yaw)
        transition.truth_flight_b_pose = (-2.485, 4.95, 4.20, 0.0)
        transition.truth_flight_b_entry_x = -2.485
        transition.truth_flight_b_heading = -np.pi / 2.0
        transition.truth_flight_b_center_deadband = 0.04
        transition.truth_flight_b_center_gain = 0.55
        transition.truth_flight_b_center_speed = 0.12
        transition.truth_flight_b_heading_gain = 0.60
        transition.truth_flight_b_max_yaw_rate = 0.16
        transition.truth_flight_b_recovery_heading_error = 0.18
        transition.truth_flight_b_recovery_release_heading_error = 0.08
        transition.truth_flight_b_recovery_forward_speed = 0.08
        transition.truth_flight_b_recovery_yaw_rate = 0.34
        transition.truth_flight_b_recovery_active = recovery_active
        transition.ascent_speed = 0.20
        return transition

    def test_run109_flight_b_large_yaw_error_enters_slow_recovery(self):
        transition = self._run109_flight_b_transition()
        center,forward,yaw_rate,center_error,heading_error = (
            transition._truth_flight_b_control())
        self.assertGreater(heading_error, 0.45)
        self.assertTrue(transition.truth_flight_b_recovery_active)
        self.assertAlmostEqual(forward, -0.08, places=6)
        self.assertAlmostEqual(yaw_rate, 0.34, places=6)
        self.assertGreater(center, 0.0)
        self.assertGreater(center_error, 0.0)

    def test_run111_flight_b_hysteresis_uses_gentle_yaw_near_alignment(self):
        transition = self._run109_flight_b_transition(
            yaw=-np.pi / 2.0-0.09, recovery_active=True)
        _center,forward,yaw_rate,_center_error,heading_error = (
            transition._truth_flight_b_control())
        self.assertAlmostEqual(abs(heading_error), 0.09, places=6)
        self.assertTrue(transition.truth_flight_b_recovery_active)
        self.assertAlmostEqual(forward, -0.20, places=6)
        self.assertAlmostEqual(yaw_rate, 0.60*0.09, places=6)

    def test_run113_flight_b_resumes_climb_inside_recovery_hysteresis(self):
        transition = self._run109_flight_b_transition(
            yaw=-np.pi / 2.0-0.152, recovery_active=True)
        transition.ascent_speed = 0.80
        _center,forward,yaw_rate,_center_error,heading_error = (
            transition._truth_flight_b_control())
        self.assertAlmostEqual(abs(heading_error), 0.152, places=6)
        self.assertTrue(transition.truth_flight_b_recovery_active)
        self.assertAlmostEqual(forward, -0.80, places=6)
        self.assertAlmostEqual(yaw_rate, 0.60*0.152, places=6)

    def test_flight_b_stall_watchdog_requires_no_forward_or_height_progress(self):
        transition = self._run109_flight_b_transition(yaw=-np.pi / 2.0)
        transition.truth_flight_b_stall_recovery_limit = 1
        transition.truth_flight_b_stall_timeout = 4.0
        transition.truth_flight_b_stall_minimum_progress = 0.08
        transition.truth_flight_b_stall_minimum_height_gain = 0.04
        transition.truth_flight_b_progress_anchor_at = None
        transition.truth_flight_b_progress_anchor_pose = None
        self.assertFalse(transition._truth_flight_b_is_stalled(now=10.0))
        transition.truth_pose = (-2.5544, 3.4219, 4.8878, -np.pi / 2.0)
        self.assertTrue(transition._truth_flight_b_is_stalled(now=14.1))
        transition.truth_pose = (-2.5544, 3.3219, 4.8878, -np.pi / 2.0)
        self.assertFalse(transition._truth_flight_b_is_stalled(now=14.2))
        self.assertAlmostEqual(
            transition.truth_flight_b_progress_anchor_at, 14.2)

    def test_flight_b_retreat_progress_uses_opposite_truth_heading(self):
        origin = (-2.40, 3.38, 4.87, -np.pi / 2.0)
        self.assertAlmostEqual(
            StairTransition._truth_flight_b_retreat_progress(
                origin, (-2.40, 3.66, 4.80, 0.0), -np.pi / 2.0),
            0.28, places=6)
        self.assertAlmostEqual(
            StairTransition._truth_flight_b_retreat_progress(
                origin, (-2.10, 3.10, 4.90, 0.0), -np.pi / 2.0),
            0.0, places=6)

    def test_flight_b_forward_progress_uses_truth_heading(self):
        origin = (-2.40, 3.38, 4.87, -np.pi / 2.0)
        self.assertAlmostEqual(
            StairTransition._truth_flight_b_forward_progress(
                origin, (-2.10, 3.10, 4.95, 0.0), -np.pi / 2.0),
            0.28, places=6)


    def test_run109_flight_b_recovery_releases_only_after_alignment(self):
        transition = self._run109_flight_b_transition(
            yaw=-np.pi / 2.0-0.05, recovery_active=True)
        _center,forward,yaw_rate,_center_error,heading_error = (
            transition._truth_flight_b_control())
        self.assertAlmostEqual(abs(heading_error), 0.05, places=6)
        self.assertFalse(transition.truth_flight_b_recovery_active)
        self.assertAlmostEqual(forward, -0.20, places=6)
        self.assertLess(abs(yaw_rate), 0.16)

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

    def test_truth_policy_restage_recovers_warmup_drift(self):
        distance, vx, vy, wz = StairTransition._truth_policy_restage_control(
            (-3.988, 1.546, 0.285, np.pi / 2.0), (-4.015, 1.75))
        self.assertGreater(distance, 0.20)
        self.assertLess(vx, 0.0)
        self.assertGreater(vy, 0.0)
        self.assertLess(abs(wz), 0.15)
        stopped = StairTransition._truth_policy_restage_control(
            (-4.014, 1.759, 0.327, np.pi / 2.0), (-4.015, 1.75))
        self.assertLess(stopped[0], 0.10)
        self.assertEqual(stopped[1:], (0.0, 0.0, 0.0))

    def test_truth_entry_world_projection_uses_live_yaw_until_flight_a(self):
        transition = StairTransition.__new__(StairTransition)
        transition.source_floor_index = 0
        transition.truth_fixed_two_flight_profile = True
        transition.truth_stair_heading = 0.0
        transition.truth_pose = (0.0, 0.0, 0.3, np.pi / 2.0)
        transition.cmd = mock.Mock()
        transition.hold_rl = mock.Mock()

        transition.phase = "TRUTH_ENTRY_GUIDE"
        transition._publish_truth_world_command(1.0, 0.0)
        entry_command = transition.cmd.publish.call_args.args[0]
        self.assertAlmostEqual(entry_command.linear.x, 0.0, places=6)
        self.assertAlmostEqual(entry_command.linear.y, -1.0, places=6)

        transition.phase = "STAIR_FLIGHT_A"
        transition._publish_truth_world_command(1.0, 0.0)
        ascent_command = transition.cmd.publish.call_args.args[0]
        self.assertAlmostEqual(ascent_command.linear.x, 1.0, places=6)
        self.assertAlmostEqual(ascent_command.linear.y, 0.0, places=6)

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
        # The signed centre error directly commands motion back to the axis.
        transition.source_floor_index = 0
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
        # Centre-only recovery must retain the effective stair-climb command.
        self.assertAlmostEqual(vy, 0.80, places=6)
        self.assertGreater(heading_error, 0.0)
        self.assertAlmostEqual(yaw_rate, 0.0597787, places=5)
        self.assertTrue(transition.truth_flight_a_recovery_active)

    def test_run112_flight_a_stall_requires_no_forward_or_height_progress(self):
        transition = StairTransition.__new__(StairTransition)
        transition.truth_pose = (-4.177, 3.235, 0.984, np.pi / 2.0)
        transition.truth_stair_heading = np.pi / 2.0
        transition.direction = np.pi / 2.0
        transition.truth_flight_a_stall_recovery_limit = 1
        transition.truth_flight_a_stall_timeout = 4.0
        transition.truth_flight_a_stall_minimum_progress = 0.08
        transition.truth_flight_a_stall_minimum_height_gain = 0.04
        transition.truth_flight_a_progress_anchor_at = None
        transition.truth_flight_a_progress_anchor_pose = None
        self.assertFalse(transition._truth_flight_a_is_stalled(now=10.0))
        transition.truth_pose = (-4.177, 3.255, 0.994, np.pi / 2.0)
        self.assertTrue(transition._truth_flight_a_is_stalled(now=14.1))
        transition.truth_pose = (-4.177, 3.345, 0.994, np.pi / 2.0)
        self.assertFalse(transition._truth_flight_a_is_stalled(now=14.2))
        self.assertAlmostEqual(
            transition.truth_flight_a_progress_anchor_at, 14.2)

    def test_run112_truth_flight_a_has_bounded_timeout(self):
        transition = StairTransition.__new__(StairTransition)
        transition.started = 100.0
        transition.ascent_timeout = 45.0
        self.assertFalse(transition._truth_flight_a_timed_out(now=144.99))
        self.assertTrue(transition._truth_flight_a_timed_out(now=145.0))

    def test_flight_a_recovery_crosses_yaw_deadband_before_guard(self):
        transition = StairTransition.__new__(StairTransition)
        # A right-side excursion must command negative world-X correction.
        transition.source_floor_index = 0
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

    def test_f2_f3_flight_a_yaw_is_negative_feedback(self):
        """Upper-flight yaw must reduce, never amplify, truth heading error."""
        transition = StairTransition.__new__(StairTransition)
        transition.source_floor_index = 1
        transition.truth_pose = (-4.015, 1.76, 2.92, 1.59)
        transition.truth_step_pose = (-4.015, 1.75, 0.0, 0.0)
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
        transition.truth_ascent_start_z = 2.92

        _, _, yaw_rate, _, heading_error = (
            transition._truth_flight_a_control())
        self.assertLess(heading_error, 0.0)
        self.assertLess(yaw_rate, 0.0)
        self.assertGreater(yaw_rate * heading_error, 0.0)

        transition.truth_pose = (-4.015, 1.76, 2.92, 1.55)
        _, _, yaw_rate, _, heading_error = (
            transition._truth_flight_a_control())
        self.assertGreater(heading_error, 0.0)
        self.assertGreater(yaw_rate, 0.0)
        self.assertGreater(yaw_rate * heading_error, 0.0)

    def test_room_exit_not_reached_is_terminal_stair_failure(self):
        self.assertTrue(SecondFloorMission._terminal_stair_failure_state(
            "STAIR_TRUTH_ROOM_EXIT_NOT_REACHED"))

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

    def test_f2_to_f3_landing_bias_gets_effective_yaw_command(self):
        transition = StairTransition.__new__(StairTransition)
        transition.landing_turn_speed = 0.45
        transition.truth_landing_minimum_yaw_rate = 0.24
        transition.landing_recovery_minimum_yaw_rate = 0.34
        transition.landing_heading_tolerance = 0.05
        transition.landing_recovery_active = False
        self.assertAlmostEqual(
            transition._truth_landing_turn_rate(-0.10), -0.24, places=6)
        self.assertAlmostEqual(
            transition._truth_landing_turn_rate(-0.049), -0.0539, places=6)

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

    def test_truth_guide_steady_sim_progress_is_not_low_rate_stall(self):
        # run178 reduced the east-clear residual from 1.588 m to 0.522 m in
        # about 12.2 ROS seconds.  Low RTF made that same motion take more
        # than 15 wall seconds; wall time must not be used as the m/s basis.
        self.assertFalse(SecondFloorMission._truth_route_low_rate_stalled(
            1.588, 0.522, 12.2, 12.0, 0.05))

    def test_truth_guide_genuine_sim_crawl_is_low_rate_stall(self):
        self.assertTrue(SecondFloorMission._truth_route_low_rate_stalled(
            1.588, 1.50, 12.2, 12.0, 0.05))

    def test_upper_floor_truth_handoff_is_three_metres_inside_corridor(self):
        corridor = {
            "x_min": -1.1, "x_max": 1.1,
            "y_min": 7.85, "y_max": 35.91,
        }
        target = SecondFloorMission._truth_corridor_entry_target(
            corridor, 3.0)
        self.assertEqual(target, (0.0, 10.85))
        waypoint = {
            "corridor_bounds": corridor,
            "minimum_ingress_y": 10.55,
        }
        self.assertFalse(SecondFloorMission._truth_corridor_arrival_valid(
            (0.0, 8.61, 2.91, 1.39), waypoint, 0.30))
        self.assertTrue(SecondFloorMission._truth_corridor_arrival_valid(
            (0.1, 10.60, 2.91, 1.55), waypoint, 0.30))

    def test_upper_floor_ingress_depth_is_wired_per_floor(self):
        launch = ET.parse(os.path.join(
            SCRIPT_DIR, "..", "launch",
            "baseline_fastlio_exploration.launch"))
        root = launch.getroot()
        args = {item.get("name"): item.get("default")
                for item in root.findall("arg")}
        second = next(item for item in root.findall("node")
                      if item.get("name") ==
                      "second_floor_exploration_manager")
        third = next(item for item in root.findall("node")
                     if item.get("name") ==
                     "third_floor_exploration_manager")
        second_params = {item.get("name"): item.get("value")
                         for item in second.findall("param")}
        third_params = {item.get("name"): item.get("value")
                        for item in third.findall("param")}
        self.assertEqual(
            args["second_floor_truth_corridor_ingress_depth_m"], "1.0")
        self.assertEqual(
            second_params["second_floor_truth_corridor_ingress_depth_m"],
            "$(arg second_floor_truth_corridor_ingress_depth_m)")
        self.assertEqual(
            third_params["second_floor_truth_corridor_ingress_depth_m"],
            "$(arg third_floor_truth_corridor_ingress_depth_m)")

    def test_second_floor_successful_exit_does_not_shutdown_roslaunch(self):
        launch = ET.parse(os.path.join(
            SCRIPT_DIR, "..", "launch",
            "baseline_fastlio_exploration.launch"))
        node = next(
            item for item in launch.getroot().findall("node")
            if item.get("name") == "second_floor_exploration_manager")
        self.assertEqual(node.get("required"), "false")

    def test_third_floor_failure_does_not_kill_full_roslaunch(self):
        launch = ET.parse(os.path.join(
            SCRIPT_DIR, "..", "launch",
            "baseline_fastlio_exploration.launch"))
        node = next(
            item for item in launch.getroot().findall("node")
            if item.get("name") == "third_floor_exploration_manager")
        self.assertEqual(
            node.get("required"),
            "$(eval not arg('enable_third_floor_descent'))")

    def test_all_floors_share_fast_room_target_with_return_headroom(self):
        launch = ET.parse(os.path.join(
            SCRIPT_DIR, "..", "launch",
            "baseline_fastlio_exploration.launch"))
        args = {item.get("name"): item.get("default")
                for item in launch.getroot().findall("arg")}
        self.assertEqual(float(args["first_floor_room_phase_target_seconds"]),
                         140.0)
        self.assertEqual(float(args["maximum_duration"]), 220.0)
        self.assertEqual(float(args["second_floor_maximum_duration"]), 220.0)
        self.assertEqual(float(args["third_floor_maximum_duration"]), 220.0)
        self.assertEqual(args["unified_fast_four_room_profile"], "true")
        self.assertEqual(args["require_all_rooms_for_floor_handoff"], "true")
        self.assertEqual(
            float(args["first_floor_corridor_entry_distance_from_mission_start_m"]),
            8.0)
        self.assertEqual(
            float(args["second_floor_truth_corridor_ingress_depth_m"]), 1.0)
        self.assertEqual(
            float(args["third_floor_truth_corridor_ingress_depth_m"]), 1.0)
        self.assertAlmostEqual(
            float(args["upper_floor_far_row_door_scan_angle_rad"]), np.pi,
            places=6)
        nodes = {item.get("name"): item
                 for item in launch.getroot().findall("node")}
        for node_name in ("baseline_exploration_manager",
                          "second_floor_exploration_manager",
                          "third_floor_exploration_manager"):
            params = {item.get("name"): item.get("value")
                      for item in nodes[node_name].findall("param")}
            self.assertEqual(
                params["require_all_rooms_for_floor_handoff"],
                "$(arg require_all_rooms_for_floor_handoff)")
            self.assertEqual(params["room_target_count"], "4")

    def test_public_fullflow_launch_forwards_fast_room_motion_profile(self):
        launch = ET.parse(os.path.join(
            SCRIPT_DIR, "..", "launch",
            "fuel_semantic_fastlio_exploration.launch"))
        root = launch.getroot()
        args = {item.get("name"): item.get("default")
                for item in root.findall("arg")}
        expected = {
            "first_floor_room_entry_speed": "1.20",
            "first_floor_room_open_transition_speed": "1.20",
            "first_floor_room_exit_speed": "0.95",
            "first_floor_room_fast_exit_approach_speed": "1.10",
            "first_floor_room_portal_crossing_speed": "0.90",
            "first_floor_room_obstacle_transition_speed": "0.90",
            "goal_executor_room_transaction_degraded_speed_cap": "1.35",
            "goal_executor_room_exit_degraded_speed_cap": "1.35",
        }
        for name, value in expected.items():
            self.assertEqual(args[name], value)
        include = root.find("include")
        forwarded = {item.get("name"): item.get("value")
                     for item in include.findall("arg")}
        for name in expected:
            self.assertEqual(forwarded[name], "$(arg %s)" % name)

    def test_f1_stair_position_state_requires_explicit_transit_arm(self):
        transition = StairTransition.__new__(StairTransition)
        transition.phase = "WAIT_F1"
        transition.return_transit_armed = False
        transition.require_return_transit_arm_for_state_handoff = True
        transition.pending_state_handoff = False
        transition._maybe_publish_truth_return_gate = lambda: None
        starts = []
        transition._begin_handoff = lambda: starts.append("started")

        with mock.patch("stair_transition_manager.rospy.logwarn_throttle"):
            transition.on_state(SimpleNamespace(data=json.dumps({
                "state": "STAIR_LOBBY_HANDOFF",
                "reason": "g2_best_effort_return",
            })))
        self.assertEqual(starts, [])
        self.assertTrue(transition.pending_state_handoff)

        transition.on_state(SimpleNamespace(data=json.dumps({
            "state": "STAIR_RETURN_TRANSIT",
            "reason": "four_rooms_complete",
        })))
        self.assertTrue(transition.return_transit_armed)
        self.assertEqual(starts, ["started"])

    def test_f1_stair_does_not_match_diagnostic_substring_as_arm(self):
        transition = StairTransition.__new__(StairTransition)
        transition.phase = "WAIT_F1"
        transition.return_transit_armed = False
        transition.require_return_transit_arm_for_state_handoff = True
        transition.pending_state_handoff = False
        transition._maybe_publish_truth_return_gate = lambda: None
        transition._begin_handoff = lambda: self.fail(
            "diagnostic state must not arm or start the stair")

        transition.on_state(SimpleNamespace(data=json.dumps({
            "state": "STAIR_RETURN_TRANSIT_REJECTED_MISSING_ROOMS",
        })))
        self.assertFalse(transition.return_transit_armed)


    def test_f3_local_reset_is_final_pose_joint_and_feedback_writer(self):
        with open(os.path.join(
                SCRIPT_DIR, "second_floor_exploration_manager.py"),
                encoding="utf-8") as stream:
            source = stream.read()
        run_body = source.split("def run(self):", 1)[1].split(
            'if __name__ == "__main__"', 1)[0]
        f3_branch = run_body.split(
            "F3 alone starts on a narrow, dynamic stair seam", 1)[1]
        self.assertLess(
            f3_branch.index("self._reset_truth_upper_landing_pose"),
            f3_branch.index("self._local_reset_controller_at_corridor"))

    def test_third_floor_worker_is_required_terminal_supervisor(self):
        launch = ET.parse(os.path.join(
            SCRIPT_DIR, "..", "launch",
            "baseline_fastlio_exploration.launch"))
        node = next(item for item in launch.getroot().findall("node")
                    if item.get("name") ==
                    "third_floor_exploration_manager")
        self.assertEqual(
            node.get("required"),
            "$(eval not arg('enable_third_floor_descent'))")

    def test_f1_flight_b_fall_uses_bounded_landing_recovery_first(self):
        with open(os.path.join(
                SCRIPT_DIR, "stair_transition_manager.py"),
                encoding="utf-8") as stream:
            source = stream.read()
        fall_branch = source.split(
            "self.truth_flight_b_fall_drop and", 1)[1].split(
                "elif self.phase", 1)[0]
        self.assertLess(
            fall_branch.index("self._recover_f1_flight_b_timeout()"),
            fall_branch.index("self.phase='STAIR_FLIGHT_B_FALL_DETECTED'"))

    def test_f1_landing_recovery_waits_for_plane_policy_then_reapplies_pose(self):
        with open(os.path.join(
                SCRIPT_DIR, "stair_transition_manager.py"),
                encoding="utf-8") as stream:
            source = stream.read()
        recovery = source.split(
            "def _recover_f1_flight_b_timeout(self):", 1)[1].split(
                "def _begin_handoff", 1)[0]
        self.assertIn(
            "self.phase='STAIR_F1_LANDING_RECOVERY_POLICY_LOADING'",
            recovery)
        self.assertLess(
            recovery.index("self._reset_f1_upper_landing_controller()"),
            recovery.index("self.policy=self.truth_flight_b_bridge_policy"))
        self.assertLess(
            recovery.index("self.policy=self.truth_flight_b_bridge_policy"),
            recovery.index("self.pub.publish(String(data=self.policy))"))
        policy_wait = source.split(
            "elif self.phase=='STAIR_F1_LANDING_RECOVERY_POLICY_LOADING':",
            1)[1].split("elif self.phase=='STAIR_ASCENT_B'", 1)[0]
        self.assertIn("if self.policy_loaded:", policy_wait)
        self.assertIn("self._place_f1_verified_upper_landing()", policy_wait)
        self.assertIn("STAIR_F1_LANDING_RECOVERY_POLICY_TIMEOUT", policy_wait)

    def test_f1_upper_landing_fall_resets_joints_and_stands_before_rl(self):
        with open(os.path.join(
                SCRIPT_DIR, "stair_transition_manager.py"),
                encoding="utf-8") as stream:
            source = stream.read()
        recovery = source.split(
            "def _reset_f1_upper_landing_controller(self):", 1)[1].split(
                "def _recover_f1_flight_b_timeout", 1)[0]
        self.assertIn(
            "rospy.set_param('/simenv/local_reset/upper_floor_guard_active', True)",
            recovery)
        self.assertIn(
            "rospy.set_param('/simenv/local_reset/use_startup_stance', True)",
            recovery)
        self.assertIn("reset.buttons[10]=1", recovery)
        self.assertIn("'/simenv/local_reset/completed'", recovery)
        self.assertIn("transaction_deadline=time.monotonic()+12.0", recovery)
        self.assertIn("reset_accepted = bool(rospy.get_param(", recovery)
        self.assertNotIn(
            "upper-landing controller did not consume local reset", recovery)
        self.assertIn("self._recover_f1_controller_stand()", recovery)
        self.assertNotIn("self.pub.publish(String(data=self.policy))", recovery)

    def test_f1_pre_riser_policy_swap_prefers_verified_hot_handoff(self):
        with open(os.path.join(
                SCRIPT_DIR, "stair_transition_manager.py"),
                encoding="utf-8") as stream:
            source = stream.read()
        hot_queue = source.split(
            "def _queue_f1_stair_policy_hot(self):", 1)[1].split(
                "def _publish_world_command", 1)[0]
        self.assertIn("_truth_pre_riser_hot_handoff_ready", hot_queue)
        self.assertIn("self.pub.publish(String(data=self.policy))", hot_queue)
        self.assertNotIn("_recover_f1_controller_stand", hot_queue)
        self.assertIn("stair_fast_takeover_enabled", hot_queue)
        settle = source.split(
            "elif self.phase=='STAIR_TRUTH_STAGE_SETTLE':", 1)[1].split(
                "elif self.phase=='STAIR_POLICY_LOADING':", 1)[0]
        self.assertLess(
            settle.index("self._queue_f1_stair_policy_hot()"),
            settle.index("self._queue_f1_stair_policy_from_fixed_stand()"))
        self.assertIn("STAIR_PRE_RISER_POSTURE_INVALID", settle)

    def test_f1_pre_riser_fixed_stand_remains_bounded_fallback(self):
        with open(os.path.join(
                SCRIPT_DIR, "stair_transition_manager.py"),
                encoding="utf-8") as stream:
            source = stream.read()
        queue = source.split(
            "def _queue_f1_stair_policy_from_fixed_stand(self):", 1)[1].split(
                "def _truth_pre_riser_posture_upright", 1)[0]
        self.assertLess(
            queue.index("_recover_f1_controller_stand(release_to_rl=False)"),
            queue.index("self.pub.publish(String(data=self.policy))"))
        self.assertLess(
            queue.index("self.pub.publish(String(data=self.policy))"),
            queue.index("self.hold_rl()"))
        self.assertIn("stair_fast_takeover_enabled", queue)
        warmup = source.split(
            "elif self.phase=='STAIR_POLICY_WARMUP':", 1)[1].split(
                "elif (self.phase=='STAIR_FLIGHT_A_STALL_RECOVERY'", 1)[0]
        self.assertIn("if not self.locomotion_ready:", warmup)
        self.assertIn("STAIR_POLICY_READY_TIMEOUT", warmup)

    def test_rl_hot_reload_consumes_stair_takeover_profile(self):
        controller = os.path.abspath(os.path.join(
            SCRIPT_DIR, "..", "..", "unitree_guide", "unitree_guide",
            "unitree_guide", "src", "FSM", "State_RL_test.cpp"))
        with open(controller, encoding="utf-8") as stream:
            source = stream.read()
        hot_reload = source.split("void State_RL::run(){", 1)[1].split(
            "void State_RL::policyRequestCallback", 1)[0]
        self.assertIn('requested.find("stair")', hot_reload)
        self.assertIn(
            'nh.getParam("/simenv/stair_fast_takeover_enabled"',
            hot_reload)
        self.assertIn("Using one-shot hot stair RL profile", hot_reload)
        self.assertLess(
            hot_reload.index("fastStairTakeover"),
            hot_reload.index("infer_thread_runnning = State_RL::STOP"))

    def test_f1_upper_landing_never_reuses_absolute_reset_z_as_offset(self):
        with open(os.path.join(
                SCRIPT_DIR, "stair_transition_manager.py"),
                encoding="utf-8") as stream:
            source = stream.read()
        placement = source.split(
            "def _place_f1_verified_upper_landing(self):", 1)[1].split(
            "def _reset_f1_upper_landing_controller", 1)[0]
        controller_reset = source.split(
            "def _reset_f1_upper_landing_controller(self):", 1)[1].split(
            "def _recover_f1_flight_b_timeout", 1)[0]
        self.assertIn("self.upper_landing_root_clearance", placement)
        self.assertIn("self.upper_landing_root_clearance", controller_reset)
        self.assertNotIn("rospy.get_param('/simenv/local_reset/z'", placement)
        self.assertNotIn("rospy.get_param(\n            '/simenv/local_reset/z'",
                         controller_reset)
        self.assertIn("self.target_floor_number-1", placement)
        self.assertIn("self.target_floor_number-1", controller_reset)

    def test_f3_rebuilds_empty_live_voxel_tree(self):
        with open(os.path.join(
                SCRIPT_DIR, "second_floor_exploration_manager.py"),
                encoding="utf-8") as stream:
            source = stream.read()
        f3_clear = source.split("if self._floor_number >= 3:", 1)[-1]
        self.assertIn("Float64(data=-2.0e9)", f3_clear)
        self.assertNotIn("Float64(data=4.30)", f3_clear)

    def test_truth_height_requires_second_floor_elevation(self):
        self.assertTrue(SecondFloorMission._truth_height_is_second_floor(
            (-2.5, 1.7, 2.91, 0.0), 2.60, 0.15))
        self.assertFalse(SecondFloorMission._truth_height_is_second_floor(
            (-2.5, 1.7, 0.31, 0.0), 2.60, 0.15))

    def test_f3_standing_posture_rejects_crouched_wrapped_handoff(self):
        failed = {
            "roll": -0.12,
            "pitch": -0.14,
            "joint_velocity_rms": 0.06,
            "joint_position_error": 1.18,
            "acceleration_norm": 10.7,
            "wrapped_target_count": 8,
        }
        # axialtruth had world z=5.38, but this is only 0.18 m above F3.
        self.assertFalse(SecondFloorMission._f3_standing_posture_valid(
            (-0.2, 8.7, 5.38, 0.0), 5.20, failed))
        # Correcting absolute height alone must not waive the joint error.
        self.assertFalse(SecondFloorMission._f3_standing_posture_valid(
            (-0.2, 8.7, 5.52, 0.0), 5.20, failed))

    def test_f3_standing_posture_accepts_flat_policy_stance(self):
        healthy = {
            "roll": 0.03,
            "pitch": -0.04,
            "joint_velocity_rms": 0.08,
            "joint_position_error": 0.06,
            "acceleration_norm": 9.81,
            "wrapped_target_count": 0,
        }
        self.assertTrue(SecondFloorMission._f3_standing_posture_valid(
            (-0.2, 8.7, 5.52, 0.0), 5.20, healthy))

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

    def test_mapping_rescan_finishes_fast_and_cannot_be_overwritten(self):
        executor = GoalExecutor.__new__(GoalExecutor)
        executor._maximum_visual_sweep_speed = 2.0
        executor._maximum_mapping_rescan_speed = 1.2
        executor._lock = threading.RLock()
        executor._locomotion_ready = True
        executor._goal = None
        executor._finished = True
        executor._rescan = None
        executor._pose = (0.0, 0.0, 0.0, 0.0)
        executor._rescan_result_pub = mock.Mock()

        first = SimpleNamespace(data=json.dumps({
            "request_id": "mapping-1",
            "angle_rad": 2.0 * np.pi,
            "angular_speed": 1.2,
            "timeout_sec": 8.0,
        }))
        executor._on_rescan_request(first)
        self.assertEqual(executor._rescan["request_id"], "mapping-1")
        self.assertAlmostEqual(executor._rescan["angle"], 2.0 * np.pi)
        self.assertAlmostEqual(executor._rescan["speed"], 1.2)
        self.assertLess(executor._rescan["duration"], 5.3)

        second = SimpleNamespace(data=json.dumps({
            "request_id": "mapping-2",
            "angle_rad": 2.0 * np.pi,
            "angular_speed": 1.2,
            "timeout_sec": 8.0,
        }))
        executor._on_rescan_request(second)
        self.assertEqual(executor._rescan["request_id"], "mapping-1")
        rejection = json.loads(
            executor._rescan_result_pub.publish.call_args.args[0].data)
        self.assertEqual(rejection["request_id"], "mapping-2")
        self.assertFalse(rejection["success"])

    @staticmethod
    def _exit_refinement_manager(checker, pose):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.enable_scan_lite = False
        manager.room_two_pose_scan_lite_enabled = False
        manager.room_two_pose_waypoint_simplification = False
        manager.scan_config = baseline_manager_module.RefinerConfig()
        manager.waypoint_spacing = 0.30
        manager.lock = threading.RLock()
        manager.pose = (pose[0], pose[1], 0.31, pose[2])
        manager.trajectory = []
        manager.fastlio_registration_healthy = True
        manager.scan_history = []
        manager.scan_duration_pub = mock.Mock()
        manager.output_dir = "/tmp"
        manager.elapsed = mock.Mock(return_value=40.0)
        manager.corridor_sweep_history = []
        manager._scan_check = mock.Mock(side_effect=checker)
        return manager

    @classmethod
    def _run60_room1_exit_refinement(cls, checker):
        anchors = [
            (5.775000587105751, -2.0249996811151494),
            (6.07500059902668, -0.974999639391898),
            (5.775000587105751, -0.07499960362911118),
            (5.776639229569645, 0.27499656042254744),
        ]
        current = (7.40660815949865, -2.8896992187080106)
        manager = cls._exit_refinement_manager(
            checker, (current[0], current[1], -2.18))
        goal = {
            "source": "lightweight_room_exit", "room_role": "EXIT",
            "room_id": "estimated_room_01",
            "position": [anchors[-1][0], anchors[-1][1], 0.31],
            "mandatory_portal_waypoints": anchors,
            "portal_preflight_verified": True,
            "corridor_extension_applied": True,
            "direct_centerline_exit_approach": True,
        }
        path = {
            "success": True, "reason": "portal_path_found",
            "path": [current] + anchors[:-1],
        }
        with mock.patch("builtins.open", mock.mock_open()):
            result = manager._refine_path(6, path, goal)
        return manager, result, anchors

    def test_exit_door_center_index_tracks_corridor_extension(self):
        index = baseline_manager_module.portal_door_center_anchor_index
        self.assertEqual(index([(0, 0), (1, 0), (2, 0)], False), 1)
        self.assertEqual(
            index([(0, 0), (1, 0), (2, 0), (3, 0)], True), 1)
        self.assertIsNone(index([(0, 0), (1, 0)], True))

    def test_run60_direct_exit_uses_audited_inside_to_corridor_chord(self):
        def clear(_x, _y, _yaw):
            return baseline_manager_module.FootprintCheck(
                map_available=True, occupied_collision=False,
                minimum_obstacle_clearance=0.30)

        manager, result, anchors = self._run60_room1_exit_refinement(
            clear)
        self.assertEqual(result["execution_waypoints"],
                         [anchors[0], anchors[-1]])
        self.assertTrue(
            manager.scan_history[-1]["direct_portal_exit_chord"])
        self.assertEqual(
            len(manager.scan_history[-1]["mandatory_portal_waypoints"]), 4)

    def test_collinear_clear_exit_removes_inside_execution_stop(self):
        def clear(_x, _y, _yaw):
            return baseline_manager_module.FootprintCheck(
                map_available=True, occupied_collision=False,
                minimum_obstacle_clearance=0.30)

        manager = self._exit_refinement_manager(
            clear, (0.0, 0.0, 0.0))
        anchors = [(2.0, 0.0), (3.0, 0.0),
                   (4.0, 0.0), (5.0, 0.0)]
        goal = {
            "source": "lightweight_room_exit", "room_role": "EXIT",
            "room_id": "room_test", "position": [5.0, 0.0, 0.31],
            "mandatory_portal_waypoints": anchors,
            "portal_preflight_verified": True,
            "corridor_extension_applied": True,
        }
        path = {"success": True, "reason": "portal_path_found",
                "path": [(0.0, 0.0)] + anchors[:-1]}
        with mock.patch("builtins.open", mock.mock_open()):
            result = manager._refine_path(99, path, goal)
        self.assertEqual(result["execution_waypoints"], [anchors[-1]])
        self.assertTrue(manager.scan_history[-1][
            "direct_room_to_corridor_exit_chord"])

    def test_run60_direct_exit_collision_retains_all_anchors(self):
        def block_direct_midpoint(x, y, _yaw):
            blocked = abs(x - 5.775) < 0.03 and -1.15 < y < -0.85
            return baseline_manager_module.FootprintCheck(
                map_available=True, occupied_collision=blocked,
                minimum_obstacle_clearance=0.0 if blocked else 0.30)

        manager, result, anchors = self._run60_room1_exit_refinement(
            block_direct_midpoint)
        self.assertEqual(result["execution_waypoints"], anchors)
        self.assertFalse(
            manager.scan_history[-1]["direct_portal_exit_chord"])

    def test_compacted_exit_drops_prefix_sample_within_eight_cm(self):
        def clear(_x, _y, _yaw):
            return baseline_manager_module.FootprintCheck(
                map_available=True, occupied_collision=False,
                minimum_obstacle_clearance=0.30)

        manager = self._exit_refinement_manager(clear, (0.0, 0.0, 0.0))
        anchors = [(2.918, 0.0), (3.5, 0.0),
                   (4.5, 0.0), (4.85, 0.0)]
        goal = {
            "source": "lightweight_room_exit", "room_role": "EXIT",
            "room_id": "room_02", "position": [4.85, 0.0, 0.31],
            "mandatory_portal_waypoints": anchors,
            "portal_preflight_verified": True,
            "corridor_extension_applied": True,
        }
        path = {"success": True, "reason": "portal_path_found",
                "path": [(0.0, 0.0)] + anchors[:-1]}
        with mock.patch("builtins.open", mock.mock_open()):
            result = manager._refine_path(10, path, goal)
        self.assertEqual(result["execution_waypoints"], [anchors[-1]])
        self.assertFalse(any(
            abs(point[0] - 2.90) < 1e-6
            for point in result["execution_waypoints"]))

    def test_three_anchor_clear_exit_uses_audited_direct_chord(self):
        def clear(_x, _y, _yaw):
            return baseline_manager_module.FootprintCheck(
                map_available=True, occupied_collision=False,
                minimum_obstacle_clearance=0.30)

        manager = self._exit_refinement_manager(
            clear, (0.0, 0.0, 0.0))
        anchors = [(2.0, 0.0), (3.0, 0.05), (4.0, 0.0)]
        goal = {
            "source": "lightweight_room_exit", "room_role": "EXIT",
            "room_id": "room_three_anchor",
            "position": [4.0, 0.0, 0.31],
            "mandatory_portal_waypoints": anchors,
            "portal_preflight_verified": True,
            "corridor_extension_applied": True,
        }
        path = {"success": True, "reason": "portal_path_found",
                "path": [(0.0, 0.0)] + anchors[:-1]}
        with mock.patch("builtins.open", mock.mock_open()):
            result = manager._refine_path(101, path, goal)
        self.assertEqual(result["execution_waypoints"], [anchors[-1]])
        self.assertTrue(manager.scan_history[-1][
            "direct_portal_exit_chord"])

    def test_verified_exit_backtrack_prefers_live_3d_direct_chord(self):
        def clear(_x, _y, _yaw):
            return baseline_manager_module.FootprintCheck(
                map_available=True, occupied_collision=False,
                minimum_obstacle_clearance=0.30)

        manager = self._exit_refinement_manager(
            clear, (0.0, 2.0, 0.0))
        anchors = [(2.0, 0.0), (3.0, 0.0),
                   (4.0, 0.0), (5.0, 0.0)]
        goal = {
            "source": "lightweight_room_exit", "room_role": "EXIT",
            "room_id": "room_verified_trace",
            "position": [5.0, 0.0, 0.31],
            "mandatory_portal_waypoints": anchors,
            "portal_preflight_verified": False,
            "corridor_extension_applied": True,
            "verified_trajectory_backtrack": True,
        }
        path = {
            "success": True,
            "reason": "recent_room_trajectory_then_centered_portal",
            "path": [(0.0, 2.0), (0.5, 2.0), (1.0, 1.0)] + anchors,
        }
        with mock.patch("builtins.open", mock.mock_open()):
            result = manager._refine_path(102, path, goal)
        self.assertEqual(result["execution_waypoints"], [anchors[-1]])
        self.assertTrue(manager.scan_history[-1][
            "direct_portal_exit_chord"])
        self.assertFalse(manager.scan_history[-1]["fallback_used"])

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

    def test_run10_obstacle_g4_audit_replaces_stale_astar_bend_with_chord(self):
        """A successful stale 2-D detour must not hide the clear gap chord."""
        def clear(_x, _y, _yaw):
            return baseline_manager_module.FootprintCheck(
                map_available=True, occupied_collision=False,
                minimum_obstacle_clearance=0.30)

        manager = self._exit_refinement_manager(clear, (0.0, 0.0, 0.0))
        manager.floor_number = 3
        manager.room_two_pose_scan_lite_enabled = True
        manager._truth_layout_obstacle_gap_segment_clear = mock.Mock(
            return_value=(
                [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0),
                 (1.5, 0.0), (2.0, 0.0)],
                {"reason": "truth_obstacle_front_gap_and_furniture_clear"}))
        goal = {
            "source": "lightweight_room_semantic",
            "room_role": "G4",
            "room_id": "floor_3_estimated_room_03",
            "estimated_door_id": "estimated_door_03",
            "position": [2.0, 0.0, 0.31],
            "room_goal_diagnostic": {
                "two_pose_room_sweep": True,
                "two_pose_strategy": "obstacle_front_gap_opposite_side",
                "two_pose_obstacle": {"front_edge_depth_m": 3.8},
            },
        }
        stale_astar = {
            "success": True, "reason": "path_found",
            "path": [(0.0, 0.0), (0.4, 0.7), (1.0, 0.9),
                     (1.6, 0.6), (2.0, 0.0)],
            "scheduler_preflight_path": True,
        }

        with mock.patch("builtins.open", mock.mock_open()):
            result = manager._refine_path(210, stale_astar, goal)

        self.assertTrue(result["success"])
        self.assertTrue(
            result["truth_obstacle_gap_disconnected_override"])
        self.assertEqual(result["execution_waypoints"], [(2.0, 0.0)])
        self.assertIn(
            "OBSTACLE_GAP_TRUTH_CHORD_DENSE_3D_AUDIT",
            [event["event"] for event in manager.corridor_sweep_history])

    def test_truth_probe_repair_inside_reached_tolerance_keeps_original(self):
        def clear(_x, _y, _yaw):
            return baseline_manager_module.FootprintCheck(
                map_available=True, occupied_collision=False,
                minimum_obstacle_clearance=0.30)

        manager = self._exit_refinement_manager(clear, (0.0, 13.546, 0.0))
        manager.floor_number = 2
        manager.allow_truth_exploration_when_registration_lost = True
        manager.reached_tolerance = 0.28
        manager.enable_scan_lite = True
        repaired = SimpleNamespace(
            success=True,
            path=[(0.0, 13.546), (0.0, 13.815)],
            repaired_waypoints=[1],
            duration_ms=1.0,
            yaws=[math.pi / 2.0, math.pi / 2.0],
            collision_checks=10,
            colliding_original_waypoints=[1],
            debug=[], unknown_queries=0, occupied_queries=1,
            minimum_obstacle_clearance=0.1,
            reason="path_refined")
        goal = {
            "source": "upper_floor_truth_corridor_probe",
            "corridor_truth_probe": True,
            "corridor_advance_m": 0.869,
            "position": [0.0, 14.415, 2.914],
        }
        path = {
            "success": True,
            "reason": "verified_upper_floor_truth_corridor_segment",
            "path": [(0.0, 13.546), (0.0, 14.415)],
        }

        with mock.patch.object(
                baseline_manager_module.PathRefiner, "refine",
                return_value=repaired), \
                mock.patch("builtins.open", mock.mock_open()):
            result = manager._refine_path(55, path, goal)

        self.assertEqual(result["execution_waypoints"], [(0.0, 14.415)])
        self.assertEqual(
            result["refinement_status"],
            "upper_floor_truth_probe_original_after_underadvance_repair")
        self.assertEqual(
            manager.corridor_sweep_history[-1]["event"],
            "UPPER_FLOOR_TRUTH_PROBE_UNDERADVANCE_REJECTED")

    def test_truth_probe_clear_dense_audit_executes_only_endpoint(self):
        """Safety samples validate the chord but are not stop waypoints."""
        def clear(_x, _y, _yaw):
            return baseline_manager_module.FootprintCheck(
                map_available=True, occupied_collision=False,
                minimum_obstacle_clearance=0.30)

        manager = self._exit_refinement_manager(clear, (0.0, 14.0, 0.0))
        manager.floor_number = 3
        manager.allow_truth_exploration_when_registration_lost = True
        manager.reached_tolerance = 0.28
        manager.enable_scan_lite = True
        dense = [(0.0, 14.0 + 0.8 * index) for index in range(17)]
        audited = SimpleNamespace(
            success=True,
            path=dense,
            repaired_waypoints=[],
            duration_ms=2.0,
            yaws=[math.pi / 2.0] * len(dense),
            collision_checks=160,
            colliding_original_waypoints=[],
            debug=[], unknown_queries=0, occupied_queries=0,
            minimum_obstacle_clearance=0.30,
            reason="path_clear")
        goal = {
            "source": "upper_floor_truth_corridor_probe",
            "corridor_truth_probe": True,
            "corridor_advance_m": 12.8,
            "position": [0.0, 26.8, 5.5],
        }
        path = {
            "success": True,
            "reason": "verified_upper_floor_truth_corridor_segment",
            "path": [(0.0, 14.0), (0.0, 26.8)],
        }

        with mock.patch.object(
                baseline_manager_module.PathRefiner, "refine",
                return_value=audited), \
                mock.patch("builtins.open", mock.mock_open()):
            result = manager._refine_path(56, path, goal)

        self.assertEqual(result["execution_waypoints"], [(0.0, 26.8)])
        events = [item["event"]
                  for item in manager.corridor_sweep_history]
        self.assertIn(
            "UPPER_FLOOR_TRUTH_PROBE_AUDITED_CHORD_COMPACTED", events)
        self.assertTrue(manager.scan_history[-1][
            "truth_corridor_probe_chord_compacted"])
        self.assertEqual(manager.scan_history[-1]["refined_waypoint_count"], 1)

    def test_f3_truth_audited_entry_retains_portal_anchors_after_stale_repair(self):
        def clear(_x, _y, _yaw):
            return baseline_manager_module.FootprintCheck(
                map_available=True, occupied_collision=False,
                minimum_obstacle_clearance=0.30)

        manager = self._exit_refinement_manager(
            clear, (0.20, 30.08, 0.0))
        manager.floor_number = 3
        manager.enable_scan_lite = True
        anchors = [
            (-0.54, 28.896),
            (-1.14, 28.897),
            (-1.99, 28.899),
        ]
        repaired = SimpleNamespace(
            success=True,
            path=[(0.20, 30.08), anchors[0], anchors[1],
                  (-1.54, 28.899)],
            repaired_waypoints=[3], duration_ms=1.0,
            yaws=[0.0, 0.0, 0.0, 0.0], collision_checks=20,
            colliding_original_waypoints=[3], debug=[],
            unknown_queries=0, occupied_queries=1,
            minimum_obstacle_clearance=0.0, reason="path_refined")
        goal = {
            "source": "lightweight_room_entry",
            "room_role": "ENTRY",
            "room_id": "floor_3_estimated_room_03",
            "estimated_door_id": "estimated_door_03",
            "position": [anchors[-1][0], anchors[-1][1], 5.51],
            "mandatory_portal_waypoints": anchors,
            "f3_truth_portal_segment_overrides": [1, 2],
        }
        path = {
            "success": True,
            "reason": "portal_path_found",
            "path": [(0.20, 30.08)] + anchors,
        }
        manager._truth_layout_portal_segment_clear = mock.Mock(
            return_value=([anchors[0], anchors[1]], {
                "reason": "truth_portal_aperture_and_furniture_clear"}))

        with mock.patch.object(
                baseline_manager_module.PathRefiner, "refine",
                return_value=repaired), \
                mock.patch("builtins.open", mock.mock_open()):
            result = manager._refine_path(199, path, goal)

        self.assertEqual(result["execution_waypoints"], anchors)
        self.assertEqual(
            result["refinement_status"],
            "f3_truth_portal_original_anchors_after_stale_3d_repair")
        self.assertTrue(
            manager.scan_history[-1]["f3_truth_portal_anchor_restored"])
        self.assertEqual(
            manager.corridor_sweep_history[-1]["event"],
            "F3_ROOM_ENTRY_TRUTH_PORTAL_ANCHORS_RESTORED")

    def test_run66_failed_refiner_does_not_claim_portal_anchors_preserved(self):
        def blocked(_x, _y, _yaw):
            return baseline_manager_module.FootprintCheck(
                map_available=True, occupied_collision=True,
                minimum_obstacle_clearance=0.0)

        current = (14.73, -0.14)
        anchors = [(14.865, 0.20), (14.865, 1.10),
                   (14.865, 2.10)]
        manager = self._exit_refinement_manager(
            blocked, (current[0], current[1], 0.0))
        manager.floor_number = 1
        manager.enable_scan_lite = True
        failed = SimpleNamespace(
            success=False,
            path=[current] + anchors,
            repaired_waypoints=[], duration_ms=1.0,
            yaws=[0.0] * 4, collision_checks=20,
            colliding_original_waypoints=[2], debug=[],
            unknown_queries=0, occupied_queries=1,
            minimum_obstacle_clearance=0.0,
            reason="no_local_replacement")
        goal = {
            "source": "lightweight_room_entry",
            "room_role": "ENTRY",
            "room_id": "estimated_room_01",
            "estimated_door_id": "estimated_door_01",
            "position": [anchors[-1][0], anchors[-1][1], 0.31],
            "mandatory_portal_waypoints": anchors,
            "portal_preflight_verified": True,
            "truth_portal_planner_override": True,
            "room_goal_diagnostic": {
                "truth_portal_center_aligned": True,
            },
        }
        path = {"success": True, "reason": "portal_path_found",
                "path": [current] + anchors}
        manager._truth_layout_portal_segment_clear = mock.Mock(
            return_value=([current, anchors[0]], {
                "reason": "truth_portal_aperture_and_furniture_clear"}))

        with mock.patch.object(
                baseline_manager_module.PathRefiner, "refine",
                return_value=failed), \
                mock.patch("builtins.open", mock.mock_open()):
            result = manager._refine_path(206, path, goal)

        self.assertEqual(result["execution_waypoints"], anchors)
        self.assertEqual(
            result["refinement_status"],
            "generated_truth_portal_after_stale_live_projection")
        self.assertTrue(
            manager.scan_history[-1]["generated_truth_portal_restored"])
        self.assertFalse(
            manager.scan_history[-1]["refinement_success"])
        self.assertEqual(
            manager.corridor_sweep_history[-1]["event"],
            "ROOM_ENTRY_GENERATED_TRUTH_PORTAL_ROUTE_RESTORED")

    def test_run66_truth_portal_uses_generated_physical_aperture_width(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.floor_number = 1
        manager.offline_truth_layout_metadata = "/tmp/layout_metadata.json"
        manager.truth_generated_room_doorway_width = 1.20
        manager._offline_truth_layout_cache = {"floors": [{"rooms": [{
            "id": "floor_0_room_1",
            "side": "right",
            "door_pose": [1.1, 14.865, 1.2, 0.0, 0.0, math.pi],
            "furniture": [],
        }]}]}
        active_door = SimpleNamespace(
            door_id="estimated_door_01",
            width=0.90,
            viewpoint_contract_locked_before_entry=True,
            truth_room_id="floor_0_room_1",
            truth_coordinate_transform="map_yaw_plus_90",
            truth_contract_match_separation_m=0.012)
        manager.room_scheduler = SimpleNamespace(
            active_door=active_door,
            config=SimpleNamespace(truth_portal_snap_max_separation=0.70))
        goal = {"estimated_door_id": "estimated_door_01"}

        samples, detail = manager._truth_layout_portal_segment_clear(
            (14.865, -0.20), (14.865, -1.10), goal)

        self.assertTrue(samples)
        self.assertAlmostEqual(detail["measured_door_width_m"], 0.90)
        self.assertAlmostEqual(
            detail["physical_truth_aperture_width_m"], 1.20)
        self.assertAlmostEqual(detail["usable_half_width_m"], 0.32)

    def test_run67_truth_portal_allows_bounded_corridor_station_alignment(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.floor_number = 1
        manager.offline_truth_layout_metadata = "/tmp/layout_metadata.json"
        manager.truth_generated_room_doorway_width = 1.20
        manager._offline_truth_layout_cache = {"floors": [{"rooms": [{
            "id": "floor_0_room_0",
            "side": "left",
            "door_pose": [-1.1, 14.865, 1.2, 0.0, 0.0, 0.0],
            "furniture": [],
        }]}]}
        active_door = SimpleNamespace(
            door_id="estimated_door_02",
            width=1.40,
            viewpoint_contract_locked_before_entry=True,
            truth_room_id="floor_0_room_0",
            truth_coordinate_transform="map_yaw_plus_90",
            truth_contract_match_separation_m=0.658)
        manager.room_scheduler = SimpleNamespace(
            active_door=active_door,
            config=SimpleNamespace(truth_portal_snap_max_separation=0.70))
        goal = {"estimated_door_id": "estimated_door_02"}

        # run67: 1.423 m lateral station alignment while the complete body is
        # 1.866 m outside the door plane.  It must not be confused with a
        # diagonal aperture crossing.
        samples, detail = manager._truth_layout_portal_segment_clear(
            (13.4419, -0.7656), (14.865, -0.7656), goal)

        self.assertTrue(samples)
        self.assertTrue(detail["bounded_corridor_alignment_allowed"])
        self.assertAlmostEqual(detail["length_m"], 1.423, places=3)

    def test_run134_entry_chain_uses_corridor_audit_for_long_station_leg(self):
        def blocked(_x, _y, _yaw):
            return baseline_manager_module.FootprintCheck(
                map_available=True, occupied_collision=True,
                minimum_obstacle_clearance=0.0)

        current = (26.980, -0.804)
        anchors = [(28.895, -0.804), (28.895, -1.216),
                   (28.895, -1.966)]
        manager = self._exit_refinement_manager(
            blocked, (current[0], current[1], 0.0))
        manager.floor_number = 1
        manager.enable_scan_lite = True
        manager.offline_truth_layout_metadata = "/tmp/layout_metadata.json"
        manager.allow_truth_exploration_when_registration_lost = True
        active_door = SimpleNamespace(
            door_id="estimated_door_03", truth_room_id="floor_0_room_3",
            truth_coordinate_transform="map_yaw_plus_90",
            truth_contract_match_separation_m=1.147,
            truth_contract_match_tangent_separation_m=1.141,
            truth_contract_match_normal_separation_m=0.116,
            viewpoint_contract_locked_before_entry=True)
        manager.room_scheduler = SimpleNamespace(
            active_door=active_door,
            config=SimpleNamespace(
                truth_portal_snap_max_separation=0.75,
                truth_portal_tangent_repair_max_separation=1.35,
                truth_portal_normal_repair_max_separation=0.35))
        manager._live_truth_contract_pose = mock.Mock(
            return_value=(current[0], current[1], 0.31, 0.0))
        goal = {
            "source": "lightweight_room_entry",
            "room_role": "ENTRY",
            "room_id": "estimated_room_03",
            "estimated_door_id": active_door.door_id,
            "position": [anchors[-1][0], anchors[-1][1], 0.31],
            "mandatory_portal_waypoints": anchors,
            "portal_preflight_verified": True,
            "room_goal_diagnostic": {
                "truth_portal_center_aligned": True,
            },
        }
        path = {"success": True, "reason": "portal_path_found",
                "path": [current] + anchors}
        portal_clear = ([anchors[0], anchors[1]], {
            "reason": "truth_portal_aperture_and_furniture_clear"})
        manager._truth_layout_portal_segment_clear = mock.Mock(
            side_effect=[
                (None, {"reason": "sample_outside_truth_portal_envelope"}),
                portal_clear, portal_clear,
            ])
        manager._truth_layout_corridor_segment_clear = mock.Mock(
            return_value=([current, anchors[0]], {
                "reason": "truth_corridor_bounds_and_furniture_clear"}))

        with mock.patch("builtins.open", mock.mock_open()):
            result = manager._refine_path(134, path, goal)

        self.assertEqual(result["execution_waypoints"], anchors)
        self.assertEqual(
            result["refinement_status"],
            "generated_truth_portal_immutable_route")
        manager._truth_layout_corridor_segment_clear.assert_called_once_with(
            current, anchors[0])
        event = next(
            item for item in manager.corridor_sweep_history
            if item.get("event") ==
            "ROOM_ENTRY_GENERATED_TRUTH_PORTAL_CHAIN_PREVALIDATED")
        self.assertEqual(
            event["truth_details"][0]["entry_chain_leg"],
            "corridor_station_alignment")
        self.assertEqual(
            event["truth_details"][0]["portal_rejection"]["reason"],
            "sample_outside_truth_portal_envelope")

    def test_run68_truth_portal_rejects_large_door_match_offset(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.offline_truth_layout_metadata = "/tmp/layout_metadata.json"
        active_door = SimpleNamespace(
            door_id="estimated_door_run68",
            viewpoint_contract_locked_before_entry=True,
            truth_room_id="floor_0_room_1",
            truth_contract_match_separation_m=0.794)
        manager.room_scheduler = SimpleNamespace(
            active_door=active_door,
            config=SimpleNamespace(truth_portal_snap_max_separation=0.70))

        samples, detail = manager._truth_layout_portal_segment_clear(
            (13.44, -0.76), (14.865, -0.76),
            {"estimated_door_id": "estimated_door_run68"})

        self.assertIsNone(samples)
        self.assertEqual(
            detail["reason"],
            "truth_portal_match_separation_too_large")

    def test_run72_truth_portal_accepts_bounded_tangent_edge_repair(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.floor_number = 1
        manager.offline_truth_layout_metadata = "/tmp/layout_metadata.json"
        manager.truth_generated_room_doorway_width = 1.20
        manager._offline_truth_layout_cache = {"floors": [{"rooms": [{
            "id": "floor_0_room_1", "side": "right",
            "door_pose": [1.1, 14.865, 1.2, 0.0, 0.0, math.pi],
            "furniture": [],
        }]}]}
        active_door = SimpleNamespace(
            door_id="estimated_door_run72", width=0.90,
            center=(13.823, -1.227),
            viewpoint_contract_locked_before_entry=True,
            truth_room_id="floor_0_room_1",
            truth_coordinate_transform="map_yaw_plus_90",
            truth_contract_match_separation_m=1.05,
            truth_contract_match_tangent_separation_m=1.042,
            truth_contract_match_normal_separation_m=0.127)
        manager.room_scheduler = SimpleNamespace(
            active_door=active_door,
            config=SimpleNamespace(
                truth_portal_snap_max_separation=0.70,
                truth_portal_tangent_repair_max_separation=1.35,
                truth_portal_normal_repair_max_separation=0.35))

        samples, detail = manager._truth_layout_portal_segment_clear(
            (13.823, -0.327), (14.865, -0.327),
            {"estimated_door_id": "estimated_door_run72"})

        self.assertTrue(samples)
        self.assertEqual(detail["truth_room_id"], "floor_0_room_1")

    def test_run116_corridor_resume_bounds_live_start_scan_checks(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.enable_scan_lite = True
        manager.room_two_pose_waypoint_simplification = False
        manager.scan_config = baseline_manager_module.RefinerConfig()
        manager.waypoint_spacing = 0.30
        manager.lock = threading.RLock()
        manager.pose = (0.0, 0.0, 2.4, 0.0)
        manager.trajectory = [
            (0.0, -0.10, 0.0, 2.4, 0.0),
            (1.0, 0.0, 0.0, 2.4, 0.0),
        ]
        manager.fastlio_registration_healthy = True
        manager.corridor_anchor = np.asarray([0.0, 0.0])
        manager.corridor_axis = np.asarray([1.0, 0.0])
        manager.scan_history = []
        manager.scan_duration_pub = mock.Mock()
        manager.output_dir = "/tmp"
        manager.elapsed = mock.Mock(return_value=1.0)
        occupied = baseline_manager_module.FootprintCheck(
            map_available=True,
            occupied_collision=True,
            unknown_queries=0,
            occupied_queries=1,
            status="stale_live_start_collision")
        clear = baseline_manager_module.FootprintCheck(
            map_available=True, occupied_collision=False,
            unknown_queries=0, occupied_queries=0,
            minimum_obstacle_clearance=0.40, status="clear")
        manager._scan_check = mock.Mock(
            side_effect=[occupied] + [clear] * 32)

        with mock.patch("builtins.open", mock.mock_open()):
            result = manager._refine_path(
                12,
                {"success": True, "path": [(0.0, 0.0), (0.55, 0.0)]},
                {"source": "corridor_resume_centerline",
                 "position": [0.55, 0.0, 2.4]})

        self.assertTrue(result["success"])
        self.assertLess(manager._scan_check.call_count, 20)
        self.assertTrue(manager.scan_history)
        record = manager.scan_history[-1]
        self.assertTrue(record["corridor_start_footprint_override"])
        self.assertEqual(record["corridor_start_footprint_override_reason"],
                         "healthy_continuous_live_corridor_pose")


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

    def test_run110_handoff_hint_flips_fresh_axis_out_of_stair_lobby(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.corridor_axis = None
        manager.corridor_forward_axis_hint = np.asarray([
            np.cos(-0.23), np.sin(-0.23)])
        oriented = manager._orient_corridor_axis_to_forward_hint(
            np.asarray([-1.0, 0.0]))
        self.assertGreater(
            float(np.dot(oriented, manager.corridor_forward_axis_hint)), 0.0)
        self.assertGreater(oriented[0], 0.0)

    def test_established_axis_can_reverse_for_terminal_return(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.corridor_axis = np.asarray([1.0, 0.0])
        manager.corridor_forward_axis_hint = np.asarray([1.0, 0.0])
        reversed_axis = manager._orient_corridor_axis_to_forward_hint(
            np.asarray([-1.0, 0.0]))
        self.assertLess(reversed_axis[0], 0.0)

    def test_runtime_truth_floor_loss_cancels_active_floor_manager(self):
        mission = SecondFloorMission.__new__(SecondFloorMission)
        mission._lock = threading.RLock()
        mission._floor_word = "SECOND"
        mission._truth_pose = None
        mission._truth_pose_received_at = -1.0
        mission._second_floor_elevation = 2.6
        mission._truth_floor_loss_margin = 0.05
        mission._truth_floor_loss_required_samples = 3
        mission._truth_floor_loss_count = 0
        mission._truth_floor_loss_reported = False
        mission._active_manager = mock.Mock()
        mission._truth_correction_pub = mock.Mock()
        mission._cmd_pub = mock.Mock()
        mission._state_pub = mock.Mock()
        mission._append_localization_diagnostic = mock.Mock()
        pose = SimpleNamespace(
            position=SimpleNamespace(x=1.0, y=2.0, z=1.9),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0))
        twist = SimpleNamespace(
            linear=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            angular=SimpleNamespace(x=0.0, y=0.0, z=0.0))
        message = SimpleNamespace(
            name=["a1_gazebo"], pose=[pose], twist=[twist])
        with mock.patch(
                "second_floor_exploration_manager.rospy.Time.now",
                return_value=mock.Mock()):
            for _ in range(3):
                mission._on_truth_states(message)
        mission._active_manager.request_external_abort.assert_called_once_with(
            "SECOND_FLOOR_TRUTH_FLOOR_HEIGHT_LOST")
        mission._cmd_pub.publish.assert_called_once()
        mission._append_localization_diagnostic.assert_called_once()

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
        overrides = SecondFloorMission._second_floor_only_parameter_overrides(
            unified_fast_profile=True)
        self.assertEqual(overrides["corridor_door_post_latch_motion_m"], 0.35)
        self.assertEqual(
            overrides["first_room_minimum_corridor_station_m"], -0.25)
        self.assertNotIn(
            "corridor_forward_recovery_minimum_advance_m", overrides)
        self.assertNotIn("corridor_forward_lateral_search_m", overrides)
        self.assertNotIn("room_require_portal_preflight_for_entry", overrides)
        self.assertNotIn("room_entry_retry_limit", overrides)
        self.assertTrue(
            overrides["corridor_short_door_commit_after_establishment"])
        self.assertNotIn(
            "enforce_room_phase_target_deadline", overrides)
        self.assertTrue(
            overrides["corridor_guided_entry_door_context"])
        self.assertAlmostEqual(
            overrides["corridor_guided_entry_minimum_progress_m"], 0.5)
        self.assertAlmostEqual(
            overrides["corridor_door_forward_commit_distance_m"], 1.5)
        for shared_speed_parameter in (
                "corridor_transit_speed",
                "corridor_lateral_speed_limit",
                "corridor_maximum_yaw_rate",
                "room_entry_speed",
                "room_lateral_speed_limit",
                "room_exit_speed",
                "room_exit_lateral_speed_limit",
                "room_fast_exit_approach_speed"):
            self.assertNotIn(shared_speed_parameter, overrides)
        self.assertAlmostEqual(
            overrides["local_door_minimum_semantic_width_m"], 0.60)
        self.assertEqual(overrides["planner_timeout"], 8.0)
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
        self.assertEqual(
            overrides["room_reversed_entry_trace_corridor_extension_m"],
            0.0)
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
        self.assertNotIn(
            "terminal_return_blocked_by_known_unvisited_door", overrides)

    def test_guided_upper_floor_arms_from_first_confirmed_door(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.corridor_door_detection_armed = False
        manager.corridor_established = False
        manager.corridor_entry_forward_lock = False
        manager.corridor_entry_anchor = None
        manager.corridor_entry_elapsed_sec = None
        manager.corridor_entry_source = None
        manager.corridor_guided_entry_door_context = True
        manager.corridor_forward_axis_hint = np.asarray([1.0, 0.0])
        manager.mission_start_pose = (0.0, 0.0, 2.46, 0.0)
        manager.corridor_local_door_entry_lock_minimum_progress = 2.0
        manager.corridor_guided_entry_minimum_progress = 3.5
        manager.corridor_centerline_membership_tolerance = 0.75
        manager.lock = threading.RLock()
        manager.local_entry_status = {"candidates": [{
            "confirmed": True,
            "astar_reachable": True,
            "scan_lite_safe": True,
            "width_m": 0.65,
            "unknown_behind_m2": 2.0,
            "free_behind_m2": 0.0,
        }]}
        manager.local_door_status = {}
        manager.local_door_minimum_semantic_width = 0.60
        manager.local_door_maximum_semantic_width = 1.50
        manager.local_door_minimum_unknown = 1.0
        manager.local_door_minimum_free = 6.0
        manager.corridor_confirmation_count = 0
        manager.corridor_confirmation_updates = 3
        manager.corridor_sweep_history = []
        manager.corridor_door_forward_commit_distance = 0.35
        manager.elapsed = lambda: 14.0

        armed = manager._update_corridor_door_phase(
            (3.6, 0.1, 2.46, 0.0), {
                "axis": [1.0, 0.0],
                "raw_is_corridor": False,
                "centerline_point": [3.6, 0.0],
            })

        self.assertFalse(armed)
        self.assertTrue(manager.corridor_established)
        self.assertTrue(manager.corridor_established_from_local_door)
        self.assertTrue(manager.corridor_entry_forward_lock)
        self.assertTrue(manager.corridor_door_forward_pending)
        self.assertFalse(manager.corridor_door_detection_armed)
        np.testing.assert_allclose(
            manager.corridor_station_origin, [0.0, 0.0])
        self.assertAlmostEqual(manager.corridor_door_arm_high_water, 3.6)
        self.assertTrue(
            manager.corridor_sweep_history[-2]["guided_entry_context"])

    def test_run108_first_real_door_pair_survives_lobby_throat_guard(self):
        manager = BaselineExplorationManager.__new__(BaselineExplorationManager)
        manager.corridor_station_axis = np.asarray([1.0, 0.0])
        manager.corridor_station_origin = np.asarray([0.0, 0.0])
        manager.corridor_forward_station_sign = 1.0
        manager.first_room_minimum_corridor_station = -0.25
        manager.room_scheduler = SimpleNamespace(detector=SimpleNamespace(doors=[]))
        manager._missing_room_retrace_active = lambda: False
        manager.initial_lobby_door_rejections = set()
        manager.corridor_sweep_history = []
        manager.elapsed = lambda: 67.83
        self.assertFalse(manager._reject_initial_lobby_throat((.173, 1.0), "left", "test"))
        self.assertFalse(manager._reject_initial_lobby_throat((-.047, -1.0), "right", "test"))
        self.assertTrue(manager._reject_initial_lobby_throat((-4.0, 1.0), "lobby", "test"))

    def test_missing_room_retrace_recovers_near_pair_but_rejects_lobby(self):
        manager = BaselineExplorationManager.__new__(BaselineExplorationManager)
        manager.corridor_station_axis = np.asarray([1.0, 0.0])
        manager.corridor_station_origin = np.asarray([0.0, 0.0])
        manager.corridor_forward_station_sign = 1.0
        manager.first_room_minimum_corridor_station = 2.0
        manager.return_room_retrace_minimum_corridor_station = -2.25
        manager.room_scheduler = SimpleNamespace(
            detector=SimpleNamespace(doors=[SimpleNamespace(visited=True)]))
        manager._missing_room_retrace_active = lambda: True
        manager.initial_lobby_door_rejections = set()
        manager.corridor_sweep_history = []
        manager.elapsed = lambda: 154.0

        self.assertFalse(manager._reject_initial_lobby_throat(
            (-1.70, 1.0), "near-room", "test"))
        self.assertTrue(manager._reject_initial_lobby_throat(
            (-3.20, 1.0), "lobby", "test"))
        self.assertEqual(
            manager.corridor_sweep_history[-1]["minimum_station_m"], -2.25)

    def test_run108_verified_short_forward_probe_bypasses_long_goal_filter(self):
        manager = BaselineExplorationManager.__new__(BaselineExplorationManager)
        manager.room_scheduler = SimpleNamespace(config=SimpleNamespace(enabled=True))
        manager.corridor_sweep_history = []
        manager.elapsed = lambda: 112.5
        goal = {
            "source": "corridor_sweep",
            "position": [37.69, -0.43, 2.46],
            "corridor_no_frontier_short_probe": True,
        }
        path = {"success": True, "path": [(37.2, -0.4), (37.69, -0.43)]}
        preserved_goal, preserved_path = manager._apply_corridor_sweep(6, goal, path)
        self.assertIs(preserved_goal, goal)
        self.assertIs(preserved_path, path)
        self.assertEqual(manager.corridor_sweep_history[-1]["event"], "CORRIDOR_NO_FRONTIER_SHORT_PROBE_PRESERVED")
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.last_room_exit_side = None
        manager.last_room_exit_pose = None
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

    def test_fix08_truth_probe_bypasses_generic_minimum_advance_filter(self):
        manager = BaselineExplorationManager.__new__(BaselineExplorationManager)
        manager.room_scheduler = SimpleNamespace(
            config=SimpleNamespace(enabled=True))
        manager.corridor_sweep_history = []
        manager.elapsed = lambda: 17.7
        goal = {
            "source": "upper_floor_truth_corridor_probe",
            "position": [-0.15, -0.84, 2.91],
            "corridor_truth_probe": True,
        }
        path = {"success": True,
                "path": [(-0.19, 0.36), (-0.15, -0.84)]}
        preserved_goal, preserved_path = manager._apply_corridor_sweep(
            6, goal, path)
        self.assertIs(preserved_goal, goal)
        self.assertIs(preserved_path, path)
        self.assertEqual(
            manager.corridor_sweep_history[-1]["event"],
            "UPPER_FLOOR_TRUTH_PROBE_PRESERVED")

    def test_f3_initial_truth_seed_bypasses_generic_minimum_advance_filter(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.room_scheduler = SimpleNamespace(
            config=SimpleNamespace(enabled=True))
        manager.corridor_sweep_history = []
        manager.elapsed = lambda: 0.3
        goal = {
            "source": "f3_initial_truth_corridor_seed",
            "position": [0.0, 14.415, 5.513],
            "corridor_truth_seed": True,
        }
        path = {
            "success": True,
            "path": [(0.0, 13.325), (0.0, 14.415)],
        }
        preserved_goal, preserved_path = manager._apply_corridor_sweep(
            1, goal, path)
        self.assertIs(preserved_goal, goal)
        self.assertIs(preserved_path, path)
        self.assertEqual(
            manager.corridor_sweep_history[-1]["event"],
            "F3_TRUTH_INITIAL_SEED_PRESERVED")

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

    def test_partial_room_cleanup_preserves_physical_exit_transactions(self):
        manager = SimpleNamespace(
            room_exit_map_snapshots=[
                {"snapshot_file": "room_exit_01_grid.npz"},
                {"snapshot_file": "room_exit_02_grid.npz"},
                {"snapshot_file": "room_exit_03_grid.npz"},
                {"snapshot_file": "room_exit_04_grid.npz"},
            ],
            room_scheduler=SimpleNamespace(events=[
                {"event": "ROOM_ENTERED", "room_id": "room_1"},
                {"event": "ROOM_GOAL_RESULT", "room_id": "room_1",
                 "role": "EXIT", "success": True},
            ]))
        self.assertEqual(
            SecondFloorMission._physical_room_exit_evidence_count(manager), 4)

    def test_f3_launch_defers_shutdown_to_descent_owner(self):
        launch = ET.parse(os.path.join(
            SCRIPT_DIR, "..", "launch",
            "baseline_fastlio_exploration.launch"))
        root = launch.getroot()
        node = next(
            item for item in root.findall("node")
            if item.get("name") == "third_floor_exploration_manager")
        params = {item.get("name"): item.get("value")
                  for item in node.findall("param")}
        self.assertEqual(
            params["defer_terminal_shutdown_to_descent"],
            "$(arg enable_third_floor_descent)")
        self.assertGreaterEqual(
            float(params["third_floor_stair_return_timeout_sec"]), 180.0)

    def test_f3_failure_does_not_defer_terminal_shutdown(self):
        source_path = os.path.join(
            SCRIPT_DIR, "second_floor_exploration_manager.py")
        with open(source_path, "r", encoding="utf-8") as stream:
            source = stream.read()
        self.assertIn(
            "handoff_authorized and self._floor_number >= 3 and\n"
            "            self._defer_terminal_shutdown_to_descent", source)

    def test_f2_f3_flight_a_top_recovery_adds_body_clearance(self):
        transition = StairTransition.__new__(StairTransition)
        transition.source_floor_index = 1
        transition.truth_f2_f3_flight_a_timeout_recoveries = 0
        transition.truth_f2_f3_flight_a_timeout_recovery_limit = 1
        transition.truth_f2_f3_physical_only_recovery = False
        transition.truth_flight_a_top = (-4.01, 4.52, 3.83, 0.0)
        transition.truth_f2_f3_flight_a_top_body_clearance = 0.38
        transition.truth_stair_heading = math.pi / 2.0
        transition.truth_flight_b_heading = -math.pi / 2.0
        transition.direction = math.pi / 2.0
        transition.truth_ascent_start_z = 2.91
        transition.truth_flight_a_peak_gain = 0.0
        transition.pose = (-4.01, 4.52, 4.21, math.pi / 2.0)
        transition.pub = mock.Mock()
        transition.state = mock.Mock()
        transition.truth_flight_b_bridge_policy = "plane"
        transition._truth_flight_b_center_x = mock.Mock(return_value=-3.72)
        transition._reset_truth_flight_b_progress_watchdog = mock.Mock()
        transition._record_trace = mock.Mock()
        response = SimpleNamespace(success=True)
        with mock.patch(
                "stair_transition_manager.rospy.wait_for_service"), mock.patch(
                "stair_transition_manager.rospy.ServiceProxy",
                return_value=mock.Mock(return_value=response)) as proxy:
            self.assertTrue(transition._recover_f2_f3_flight_a_timeout())
        state = proxy.return_value.call_args.args[0]
        self.assertAlmostEqual(state.pose.position.z, 4.21, places=6)
        self.assertAlmostEqual(state.pose.position.x, -3.72, places=6)
        self.assertAlmostEqual(
            state.pose.orientation.z, math.sin(-math.pi / 4.0), places=6)
        self.assertEqual(transition.phase, "STAIR_ASCENT_B")
        transition._reset_truth_flight_b_progress_watchdog.assert_called_once()

    def test_open_room_stale_map_collision_has_truth_bounded_fallback(self):
        source_path = os.path.join(
            SCRIPT_DIR, "baseline_exploration_manager.py")
        with open(source_path, "r", encoding="utf-8") as stream:
            source = stream.read()
        block = source[source.index(
            "truth_open_semantic_stale_map_fallback = bool("):
            source.index("        stale_two_pose_direct_disconnect = bool(")]
        self.assertIn('source == "lightweight_room_semantic"', block)
        self.assertIn('room_role in ("G3", "G4")', block)
        self.assertIn("_truth_layout_open_room_segment_clear", block)
        self.assertIn(
            "OPEN_ROOM_STALE_MAP_COLLISION_TRUTH_ROUTE_OVERRIDE", block)
        self.assertNotIn("obstacle_front_opposite_sides", block)

    def test_deep_obstacle_stale_map_collision_keeps_strict_gap_contract(self):
        source_path = os.path.join(
            SCRIPT_DIR, "baseline_exploration_manager.py")
        with open(source_path, "r", encoding="utf-8") as stream:
            source = stream.read()
        block = source[source.index(
            "truth_obstacle_semantic_stale_map_fallback = bool("):
            source.index(
                "        truth_open_semantic_stale_map_fallback = bool(")]
        self.assertIn('room_role in ("G3", "G4")', block)
        self.assertIn("truth_disconnected_obstacle_gap_path", block)
        self.assertIn('"deep_outer_side_peek"', block)
        self.assertIn(
            "_truth_layout_obstacle_gap_segment_clear", block)
        self.assertIn(
            "OBSTACLE_GAP_STALE_MAP_COLLISION_TRUTH_ROUTE_OVERRIDE",
            block)
        self.assertIn('"physical_room_credit": False', block)
        self.assertIn("execution_waypoints", block)

    def test_truth_open_deep_view_keeps_inflated_far_wall_margin(self):
        source_path = os.path.join(
            SCRIPT_DIR, "baseline_exploration_manager.py")
        with open(source_path, "r", encoding="utf-8") as stream:
            source = stream.read()
        self.assertIn("max(0.0, room_depth - 1.25)", source)
        self.assertIn("                    5.70,", source)

    def test_failure_token_precedes_visualization(self):
        source_path = os.path.join(
            SCRIPT_DIR, "second_floor_exploration_manager.py")
        with open(source_path, "r", encoding="utf-8") as stream:
            source = stream.read()
        method = source[source.index("    def _publish_exploration_failure"):
                        source.index("    def _seed_baseline_corridor_entry_anchor")]
        self.assertLess(
            method.index("self._state_pub.publish(String(data=generic_state))"),
            method.index("self._generate_combined_visualization()"))

    def test_f3_corridor_end_return_is_not_gated_by_room_count(self):
        source_path = os.path.join(
            SCRIPT_DIR, "second_floor_exploration_manager.py")
        with open(source_path, "r", encoding="utf-8") as stream:
            source = stream.read()
        self.assertIn(
            "(handoff_authorized and strict_four_rooms) or\n"
            "                bounded_corridor_end_return", source)
        self.assertNotIn(
            "physical_exit_evidence_count >=\n"
            "                self._partial_floor_handoff_minimum_exited_rooms and\n"
            "                (manager.corridor_terminal_return_latched", source)

    def test_f3_partial_quality_bypasses_redundant_anchor_return(self):
        source_path = os.path.join(
            SCRIPT_DIR, "second_floor_exploration_manager.py")
        with open(source_path, "r", encoding="utf-8") as stream:
            source = stream.read()
        recovery = source[source.index(
            "# Flow-continuity recovery"):
            source.index("handoff_authorized =", source.index(
                "# Flow-continuity recovery"))]
        self.assertIn("self._floor_number == 2", recovery)
        self.assertIn("f3_physical_return_continuity", recovery)
        self.assertIn("self._f3_physical_return_safe", recovery)
        self.assertNotIn(
            "self._floor_number in (2, 3)", recovery)

    def test_f3_physical_return_safety_is_not_room_quality_gate(self):
        mission = SecondFloorMission.__new__(SecondFloorMission)
        mission._floor_number = 3
        mission._second_floor_elevation = 5.35
        mission._second_floor_height_margin = 0.8
        mission._truth_pose = (0.0, 25.0, 5.50, 0.0)
        mission._lock = mock.MagicMock()
        mission._truth_height_is_second_floor = mock.Mock(return_value=True)
        manager = SimpleNamespace(
            corridor_established=True,
            room_scheduler=SimpleNamespace(active_door=None))

        self.assertTrue(mission._f3_physical_return_safe(manager, 0))
        mission._truth_height_is_second_floor.assert_called_once()

        manager.room_scheduler.active_door = SimpleNamespace()
        self.assertFalse(mission._f3_physical_return_safe(manager, 3))

    def test_f3_stair_return_never_backtracks_to_corridor_station(self):
        source_path = os.path.join(
            SCRIPT_DIR, "second_floor_exploration_manager.py")
        with open(source_path, "r", encoding="utf-8") as stream:
            source = stream.read()
        self.assertIn(
            "if float(initial[1]) > corridor_y + 0.50:", source)
        self.assertIn(
            "targets.extend([\n"
            "            (clear_x, landing_y, \"corridor_reverse_to_landing\")",
            source)
        self.assertIn('"stair_wait_lip_pre_align"', source)
        self.assertIn(
            '0.70 if stage == "stair_wait_lip_pre_align" else 0.50',
            source)
        self.assertIn(
            '0.03 if stage == "stair_wait_lip_pre_align" else 0.08',
            source)

    def test_f3_stair_wait_is_bounded_handoff_zone(self):
        source_path = os.path.join(
            SCRIPT_DIR, "second_floor_exploration_manager.py")
        with open(source_path, "r", encoding="utf-8") as stream:
            source = stream.read()
        method = source[
            source.index("    def _third_floor_return_to_stair_wait"):
            source.index("    def _quiesce_exploration_executor_for_f3_return")]
        self.assertIn(
            'third_floor_stair_wait_handoff_tolerance_m', method)
        self.assertIn(
            'stair_wait_handoff_tolerance = max(0.55, min(0.75', method)
        self.assertIn(
            'final_distance <= stair_wait_handoff_tolerance and upright',
            method)
        self.assertNotIn('final_distance <= 0.55', method)

    def test_locked_truth_open_g4_is_not_reclassified_as_g3(self):
        source_path = os.path.join(SCRIPT_DIR, "lightweight_room_core.py")
        with open(source_path, "r", encoding="utf-8") as stream:
            source = stream.read()
        self.assertIn(
            'str(proposal.get("two_pose_strategy", "")) !=\n'
            '                    "truth_layout_open_centerline_deep_near"',
            source)

    def test_f3_stair_return_uses_forward_only_policy_commands(self):
        source_path = os.path.join(
            SCRIPT_DIR, "second_floor_exploration_manager.py")
        with open(source_path, "r", encoding="utf-8") as stream:
            source = stream.read()
        method = source[
            source.index("    def _third_floor_return_to_stair_wait"):
            source.index("    def _truth_return_to_corridor_anchor")]
        self.assertIn("heading_gate = max(0.0, math.cos(yaw_error))", method)
        self.assertIn("if abs(yaw_error) > 0.30:", method)
        self.assertIn("prealign_heading,", method)
        self.assertNotIn("self._f3_stationary_turn_assist(", method)
        self.assertIn('"stationary_prealign_skipped": True', method)
        self.assertIn("to_sec() < 0.20", method)
        self.assertIn("_publish_f3_world_command(", method)
        self.assertIn("forward_speed * math.cos(pose[3])", method)
        self.assertIn("forward_speed * math.sin(pose[3])", method)
        self.assertNotIn("speed * dx / max(distance, 1e-6)", method)

    def test_run152_every_return_leg_uses_one_bounded_yaw_controller(self):
        source = inspect.getsource(
            SecondFloorMission._third_floor_return_to_stair_wait)
        self.assertIn(
            '"action": "bounded_route_leg_yaw_closure"', source)
        self.assertIn('"stationary_prealign_skipped": True', source)
        self.assertNotIn("_f3_stationary_turn_assist(", source)
        self.assertNotIn(
            '"reason": "f3_stair_return_prealign_failed"', source)
        self.assertIn("if abs(yaw_error) > 0.30:", source)
        self.assertIn("forward_speed = 0.0", source)

    def test_run152_post_entry_replan_is_one_fresh_grid_event(self):
        launch_path = os.path.abspath(os.path.join(
            SCRIPT_DIR, "..", "launch",
            "baseline_fastlio_exploration.launch"))
        root = ET.parse(launch_path).getroot()
        argument = next(item for item in root.findall("arg")
                        if item.get("name") ==
                        "first_floor_room_entry_replan_minimum_grid_updates")
        self.assertEqual(argument.get("default"), "1")
        parameter = next(item for item in root.findall(".//param")
                         if item.get("name") ==
                         "room_entry_replan_minimum_grid_updates")
        self.assertEqual(
            parameter.get("value"),
            "$(arg first_floor_room_entry_replan_minimum_grid_updates)")

    def test_run152_upper_floor_planar_ceiling_preserves_role_limits(self):
        launch_path = os.path.abspath(os.path.join(
            SCRIPT_DIR, "..", "launch",
            "baseline_fastlio_exploration.launch"))
        root = ET.parse(launch_path).getroot()
        arguments = {item.get("name"): item.get("default")
                     for item in root.findall("arg")}
        self.assertEqual(arguments["upper_floor_goal_maximum_speed"], "1.35")
        executor = next(item for item in root.findall(".//include")
                        if item.get("file", "").endswith(
                            "goal_executor.launch"))
        forwarded = {item.get("name"): item.get("value")
                     for item in executor.findall("arg")}
        self.assertEqual(
            forwarded["upper_floor_maximum_speed"],
            "$(arg upper_floor_goal_maximum_speed)")
        public_root = ET.parse(os.path.join(
            SCRIPT_DIR, "..", "launch",
            "fuel_semantic_fastlio_exploration.launch")).getroot()
        public_arguments = {item.get("name"): item.get("default")
                            for item in public_root.findall("arg")}
        self.assertLessEqual(
            float(arguments["upper_floor_goal_maximum_speed"]),
            float(public_arguments[
                "goal_executor_room_transaction_degraded_speed_cap"]))

    def test_run150_f3_north_seed_recenters_before_west_jamb(self):
        bounds = {"x_min": -1.10, "x_max": 1.10}
        stair = {"y_max": 6.85}
        self.assertEqual(
            SecondFloorMission._f3_north_seed_lateral_speed(
                (-1.12, 5.80, 5.51, math.pi / 2.0), bounds, stair),
            0.0)
        self.assertGreaterEqual(
            SecondFloorMission._f3_north_seed_lateral_speed(
                (-1.12, 6.82, 5.51, math.pi / 2.0), bounds, stair),
            0.30)
        self.assertLessEqual(
            SecondFloorMission._f3_north_seed_lateral_speed(
                (1.02, 7.20, 5.51, math.pi / 2.0), bounds, stair),
            -0.30)
        self.assertEqual(
            SecondFloorMission._f3_north_seed_lateral_speed(
                (0.02, 8.20, 5.51, math.pi / 2.0), bounds, stair),
            0.0)

    def test_f3_long_corridor_return_has_separate_bounded_speed_cap(self):
        source_path = os.path.join(
            SCRIPT_DIR, "second_floor_exploration_manager.py")
        with open(source_path, "r", encoding="utf-8") as stream:
            source = stream.read()
        method = source[
            source.index("    def _third_floor_return_to_stair_wait"):
            source.index("    def _truth_return_to_corridor_anchor")]
        self.assertIn("third_floor_stair_return_corridor_speed_mps", source)
        self.assertIn("third_floor_stair_return_lip_speed_mps", source)
        self.assertIn(
            'if stage == "corridor_reverse_to_landing" else', method)
        self.assertIn("speed = min(speed_cap, max(0.25, 0.70 * distance))",
                      method)

    def test_f3_return_quiesces_executor_before_truth_cmd_ownership(self):
        source_path = os.path.join(
            SCRIPT_DIR, "second_floor_exploration_manager.py")
        with open(source_path, "r", encoding="utf-8") as stream:
            source = stream.read()
        finalize = source[
            source.index("        strict_room_contract ="):
            source.index("        partial_handoff = bool(")]
        self.assertLess(
            finalize.index("_quiesce_exploration_executor_for_f3_return"),
            finalize.index("_third_floor_return_to_stair_wait"))
        self.assertIn("manager.active_goal_pub.publish(Int32(data=-1))", source)
        self.assertIn("F3_EXPLORATION_EXECUTOR_QUIESCED", source)

    def test_upper_stair_trigger_requires_exact_success_token(self):
        token = "SECOND_FLOOR_EXPLORATION_COMPLETE"
        self.assertTrue(StairTransition._mission_trigger_matches(token, token))
        self.assertFalse(StairTransition._mission_trigger_matches(
            token + "_DIAGNOSTIC", token))
        self.assertFalse(StairTransition._mission_trigger_matches(
            "SECOND_FLOOR_EXPLORATION_FAILED", token))

    def test_f2_upper_landing_uses_strict_east_clear_and_bounded_north_legs(self):
        source_path = os.path.join(
            SCRIPT_DIR, "second_floor_exploration_manager.py")
        with open(source_path, "r", encoding="utf-8") as stream:
            source = stream.read()
        route = source[
            source.index("    def _second_floor_truth_route"):
            source.index("    def _truth_reposition_to_corridor")]
        guide = source[
            source.index("    def _guide_to_second_floor_corridor"):
            source.index("    def _wait_for_real_locomotion_ready")]
        self.assertIn('opening_clear_y = float(stair["y_max"]) + 0.35', route)
        self.assertNotIn('north_mid_y = landing_y + 0.5 * (', route)
        self.assertIn('north_targets = [opening_clear_y, north_y]', route)
        self.assertIn('"upper_landing_north_step_{:02d}"', route)
        self.assertNotIn('"stage": "upper_landing_north_clear"', route)
        self.assertIn('stage == "upper_landing_east_clear"', guide)
        self.assertIn('self._east_landing_clear_arrival(', guide)
        self.assertIn('stage.startswith("upper_landing_north_step_")', guide)

    def test_truth_guide_body_envelope_preserves_direction_while_bounding_reverse(self):
        forward, lateral, scale = SecondFloorMission._bounded_truth_body_command(
            -1.465, -0.643, 1.05, 0.70, 0.55, 1.05)
        self.assertAlmostEqual(forward, -0.70, places=6)
        self.assertAlmostEqual(lateral / forward, -0.643 / -1.465, places=6)
        self.assertLess(scale, 1.0)
        self.assertLessEqual(abs(lateral), 0.55)
        self.assertLessEqual(math.hypot(forward, lateral), 1.05)

    def test_truth_guide_body_envelope_keeps_safe_forward_command(self):
        bounded = SecondFloorMission._bounded_truth_body_command(
            0.80, 0.10, 1.05, 0.70, 0.55, 1.05)
        self.assertEqual(bounded, (0.80, 0.10, 1.0))

    def test_upper_floor_map_is_prepared_before_guide_and_not_cleared_twice(self):
        source_path = os.path.join(
            SCRIPT_DIR, "second_floor_exploration_manager.py")
        with open(source_path, "r", encoding="utf-8") as stream:
            source = stream.read()
        run = inspect.getsource(SecondFloorMission._run_mission)
        prepare = inspect.getsource(
            SecondFloorMission._prepare_upper_floor_live_map_for_ingress)
        lobby = inspect.getsource(SecondFloorMission._map_second_floor_lobby)
        self.assertLess(
            run.index("_prepare_upper_floor_live_map_for_ingress()"),
            run.index("_guide_to_second_floor_corridor()"))
        self.assertIn("self._voxel_clear_pub.publish", prepare)
        self.assertIn("self._navigation_height_reanchor_pub.publish", prepare)
        self.assertNotIn("set_model_state", prepare)
        self.assertIn("if self._live_map_prepared_before_guide:", lobby)
        self.assertIn('CORRIDOR_MAP_READY_FROM_INGRESS', lobby)
        self.assertIn('map_fill_mode="physical_ingress_callbacks"', lobby)

    def test_f3_corridor_guide_combines_heading_with_translation(self):
        source_path = os.path.join(
            SCRIPT_DIR, "second_floor_exploration_manager.py")
        with open(source_path, "r", encoding="utf-8") as stream:
            source = stream.read()
        guide = source[
            source.index("    def _guide_f3_stair_platform_to_corridor"):
            source.index("    def _guide_to_second_floor_corridor")]
        stages = guide[
            guide.index("        stages = ("):
            guide.index("        for stage, mode, target in stages:")]
        self.assertIn('(\"stair_exit_drive_out\", \"east\", bridge_x)', stages)
        self.assertIn('(\"corridor_seed_drive\", \"north\", ingress_y)', stages)
        self.assertNotIn('stair_exit_turn', stages)
        self.assertNotIn('stair_exit_face_corridor', stages)
        self.assertIn('expected_heading = 0.0 if mode == "east" else math.pi / 2.0',
                      guide)
        self.assertIn('_publish_f3_world_command(world_x, world_y, yaw_rate, pose)',
                      guide)

    def test_run136_supported_east_landing_pose_is_accepted(self):
        self.assertTrue(SecondFloorMission._east_landing_clear_arrival(
            (-1.319, 1.50, 2.913), -0.90, 1.55, 0.60))

    def test_run131_stair_seam_pose_is_rejected(self):
        self.assertFalse(SecondFloorMission._east_landing_clear_arrival(
            (-1.44, 1.55, 2.913), -0.90, 1.55, 0.60))

    def test_east_landing_exact_pose_is_accepted(self):
        self.assertTrue(SecondFloorMission._east_landing_clear_arrival(
            (-0.90, 1.55, 2.913), -0.90, 1.55, 0.60))

    def test_east_landing_large_lateral_error_is_rejected(self):
        self.assertFalse(SecondFloorMission._east_landing_clear_arrival(
            (-0.90, 1.91, 2.913), -0.90, 1.55, 0.60))

    def test_run137_supported_f3_bridge_pose_is_accepted(self):
        bounds = {"x_min": -1.10, "x_max": 1.10}
        self.assertTrue(SecondFloorMission._f3_east_bridge_arrival(
            (-0.92, 1.19, 5.52), -0.75, bounds))

    def test_run138_supported_f3_bridge_pose_is_accepted(self):
        bounds = {"x_min": -1.10, "x_max": 1.10}
        self.assertTrue(SecondFloorMission._f3_east_bridge_arrival(
            (-1.01, 1.23, 5.52), -0.75, bounds))

    def test_f3_bridge_pose_too_close_to_west_seam_is_rejected(self):
        bounds = {"x_min": -1.10, "x_max": 1.10}
        self.assertFalse(SecondFloorMission._f3_east_bridge_arrival(
            (-1.06, 1.19, 5.52), -0.75, bounds))

    def test_f3_bridge_pose_far_short_of_target_is_rejected(self):
        bounds = {"x_min": -1.30, "x_max": 1.10}
        self.assertFalse(SecondFloorMission._f3_east_bridge_arrival(
            (-1.14, 1.19, 5.52), -0.75, bounds))

    def test_run114_pre_handoff_wait_can_be_unbounded(self):
        self.assertFalse(SecondFloorMission._handoff_wait_expired(
            0.0, None, 0.0, 1200.0, 5000.0))
        self.assertFalse(SecondFloorMission._handoff_wait_expired(
            0.0, 100.0, 0.0, 1200.0, 1299.9))
        self.assertTrue(SecondFloorMission._handoff_wait_expired(
            0.0, 100.0, 0.0, 1200.0, 1300.0))

    def test_third_floor_waiter_does_not_expire_during_f2_exploration(self):
        launch = ET.parse(os.path.join(
            SCRIPT_DIR, "..", "launch",
            "baseline_fastlio_exploration.launch"))
        root = launch.getroot()
        node = next(
            item for item in root.findall("node")
            if item.get("name") == "third_floor_exploration_manager")
        params = {item.get("name"): item.get("value")
                  for item in node.findall("param")}
        args = {item.get("name"): item.get("default")
                for item in root.findall("arg")}
        self.assertEqual(
            node.get("required"),
            "$(eval not arg('enable_third_floor_descent'))")
        self.assertEqual(
            params["pre_handoff_wait_timeout_sec"],
            "$(arg third_floor_pre_handoff_wait_timeout_sec)")
        self.assertEqual(
            args["third_floor_pre_handoff_wait_timeout_sec"], "0.0")
        fuel_launch = ET.parse(os.path.join(
            SCRIPT_DIR, "..", "launch",
            "fuel_semantic_fastlio_exploration.launch"))
        fuel_root = fuel_launch.getroot()
        fuel_args = {item.get("name"): item.get("default")
                     for item in fuel_root.findall("arg")}
        baseline_include = next(
            item for item in fuel_root.findall("include")
            if item.get("file", "").endswith(
                "baseline_fastlio_exploration.launch"))
        forwarded = {item.get("name"): item.get("value")
                     for item in baseline_include.findall("arg")}
        self.assertEqual(
            fuel_args["third_floor_pre_handoff_wait_timeout_sec"], "0.0")
        self.assertEqual(forwarded[
            "third_floor_pre_handoff_wait_timeout_sec"],
            "$(arg third_floor_pre_handoff_wait_timeout_sec)")

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

    def test_terminal_return_policy_rejects_partial_room_count(self):
        self.assertFalse(SecondFloorMission._exploration_handoff_authorized(
            "STAIR_CORRIDOR_EXIT_HANDOFF_TRUTH_GATE", 3, 4, True))
        self.assertFalse(SecondFloorMission._exploration_handoff_authorized(
            "STAIR_CORRIDOR_EXIT_HANDOFF_TRUTH_GATE", 3, 4, False))
        self.assertFalse(SecondFloorMission._exploration_handoff_authorized(
            "STAIR_CORRIDOR_EXIT_HANDOFF_TRUTH_GATE", 3, 3, True))

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
        self.assertFalse(
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
        manager.corridor_no_frontier_exhaustion_streak = 0
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
        manager.corridor_far_pair_search_target_advance = 0.0
        manager.corridor_far_pair_search_after_exits = 2
        manager.corridor_forward_recovery_minimum_advance = 0.35
        manager.corridor_forward_lateral_search = 0.0
        manager.corridor_minimum_advance = 2.5
        manager.clearance = 0.38
        manager.reached_tolerance = 0.15
        manager.corridor_reversed = False
        manager.corridor_no_frontier_exhaustion_streak = 0
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
        overrides = SecondFloorMission._second_floor_only_parameter_overrides(
            unified_fast_profile=True)
        for shared_parameter in (
                "corridor_far_pair_search_maximum_advance_m",
                "corridor_partial_return_minimum_outbound_progress_m",
                "corridor_partial_return_reserve_seconds",
                "corridor_partial_return_room_minimum_remaining_seconds"):
            self.assertNotIn(shared_parameter, overrides)

        continuity = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        continuity.corridor_partial_return_minimum_outbound_progress = 20.0
        continuity.corridor_partial_return_reserve = 160.0
        continuity.corridor_forward_station_high_water = 19.8
        continuity.maximum_duration = 400.0
        continuity.room_target_count = 4
        continuity.room_scheduler = SimpleNamespace(
            active_door=None, detector=SimpleNamespace(doors=[
                SimpleNamespace(visited=True) for _ in range(4)
            ]))
        continuity.elapsed = lambda: 200.0
        self.assertEqual(continuity._partial_corridor_return_trigger(),
                         "online_outbound_progress")
        continuity.corridor_forward_station_high_water = 5.0
        continuity.elapsed = lambda: 240.0
        self.assertEqual(continuity._partial_corridor_return_trigger(),
                         "next_floor_time_reserve")

    def test_run115_room_deadline_defers_until_four_exits_or_hard_limit(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.enforce_room_phase_target_deadline = True
        manager.room_phase_target_seconds = 150.0
        manager.room_phase_deadline_minimum_exited_rooms = 4
        manager.room_phase_deadline_hard_overrun = 150.0
        manager.room_phase_deadline_deferred_logged = False
        manager.corridor_sweep_history = []
        manager.room_scheduler = SimpleNamespace(
            first_room_started_at=10.0,
            detector=SimpleNamespace(doors=[
                SimpleNamespace(visited=index < 2) for index in range(4)
            ]))
        manager.elapsed = mock.Mock(return_value=170.0)

        self.assertIsNone(manager._partial_corridor_return_trigger())
        self.assertEqual(
            manager.corridor_sweep_history[-1]["event"],
            "ROOM_PHASE_DEADLINE_DEFERRED_FOR_MISSING_ROOMS")

        manager.elapsed.return_value = 310.0
        self.assertEqual(manager._partial_corridor_return_trigger(),
                         "room_phase_deadline")

    def test_strict_final_room_gets_one_bounded_completion_window(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.maximum_duration = 300.0
        manager.stair_return_grace_seconds = 90.0
        manager.strict_final_room_completion_grace_seconds = 90.0
        manager.stair_return_transit_announced = False
        manager.room_target_count = 4
        manager.elapsed = mock.Mock(return_value=322.0)
        active = SimpleNamespace(
            door_id="final-door", entered_at=321.399,
            visited=False, completed=False)
        manager.room_scheduler = SimpleNamespace(
            active_door=active,
            detector=SimpleNamespace(doors=[
                SimpleNamespace(visited=True),
                SimpleNamespace(visited=True),
                SimpleNamespace(visited=True),
                active,
            ]))

        self.assertAlmostEqual(
            manager._stair_return_deadline(), 411.399, places=3)

        # Re-entering the same partial room must not renew that window.
        active.entered_at = 399.0
        self.assertAlmostEqual(
            manager._stair_return_deadline(), 411.399, places=3)

        # The extension is strict-final-room-only; an early partial floor
        # keeps the configured hard deadline.
        manager.room_scheduler.detector.doors[2].visited = False
        self.assertEqual(manager._stair_return_deadline(), 300.0)


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

    def test_run65_f1_lobby_ingress_gate_rejects_early_corridor_lock(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.floor_number = 1
        manager.corridor_entry_distance_from_mission_start = 8.0
        manager.corridor_entry_forward_lock_distance = 1.0
        manager.mission_start_pose = (0.0, 0.0, 0.0, 0.0)

        ready, detail = manager._first_floor_lobby_ingress_gate(
            (2.39, 0.0, 0.0, 0.0))
        self.assertFalse(ready)
        self.assertAlmostEqual(detail["required_displacement_m"], 7.5)
        self.assertAlmostEqual(detail["measured_displacement_m"], 2.39)

        ready, detail = manager._first_floor_lobby_ingress_gate(
            (7.63, 0.0, 0.0, 0.0))
        self.assertTrue(ready)
        self.assertAlmostEqual(detail["measured_displacement_m"], 7.63)

    def test_run65_lobby_door_cannot_preempt_even_if_phase_latch_is_stale(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.floor_number = 1
        manager.corridor_entry_distance_from_mission_start = 8.0
        manager.corridor_entry_forward_lock_distance = 1.0
        manager.mission_start_pose = (0.0, 0.0, 0.0, 0.0)
        manager.enable_local_doorway_detector = True
        manager.corridor_door_detection_armed = True
        manager.room_scheduler = SimpleNamespace(active_door=None)
        manager.lock = threading.RLock()
        manager.local_entry_status = {"candidates": [{
            "candidate_id": "run65-lobby-false-door",
            "confirmed": True,
        }]}
        manager.pose = (2.39, 0.0, 0.0, 0.0)
        manager.grid = object()
        manager.last_local_entry_received = 0.5
        manager.local_candidate_freshness = 2.0
        manager.corridor_sweep_history = []
        manager.elapsed = mock.Mock(return_value=1.0)

        self.assertIsNone(manager._local_door_preemption_candidate())
        self.assertEqual(
            manager.corridor_sweep_history[-1]["event"],
            "F1_LOBBY_LOCAL_DOOR_PREEMPTION_REJECTED")

    def test_upper_floor_and_corridor_only_launch_bypass_f1_lobby_gate(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.floor_number = 2
        manager.corridor_entry_distance_from_mission_start = 8.0
        manager.mission_start_pose = (0.0, 0.0, 2.5, 0.0)
        ready, _ = manager._first_floor_lobby_ingress_gate(
            (0.1, 0.0, 2.5, 0.0))
        self.assertTrue(ready)

        manager.floor_number = 1
        manager.corridor_entry_distance_from_mission_start = 0.0
        ready, _ = manager._first_floor_lobby_ingress_gate(
            (0.1, 0.0, 0.0, 0.0))
        self.assertTrue(ready)

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

    def test_run58_exit_restage_returns_to_entry_without_exit_credit(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.allow_nonphysical_model_state_recovery = True
        manager._active_room_truth_anchor = {
            "door_id": "door-run58",
            "map_pose": (14.70, 1.55, 0.55, -0.70),
            "truth_pose": (-1.49, 15.37, 0.55, 0.87),
        }
        manager.lock = threading.RLock()
        manager.truth_pose = (-1.70, 15.67, 0.55, 0.30)
        manager.truth_pose_received_wall = baseline_manager_module.time.monotonic()
        manager.locomotion_ready = True
        manager.truth_model_name = "a1_gazebo"
        manager.corridor_sweep_history = []
        manager.elapsed = mock.Mock(return_value=154.0)
        manager._cmd_pub = SimpleNamespace(publish=mock.Mock())
        manager._scan_check = mock.Mock(return_value=SimpleNamespace(
            map_available=True, occupied_collision=False,
            status="clear_saved_entry"))
        door = SimpleNamespace(
            door_id="door-run58", normal=np.asarray([0.0, 1.0]),
            visited=False)
        manager.room_scheduler = SimpleNamespace(
            active_door=door, successful_roles={"ENTRY", "G3", "G4"})
        response = SimpleNamespace(success=True, status_message="ok")

        with mock.patch.object(
                baseline_manager_module.rospy, "wait_for_service"), \
                mock.patch.object(
                    baseline_manager_module.rospy, "ServiceProxy",
                    return_value=mock.Mock(return_value=response)):
            recovered = manager._restage_active_room_exit_at_truth_entry({
                "room_id": "floor_1_room_1",
                "estimated_door_id": "door-run58",
                "room_role": "EXIT"})

        self.assertTrue(recovered)
        self.assertFalse(door.visited)
        self.assertEqual(manager.room_scheduler.successful_roles,
                         {"ENTRY", "G3", "G4"})
        self.assertEqual(
            manager.corridor_sweep_history[-1]["event"],
            "ROOM_EXIT_TRUTH_ENTRY_ANCHOR_RESTAGED")

    def test_strict_fullflow_disables_all_room_model_state_recovery(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.allow_nonphysical_model_state_recovery = False
        manager.floor_number = 3
        manager.f3_room_start_restage_enabled = True
        manager._f3_room_start_blocked_key = None
        manager._f3_room_start_blocked_count = 0
        manager.f3_room_start_restage_failure_threshold = 1
        manager._f3_room_start_restage_used_doors = set()
        service = mock.Mock()
        with mock.patch.object(
                baseline_manager_module.rospy, "ServiceProxy", service):
            self.assertFalse(manager._restage_active_room_exit_at_truth_entry())
            self.assertFalse(manager._reseat_f1_corridor_resume_from_truth())
            self.assertFalse(manager._maybe_restage_f3_blocked_room_start(
                {"estimated_door_id": "d", "room_role": "G3"},
                False, "start_footprint_blocked"))
        service.assert_not_called()

    def test_run62_generated_portal_rejects_entry_frame_false_exit(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager._active_room_truth_anchor = {
            "door_id": "estimated_door_01",
            # The run62 entry transform incorrectly mapped the final body to
            # y=0.97, outside the estimated y=1.22 door plane.
            "map_pose": (14.663, 1.845, 0.32, 1.590),
            "truth_pose": (-2.317, 15.236, 0.32, -3.085),
        }
        manager.lock = threading.RLock()
        manager.truth_pose = (-1.44, 15.28, 0.32, 0.0)
        manager.truth_pose_received_wall = \
            baseline_manager_module.time.monotonic()
        manager.corridor_sweep_history = []
        manager.elapsed = mock.Mock(return_value=145.0)
        door = SimpleNamespace(
            door_id="estimated_door_01",
            truth_coordinate_transform="map_yaw_plus_90",
            truth_contract_center=(14.865, 1.1),
            truth_contract_normal_direction=math.pi / 2.0)
        manager.room_scheduler = SimpleNamespace(active_door=door)

        confirmed, reason = manager._room_exit_truth_crossing_confirmed({
            "room_id": "estimated_room_01",
            "estimated_door_id": "estimated_door_01"})

        self.assertFalse(confirmed)
        self.assertEqual(reason, "exit_truth_door_plane_not_crossed")
        event = manager.corridor_sweep_history[-1]
        self.assertEqual(event["geometry_source"],
                         "generated_layout_direct_gazebo_truth")
        self.assertAlmostEqual(event["truth_door_depth_m"], 0.34, places=2)

    def test_run62_exit_restage_centres_inside_generated_portal(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.allow_nonphysical_model_state_recovery = True
        manager._active_room_truth_anchor = {
            "door_id": "estimated_door_01",
            "map_pose": (14.663, 1.845, 0.32, 1.590),
            "truth_pose": (-2.317, 15.236, 0.32, -3.085),
        }
        manager.lock = threading.RLock()
        manager.truth_pose = (-1.44, 15.28, 0.32, 0.0)
        manager.truth_pose_received_wall = \
            baseline_manager_module.time.monotonic()
        manager.locomotion_ready = True
        manager.truth_model_name = "a1_gazebo"
        manager.corridor_sweep_history = []
        manager.elapsed = mock.Mock(return_value=145.0)
        manager._cmd_pub = SimpleNamespace(publish=mock.Mock())
        manager._scan_check = mock.Mock(return_value=SimpleNamespace(
            map_available=True, occupied_collision=False,
            status="clear_generated_portal_inside"))
        door = SimpleNamespace(
            door_id="estimated_door_01",
            truth_coordinate_transform="map_yaw_plus_90",
            truth_contract_center=(14.865, 1.1),
            truth_contract_normal_direction=math.pi / 2.0,
            visited=False)
        manager.room_scheduler = SimpleNamespace(
            active_door=door, successful_roles={"ENTRY", "G3", "G4"})
        response = SimpleNamespace(success=True, status_message="ok")
        service = mock.Mock(return_value=response)

        with mock.patch.object(
                baseline_manager_module.rospy, "wait_for_service"), \
                mock.patch.object(
                    baseline_manager_module.rospy, "ServiceProxy",
                    return_value=service):
            recovered = manager._restage_active_room_exit_at_truth_entry({
                "room_id": "estimated_room_01",
                "estimated_door_id": "estimated_door_01",
                "room_role": "EXIT"})

        self.assertTrue(recovered)
        state = service.call_args.args[0]
        # Door is world x=-1.10, y=14.865.  Room interior is negative x.
        self.assertAlmostEqual(state.pose.position.x, -1.65, places=3)
        self.assertAlmostEqual(state.pose.position.y, 14.865, places=3)
        self.assertFalse(door.visited)
        self.assertEqual(manager.room_scheduler.successful_roles,
                         {"ENTRY", "G3", "G4"})
        event = manager.corridor_sweep_history[-1]
        self.assertTrue(event["generated_truth_contract"])
        self.assertIn("generated_portal_inside_center", event["policy"])

    def test_run59_f1_corridor_resume_reseats_once_without_room_credit(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.allow_nonphysical_model_state_recovery = True
        manager.floor_number = 1
        manager.offline_truth_layout_metadata = "layout.json"
        manager._offline_truth_layout_cache = {"floors": [{
            "corridor_bounds": {
                "x_min": -1.1, "x_max": 1.1,
                "y_min": 7.85, "y_max": 35.91}}]}
        manager._f1_corridor_truth_reseat_used = False
        manager.lock = threading.RLock()
        manager.truth_pose = (-1.43, 16.60, 0.31, 1.55)
        manager.truth_pose_received_wall = \
            baseline_manager_module.time.monotonic()
        manager.locomotion_ready = True
        manager.truth_model_name = "a1_gazebo"
        manager.corridor_sweep_history = []
        manager.elapsed = mock.Mock(return_value=145.0)
        manager._cmd_pub = SimpleNamespace(publish=mock.Mock())
        door = SimpleNamespace(visited=True, completed=True)
        manager.room_scheduler = SimpleNamespace(
            detector=SimpleNamespace(doors=[door]))
        response = SimpleNamespace(success=True, status_message="ok")

        with mock.patch.object(
                baseline_manager_module.rospy, "wait_for_service"), \
                mock.patch.object(
                    baseline_manager_module.rospy, "ServiceProxy",
                    return_value=mock.Mock(return_value=response)):
            first = manager._reseat_f1_corridor_resume_from_truth({
                "position": [15.84, -0.38, 0.31]})
            second = manager._reseat_f1_corridor_resume_from_truth({
                "position": [15.84, -0.38, 0.31]})

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertTrue(door.visited)
        self.assertTrue(door.completed)
        self.assertEqual(manager._cmd_pub.publish.call_count, 1)
        self.assertEqual(
            manager.corridor_sweep_history[-1]["event"],
            "F1_CORRIDOR_RESUME_TRUTH_RESEATED")
        self.assertEqual(
            manager.corridor_sweep_history[-1]["to_truth_pose"][:2],
            [0.0, 16.60])

    def test_run60_f1_exit_rearms_exhausted_opposite_portal_once(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.floor_number = 1
        manager.upper_floor_truth_room_fallback_active = False
        manager.offline_truth_layout_metadata = "layout.json"
        manager.corridor_established = True
        manager.corridor_station_origin = np.asarray([7.7, 0.0])
        manager.corridor_station_axis = np.asarray([1.0, 0.0])
        manager.last_room_exit_side = 1
        manager.corridor_axis = np.asarray([1.0, 0.0])
        manager.truth_room_fallback_retry_limit = 3
        manager.f3_truth_room_fallback_attempts = {
            "near:-1": 3, "near:1": 2}
        manager.truth_room_fallback_last_grid_update = {
            "near:-1": 77}
        manager.truth_room_fallback_prescans = {
            "near:-1:attempt-3", "near:1:attempt-2"}
        manager.corridor_sweep_history = []
        manager.elapsed = mock.Mock(return_value=154.0)
        manager._mission_rooms_physically_exited = mock.Mock(
            return_value=False)

        manager._rearm_recent_exit_opposite_truth_portal(
            (14.88, 0.02, 0.31, 0.0))

        self.assertEqual(
            manager.f3_truth_room_fallback_attempts["near:-1"], 2)
        self.assertEqual(manager.f3_truth_room_fallback_attempts["near:1"], 2)
        self.assertNotIn(
            "near:-1", manager.truth_room_fallback_last_grid_update)
        self.assertNotIn(
            "near:-1:attempt-3", manager.truth_room_fallback_prescans)
        self.assertIn(
            "near:1:attempt-2", manager.truth_room_fallback_prescans)
        self.assertEqual(
            manager.corridor_sweep_history[-1]["event"],
            "RECENT_EXIT_OPPOSITE_TRUTH_PORTAL_REARMED")
        self.assertEqual(
            manager.corridor_sweep_history[-1]["opposite_side"], -1)

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

    def test_open_partial_retry_discards_false_near_success(self):
        scheduler = LightweightRoomScheduler.__new__(
            LightweightRoomScheduler)
        scheduler.active_room_id = "floor_2_estimated_room_04"
        scheduler.events = []
        scheduler.successful_roles = {"G3", "G4"}
        scheduler.completed_points = [(4.615, 0.0), (6.428, 0.0)]
        scheduler.active_door = EstimatedDoorway(
            door_id="estimated_door_04",
            center=(0.0, 0.0),
            normal_direction=0.0,
            width=2.0,
            left_frame_point=(0.0, 1.0),
            right_frame_point=(0.0, -1.0),
            corridor_side=(-1.0, 0.0),
            interior_side=(1.0, 0.0),
            confidence=1.0,
            entered_at=0.0,
            room_id="floor_2_estimated_room_04",
            viewpoint_contract="open_deep_near",
            g3_truth_pose=(4.615, 0.0),
            g4_truth_pose=(6.428, 0.0))
        evidence = {
            "physical_two_view_contract_met": False,
            "deep_view_met": True,
            "near_view_met": False,
            "g3_depth_m": 4.615,
            "g4_depth_m": 6.428,
        }

        scheduler._canonicalize_open_partial_contract(evidence, 620.0)

        self.assertEqual(scheduler.successful_roles, {"G3"})
        self.assertEqual(scheduler.completed_points, [(6.428, 0.0)])
        self.assertEqual(scheduler.active_door.g3_truth_pose,
                         (6.428, 0.0))
        self.assertIsNone(scheduler.active_door.g4_truth_pose)
        self.assertEqual(scheduler.events[-1]["repair_role"], "G4")
        self.assertEqual(
            scheduler.events[-1]["policy"],
            "bounded_retry_only_missing_truth_depth_class")

    def test_f3_blocked_room_start_restages_once_without_view_credit(self):
        manager = BaselineExplorationManager.__new__(
            BaselineExplorationManager)
        manager.allow_nonphysical_model_state_recovery = True
        manager.floor_number = 3
        manager.f3_room_start_restage_enabled = True
        manager.f3_room_start_restage_failure_threshold = 1
        manager.f3_room_start_restage_max_distance = 2.75
        manager._f3_room_start_blocked_key = None
        manager._f3_room_start_blocked_count = 0
        manager._f3_room_start_restage_used_doors = set()
        manager._active_room_truth_anchor = {
            "door_id": "door-3",
            "map_pose": (1.90, 28.90, 5.58, 0.0),
            "truth_pose": (1.90, 28.90, 5.58, 0.0),
        }
        manager.lock = threading.RLock()
        # The first F3 open-room near mapping pose can be about two metres
        # laterally from the physically crossed ENTRY anchor.
        manager.truth_pose = (3.65, 30.20, 5.58, 1.65)
        manager.truth_pose_received_wall = baseline_manager_module.time.monotonic()
        manager.locomotion_ready = True
        manager.truth_model_name = "a1_gazebo"
        manager.corridor_sweep_history = []
        manager.elapsed = lambda: 200.0
        manager._cmd_pub = SimpleNamespace(publish=mock.Mock())
        manager._scan_check = mock.Mock(return_value=SimpleNamespace(
            map_available=True, occupied_collision=False,
            status="clear_saved_entry"))
        scheduler = SimpleNamespace(
            active_door=SimpleNamespace(door_id="door-3"),
            role_attempts={"G4": 9}, failed_roles={"G4"},
            visual_geometry_retry_roles={"G4"},
            adaptive_route_queue=[{"role": "G4"}],
            visual_deepening_requested=False,
            visual_geometry_map_refresh_pending=None,
            next_role_index=3, _set_state=mock.Mock())
        manager.room_scheduler = scheduler
        goal = {
            "room_id": "floor_3_estimated_room_03",
            "estimated_door_id": "door-3", "room_role": "G4"}
        response = SimpleNamespace(success=True, status_message="ok")

        with mock.patch.object(
                baseline_manager_module.rospy, "wait_for_service"), \
                mock.patch.object(
                    baseline_manager_module.rospy, "ServiceProxy",
                    return_value=mock.Mock(return_value=response)):
            first = manager._maybe_restage_f3_blocked_room_start(
                goal, False, "start_footprint_blocked")
            second = manager._maybe_restage_f3_blocked_room_start(
                goal, False, "start_footprint_blocked")

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(manager._cmd_pub.publish.call_count, 1)
        self.assertNotIn("G4", scheduler.role_attempts)
        self.assertNotIn("G4", scheduler.failed_roles)
        self.assertNotIn("G4", scheduler.visual_geometry_retry_roles)
        self.assertEqual(scheduler.adaptive_route_queue, [])
        self.assertEqual(scheduler.visual_geometry_map_refresh_pending, "G4")
        self.assertEqual(scheduler.next_role_index, 2)
        self.assertNotIn("G4", getattr(scheduler, "successful_roles", set()))
        self.assertEqual(
            manager.corridor_sweep_history[-1]["event"],
            "F3_ROOM_START_TRUTH_ENTRY_ANCHOR_RESTAGED")


if __name__ == "__main__":
    unittest.main()
