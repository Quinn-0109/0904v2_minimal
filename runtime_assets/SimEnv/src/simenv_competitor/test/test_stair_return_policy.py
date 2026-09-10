#!/usr/bin/env python3
"""Regression tests for first-floor to stair ownership transfer."""

import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np


SCRIPT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "scripts"))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from baseline_exploration_manager import (
    BaselineExplorationManager, two_pose_route_requires_obstacle_bend)
from lightweight_room_core import LightweightRoomConfig, LightweightRoomScheduler
from stair_visual_depth_detector import normalize_depth_image


def manager_with(exited, terminal=False, active_door=None, enabled=True, strict=True):
    manager = object.__new__(BaselineExplorationManager)
    manager.stair_handoff_on_corridor_exit = enabled
    manager.require_all_rooms_for_floor_handoff = strict
    manager.room_target_count = 4
    manager.corridor_terminal_return_latched = terminal
    manager.room_scheduler = SimpleNamespace(
        active_door=active_door,
        detector=SimpleNamespace(doors=[
            SimpleNamespace(visited=index < exited,
                            completed=index < exited)
            for index in range(4)
        ]))
    return manager


class StairReturnPolicyTest(unittest.TestCase):
    def test_integrated_role_timing_uses_ros_state_residency(self):
        manager = object.__new__(BaselineExplorationManager)
        manager.mission_clock_started = True
        manager.state_entered_sim_elapsed = 10.0
        manager.state = "EXECUTE_GOAL"
        manager.state_residency_ros_sim_s = {}
        manager.room_role_timing_ros_sim = {}
        manager.telemetry_room_id = "track-2"
        manager.telemetry_room_role = "G3"
        manager.simulation_elapsed = lambda: 16.25

        manager._flush_state_residency()

        self.assertEqual(manager.state_residency_ros_sim_s,
                         {"EXECUTE_GOAL": 6.25})
        record = manager.room_role_timing_ros_sim[
            "room=track-2|role=G3"]
        self.assertEqual(record["goal_motion_ros_sim_s"], 6.25)
        self.assertEqual(record["visual_scan_ros_sim_s"], 0.0)

    def test_mandatory_g3_scan_and_settle_have_distinct_categories(self):
        self.assertEqual(
            BaselineExplorationManager._state_timing_category(
                "VISUAL_SWEEP_SETTLE", "G3"),
            "stop_settle_ros_sim_s")
        self.assertEqual(
            BaselineExplorationManager._state_timing_category(
                "LOCAL_RESCAN", "G3"),
            "visual_scan_ros_sim_s")
        self.assertEqual(
            BaselineExplorationManager._state_timing_category(
                "LOCAL_RESCAN", "ENTRY"),
            "retry_recovery_ros_sim_s")

    def test_actual_corridor_goal_clears_previous_exit_timing_context(self):
        manager = object.__new__(BaselineExplorationManager)
        manager.mission_clock_started = False
        manager.telemetry_room_id = "room-1"
        manager.telemetry_room_role = "EXIT"
        manager.room_scheduler = SimpleNamespace(active_door=None)

        manager._set_telemetry_goal_context({
            "source": "corridor_sweep", "room_role": None})

        self.assertIsNone(manager.telemetry_room_id)
        self.assertIsNone(manager.telemetry_room_role)

    def test_actual_room_goal_sets_role_timing_context(self):
        manager = object.__new__(BaselineExplorationManager)
        manager.mission_clock_started = False
        manager.telemetry_room_id = None
        manager.telemetry_room_role = None
        manager.room_scheduler = SimpleNamespace(active_door=None)

        manager._set_telemetry_goal_context({
            "source": "lightweight_room_semantic",
            "room_id": "room-4", "room_role": "G4"})

        self.assertEqual(manager.telemetry_room_id, "room-4")
        self.assertEqual(manager.telemetry_room_role, "G4")

    def test_deep_obstacle_edge_recenter_is_slow_but_mandatory_fan_unchanged(self):
        manager = object.__new__(BaselineExplorationManager)
        manager.room_camera_sweep_speed = 2.10
        manager.deep_obstacle_edge_recenter_speed = 0.90
        manager.room_scheduler = SimpleNamespace(active_door=None)

        speed = manager._edge_recenter_speed_for_goal({
            "room_goal_diagnostic": {
                "obstacle_view_policy": "deep_outer_side_peek"}})
        ordinary = manager._edge_recenter_speed_for_goal({
            "room_goal_diagnostic": {
                "obstacle_view_policy": "shallow_front_chord"}})

        self.assertEqual(speed, 0.90)
        self.assertEqual(ordinary, 2.10)
        self.assertEqual(manager.room_camera_sweep_speed, 2.10)

    def test_named_obstacle_route_requires_bend(self):
        self.assertTrue(two_pose_route_requires_obstacle_bend({
            "two_pose_strategy": "obstacle_deep_then_opposite_near",
        }))

    def test_unknown_deep_region_provisional_pair_remains_open(self):
        self.assertFalse(two_pose_route_requires_obstacle_bend({
            "two_pose_strategy": "provisional_obstacle_side_split",
            "two_pose_obstacle": {
                "classification": "deep_region_not_yet_observed",
                "provisional_side_split": True,
            },
        }))

    def test_measured_obstacle_provisional_pair_uses_safe_speed_mode(self):
        diagnostic = {
            "two_pose_strategy": "provisional_obstacle_side_split",
            "two_pose_index": 2,
            "two_pose_obstacle": {
                "classification": "occupied_centre_component",
                "area_m2": 0.225,
            },
        }
        self.assertTrue(two_pose_route_requires_obstacle_bend(diagnostic))
        manager = object.__new__(BaselineExplorationManager)
        manager.room_scheduler = SimpleNamespace(
            active_door=SimpleNamespace(width=1.2))
        manager.obstacle_room_transition_speed = 1.20
        manager.obstacle_room_transition_lateral_speed = 0.30
        manager.obstacle_room_transition_heading_gain = 0.80
        manager.obstacle_room_transition_maximum_yaw_rate = 0.35
        manager.obstacle_room_transition_distance_gain = 0.75
        manager.obstacle_room_transition_minimum_speed = 0.12
        manager.goal_minimum_speed = 0.10
        manager.elapsed = lambda: 31.5
        manager.corridor_sweep_history = []
        published = {}
        manager.speed_limit_pub = SimpleNamespace(
            publish=lambda msg: published.__setitem__("speed", msg.data))
        manager.minimum_speed_pub = SimpleNamespace(
            publish=lambda msg: published.__setitem__("minimum_speed", msg.data))
        manager.goal_degraded_speed_cap_pub = SimpleNamespace(
            publish=lambda msg: None)
        manager.lateral_speed_limit_pub = SimpleNamespace(
            publish=lambda msg: published.__setitem__("lateral", msg.data))
        manager.heading_gain_pub = SimpleNamespace(
            publish=lambda msg: published.__setitem__("heading", msg.data))
        manager.maximum_yaw_rate_pub = SimpleNamespace(
            publish=lambda msg: published.__setitem__("yaw", msg.data))
        manager.distance_gain_pub = SimpleNamespace(
            publish=lambda msg: published.__setitem__("distance", msg.data))
        manager._publish_speed_for_goal({
            "source": "lightweight_room_semantic", "room_role": "G4",
            "room_goal_diagnostic": diagnostic})
        self.assertEqual(published, {
            "speed": 1.20, "lateral": 0.30, "heading": 0.80,
            "yaw": 0.35, "distance": 0.75, "minimum_speed": 0.12})
        self.assertEqual(manager.corridor_sweep_history[-1]["mode"],
                         "ROOM_OBSTACLE_BEND_SAFE")

    def test_four_physical_room_exits_authorize_handoff(self):
        manager = manager_with(4)
        self.assertTrue(manager._stair_return_handoff_authorized())

    def test_terminal_wall_starts_supplement_but_not_partial_handoff(self):
        manager = manager_with(3, terminal=True)
        self.assertFalse(manager._stair_return_handoff_authorized())
        self.assertTrue(manager._terminal_return_room_supplement_active())

    def test_strict_mode_never_lowers_physical_target_below_four(self):
        manager = manager_with(3, terminal=True)
        manager.room_target_count = 3
        self.assertFalse(manager._stair_return_handoff_authorized())

    def test_explicit_legacy_mode_retains_terminal_best_effort_handoff(self):
        manager = manager_with(3, terminal=True, strict=False)
        self.assertTrue(manager._stair_return_handoff_authorized())

    def test_strict_requirement_dominates_enabled_partial_continuity(self):
        manager = manager_with(3, terminal=True, strict=True)
        manager.allow_partial_floor_handoff = True
        manager.partial_floor_handoff_minimum_exited_rooms = 3
        manager.emergency_partial_handoff_armed = False
        for door in manager.room_scheduler.detector.doors:
            door.completed = door.visited
        self.assertFalse(manager._floor_handoff_room_requirement_met())

        manager.room_scheduler.detector.doors[3].visited = True
        manager.room_scheduler.detector.doors[3].completed = True
        self.assertTrue(manager._floor_handoff_room_requirement_met())


    def test_partial_return_without_terminal_confirmation_is_rejected(self):
        manager = manager_with(3, terminal=False)
        self.assertFalse(manager._stair_return_handoff_authorized())

    def test_active_room_blocks_handoff(self):
        manager = manager_with(4, terminal=True, active_door=object())
        self.assertFalse(manager._stair_return_handoff_authorized())

    def test_four_exits_end_terminal_supplement(self):
        manager = manager_with(4, terminal=True)
        self.assertFalse(manager._terminal_return_room_supplement_active())

    def test_terminal_supplement_bypasses_outbound_high_water_gate(self):
        manager = manager_with(3, terminal=True)
        manager._missing_room_retrace_active = lambda: False
        manager._update_corridor_forward_high_water = lambda pose, context: (
            self.fail("reverse supplement must not use outbound high-water"))
        self.assertTrue(manager._at_corridor_forward_high_water(
            pose=(0.0, 0.0, 0.0), context=None))

    def test_first_floor_room_phase_deadline_starts_best_effort_return(self):
        manager = object.__new__(BaselineExplorationManager)
        manager.enforce_room_phase_target_deadline = True
        manager.room_phase_target_seconds = 150.0
        manager.corridor_partial_return_minimum_outbound_progress = 0.0
        manager.corridor_partial_return_reserve = 0.0
        manager.maximum_duration = 700.0
        manager.room_scheduler = SimpleNamespace(first_room_started_at=50.0)
        manager.elapsed = lambda: 199.9
        self.assertIsNone(manager._partial_corridor_return_trigger())
        manager.elapsed = lambda: 200.0
        self.assertEqual(manager._partial_corridor_return_trigger(),
                         "room_phase_deadline")

    def test_upper_floor_can_disable_first_floor_phase_deadline(self):
        manager = object.__new__(BaselineExplorationManager)
        manager.enforce_room_phase_target_deadline = False
        manager.room_phase_target_seconds = 150.0
        manager.corridor_partial_return_minimum_outbound_progress = 0.0
        manager.corridor_partial_return_reserve = 0.0
        manager.maximum_duration = 700.0
        manager.room_scheduler = SimpleNamespace(first_room_started_at=50.0)
        manager.elapsed = lambda: 250.0
        self.assertIsNone(manager._partial_corridor_return_trigger())

    def test_room_phase_deadline_allows_bounded_opposite_probe_after_exit(self):
        manager = object.__new__(BaselineExplorationManager)
        manager.enforce_room_phase_target_deadline = True
        manager.room_phase_target_seconds = 150.0
        manager.corridor_partial_return_minimum_outbound_progress = 0.0
        manager.corridor_partial_return_reserve = 0.0
        manager.maximum_duration = 700.0
        manager.room_scheduler = SimpleNamespace(first_room_started_at=50.0)
        manager.last_room_exit_side = -1
        manager.last_room_exit_at = 199.0
        manager.room_phase_post_exit_opposite_probe_grace = 8.0
        manager.room_phase_opposite_probe_logged_exit_at = None
        manager.corridor_sweep_history = []

        manager.elapsed = lambda: 203.0
        self.assertIsNone(manager._partial_corridor_return_trigger())
        self.assertEqual(
            manager.corridor_sweep_history[-1]["event"],
            "ROOM_PHASE_DEADLINE_OPPOSITE_PROBE_GRACE")

        manager.elapsed = lambda: 207.1
        self.assertEqual(manager._partial_corridor_return_trigger(),
                         "room_phase_deadline")

    def test_room_observation_budget_restarts_after_confirmed_entry(self):
        scheduler = LightweightRoomScheduler(LightweightRoomConfig(
            enabled=True, budget_starts_after_entry=True))
        scheduler.active_door = SimpleNamespace(door_id="door")
        scheduler.room_started_at = 10.0
        scheduler._confirm_entry((1.0, 2.0), 42.0, "test")
        self.assertEqual(scheduler.room_started_at, 42.0)
        self.assertEqual(scheduler.first_room_started_at, 42.0)
        self.assertTrue(scheduler.entry_confirmed)
        self.assertEqual(scheduler.return_anchor, (1.0, 2.0))
        scheduler._clear_active()
        self.assertEqual(scheduler.first_room_started_at, 42.0)

    def test_far_room_exit_accepts_door_local_corridor_endpoint(self):
        manager = object.__new__(BaselineExplorationManager)
        manager.room_exit_require_raw_corridor_confirmation = False
        manager.room_exit_require_station_corridor_confirmation = True
        manager.corridor_centerline_membership_tolerance = 0.75
        manager.room_exit_station_corridor_tolerance = 0.75
        manager._on_station_corridor_centerline = lambda point, tolerance=None: False
        manager.elapsed = lambda: 500.0
        manager.corridor_sweep_history = []
        door = SimpleNamespace(
            depth=lambda point: float(point[1]) - 1.24)
        manager.room_scheduler = SimpleNamespace(
            active_door=door,
            config=SimpleNamespace(semantic_completion_tolerance=0.20))
        accepted, reason = manager._room_exit_corridor_confirmed(
            (28.470, -0.105), None,
            {"room_id": "room_03",
             "position": [28.483, -0.272, 0.0]})
        self.assertTrue(accepted)
        self.assertIsNone(reason)
        self.assertEqual(
            manager.corridor_sweep_history[-1]["policy"],
            "door_local_corridor_endpoint_plus_crossing")

    def test_all_invalid_depth_frame_does_not_break_stair_detection(self):
        depth = np.full((4, 6), np.nan, dtype=np.float32)
        normalized = normalize_depth_image(depth)
        self.assertEqual(normalized.shape, depth.shape)
        self.assertTrue(np.isnan(normalized).all())

    def test_recent_exit_opposite_check_releases_after_opposite_entry(self):
        manager = object.__new__(BaselineExplorationManager)
        manager.last_room_exit_side = 1
        manager.last_room_exit_pose = (14.0, 0.1, 0.0, 0.0)
        manager.elapsed = lambda: 120.0
        manager.room_exit_resume_until = 80.0
        manager.room_exit_opposite_door_revisit_time = 40.0
        manager.room_exit_opposite_door_revisit_radius = 3.5
        manager.corridor_branch_station_tolerance = 1.5
        manager._corridor_station = lambda pose: 6.2
        manager.branch_scheduler = SimpleNamespace(branches=[
            SimpleNamespace(state="COVERED", side=1, station=6.2),
            SimpleNamespace(state="ENTERED_PARTIAL", side=-1, station=6.2),
        ])
        manager.room_scheduler = SimpleNamespace(
            active_door=SimpleNamespace(
                door_id="estimated_door_02",
                corridor_side=(14.02, 0.11)),
            entry_confirmed=True)
        manager._recent_room_exit_opposite_side = lambda pose: 1
        self.assertFalse(manager._recent_exit_needs_opposite_check(
            (14.05, -1.54, 0.0, 0.0)))

    def test_completed_measured_opposite_releases_stale_branch_hold(self):
        manager = object.__new__(BaselineExplorationManager)
        manager.last_room_exit_side = -1
        manager.last_room_exit_pose = (6.2247, -0.5009)
        manager.last_room_exit_at = 85.316
        manager.elapsed = lambda: 90.0
        manager.room_exit_resume_until = 90.316
        manager.room_exit_opposite_door_revisit_time = 40.0
        manager.room_exit_opposite_door_revisit_radius = 3.5
        manager.post_exit_opposite_entry_failures = 0
        manager.post_exit_opposite_entry_failure_limit = 3
        manager.post_exit_opposite_station_holds = 0
        manager.post_exit_opposite_station_hold_limit = 8
        manager.corridor_branch_station_tolerance = 1.5
        manager.corridor_station_axis = np.asarray(
            [0.9999613025224572, 0.008797355147476723])
        manager.corridor_station_origin = np.asarray(
            [3.8785081653690967, -0.20222333332272757])
        manager.branch_scheduler = SimpleNamespace(branches=[
            SimpleNamespace(state="COVERED", side=1, station=4.833),
            SimpleNamespace(state="COVERED", side=-1, station=2.211),
        ])
        opposite = SimpleNamespace(
            visited=True, completed=True, center=(6.1272, 0.5423),
            interior_side=(6.1035, 1.542))
        just_exited = SimpleNamespace(
            visited=True, completed=True, center=(6.2241, -1.0566),
            interior_side=(6.2307, -1.8069))
        manager.room_scheduler = SimpleNamespace(
            active_door=None, entry_confirmed=False,
            detector=SimpleNamespace(doors=[opposite, just_exited]))

        self.assertFalse(manager._recent_exit_needs_opposite_check(
            (6.2247, -0.5009)))

    def test_recent_exit_opposite_side_releases_after_pairing_deadline(self):
        manager = object.__new__(BaselineExplorationManager)
        manager.last_room_exit_side = -1
        manager.last_room_exit_pose = (14.0, 0.1)
        manager.last_room_exit_at = 100.0
        manager.elapsed = lambda: 145.0
        manager.room_exit_resume_until = 105.0
        manager.room_exit_opposite_door_revisit_time = 40.0
        manager.room_exit_opposite_door_revisit_radius = 3.5
        manager.post_exit_opposite_entry_failures = 0
        manager.post_exit_opposite_entry_failure_limit = 3
        manager.post_exit_opposite_station_holds = 0
        manager.post_exit_opposite_station_hold_limit = 8
        manager.corridor_branch_station_tolerance = 1.5
        manager._corridor_station = lambda pose: 6.2
        manager.branch_scheduler = SimpleNamespace(branches=[
            SimpleNamespace(state="COVERED", side=-1, station=6.2),
        ])
        self.assertIsNone(manager._recent_room_exit_opposite_side(
            (14.05, 0.12)))

    def test_recent_exit_opposite_side_releases_after_entry_failures(self):
        manager = object.__new__(BaselineExplorationManager)
        manager.last_room_exit_side = -1
        manager.last_room_exit_pose = (14.0, 0.1)
        manager.last_room_exit_at = 120.0
        manager.elapsed = lambda: 125.0
        manager.room_exit_resume_until = 125.0
        manager.room_exit_opposite_door_revisit_time = 120.0
        manager.room_exit_opposite_door_revisit_radius = 3.5
        manager.post_exit_opposite_entry_failures = 3
        manager.post_exit_opposite_entry_failure_limit = 3
        manager.post_exit_opposite_station_holds = 0
        manager.post_exit_opposite_station_hold_limit = 8
        manager.corridor_branch_station_tolerance = 1.5
        manager._corridor_station = lambda pose: 6.2
        manager.branch_scheduler = SimpleNamespace(branches=[
            SimpleNamespace(state="COVERED", side=-1, station=6.2),
        ])
        self.assertIsNone(manager._recent_room_exit_opposite_side(
            (14.05, 0.12)))

    def test_transit_state_is_announced_once(self):
        manager = manager_with(4, terminal=True)
        manager.stair_return_transit_announced = False
        published = []
        manager.stair_return_transit_pub = SimpleNamespace(
            publish=lambda message: published.append(bool(message.data)))
        states = []
        manager._set_state = lambda state, reason=None: states.append(
            (state, reason))
        manager._announce_stair_return_transit("terminal")
        manager._announce_stair_return_transit("duplicate")
        self.assertEqual(states, [("STAIR_RETURN_TRANSIT", "terminal")])
        self.assertEqual(published, [True])


if __name__ == "__main__":
    unittest.main()
